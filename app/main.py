import json
import os
import tempfile
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
from fractions import Fraction

from PIL import Image, ImageOps
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pyhanko import stamp
from pyhanko.pdf_utils.text import TextBoxStyle
from pyhanko.pdf_utils.font.basic import SimpleFontEngineFactory, COURIER_META
from pyhanko.pdf_utils import images
from pyhanko.pdf_utils.incremental_writer import IncrementalPdfFileWriter
from pyhanko.sign import fields, signers
from pyhanko.pdf_utils.layout import SimpleBoxLayoutRule, Margins, AxisAlignment, InnerScaling


app = FastAPI(title="PDF Signing API")

# Smallest signature box allowed (width, height) in PDF points.
MIN_SIGNATURE_SIZE = (144, 54)


@dataclass
class SignatureConfig:
    box: Tuple[int, int]
    pfx_path: str
    passphrase: bytes
    background_img: Optional[str] = None
    size: Tuple[int, int] = MIN_SIGNATURE_SIZE


def clamp_signature_size(width: Optional[int], height: Optional[int]) -> Tuple[int, int]:
    """Use the requested size, but never go below MIN_SIGNATURE_SIZE (per dimension)."""
    w = width if width is not None else MIN_SIGNATURE_SIZE[0]
    h = height if height is not None else MIN_SIGNATURE_SIZE[1]
    return max(w, MIN_SIGNATURE_SIZE[0]), max(h, MIN_SIGNATURE_SIZE[1])


@dataclass
class PageSignature:
    """One visible signature: which page, where, and how big."""
    page: int
    box_x: int
    box_y: int
    width: int = MIN_SIGNATURE_SIZE[0]
    height: int = MIN_SIGNATURE_SIZE[1]

    @property
    def size(self) -> Tuple[int, int]:
        return self.width, self.height


_ALLOWED_KEYS = {"page", "box_x", "box_y", "signature_width", "signature_height"}
_REQUIRED_KEYS = ("page", "box_x", "box_y")


def _as_int(value: Any, label: str) -> int:
    # bool is a subclass of int in Python, and floats like 1.5 must not be truncated silently
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be a whole number, got {value!r}")
    return value


def parse_page_settings(raw: str) -> List[PageSignature]:
    """Parse the `page_settings` JSON array into validated PageSignature objects.

    Example:
        [{"page": 0, "box_x": 50, "box_y": 50},
         {"page": 1, "box_x": 300, "box_y": 600, "signature_width": 250, "signature_height": 90}]
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"page_settings is not valid JSON: {e}")

    if not isinstance(data, list) or not data:
        raise ValueError("page_settings must be a non-empty JSON array")

    result: List[PageSignature] = []
    for i, item in enumerate(data):
        where = f"page_settings[{i}]"
        if not isinstance(item, dict):
            raise ValueError(f"{where} must be an object")

        missing = [k for k in _REQUIRED_KEYS if k not in item]
        if missing:
            raise ValueError(f"{where} is missing required field(s): {', '.join(missing)}")
        unknown = sorted(set(item) - _ALLOWED_KEYS)
        if unknown:
            raise ValueError(
                f"{where} has unknown field(s): {', '.join(unknown)} "
                f"(allowed: {', '.join(sorted(_ALLOWED_KEYS))})"
            )

        page = _as_int(item["page"], f"{where}.page")
        if page < 0:
            raise ValueError(f"{where}.page must be 0 or greater")
        box_x = _as_int(item["box_x"], f"{where}.box_x")
        box_y = _as_int(item["box_y"], f"{where}.box_y")

        width = item.get("signature_width")
        height = item.get("signature_height")
        width = _as_int(width, f"{where}.signature_width") if width is not None else None
        height = _as_int(height, f"{where}.signature_height") if height is not None else None
        w, h = clamp_signature_size(width, height)

        result.append(PageSignature(page, box_x, box_y, w, h))
    return result


def delete_temp_files(files: list):
    for file in files:
        if file and os.path.exists(file):
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error deleting file '{file}': {e}")


# Courier-Bold is a standard PDF font (no embedding needed) with the same glyph
# width as Courier (0.6 x size), so fit_font_size() stays accurate.
BOLD_FONT = SimpleFontEngineFactory("Courier-Bold", 0.6, COURIER_META)


def adobe_date(dt: datetime) -> str:
    """Format like Adobe Acrobat's default stamp: 2026.10.04 13:55:23 +08'00'"""
    offset = dt.strftime("%z")  # e.g. +0800
    return dt.strftime("%Y.%m.%d %H:%M:%S ") + f"{offset[:3]}'{offset[3:]}'"


def signing_now() -> datetime:
    """Current time in SIGN_TIMEZONE (e.g. 'Asia/Manila'), else the system timezone."""
    tz_name = os.getenv("SIGN_TIMEZONE")
    return datetime.now(ZoneInfo(tz_name)) if tz_name else datetime.now().astimezone()


def get_signer_email(signer) -> str:
    """Read the email from the certificate subject (emailAddress) or SAN."""
    cert = signer.signing_cert
    email = cert.subject.native.get("email_address")
    if email:
        return email
    try:
        for ext in cert["tbs_certificate"]["extensions"]:
            if ext["extn_id"].native == "subject_alt_name":
                for name in ext["extn_value"].parsed:
                    if name.name == "rfc822_name":
                        return name.native
    except Exception:
        pass
    return "N/A"


def prepare_background(
    src_path: str,
    dst_path: str,
    box_w: float,
    box_h: float,
    fade: float = 0.35,
    padding_ratio: float = 0.06,
    scale: int = 4,
) -> str:
    """Fit the image onto a TRANSPARENT canvas that matches the signature box.

    - keeps the image's aspect ratio and centers it
    - leftover space and the image's own white background become transparent,
      so the page content underneath stays visible
    - fades the image (fade=1 -> full strength) so the text on top stays readable
    """
    w, h = int(box_w * scale), int(box_h * scale)

    img = Image.open(src_path)
    if img.mode in ("RGBA", "LA") or "transparency" in img.info:
        img = img.convert("RGBA")
        alpha = img.getchannel("A")
    else:
        # No alpha (e.g. JPG, ink on white paper): treat brightness as opacity,
        # so white -> fully transparent and dark ink -> fully opaque.
        rgb = img.convert("RGB")
        alpha = ImageOps.invert(rgb.convert("L"))
        # Clean up near-white noise / JPEG artifacts around the strokes.
        alpha = alpha.point(lambda a: 0 if a < 24 else min(255, int((a - 24) * 255 / (255 - 24))))
        img = rgb.convert("RGBA")

    img.putalpha(alpha)
    # Fade the strokes. Only the alpha is scaled, so the ink colour stays crisp.
    img.putalpha(img.getchannel("A").point(lambda a: int(a * fade)))

    inner = (int(w * (1 - 2 * padding_ratio)), int(h * (1 - 2 * padding_ratio)))
    img = ImageOps.contain(img, inner)  # scales up or down, keeps aspect ratio

    canvas = Image.new("RGBA", (w, h), (255, 255, 255, 0))  # fully transparent
    canvas.alpha_composite(img, ((w - img.width) // 2, (h - img.height) // 2))
    canvas.save(dst_path, "PNG")
    return dst_path


def fit_font_size(
    lines: List[str],
    box_w: float,
    box_h: float,
    padding: float = 4,
    max_size: float = 10,
    min_size: float = 4,
    leading_factor: float = 1.2,
) -> Fraction:
    """Largest font size at which all lines fit inside the box.

    Assumes pyHanko's default monospaced font (Courier: glyph width = 0.6 x size).
    """
    avail_w = box_w - 2 * padding
    avail_h = box_h - 2 * padding
    longest = max(len(line) for line in lines)
    size_by_width = avail_w / (longest * 0.6)
    size_by_height = avail_h / (len(lines) * leading_factor)
    size = min(max_size, size_by_width, size_by_height)
    # pyHanko's layout needs exact numbers (int/Fraction), not floats
    return Fraction(int(max(min_size, size) * 10), 10)  # round down to 0.1 pt


def sign_pdf(input_pdf: str, output_pdf: str, signature: SignatureConfig, page: int) -> str:
    stamp_text = "Digitally signed by:\n%(name)s\n<%(email)s>\nDate: %(date)s"
    signature_size = clamp_signature_size(*signature.size)

    signature_x = signature.box[0]
    signature_y = signature.box[1]
    signature_box = signature.box + (
        signature_x + signature_size[0],
        signature_y + signature_size[1],
    )

    signer = signers.pdf_cms.SimpleSigner.load_pkcs12(
        pfx_file=signature.pfx_path,
        passphrase=signature.passphrase,
    )

    signer_email = get_signer_email(signer)
    signer_name = signer.signing_cert.subject.native.get("common_name", "Unknown")

    date_str = adobe_date(signing_now())
    sample_lines = [
        "Digitally signed by:",
        f"{signer_name}",
        f"<{signer_email}>",
        f"Date: {date_str}",
    ]
    font_size = fit_font_size(sample_lines, signature_size[0], signature_size[1])

    bg_path = None
    if signature.background_img:
        bg_path = prepare_background(
            signature.background_img,
            output_pdf + ".bg.png",
            signature_size[0],
            signature_size[1],
        )

    try:
        with open(input_pdf, "rb") as inf:
            writer = IncrementalPdfFileWriter(inf)
            signature_field_name = f"Signature_{uuid.uuid4().hex[:8]}"

            fields.append_signature_field(
                writer,
                fields.SigFieldSpec(
                    signature_field_name,
                    box=signature_box,
                    on_page=page,
                ),
            )

            meta = signers.PdfSignatureMetadata(field_name=signature_field_name)

            style = stamp.TextStampStyle(
                background_opacity=1,   # fading is baked into the image
                border_width=0,
                text_box_style=TextBoxStyle(font=BOLD_FONT, font_size=font_size, leading=font_size * Fraction(6, 5)),
                inner_content_layout=SimpleBoxLayoutRule(
                    x_align=AxisAlignment.ALIGN_MID,
                    y_align=AxisAlignment.ALIGN_MID,
                    margins=Margins.uniform(4),
                ),
                background_layout=SimpleBoxLayoutRule(
                    x_align=AxisAlignment.ALIGN_MID,
                    y_align=AxisAlignment.ALIGN_MID,
                    margins=Margins(),   # image already matches the box exactly
                    inner_content_scaling=InnerScaling.SHRINK_TO_FIT,
                ),
                stamp_text=stamp_text,
                background=images.PdfImage(bg_path) if bg_path else None,
            )

            pdf_signer = signers.PdfSigner(meta, signer=signer, stamp_style=style)

            with open(output_pdf, "wb") as outf:
                pdf_signer.sign_pdf(
                    writer,
                    output=outf,
                    appearance_text_params={
                        "name": signer_name,
                        "email": signer_email,
                        "date": date_str,
                    },
                )
    finally:
        delete_temp_files([bg_path])

    return output_pdf


def sign_pdf_multiple(
    input_pdf: str,
    output_pdf: str,
    pfx_path: str,
    passphrase: bytes,
    placements: List[PageSignature],
    background_img: Optional[str] = None,
) -> str:
    """Sign the PDF once per placement, each with its own real signature field.

    Each pass signs the output of the previous one (incremental updates), so earlier
    signatures stay valid. The actual signing is done by the existing sign_pdf().
    """
    # Check every page up front so we fail before doing any signing work.
    with open(input_pdf, "rb") as f:
        page_count = int(IncrementalPdfFileWriter(f).root["/Pages"]["/Count"])
    for i, p in enumerate(placements):
        if p.page >= page_count:
            raise ValueError(
                f"page_settings[{i}].page {p.page} is out of range: "
                f"the document has {page_count} page(s) (0-indexed)"
            )

    current_input = input_pdf
    intermediates: List[str] = []

    try:
        for idx, p in enumerate(placements):
            is_last = idx == len(placements) - 1
            step_output = output_pdf if is_last else f"{output_pdf}.step{idx}.pdf"

            sign_pdf(
                input_pdf=current_input,
                output_pdf=step_output,
                signature=SignatureConfig(
                    box=(p.box_x, p.box_y),
                    pfx_path=pfx_path,
                    passphrase=passphrase,
                    background_img=background_img,
                    size=p.size,
                ),
                page=p.page,
            )

            if not is_last:
                intermediates.append(step_output)
            current_input = step_output
    finally:
        delete_temp_files(intermediates)

    return output_pdf


@app.post("/sign-pdf")
def sign_pdf_endpoint(
    pdf_file: UploadFile = File(..., description="The PDF document to sign"),
    pfx_file: UploadFile = File(..., description="The PKCS#12 (.pfx/.p12) certificate file"),
    passphrase: str = Form(..., description="Passphrase for the certificate"),
    box_x: int = Form(..., description="X coordinate for the signature box"),
    box_y: int = Form(..., description="Y coordinate for the signature box"),
    page: int = Form(0, description="Page number (0-indexed) where signature should be placed"),
    signature_width: Optional[int] = Form(None, description="Signature box width in points (minimum 144)"),
    signature_height: Optional[int] = Form(None, description="Signature box height in points (minimum 54)"),
    background_img: Optional[UploadFile] = File(None, description="Optional background image file"),
):
    temp_files = []

    try:
        temp_dir = tempfile.mkdtemp()

        # Note: Use synchronous .file.read() instead of await .read()

        # 1. Save uploaded PDF
        input_pdf_path = os.path.join(temp_dir, f"input_{uuid.uuid4().hex}.pdf")
        with open(input_pdf_path, "wb") as f:
            f.write(pdf_file.file.read())
        temp_files.append(input_pdf_path)

        # 2. Save uploaded PFX
        pfx_path = os.path.join(temp_dir, f"cert_{uuid.uuid4().hex}.pfx")
        with open(pfx_path, "wb") as f:
            f.write(pfx_file.file.read())
        temp_files.append(pfx_path)

        # 3. Save optional background image
        bg_img_path = None
        if background_img and background_img.filename:
            bg_img_path = os.path.join(temp_dir, f"bg_{uuid.uuid4().hex}_{background_img.filename}")
            with open(bg_img_path, "wb") as f:
                f.write(background_img.file.read())
            temp_files.append(bg_img_path)

        output_pdf_path = os.path.join(temp_dir, f"signed_{uuid.uuid4().hex}.pdf")

        sig_config = SignatureConfig(
            box=(box_x, box_y),
            pfx_path=pfx_path,
            passphrase=passphrase.encode("utf-8"),
            background_img=bg_img_path,
            size=clamp_signature_size(signature_width, signature_height),
        )

        sign_pdf(
            input_pdf=input_pdf_path,
            output_pdf=output_pdf_path,
            signature=sig_config,
            page=page,
        )

        return FileResponse(
            path=output_pdf_path,
            filename=f"signed_{pdf_file.filename}",
            media_type="application/pdf",
        )

    except Exception as e:
        delete_temp_files(temp_files)
        raise HTTPException(status_code=400, detail=f"PDF signing failed: {str(e)}")


@app.post("/sign-pdf-multiple")
def sign_pdf_multiple_endpoint(
    pdf_file: UploadFile = File(..., description="The PDF document to sign"),
    pfx_file: UploadFile = File(..., description="The PKCS#12 (.pfx/.p12) certificate file"),
    passphrase: str = Form(..., description="Passphrase for the certificate"),
    page_settings: str = Form(
        ...,
        description=(
            "JSON array, one object per signature, applied in order: "
            '[{"page": 0, "box_x": 50, "box_y": 50, "signature_width": 144, "signature_height": 54}, ...]. '
            "page (0-indexed), box_x and box_y are required; signature_width / signature_height "
            "are optional and never go below 144 x 54."
        ),
    ),
    background_img: Optional[UploadFile] = File(None, description="Optional background image file"),
):
    temp_files = []

    try:
        # Validate the settings first, before touching any files.
        placements = parse_page_settings(page_settings)

        temp_dir = tempfile.mkdtemp()

        # 1. Save uploaded PDF
        input_pdf_path = os.path.join(temp_dir, f"input_{uuid.uuid4().hex}.pdf")
        with open(input_pdf_path, "wb") as f:
            f.write(pdf_file.file.read())
        temp_files.append(input_pdf_path)

        # 2. Save uploaded PFX
        pfx_path = os.path.join(temp_dir, f"cert_{uuid.uuid4().hex}.pfx")
        with open(pfx_path, "wb") as f:
            f.write(pfx_file.file.read())
        temp_files.append(pfx_path)

        # 3. Save optional background image
        bg_img_path = None
        if background_img and background_img.filename:
            bg_img_path = os.path.join(temp_dir, f"bg_{uuid.uuid4().hex}_{background_img.filename}")
            with open(bg_img_path, "wb") as f:
                f.write(background_img.file.read())
            temp_files.append(bg_img_path)

        output_pdf_path = os.path.join(temp_dir, f"signed_{uuid.uuid4().hex}.pdf")

        sign_pdf_multiple(
            input_pdf=input_pdf_path,
            output_pdf=output_pdf_path,
            pfx_path=pfx_path,
            passphrase=passphrase.encode("utf-8"),
            placements=placements,
            background_img=bg_img_path,
        )

        return FileResponse(
            path=output_pdf_path,
            filename=f"signed_{pdf_file.filename}",
            media_type="application/pdf",
        )

    except Exception as e:
        delete_temp_files(temp_files)
        raise HTTPException(status_code=400, detail=f"PDF signing failed: {str(e)}")