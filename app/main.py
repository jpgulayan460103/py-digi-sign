import os
import tempfile
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
from fractions import Fraction

from PIL import Image, ImageOps
from dataclasses import dataclass
from typing import List, Tuple, Optional

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
    """Fit the image onto a solid white canvas that matches the signature box.

    - keeps the image's aspect ratio and centers it
    - fills all leftover space with white
    - fades the image (fade=1 -> full strength) so the text on top stays readable
    """
    w, h = int(box_w * scale), int(box_h * scale)
    canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))

    img = Image.open(src_path).convert("RGBA")
    inner = (int(w * (1 - 2 * padding_ratio)), int(h * (1 - 2 * padding_ratio)))
    img = ImageOps.contain(img, inner)  # scales up or down, keeps aspect ratio
    canvas.alpha_composite(img, ((w - img.width) // 2, (h - img.height) // 2))

    white = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    Image.blend(white, canvas, fade).convert("RGB").save(dst_path, "PNG")
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