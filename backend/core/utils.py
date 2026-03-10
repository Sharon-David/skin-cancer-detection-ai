"""
utils.py — Shared helper functions
"""
from fastapi import HTTPException
from PIL import Image
import io

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_FILE_SIZE_MB = 10

def validate_image_upload(file_bytes: bytes, filename: str) -> None:
    """Raise HTTP 400 if the upload is invalid."""
    # Check file size
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f}MB). Maximum is {MAX_FILE_SIZE_MB}MB."
        )

    # Check extension
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Check it's actually a valid image
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="File does not appear to be a valid image."
        )

def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert a PIL image to bytes."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()
