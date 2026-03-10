"""
ood_detection.py — Out-of-Distribution detection

Flags images that are too different from training data
(e.g. photos of faces, food, landscapes instead of skin lesions).
The model should not be trusted on these inputs.
"""

import io
import numpy as np
from PIL import Image, ImageStat

# Thresholds tuned on HAM10000 training set characteristics
MIN_BRIGHTNESS  = 20    # Too dark
MAX_BRIGHTNESS  = 235   # Too bright / washed out
MIN_RESOLUTION  = 64    # Too small
MAX_RESOLUTION  = 5000  # Unreasonably large
MIN_SATURATION  = 5     # Completely greyscale (unlikely dermoscopy)
MAX_ASPECT_RATIO = 4.0  # Extreme panoramic / portrait crops

def is_ood(image_bytes: bytes) -> dict:
    """
    Check if an image is likely out-of-distribution.

    Returns:
        {
            "is_ood": bool,
            "confidence": float,   # 0.0 = definitely in-dist, 1.0 = definitely OOD
            "reasons": list[str]   # Human-readable flags
        }
    """
    reasons = []
    flags   = 0

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return {"is_ood": True, "confidence": 1.0, "reasons": ["Cannot open image"]}

    w, h = img.size

    # ── Resolution check ──────────────────────────────────────────────────────
    if w < MIN_RESOLUTION or h < MIN_RESOLUTION:
        reasons.append(f"Image too small ({w}x{h}px). Minimum: {MIN_RESOLUTION}px")
        flags += 2

    if w > MAX_RESOLUTION or h > MAX_RESOLUTION:
        reasons.append(f"Image unusually large ({w}x{h}px)")
        flags += 1

    # ── Aspect ratio check ────────────────────────────────────────────────────
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > MAX_ASPECT_RATIO:
        reasons.append(f"Unusual aspect ratio ({aspect:.1f}:1). Likely not a lesion image.")
        flags += 2

    # ── Brightness check ──────────────────────────────────────────────────────
    stat       = ImageStat.Stat(img)
    brightness = sum(stat.mean) / 3

    if brightness < MIN_BRIGHTNESS:
        reasons.append(f"Image too dark (brightness={brightness:.0f}/255)")
        flags += 1

    if brightness > MAX_BRIGHTNESS:
        reasons.append(f"Image too bright/washed out (brightness={brightness:.0f}/255)")
        flags += 1

    # ── Saturation check ──────────────────────────────────────────────────────
    img_hsv    = img.convert("HSV") if hasattr(img, "convert") else img
    try:
        img_hsv    = img.convert("HSV")
        sat_stat   = ImageStat.Stat(img_hsv)
        saturation = sat_stat.mean[1]
        if saturation < MIN_SATURATION:
            reasons.append(f"Very low saturation ({saturation:.0f}) — may not be a skin image")
            flags += 1
    except Exception:
        pass

    # ── Scoring ───────────────────────────────────────────────────────────────
    is_ood_result  = flags >= 2
    confidence     = min(1.0, flags / 4.0)

    return {
        "is_ood":      is_ood_result,
        "confidence":  round(confidence, 2),
        "reasons":     reasons,
        "flags":       flags,
    }
