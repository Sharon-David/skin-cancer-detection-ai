"""
inference.py — Ensemble inference for skin cancer detection
============================================================
Model priority:
  1. Ensemble: ham10000_baseline + pad_ufes_finetuned (best)
  2. Single:   ham10000_baseline (dermoscopy only)

Domain shift finding:
  - HAM10000 model: trained on dermoscopy images
  - PAD-UFES model: trained on smartphone clinical photos
  - Ensemble: averages both, robust across image types
"""

import os
import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
import albumentations as A

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(BASE_DIR)
SAVED_DIR   = os.path.join(BACKEND_DIR, "saved_models")

MODEL_PATHS = {
    "dermoscopy": os.path.join(SAVED_DIR, "ham10000_baseline.keras"),
    "smartphone": os.path.join(SAVED_DIR, "pad_ufes_finetuned.keras"),
    "ddi":        os.path.join(SAVED_DIR, "ddi_finetuned.keras"),
}

IMG_SIZE = 224

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# ── Model loader ──────────────────────────────────────────────────────────────
_models = {}

def _load_model(key: str):
    if key not in _models:
        path = MODEL_PATHS[key]
        if os.path.exists(path):
            print(f"Loading model: {key} from {path}")
            _models[key] = tf.keras.models.load_model(path)
        else:
            _models[key] = None
    return _models[key]

def get_available_models():
    available = []
    for key, path in MODEL_PATHS.items():
        if os.path.exists(path):
            available.append(key)
    return available

def _preprocess(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB")
    img = np.array(img)
    img = transform(image=img)["image"]
    return np.expand_dims(img, axis=0).astype(np.float32)

# ── Risk level helper ─────────────────────────────────────────────────────────
def _risk_level(prob: float) -> tuple[str, str]:
    if prob >= 0.7:
        return "High", "Consult a dermatologist promptly."
    elif prob >= 0.4:
        return "Moderate", "Monitor closely and consider a dermatologist visit."
    else:
        return "Low", "Low concern, but monitor for changes."

# ── Single model predict ──────────────────────────────────────────────────────
def predict(pil_image: Image.Image) -> dict:
    """Single model prediction using best available model."""
    img = _preprocess(pil_image)

    # Load best single model
    model = _load_model("dermoscopy")
    if model is None:
        raise RuntimeError("No model available")

    prob = float(model.predict(img, verbose=0)[0][0])
    risk, recommendation = _risk_level(prob)

    return {
        "prediction":    "Malignant" if prob >= 0.5 else "Benign",
        "confidence":    round(max(prob, 1 - prob) * 100, 1),
        "prob_malignant": round(prob * 100, 1),
        "prob_benign":    round((1 - prob) * 100, 1),
        "risk_level":    risk,
        "recommendation": recommendation,
        "model_used":    "dermoscopy",
        "disclaimer":    "Research tool only. Not a clinical diagnosis.",
    }

# ── Monte Carlo uncertainty ───────────────────────────────────────────────────
def _mc_predict(model, img: np.ndarray, n_samples: int = 20) -> tuple[float, float]:
    """Run MC dropout inference."""
    preds = []
    for _ in range(n_samples):
        p = float(model(img, training=True)[0][0])
        preds.append(p)
    return float(np.mean(preds)), float(np.std(preds))

# ── ENSEMBLE predict (main function) ─────────────────────────────────────────
def predict_with_uncertainty(pil_image: Image.Image) -> dict:
    """
    Ensemble prediction with uncertainty estimation.
    
    Uses dermoscopy model + smartphone model if both available.
    Falls back to single model with MC dropout if only one available.
    
    Returns full result including:
    - ensemble prediction + confidence
    - per-model predictions (for frontend display)
    - uncertainty estimate
    - domain shift note if models disagree significantly
    """
    img = _preprocess(pil_image)

    derm_model = _load_model("dermoscopy")
    sphone_model = _load_model("smartphone")

    available = []
    if derm_model is not None:
        available.append("dermoscopy")
    if sphone_model is not None:
        available.append("smartphone")

    if not available:
        raise RuntimeError("No models available")

    # ── Single model fallback ──────────────────────────────────────────────────
    if len(available) == 1:
        model = derm_model or sphone_model
        mean_prob, std_prob = _mc_predict(model, img)

        if std_prob > 0.15:
            uncertainty_label = "High"
        elif std_prob > 0.08:
            uncertainty_label = "Moderate"
        else:
            uncertainty_label = "Low"

        risk, recommendation = _risk_level(mean_prob)

        return {
            "prediction":      "Malignant" if mean_prob >= 0.5 else "Benign",
            "confidence":      round(max(mean_prob, 1 - mean_prob) * 100, 1),
            "prob_malignant":  round(mean_prob * 100, 1),
            "prob_benign":     round((1 - mean_prob) * 100, 1),
            "risk_level":      risk,
            "recommendation":  recommendation,
            "model_used":      available[0],
            "ensemble":        False,
            "uncertainty": {
                "std":   round(std_prob * 100, 1),
                "label": uncertainty_label,
                "samples": 20,
            },
            "per_model": {
                available[0]: round(mean_prob * 100, 1),
            },
            "domain_shift_note": None,
            "disclaimer": "Research tool only. Not a clinical diagnosis.",
        }

    # ── Ensemble: both models available ──────────────────────────────────────
    derm_mean,   derm_std   = _mc_predict(derm_model,   img, n_samples=20)
    sphone_mean, sphone_std = _mc_predict(sphone_model, img, n_samples=20)

    # Weighted average — dermoscopy model has higher validated AUC
    derm_weight   = 0.55
    sphone_weight = 0.45
    ensemble_prob = derm_weight * derm_mean + sphone_weight * sphone_mean
    ensemble_std  = np.sqrt(
        derm_weight**2 * derm_std**2 + sphone_weight**2 * sphone_std**2
    )

    # Domain shift detection — models disagree significantly
    model_disagreement = abs(derm_mean - sphone_mean)
    domain_shift_note = None
    if model_disagreement > 0.25:
        domain_shift_note = (
            f"Models disagree by {round(model_disagreement*100)}% — "
            f"image may have features atypical for both dermoscopy and smartphone domains. "
            f"Higher uncertainty applies."
        )

    if ensemble_std > 0.15:
        uncertainty_label = "High"
    elif ensemble_std > 0.08:
        uncertainty_label = "Moderate"
    else:
        uncertainty_label = "Low"

    risk, recommendation = _risk_level(ensemble_prob)

    return {
        "prediction":     "Malignant" if ensemble_prob >= 0.5 else "Benign",
        "confidence":     round(max(ensemble_prob, 1 - ensemble_prob) * 100, 1),
        "prob_malignant": round(ensemble_prob * 100, 1),
        "prob_benign":    round((1 - ensemble_prob) * 100, 1),
        "risk_level":     risk,
        "recommendation": recommendation,
        "model_used":     "ensemble",
        "ensemble":       True,
        "uncertainty": {
            "std":     round(float(ensemble_std) * 100, 1),
            "label":   uncertainty_label,
            "samples": 20,
        },
        "per_model": {
            "dermoscopy": round(derm_mean * 100, 1),
            "smartphone": round(sphone_mean * 100, 1),
        },
        "domain_shift_note": domain_shift_note,
        "disclaimer": "Research tool only. Not a clinical diagnosis.",
    }
