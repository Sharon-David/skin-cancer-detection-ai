"""
inference.py — Model loading, prediction, and Monte Carlo uncertainty estimation
"""
import os
import io
import numpy as np
from PIL import Image
import tensorflow as tf
import albumentations as A

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_XLA"]        = "0"
os.environ["TF_XLA_FLAGS"]         = "--tf_xla_auto_jit=0"

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE   = 224
MC_SAMPLES = 20   # Monte Carlo dropout forward passes

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_DIR  = os.path.join(BASE_DIR, "saved_models")

# Model priority: use best available model
def _find_model():
    candidates = [
        "pad_ufes_finetuned.keras",
        "ddi_finetuned.keras",
        "ham10000_baseline.keras",
    ]
    for name in candidates:
        path = os.path.join(SAVED_DIR, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No model found in {SAVED_DIR}")

MODEL_PATH = _find_model()

# ── Lazy singleton ────────────────────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        print(f"Loading model: {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✓ Model loaded ({len(_model.layers)} layers)")
    return _model

# ── Preprocessing ─────────────────────────────────────────────────────────────
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

def preprocess(image_bytes: bytes) -> np.ndarray:
    img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr    = np.array(img)
    arr    = val_transform(image=arr)["image"]
    return np.expand_dims(arr, axis=0).astype(np.float32)

# ── Risk helpers ──────────────────────────────────────────────────────────────
def get_risk_level(prob: float) -> str:
    if prob >= 75:  return "High"
    if prob >= 40:  return "Medium"
    return "Low"

def get_recommendation(prob: float) -> str:
    if prob >= 75:
        return "High probability of malignancy detected. Please consult a dermatologist promptly."
    if prob >= 40:
        return "Moderate risk detected. Consider a dermatologist consultation for further evaluation."
    return "Low risk detected. Continue regular skin self-examinations and annual dermatologist checkups."

# ── Standard prediction ───────────────────────────────────────────────────────
def predict(image_bytes: bytes) -> dict:
    model          = get_model()
    x              = preprocess(image_bytes)
    prob_malignant = float(model.predict(x, verbose=0)[0][0]) * 100
    prob_benign    = 100.0 - prob_malignant
    is_malignant   = prob_malignant >= 50
    confidence     = prob_malignant if is_malignant else prob_benign

    return {
        "prediction":      "Malignant" if is_malignant else "Benign",
        "confidence":      round(confidence, 1),
        "prob_malignant":  round(prob_malignant, 1),
        "prob_benign":     round(prob_benign, 1),
        "risk_level":      get_risk_level(prob_malignant),
        "recommendation":  get_recommendation(prob_malignant),
        "disclaimer":      "This is a research tool only. Not a medical device. Always consult a qualified dermatologist.",
        "gradcam_image":   None,
        "uncertainty":     None,
    }

# ── Monte Carlo Dropout uncertainty estimation ────────────────────────────────
def predict_with_uncertainty(image_bytes: bytes) -> dict:
    """
    Run MC_SAMPLES forward passes with dropout ENABLED.
    Returns mean prediction + uncertainty (std deviation).
    Wide std = model is uncertain = tell user to see a doctor.
    """
    model = get_model()
    x     = preprocess(image_bytes)

    # Run multiple stochastic forward passes (dropout stays ON)
    preds = []
    for _ in range(MC_SAMPLES):
        # training=True keeps dropout active during inference
        p = model(x, training=True).numpy()[0][0]
        preds.append(p)

    preds          = np.array(preds)
    mean_prob      = float(np.mean(preds)) * 100
    std_prob       = float(np.std(preds))  * 100   # uncertainty
    prob_benign    = 100.0 - mean_prob
    is_malignant   = mean_prob >= 50
    confidence     = mean_prob if is_malignant else prob_benign

    # Uncertainty interpretation
    if std_prob > 15:
        uncertainty_label = "High — model is uncertain, strongly recommend clinical review"
    elif std_prob > 8:
        uncertainty_label = "Moderate — consider clinical review"
    else:
        uncertainty_label = "Low — model is confident"

    return {
        "prediction":       "Malignant" if is_malignant else "Benign",
        "confidence":       round(confidence, 1),
        "prob_malignant":   round(mean_prob, 1),
        "prob_benign":      round(prob_benign, 1),
        "risk_level":       get_risk_level(mean_prob),
        "recommendation":   get_recommendation(mean_prob),
        "disclaimer":       "This is a research tool only. Not a medical device. Always consult a qualified dermatologist.",
        "gradcam_image":    None,
        "uncertainty":      {
            "std":   round(std_prob, 1),
            "label": uncertainty_label,
            "samples": MC_SAMPLES,
        },
    }
