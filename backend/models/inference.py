"""
inference.py — Model loading and prediction
"""
import os
import io
import numpy as np
from PIL import Image
import tensorflow as tf
import albumentations as A
import base64

# ── Suppress TF noise ─────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_XLA"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE = 224
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "ham10000_baseline.keras")

# ── Lazy singleton — model loads once on first request ────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        print(f"Loading model from {MODEL_PATH}...")
        _model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded.")
    return _model

# ── Preprocessing ─────────────────────────────────────────────────────────────
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)
    img_np = val_transform(image=img_np)["image"]
    return np.expand_dims(img_np, axis=0).astype(np.float32)

# ── Risk helpers ──────────────────────────────────────────────────────────────
def get_risk_level(prob: float) -> str:
    if prob >= 0.75:  return "High"
    if prob >= 0.40:  return "Medium"
    return "Low"

def get_recommendation(prob: float) -> str:
    if prob >= 0.75:
        return "High probability of malignancy detected. Please consult a dermatologist promptly."
    if prob >= 0.40:
        return "Moderate risk detected. Consider a dermatologist consultation for further evaluation."
    return "Low risk detected. Continue regular skin self-examinations and annual dermatologist checkups."

# ── Main predict function ─────────────────────────────────────────────────────
def predict(image_bytes: bytes) -> dict:
    model = get_model()
    x = preprocess(image_bytes)

    prob_malignant = float(model.predict(x, verbose=0)[0][0])
    prob_benign    = 1.0 - prob_malignant
    is_malignant   = prob_malignant >= 0.5
    confidence     = prob_malignant if is_malignant else prob_benign

    return {
        "prediction":      "Malignant" if is_malignant else "Benign",
        "confidence":      round(confidence * 100, 1),
        "prob_malignant":  round(prob_malignant * 100, 1),
        "prob_benign":     round(prob_benign * 100, 1),
        "risk_level":      get_risk_level(prob_malignant),
        "recommendation":  get_recommendation(prob_malignant),
        "disclaimer":      "This is a research tool only. Not a medical device. Always consult a qualified dermatologist.",
        "gradcam_image":   None,  # Will be added when explainability.py is wired in
    }
