"""
evaluate.py — Model evaluation on HAM10000 validation set

Usage:
    python backend/models/evaluate.py
    python backend/models/evaluate.py --model_path backend/saved_models/ddi_finetuned.keras
"""

import os
import sys
import argparse

os.environ["TF_XLA_FLAGS"]          = "--tf_xla_auto_jit=0"
os.environ["TF_ENABLE_XLA"]         = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
tf.config.optimizer.set_jit(False)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, roc_curve, average_precision_score
)
import albumentations as A

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str,
    default="backend/saved_models/ham10000_baseline.keras")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--img_size",   type=int, default=224)
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR   = os.path.dirname(BASE_DIR)
DATA_DIR      = os.path.join(BACKEND_DIR, "data", "ham10000")
IMAGES_DIR    = os.path.join(DATA_DIR, "images")
METADATA_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
MODEL_PATH    = os.path.join(BACKEND_DIR, "..", args.model_path) \
                if not os.path.isabs(args.model_path) else args.model_path

print(f"Evaluating: {MODEL_PATH}")

# ── GPU ───────────────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ── Load model ────────────────────────────────────────────────────────────────
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✓ Model loaded. Layers: {len(model.layers)}")

# ── Load HAM10000 val set ─────────────────────────────────────────────────────
df = pd.read_csv(METADATA_PATH)
df["image_path"] = df["image_id"].apply(
    lambda x: os.path.join(IMAGES_DIR, f"{x}.jpg"))
df = df[df["image_path"].apply(os.path.exists)].copy()

malignant_classes = ["mel", "bcc", "akiec"]
df["label"] = df["dx"].apply(lambda x: 1 if x in malignant_classes else 0)

_, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
print(f"Val set: {len(val_df)} images")

# ── Transform ─────────────────────────────────────────────────────────────────
val_transform = A.Compose([
    A.Resize(args.img_size, args.img_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# ── Run inference ─────────────────────────────────────────────────────────────
print("Running inference...")
y_true, y_proba = [], []

for i in range(0, len(val_df), args.batch_size):
    batch = val_df.iloc[i:i + args.batch_size]
    images, labels = [], []
    for _, row in batch.iterrows():
        try:
            img = np.array(Image.open(row["image_path"]).convert("RGB"))
            img = val_transform(image=img)["image"]
            images.append(img)
            labels.append(row["label"])
        except Exception:
            pass
    if images:
        preds = model.predict(np.array(images, dtype=np.float32), verbose=0)
        y_true.extend(labels)
        y_proba.extend(preds.flatten())

y_true  = np.array(y_true)
y_proba = np.array(y_proba)
auc     = roc_auc_score(y_true, y_proba)
ap      = average_precision_score(y_true, y_proba)

# ── Print results ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"EVALUATION RESULTS")
print(f"{'='*60}")
print(f"ROC-AUC  : {auc:.4f}")
print(f"Avg Prec : {ap:.4f}")

print(f"\n── Per-threshold breakdown ──")
for thresh in [0.3, 0.4, 0.5]:
    y_pred = (y_proba > thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"\nThreshold = {thresh}")
    print(f"  Sensitivity (recall malignant) : {sensitivity:.3f}")
    print(f"  Specificity (recall benign)    : {specificity:.3f}")
    print(classification_report(y_true, y_pred,
          target_names=["Benign", "Malignant"]))

print(f"\n── Per-class breakdown ──")
for dx in val_df["dx"].unique():
    subset = val_df[val_df["dx"] == dx]
    idxs   = subset.index
    mask   = np.isin(np.where(val_df.index)[0] if False else
             [val_df.index.get_loc(i) for i in idxs if i in val_df.index], range(len(y_true)))
    # Simple per-class count
    n_mal  = (subset["label"] == 1).sum()
    n_ben  = (subset["label"] == 0).sum()
    print(f"  {dx:8s}: {len(subset):4d} samples  (mal={n_mal}, ben={n_ben})")
