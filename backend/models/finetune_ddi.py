"""
finetune_ddi.py — Step 2 of sequential fine-tuning pipeline
=============================================================
Foundation : ham10000_baseline.keras (AUC 0.769)
Goal       : Improve fairness across skin tones using DDI dataset
Output     : ddi_finetuned.keras

Usage:
    python backend/models/finetune_ddi.py
"""

import os
import sys

os.environ["TF_XLA_FLAGS"]          = "--tf_xla_auto_jit=0"
os.environ["TF_ENABLE_XLA"]         = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["XLA_FLAGS"]             = "--xla_gpu_autotune_level=0"

import pickle
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
tf.config.optimizer.set_jit(False)

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import albumentations as A

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE        = 224
BATCH_SIZE      = 8
FINETUNE_EPOCHS = 30
FINETUNE_LR     = 1e-6      # Very small — preserve HAM10000 knowledge
UNFREEZE_LAYERS = 20

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(BASE_DIR)
SAVED_DIR   = os.path.join(BACKEND_DIR, "saved_models")

FOUNDATION_MODEL = os.path.join(SAVED_DIR, "ham10000_baseline.keras")
WARMUP_CKPT      = os.path.join(SAVED_DIR, "ddi_warmup.keras")
OUTPUT_MODEL     = os.path.join(SAVED_DIR, "ddi_finetuned.keras")
HISTORY_PATH     = os.path.join(SAVED_DIR, "ddi_history.pkl")

DDI_DIR        = os.path.join(BACKEND_DIR, "data", "ddi")
DDI_IMAGES_DIR = os.path.join(DDI_DIR, "images")
DDI_META       = os.path.join(DDI_DIR, "ddi_metadata.csv")

# ── GPU ───────────────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU: {[g.name for g in gpus]}")
    except RuntimeError as e:
        print(f"GPU error: {e}")
else:
    print("⚠ No GPU detected — training on CPU")

print(f"\n── Config ──")
print(f"Foundation : {FOUNDATION_MODEL}")
print(f"Output     : {OUTPUT_MODEL}")
print(f"LR         : {FINETUNE_LR}  (tiny — preserve HAM10000 knowledge)")
print(f"Unfreeze   : {UNFREEZE_LAYERS} layers")

# ── Load foundation model ─────────────────────────────────────────────────────
print(f"\n── Loading foundation model ──")
if not os.path.exists(FOUNDATION_MODEL):
    raise FileNotFoundError(f"Foundation model not found: {FOUNDATION_MODEL}")

model = tf.keras.models.load_model(FOUNDATION_MODEL)
print(f"✓ Loaded. Total layers: {len(model.layers)}")

# ── Load DDI metadata ─────────────────────────────────────────────────────────
print(f"\n── Loading DDI dataset ──")
df = pd.read_csv(DDI_META)
print(f"Columns: {list(df.columns)}")

# Build image paths
def find_image(row):
    for col in ["DDI_file", "file", "image_file", "filename", "image_id"]:
        if col in row.index:
            fname = str(row[col])
            full  = os.path.join(DDI_IMAGES_DIR, fname)
            if os.path.exists(full):
                return full
            full = os.path.join(DDI_IMAGES_DIR, os.path.basename(fname))
            if os.path.exists(full):
                return full
    return None

df["image_path"] = df.apply(find_image, axis=1)
df = df[df["image_path"].notna()].copy()
print(f"Images found: {len(df)}")

# Detect label column
label_col = None
for col in ["malignant", "label", "diagnosis", "benign_malignant"]:
    if col in df.columns:
        label_col = col
        break
if label_col is None:
    raise ValueError(f"Cannot find label column. Columns: {list(df.columns)}")

sample = df[label_col].iloc[0]
if isinstance(sample, bool) or str(sample).lower() in ("true", "false"):
    df["label"] = df[label_col].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0).astype(int)
elif df[label_col].dtype in [np.int64, np.float64]:
    df["label"] = df[label_col].astype(int)
else:
    malignant_terms = ["malignant", "melanoma", "bcc", "scc"]
    df["label"] = df[label_col].str.lower().apply(
        lambda x: 1 if any(t in str(x) for t in malignant_terms) else 0)

print(f"Malignant : {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
print(f"Benign    : {(df['label']==0).sum()} ({(1-df['label'].mean())*100:.1f}%)")

# Skin tone breakdown if available
for col in ["skin_tone", "fitzpatrick", "fitzpatrick_scale"]:
    if col in df.columns:
        print(f"\nSkin tone distribution ({col}):")
        print(df[col].value_counts())
        break

# ── Train / val split ─────────────────────────────────────────────────────────
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42)
print(f"\nTrain: {len(train_df)}  |  Val: {len(val_df)}")

# ── Sample weights ────────────────────────────────────────────────────────────
n_neg        = (train_df["label"] == 0).sum()
n_pos        = (train_df["label"] == 1).sum()
weight_for_0 = 1.0
weight_for_1 = float(np.sqrt(float(n_neg) / float(n_pos))) if n_pos > 0 else 1.0
print(f"Sample weight: malignant = {weight_for_1:.2f}x benign")

# ── Augmentation ──────────────────────────────────────────────────────────────
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.RandomRotate90(),
    A.Flip(),
    A.Transpose(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, p=0.5),
    A.GaussNoise(p=0.3),
    A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# ── Generator ─────────────────────────────────────────────────────────────────
def data_generator(dataframe, transform, batch_size, shuffle=True, weighted=False):
    while True:
        df_loop = dataframe.sample(frac=1).reset_index(drop=True) if shuffle else dataframe
        for i in range(0, len(df_loop), batch_size):
            batch = df_loop.iloc[i:i + batch_size]
            images, labels = [], []
            for _, row in batch.iterrows():
                try:
                    img = np.array(Image.open(row["image_path"]).convert("RGB"))
                    img = transform(image=img)["image"]
                    images.append(img)
                    labels.append(row["label"])
                except Exception:
                    pass
            if images:
                imgs_arr   = np.array(images, dtype=np.float32)
                labels_arr = np.array(labels, dtype=np.float32)
                if weighted:
                    sw = np.where(labels_arr == 1, weight_for_1, weight_for_0)
                    yield imgs_arr, labels_arr, sw
                else:
                    yield imgs_arr, labels_arr

steps_train = max(1, len(train_df) // BATCH_SIZE)
steps_val   = max(1, len(val_df)   // BATCH_SIZE)
print(f"Steps/epoch: train={steps_train}, val={steps_val}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Short warmup with DDI (head only, base frozen)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"PHASE 1: Head warmup on DDI — 5 epochs, base FROZEN")
print(f"{'='*60}")

# Freeze entire base
for layer in model.layers:
    layer.trainable = False
# Unfreeze head (last 4 layers = dense layers)
for layer in model.layers[-4:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    run_eagerly=True,
)

cb_warmup = [
    ModelCheckpoint(WARMUP_CKPT, monitor="val_auc",
                    save_best_only=True, mode="max", verbose=1),
    EarlyStopping(monitor="val_auc", patience=4,
                  restore_best_weights=True, mode="max", verbose=1),
]

h1 = model.fit(
    data_generator(train_df, train_transform, BATCH_SIZE, shuffle=True,  weighted=True),
    steps_per_epoch=steps_train,
    validation_data=data_generator(val_df, val_transform, BATCH_SIZE, shuffle=False, weighted=False),
    validation_steps=steps_val,
    epochs=5,
    callbacks=cb_warmup,
)
best_p1 = max(h1.history["val_auc"])
print(f"\n✓ Phase 1 complete. Best val_auc = {best_p1:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Fine-tune top layers on DDI
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"PHASE 2: Fine-tuning top {UNFREEZE_LAYERS} layers on DDI")
print(f"  LR = {FINETUNE_LR}  (tiny — must not destroy HAM10000 knowledge)")
print(f"{'='*60}")

# Load best warmup weights before unfreezing
model.load_weights(WARMUP_CKPT)
print(f"✓ Loaded warmup checkpoint")

# Get the base model (first layer that is EfficientNet)
base_model = model.layers[1] if hasattr(model.layers[1], 'layers') else None
if base_model:
    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False
    unfrozen = sum(1 for l in base_model.layers if l.trainable)
    print(f"Unfrozen: {unfrozen} / {len(base_model.layers)} base layers")
else:
    # Fallback: unfreeze top N layers of whole model
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[-(UNFREEZE_LAYERS + 4):]:
        layer.trainable = True
    print(f"Unfrozen top {UNFREEZE_LAYERS + 4} layers of full model")

model.compile(
    optimizer=Adam(learning_rate=FINETUNE_LR),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    run_eagerly=True,
)

cb_finetune = [
    ModelCheckpoint(OUTPUT_MODEL, monitor="val_auc",
                    save_best_only=True, mode="max", verbose=1),
    EarlyStopping(monitor="val_auc", patience=8,
                  restore_best_weights=True, mode="max", verbose=1),
    ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=4,
                      min_lr=1e-9, mode="max", verbose=1),
]

h2 = model.fit(
    data_generator(train_df, train_transform, BATCH_SIZE, shuffle=True,  weighted=True),
    steps_per_epoch=steps_train,
    validation_data=data_generator(val_df, val_transform, BATCH_SIZE, shuffle=False, weighted=False),
    validation_steps=steps_val,
    epochs=FINETUNE_EPOCHS,
    callbacks=cb_finetune,
)

# ── Save history ──────────────────────────────────────────────────────────────
combined = {}
for k in h1.history:
    combined[k] = h1.history[k] + h2.history.get(k, [])
with open(HISTORY_PATH, "wb") as f:
    pickle.dump(combined, f)
print(f"\n✓ History saved: {HISTORY_PATH}")

# ── Evaluation ────────────────────────────────────────────────────────────────
print("\nEvaluating on DDI validation set...")
val_gen = data_generator(val_df, val_transform, BATCH_SIZE, shuffle=False, weighted=False)
y_true, y_proba = [], []
for _ in range(steps_val):
    imgs, lbls = next(val_gen)
    preds = model.predict(imgs, verbose=0)
    y_true.extend(lbls)
    y_proba.extend(preds.flatten())

y_true  = np.array(y_true)
y_proba = np.array(y_proba)
auc     = roc_auc_score(y_true, y_proba)

print("\n── Classification report (threshold=0.4) ──")
y_pred = (y_proba > 0.4).astype(int)
print(classification_report(y_true, y_pred, target_names=["Benign", "Malignant"]))

print(f"\n{'='*60}")
print("SUMMARY — DDI Fine-tuning")
print(f"{'='*60}")
print(f"Phase 1 best val_auc : {best_p1:.4f}")
print(f"Phase 2 best val_auc : {max(h2.history['val_auc']):.4f}")
print(f"Final ROC-AUC        : {auc:.4f}")
print(f"Model saved to       : {OUTPUT_MODEL}")
print()
print(f"Next: python backend/models/finetune_pad_ufes.py")
