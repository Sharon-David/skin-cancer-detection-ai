"""
finetune_ddi.py — Step 2: Mixed HAM10000 + DDI training
=========================================================
Problem with DDI-only: only 656 images — too small to fine-tune alone.
Solution: mix DDI with HAM10000 (oversample DDI to match HAM10000 scale).
This teaches the model skin tone diversity without forgetting dermoscopy.

Foundation : ham10000_baseline.keras (AUC 0.769)
Output     : ddi_finetuned.keras

Usage:
    python backend/models/finetune_ddi.py
"""

import os
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
WARMUP_EPOCHS   = 5
FINETUNE_EPOCHS = 30
WARMUP_LR       = 1e-4
FINETUNE_LR     = 1e-6
UNFREEZE_LAYERS = 20
DDI_OVERSAMPLE  = 8    # Repeat DDI samples 8x to match HAM10000 scale

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(BASE_DIR)
SAVED_DIR   = os.path.join(BACKEND_DIR, "saved_models")

FOUNDATION_MODEL = os.path.join(SAVED_DIR, "ham10000_baseline.keras")
WARMUP_CKPT      = os.path.join(SAVED_DIR, "ddi_warmup.keras")
OUTPUT_MODEL     = os.path.join(SAVED_DIR, "ddi_finetuned.keras")
HISTORY_PATH     = os.path.join(SAVED_DIR, "ddi_history.pkl")

HAM_DIR        = os.path.join(BACKEND_DIR, "data", "ham10000")
HAM_IMAGES_DIR = os.path.join(HAM_DIR, "images")
HAM_META       = os.path.join(HAM_DIR, "HAM10000_metadata.csv")

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

print(f"\n── Config ──")
print(f"Foundation       : {FOUNDATION_MODEL}")
print(f"DDI oversample   : {DDI_OVERSAMPLE}x (to balance with HAM10000)")
print(f"Finetune LR      : {FINETUNE_LR}")

# ── Load foundation model ─────────────────────────────────────────────────────
print(f"\n── Loading foundation model ──")
model = tf.keras.models.load_model(FOUNDATION_MODEL)
print(f"✓ Loaded. Layers: {len(model.layers)}")

# ── Load HAM10000 ─────────────────────────────────────────────────────────────
print(f"\n── Loading HAM10000 ──")
ham_df = pd.read_csv(HAM_META)
ham_df["image_path"] = ham_df["image_id"].apply(
    lambda x: os.path.join(HAM_IMAGES_DIR, f"{x}.jpg"))
ham_df = ham_df[ham_df["image_path"].apply(os.path.exists)].copy()
malignant_classes = ["mel", "bcc", "akiec"]
ham_df["label"] = ham_df["dx"].apply(lambda x: 1 if x in malignant_classes else 0)
ham_df["source"] = "ham10000"
print(f"HAM10000: {len(ham_df)} images  (mal={ham_df['label'].sum()}, ben={(ham_df['label']==0).sum()})")

# ── Load DDI ──────────────────────────────────────────────────────────────────
print(f"\n── Loading DDI ──")
ddi_df = pd.read_csv(DDI_META)
ddi_df["image_path"] = ddi_df["DDI_file"].apply(
    lambda x: os.path.join(DDI_IMAGES_DIR, str(x)))
ddi_df = ddi_df[ddi_df["image_path"].apply(os.path.exists)].copy()
ddi_df["label"] = ddi_df["malignant"].astype(str).str.lower().map(
    {"true": 1, "false": 0}).fillna(0).astype(int)
ddi_df["source"] = "ddi"
print(f"DDI: {len(ddi_df)} images  (mal={ddi_df['label'].sum()}, ben={(ddi_df['label']==0).sum()})")
print(f"Skin tones: {ddi_df['skin_tone'].value_counts().to_dict()}")

# ── Split both datasets ───────────────────────────────────────────────────────
ham_train, ham_val = train_test_split(
    ham_df, test_size=0.2, stratify=ham_df["label"], random_state=42)
ddi_train, ddi_val = train_test_split(
    ddi_df, test_size=0.2, stratify=ddi_df["label"], random_state=42)

# ── Oversample DDI in training set ───────────────────────────────────────────
ddi_train_oversampled = pd.concat([ddi_train] * DDI_OVERSAMPLE, ignore_index=True)
print(f"\nDDI train after {DDI_OVERSAMPLE}x oversample: {len(ddi_train_oversampled)}")

# ── Combine: HAM10000 + oversampled DDI ──────────────────────────────────────
train_df = pd.concat([ham_train, ddi_train_oversampled], ignore_index=True)
# Val: keep separate so we can evaluate fairly on each
val_df   = pd.concat([ham_val, ddi_val], ignore_index=True)

print(f"\nCombined train : {len(train_df)}")
print(f"Combined val   : {len(val_df)}")
print(f"  HAM10000 val : {len(ham_val)}")
print(f"  DDI val      : {len(ddi_val)}")

# ── Sample weights ────────────────────────────────────────────────────────────
n_neg        = (train_df["label"] == 0).sum()
n_pos        = (train_df["label"] == 1).sum()
weight_for_0 = 1.0
weight_for_1 = float(np.sqrt(float(n_neg) / float(n_pos)))
print(f"\nSample weight: malignant = {weight_for_1:.2f}x benign")

# ── Augmentation ──────────────────────────────────────────────────────────────
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.RandomRotate90(),
    A.HorizontalFlip(),
    A.Transpose(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, p=0.5),
    A.GaussNoise(p=0.2),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
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
# PHASE 1 — Short warmup (head only)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"PHASE 1: Head warmup — {WARMUP_EPOCHS} epochs")
print(f"{'='*60}")

for layer in model.layers:
    layer.trainable = False
for layer in model.layers[-4:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=WARMUP_LR),
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
    data_generator(train_df, train_transform, BATCH_SIZE, shuffle=True, weighted=True),
    steps_per_epoch=steps_train,
    validation_data=data_generator(val_df, val_transform, BATCH_SIZE, shuffle=False, weighted=False),
    validation_steps=steps_val,
    epochs=WARMUP_EPOCHS,
    callbacks=cb_warmup,
)
best_p1 = max(h1.history["val_auc"])
print(f"\n✓ Phase 1 complete. Best val_auc = {best_p1:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Fine-tune top layers
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"PHASE 2: Fine-tuning top {UNFREEZE_LAYERS} layers")
print(f"{'='*60}")

model.load_weights(WARMUP_CKPT)
print(f"✓ Loaded warmup checkpoint")

# Unfreeze top N layers of base model
base_model = None
for layer in model.layers:
    if hasattr(layer, 'layers') and len(layer.layers) > 10:
        base_model = layer
        break

if base_model:
    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False
    unfrozen = sum(1 for l in base_model.layers if l.trainable)
    print(f"Unfrozen: {unfrozen} / {len(base_model.layers)} base layers")
else:
    for layer in model.layers[-(UNFREEZE_LAYERS + 4):]:
        layer.trainable = True

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
    data_generator(train_df, train_transform, BATCH_SIZE, shuffle=True, weighted=True),
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
print(f"\n✓ History saved")

# ── Evaluation ────────────────────────────────────────────────────────────────
print("\n── Evaluating on HAM10000 val (regression test) ──")
ham_gen = data_generator(ham_val, val_transform, BATCH_SIZE, shuffle=False, weighted=False)
y_true, y_proba = [], []
for _ in range(max(1, len(ham_val) // BATCH_SIZE)):
    imgs, lbls = next(ham_gen)
    preds = model.predict(imgs, verbose=0)
    y_true.extend(lbls)
    y_proba.extend(preds.flatten())
ham_auc = roc_auc_score(np.array(y_true), np.array(y_proba))
print(f"HAM10000 val AUC : {ham_auc:.4f}  (baseline was 0.7691 — should be close)")

print("\n── Evaluating on DDI val (fairness test) ──")
ddi_gen = data_generator(ddi_val, val_transform, BATCH_SIZE, shuffle=False, weighted=False)
y_true, y_proba = [], []
for _ in range(max(1, len(ddi_val) // BATCH_SIZE)):
    imgs, lbls = next(ddi_gen)
    preds = model.predict(imgs, verbose=0)
    y_true.extend(lbls)
    y_proba.extend(preds.flatten())
ddi_auc = roc_auc_score(np.array(y_true), np.array(y_proba))
print(f"DDI val AUC      : {ddi_auc:.4f}")

# Per skin tone AUC
ddi_val_copy = ddi_val.copy().reset_index(drop=True)
print("\n── DDI AUC per skin tone ──")
for tone in sorted(ddi_val_copy["skin_tone"].unique()):
    mask   = ddi_val_copy["skin_tone"] == tone
    subset = ddi_val_copy[mask]
    if len(subset) < 10:
        continue
    gen = data_generator(subset, val_transform, BATCH_SIZE, shuffle=False, weighted=False)
    yt, yp = [], []
    for _ in range(max(1, len(subset) // BATCH_SIZE)):
        imgs, lbls = next(gen)
        preds = model.predict(imgs, verbose=0)
        yt.extend(lbls); yp.extend(preds.flatten())
    if len(set(yt)) > 1:
        tone_auc = roc_auc_score(np.array(yt), np.array(yp))
        print(f"  Skin tone {tone}: AUC={tone_auc:.4f}  (n={len(subset)})")

print(f"\n{'='*60}")
print("SUMMARY — DDI Mixed Fine-tuning")
print(f"{'='*60}")
print(f"Phase 1 best val_auc : {best_p1:.4f}")
print(f"Phase 2 best val_auc : {max(h2.history['val_auc']):.4f}")
print(f"HAM10000 val AUC     : {ham_auc:.4f}  (was 0.7691)")
print(f"DDI val AUC          : {ddi_auc:.4f}")
print(f"Model saved to       : {OUTPUT_MODEL}")
print(f"\nNext: python backend/models/finetune_pad_ufes.py")
