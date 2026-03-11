"""
finetune_pad_ufes.py — Step 3: PAD-UFES-20 fine-tuning
========================================================
Foundation : ddi_finetuned.keras (or ham10000_baseline.keras)
Dataset    : PAD-UFES-20 — smartphone clinical photos, Brazil
Goal       : Adapt to non-dermoscopy images, improve real-world generalization
Output     : pad_ufes_finetuned.keras

All runs automatically logged to MLflow.

Usage:
    python3 backend/models/finetune_pad_ufes.py
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

# MLflow
try:
    import mlflow
    import mlflow.keras
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("skin-cancer-detection")
    MLFLOW_ENABLED = True
    print("✓ MLflow enabled")
except ImportError:
    MLFLOW_ENABLED = False
    print("⚠ MLflow not installed — run: pip install mlflow --break-system-packages")

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE        = 224
BATCH_SIZE      = 8
WARMUP_EPOCHS   = 5
FINETUNE_EPOCHS = 30
WARMUP_LR       = 5e-5    # Smaller than DDI — we're further from source domain
FINETUNE_LR     = 5e-7
UNFREEZE_LAYERS = 15      # Fewer — preserve more learned features

PARAMS = {
    "model":            "EfficientNetB0",
    "dataset":          "PAD-UFES-20",
    "img_size":         IMG_SIZE,
    "batch_size":       BATCH_SIZE,
    "warmup_epochs":    WARMUP_EPOCHS,
    "finetune_epochs":  FINETUNE_EPOCHS,
    "warmup_lr":        WARMUP_LR,
    "finetune_lr":      FINETUNE_LR,
    "unfreeze_layers":  UNFREEZE_LAYERS,
    "loss":             "binary_crossentropy",
    "domain_shift":     "dermoscopy->smartphone",
    "notes":            "Final fine-tuning step. Smaller LR to preserve DDI+HAM knowledge.",
}

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(BASE_DIR)
SAVED_DIR   = os.path.join(BACKEND_DIR, "saved_models")

# Use best available foundation
def find_foundation():
    for name in ["ddi_finetuned.keras", "ham10000_baseline.keras"]:
        p = os.path.join(SAVED_DIR, name)
        if os.path.exists(p):
            print(f"Foundation: {name}")
            return p
    raise FileNotFoundError("No foundation model found")

FOUNDATION_MODEL = find_foundation()
WARMUP_CKPT      = os.path.join(SAVED_DIR, "pad_ufes_warmup.keras")
OUTPUT_MODEL     = os.path.join(SAVED_DIR, "pad_ufes_finetuned.keras")
HISTORY_PATH     = os.path.join(SAVED_DIR, "pad_ufes_history.pkl")

PAD_DIR        = os.path.join(BACKEND_DIR, "data", "pad_ufes_20")
PAD_META       = os.path.join(PAD_DIR, "metadata.csv")

# ── GPU ───────────────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✓ GPU: {[g.name for g in gpus]}")

# ── Load foundation model ─────────────────────────────────────────────────────
print(f"\n── Loading foundation model ──")
model = tf.keras.models.load_model(FOUNDATION_MODEL)
print(f"✓ Loaded. Layers: {len(model.layers)}")

# ── Load PAD-UFES-20 ──────────────────────────────────────────────────────────
print(f"\n── Loading PAD-UFES-20 ──")
df = pd.read_csv(PAD_META)
print(f"Columns: {list(df.columns)}")
print(f"Total rows: {len(df)}")

# Find image path column
img_col = None
for col in ["img_id", "image_id", "filename", "file", "image"]:
    if col in df.columns:
        img_col = col
        break

if img_col is None:
    # Try to find images folder and match
    images_dir = os.path.join(PAD_DIR, "images")
    if not os.path.exists(images_dir):
        # Some versions have images directly in pad_ufes_20/
        images_dir = PAD_DIR
else:
    images_dir = os.path.join(PAD_DIR, "images")
    if not os.path.exists(images_dir):
        images_dir = PAD_DIR

print(f"Images dir: {images_dir}")

def build_image_path(row):
    if img_col:
        fname = str(row[img_col])
        for ext in ["", ".png", ".jpg", ".jpeg"]:
            p = os.path.join(images_dir, fname + ext)
            if os.path.exists(p):
                return p
    return None

df["image_path"] = df.apply(build_image_path, axis=1)
df = df[df["image_path"].notna()].copy()
print(f"Images found: {len(df)}")

if len(df) == 0:
    print(f"\n⚠ No images found. Contents of {PAD_DIR}:")
    for f in os.listdir(PAD_DIR):
        print(f"  {f}")
    raise FileNotFoundError("Could not find PAD-UFES images. Check folder structure.")

# Detect label column
malignant_terms = ["mel", "melanoma", "bcc", "scc", "malignant",
                   "squamous", "basal", "actinic"]
label_col = None
for col in ["diagnostic", "label", "diagnosis", "malignant", "benign_malignant"]:
    if col in df.columns:
        label_col = col
        break

if label_col is None:
    raise ValueError(f"Cannot find label column. Columns: {list(df.columns)}")

sample = str(df[label_col].iloc[0]).lower()
if sample in ("true", "false", "0", "1"):
    df["label"] = df[label_col].astype(str).str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0}).fillna(0).astype(int)
else:
    df["label"] = df[label_col].str.lower().apply(
        lambda x: 1 if any(t in str(x) for t in malignant_terms) else 0)

print(f"Malignant : {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
print(f"Benign    : {(df['label']==0).sum()} ({(1-df['label'].mean())*100:.1f}%)")

# ── Split ─────────────────────────────────────────────────────────────────────
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42)
print(f"\nTrain: {len(train_df)}  |  Val: {len(val_df)}")

# ── Sample weights ────────────────────────────────────────────────────────────
n_neg        = (train_df["label"] == 0).sum()
n_pos        = (train_df["label"] == 1).sum()
weight_for_0 = 1.0
weight_for_1 = float(np.sqrt(float(n_neg) / float(n_pos))) if n_pos > 0 else 1.0
print(f"Sample weight: malignant = {weight_for_1:.2f}x")

# ── Augmentation ──────────────────────────────────────────────────────────────
# PAD-UFES are smartphone photos — add more realistic augmentations
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.RandomRotate90(),
    A.HorizontalFlip(),
    A.Transpose(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.6),
    A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=35, p=0.5),
    A.GaussNoise(p=0.3),
    A.Blur(blur_limit=3, p=0.2),           # Simulate phone camera blur
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
# TRAINING with MLflow tracking
# ══════════════════════════════════════════════════════════════════════════════
run_context = mlflow.start_run(run_name="pad_ufes_finetuning") \
    if MLFLOW_ENABLED else __import__("contextlib").nullcontext()

with run_context:
    if MLFLOW_ENABLED:
        mlflow.log_params(PARAMS)
        mlflow.log_params({
            "n_train":     len(train_df),
            "n_val":       len(val_df),
            "n_malignant": int(df["label"].sum()),
            "n_benign":    int((df["label"]==0).sum()),
            "weight_pos":  round(weight_for_1, 3),
            "foundation":  os.path.basename(FOUNDATION_MODEL),
        })

    # ── PHASE 1: Warmup ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 1: Head warmup — {WARMUP_EPOCHS} epochs, base FROZEN")
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
        data_generator(train_df, train_transform, BATCH_SIZE, weighted=True),
        steps_per_epoch=steps_train,
        validation_data=data_generator(val_df, val_transform, BATCH_SIZE, shuffle=False),
        validation_steps=steps_val,
        epochs=WARMUP_EPOCHS,
        callbacks=cb_warmup,
    )
    best_p1 = max(h1.history["val_auc"])
    print(f"\n✓ Phase 1 best val_auc = {best_p1:.4f}")

    if MLFLOW_ENABLED:
        mlflow.log_metric("phase1_best_auc", best_p1)
        for i, auc in enumerate(h1.history["val_auc"]):
            mlflow.log_metric("warmup_val_auc", auc, step=i)

    # ── PHASE 2: Fine-tune ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 2: Fine-tuning top {UNFREEZE_LAYERS} layers, LR={FINETUNE_LR}")
    print(f"{'='*60}")

    model.load_weights(WARMUP_CKPT)

    base_model = None
    for layer in model.layers:
        if hasattr(layer, "layers") and len(layer.layers) > 10:
            base_model = layer
            break

    if base_model:
        base_model.trainable = True
        for layer in base_model.layers[:-UNFREEZE_LAYERS]:
            layer.trainable = False
        print(f"Unfrozen: {sum(1 for l in base_model.layers if l.trainable)} base layers")
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
        data_generator(train_df, train_transform, BATCH_SIZE, weighted=True),
        steps_per_epoch=steps_train,
        validation_data=data_generator(val_df, val_transform, BATCH_SIZE, shuffle=False),
        validation_steps=steps_val,
        epochs=FINETUNE_EPOCHS,
        callbacks=cb_finetune,
    )
    best_p2 = max(h2.history["val_auc"])

    # ── Save history ──────────────────────────────────────────────────────────
    combined = {}
    for k in h1.history:
        combined[k] = h1.history[k] + h2.history.get(k, [])
    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(combined, f)

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n── Evaluating on PAD-UFES val set ──")
    val_gen = data_generator(val_df, val_transform, BATCH_SIZE, shuffle=False)
    y_true, y_proba = [], []
    for _ in range(steps_val):
        imgs, lbls = next(val_gen)
        preds = model.predict(imgs, verbose=0)
        y_true.extend(lbls)
        y_proba.extend(preds.flatten())

    y_true  = np.array(y_true)
    y_proba = np.array(y_proba)
    final_auc = roc_auc_score(y_true, y_proba)

    print(classification_report(
        y_true, (y_proba > 0.4).astype(int),
        target_names=["Benign", "Malignant"]))

    if MLFLOW_ENABLED:
        mlflow.log_metric("phase2_best_auc", best_p2)
        mlflow.log_metric("final_val_auc",   final_auc)
        mlflow.log_metric("weight_pos",      weight_for_1)
        for i, auc in enumerate(h2.history["val_auc"]):
            mlflow.log_metric("finetune_val_auc", auc, step=i)
        mlflow.log_artifact(OUTPUT_MODEL, artifact_path="models")
        mlflow.log_artifact(HISTORY_PATH, artifact_path="history")
        mlflow.set_tag("stage",  "pad_ufes_finetuning")
        mlflow.set_tag("status", "completed")
        print("✓ Run logged to MLflow")

    print(f"\n{'='*60}")
    print("SUMMARY — PAD-UFES Fine-tuning")
    print(f"{'='*60}")
    print(f"Phase 1 best val_auc : {best_p1:.4f}")
    print(f"Phase 2 best val_auc : {best_p2:.4f}")
    print(f"Final ROC-AUC        : {final_auc:.4f}")
    print(f"Model saved to       : {OUTPUT_MODEL}")
    print(f"\n✓ Pipeline complete!")
    print(f"  Foundation (HAM10000) : 0.7691 AUC")
    print(f"  After DDI fine-tuning : see ddi_history.pkl")
    print(f"  After PAD-UFES        : {final_auc:.4f} AUC")
    print(f"\nNext steps:")
    print(f"  1. python3 backend/models/log_existing_runs.py")
    print(f"  2. mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001")
    print(f"  3. python3 backend/models/evaluate.py --model_path backend/saved_models/pad_ufes_finetuned.keras")
