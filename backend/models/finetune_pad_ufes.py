"""
finetune_pad_ufes.py — Step 3: PAD-UFES-20 fine-tuning
========================================================
Foundation : ddi_finetuned.keras (or ham10000_baseline.keras)
Dataset    : PAD-UFES-20 — 2298 smartphone clinical photos, Brazil
Labels     : BCC/SCC/MEL = malignant | NEV/ACK/SEK = benign
Output     : pad_ufes_finetuned.keras

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
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("skin-cancer-detection")
    MLFLOW_ENABLED = True
    print("✓ MLflow enabled")
except ImportError:
    MLFLOW_ENABLED = False

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE        = 224
BATCH_SIZE      = 8
WARMUP_EPOCHS   = 10
FINETUNE_EPOCHS = 40
WARMUP_LR       = 1e-4
FINETUNE_LR     = 5e-7
UNFREEZE_LAYERS = 15

# PAD-UFES label mapping — confirmed from metadata
MALIGNANT = {"bcc", "scc", "mel"}
BENIGN    = {"nev", "ack", "sek"}

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(BASE_DIR)
SAVED_DIR   = os.path.join(BACKEND_DIR, "saved_models")

def find_foundation():
    for name in ["ham10000_baseline.keras"]:
        p = os.path.join(SAVED_DIR, name)
        if os.path.exists(p):
            print(f"Foundation: {name}")
            return p, name
    raise FileNotFoundError("No foundation model found")

FOUNDATION_MODEL, FOUNDATION_NAME = find_foundation()
WARMUP_CKPT  = os.path.join(SAVED_DIR, "pad_ufes_warmup.keras")
OUTPUT_MODEL = os.path.join(SAVED_DIR, "pad_ufes_finetuned.keras")
HISTORY_PATH = os.path.join(SAVED_DIR, "pad_ufes_history.pkl")

PAD_DIR        = os.path.join(BACKEND_DIR, "data", "pad_ufes_20")
PAD_IMAGES_DIR = os.path.join(PAD_DIR, "images")
PAD_META       = os.path.join(PAD_DIR, "metadata.csv")

# ── GPU ───────────────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✓ GPU ready")

# ── Load foundation model ─────────────────────────────────────────────────────
print(f"\n── Loading foundation model ──")
model = tf.keras.models.load_model(FOUNDATION_MODEL)
print(f"✓ Loaded. Layers: {len(model.layers)}")

# ── Load PAD-UFES-20 ──────────────────────────────────────────────────────────
print(f"\n── Loading PAD-UFES-20 ──")
df = pd.read_csv(PAD_META)

# Build image paths using img_id column
df["image_path"] = df["img_id"].apply(
    lambda x: os.path.join(PAD_IMAGES_DIR, str(x)))
df = df[df["image_path"].apply(os.path.exists)].copy()
print(f"Images found: {len(df)}")

# Label mapping — explicit, no guessing
df["label"] = df["diagnostic"].str.upper().apply(
    lambda x: 1 if x in {"BCC", "SCC", "MEL"} else 0)

print(f"\nClass distribution:")
print(df["diagnostic"].value_counts().to_string())
print(f"\nMalignant (BCC+SCC+MEL) : {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
print(f"Benign    (NEV+ACK+SEK) : {(df['label']==0).sum()} ({(1-df['label'].mean())*100:.1f}%)")

if len(df) == 0:
    raise FileNotFoundError(f"No images found in {PAD_IMAGES_DIR}")

# ── Split ─────────────────────────────────────────────────────────────────────
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42)
print(f"\nTrain: {len(train_df)}  |  Val: {len(val_df)}")

# ── Sample weights ────────────────────────────────────────────────────────────
n_neg        = (train_df["label"] == 0).sum()
n_pos        = (train_df["label"] == 1).sum()
weight_for_0 = 1.0
weight_for_1 = float(np.sqrt(float(n_neg) / float(n_pos)))
print(f"Sample weight: malignant = {weight_for_1:.3f}x benign")

# ── Augmentation ──────────────────────────────────────────────────────────────
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.RandomRotate90(),
    A.HorizontalFlip(),
    A.Transpose(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.6),
    A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=35, p=0.5),
    A.GaussNoise(p=0.3),
    A.Blur(blur_limit=3, p=0.2),
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
run_ctx = mlflow.start_run(run_name="pad_ufes_finetuning") \
    if MLFLOW_ENABLED else __import__("contextlib").nullcontext()

with run_ctx:
    if MLFLOW_ENABLED:
        mlflow.log_params({
            "model":            "EfficientNetB0",
            "dataset":          "PAD-UFES-20",
            "foundation":       FOUNDATION_NAME,
            "n_train":          len(train_df),
            "n_val":            len(val_df),
            "n_malignant":      int(df["label"].sum()),
            "n_benign":         int((df["label"]==0).sum()),
            "warmup_lr":        WARMUP_LR,
            "finetune_lr":      FINETUNE_LR,
            "unfreeze_layers":  UNFREEZE_LAYERS,
            "warmup_epochs":    WARMUP_EPOCHS,
            "finetune_epochs":  FINETUNE_EPOCHS,
            "label_mapping":    "BCC+SCC+MEL=1, NEV+ACK+SEK=0",
            "weight_pos":       round(weight_for_1, 3),
            "domain_shift":     "dermoscopy->smartphone",
        })

    # ── PHASE 1: Warmup ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 1: Warmup — {WARMUP_EPOCHS} epochs, base FROZEN")
    print(f"{'='*60}")

    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=WARMUP_LR),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        run_eagerly=True,
    )

    h1 = model.fit(
        data_generator(train_df, train_transform, BATCH_SIZE, weighted=True),
        steps_per_epoch=steps_train,
        validation_data=data_generator(val_df, val_transform, BATCH_SIZE, shuffle=False),
        validation_steps=steps_val,
        epochs=WARMUP_EPOCHS,
        callbacks=[
            ModelCheckpoint(WARMUP_CKPT, monitor="val_auc",
                            save_best_only=True, mode="max", verbose=1),
            EarlyStopping(monitor="val_auc", patience=5,
                          restore_best_weights=True, mode="max", verbose=1),
        ],
    )
    best_p1 = max(h1.history["val_auc"])
    print(f"\n✓ Phase 1 best val_auc = {best_p1:.4f}")

    if MLFLOW_ENABLED:
        mlflow.log_metric("phase1_best_auc", best_p1)

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

    h2 = model.fit(
        data_generator(train_df, train_transform, BATCH_SIZE, weighted=True),
        steps_per_epoch=steps_train,
        validation_data=data_generator(val_df, val_transform, BATCH_SIZE, shuffle=False),
        validation_steps=steps_val,
        epochs=FINETUNE_EPOCHS,
        callbacks=[
            ModelCheckpoint(OUTPUT_MODEL, monitor="val_auc",
                            save_best_only=True, mode="max", verbose=1),
            EarlyStopping(monitor="val_auc", patience=8,
                          restore_best_weights=True, mode="max", verbose=1),
            ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=4,
                              min_lr=1e-9, mode="max", verbose=1),
        ],
    )
    best_p2 = max(h2.history["val_auc"])

    # ── Save history ──────────────────────────────────────────────────────────
    combined = {}
    for k in h1.history:
        combined[k] = h1.history[k] + h2.history.get(k, [])
    with open(HISTORY_PATH, "wb") as f:
        pickle.dump(combined, f)
    print(f"✓ History saved")

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\n── Evaluating on PAD-UFES val ──")
    val_gen = data_generator(val_df, val_transform, BATCH_SIZE, shuffle=False)
    y_true, y_proba = [], []
    for _ in range(steps_val):
        imgs, lbls = next(val_gen)
        preds = model.predict(imgs, verbose=0)
        y_true.extend(lbls)
        y_proba.extend(preds.flatten())

    y_true    = np.array(y_true)
    y_proba   = np.array(y_proba)
    final_auc = roc_auc_score(y_true, y_proba)

    print(f"\n── Per-class breakdown ──")
    print(df["diagnostic"].value_counts().to_string())

    print(f"\n── Classification report (threshold=0.4) ──")
    print(classification_report(
        y_true, (y_proba > 0.4).astype(int),
        target_names=["Benign", "Malignant"], zero_division=0))

    if MLFLOW_ENABLED:
        mlflow.log_metric("phase2_best_auc", best_p2)
        mlflow.log_metric("final_val_auc",   final_auc)
        for i, auc in enumerate(h2.history["val_auc"]):
            mlflow.log_metric("finetune_val_auc", auc, step=i)
        mlflow.log_artifact(OUTPUT_MODEL, artifact_path="models")
        mlflow.set_tag("stage",  "pad_ufes_finetuning")
        mlflow.set_tag("status", "completed")
        print("✓ Logged to MLflow")

    print(f"\n{'='*60}")
    print("SUMMARY — PAD-UFES Fine-tuning")
    print(f"{'='*60}")
    print(f"Foundation           : {FOUNDATION_NAME}")
    print(f"Phase 1 best val_auc : {best_p1:.4f}")
    print(f"Phase 2 best val_auc : {best_p2:.4f}")
    print(f"Final ROC-AUC        : {final_auc:.4f}")
    print(f"Model saved to       : {OUTPUT_MODEL}")
    print(f"\n── Pipeline complete! ──")
    print(f"  HAM10000 baseline  : 0.7691")
    print(f"  After DDI          : 0.7483 (HAM val)")
    print(f"  After PAD-UFES     : {final_auc:.4f}")
    print(f"\nNext:")
    print(f"  python3 backend/models/fairness_audit.py --model pad_ufes_finetuned.keras")
    print(f"  python3 backend/models/evaluate.py --model_path backend/saved_models/pad_ufes_finetuned.keras")
