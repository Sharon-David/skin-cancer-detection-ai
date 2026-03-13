"""
train_pad_ufes.py — PAD-UFES-20 training with tf.data pipeline
===============================================================
Uses tf.data instead of Python generators — stable on Blackwell GPU.
Fresh EfficientNetB0 from ImageNet weights.

Usage:
    python3 backend/models/train_pad_ufes.py
"""

import os
os.environ["TF_XLA_FLAGS"]          = "--tf_xla_auto_jit=0"
os.environ["TF_ENABLE_XLA"]         = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pickle
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

import tensorflow as tf
tf.config.optimizer.set_jit(False)

from keras import layers, models, regularizers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

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
BATCH_SIZE      = 16   # larger batch = more stable gradients
WARMUP_EPOCHS   = 20
FINETUNE_EPOCHS = 40
WARMUP_LR       = 1e-3
FINETUNE_LR     = 1e-5
UNFREEZE_LAYERS = 30
AUTOTUNE        = tf.data.AUTOTUNE

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR  = os.path.dirname(BASE_DIR)
SAVED_DIR    = os.path.join(BACKEND_DIR, "saved_models")
PAD_DIR      = os.path.join(BACKEND_DIR, "data", "pad_ufes_20")
PAD_IMAGES   = os.path.join(PAD_DIR, "images")
PAD_META     = os.path.join(PAD_DIR, "metadata.csv")
WARMUP_CKPT  = os.path.join(SAVED_DIR, "pad_ufes_warmup.keras")
OUTPUT_MODEL = os.path.join(SAVED_DIR, "pad_ufes_finetuned.keras")
HISTORY_PATH = os.path.join(SAVED_DIR, "pad_ufes_history.pkl")

# ── GPU ───────────────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✓ GPU: {len(gpus)} device(s)")

# ── Load metadata ─────────────────────────────────────────────────────────────
print("\n── Loading PAD-UFES-20 ──")
df = pd.read_csv(PAD_META)
df["image_path"] = df["img_id"].apply(
    lambda x: os.path.join(PAD_IMAGES, str(x)))
df = df[df["image_path"].apply(os.path.exists)].copy()
df["label"] = df["diagnostic"].str.upper().apply(
    lambda x: 1 if x in {"BCC", "SCC", "MEL"} else 0)

print(f"Images: {len(df)}")
print(df["diagnostic"].value_counts().to_string())
print(f"\nMalignant : {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
print(f"Benign    : {(df['label']==0).sum()} ({(1-df['label'].mean())*100:.1f}%)")

# ── Split ─────────────────────────────────────────────────────────────────────
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42)
print(f"\nTrain: {len(train_df)}  |  Val: {len(val_df)}")

# ── Sample weights ────────────────────────────────────────────────────────────
n_neg        = (train_df["label"] == 0).sum()
n_pos        = (train_df["label"] == 1).sum()
weight_for_1 = float(np.sqrt(float(n_neg) / float(n_pos)))
weight_for_0 = 1.0
print(f"Sample weight malignant: {weight_for_1:.3f}x")

# ── Pre-process images to numpy arrays once ───────────────────────────────────
# Load everything into memory — 2298 images at 224x224 ~ 1.4GB, fits in RAM
print("\n── Pre-loading images into RAM (faster training) ──")

def load_images(dataframe):
    images, labels = [], []
    for _, row in dataframe.iterrows():
        try:
            img = Image.open(row["image_path"]).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            images.append(np.array(img, dtype=np.uint8))
            labels.append(int(row["label"]))
        except Exception as e:
            print(f"  Skip: {row['image_path']} — {e}")
    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.float32)

print("  Loading train images...")
X_train, y_train = load_images(train_df)
print(f"  ✓ Train: {X_train.shape}")

print("  Loading val images...")
X_val, y_val = load_images(val_df)
print(f"  ✓ Val:   {X_val.shape}")

# ── Sample weights array ──────────────────────────────────────────────────────
sw_train = np.where(y_train == 1, weight_for_1, weight_for_0).astype(np.float32)

# ── tf.data pipeline ──────────────────────────────────────────────────────────
# ImageNet normalization
MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

def normalize(image):
    image = tf.cast(image, tf.float32) / 255.0
    return (image - MEAN) / STD

@tf.function
def augment(image, label, sw):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    image = tf.image.random_hue(image, max_delta=0.1)
    image = tf.clip_by_value(image, 0, 255)
    return normalize(image), label, sw

@tf.function
def preprocess_val(image, label):
    return normalize(image), label

def make_train_dataset(X, y, sw, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((X, y, sw))
    ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds

def make_val_dataset(X, y, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.map(preprocess_val, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds

train_ds = make_train_dataset(X_train, y_train, sw_train, BATCH_SIZE)
val_ds   = make_val_dataset(X_val, y_val, BATCH_SIZE)
print(f"\nSteps/epoch: train={len(X_train)//BATCH_SIZE}, val={len(X_val)//BATCH_SIZE}")

# ── Build model ───────────────────────────────────────────────────────────────
print("\n── Building EfficientNetB0 (ImageNet weights) ──")
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation="relu",
                  kernel_regularizer=regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = models.Model(inputs, outputs)
print(f"✓ Model ready. Base layers: {len(base.layers)}")

# ── Wrapper to handle (image, label, sample_weight) from tf.data ─────────────
# Keras fit() with sample_weights needs a special dataset format
def unpack_ds(ds):
    """Convert (img, label, sw) dataset to format keras accepts."""
    return ds.map(lambda img, lbl, sw: (img, lbl, sw))

# ══════════════════════════════════════════════════════════════════════════════
run_ctx = mlflow.start_run(run_name="pad_ufes_tfdata") \
    if MLFLOW_ENABLED else __import__("contextlib").nullcontext()

with run_ctx:
    if MLFLOW_ENABLED:
        mlflow.log_params({
            "model":         "EfficientNetB0",
            "weights":       "imagenet",
            "dataset":       "PAD-UFES-20",
            "pipeline":      "tf.data",
            "batch_size":    BATCH_SIZE,
            "n_train":       len(X_train),
            "n_val":         len(X_val),
            "warmup_lr":     WARMUP_LR,
            "finetune_lr":   FINETUNE_LR,
            "label_mapping": "BCC+SCC+MEL=1, NEV+ACK+SEK=0",
        })

    # ── PHASE 1: Warmup ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 1: Warmup {WARMUP_EPOCHS} epochs | LR={WARMUP_LR} | Base FROZEN")
    print(f"{'='*60}")

    model.compile(
        optimizer=Adam(learning_rate=WARMUP_LR),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        run_eagerly=True,
    )

    h1 = model.fit(
        unpack_ds(train_ds),
        validation_data=val_ds,
        epochs=WARMUP_EPOCHS,
        callbacks=[
            ModelCheckpoint(WARMUP_CKPT, monitor="val_auc",
                            save_best_only=True, mode="max", verbose=1),
            EarlyStopping(monitor="val_auc", patience=6,
                          restore_best_weights=True, mode="max", verbose=1),
        ],
    )
    best_p1 = max(h1.history["val_auc"])
    print(f"\n✓ Phase 1 best val_auc = {best_p1:.4f}")

    if MLFLOW_ENABLED:
        mlflow.log_metric("phase1_best_auc", best_p1)
        for i, v in enumerate(h1.history["val_auc"]):
            mlflow.log_metric("warmup_val_auc", v, step=i)

    # ── PHASE 2: Fine-tune ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PHASE 2: Unfreeze top {UNFREEZE_LAYERS} | LR={FINETUNE_LR}")
    print(f"{'='*60}")

    model.load_weights(WARMUP_CKPT)
    base.trainable = True
    for layer in base.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False
    print(f"Unfrozen: {sum(1 for l in base.layers if l.trainable)} base layers")

    model.compile(
        optimizer=Adam(learning_rate=FINETUNE_LR),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        run_eagerly=True,
    )

    h2 = model.fit(
        unpack_ds(train_ds),
        validation_data=val_ds,
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
    print("✓ History saved")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n── Final evaluation on val set ──")
    y_proba = model.predict(
        make_val_dataset(X_val, y_val, BATCH_SIZE), verbose=1
    ).flatten()
    final_auc = roc_auc_score(y_val, y_proba)

    print(classification_report(
        y_val, (y_proba > 0.4).astype(int),
        target_names=["Benign", "Malignant"], zero_division=0))

    if MLFLOW_ENABLED:
        mlflow.log_metric("phase2_best_auc", best_p2)
        mlflow.log_metric("final_val_auc",   final_auc)
        for i, v in enumerate(h2.history["val_auc"]):
            mlflow.log_metric("finetune_val_auc", v, step=i)
        mlflow.log_artifact(OUTPUT_MODEL, artifact_path="models")
        mlflow.set_tags({"stage": "pad_ufes", "pipeline": "tf.data", "status": "completed"})
        print("✓ MLflow logged")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Phase 1 best val_auc : {best_p1:.4f}")
    print(f"Phase 2 best val_auc : {best_p2:.4f}")
    print(f"Final ROC-AUC        : {final_auc:.4f}")
    print(f"Model saved          : {OUTPUT_MODEL}")
    print(f"\nNext:")
    print(f"  python3 backend/models/domain_shift_analysis.py")
    print(f"  python3 backend/models/fairness_audit.py --model pad_ufes_finetuned.keras")
