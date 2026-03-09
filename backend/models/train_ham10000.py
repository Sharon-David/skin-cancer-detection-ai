"""
train_ham10000.py — Step 1b of sequential fine-tuning pipeline
================================================================
Goal    : Train the strongest possible model on HAM10000 alone.
          This becomes the foundation for DDI and PAD-UFES fine-tuning.

Usage:
    python backend/models/train_ham10000.py

Next step:
    python backend/models/finetune_ddi.py
"""

import os
import sys

# ── Must be set BEFORE any TF/Keras import ────────────────────────────────────
os.environ["TF_XLA_FLAGS"]          = "--tf_xla_auto_jit=0"
os.environ["TF_ENABLE_XLA"]         = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["XLA_FLAGS"]             = "--xla_gpu_autotune_level=0"

if os.name == 'nt':
    base_path = sys.prefix
    nvidia_libs = os.path.join(base_path, 'Lib', 'site-packages', 'nvidia')
    if os.path.exists(nvidia_libs):
        for root, dirs, files in os.walk(nvidia_libs):
            if 'bin' in dirs:
                os.add_dll_directory(os.path.join(root, 'bin'))

import pickle
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
tf.config.optimizer.set_jit(False)

from keras import layers, models
from keras.applications import EfficientNetB0
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import albumentations as A

# ── Config ─────────────────────────────────────────────────────────────────────
IMG_SIZE        = 224
BATCH_SIZE      = 8
WARMUP_EPOCHS   = 20
FINETUNE_EPOCHS = 50
WARMUP_LR       = 1e-3
FINETUNE_LR     = 1e-5    # Back to what worked in original train.py (0.769 AUC)
UNFREEZE_LAYERS = 20      # Middle ground

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR   = os.path.dirname(BASE_DIR)
DATA_DIR      = os.path.join(BACKEND_DIR, "data", "ham10000")
IMAGES_DIR    = os.path.join(DATA_DIR, "images")
METADATA_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
SAVED_DIR     = os.path.join(BACKEND_DIR, "saved_models")
os.makedirs(SAVED_DIR, exist_ok=True)

MODEL_PATH       = os.path.join(SAVED_DIR, "ham10000_final.keras")
WARMUP_CKPT_PATH = os.path.join(SAVED_DIR, "ham10000_final_warmup.keras")
HISTORY_PATH     = os.path.join(SAVED_DIR, "ham10000_final_history.pkl")

# ── GPU ────────────────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU: {[g.name for g in gpus]}")
    except RuntimeError as e:
        print(f"GPU error: {e}")
else:
    print("⚠ No GPU detected")

print(f"\n── Config ──")
print(f"Image size      : {IMG_SIZE}x{IMG_SIZE}")
print(f"Batch size      : {BATCH_SIZE}")
print(f"Warmup epochs   : {WARMUP_EPOCHS} @ LR={WARMUP_LR}")
print(f"Finetune epochs : {FINETUNE_EPOCHS} @ LR={FINETUNE_LR}")
print(f"Unfreeze layers : {UNFREEZE_LAYERS}")
print(f"Loss            : binary_crossentropy + sqrt sample weights")
print(f"Output model    : {MODEL_PATH}")

# ── Load HAM10000 ──────────────────────────────────────────────────────────────
print(f"\n── Loading HAM10000 ──")
df = pd.read_csv(METADATA_PATH)
df['image_path'] = df['image_id'].apply(
    lambda x: os.path.join(IMAGES_DIR, f"{x}.jpg"))
df = df[df['image_path'].apply(os.path.exists)].copy()

malignant_classes = ['mel', 'bcc', 'akiec']
df['label'] = df['dx'].apply(lambda x: 1 if x in malignant_classes else 0)

print(f"Total images : {len(df)}")
print(f"Malignant    : {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
print(f"Benign       : {(df['label']==0).sum()} ({(1-df['label'].mean())*100:.1f}%)")

# ── Train / val split ──────────────────────────────────────────────────────────
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42)
print(f"Train: {len(train_df)}  |  Val: {len(val_df)}")

# ── Sample weights (sqrt dampening) ───────────────────────────────────────────
n_neg        = (train_df['label'] == 0).sum()
n_pos        = (train_df['label'] == 1).sum()
weight_for_0 = 1.0
weight_for_1 = float(np.sqrt(float(n_neg) / float(n_pos)))
print(f"Sample weight: malignant = {weight_for_1:.2f}x benign")

# ── Augmentation ───────────────────────────────────────────────────────────────
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.RandomRotate90(),
    A.Flip(),
    A.Transpose(p=0.3),
    A.RandomBrightnessContrast(p=0.4),
    A.HueSaturationValue(p=0.4),
    A.GaussNoise(p=0.2),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# ── Generator ──────────────────────────────────────────────────────────────────
def data_generator(dataframe, transform, batch_size, shuffle=True, weighted=False):
    while True:
        df_loop = dataframe.sample(frac=1).reset_index(drop=True) if shuffle else dataframe
        for i in range(0, len(df_loop), batch_size):
            batch = df_loop.iloc[i:i + batch_size]
            images, labels = [], []
            for _, row in batch.iterrows():
                try:
                    img = np.array(Image.open(row['image_path']).convert('RGB'))
                    img = transform(image=img)['image']
                    images.append(img)
                    labels.append(row['label'])
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
print(f"Steps/epoch  : train={steps_train}, val={steps_val}")

# ── Build model ────────────────────────────────────────────────────────────────
print(f"\nBuilding EfficientNetB0...")
base_model = EfficientNetB0(
    weights='imagenet', include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

x      = layers.GlobalAveragePooling2D()(base_model.output)
x      = layers.BatchNormalization()(x)
x      = layers.Dropout(0.4)(x)
x      = layers.Dense(256, activation='relu',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x      = layers.BatchNormalization()(x)
x      = layers.Dropout(0.3)(x)
x      = layers.Dense(64, activation='relu')(x)
x      = layers.Dropout(0.2)(x)
output = layers.Dense(1, activation='sigmoid')(x)
model  = models.Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=WARMUP_LR),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    run_eagerly=True,
)

trainable = sum(np.prod(v.shape) for v in model.trainable_variables)
total     = sum(np.prod(v.shape) for v in model.variables)
print(f"Params: trainable={trainable:,}  total={total:,}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Head warmup (base fully frozen)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"PHASE 1: Head warmup — {WARMUP_EPOCHS} epochs, base FROZEN")
print(f"  LR = {WARMUP_LR}  |  Loss = binary_crossentropy")
print(f"{'='*60}")

cb_warmup = [
    ModelCheckpoint(WARMUP_CKPT_PATH, monitor='val_auc',
                    save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_auc', patience=6,
                  restore_best_weights=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3,
                      min_lr=1e-6, mode='max', verbose=1),
]

h1 = model.fit(
    data_generator(train_df, train_transform, BATCH_SIZE, shuffle=True,  weighted=True),
    steps_per_epoch=steps_train,
    validation_data=data_generator(val_df, val_transform, BATCH_SIZE, shuffle=False, weighted=False),
    validation_steps=steps_val,
    epochs=WARMUP_EPOCHS,
    callbacks=cb_warmup,
)

best_p1 = max(h1.history['val_auc'])
print(f"\n✓ Phase 1 complete. Best val_auc = {best_p1:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Fine-tune top UNFREEZE_LAYERS layers
# CRITICAL FIX: explicitly load warmup checkpoint before unfreezing.
# Without this, compile() resets optimizer state and Phase 2 starts
# from wrong weights — proven cause of Phase 1 → Phase 2 AUC drop.
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"PHASE 2: Fine-tuning top {UNFREEZE_LAYERS} base layers")
print(f"  LR = {FINETUNE_LR}  |  Loading best warmup weights first")
print(f"{'='*60}")

# Load best warmup weights explicitly
model.load_weights(WARMUP_CKPT_PATH)
print(f"✓ Warmup checkpoint loaded — Phase 2 starts from Phase 1 best")

base_model.trainable = True
for layer in base_model.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False

unfrozen = sum(1 for l in base_model.layers if l.trainable)
print(f"Unfrozen: {unfrozen} / {len(base_model.layers)} base layers")

model.compile(
    optimizer=Adam(learning_rate=FINETUNE_LR),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    run_eagerly=True,
)

cb_finetune = [
    ModelCheckpoint(MODEL_PATH, monitor='val_auc',
                    save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_auc', patience=10,
                  restore_best_weights=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5,
                      min_lr=1e-8, mode='max', verbose=1),
]

h2 = model.fit(
    data_generator(train_df, train_transform, BATCH_SIZE, shuffle=True,  weighted=True),
    steps_per_epoch=steps_train,
    validation_data=data_generator(val_df, val_transform, BATCH_SIZE, shuffle=False, weighted=False),
    validation_steps=steps_val,
    epochs=FINETUNE_EPOCHS,
    callbacks=cb_finetune,
)

# ── Save history ───────────────────────────────────────────────────────────────
combined = {}
for k in h1.history:
    combined[k] = h1.history[k] + h2.history.get(k, [])
with open(HISTORY_PATH, 'wb') as f:
    pickle.dump(combined, f)
print(f"\n✓ History saved: {HISTORY_PATH}")

# ── Evaluation ─────────────────────────────────────────────────────────────────
print("\nEvaluating on HAM10000 validation set...")
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

print("\n── Classification report at multiple thresholds ──")
for thresh in [0.3, 0.4, 0.5]:
    y_pred = (y_proba > thresh).astype(int)
    print(f"\nThreshold = {thresh}")
    print(classification_report(y_true, y_pred, target_names=['Benign', 'Malignant']))

print(f"\n{'='*60}")
print("SUMMARY — HAM10000 Foundation Model")
print(f"{'='*60}")
print(f"Phase 1 best val_auc  : {best_p1:.4f}")
print(f"Phase 2 best val_auc  : {max(h2.history['val_auc']):.4f}")
print(f"Final ROC-AUC         : {auc:.4f}")
print(f"Previous best         : 0.7691  (ham10000_baseline.keras)")
print(f"Model saved to        : {MODEL_PATH}")
print()
if auc >= best_p1:
    print(f"✓ Fine-tuning helped: {best_p1:.4f} → {auc:.4f}")
    print(f"  Next: python backend/models/finetune_ddi.py")
else:
    print(f"✗ Fine-tuning did not improve over warmup")
    print(f"  Use ham10000_baseline.keras (0.7691) as foundation instead")
    print(f"  Next: python backend/models/finetune_ddi.py")
