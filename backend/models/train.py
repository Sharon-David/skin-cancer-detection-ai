import os
import sys

if os.name == 'nt':
    base_path = sys.prefix
    nvidia_libs = os.path.join(base_path, 'Lib', 'site-packages', 'nvidia')
    if os.path.exists(nvidia_libs):
        for root, dirs, files in os.walk(nvidia_libs):
            if 'bin' in dirs:
                os.add_dll_directory(os.path.join(root, 'bin'))

import argparse
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from keras import layers, models
from keras.applications import EfficientNetB3
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import albumentations as A

# GPU verification (enhanced with print)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("NO GPU detected, training on CPU (slow)")

# Command line args (added --model_path)
parser = argparse.ArgumentParser(description="Train baseline on HAM10000")
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--use_metadata', action='store_true', help='Include metadata (age/sex/location) - not implemented yet')
parser.add_argument('--model_path', type=str, default='saved_models/ham10000_baseline.keras', help='Path to save the model')
args = parser.parse_args()

# Paths (with debug prints)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          
BACKEND_DIR = os.path.dirname(BASE_DIR)                 
PROJECT_DIR = os.path.dirname(BACKEND_DIR)                  

DATA_DIR = os.path.join(BACKEND_DIR, "data", "ham10000")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
METADATA_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")

MODEL_DIR = os.path.dirname(args.model_path)
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Images dir: {IMAGES_DIR} (exists: {os.path.exists(IMAGES_DIR)})")
print(f"Metadata path: {METADATA_PATH} (exists: {os.path.exists(METADATA_PATH)})")
print(f"Saving model to: {args.model_path}")

# Load and prepare data
df = pd.read_csv(METADATA_PATH)
df['image_path'] = df['image_id'].apply(lambda x: os.path.join(IMAGES_DIR, f"{x}.jpg"))

# EARLY FAILURE CHECK (enhanced)
if not os.path.exists(df['image_path'].iloc[0]):
    raise FileNotFoundError(f"Image not found: {df['image_path'].iloc[0]}")

malignant_classes = ['mel', 'bcc', 'akiec']  # Fixed typo if any
df['label'] = df['dx'].apply(lambda x: 1 if x in malignant_classes else 0)

print("Class distribution:\n", df['label'].value_counts(normalize=True))

# Splitting the dataset
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Data augmentation
train_transform = A.Compose([
    A.Resize(300, 300),
    A.RandomRotate90(),
    A.Flip(),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.GaussNoise(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

val_transform = A.Compose([
    A.Resize(300, 300),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))   
])

# Data generators (enhanced try-except with logging)
def data_generator(df, transform, batch_size, shuffle=True):
    while True:
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)  # Shuffle here
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            images, labels = [], []
            skipped = 0

            for _, row in batch.iterrows():
                try:
                    img = Image.open(row['image_path']).convert('RGB')
                    img = np.array(img)
                    img = transform(image=img)['image']
                    images.append(img)
                    labels.append(row['label'])
                except Exception as e:
                    print(f"Skipping {row['image_path']}: {e}")
                    skipped += 1

            if images:
                yield np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)
            else:
                print(f"Batch skipped entirely ({skipped} errors)")

train_gen = data_generator(train_df, train_transform, args.batch_size, shuffle=True)
val_gen = data_generator(val_df, val_transform, args.batch_size, shuffle=False)  # No shuffle for val

steps_train = max(1, len(train_df) // args.batch_size)  
steps_val = max(1, len(val_df) // args.batch_size)

# Build model
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(300,300,3))
base_model.trainable = True  # Unfreeze all layers

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)

predictions = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=args.lr),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# Train
checkpoint = ModelCheckpoint(args.model_path, monitor='val_auc', save_best_only=True, mode='max', verbose=1)
early_stop = EarlyStopping(monitor='val_auc', patience=10, restore_best_weights=True, mode='max')

history = model.fit(
    train_gen,
    steps_per_epoch=steps_train,
    validation_data=val_gen,
    validation_steps=steps_val,
    epochs=args.epochs,
    callbacks=[checkpoint, early_stop]
)

# Save history for later plotting (optional but useful)
import pickle
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Evaluation (moved to separate notebook later, but basic print here)
print("\nEvaluating model...")
val_gen_eval = data_generator(val_df, val_transform, args.batch_size, shuffle=False)

y_true, y_pred = [], []

for _ in range(steps_val):
    images, labels = next(val_gen_eval)
    preds = model.predict(images, verbose=0)
    y_true.extend(labels)
    y_pred.extend((preds > 0.5).astype(int).flatten())

print("\nFinal Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Benign', 'Malignant']))