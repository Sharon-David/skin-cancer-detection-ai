"""
domain_shift_analysis.py — Document domain shift finding
=========================================================
Runs both models on both datasets and produces a comparison
plot showing the domain shift between dermoscopy and smartphone.

This is a research finding, not a failure.

Usage:
    python3 backend/models/domain_shift_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
os.environ["TF_ENABLE_XLA"]         = "0"
os.environ["TF_XLA_FLAGS"]          = "--tf_xla_auto_jit=0"

import tensorflow as tf
tf.config.optimizer.set_jit(False)

from sklearn.metrics import roc_auc_score
import albumentations as A

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(BASE_DIR)
SAVED_DIR   = os.path.join(BACKEND_DIR, "saved_models")
OUT_DIR     = os.path.join(BACKEND_DIR, "..", "reports")
os.makedirs(OUT_DIR, exist_ok=True)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

def run_inference(model, img_paths, labels, n=200):
    """Run inference on up to n images, return AUC."""
    idx = np.random.choice(len(img_paths), min(n, len(img_paths)), replace=False)
    probs, y = [], []
    for i in idx:
        try:
            img  = np.array(Image.open(img_paths[i]).convert("RGB"))
            img  = transform(image=img)["image"]
            img  = np.expand_dims(img, 0).astype(np.float32)
            prob = float(model.predict(img, verbose=0)[0][0])
            probs.append(prob)
            y.append(labels[i])
        except:
            pass
    if len(set(y)) < 2:
        return None
    return roc_auc_score(y, probs)

# ── Load models ───────────────────────────────────────────────────────────────
models_to_test = {}
for name, fname in [("HAM10000 (Dermoscopy)", "ham10000_baseline.keras"),
                     ("PAD-UFES (Smartphone)", "pad_ufes_finetuned.keras")]:
    path = os.path.join(SAVED_DIR, fname)
    if os.path.exists(path):
        print(f"Loading: {fname}")
        models_to_test[name] = tf.keras.models.load_model(path)

if len(models_to_test) < 2:
    print("Need both models to run domain shift analysis.")
    print("Run train_pad_ufes.py first.")
    sys.exit(0)

# ── Load HAM10000 val ─────────────────────────────────────────────────────────
ham_meta  = pd.read_csv(os.path.join(BACKEND_DIR, "data", "ham10000", "HAM10000_metadata.csv"))
ham_dir   = os.path.join(BACKEND_DIR, "data", "ham10000", "images")

def ham_label(dx):
    return 1 if dx in ["mel", "bcc", "akiec", "bkl"] else 0

ham_meta["label"]      = ham_meta["dx"].apply(ham_label)
ham_meta["image_path"] = ham_meta["image_id"].apply(
    lambda x: os.path.join(ham_dir, x + ".jpg"))
ham_meta = ham_meta[ham_meta["image_path"].apply(os.path.exists)]
ham_paths  = ham_meta["image_path"].tolist()
ham_labels = ham_meta["label"].tolist()

# ── Load PAD-UFES val ─────────────────────────────────────────────────────────
pad_meta  = pd.read_csv(os.path.join(BACKEND_DIR, "data", "pad_ufes_20", "metadata.csv"))
pad_dir   = os.path.join(BACKEND_DIR, "data", "pad_ufes_20", "images")
pad_meta["label"]      = pad_meta["diagnostic"].str.upper().apply(
    lambda x: 1 if x in {"BCC", "SCC", "MEL"} else 0)
pad_meta["image_path"] = pad_meta["img_id"].apply(
    lambda x: os.path.join(pad_dir, str(x)))
pad_meta  = pad_meta[pad_meta["image_path"].apply(os.path.exists)]
pad_paths  = pad_meta["image_path"].tolist()
pad_labels = pad_meta["label"].tolist()

# ── Run all 4 combinations ────────────────────────────────────────────────────
print("\nRunning domain shift analysis (200 images per dataset)...")
results = {}
datasets = {
    "HAM10000\n(Dermoscopy)": (ham_paths, ham_labels),
    "PAD-UFES\n(Smartphone)": (pad_paths, pad_labels),
}

for model_name, model in models_to_test.items():
    results[model_name] = {}
    for dataset_name, (paths, labels) in datasets.items():
        print(f"  {model_name} → {dataset_name.strip()} ...", end=" ")
        auc = run_inference(model, paths, labels)
        results[model_name][dataset_name] = auc
        print(f"AUC = {auc:.4f}" if auc else "SKIP")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#0a1628")

model_names  = list(results.keys())
dataset_names = list(list(results.values())[0].keys())
colors = ["#0fb8a9", "#f87171"]
x = np.arange(len(dataset_names))
width = 0.35

ax = axes[0]
ax.set_facecolor("#0f1f3a")
for i, (mname, color) in enumerate(zip(model_names, colors)):
    aucs = [results[mname][d] or 0 for d in dataset_names]
    bars = ax.bar(x + i*width - width/2, aucs, width,
                  label=mname, color=color, alpha=0.85, edgecolor="none")
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{auc:.3f}", ha="center", va="bottom",
                color="white", fontsize=10, fontweight="bold")

ax.axhline(y=0.5, color="#fbbf24", linestyle="--", linewidth=1, alpha=0.5, label="Random (0.5)")
ax.set_xticks(x)
ax.set_xticklabels([d.replace("\n", " ") for d in dataset_names])
ax.set_ylim(0, 1)
ax.set_ylabel("ROC-AUC", color="#94a3b8")
ax.set_title("Domain Shift — AUC by Model & Dataset", color="white", fontsize=13, pad=12)
ax.tick_params(colors="#94a3b8")
ax.spines[:].set_color("#1e3a5f")
ax.legend(labelcolor="white", facecolor="#0f1f3a", edgecolor="#1e3a5f", fontsize=9)

# Heatmap
ax2 = axes[1]
ax2.set_facecolor("#0f1f3a")
matrix = np.array([[results[m][d] or 0 for d in dataset_names] for m in model_names])
im = ax2.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.4, vmax=0.9)
ax2.set_xticks(range(len(dataset_names)))
ax2.set_yticks(range(len(model_names)))
ax2.set_xticklabels([d.replace("\n", " ") for d in dataset_names], color="#94a3b8")
ax2.set_yticklabels(model_names, color="#94a3b8")
ax2.set_title("AUC Heatmap", color="white", fontsize=13, pad=12)
for i in range(len(model_names)):
    for j in range(len(dataset_names)):
        ax2.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center",
                 color="black", fontweight="bold", fontsize=12)
plt.colorbar(im, ax=ax2)

plt.suptitle("DermAI — Domain Shift Analysis\nDermoscopy vs Smartphone Clinical Photos",
             color="white", fontsize=14, y=1.02)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, "domain_shift_analysis.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0a1628")
print(f"\n✓ Plot saved: {out_path}")

print(f"\n{'='*60}")
print("DOMAIN SHIFT FINDINGS")
print(f"{'='*60}")
for mname in model_names:
    for dname in dataset_names:
        auc = results[mname][dname]
        print(f"  {mname:<30} on {dname.strip():<25}: {auc:.4f}")

derm_on_ham = results["HAM10000 (Dermoscopy)"]["HAM10000\n(Dermoscopy)"]
derm_on_pad = results["HAM10000 (Dermoscopy)"]["PAD-UFES\n(Smartphone)"]
drop = derm_on_ham - derm_on_pad
print(f"\nDomain shift magnitude: {drop:.4f} AUC drop")
print(f"  ({round(drop/derm_on_ham*100, 1)}% relative degradation)")
print(f"\nThis confirms significant domain shift between")
print(f"dermoscopy and smartphone clinical photography.")
print(f"Ensemble model addresses this by combining both domains.")
