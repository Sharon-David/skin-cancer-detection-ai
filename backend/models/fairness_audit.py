"""
fairness_audit.py — Per skin tone AUC fairness analysis using DDI dataset
=========================================================================
Computes model performance broken down by Fitzpatrick skin tone.
This is critical for responsible AI in medical imaging.

Skin tone groups in DDI:
  12 = Light (Fitzpatrick I-II)
  34 = Medium (Fitzpatrick III-IV)  
  56 = Dark (Fitzpatrick V-VI)

Usage:
    python3 backend/models/fairness_audit.py
    python3 backend/models/fairness_audit.py --model ham10000_baseline.keras
    python3 backend/models/fairness_audit.py --model ddi_finetuned.keras
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
os.environ["TF_ENABLE_XLA"]         = "0"
os.environ["TF_XLA_FLAGS"]          = "--tf_xla_auto_jit=0"

import tensorflow as tf
tf.config.optimizer.set_jit(False)

from sklearn.metrics import roc_auc_score, confusion_matrix
import albumentations as A

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="ham10000_baseline.keras")
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(BASE_DIR)
SAVED_DIR   = os.path.join(BACKEND_DIR, "saved_models")
DDI_DIR     = os.path.join(BACKEND_DIR, "data", "ddi")
OUT_DIR     = os.path.join(BACKEND_DIR, "..", "reports")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(SAVED_DIR, args.model)
if not os.path.exists(MODEL_PATH):
    print(f"Model not found: {MODEL_PATH}")
    sys.exit(1)

# ── GPU ───────────────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading model: {args.model}")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✓ Loaded")

# ── Load DDI ──────────────────────────────────────────────────────────────────
print(f"\nLoading DDI dataset...")
df = pd.read_csv(os.path.join(DDI_DIR, "ddi_metadata.csv"))
df["image_path"] = df["DDI_file"].apply(
    lambda x: os.path.join(DDI_DIR, "images", str(x)))
df = df[df["image_path"].apply(os.path.exists)].copy()
df["label"] = df["malignant"].astype(str).str.lower().map(
    {"true": 1, "false": 0}).fillna(0).astype(int)

print(f"Total DDI images: {len(df)}")
print(f"Malignant: {df['label'].sum()} | Benign: {(df['label']==0).sum()}")

# ── Transform ─────────────────────────────────────────────────────────────────
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# ── Run inference on ALL DDI images ──────────────────────────────────────────
print(f"\nRunning inference on {len(df)} images...")
all_probs = []
for _, row in df.iterrows():
    try:
        img  = np.array(Image.open(row["image_path"]).convert("RGB"))
        img  = transform(image=img)["image"]
        img  = np.expand_dims(img, axis=0).astype(np.float32)
        prob = float(model.predict(img, verbose=0)[0][0])
        all_probs.append(prob)
    except Exception as e:
        all_probs.append(0.5)

df["prob_malignant"] = all_probs
print("✓ Inference complete")

# ── Overall AUC ──────────────────────────────────────────────────────────────
overall_auc = roc_auc_score(df["label"], df["prob_malignant"])
print(f"\nOverall DDI AUC: {overall_auc:.4f}")

# ── Per skin tone analysis ────────────────────────────────────────────────────
TONE_LABELS = {12: "Light (I-II)", 34: "Medium (III-IV)", 56: "Dark (V-VI)"}
results = {}

print(f"\n{'='*60}")
print("FAIRNESS AUDIT — AUC per Fitzpatrick Skin Tone")
print(f"{'='*60}")
print(f"{'Skin Tone':<20} {'N':>5} {'Mal':>5} {'AUC':>8} {'Sens@0.4':>10} {'Spec@0.4':>10}")
print("-"*60)

for tone in sorted(df["skin_tone"].unique()):
    subset = df[df["skin_tone"] == tone].copy()
    label  = TONE_LABELS.get(tone, str(tone))

    if len(subset) < 10 or subset["label"].nunique() < 2:
        print(f"{label:<20} {'N/A':>5} — insufficient data")
        continue

    auc  = roc_auc_score(subset["label"], subset["prob_malignant"])
    pred = (subset["prob_malignant"] > 0.4).astype(int)
    cm   = confusion_matrix(subset["label"], pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sens = spec = 0

    results[tone] = {
        "label":  label,
        "n":      len(subset),
        "n_mal":  int(subset["label"].sum()),
        "auc":    auc,
        "sens":   sens,
        "spec":   spec,
    }
    print(f"{label:<20} {len(subset):>5} {int(subset['label'].sum()):>5} {auc:>8.4f} {sens:>10.3f} {spec:>10.3f}")

print("-"*60)
print(f"{'Overall':<20} {len(df):>5} {int(df['label'].sum()):>5} {overall_auc:>8.4f}")

# ── Fairness gap ──────────────────────────────────────────────────────────────
if len(results) >= 2:
    aucs      = [v["auc"] for v in results.values()]
    max_gap   = max(aucs) - min(aucs)
    best_tone = max(results, key=lambda k: results[k]["auc"])
    worst_tone = min(results, key=lambda k: results[k]["auc"])

    print(f"\n── Fairness Gap ──")
    print(f"Best  AUC: {results[best_tone]['label']} = {results[best_tone]['auc']:.4f}")
    print(f"Worst AUC: {results[worst_tone]['label']} = {results[worst_tone]['auc']:.4f}")
    print(f"Gap: {max_gap:.4f}")
    if max_gap < 0.05:
        print("✓ FAIR — gap < 0.05 (excellent)")
    elif max_gap < 0.10:
        print("⚠ MODERATE BIAS — gap 0.05-0.10")
    else:
        print("✗ SIGNIFICANT BIAS — gap > 0.10 — consider debiasing")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor("#0a1628")

colors = {"Light (I-II)": "#60a5fa", "Medium (III-IV)": "#34d399", "Dark (V-VI)": "#f87171"}

# Plot 1: AUC per skin tone
ax1 = axes[0]
ax1.set_facecolor("#0f1f3a")
tone_labels = [v["label"] for v in results.values()]
tone_aucs   = [v["auc"]   for v in results.values()]
bar_colors  = [colors.get(l, "#94a3b8") for l in tone_labels]
bars = ax1.bar(tone_labels, tone_aucs, color=bar_colors, alpha=0.85, edgecolor="none")
ax1.axhline(y=overall_auc, color="#fbbf24", linestyle="--", linewidth=1.5, label=f"Overall ({overall_auc:.3f})")
ax1.set_ylim(0, 1)
ax1.set_title("AUC by Skin Tone", color="white", fontsize=13, pad=12)
ax1.set_ylabel("ROC-AUC", color="#94a3b8")
ax1.tick_params(colors="#94a3b8")
ax1.spines[:].set_color("#1e3a5f")
for bar, auc in zip(bars, tone_aucs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{auc:.3f}", ha="center", va="bottom", color="white", fontsize=10, fontweight="bold")
ax1.legend(labelcolor="white", facecolor="#0f1f3a", edgecolor="#1e3a5f")

# Plot 2: Sensitivity per skin tone
ax2 = axes[1]
ax2.set_facecolor("#0f1f3a")
tone_sens = [v["sens"] for v in results.values()]
bars2 = ax2.bar(tone_labels, tone_sens, color=bar_colors, alpha=0.85, edgecolor="none")
ax2.set_ylim(0, 1)
ax2.set_title("Sensitivity (Malignant Recall) by Skin Tone", color="white", fontsize=13, pad=12)
ax2.set_ylabel("Sensitivity", color="#94a3b8")
ax2.tick_params(colors="#94a3b8")
ax2.spines[:].set_color("#1e3a5f")
for bar, s in zip(bars2, tone_sens):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{s:.3f}", ha="center", va="bottom", color="white", fontsize=10, fontweight="bold")

# Plot 3: Sample distribution
ax3 = axes[2]
ax3.set_facecolor("#0f1f3a")
tone_n     = [v["n"]     for v in results.values()]
tone_n_mal = [v["n_mal"] for v in results.values()]
tone_n_ben = [n - m for n, m in zip(tone_n, tone_n_mal)]
x = range(len(tone_labels))
ax3.bar(x, tone_n_ben, label="Benign",    color="#34d399", alpha=0.8, edgecolor="none")
ax3.bar(x, tone_n_mal, bottom=tone_n_ben, label="Malignant", color="#f87171", alpha=0.8, edgecolor="none")
ax3.set_xticks(list(x))
ax3.set_xticklabels(tone_labels)
ax3.set_title("Dataset Distribution by Skin Tone", color="white", fontsize=13, pad=12)
ax3.set_ylabel("Count", color="#94a3b8")
ax3.tick_params(colors="#94a3b8")
ax3.spines[:].set_color("#1e3a5f")
ax3.legend(labelcolor="white", facecolor="#0f1f3a", edgecolor="#1e3a5f")

plt.suptitle(f"DermAI Fairness Audit — {args.model}\nOverall AUC: {overall_auc:.4f}",
             color="white", fontsize=14, y=1.02)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, f"fairness_audit_{args.model.replace('.keras','')}.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0a1628")
print(f"\n✓ Plot saved: {out_path}")

# ── Save CSV report ───────────────────────────────────────────────────────────
report_df = pd.DataFrame(results).T
csv_path  = os.path.join(OUT_DIR, f"fairness_audit_{args.model.replace('.keras','')}.csv")
report_df.to_csv(csv_path)
print(f"✓ CSV saved:  {csv_path}")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Model        : {args.model}")
print(f"Overall AUC  : {overall_auc:.4f}")
for tone, r in results.items():
    print(f"  {r['label']:<20}: AUC={r['auc']:.4f}  Sensitivity={r['sens']:.3f}")
print(f"\nRun again after PAD-UFES training to compare:")
print(f"  python3 backend/models/fairness_audit.py --model pad_ufes_finetuned.keras")
