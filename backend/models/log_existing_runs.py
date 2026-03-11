"""
log_existing_runs.py — Log all existing training results into MLflow
Run this once to populate MLflow with your completed experiments.

Usage:
    python3 backend/models/log_existing_runs.py
"""
import os
import pickle
import mlflow
import mlflow.keras
import numpy as np

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_DIR = os.path.join(BASE_DIR, "saved_models")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("skin-cancer-detection")

# ── Run 1: HAM10000 baseline (your best model) ────────────────────────────────
print("Logging HAM10000 baseline run...")
with mlflow.start_run(run_name="ham10000_baseline"):
    # Params
    mlflow.log_params({
        "model":            "EfficientNetB0",
        "dataset":          "HAM10000",
        "img_size":         224,
        "batch_size":       8,
        "warmup_epochs":    10,
        "finetune_epochs":  50,
        "warmup_lr":        1e-3,
        "finetune_lr":      1e-5,
        "unfreeze_layers":  10,
        "loss":             "binary_crossentropy",
        "sample_weight":    2.03,
        "augmentation":     "RandomRotate90,Flip,BrightnessContrast,HueSaturation,GaussNoise,CoarseDropout",
        "optimizer":        "Adam",
    })

    # Metrics
    mlflow.log_metrics({
        "phase1_best_auc":  0.7527,
        "phase2_best_auc":  0.7691,
        "final_val_auc":    0.7691,
        "n_train_images":   8012,
        "n_val_images":     2003,
        "n_malignant":      1954,
        "n_benign":         8061,
    })

    # Tags
    mlflow.set_tags({
        "stage":    "foundation",
        "status":   "best_model",
        "gpu":      "RTX5060_Blackwell",
        "notes":    "Best HAM10000 result. Phase 2 fine-tuning improved over warmup.",
    })

    # Log model if exists
    model_path = os.path.join(SAVED_DIR, "ham10000_baseline.keras")
    if os.path.exists(model_path):
        mlflow.log_artifact(model_path, artifact_path="models")
        print(f"  ✓ Logged model: ham10000_baseline.keras")

    # Log history if exists
    hist_path = os.path.join(SAVED_DIR, "ham10000_final_history.pkl")
    if os.path.exists(hist_path):
        with open(hist_path, "rb") as f:
            history = pickle.load(f)
        # Log epoch-by-epoch AUC
        for i, auc in enumerate(history.get("val_auc", [])):
            mlflow.log_metric("val_auc_per_epoch", auc, step=i)
        print(f"  ✓ Logged {len(history.get('val_auc', []))} epoch metrics")

print("  ✓ HAM10000 baseline logged\n")

# ── Run 2: HAM10000 final (run 5 with load_weights fix) ──────────────────────
print("Logging HAM10000 final run...")
with mlflow.start_run(run_name="ham10000_final_loadweights_fix"):
    mlflow.log_params({
        "model":            "EfficientNetB0",
        "dataset":          "HAM10000",
        "img_size":         224,
        "batch_size":       8,
        "warmup_epochs":    20,
        "finetune_epochs":  50,
        "warmup_lr":        1e-3,
        "finetune_lr":      1e-5,
        "unfreeze_layers":  20,
        "loss":             "binary_crossentropy",
        "fix_applied":      "load_weights_before_unfreeze",
    })
    mlflow.log_metrics({
        "phase1_best_auc":  0.7527,
        "phase2_best_auc":  0.7607,
        "final_val_auc":    0.7578,
    })
    mlflow.set_tags({
        "stage":  "foundation",
        "status": "completed",
        "notes":  "Phase 2 improved over Phase 1 for first time due to load_weights fix. Still below baseline.",
    })
print("  ✓ Logged\n")

# ── Run 3: DDI fine-tuning (failed attempt) ───────────────────────────────────
print("Logging DDI failed attempt...")
with mlflow.start_run(run_name="ddi_finetuning_v1_failed"):
    mlflow.log_params({
        "model":            "EfficientNetB0",
        "dataset":          "DDI_only",
        "ddi_images":       656,
        "train_split":      524,
        "loss":             "binary_crossentropy",
        "finetune_lr":      1e-6,
    })
    mlflow.log_metrics({
        "phase1_best_auc":  0.4936,
        "phase2_best_auc":  0.4972,
        "final_val_auc":    0.4659,
    })
    mlflow.set_tags({
        "stage":      "ddi_finetuning",
        "status":     "failed",
        "root_cause": "DDI too small (656 images) for standalone fine-tuning",
        "fix":        "Mix DDI with HAM10000, oversample DDI 8x",
    })
print("  ✓ Logged\n")

# ── Run 4: DDI mixed training (current/completed) ─────────────────────────────
print("Logging DDI mixed training...")
with mlflow.start_run(run_name="ddi_mixed_ham10000_v2"):
    mlflow.log_params({
        "model":             "EfficientNetB0",
        "datasets":          "HAM10000+DDI",
        "ddi_oversample":    8,
        "combined_train":    12204,
        "combined_val":      2135,
        "warmup_epochs":     5,
        "finetune_epochs":   30,
        "warmup_lr":         1e-4,
        "finetune_lr":       1e-6,
        "unfreeze_layers":   20,
        "loss":              "binary_crossentropy",
        "fairness_aware":    True,
        "skin_tones":        "12,34,56",
    })
    mlflow.set_tags({
        "stage":   "ddi_finetuning",
        "status":  "completed",
        "novelty": "Fairness-aware training across Fitzpatrick skin tones",
    })

    # Log DDI history if exists
    hist_path = os.path.join(SAVED_DIR, "ddi_history.pkl")
    if os.path.exists(hist_path):
        with open(hist_path, "rb") as f:
            history = pickle.load(f)
        best_auc = max(history.get("val_auc", [0]))
        mlflow.log_metric("best_val_auc", best_auc)
        for i, auc in enumerate(history.get("val_auc", [])):
            mlflow.log_metric("val_auc_per_epoch", auc, step=i)
        print(f"  ✓ Best AUC from history: {best_auc:.4f}")

    model_path = os.path.join(SAVED_DIR, "ddi_finetuned.keras")
    if os.path.exists(model_path):
        mlflow.log_artifact(model_path, artifact_path="models")
        print(f"  ✓ Logged model: ddi_finetuned.keras")
print("  ✓ Logged\n")

print("="*50)
print("✓ All runs logged to MLflow!")
print("="*50)
print("\nTo view the MLflow UI, run:")
print("  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001")
print("Then open: http://127.0.0.1:5001")
