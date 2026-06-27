"""
Testing / Evaluation Script
============================
Evaluates a trained model on pre-engineered test or validation data.
Implements the full EDA-recommended evaluation protocol:

  Primary metric    : PR-AUC (Average Precision)
  Secondary metrics : Recall @ chosen threshold, Precision, F-beta (β=2)
  NOT the primary   : Accuracy (misleading at 533:1 imbalance — shown last with warning)
  Interpretability  : SHAP values (required for financial services compliance)
  Visualisations    : Precision-Recall curve, Confusion matrix, SHAP summary

Usage:
    # Evaluate on test data (uses threshold saved with model)
    python testing.py \\
        --model-path models/trained_model.pkl \\
        --data-path  data/engineered/test_features.parquet

    # Evaluate with a specific threshold
    python testing.py \\
        --model-path models/trained_model.pkl \\
        --data-path  data/engineered/test_features.parquet \\
        --threshold 0.35

    # Evaluate on validation data, skip SHAP (faster)
    python testing.py \\
        --model-path models/trained_model.pkl \\
        --data-path  data/engineered/val_features.parquet \\
        --output-dir eval/validation \\
        --no-shap

    # Evaluate with custom cost ratio for cost-aware metrics
    python testing.py \\
        --model-path models/trained_model.pkl \\
        --data-path  data/engineered/test_features.parquet \\
        --cost-fn-ratio 10
"""

import os
import json
import argparse
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    accuracy_score,
)

warnings.filterwarnings("ignore")

FRAUD_C  = "#C0392B"
NORMAL_C = "#2471A3"
ACCENT_C = "#117A65"


# ──────────────────────────────────────────────────────────────────────────────
# LOAD HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str) -> dict:
    """Load model bundle saved by training.py."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def load_data(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None]:
    """Load parquet, separate X and y."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")
    df = pd.read_parquet(data_path)
    y  = df["Class"] if "Class" in df.columns else None
    X  = df.drop(columns=["Class"]) if y is not None else df
    return df, X, y


# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(model, X: pd.DataFrame,
                  threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Return binary predictions and fraud probabilities."""
    y_proba = model.predict_proba(X)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)
    return y_pred, y_proba


# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray, threshold: float,
                    cost_fn_ratio: float = 10.0) -> dict:
    """
    Compute the full EDA-recommended metric set.

    Metric ordering follows the EDA priority:
      1. PR-AUC     — primary, threshold-free
      2. Recall     — minimise missed fraud
      3. Precision  — operational feasibility
      4. F2         — single-number comparison (weights recall 2× precision)
      5. F1         — standard harmonic mean for reference
      6. Total cost — FN × cost_fn_ratio + FP × 1
      7. ROC-AUC    — reference only (inflated by imbalance)
      8. Accuracy   — shown LAST with explicit warning

    Args:
        y_true        : Ground truth labels
        y_pred        : Binary predictions at chosen threshold
        y_proba       : Raw fraud probability scores
        threshold     : Classification threshold used
        cost_fn_ratio : Cost of 1 missed fraud relative to 1 false alarm
    """
    pr_auc  = average_precision_score(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    recall    = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    f2        = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    accuracy  = accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Operational cost — requires business input on cost_fn_ratio
    total_cost = cost_fn_ratio * fn + fp

    return {
        # PRIMARY
        "pr_auc"          : float(pr_auc),
        # THRESHOLD-DEPENDENT (core business metrics)
        "recall"          : float(recall),
        "precision"       : float(precision),
        "f2_score"        : float(f2),
        "f1_score"        : float(f1),
        # COST
        "total_cost"      : float(total_cost),
        "cost_fn_ratio"   : float(cost_fn_ratio),
        "fn_cost"         : float(cost_fn_ratio * fn),
        "fp_cost"         : float(fp),
        # CONFUSION MATRIX
        "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                             "fn": int(fn), "tp": int(tp)},
        "specificity"     : float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        # REFERENCE ONLY
        "roc_auc"         : float(roc_auc),
        # LAST — misleading under imbalance
        "accuracy"        : float(accuracy),
        "threshold"       : float(threshold),
    }


def print_metrics(metrics: dict) -> None:
    """Print metrics in EDA-recommended priority order."""
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    n_fraud = tp + fn
    n_total = tn + fp + fn + tp

    print("\n" + "═" * 60)
    print("  EVALUATION RESULTS")
    print("═" * 60)

    print(f"\n  ── PRIMARY METRIC ──────────────────────────────────────")
    print(f"  PR-AUC (Average Precision) : {metrics['pr_auc']:.4f}")
    print(f"  [Target: > 0.85  |  Random baseline ≈ 0.002]")

    print(f"\n  ── THRESHOLD-DEPENDENT METRICS  (@ {metrics['threshold']:.4f}) ──")
    print(f"  Recall    (fraud caught)   : {metrics['recall']:.4f}  "
          f"[{tp}/{n_fraud} frauds detected]")
    print(f"  Precision (flag accuracy)  : {metrics['precision']:.4f}  "
          f"[{tp}/{tp+fp} flagged are real fraud]")
    print(f"  F2 score  (recall×2 weight): {metrics['f2_score']:.4f}")
    print(f"  F1 score                   : {metrics['f1_score']:.4f}")

    print(f"\n  ── OPERATIONAL COST  (FN cost = {metrics['cost_fn_ratio']:.0f}× FP cost) ──")
    print(f"  Missed fraud (FN)    : {fn}  → cost = {metrics['fn_cost']:.0f} units")
    print(f"  False alarms (FP)    : {fp}  → cost = {metrics['fp_cost']:.0f} units")
    print(f"  Total cost           : {metrics['total_cost']:.0f} units")

    print(f"\n  ── CONFUSION MATRIX ─────────────────────────────────────")
    print(f"  {'':>18}  Predicted Normal  Predicted Fraud")
    print(f"  {'Actual Normal':>18}  {tn:>16,}  {fp:>15,}")
    print(f"  {'Actual Fraud':>18}  {fn:>16,}  {tp:>15,}")

    print(f"\n  ── REFERENCE METRICS ────────────────────────────────────")
    print(f"  ROC-AUC     : {metrics['roc_auc']:.4f}  "
          f"[inflated by {tn:,} true negatives — not primary]")
    print(f"  Specificity : {metrics['specificity']:.4f}")

    print(f"\n  ── ⚠  ACCURACY (DO NOT USE AS PRIMARY METRIC) ──────────")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  A model predicting ALL transactions as normal would score "
          f"{1 - n_fraud/n_total:.4f}")
    print("═" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# VISUALISATIONS
# ──────────────────────────────────────────────────────────────────────────────

def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray,
                                threshold: float, pr_auc: float,
                                output_path: str) -> None:
    """
    Plot the Precision-Recall curve with the chosen threshold marked.
    This is the primary evaluation visualisation for imbalanced datasets.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    baseline = y_true.mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Precision-Recall Analysis", fontsize=13, fontweight="bold")

    # ── Left: PR curve ────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(recalls, precisions, color=FRAUD_C, lw=2, label=f"PR curve (AUC = {pr_auc:.4f})")
    ax.axhline(baseline, color="grey", lw=1, ls="--",
               label=f"Random baseline ({baseline:.4f})")

    # Mark chosen threshold
    idx = np.argmin(np.abs(thresholds - threshold))
    ax.scatter(recalls[idx], precisions[idx], s=120, color=ACCENT_C, zorder=5,
               label=f"Chosen threshold ({threshold:.3f})\n"
                     f"P={precisions[idx]:.3f}, R={recalls[idx]:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.4)

    # ── Right: Precision and Recall vs Threshold ───────────────────────────────
    ax2 = axes[1]
    ax2.plot(thresholds, precisions[:-1], color=NORMAL_C, lw=2, label="Precision")
    ax2.plot(thresholds, recalls[:-1],    color=FRAUD_C,  lw=2, label="Recall")
    ax2.axvline(threshold, color=ACCENT_C, lw=1.5, ls="--",
                label=f"Chosen threshold ({threshold:.3f})")
    ax2.axhline(0.80, color="grey", lw=1, ls=":", alpha=0.7, label="Recall target (0.80)")
    ax2.axhline(0.70, color="grey", lw=1, ls=":", alpha=0.5, label="Precision target (0.70)")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Score")
    ax2.set_title("Precision & Recall vs Threshold", fontweight="bold")
    ax2.legend(loc="center left", fontsize=8)
    ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.05])
    ax2.grid(alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  PR curve saved → {output_path}")


def plot_confusion_matrix(metrics: dict, output_path: str) -> None:
    """Plot styled confusion matrix with recall and precision annotations."""
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Confusion Matrix Analysis", fontsize=13, fontweight="bold")

    # ── Raw counts ────────────────────────────────────────────────────────────
    matrix      = np.array([[tn, fp], [fn, tp]])
    row_sums    = matrix.sum(axis=1, keepdims=True)
    norm_matrix = matrix / row_sums.clip(min=1)

    im = axes[0].imshow(norm_matrix, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["Pred Normal", "Pred Fraud"])
    axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["Actual Normal", "Actual Fraud"])
    axes[0].set_title("Normalised Confusion Matrix\n(row = actual class)",
                       fontweight="bold")
    for i in range(2):
        for j in range(2):
            raw = matrix[i, j]
            pct = norm_matrix[i, j]
            color = "white" if pct > 0.6 else "black"
            axes[0].text(j, i, f"{raw:,}\n({pct:.1%})",
                         ha="center", va="center", color=color, fontsize=10)
    plt.colorbar(im, ax=axes[0])

    # ── Business cost breakdown ───────────────────────────────────────────────
    cost_labels   = ["Correct\n(TN)", "False Alarm\n(FP)", "Missed Fraud\n(FN)", "Correct\n(TP)"]
    cost_values   = [tn, fp, fn, tp]
    cost_colors   = [NORMAL_C, WARN_C, FRAUD_C, ACCENT_C]
    WARN_C_local  = "#D4AC0D"
    cost_colors   = [NORMAL_C, WARN_C_local, FRAUD_C, ACCENT_C]
    bars = axes[1].bar(cost_labels, cost_values, color=cost_colors, alpha=0.8, edgecolor="white")
    axes[1].set_title(f"Outcome Breakdown\n"
                       f"(FN cost = {metrics['cost_fn_ratio']:.0f}× FP cost  →  "
                       f"Total = {metrics['total_cost']:.0f} units)",
                       fontweight="bold")
    axes[1].set_ylabel("Count")
    for bar, val in zip(bars, cost_values):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + max(cost_values) * 0.01,
                     f"{val:,}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved → {output_path}")


def run_shap_analysis(model, X: pd.DataFrame, output_dir: str,
                      max_samples: int = 500) -> None:
    """
    Compute and plot SHAP values for model interpretability.

    Required for financial services regulatory compliance.
    EDA predicted: V12_amount and V14 will dominate top SHAP values.
    SHAP explains WHY each individual prediction was made — critical for
    fraud investigation teams to understand model decisions.

    Args:
        model       : Trained XGBoost or LightGBM model
        X           : Feature matrix (subsample for speed)
        output_dir  : Directory to save SHAP plots
        max_samples : Number of samples for SHAP computation (500 is enough for summary)
    """
    try:
        import shap
    except ImportError:
        print("  ⚠  SHAP not installed. Run: pip install shap")
        return

    print(f"\nComputing SHAP values (n={min(max_samples, len(X))})...")

    # Subsample for speed — SHAP TreeExplainer scales O(n) but 500 is representative
    if len(X) > max_samples:
        idx       = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X_sample  = X.iloc[idx]
    else:
        X_sample  = X

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classifiers, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # ── Summary bar plot (mean |SHAP|) ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    plt.sca(axes[0])
    shap.summary_plot(shap_values, X_sample, plot_type="bar",
                      show=False, max_display=20)
    axes[0].set_title("SHAP Feature Importance\n(Mean |SHAP value|)",
                       fontweight="bold", fontsize=11)

    # ── Beeswarm plot (direction and magnitude) ───────────────────────────────
    plt.sca(axes[1])
    shap.summary_plot(shap_values, X_sample, plot_type="dot",
                      show=False, max_display=20)
    axes[1].set_title("SHAP Summary Plot\n(Red = high feature value increases fraud probability)",
                       fontweight="bold", fontsize=11)

    plt.suptitle("SHAP Model Interpretability — Credit Card Fraud Model",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    shap_path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(shap_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  SHAP summary saved → {shap_path}")

    # ── Save mean |SHAP| values as JSON for downstream use ───────────────────
    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X_sample.columns
    ).sort_values(ascending=False)

    shap_json_path = os.path.join(output_dir, "shap_importance.json")
    with open(shap_json_path, "w") as f:
        json.dump(mean_shap.to_dict(), f, indent=2)
    print(f"  SHAP importance  → {shap_json_path}")

    print(f"\n  Top 10 features by mean |SHAP|:")
    for feat, val in mean_shap.head(10).items():
        print(f"    {feat:25s}: {val:.5f}")


# ──────────────────────────────────────────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────────────────────────────────────────

def save_predictions(X: pd.DataFrame, y_pred: np.ndarray,
                     y_proba: np.ndarray, output_path: str) -> None:
    """Save predictions alongside fraud probability score."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pd.DataFrame({
        "predicted_label"      : y_pred,
        "fraud_probability"    : y_proba,
    }).to_csv(output_path, index=False)
    print(f"  Predictions saved → {output_path}")


def save_metrics(metrics: dict, output_path: str) -> None:
    """Save all metrics to JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved    → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Model Evaluation Pipeline — Credit Card Fraud Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", type=str,
                        default="models/trained_model.pkl",
                        help="Path to trained model pickle")
    parser.add_argument("--data-path", type=str,
                        default="data/engineered/test_features.parquet",
                        help="Path to engineered test/validation parquet")
    parser.add_argument("--output-dir", type=str, default="eval",
                        help="Output directory for metrics, plots, predictions")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override threshold (default: load from model)")
    parser.add_argument("--cost-fn-ratio", type=float, default=10.0,
                        help="Cost of 1 FN relative to 1 FP (default: 10.0)")
    parser.add_argument("--no-shap", action="store_true",
                        help="Skip SHAP analysis (faster, but loses interpretability)")
    parser.add_argument("--shap-samples", type=int, default=500,
                        help="Number of samples for SHAP computation (default: 500)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model and data ───────────────────────────────────────────────────
    print(f"Loading model  : {args.model_path}")
    model_bundle = load_model(args.model_path)
    model        = model_bundle["model"]
    model_type   = model_bundle.get("model_type", "unknown")
    saved_thresh = model_bundle.get("threshold", 0.5)

    threshold = args.threshold if args.threshold is not None else saved_thresh
    print(f"Model type     : {model_type}")
    print(f"Threshold      : {threshold:.4f}"
          + (" (from model)" if args.threshold is None else " (manual override)"))

    print(f"\nLoading data   : {args.data_path}")
    df, X, y = load_data(args.data_path)
    print(f"Data shape     : {X.shape}")

    if y is None:
        # Inference-only mode (no ground truth)
        print("\n⚠  No 'Class' column found. Running inference only.")
        y_pred, y_proba = run_inference(model, X, threshold)
        save_predictions(X, y_pred, y_proba,
                         os.path.join(args.output_dir, "predictions.csv"))
        print("✓  Predictions saved. No metrics computed (no ground truth).")
        return

    # ── Inference ─────────────────────────────────────────────────────────────
    print(f"\nRunning inference on {len(X):,} samples...")
    y_pred, y_proba = run_inference(model, X, threshold)

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\nComputing metrics...")
    metrics = compute_metrics(
        y.values, y_pred, y_proba,
        threshold=threshold,
        cost_fn_ratio=args.cost_fn_ratio
    )
    print_metrics(metrics)

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\nSaving outputs...")
    save_predictions(X, y_pred, y_proba,
                     os.path.join(args.output_dir, "predictions.csv"))
    save_metrics(metrics,
                 os.path.join(args.output_dir, "evaluation_metrics.json"))

    plot_precision_recall_curve(
        y.values, y_proba, threshold, metrics["pr_auc"],
        os.path.join(args.output_dir, "precision_recall_curve.png")
    )
    plot_confusion_matrix(
        metrics,
        os.path.join(args.output_dir, "confusion_matrix.png")
    )

    # ── SHAP ──────────────────────────────────────────────────────────────────
    if not args.no_shap:
        if model_type in ("xgboost", "lightgbm"):
            run_shap_analysis(model, X, args.output_dir, args.shap_samples)
        else:
            print(f"\n  ⚠  SHAP TreeExplainer requires XGBoost or LightGBM. "
                  f"Skipping SHAP for {model_type}.")
    else:
        print("\n  SHAP skipped (--no-shap flag set).")

    print(f"\n✓  Evaluation complete. All outputs in: {args.output_dir}/")


if __name__ == "__main__":
    main()