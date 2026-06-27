"""
Training Script
================
Trains XGBoost or LightGBM on pre-engineered features.
Implements stratified K-fold cross-validation and threshold tuning
via the Precision-Recall curve.

EDA-driven design decisions:
  - Primary model  : XGBoost / LightGBM with scale_pos_weight = 533
  - NOT recommended: Logistic Regression, Linear SVM (max r = 0.26 with fraud)
  - NOT recommended: KNN (scale-dependent, slow at 170K rows)
  - Imbalance      : scale_pos_weight first; SMOTE applied in feature_engineering.py
  - Evaluation     : PR-AUC (primary), Recall @ Precision≥0.70, F-beta (β=2)
  - Threshold      : tuned on validation PR curve, NOT fixed at 0.5
  - CV             : StratifiedKFold (5 or 10 folds) — only 320 fraud cases

Usage:
    # Train XGBoost (recommended — matches EDA recommendation)
    python training.py --model-type xgboost \\
        --train-data data/engineered/train_features.parquet \\
        --val-data   data/engineered/val_features.parquet

    # Train LightGBM (alternative — faster, similar performance)
    python training.py --model-type lightgbm \\
        --train-data data/engineered/train_features.parquet \\
        --val-data   data/engineered/val_features.parquet

    # Train with custom cost ratio (cost_fn_ratio = cost_of_miss / cost_of_fp)
    python training.py --model-type xgboost \\
        --train-data data/engineered/train_features.parquet \\
        --val-data   data/engineered/val_features.parquet \\
        --cost-fn-ratio 10

    # Train with manual threshold override
    python training.py --model-type xgboost \\
        --train-data data/engineered/train_features.parquet \\
        --val-data   data/engineered/val_features.parquet \\
        --threshold 0.35

    # Run 5-fold stratified CV before final training
    python training.py --model-type xgboost \\
        --train-data data/engineered/train_features.parquet \\
        --val-data   data/engineered/val_features.parquet \\
        --cv-folds 5
"""

import os
import json
import argparse
import joblib
import warnings
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    recall_score,
    precision_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# MODEL BUILDERS
# ──────────────────────────────────────────────────────────────────────────────

RECOMMENDED_MODELS = ("xgboost", "lightgbm")
NOT_RECOMMENDED    = ("logistic", "knn", "randomforest")

NOT_RECOMMENDED_MSG = (
    "⚠  {model} is NOT recommended for this dataset.\n"
    "   EDA finding: max Pearson r with fraud = 0.26 — patterns are non-linear.\n"
    "   Logistic Regression and KNN perform poorly on non-linear PCA-feature fraud signals.\n"
    "   Recommended: --model-type xgboost  or  --model-type lightgbm"
)


def compute_scale_pos_weight(y: pd.Series) -> float:
    """
    Compute scale_pos_weight = n_negative / n_positive.
    EDA found 533:1 ratio. This is passed directly to XGBoost/LightGBM
    so the model internally penalises fraud misclassification 533× more.
    """
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    spw   = n_neg / n_pos if n_pos > 0 else 1.0
    return float(spw)


def build_xgboost(scale_pos_weight: float, **kwargs):
    """
    Build XGBoost with EDA-recommended defaults.

    Key hyperparameters for tuning (from EDA Section 14.4):
        max_depth       : 4–7  (deeper = more complex boundaries)
        min_child_weight: 1–10 (higher = more conservative, prevents overfitting on 320 fraud cases)
        subsample       : 0.7–1.0
        colsample_bytree: 0.7–1.0
        learning_rate   : 0.05–0.15
    """
    from xgboost import XGBClassifier

    defaults = dict(
        n_estimators      = 500,
        learning_rate     = 0.05,
        max_depth         = 6,
        min_child_weight  = 3,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = scale_pos_weight,
        eval_metric       = "aucpr",     # optimise PR-AUC directly
        early_stopping_rounds = 30,
        random_state      = 42,
        n_jobs            = -1,
        verbosity         = 0,
    )
    defaults.update(kwargs)
    return XGBClassifier(**defaults)


def build_lightgbm(scale_pos_weight: float, **kwargs):
    """
    Build LightGBM with EDA-recommended defaults.
    LightGBM equivalent of scale_pos_weight is is_unbalance=True or scale_pos_weight.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        raise ImportError("LightGBM required: pip install lightgbm")

    defaults = dict(
        n_estimators       = 500,
        learning_rate      = 0.05,
        max_depth          = 6,
        min_child_samples  = 20,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        scale_pos_weight   = scale_pos_weight,
        metric             = "average_precision",
        early_stopping_round = 30,
        random_state       = 42,
        n_jobs             = -1,
        verbose            = -1,
    )
    defaults.update(kwargs)
    return LGBMClassifier(**defaults)


def build_model(model_type: str, scale_pos_weight: float):
    """Factory function — returns an untrained model."""
    if model_type in NOT_RECOMMENDED:
        print(NOT_RECOMMENDED_MSG.format(model=model_type))

    if model_type == "xgboost":
        return build_xgboost(scale_pos_weight)
    elif model_type == "lightgbm":
        return build_lightgbm(scale_pos_weight)
    elif model_type == "randomforest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            n_jobs=-1, random_state=42
        )
    elif model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                         f"Choose from: xgboost, lightgbm, randomforest, logistic")


# ──────────────────────────────────────────────────────────────────────────────
# THRESHOLD SELECTION
# ──────────────────────────────────────────────────────────────────────────────

def select_threshold(y_true: np.ndarray,
                     y_proba: np.ndarray,
                     min_recall: float = 0.80,
                     cost_fn_ratio: float = 10.0) -> dict:
    """
    Select the optimal classification threshold using the Precision-Recall curve.

    Two selection strategies — both are computed and reported:

    Strategy A — Recall-floor: highest Precision where Recall >= min_recall.
      Use when you have a hard business requirement on minimum fraud catch rate.
      EDA recommendation: min_recall = 0.80 (catch ≥ 80% of fraud).

    Strategy B — Cost-ratio: minimise total expected cost.
      cost = FN × cost_fn_ratio + FP × 1
      Use when you can quantify the relative cost of a missed fraud vs false alarm.
      EDA recommendation: cost_fn_ratio = 10 (missing fraud is 10× worse than FP).
      Adjust based on actual business data.

    Args:
        y_true        : Ground truth labels
        y_proba       : Model probability scores for the positive (fraud) class
        min_recall    : Minimum acceptable recall (Strategy A)
        cost_fn_ratio : Cost of 1 FN relative to 1 FP (Strategy B)

    Returns:
        dict with both thresholds and their full metrics
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns one extra precision/recall point at the end
    # Align thresholds with precision/recall arrays
    precisions = precisions[:-1]
    recalls    = recalls[:-1]

    # ── Strategy A: Recall-floor ──────────────────────────────────────────────
    valid_mask = recalls >= min_recall
    if valid_mask.any():
        best_idx_a    = np.argmax(precisions[valid_mask])
        threshold_a   = thresholds[valid_mask][best_idx_a]
        precision_a   = precisions[valid_mask][best_idx_a]
        recall_a      = recalls[valid_mask][best_idx_a]
    else:
        # If no threshold achieves min_recall, take the threshold with max recall
        best_idx_a    = np.argmax(recalls)
        threshold_a   = thresholds[best_idx_a]
        precision_a   = precisions[best_idx_a]
        recall_a      = recalls[best_idx_a]
        print(f"  ⚠  No threshold achieves Recall ≥ {min_recall:.0%}. "
              f"Best achievable: {recall_a:.3f}")

    # ── Strategy B: Cost-ratio minimisation ───────────────────────────────────
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    # At each threshold: FN = n_pos × (1 - recall), FP = n_neg × (1 - precision/precision_at_baseline)
    # Approximate FP from precision and recall:  FP = TP × (1/precision - 1)
    tp_approx   = recalls * n_pos
    fp_approx   = np.where(precisions > 0, tp_approx * (1 / precisions - 1), n_neg)
    fn_approx   = n_pos * (1 - recalls)
    total_cost  = cost_fn_ratio * fn_approx + fp_approx
    best_idx_b  = np.argmin(total_cost)
    threshold_b = thresholds[best_idx_b]
    precision_b = precisions[best_idx_b]
    recall_b    = recalls[best_idx_b]

    return {
        "strategy_a": {
            "name"        : f"Recall-floor (min_recall={min_recall:.0%})",
            "threshold"   : float(threshold_a),
            "precision"   : float(precision_a),
            "recall"      : float(recall_a),
            "f1"          : float(2 * precision_a * recall_a /
                                  (precision_a + recall_a + 1e-9)),
        },
        "strategy_b": {
            "name"        : f"Cost-ratio (FN cost = {cost_fn_ratio}× FP cost)",
            "threshold"   : float(threshold_b),
            "precision"   : float(precision_b),
            "recall"      : float(recall_b),
            "f1"          : float(2 * precision_b * recall_b /
                                  (precision_b + recall_b + 1e-9)),
        },
        "pr_curve": {
            "precisions"  : precisions.tolist(),
            "recalls"     : recalls.tolist(),
            "thresholds"  : thresholds.tolist(),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# CROSS-VALIDATION
# ──────────────────────────────────────────────────────────────────────────────

def run_cross_validation(X: pd.DataFrame, y: pd.Series,
                         model_type: str, scale_pos_weight: float,
                         n_folds: int = 5) -> dict:
    """
    Run stratified K-fold cross-validation.

    Stratification ensures each fold has the same fraud prevalence (~0.19%).
    With only 320 fraud cases across 5 folds, each fold gets ~64 fraud samples.
    Primary metric is PR-AUC — NOT accuracy.

    Args:
        X               : Feature matrix (after preprocessing)
        y               : Labels
        model_type      : 'xgboost' or 'lightgbm'
        scale_pos_weight: Imbalance ratio for the model
        n_folds         : Number of CV folds (5 or 10 recommended)

    Returns:
        dict with per-fold and aggregate metrics
    """
    skf      = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []

    print(f"\nRunning {n_folds}-fold stratified cross-validation...")
    print(f"{'Fold':>4}  {'PR-AUC':>8}  {'Recall':>8}  {'Precision':>10}  "
          f"{'F2':>8}  {'ROC-AUC':>8}")
    print("-" * 55)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = build_model(model_type, scale_pos_weight)

        # XGBoost / LightGBM support eval_set for early stopping
        if model_type in ("xgboost", "lightgbm"):
            model.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
        else:
            model.fit(X_tr, y_tr)

        y_proba = model.predict_proba(X_val)[:, 1]

        pr_auc     = average_precision_score(y_val, y_proba)
        roc_auc    = roc_auc_score(y_val, y_proba)

        # Use threshold from recall-floor strategy for fold metrics
        th_info    = select_threshold(y_val.values, y_proba, min_recall=0.80)
        thresh     = th_info["strategy_a"]["threshold"]
        y_pred     = (y_proba >= thresh).astype(int)

        rec        = recall_score(y_val, y_pred, zero_division=0)
        prec       = precision_score(y_val, y_pred, zero_division=0)
        f2         = fbeta_score(y_val, y_pred, beta=2, zero_division=0)

        fold_metrics.append({
            "fold": fold, "pr_auc": pr_auc, "roc_auc": roc_auc,
            "recall": rec, "precision": prec, "f2": f2, "threshold": thresh,
        })
        print(f"{fold:>4}  {pr_auc:>8.4f}  {rec:>8.4f}  {prec:>10.4f}  "
              f"{f2:>8.4f}  {roc_auc:>8.4f}")

    # Aggregate
    pr_aucs = [m["pr_auc"] for m in fold_metrics]
    recalls = [m["recall"] for m in fold_metrics]
    f2s     = [m["f2"]     for m in fold_metrics]

    print("-" * 55)
    print(f"{'Mean':>4}  {np.mean(pr_aucs):>8.4f}  {np.mean(recalls):>8.4f}  "
          f"{'':>10}  {np.mean(f2s):>8.4f}")
    print(f"{'Std':>4}  {np.std(pr_aucs):>8.4f}  {np.std(recalls):>8.4f}  "
          f"{'':>10}  {np.std(f2s):>8.4f}")

    return {
        "folds"      : fold_metrics,
        "mean_pr_auc": float(np.mean(pr_aucs)),
        "std_pr_auc" : float(np.std(pr_aucs)),
        "mean_recall": float(np.mean(recalls)),
        "mean_f2"    : float(np.mean(f2s)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# FINAL TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def train_final_model(X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series,
                      model_type: str, scale_pos_weight: float):
    """
    Train the final model on full training data with val set for early stopping.
    Early stopping prevents overfitting given only 320 fraud training examples.
    """
    model = build_model(model_type, scale_pos_weight)

    print(f"\nTraining final {model_type} model...")
    print(f"  Train: {X_train.shape}  Class dist: {Counter(y_train)}")
    print(f"  Val  : {X_val.shape}   Class dist: {Counter(y_val)}")

    if model_type in ("xgboost", "lightgbm"):
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
    else:
        model.fit(X_train, y_train)

    return model


# ──────────────────────────────────────────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────────────────────────────────────────

def save_model(model, output_path: str, model_type: str,
               threshold: float, feature_names: list,
               scale_pos_weight: float, metadata: dict) -> None:
    """Save model with full metadata bundle."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    bundle = {
        "model"           : model,
        "model_type"      : model_type,
        "threshold"       : threshold,
        "feature_names"   : feature_names,
        "scale_pos_weight": scale_pos_weight,
        "metadata"        : metadata,
    }
    joblib.dump(bundle, output_path)
    print(f"  Saved model → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Model Training Pipeline — Credit Card Fraud Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-type", choices=["xgboost", "lightgbm",
                                                   "randomforest", "logistic"],
                        default="xgboost",
                        help="Model type (recommended: xgboost or lightgbm)")
    parser.add_argument("--train-data", type=str,
                        default="data/engineered/train_features.parquet",
                        help="Path to engineered training parquet")
    parser.add_argument("--val-data", type=str,
                        default="data/engineered/val_features.parquet",
                        help="Path to engineered validation parquet")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Output directory for model and metrics")
    parser.add_argument("--cv-folds", type=int, default=0,
                        help="Number of CV folds (0 = skip CV, just train final)")
    parser.add_argument("--min-recall", type=float, default=0.80,
                        help="Minimum recall for threshold Strategy A (default: 0.80)")
    parser.add_argument("--cost-fn-ratio", type=float, default=10.0,
                        help="Cost of 1 FN relative to 1 FP for threshold Strategy B "
                             "(default: 10.0 — missing fraud is 10× worse than false alarm)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override threshold (skips automatic PR-curve selection)")

    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading training data: {args.train_data}")
    train_df = pd.read_parquet(args.train_data)
    print(f"Loading validation data: {args.val_data}")
    val_df   = pd.read_parquet(args.val_data)

    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"]
    X_val   = val_df.drop(columns=["Class"])
    y_val   = val_df["Class"]

    scale_pos_weight = compute_scale_pos_weight(y_train)
    print(f"\nscale_pos_weight = {scale_pos_weight:.1f}  "
          f"(n_normal={int((y_train==0).sum())}, n_fraud={int((y_train==1).sum())})")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Optional CV ────────────────────────────────────────────────────────────
    cv_results = None
    if args.cv_folds > 0:
        cv_results = run_cross_validation(
            X_train, y_train, args.model_type,
            scale_pos_weight, n_folds=args.cv_folds
        )
        cv_path = os.path.join(args.output_dir, "cv_results.json")
        with open(cv_path, "w") as f:
            json.dump(cv_results, f, indent=2)
        print(f"\n  CV results saved → {cv_path}")
        print(f"  Mean PR-AUC : {cv_results['mean_pr_auc']:.4f} "
              f"± {cv_results['std_pr_auc']:.4f}")

    # ── Final training ─────────────────────────────────────────────────────────
    model = train_final_model(
        X_train, y_train, X_val, y_val,
        args.model_type, scale_pos_weight
    )

    # ── Threshold selection ────────────────────────────────────────────────────
    print("\nSelecting optimal threshold on validation set PR curve...")
    y_val_proba  = model.predict_proba(X_val)[:, 1]
    val_pr_auc   = average_precision_score(y_val, y_val_proba)
    print(f"  Validation PR-AUC : {val_pr_auc:.4f}")

    if args.threshold is not None:
        # Manual override
        chosen_threshold = args.threshold
        print(f"  Using manual threshold override: {chosen_threshold:.4f}")
        threshold_info = {"manual_override": chosen_threshold}
    else:
        threshold_info = select_threshold(
            y_val.values, y_val_proba,
            min_recall   = args.min_recall,
            cost_fn_ratio= args.cost_fn_ratio,
        )
        print(f"\n  Strategy A ({threshold_info['strategy_a']['name']}):")
        sa = threshold_info["strategy_a"]
        print(f"    Threshold = {sa['threshold']:.4f}  "
              f"Precision = {sa['precision']:.4f}  "
              f"Recall = {sa['recall']:.4f}  "
              f"F1 = {sa['f1']:.4f}")

        print(f"\n  Strategy B ({threshold_info['strategy_b']['name']}):")
        sb = threshold_info["strategy_b"]
        print(f"    Threshold = {sb['threshold']:.4f}  "
              f"Precision = {sb['precision']:.4f}  "
              f"Recall = {sb['recall']:.4f}  "
              f"F1 = {sb['f1']:.4f}")

        # Default to Strategy A for the saved model
        chosen_threshold = sa["threshold"]
        print(f"\n  → Saving model with Strategy A threshold: {chosen_threshold:.4f}")
        print(f"    (Override with --threshold X.XX if Strategy B is preferred)")

    # ── Save model ─────────────────────────────────────────────────────────────
    model_path = os.path.join(args.output_dir, "trained_model.pkl")
    save_model(
        model, model_path, args.model_type,
        threshold=chosen_threshold,
        feature_names=X_train.columns.tolist(),
        scale_pos_weight=scale_pos_weight,
        metadata={
            "val_pr_auc"     : val_pr_auc,
            "threshold_info" : threshold_info,
            "cv_results"     : cv_results,
            "train_shape"    : list(X_train.shape),
            "val_shape"      : list(X_val.shape),
            "cost_fn_ratio"  : args.cost_fn_ratio,
            "min_recall_target": args.min_recall,
        }
    )

    # Save threshold report
    th_path = os.path.join(args.output_dir, "threshold_report.json")
    report  = {
        "chosen_threshold" : chosen_threshold,
        "val_pr_auc"       : val_pr_auc,
        "threshold_info"   : {
            k: v for k, v in threshold_info.items() if k != "pr_curve"
        },
    }
    with open(th_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Training complete.")
    print(f"  Model     → {model_path}")
    print(f"  Threshold → {th_path}")
    if cv_results:
        print(f"  CV        → {os.path.join(args.output_dir, 'cv_results.json')}")


if __name__ == "__main__":
    main()