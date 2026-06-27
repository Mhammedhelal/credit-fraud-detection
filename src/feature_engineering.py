"""
Feature Engineering Script
===========================
Transforms raw credit card transaction data into the production feature set
defined by the EDA investigation (EDA_Story_Driven.ipynb).

Key EDA decisions implemented here:
  - log_amount     : replaces raw Amount (skewness 19.99 → 0.80)
  - amount_bin     : 5-quantile bins fitted on TRAIN only (U-shaped fraud rate)
  - Hour           : extracted from Time (seconds → 0-23)
  - sin_hour/cos_hour : cyclical encoding (fraud window spans midnight boundary)
  - V12_amount     : V12 × log_amount  (r = -0.21 with fraud)
  - V12_hour       : V12 × Hour        (r = -0.21 with fraud)
  - V7_amount      : V7  × log_amount  (independent secondary signal)
  - V11_hour       : V11 × Hour        (independent secondary signal)

Features deliberately NOT created (confirmed useless in EDA):
  - is_outlier_amount, V*_is_outlier   (r ≈ 0.009 — noise)
  - amount_hour_interaction            (r = -0.01 — noise)
  - V17/V14/V10 × any feature          (dilutes strong base signal)

Preprocessing:
  - OrdinalEncoder on amount_bin with explicit category order
  - NO StandardScaler — XGBoost/LightGBM are scale-invariant
  - Fitted ONLY on training data; applied to val/test without refitting

Resampling (train mode only):
  - Primary strategy: scale_pos_weight = 533 (passed to model, not applied here)
  - Optional SMOTE fallback: --apply-smote flag, applied AFTER split, inside fold

Usage:
    # Step 1: Process training data (fit transformers)
    python feature_engineering.py --mode train \\
        --input-data data/train.csv \\
        --output-dir data/engineered

    # Step 2: Process validation data (apply saved transformers)
    python feature_engineering.py --mode val \\
        --input-data data/val.csv \\
        --output-dir data/engineered

    # Step 3: Process test data (apply saved transformers)
    python feature_engineering.py --mode test \\
        --input-data data/test.csv \\
        --output-dir data/engineered

    # Optional: process train WITH SMOTE fallback (use only if PR-AUC < 0.70 without it)
    python feature_engineering.py --mode train \\
        --input-data data/train.csv \\
        --output-dir data/engineered \\
        --apply-smote
"""

import os
import json
import argparse
import joblib
import warnings
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# Explicit ordinal order — critical so XGBoost sees monotonic integer encoding
# even though fraud rate is U-shaped (tree splits handle non-monotonicity)
AMOUNT_BIN_ORDER = ["Very Low", "Low", "Medium", "High", "Very High"]
N_AMOUNT_BINS    = 5

# Production feature set from EDA Section 13.6
V_FEATURES = [f"V{i}" for i in range(1, 29)]

PRODUCTION_FEATURES = (
    V_FEATURES
    + ["log_amount", "amount_bin", "Hour", "sin_hour", "cos_hour",
       "V12_amount", "V12_hour", "V7_amount", "V11_hour"]
)


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────────────

def apply_feature_engineering(df: pd.DataFrame,
                               train_stats: dict | None = None
                               ) -> tuple[pd.DataFrame, dict]:
    """
    Apply all EDA-motivated feature transformations.

    When train_stats is None (training mode): compute and return bin edges.
    When train_stats is provided (val/test mode): apply pre-fitted bin edges.

    Args:
        df          : Raw dataframe with columns Time, V1-V28, Amount, Class
        train_stats : Dict with 'amount_bin_edges' from training fit (None = train mode)

    Returns:
        df_out      : Dataframe with all engineered features added
        stats       : Dict with fit statistics (bin edges) to save for val/test
    """
    df = df.copy()

    # ── 1. Hour of day (0–23) ─────────────────────────────────────────────────
    # Modulo 24 maps any elapsed-seconds value to the correct hour
    df["Hour"] = (df["Time"] // 3600) % 24

    # ── 2. log_amount — replaces raw Amount ───────────────────────────────────
    # log1p handles Amount = 0 (log(0+1) = 0)
    df["log_amount"] = np.log1p(df["Amount"])

    # ── 3. amount_bin — 5-quantile binning fitted on TRAIN only ───────────────
    if train_stats is None:
        # Training mode: fit the bin edges on this data
        _, bin_edges = pd.qcut(
            df["log_amount"], q=N_AMOUNT_BINS,
            labels=AMOUNT_BIN_ORDER, retbins=True, duplicates="drop"
        )
        stats = {"amount_bin_edges": bin_edges.tolist()}
        print(f"  amount_bin edges (log scale): {np.round(bin_edges, 3).tolist()}")
        print(f"  amount_bin edges ($ scale)  : "
              f"{['${:.2f}'.format(np.expm1(e)) for e in bin_edges]}")
    else:
        # Val/test mode: reuse training bin edges — never refit
        bin_edges = np.array(train_stats["amount_bin_edges"])
        stats = train_stats

    # Apply binning with pre-fitted edges (both modes)
    # Clip to avoid out-of-range values in val/test
    df["amount_bin"] = pd.cut(
        df["log_amount"],
        bins=bin_edges,
        labels=AMOUNT_BIN_ORDER,
        include_lowest=True
    )
    # Fill any NaN (out-of-range val/test values) with nearest bin
    df["amount_bin"] = df["amount_bin"].cat.add_categories(["Very Low", "Very High"])
    df.loc[df["log_amount"] < bin_edges[0], "amount_bin"] = "Very Low"
    df.loc[df["log_amount"] > bin_edges[-1], "amount_bin"] = "Very High"
    df["amount_bin"] = df["amount_bin"].astype(str)

    # ── 4. Cyclical time encoding ──────────────────────────────────────────────
    # sin/cos encoding preserves the circular structure of hour-of-day.
    # Without this, a linear model treats hour 23 and hour 0 as 23 steps apart;
    # with sin/cos they are adjacent on the unit circle — correct for the
    # midnight fraud window (hours 22–2) discovered in EDA Section 7.
    df["sin_hour"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["Hour"] / 24)

    # ── 5. Interaction features — validated in EDA Section 11.2 & 13.4 ────────
    # V12 is the ONLY V-feature correlated with Hour (r = 0.35).
    # Both products outperform V12 alone (r = -0.26) in non-linear signal.
    # V17, V14, V10 interactions were tested and REJECTED (dilute base signal).
    df["V12_amount"] = df["V12"] * df["log_amount"]   # r = -0.21 with fraud
    df["V12_hour"]   = df["V12"] * df["Hour"]          # r = -0.21 with fraud
    df["V7_amount"]  = df["V7"]  * df["log_amount"]    # r = -0.09, independent
    df["V11_hour"]   = df["V11"] * df["Hour"]           # r = +0.10, independent

    return df, stats


# ──────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Build ColumnTransformer.

    DESIGN NOTE — No StandardScaler:
    XGBoost and LightGBM are scale-invariant (tree splits on rank order,
    not absolute values). Scaling the V-features, which are already PCA-
    standardised, would be a no-op at best and distorting at worst.
    The only transformation needed is OrdinalEncoder for amount_bin.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "ordinal_amount_bin",
                OrdinalEncoder(
                    categories=[AMOUNT_BIN_ORDER],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ["amount_bin"],
            )
        ],
        remainder="passthrough",   # all other features pass through unchanged
        verbose_feature_names_out=False,
    )
    return preprocessor


def fit_and_transform(X: pd.DataFrame,
                      preprocessor: ColumnTransformer
                      ) -> tuple[ColumnTransformer, pd.DataFrame]:
    """Fit preprocessor on X and return both fitted preprocessor and transformed X."""
    preprocessor.fit(X)
    X_out = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()
    return preprocessor, pd.DataFrame(X_out, columns=feature_names, index=X.index)


def apply_transform(X: pd.DataFrame,
                    preprocessor: ColumnTransformer) -> pd.DataFrame:
    """Apply a fitted preprocessor to X without refitting."""
    X_out = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()
    return pd.DataFrame(X_out, columns=feature_names, index=X.index)


# ──────────────────────────────────────────────────────────────────────────────
# RESAMPLING  (optional SMOTE fallback — EDA Section 14.3)
# ──────────────────────────────────────────────────────────────────────────────

def apply_smote(X: pd.DataFrame, y: pd.Series,
                sampling_strategy: float = 0.1,
                k_neighbors: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE oversampling to the TRAINING fold only.

    EDA recommendation: use scale_pos_weight = 533 first.
    Apply SMOTE only as a fallback when PR-AUC < 0.70 without it.

    NEVER call this function on validation or test data.

    sampling_strategy = 0.1 means fraud will be 10% of the resampled
    training set — a 9:1 ratio. This is less aggressive than full balancing
    and avoids overfitting to synthetic samples while giving the model
    enough minority class examples to learn patterns.

    Args:
        X                 : Training feature matrix (after preprocessing)
        y                 : Training labels
        sampling_strategy : Target ratio of minority/majority after resampling
        k_neighbors       : Number of nearest neighbours for SMOTE interpolation

    Returns:
        X_res, y_res      : Resampled feature matrix and labels
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError("imbalanced-learn required: pip install imbalanced-learn")

    print(f"  Before SMOTE: {Counter(y)}")
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=42,
        n_jobs=-1,
    )
    X_res, y_res = smote.fit_resample(X, y)
    print(f"  After  SMOTE: {Counter(y_res)}")
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)


# ──────────────────────────────────────────────────────────────────────────────
# SAVE / LOAD HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def save_features(df: pd.DataFrame, output_path: str,
                  metadata_path: str | None = None) -> None:
    """Save engineered feature dataframe to parquet with optional metadata."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  Saved features  → {output_path}  {df.shape}")

    if metadata_path:
        os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)
        meta = {
            "columns": df.columns.tolist(),
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "has_target": "Class" in df.columns,
            "class_distribution": (
                df["Class"].value_counts().to_dict() if "Class" in df.columns else None
            ),
        }
        with open(metadata_path, "w") as f:
            json.dump(meta, f, indent=2, default=int)
        print(f"  Saved metadata  → {metadata_path}")


def save_artifacts(preprocessor, train_stats: dict,
                   feature_names: list, artifacts_path: str) -> None:
    """Save all training-fit artifacts needed to process val/test data."""
    os.makedirs(os.path.dirname(artifacts_path) or ".", exist_ok=True)
    bundle = {
        "preprocessor":   preprocessor,
        "train_stats":    train_stats,
        "feature_names":  feature_names,
        "amount_bin_order": AMOUNT_BIN_ORDER,
    }
    joblib.dump(bundle, artifacts_path)
    print(f"  Saved artifacts → {artifacts_path}")


def load_artifacts(artifacts_path: str) -> dict:
    """Load training-fit artifacts for val/test processing."""
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(
            f"Artifacts not found: {artifacts_path}\n"
            f"Run 'python feature_engineering.py --mode train' first."
        )
    return joblib.load(artifacts_path)


# ──────────────────────────────────────────────────────────────────────────────
# PROCESSING MODES
# ──────────────────────────────────────────────────────────────────────────────

def process_train(args) -> None:
    """Fit all transformers on training data and save artifacts."""
    print("\n=== Feature Engineering — TRAIN mode ===")

    # Load raw CSV
    df = pd.read_csv(args.input_data)
    print(f"Raw data: {df.shape}   Class dist: {Counter(df['Class'])}")

    # Feature engineering (fit bin edges)
    print("\nApplying feature engineering...")
    df_eng, train_stats = apply_feature_engineering(df, train_stats=None)

    # Select production feature columns + target
    missing = [f for f in PRODUCTION_FEATURES if f not in df_eng.columns]
    if missing:
        raise ValueError(f"Missing production features after engineering: {missing}")

    X = df_eng[PRODUCTION_FEATURES]
    y = df_eng["Class"]

    print(f"\nFeature matrix: {X.shape}")
    print(f"  V-features      : {len(V_FEATURES)}")
    print(f"  Engineered feats: {len(PRODUCTION_FEATURES) - len(V_FEATURES)}")

    # Fit preprocessor (OrdinalEncoder for amount_bin only)
    print("\nFitting preprocessor (OrdinalEncoder for amount_bin)...")
    preprocessor = build_preprocessor()
    preprocessor, X_proc = fit_and_transform(X, preprocessor)
    print(f"  Processed shape : {X_proc.shape}")

    # Optional SMOTE (fallback strategy — use only if PR-AUC insufficient)
    if args.apply_smote:
        print("\nApplying SMOTE (fallback — only use if scale_pos_weight alone is insufficient)...")
        X_proc, y = apply_smote(X_proc, y, sampling_strategy=0.1)

    # Assemble output dataframe
    out_df = X_proc.copy()
    out_df["Class"] = y.values

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path  = os.path.join(args.output_dir, "train_features.parquet")
    meta_path = os.path.join(args.output_dir, "train_features_metadata.json")
    save_features(out_df, out_path, meta_path)

    art_path = args.artifacts_path
    save_artifacts(preprocessor, train_stats, X_proc.columns.tolist(), art_path)

    print(f"\n✓ Training features ready.")
    print(f"  scale_pos_weight for model = "
          f"{int((y == 0).sum())} / {int((y == 1).sum())} "
          f"= {(y == 0).sum() / (y == 1).sum():.1f}")


def process_inference(args, mode: str) -> None:
    """Apply training-fit transformers to val or test data (no refitting)."""
    print(f"\n=== Feature Engineering — {mode.upper()} mode ===")

    # Load training artifacts
    artifacts  = load_artifacts(args.artifacts_path)
    preprocessor = artifacts["preprocessor"]
    train_stats  = artifacts["train_stats"]
    print(f"  Loaded artifacts from {args.artifacts_path}")

    # Load raw CSV
    df = pd.read_csv(args.input_data)
    print(f"  Raw data: {df.shape}")
    if "Class" in df.columns:
        print(f"  Class dist: {Counter(df['Class'])}")

    # Feature engineering (apply pre-fitted bin edges — NO refit)
    df_eng, _ = apply_feature_engineering(df, train_stats=train_stats)

    X = df_eng[PRODUCTION_FEATURES]
    y = df_eng["Class"] if "Class" in df_eng.columns else None

    # Apply preprocessing (NO refit)
    X_proc = apply_transform(X, preprocessor)

    # Assemble output
    out_df = X_proc.copy()
    if y is not None:
        out_df["Class"] = y.values

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path  = os.path.join(args.output_dir, f"{mode}_features.parquet")
    meta_path = os.path.join(args.output_dir, f"{mode}_features_metadata.json")
    save_features(out_df, out_path, meta_path)

    print(f"\n✓ {mode.capitalize()} features ready.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Feature Engineering Pipeline — Credit Card Fraud Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Process training data (fits transformers, saves artifacts)
  python feature_engineering.py --mode train \\
      --input-data data/train.csv --output-dir data/engineered

  # 2. Process validation data (uses training artifacts)
  python feature_engineering.py --mode val \\
      --input-data data/val.csv --output-dir data/engineered

  # 3. Process test data (uses training artifacts)
  python feature_engineering.py --mode test \\
      --input-data data/test.csv --output-dir data/engineered

  # 4. Train with SMOTE fallback (only if PR-AUC < 0.70 without it)
  python feature_engineering.py --mode train \\
      --input-data data/train.csv --output-dir data/engineered \\
      --apply-smote
        """
    )
    parser.add_argument("--mode", choices=["train", "val", "test"],
                        default="train", help="Processing mode")
    parser.add_argument("--input-data", type=str, default=None,
                        help="Path to raw CSV (default: data/{mode}.csv)")
    parser.add_argument("--output-dir", type=str, default="data/engineered",
                        help="Output directory for parquet files")
    parser.add_argument("--artifacts-path", type=str,
                        default="data/engineered/feature_artifacts.pkl",
                        help="Path to save/load feature engineering artifacts")
    parser.add_argument("--apply-smote", action="store_true",
                        help="Apply SMOTE oversampling (fallback — train mode only)")

    args = parser.parse_args()

    if args.input_data is None:
        args.input_data = f"data/{args.mode}.csv"

    if not os.path.exists(args.input_data):
        raise FileNotFoundError(f"Input file not found: {args.input_data}")

    if args.mode == "train":
        process_train(args)
    else:
        process_inference(args, args.mode)


if __name__ == "__main__":
    main()