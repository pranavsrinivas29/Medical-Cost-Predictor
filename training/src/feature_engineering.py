"""
training/src/feature_engineering.py

Purpose:
- Perform feature engineering / preprocessing for the insurance dataset
  (one-hot encoding for categoricals + scaling for numerics)
- Apply optional log1p transform to target (charges)
- Save all preprocessing artifacts needed for training experiments AND inference:
    - fitted preprocessor (encoder + scaler)
    - feature names (final column order)
    - transformed train/test matrices (optional, as .npz)
    - transformed targets (optional)
    - metadata (json)

IMPORTANT:
- This script intentionally DOES NOT train a model yet.
  It prepares everything so you can plug in model training/HPO next.

Inputs (expected from preprocess stage):
  data/processed/X_train.csv
  data/processed/X_test.csv
  data/processed/y_train.csv
  data/processed/y_test.csv

Outputs (default):
  models/feature_engineering/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class FEConfig:
    processed_dir: Path
    out_dir: Path
    numeric_features: List[str]
    categorical_features: List[str]
    scale_numeric: bool
    onehot_drop: Optional[str]  # "first" or None
    apply_log_target: bool


def load_processed_splits(processed_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x_train_path = processed_dir / "X_train.csv"
    x_test_path = processed_dir / "X_test.csv"
    y_train_path = processed_dir / "y_train.csv"
    y_test_path = processed_dir / "y_test.csv"

    for p in [x_train_path, x_test_path, y_train_path, y_test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required processed split: {p}")

    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)

    # y files were written as 1-column CSV with header
    y_train = pd.read_csv(y_train_path).iloc[:, 0]
    y_test = pd.read_csv(y_test_path).iloc[:, 0]

    return X_train, X_test, y_train, y_test


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    scale_numeric: bool = True,
    onehot_drop: Optional[str] = "first",
) -> ColumnTransformer:
    if scale_numeric:
        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    else:
        # passthrough numerics unchanged
        numeric_transformer = "passthrough"

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    drop=onehot_drop,            # "first" reduces multicollinearity; set None to keep all
                    sparse_output=False,          # easier for debugging; can be True for large data
                    handle_unknown="ignore",
                ),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Returns final feature names after ColumnTransformer fit.
    """
    if not hasattr(preprocessor, "get_feature_names_out"):
        raise RuntimeError("Preprocessor is not fitted or does not support get_feature_names_out().")

    names = preprocessor.get_feature_names_out()
    return [str(n) for n in names]


def transform_target(y: pd.Series, apply_log: bool) -> np.ndarray:
    y_arr = y.to_numpy(dtype=float)
    if apply_log:
        return np.log1p(y_arr)
    return y_arr


def save_artifacts(
    out_dir: Path,
    preprocessor: ColumnTransformer,
    feature_names: List[str],
    X_train_t: np.ndarray,
    X_test_t: np.ndarray,
    y_train_t: np.ndarray,
    y_test_t: np.ndarray,
    cfg: FEConfig,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Save fitted preprocessor
    joblib.dump(preprocessor, out_dir / "preprocessor.joblib")

    # 2) Save feature names / column order
    (out_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")

    # 3) Save transformed arrays (optional but convenient for experiments)
    np.savez_compressed(
        out_dir / "train_test_transformed.npz",
        X_train=X_train_t,
        X_test=X_test_t,
        y_train=y_train_t,
        y_test=y_test_t,
    )

    # 4) Save metadata
    meta = {
        "processed_dir": str(cfg.processed_dir),
        "numeric_features": cfg.numeric_features,
        "categorical_features": cfg.categorical_features,
        "scale_numeric": cfg.scale_numeric,
        "onehot_drop": cfg.onehot_drop,
        "apply_log_target": cfg.apply_log_target,
        "n_train": int(X_train_t.shape[0]),
        "n_test": int(X_test_t.shape[0]),
        "n_features_after_transform": int(X_train_t.shape[1]),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature engineering prep.")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Directory with X/y splits.")
    parser.add_argument("--out-dir", type=str, default="data/feature_engineering", help="Where to save artifacts.")
    parser.add_argument("--scale-numeric", action="store_true", help="Scale numeric features with StandardScaler.")
    parser.add_argument("--no-scale-numeric", dest="scale_numeric", action="store_false", help="Do not scale numerics.")
    parser.set_defaults(scale_numeric=True)

    parser.add_argument(
        "--onehot-drop",
        type=str,
        default="first",
        choices=["first", "none"],
        help="Drop strategy for one-hot encoding: 'first' or 'none'.",
    )
    parser.add_argument(
        "--log-target",
        action="store_true",
        help="Apply log1p transform to target charges (recommended).",
    )

    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    onehot_drop = None if args.onehot_drop == "none" else args.onehot_drop
    apply_log_target = bool(args.log_target)

    # Define columns for this dataset
    numeric_features = ["age", "bmi", "children"]
    categorical_features = ["sex", "smoker", "region"]

    cfg = FEConfig(
        processed_dir=processed_dir,
        out_dir=out_dir,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=bool(args.scale_numeric),
        onehot_drop=onehot_drop,
        apply_log_target=apply_log_target,
    )

    # Load processed splits (created by preprocess.py)
    X_train, X_test, y_train, y_test = load_processed_splits(processed_dir)

    # Basic schema checks
    for col in numeric_features + categorical_features:
        if col not in X_train.columns:
            raise ValueError(f"Expected column '{col}' not found in X_train columns: {list(X_train.columns)}")

    # Build & fit preprocessor ONLY on train
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=cfg.scale_numeric,
        onehot_drop=cfg.onehot_drop,
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # Feature names after encoding (for debugging + stable inference ordering)
    feature_names = get_feature_names(preprocessor)

    # Transform target if requested
    y_train_t = transform_target(y_train, apply_log=cfg.apply_log_target)
    y_test_t = transform_target(y_test, apply_log=cfg.apply_log_target)

    # Save artifacts for later model training / HPO and inference bundling
    save_artifacts(
        out_dir=cfg.out_dir,
        preprocessor=preprocessor,
        feature_names=feature_names,
        X_train_t=np.asarray(X_train_t),
        X_test_t=np.asarray(X_test_t),
        y_train_t=np.asarray(y_train_t),
        y_test_t=np.asarray(y_test_t),
        cfg=cfg,
    )

    print("✅ Feature engineering artifacts prepared (no model training).")
    print(f"Processed splits: {processed_dir}")
    print(f"Artifacts saved:  {out_dir}")
    print(f"Train shape:      {X_train_t.shape} | Test shape: {X_test_t.shape}")
    print(f"Target transform: {'log1p' if cfg.apply_log_target else 'none'}")
    print(f"OneHot drop:      {cfg.onehot_drop if cfg.onehot_drop is not None else 'none'}")


if __name__ == "__main__":
    main()
