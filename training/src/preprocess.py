"""
training/src/preprocess.py

Preprocessing for the insurance charges dataset.

Responsibilities:
- Read raw CSV from data/raw/
- Validate required columns
- Basic cleaning (types, missing values)
- Deterministic train/test split
- Write processed outputs to data/processed/ as CSVs:
    X_train.csv, X_test.csv, y_train.csv, y_test.csv

IMPORTANT:
- This stage does NOT fit encoders/scalers (recommended to do in train.py for MLflow bundling).
- No model training happens here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS: List[str] = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]

DEFAULT_RAW_PATH = Path("data/raw/insurance.csv")
DEFAULT_OUT_DIR = Path("data/processed")


def _validate_input(df: pd.DataFrame, required_cols: List[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")


def _basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning:
    - strip whitespace from column names/strings
    - coerce numeric columns
    - normalize categorical strings to lowercase
    - drop rows with missing required values
    """
    df = df.copy()

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Strip whitespace from string cells
    for col in ["sex", "smoker", "region"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # Coerce numeric columns
    numeric_cols = ["age", "bmi", "children", "charges"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing required values after coercion
    df = df.dropna(subset=REQUIRED_COLUMNS)

    # Ensure integer-like columns are ints where appropriate
    df["age"] = df["age"].astype(int)
    df["children"] = df["children"].astype(int)

    return df


def _split_features_target(df: pd.DataFrame, target_col: str = "charges") -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def _ensure_outdir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _write_outputs(
    out_dir: Path,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    _ensure_outdir(out_dir)
    X_train.to_csv(out_dir / "X_train.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False, header=True)
    y_test.to_csv(out_dir / "y_test.csv", index=False, header=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess insurance dataset into train/test splits.")
    parser.add_argument("--raw-path", type=str, default=str(DEFAULT_RAW_PATH), help="Path to raw CSV.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory for processed CSVs.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for split.")
    args = parser.parse_args()

    raw_path = Path(args.raw_path)
    out_dir = Path(args.out_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw CSV not found at: {raw_path}")

    df = pd.read_csv(raw_path)
    _validate_input(df, REQUIRED_COLUMNS)

    df = _basic_cleaning(df)

    # Optional sanity checks (can be expanded later)
    if df.empty:
        raise ValueError("Dataset is empty after cleaning. Check raw data and cleaning rules.")
    if (df["charges"] < 0).any():
        raise ValueError("Found negative charges after cleaning. Check raw data.")

    X, y = _split_features_target(df, target_col="charges")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=True,
    )

    _write_outputs(out_dir, X_train, X_test, y_train, y_test)

    print("✅ Preprocessing complete.")
    print(f"Raw:        {raw_path}  (rows: {len(df)})")
    print(f"Processed:  {out_dir}")
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")


if __name__ == "__main__":
    main()
