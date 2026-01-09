from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PREPROCESSOR_PATH,
    METADATA_PATH,
    MODEL_PATH,
    FEATURE_NAMES_PATH,  # not required for inference, but useful for debug
)


REQUIRED_INPUT_COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region"]


@dataclass(frozen=True)
class InferenceArtifacts:
    model: Any
    preprocessor: Any
    apply_log_target: bool


_ARTIFACTS: InferenceArtifacts | None = None


def _file_nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _load_apply_log_target(metadata_path: Path) -> bool:
    # Your project decision: default True if missing
    if not metadata_path.exists():
        return True
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    return bool(meta.get("apply_log_target", True))


def load_artifacts(force_reload: bool = False) -> InferenceArtifacts:
    """
    Load and cache model + preprocessor + metadata once.
    """
    global _ARTIFACTS
    if _ARTIFACTS is not None and not force_reload:
        return _ARTIFACTS

    if not _file_nonempty(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            f"Preprocessor not found or empty: {PREPROCESSOR_PATH}. "
            f"Run feature engineering first."
        )

    if not _file_nonempty(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found or empty: {MODEL_PATH}. "
            f"Run training first."
        )

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    apply_log_target = _load_apply_log_target(METADATA_PATH)

    _ARTIFACTS = InferenceArtifacts(model=model, preprocessor=preprocessor, apply_log_target=apply_log_target)
    return _ARTIFACTS


def _validate_and_build_df(payload: Dict[str, Any]) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")

    df = pd.DataFrame([payload])

    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}. Expected: {REQUIRED_INPUT_COLUMNS}")

    df = df[REQUIRED_INPUT_COLUMNS].copy()

    df["age"] = pd.to_numeric(df["age"], errors="raise")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="raise")
    df["children"] = pd.to_numeric(df["children"], errors="raise")

    for c in ["sex", "smoker", "region"]:
        df[c] = df[c].astype(str).str.strip().str.lower()

    return df


def predict_one(payload: Dict[str, Any]) -> float:
    """
    Single prediction (original scale). Uses cached artifacts.
    """
    artifacts = load_artifacts(force_reload=False)
    df = _validate_and_build_df(payload)

    X = artifacts.preprocessor.transform(df)
    y_pred = artifacts.model.predict(X)
    y_pred = float(np.asarray(y_pred).reshape(-1)[0])

    if artifacts.apply_log_target:
        y_pred = float(np.expm1(y_pred))

    return y_pred


if __name__ == "__main__":
    # quick local test
    sample = {
        "age": 19,
        "sex": "female",
        "bmi": 27.9,
        "children": 0,
        "smoker": "yes",
        "region": "southwest",
    }
    print({"prediction": predict_one(sample)})