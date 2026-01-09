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


def _file_nonempty(path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _load_apply_log_target(metadata_path) -> bool:
    if not metadata_path.exists():
        # If metadata is missing, assume True as per your project decision
        return True
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    return bool(meta.get("apply_log_target", True))


def load_artifacts() -> InferenceArtifacts:
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

    return InferenceArtifacts(model=model, preprocessor=preprocessor, apply_log_target=apply_log_target)


def _validate_and_build_df(payload: Union[Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
    if isinstance(payload, dict):
        rows = [payload]
    elif isinstance(payload, list) and all(isinstance(x, dict) for x in payload):
        rows = payload
    else:
        raise TypeError("payload must be a dict or a list of dicts")

    df = pd.DataFrame(rows)

    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}. Expected: {REQUIRED_INPUT_COLUMNS}")

    # Keep only expected columns (protects against extra keys)
    df = df[REQUIRED_INPUT_COLUMNS].copy()

    # Basic type normalization (keeps inference robust)
    df["age"] = pd.to_numeric(df["age"], errors="raise")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="raise")
    df["children"] = pd.to_numeric(df["children"], errors="raise")

    # Normalize categorical strings (optional but helps consistency)
    for c in ["sex", "smoker", "region"]:
        df[c] = df[c].astype(str).str.strip().str.lower()

    return df


def predict(payload: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Returns:
      - If single dict input: {"prediction": float}
      - If list input: {"predictions": [float, ...]}
    """
    artifacts = load_artifacts()
    df = _validate_and_build_df(payload)

    X = artifacts.preprocessor.transform(df)
    y_pred = artifacts.model.predict(X)

    # Ensure 1D numpy
    y_pred = np.asarray(y_pred).reshape(-1)

    # Inverse target transform if training used log1p
    if artifacts.apply_log_target:
        y_pred = np.expm1(y_pred)

    if isinstance(payload, dict):
        return {"prediction": float(y_pred[0])}
    return {"predictions": [float(v) for v in y_pred]}


if __name__ == "__main__":
    # Quick manual test
    sample = {
        "age": 28,
        "sex": "male",
        "bmi": 31.2,
        "children": 2,
        "smoker": "yes",
        "region": "northwest",
    }
    print(predict(sample))
