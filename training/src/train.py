"""
training/src/train.py

Requirements satisfied:
1) Always stores model locally (joblib) + params locally (json)
2) Always logs model to MLflow
3) If local model joblib exists and is non-empty -> loads model + params and reuses them
   (skips HPO) unless --force-train is set.

Bayesian HPO style:
- Uses scikit-optimize BayesSearchCV (sklearn-compatible)

Inputs:
  data/feature_engineering/
    - train_test_transformed.npz
    - preprocessor.joblib
    - feature_names.json
    - metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb

from skopt import BayesSearchCV
from skopt.space import Real, Integer

import mlflow
import mlflow.xgboost
from mlflow.models import infer_signature


# -------------------------
# Helpers
# -------------------------

def _file_nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def load_fe_artifacts(fe_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    npz_path = fe_dir / "train_test_transformed.npz"
    meta_path = fe_dir / "metadata.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"Missing: {npz_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing: {meta_path}")

    data = np.load(npz_path)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return X_train, X_test, y_train, y_test, metadata


def inverse_target(y: np.ndarray) -> np.ndarray:
    # inverse of log1p
    return np.expm1(y)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def log_preprocessing_artifacts(fe_dir: Path) -> None:
    # Log artifacts needed for inference consistency
    for fname in ["preprocessor.joblib", "feature_names.json", "metadata.json"]:
        p = fe_dir / fname
        if p.exists():
            mlflow.log_artifact(str(p), artifact_path="preprocessing")


# -------------------------
# Training / HPO
# -------------------------

def run_bayes_hpo(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
    n_iter: int,
    cv_folds: int,
) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
    base_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )

    search_spaces = {
        "n_estimators": Integer(300, 3000),
        "learning_rate": Real(0.005, 0.2, prior="log-uniform"),
        "max_depth": Integer(2, 10),
        "min_child_weight": Real(1.0, 20.0),
        "subsample": Real(0.5, 1.0),
        "colsample_bytree": Real(0.5, 1.0),
        "reg_alpha": Real(1e-8, 10.0, prior="log-uniform"),
        "reg_lambda": Real(1e-8, 50.0, prior="log-uniform"),
        "gamma": Real(0.0, 10.0),
    }

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    opt = BayesSearchCV(
        estimator=base_model,
        search_spaces=search_spaces,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,          # refit best model on full X_train
        random_state=random_state,
        verbose=0,
    )

    opt.fit(X_train, y_train)
    best_model: xgb.XGBRegressor = opt.best_estimator_
    best_params: Dict[str, Any] = dict(opt.best_params_)
    return best_model, best_params


def save_local_model_and_params(model: xgb.XGBRegressor, params: Dict[str, Any], model_path: Path, params_path: Path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")


def load_local_model_and_params(model_path: Path, params_path: Path) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
    model = joblib.load(model_path)
    params = {}
    if params_path.exists():
        params = json.loads(params_path.read_text(encoding="utf-8"))
    return model, params


# -------------------------
# Main
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost with sklearn-style Bayesian HPO + MLflow + local saving.")
    parser.add_argument("--fe-dir", type=str, default="data/feature_engineering", help="Feature engineering artifacts dir.")
    parser.add_argument("--experiment-name", type=str, default="insurance-xgboost", help="MLflow experiment name.")
    parser.add_argument("--tracking-uri", type=str, default="", help="Optional MLflow tracking URI.")

    parser.add_argument("--n-iter", type=int, default=40, help="BayesSearchCV iterations (only used if training runs).")
    parser.add_argument("--cv-folds", type=int, default=5, help="CV folds for BayesSearchCV.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")

    # Always-save local model (as you requested)
    parser.add_argument("--model-out", type=str, default="models/xgboost/best_model.joblib", help="Local model path.")
    parser.add_argument("--params-out", type=str, default="models/xgboost/best_params.json", help="Local params path.")

    # Control behavior when local model already exists
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Force rerun Bayesian HPO and overwrite local model/params even if they already exist.",
    )

    args = parser.parse_args()

    fe_dir = Path(args.fe_dir)
    model_path = Path(args.model_out)
    params_path = Path(args.params_out)

    X_train, X_test, y_train, y_test, meta = load_fe_artifacts(fe_dir)

    # Your preference: target transform should be true in FE stage
    apply_log_target = bool(meta.get("apply_log_target", True))

    # MLflow setup
    if args.tracking_uri.strip():
        mlflow.set_tracking_uri(args.tracking_uri.strip())
    mlflow.set_experiment(args.experiment_name)

    # Decide whether to train or reuse local
    reuse_local = _file_nonempty(model_path) and not args.force_train

    if reuse_local:
        model, best_params = load_local_model_and_params(model_path, params_path)
        run_mode = "reused_local_model"
    else:
        model, best_params = run_bayes_hpo(
            X_train=X_train,
            y_train=y_train,
            random_state=args.random_state,
            n_iter=args.n_iter,
            cv_folds=args.cv_folds,
        )
        # Always save locally (required)
        save_local_model_and_params(model, best_params, model_path, params_path)
        run_mode = "trained_with_bayes_hpo"

    # Evaluate on train/test in target space
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics_train = compute_metrics(y_train, y_train_pred)
    metrics_test = compute_metrics(y_test, y_test_pred)

    # Evaluate also in original space if log target was used
    metrics_train_orig = {}
    metrics_test_orig = {}
    if apply_log_target:
        y_train_orig = inverse_target(y_train)
        y_test_orig = inverse_target(y_test)
        y_train_pred_orig = inverse_target(y_train_pred)
        y_test_pred_orig = inverse_target(y_test_pred)

        metrics_train_orig = {f"orig_{k}": v for k, v in compute_metrics(y_train_orig, y_train_pred_orig).items()}
        metrics_test_orig = {f"orig_{k}": v for k, v in compute_metrics(y_test_orig, y_test_pred_orig).items()}

    # Always log to MLflow (required)
    with mlflow.start_run(run_name=f"xgboost_{run_mode}") as run:
        # Params
        if best_params:
            mlflow.log_params(best_params)
        mlflow.log_param("run_mode", run_mode)
        mlflow.log_param("apply_log_target", apply_log_target)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("cv_folds", args.cv_folds)
        mlflow.log_param("bayes_n_iter", args.n_iter)

        # Metrics
        mlflow.log_metrics({f"train_{k}": v for k, v in metrics_train.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics_test.items()})
        if apply_log_target:
            mlflow.log_metrics({f"train_{k}": v for k, v in metrics_train_orig.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics_test_orig.items()})

        # Log preprocessing artifacts (for inference consistency)
        log_preprocessing_artifacts(fe_dir)

        # Log local model + params as artifacts too (nice to have)
        if model_path.exists():
            mlflow.log_artifact(str(model_path), artifact_path="local_export")
        if params_path.exists():
            mlflow.log_artifact(str(params_path), artifact_path="local_export")

        # Log model in MLflow model format
        signature = infer_signature(X_train[:50], y_train_pred[:50])
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            signature=signature,
        )

        print("✅ Done.")
        print(f"Mode: {run_mode}")
        print(f"Local model:  {model_path} (exists={model_path.exists()}, size={model_path.stat().st_size if model_path.exists() else 0})")
        print(f"Local params: {params_path} (exists={params_path.exists()})")
        print(f"MLflow run_id: {run.info.run_id}")
        print("Test metrics (target space):", metrics_test)
        if apply_log_target:
            print("Test metrics (original space):", metrics_test_orig)


if __name__ == "__main__":
    main()
