from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict

from fastapi import FastAPI, HTTPException

# Ensure project root import works when running uvicorn
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas import PredictRequest, PredictResponse
from inference.inference import load_artifacts, predict_one


app = FastAPI(title="Insurance Charges Predictor", version="1.0.0")


@app.on_event("startup")
def _startup() -> None:
    # Load once on startup (fast inference)
    try:
        load_artifacts(force_reload=False)
    except Exception as e:
        # Keep server up; health endpoint will show degraded status
        print(f"[startup] Artifact load failed: {e}")


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        a = load_artifacts(force_reload=False)
        return {"status": "ok", "artifacts_loaded": True, "apply_log_target": a.apply_log_target}
    except Exception as e:
        return {"status": "degraded", "artifacts_loaded": False, "error": str(e)}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: PredictRequest) -> PredictResponse:
    try:
        y = predict_one(payload.model_dump())
        return PredictResponse(prediction=float(y))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
