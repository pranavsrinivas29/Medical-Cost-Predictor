from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    age: int = Field(..., ge=0)
    sex: str
    bmi: float = Field(..., gt=0)
    children: int = Field(..., ge=0)
    smoker: str
    region: str


class PredictResponse(BaseModel):
    prediction: float
