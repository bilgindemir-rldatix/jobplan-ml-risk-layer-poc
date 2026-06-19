from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI

from .schemas import JobPlanFeatureInput
from .score import score

app = FastAPI(title="JobPlan ML Risk Scorer", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/score")
def score_jobplans(payload: list[JobPlanFeatureInput]) -> list[dict]:
    df = pd.DataFrame([p.model_dump() for p in payload])
    return score(df, Path("models"))
