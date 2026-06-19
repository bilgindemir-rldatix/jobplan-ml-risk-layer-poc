from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .explain import risk_category, rule_risk_component, top_drivers
from .features import MODEL_FEATURES, data_confidence, prepare_features


def clip(value: float, lo: float = 0, hi: float = 100) -> float:
    return float(max(lo, min(hi, value)))


def model_score(model, metadata: dict, X: pd.DataFrame) -> np.ndarray:
    mode = metadata.get("model_mode", "")

    if mode == "isolation_forest_fallback":
        raw = -model.decision_function(X)
        lo = metadata.get("score_min", float(np.nanmin(raw)))
        hi = metadata.get("score_max", float(np.nanmax(raw)))
        denom = max(1e-9, hi - lo)
        return np.clip((raw - lo) / denom * 100, 0, 100)

    if hasattr(model, "predict_proba"):
        return np.clip(model.predict_proba(X)[:, 1] * 100, 0, 100)

    return np.zeros(len(X))


def score(raw: pd.DataFrame, model_dir: Path = Path("models")) -> list[dict[str, Any]]:
    model = joblib.load(model_dir / "risk_model.joblib")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))

    features = prepare_features(raw)
    X = features[MODEL_FEATURES]

    model_scores = model_score(model, metadata, X)

    outputs = []

    for position, (_, row) in enumerate(features.iterrows()):
        rule_component = rule_risk_component(row)
        model_component = float(model_scores[position])

        final_score = clip(0.55 * model_component + 0.45 * rule_component)

        confidence, missing = data_confidence(row)

        outputs.append(
            {
                "jobPlanCode": str(row.get("job_plan_code", "")),
                "riskScore": round(final_score, 1),
                "riskCategory": risk_category(final_score),
                "mainDrivers": top_drivers(row),
                "dataConfidence": confidence,
                "missingData": missing,
                "modelMode": metadata.get("model_mode"),
                "modelRiskComponent": round(model_component, 1),
                "ruleRiskComponent": round(rule_component, 1),
                "riskInterpretation": "Operational prioritisation risk; not a definitive compliance or contract decision.",
            }
        )

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample_plan_features.csv")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(json.dumps(score(df, Path(args.model_dir)), indent=2))


if __name__ == "__main__":
    main()
