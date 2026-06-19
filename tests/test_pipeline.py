from pathlib import Path

import pandas as pd

from jobplan_risk.score import score
from jobplan_risk.train import train


def test_training_and_scoring(tmp_path):
    input_csv = Path("data/sample_plan_features.csv")
    model_dir = tmp_path / "models"

    metadata = train(input_csv, model_dir)

    assert metadata["training_rows"] > 0
    assert (model_dir / "risk_model.joblib").exists()

    df = pd.read_csv(input_csv).head(3)
    results = score(df, model_dir)

    assert len(results) == 3
    assert "riskScore" in results[0]
    assert results[0]["riskCategory"] in {"Low", "Medium", "High"}
    assert "mainDrivers" in results[0]
