from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .features import MODEL_FEATURES, prepare_features
from .labels import build_pseudo_labels
from .train import build_preprocessor


def safe_metric(fn, y_true, y_pred_or_score):
    try:
        return float(fn(y_true, y_pred_or_score))
    except Exception:
        return None


def evaluate(input_csv: Path) -> dict:
    raw = pd.read_csv(input_csv)
    features = prepare_features(raw)
    y = build_pseudo_labels(features)
    X = features[MODEL_FEATURES]

    # Prefer time-based validation if multiple planning years exist.
    if "planning_year" in raw.columns and raw["planning_year"].nunique() >= 2:
        max_year = raw["planning_year"].max()
        train_idx = raw["planning_year"] < max_year
        test_idx = raw["planning_year"] == max_year
        split_strategy = f"time_based_train_before_{max_year}_test_{max_year}"
    else:
        from sklearn.model_selection import train_test_split

        train_idx, test_idx = train_test_split(
            np.arange(len(raw)),
            test_size=0.25,
            random_state=42,
            stratify=y if y.nunique() == 2 else None,
        )
        split_strategy = "random_stratified_split"

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]
    y_train = y.loc[train_idx]
    y_test = y.loc[test_idx]

    models = {
        "logistic_regression": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=100,
                        learning_rate=0.05,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    results = {
        "splitStrategy": split_strategy,
        "rowsTotal": int(len(raw)),
        "rowsTrain": int(len(X_train)),
        "rowsTest": int(len(X_test)),
        "positiveRateTrain": float(y_train.mean()),
        "positiveRateTest": float(y_test.mean()),
        "models": {},
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.predict(X_test)

        # Threshold can be tuned later. 0.5 is OK for first evaluation.
        y_pred = (y_score >= 0.5).astype(int)

        results["models"][name] = {
            "accuracy": safe_metric(accuracy_score, y_test, y_pred),
            "precision": safe_metric(lambda a, b: precision_score(a, b, zero_division=0), y_test, y_pred),
            "recall": safe_metric(lambda a, b: recall_score(a, b, zero_division=0), y_test, y_pred),
            "f1": safe_metric(lambda a, b: f1_score(a, b, zero_division=0), y_test, y_pred),
            "rocAuc": safe_metric(roc_auc_score, y_test, y_score),
            "averagePrecision": safe_metric(average_precision_score, y_test, y_score),
            "confusionMatrix": confusion_matrix(y_test, y_pred).tolist(),
            "classificationReport": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/staging_plan_features.csv")
    parser.add_argument("--output", default="outputs/evaluation_report.json")
    args = parser.parse_args()

    report = evaluate(Path(args.input))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
