from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    prepare_features,
)
from .labels import build_pseudo_labels


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            ("bin", SimpleImputer(strategy="most_frequent"), BINARY_FEATURES),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def safe_metric(metric_fn, y_true, y_score) -> float:
    try:
        return float(metric_fn(y_true, y_score))
    except Exception:
        return float("nan")


def train(input_csv: Path, output_dir: Path) -> dict:
    raw = pd.read_csv(input_csv)
    features = prepare_features(raw)
    y = build_pseudo_labels(features)

    X = features[MODEL_FEATURES]

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "training_rows": int(len(features)),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
        "feature_columns": MODEL_FEATURES,
        "label_type": "pseudo_atRisk",
        "warning": "POC model: pseudo-labels are operational risk proxies, not true clinical/contract outcomes.",
    }

    enough_supervised = len(features) >= 20 and y.nunique() == 2 and y.value_counts().min() >= 3

    if enough_supervised:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )

        candidates = []

        logistic = Pipeline(
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
        )

        logistic.fit(X_train, y_train)
        logistic_prob = logistic.predict_proba(X_test)[:, 1]
        candidates.append(("logistic_regression", logistic, logistic_prob))

        hgb = Pipeline(
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
        )

        hgb.fit(X_train, y_train)
        hgb_prob = hgb.predict_proba(X_test)[:, 1]
        candidates.append(("hist_gradient_boosting", hgb, hgb_prob))

        scored = []

        for name, model, probability in candidates:
            roc = safe_metric(roc_auc_score, y_test, probability)
            pr = safe_metric(average_precision_score, y_test, probability)
            scored.append((name, model, roc, pr))

        best = sorted(
            scored,
            key=lambda x: (np.nan_to_num(x[3]), np.nan_to_num(x[2])),
            reverse=True,
        )[0]

        model_mode, model, roc_auc, average_precision = best

        metadata.update(
            {
                "model_mode": model_mode,
                "roc_auc": roc_auc,
                "average_precision": average_precision,
            }
        )

    else:
        model = Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    IsolationForest(
                        n_estimators=250,
                        contamination=0.15,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        model.fit(X)

        raw_scores = -model.decision_function(X)

        metadata.update(
            {
                "model_mode": "isolation_forest_fallback",
                "score_min": float(np.nanmin(raw_scores)),
                "score_max": float(np.nanmax(raw_scores)),
            }
        )

    joblib.dump(model, output_dir / "risk_model.joblib")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample_plan_features.csv")
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    metadata = train(Path(args.input), Path(args.output_dir))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
