# JobPlan ML Risk Layer POC

This repository contains the **JobPlan ML Risk Layer POC** for the AI JobPlan Analyser demo.

The purpose of this POC is to show how JobPlans can be scored, prioritised, and explained using a lightweight ML layer. The model produces a **risk score from 0 to 100**, assigns a risk category, and identifies the main operational risk drivers behind each JobPlan.

The current version is designed for demo and stakeholder review. It includes the dashboard, API, trained model, metadata, evaluation report, sample dataset, and generated risk score outputs.

---

## Important POC Note

This is a **proof of concept**, not a production model.

The current dataset is **synthetic / staging-shaped** and the labels are **pseudo-labelled** using operational risk rules. The model validates the ML pipeline, API, dashboard, and integration shape.

The next step is to replace the synthetic dataset with a real staging extract and validate the highest-risk plans with domain experts.

---

## What This POC Provides

The POC supports:

* JobPlan risk scoring from 0 to 100
* Risk categories: Low, Medium, High
* Highest-risk JobPlan review queue
* Main risk drivers for each JobPlan
* Evaluation report for the trained model
* Dashboard for product and stakeholder review
* FastAPI service for integration
* Exportable CSV and JSON outputs
* Pre-trained model artifact included for demo use

---

## Current Dashboard Views

The Streamlit dashboard includes:

1. **Risk Scores**
   Summary of scored JobPlans, risk distribution, hotspots, and review signals.

2. **Highest-Risk JobPlans**
   Shows all High Risk JobPlans sorted from highest to lowest risk score.
   This is the main stakeholder-facing review queue.

3. **Evaluation**
   Displays model evaluation results such as accuracy, precision, recall, F1, ROC-AUC, average precision, and confusion matrix.

4. **Training Details**
   Shows model metadata, feature list, training rows, model mode, and pseudo-label warning.

5. **Dataset**
   Shows the staging-shaped JobPlan feature dataset used by the POC.

6. **Export / Integration**
   Provides structured outputs that can be used by other services, rule engines, or LLM explanation layers.

---

## Project Structure

```text
jobplan-ml-risk-layer-poc/
├── dashboard/
│   └── app.py
├── data/
│   └── staging_plan_features.csv
├── models/
│   ├── metadata.json
│   └── risk_model.joblib
├── outputs/
│   ├── evaluation_report.json
│   ├── risk_scores.csv
│   ├── risk_scores.json
│   ├── high_risk_jobplans.csv
│   └── high_risk_jobplans.json
├── scripts/
│   ├── start_api.sh
│   └── start_dashboard.sh
├── src/
│   ├── analyser_ml/
│   │   └── api.py
│   └── jobplan_risk/
│       ├── api.py
│       ├── evaluate.py
│       ├── explain.py
│       ├── features.py
│       ├── labels.py
│       ├── score.py
│       └── train.py
├── tests/
├── requirements.txt
└── README.md
```

---

## Dataset

The POC uses a staging-shaped dataset:

```text
data/staging_plan_features.csv
```

Each row represents one JobPlan or JobPlan version.

Example feature groups:

* PA values: total PAs, DCC PAs, SPA PAs, CP PAs, other PAs
* Workflow: plan stage, days in current stage, history changes
* Contract context: annualised, new contract flag, 5th week usage
* Trend signals: SPA delta, DCC delta, total PA delta
* Peer comparison: peer median SPA/DCC share, z-scores
* Team alignment: team-plan link, alignment score, capacity gap
* Risk flags: PA limit breach, mediation/appeal, returned to discussion
* Clinician context: consultant flag, locum flag, grade code

---

## Model Approach

The current model is a lightweight scikit-learn model.

Current model mode:

```text
HistGradientBoostingClassifier
```

The model predicts a pseudo-labelled operational risk target called:

```text
pseudo_atRisk
```

The final score combines:

```text
ML model risk component + rule-based risk component
```

The output includes:

* `riskScore`
* `riskCategory`
* `mainDrivers`
* `modelRiskComponent`
* `ruleRiskComponent`
* `dataConfidence`
* `riskInterpretation`

---

## Main Risk Drivers

Main risk drivers are currently identified using deterministic explanation logic.

The ML model predicts the risk component, while the explanation layer ranks the strongest operational risk signals for each JobPlan.

Example drivers include:

* PA limit breach
* SPA increase compared with prior plan
* DCC reduction compared with prior plan
* SPA above peer threshold
* Low team-plan alignment score
* Team capacity gap
* High history change count
* Mediation or appeal signal
* Plan returned to discussion after sign-off
* Missing TeamPlan link
* Locum flag

This makes the output easier to explain to product and operational stakeholders.

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Dashboard

```bash
./scripts/start_dashboard.sh
```

Or manually:

```bash
export PYTHONPATH=src

python -m streamlit run dashboard/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

Open port:

```text
8501
```

---

## Run the API

```bash
./scripts/start_api.sh
```

Or manually:

```bash
export PYTHONPATH=src

python -m uvicorn analyser_ml.api:app \
  --app-dir src \
  --reload \
  --host 0.0.0.0 \
  --port 8000
```

Open port:

```text
8000
```

---

## API Endpoints

Health check:

```http
GET /health
```

Score one JobPlan:

```http
GET /api/v1/jobplans/{jobPlanCode}/analysis
```

Department summary:

```http
GET /api/v1/departments/{trustLevelCode}/summary
```

Batch analysis:

```http
POST /api/v1/analysis/batch
```

Scenario simulation:

```http
POST /api/v1/scenarios/simulate
```

Legacy scoring endpoint:

```http
POST /score
```

---

## Example API Output Shape

```json
{
  "jobPlanCode": "JP-2026-001",
  "riskScore": 87.4,
  "riskCategory": "High",
  "mainDrivers": [
    "SPA increased compared with prior plan.",
    "Team-plan alignment score is low.",
    "High history change count detected."
  ],
  "dataConfidence": "High",
  "modelMode": "hist_gradient_boosting",
  "modelRiskComponent": 91.2,
  "ruleRiskComponent": 82.7,
  "riskInterpretation": "This JobPlan should be prioritised for review."
}
```

---

## Train the Model

The repository already includes a trained model for demo use.

To retrain:

```bash
export PYTHONPATH=src

python -m jobplan_risk.train \
  --input data/staging_plan_features.csv \
  --output-dir models
```

This creates:

```text
models/risk_model.joblib
models/metadata.json
```

---

## Evaluate the Model

```bash
export PYTHONPATH=src

python -m jobplan_risk.evaluate \
  --input data/staging_plan_features.csv \
  --output outputs/evaluation_report.json
```

This creates:

```text
outputs/evaluation_report.json
```

---

## Generate Risk Score Outputs

```bash
export PYTHONPATH=src

python - <<'PY'
import json
from pathlib import Path
import pandas as pd

from jobplan_risk.score import score

DATA_PATH = Path("data/staging_plan_features.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
scored = score(df, MODEL_DIR)
scored_df = pd.DataFrame(scored)

scored_df.to_csv(OUTPUT_DIR / "risk_scores.csv", index=False)
(OUTPUT_DIR / "risk_scores.json").write_text(
    json.dumps(scored, indent=2),
    encoding="utf-8"
)

high_risk_df = scored_df[scored_df["riskCategory"] == "High"].copy()
high_risk_df = high_risk_df.sort_values("riskScore", ascending=False)

high_risk_df.to_csv(OUTPUT_DIR / "high_risk_jobplans.csv", index=False)
(OUTPUT_DIR / "high_risk_jobplans.json").write_text(
    high_risk_df.to_json(orient="records", indent=2),
    encoding="utf-8"
)

print("Generated risk score outputs.")
PY
```

This creates:

```text
outputs/risk_scores.csv
outputs/risk_scores.json
outputs/high_risk_jobplans.csv
outputs/high_risk_jobplans.json
```

---

## Demo Outputs Included

The repo includes the following demo outputs:

```text
outputs/evaluation_report.json
outputs/risk_scores.csv
outputs/risk_scores.json
outputs/high_risk_jobplans.csv
outputs/high_risk_jobplans.json
```

This allows the dashboard and API to be used for demo purposes without retraining.

---

## Current Model Metadata

The model metadata is stored in:

```text
models/metadata.json
```

It includes:

* Training rows
* Positive rate
* Feature columns
* Label type
* Model mode
* ROC-AUC
* Average precision
* POC warning

---

## Demo Talk Track

A safe way to explain the POC:

```text
This is the JobPlan ML risk scoring layer. It scores each JobPlan from 0 to 100 and prioritises which plans may need review first.

The score is based on operational risk signals such as SPA/DCC drift, PA limit breaches, team-plan misalignment, workflow instability, and peer deviation.

The current dataset is synthetic and pseudo-labelled, so this validates the pipeline, dashboard, and integration shape. The next step is to replace the data with a real staging extract and validate the highest-risk plans with domain experts.
```

---

## What This POC Does Not Do Yet

This POC does not yet provide:

* Production-grade risk prediction
* Real customer/staging validation
* Full contractual compliance engine
* Final clinical or managerial decisioning
* LLM-generated final narrative
* Full what-if optimisation
* Production deployment pipeline

---

## Recommended Next Steps

1. Replace synthetic dataset with real staging extract.
2. Validate highest-risk JobPlans with domain experts.
3. Separate pure ML features from rule-derived pseudo-label features.
4. Add SHAP or model-level feature contribution explanations.
5. Integrate rule-engine findings with ML output.
6. Add LLM-generated executive explanation from structured evidence.
7. Add governance around model versioning, monitoring, and retraining.
8. Prepare production architecture if the POC is accepted.

---

## Production Caution

The current model should not be used for final decisions.

It is suitable for:

* Demo
* Stakeholder review
* Pipeline validation
* Integration testing
* Product discovery
* Review prioritisation concept

It is not yet suitable for:

* Production risk classification
* Contractual compliance decisions
* Clinical decision-making
* Automated approval or rejection of JobPlans

---

## Quick Start for Demo

```bash
pip install -r requirements.txt
./scripts/start_dashboard.sh
```

Open port:

```text
8501
```

For API demo:

```bash
./scripts/start_api.sh
```

Open port:

```text
8000
```

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```