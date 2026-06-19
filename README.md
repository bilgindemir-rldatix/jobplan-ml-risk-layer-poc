# AI JobPlan Risk Layer POC

This repository contains the Machine Learning risk-scoring layer for the **AI-Driven Job Plan Analyser** POC.

The goal of this POC is to score JobPlans, identify operational risk, and highlight which plans should be prioritised for review by Clinical Directors, Medical Staffing, Workforce Planning, Product, or Engineering stakeholders.

## What this POC provides

* JobPlan risk scoring
* High / Medium / Low risk categorisation
* Main risk drivers per JobPlan
* Trust / specialty hotspot analysis
* Model training and evaluation
* Stakeholder-facing Streamlit dashboard
* Structured CSV / JSON outputs
* API-ready scoring output for integration with Bedrock, n8n, rule engine, or future JobPlan analyser services

---

## Important POC Note

The current dataset used in this repository is **synthetic staging-shaped data**.

It is **not real customer data** and **not real production/staging data**.

The current model results validate:

* The ML pipeline
* The scoring flow
* The dashboard experience
* The integration output shape
* The ability to prioritise JobPlans using risk signals

The current results should **not** be presented as production accuracy.

Correct POC wording:

> The POC validates that the ML risk-scoring pipeline can prioritise JobPlans and explain likely risk drivers using synthetic and pseudo-labelled data.

Do not claim:

> The model is production accurate.

Production validation will require:

* Real staging or historical JobPlan data
* Real outcome labels
* Domain expert review
* Data protection review
* Model monitoring
* Governance and product approval

---

## POC Objective

The ML layer supports the AI Job Plan Analyser by producing a structured risk score for each JobPlan.

The output helps answer:

* Which JobPlans need immediate review?
* Which Trust areas or specialties have the highest concentration of risk?
* Which plans show SPA/DCC imbalance?
* Which plans may not align with demand and capacity?
* Which plans show workflow instability?
* Which plans have compliance or PA limit risk?
* What are the main explainable drivers behind each risk score?

---

## Dataset

Expected input file:

```text
data/staging_plan_features.csv
```

Dataset grain:

```text
1 row = 1 JobPlan
```

The dataset is a prepared ML feature table. It is **not raw WorkEpisode data**.

The expected dataset includes prepared and cleaned features from areas such as:

* JobPlan identity
* Trust / department / specialty context
* Planning year
* Workflow stage
* Total PA values
* DCC PA values
* SPA PA values
* CP / other PA values
* Prior-year PA values
* Peer benchmark values
* PA limit breach signal
* TeamJobPlan alignment signal
* Workflow instability signal
* Mediation / appeal signal
* Locum and grade context

The extract should avoid:

* Clinician names
* GMC numbers
* Free-text comments
* Patient data
* Sensitive identifiers
* Raw WorkEpisode rows
* LLM-generated labels

Hashed identifiers are enough for the POC.

---

## Expected Data Shape

The main dataset should contain columns similar to:

```text
job_plan_code
user_code_hash
trust_level_code
parent_trust_level_code
department
specialty
planning_year
plan_stage
currentStageOrdinal
daysInCurrentStage
pAsPerWeek
paidPA
minsPerWeek
ntPerPA
workType
doctorClassification
isSignedUpToNewContract
isAnnualised
uses5thWeekPerQuarter
hasTeamPlanLink
totalPAs
dccPAs
spaPAs
cpPAs
otherPAs
isOnCall
isShift
isCPD
priorTotalPAs
priorDccPAs
priorSpaPAs
yearsSinceLastPlan
peerMedianSpaShare
peerMedianDccShare
peerGroupSize
paLimitBreach
spaAbovePeerThreshold
missingTeamPlanLink
teamPlanAlignmentScore
teamDemandPAs
teamPlannedCapacityPAs
historyChangeCount
hasMediationOrAppeal
hasNewManagerChanges
planReturnedToDiscussionAfterSignoff
isLocum
gradeCode
isConsultant
```

---

## Project Structure

```text
.
├── dashboard/
│   └── app.py
├── data/
│   └── staging_plan_features.csv
├── models/
│   └── generated locally after training
├── outputs/
│   └── generated locally after scoring/dashboard
├── src/
│   └── jobplan_risk/
│       ├── __init__.py
│       ├── api.py
│       ├── explain.py
│       ├── features.py
│       ├── labels.py
│       ├── schemas.py
│       ├── score.py
│       └── train.py
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the Python path:

```bash
export PYTHONPATH=src
```

---

## Train the Model

```bash
python -m jobplan_risk.train \
  --input data/staging_plan_features.csv \
  --output-dir models
```

This creates:

```text
models/risk_model.joblib
models/metadata.json
```

The training step prepares features, builds pseudo-risk labels, trains the model, and saves the model artifact.

---

## Evaluate the Model

```bash
python -m jobplan_risk.evaluate \
  --input data/staging_plan_features.csv \
  --output outputs/evaluation_report.json
```

The evaluation report includes:

* Accuracy
* Precision
* Recall
* F1 score
* ROC-AUC
* Average precision
* Confusion matrix

These metrics are useful for validating the POC pipeline.

Because the current dataset is synthetic and pseudo-labelled, high metrics prove that the pipeline works, but they do not prove production accuracy.

---

## Score JobPlans

```bash
python -m jobplan_risk.score \
  --input data/staging_plan_features.csv \
  --model-dir models > outputs/risk_scores.json
```

Example output:

```json
{
  "jobPlanCode": "JP-2026-000123",
  "riskScore": 82.4,
  "riskCategory": "High",
  "mainDrivers": [
    "SPA increased compared with prior plan.",
    "Team-plan alignment score is low.",
    "High history change count detected."
  ],
  "dataConfidence": "High",
  "modelMode": "supervised",
  "modelRiskComponent": 88.2,
  "ruleRiskComponent": 75.1
}
```

---

## Run the Dashboard

```bash
export PYTHONPATH=src

python -m streamlit run dashboard/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

In GitHub Codespaces, open the forwarded port:

```text
8501
```

---

## Dashboard Tabs

The dashboard contains the following tabs:

```text
Risk Scores
Evaluation
Training Details
Dataset
Export / Integration
```

The **Risk Scores** tab is the main stakeholder view.

It shows:

* Executive summary
* High / Medium / Low risk counts
* NHS-style operational risk signals
* Trust / specialty hotspots
* Top risk drivers
* Immediate review queue
* Top highest-risk JobPlans
* Full risk score table
* Downloadable CSV output

The **Evaluation** tab shows model metrics.

The **Training Details** tab shows training metadata and model details.

The **Dataset** tab shows the input feature table and dataset distribution.

The **Export / Integration** tab shows generated files and the scoring output contract.

---

## API

Run the API:

```bash
export PYTHONPATH=src

uvicorn jobplan_risk.api:app \
  --reload \
  --host 0.0.0.0 \
  --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

The scoring endpoint can be used by the wider POC flow, including:

* Bedrock narrative layer
* n8n workflow
* rule engine
* demo UI
* future JobPlan analyser service

---

## POC Architecture

```text
Prepared JobPlan feature extract
        ↓
Feature engineering
        ↓
Pseudo-risk label generation
        ↓
ML training and evaluation
        ↓
Risk scoring
        ↓
Dashboard / API / Bedrock / n8n integration
```

---

## Model Approach

The POC combines three parts.

### 1. ML Component

The ML component learns patterns from:

* PA mix
* DCC / SPA balance
* Prior-year changes
* Peer context
* Workflow signals
* TeamJobPlan alignment
* Plan stage and history signals

### 2. Rule Component

The rule component captures deterministic signals such as:

* PA limit breach
* SPA increase
* Team-plan misalignment
* High history change count
* Mediation / appeal
* Returned-to-discussion signal
* Missing TeamJobPlan link
* Locum-related risk signal

### 3. Final Risk Score

The final score blends the ML component and the rule component into a 0-100 risk score.

Risk categories:

```text
High
Medium
Low
```

The goal is not to replace clinical or management judgement. The goal is to prioritise review and highlight explainable risk drivers.

---

## Example Business Interpretation

A high-risk JobPlan may be flagged because:

* SPA has increased compared with the previous plan
* DCC has reduced compared with the previous plan
* TeamPlan alignment is low
* The plan has a high number of changes
* The plan has been returned to discussion
* The plan breaches PA limits
* The plan has mediation or appeal signals

Example stakeholder message:

> This JobPlan has been prioritised for review because it shows SPA/DCC drift, low demand-capacity alignment, and workflow instability. A Clinical Director may want to review the plan before sign-off.

---

## Generated Output Files

After running the dashboard or CLI scoring, the following files may be generated:

```text
outputs/training_metadata.json
outputs/evaluation_report.json
outputs/risk_scores.csv
outputs/risk_scores.json
```

Model artifacts are generated under:

```text
models/
```

These are local/generated outputs and do not need to be committed unless specifically required for demo packaging.

---

## Current Status

Implemented:

* Synthetic staging-shaped dataset
* Feature preparation
* Pseudo-label generation
* Model training
* Model evaluation
* Risk scoring
* Streamlit dashboard
* Stakeholder-friendly Risk Scores tab
* Trust / specialty hotspot view
* Immediate review queue
* Exportable CSV and JSON outputs
* FastAPI scoring endpoint

---

## Next Steps

Recommended next steps:

1. Replace synthetic data with real read-only staging extract.
2. Validate top-risk JobPlans with domain experts.
3. Confirm which risk drivers are most useful for Product and Clinical stakeholders.
4. Connect risk output to Bedrock / n8n / rule-engine flow.
5. Review data protection and governance requirements.
6. Add monitoring for model drift and score stability.
7. Define production-grade outcome labels.
8. Validate fairness and bias across departments, specialties and staff groups.
9. Decide whether the ML layer should remain standalone or become part of a wider JobPlan analyser service.

---

## Real Staging Data Request

For real validation, the team should provide a read-only extract with:

```text
1 row per active JobPlan
```

Required data areas:

* JobPlan code
* Hashed user identifier
* Trust / department / specialty
* Planning year
* Plan stage
* PA totals
* DCC / SPA split
* Prior-year PA values
* Peer medians
* PA limit breach flag
* TeamPlan link / alignment
* History change count
* Mediation / appeal signal
* Returned-to-discussion signal
* Locum flag
* Grade code

Do not include:

* Names
* GMC numbers
* Patient data
* Free-text comments
* Raw WorkEpisode rows
* Sensitive identifiers

---

## Production Caution

This POC should not be presented as a production-validated clinical, workforce, or compliance model.

The right message is:

> The POC validates that a JobPlan ML risk layer can score plans, explain likely risk drivers, and highlight operational hotspots using synthetic/pseudo-labelled data.

Before production use, the model requires:

* Real historical data
* Real outcome labels
* Data quality validation
* Domain expert validation
* Data protection review
* Governance approval
* Monitoring
* Explainability review
* Product sign-off

---

## Demo Talk Track

Suggested demo wording:

> The first tab is the stakeholder view. It shows operational impact first: which JobPlans, Trust areas, and specialties need review, why they are risky, and what action should be taken.

> The ML layer produces a structured risk score and top drivers for each JobPlan. This output can be passed to the rule engine, Bedrock narrative layer, or n8n workflow to generate manager-friendly explanations.

> The current dataset is synthetic and pseudo-labelled, so this validates the pipeline and the dashboard flow. The next step is to replace it with real staging data and validate the top-risk plans with domain experts.

---

## Useful Commands

Install:

```bash
pip install -r requirements.txt
```

Set Python path:

```bash
export PYTHONPATH=src
```

Train:

```bash
python -m jobplan_risk.train \
  --input data/staging_plan_features.csv \
  --output-dir models
```

Evaluate:

```bash
python -m jobplan_risk.evaluate \
  --input data/staging_plan_features.csv \
  --output outputs/evaluation_report.json
```

Score:

```bash
python -m jobplan_risk.score \
  --input data/staging_plan_features.csv \
  --model-dir models > outputs/risk_scores.json
```

Run dashboard:

```bash
python -m streamlit run dashboard/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501
```

Run API:

```bash
uvicorn jobplan_risk.api:app \
  --reload \
  --host 0.0.0.0 \
  --port 8000
```

Run tests:

```bash
pytest
```
