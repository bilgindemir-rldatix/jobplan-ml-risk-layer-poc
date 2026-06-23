from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Body, FastAPI, HTTPException

from .score import score
from .train import train


SERVICE_NAME = "analyser-ml"
DATA_PATH = Path("data/staging_plan_features.csv")
MODEL_DIR = Path("models")


app = FastAPI(
    title="JobPlan Analyser ML API",
    description=(
        "analyser-ml service for JobPlan operational risk scoring. "
        "This service provides riskScore, riskCategory and explainable drivers "
        "for the AI Job Plan Analyser POC."
    ),
    version="0.1.0",
)


def ensure_dataset_exists() -> None:
    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Dataset not found: {DATA_PATH}",
        )


def ensure_model_exists() -> None:
    ensure_dataset_exists()

    model_file = MODEL_DIR / "risk_model.joblib"

    if not model_file.exists():
        train(DATA_PATH, MODEL_DIR)


def load_feature_data() -> pd.DataFrame:
    ensure_dataset_exists()
    return pd.read_csv(DATA_PATH)


def get_scored_data() -> pd.DataFrame:
    ensure_model_exists()

    df = load_feature_data()
    scored = score(df, MODEL_DIR)
    scored_df = pd.DataFrame(scored)

    enrichment_cols = [
        "job_plan_code",
        "trust_level_code",
        "parent_trust_level_code",
        "department",
        "specialty",
        "planning_year",
        "plan_stage",
        "totalPAs",
        "dccPAs",
        "spaPAs",
        "priorTotalPAs",
        "priorDccPAs",
        "priorSpaPAs",
        "teamPlanAlignmentScore",
        "teamDemandPAs",
        "teamPlannedCapacityPAs",
        "historyChangeCount",
        "paLimitBreach",
        "spaAbovePeerThreshold",
        "missingTeamPlanLink",
        "hasMediationOrAppeal",
        "hasNewManagerChanges",
        "planReturnedToDiscussionAfterSignoff",
        "isLocum",
        "gradeCode",
    ]

    available_cols = [c for c in enrichment_cols if c in df.columns]
    enrich = df[available_cols].copy()

    if "job_plan_code" in enrich.columns:
        enrich = enrich.rename(columns={"job_plan_code": "jobPlanCode"})
        scored_df = scored_df.merge(enrich, on="jobPlanCode", how="left")

    if "mainDrivers" in scored_df.columns:
        scored_df["mainDriversText"] = scored_df["mainDrivers"].apply(
            lambda value: " | ".join(value) if isinstance(value, list) else str(value)
        )

    return scored_df


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "modelPath": str(MODEL_DIR / "risk_model.joblib"),
        "datasetPath": str(DATA_PATH),
    }


@app.get("/api/v1/jobplans/{jobPlanCode}/analysis")
def analyse_jobplan(jobPlanCode: str) -> dict[str, Any]:
    """
    Vlatko-aligned endpoint.

    Returns:
    compliance/risk findings, peer/trend context and ML riskScore for one JobPlan.
    """
    scored_df = get_scored_data()

    result = scored_df[scored_df["jobPlanCode"] == jobPlanCode]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail=f"JobPlan not found: {jobPlanCode}",
        )

    row = result.iloc[0].to_dict()

    return {
        "service": SERVICE_NAME,
        "jobPlanCode": row.get("jobPlanCode"),
        "riskScore": row.get("riskScore"),
        "riskCategory": row.get("riskCategory"),
        "mainDrivers": row.get("mainDrivers"),
        "dataConfidence": row.get("dataConfidence"),
        "modelMode": row.get("modelMode"),
        "modelRiskComponent": row.get("modelRiskComponent"),
        "ruleRiskComponent": row.get("ruleRiskComponent"),
        "context": {
            "trustLevelCode": row.get("trust_level_code"),
            "department": row.get("department"),
            "specialty": row.get("specialty"),
            "planningYear": row.get("planning_year"),
            "planStage": row.get("plan_stage"),
            "gradeCode": row.get("gradeCode"),
            "isLocum": row.get("isLocum"),
        },
        "evidence": {
            "totalPAs": row.get("totalPAs"),
            "dccPAs": row.get("dccPAs"),
            "spaPAs": row.get("spaPAs"),
            "priorTotalPAs": row.get("priorTotalPAs"),
            "priorDccPAs": row.get("priorDccPAs"),
            "priorSpaPAs": row.get("priorSpaPAs"),
            "teamPlanAlignmentScore": row.get("teamPlanAlignmentScore"),
            "teamDemandPAs": row.get("teamDemandPAs"),
            "teamPlannedCapacityPAs": row.get("teamPlannedCapacityPAs"),
            "historyChangeCount": row.get("historyChangeCount"),
            "paLimitBreach": row.get("paLimitBreach"),
            "spaAbovePeerThreshold": row.get("spaAbovePeerThreshold"),
            "hasMediationOrAppeal": row.get("hasMediationOrAppeal"),
            "planReturnedToDiscussionAfterSignoff": row.get("planReturnedToDiscussionAfterSignoff"),
        },
        "recommendedAction": build_recommended_action(row),
    }


@app.get("/api/v1/departments/{trustLevelCode}/summary")
def department_summary(trustLevelCode: str) -> dict[str, Any]:
    """
    Vlatko-aligned endpoint.

    Returns:
    department/trust-level summary, top risks and capacity/risk hotspot information.
    """
    scored_df = get_scored_data()

    dept = scored_df[scored_df["trust_level_code"] == trustLevelCode]

    if dept.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No JobPlans found for trustLevelCode: {trustLevelCode}",
        )

    total = len(dept)
    high = int((dept["riskCategory"] == "High").sum())
    medium = int((dept["riskCategory"] == "Medium").sum())
    low = int((dept["riskCategory"] == "Low").sum())

    top_risks = (
        dept.sort_values("riskScore", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )

    specialty_summary = []
    if "specialty" in dept.columns:
        grouped = (
            dept.groupby("specialty")
            .agg(
                totalPlans=("jobPlanCode", "count"),
                highRiskPlans=("riskCategory", lambda x: int((x == "High").sum())),
                mediumRiskPlans=("riskCategory", lambda x: int((x == "Medium").sum())),
                averageRiskScore=("riskScore", "mean"),
                maxRiskScore=("riskScore", "max"),
            )
            .reset_index()
            .sort_values(["highRiskPlans", "averageRiskScore"], ascending=False)
        )

        specialty_summary = grouped.to_dict(orient="records")

    return {
        "service": SERVICE_NAME,
        "trustLevelCode": trustLevelCode,
        "summary": {
            "totalPlans": total,
            "highRiskPlans": high,
            "mediumRiskPlans": medium,
            "lowRiskPlans": low,
            "highRiskRate": high / total if total else 0,
            "averageRiskScore": float(dept["riskScore"].mean()),
            "maxRiskScore": float(dept["riskScore"].max()),
        },
        "topRisks": top_risks,
        "specialtyHotspots": specialty_summary[:10],
    }


@app.post("/api/v1/analysis/batch")
def batch_analysis(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    """
    Vlatko-aligned endpoint.

    For MVP this runs immediate batch scoring against the current feature dataset.
    Later this can queue department-wide analysis via n8n/SQS/worker.
    """
    scored_df = get_scored_data()

    trust_level_code = payload.get("trustLevelCode")

    if trust_level_code:
        scored_df = scored_df[scored_df["trust_level_code"] == trust_level_code]

    return {
        "service": SERVICE_NAME,
        "status": "completed",
        "mode": "synchronous_mvp",
        "trustLevelCode": trust_level_code,
        "scoredPlans": len(scored_df),
        "highRiskPlans": int((scored_df["riskCategory"] == "High").sum()),
        "mediumRiskPlans": int((scored_df["riskCategory"] == "Medium").sum()),
        "lowRiskPlans": int((scored_df["riskCategory"] == "Low").sum()),
        "topRisks": scored_df.sort_values("riskScore", ascending=False)
        .head(10)
        .to_dict(orient="records"),
    }


@app.post("/api/v1/scenarios/simulate")
def simulate_scenario(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """
    Vlatko-aligned endpoint.

    MVP placeholder for PA rebalance simulation.
    This does not recalculate full WorkEpisode logic.
    """
    department_code = payload.get("departmentCode") or payload.get("trustLevelCode")
    adjustments = payload.get("adjustments", [])

    if not department_code:
        raise HTTPException(
            status_code=400,
            detail="departmentCode or trustLevelCode is required",
        )

    scored_df = get_scored_data()
    dept = scored_df[scored_df["trust_level_code"] == department_code]

    if dept.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No JobPlans found for departmentCode/trustLevelCode: {department_code}",
        )

    current_total_spa = float(pd.to_numeric(dept.get("spaPAs"), errors="coerce").fillna(0).sum())
    current_total_dcc = float(pd.to_numeric(dept.get("dccPAs"), errors="coerce").fillna(0).sum())

    simulated_spa = current_total_spa
    simulated_dcc = current_total_dcc

    for adjustment in adjustments:
        category = str(adjustment.get("category", "")).upper()
        delta_pa = float(adjustment.get("deltaPA", 0))

        if category == "SPA":
            simulated_spa += delta_pa * len(dept)
            simulated_dcc -= delta_pa * len(dept)
        elif category == "DCC":
            simulated_dcc += delta_pa * len(dept)
            simulated_spa -= delta_pa * len(dept)

    return {
        "service": SERVICE_NAME,
        "departmentCode": department_code,
        "scenarioType": "lightweight_pa_rebalance_mvp",
        "warning": "This is a lightweight MVP simulation and does not recalculate full WorkEpisode logic.",
        "before": {
            "totalSpaPAs": round(current_total_spa, 2),
            "totalDccPAs": round(current_total_dcc, 2),
            "highRiskPlans": int((dept["riskCategory"] == "High").sum()),
            "averageRiskScore": float(dept["riskScore"].mean()),
        },
        "after": {
            "simulatedTotalSpaPAs": round(simulated_spa, 2),
            "simulatedTotalDccPAs": round(simulated_dcc, 2),
        },
        "adjustments": adjustments,
        "narrativeHint": (
            "Use Bedrock to narrate the trade-off: reducing SPA may improve SPA/DCC balance "
            "but could widen capacity or contractual discussion depending on context."
        ),
    }


@app.post("/score")
def legacy_score(payload: list[dict[str, Any]] | None = Body(default=None)) -> list[dict[str, Any]]:
    """
    Existing simple scoring endpoint kept for backward compatibility.
    """
    ensure_model_exists()

    if payload:
        df = pd.DataFrame(payload)
    else:
        df = load_feature_data()

    return score(df, MODEL_DIR)


def build_recommended_action(row: dict[str, Any]) -> str:
    drivers = str(row.get("mainDriversText", "")).lower()
    risk_score = float(row.get("riskScore", 0))

    if risk_score >= 85:
        return "Immediate Clinical Director review"
    if "team" in drivers or "alignment" in drivers:
        return "Review demand/capacity alignment"
    if "spa" in drivers or "dcc" in drivers:
        return "Review SPA/DCC balance"
    if "history" in drivers or "returned" in drivers:
        return "Review workflow instability and plan changes"
    if risk_score >= 70:
        return "Prioritise in next job-planning review"
    return "Monitor"
