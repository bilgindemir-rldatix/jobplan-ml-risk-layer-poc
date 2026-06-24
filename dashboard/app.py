from __future__ import annotations

import ast
import html
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT = Path(".")
DATA_PATH = ROOT / "data" / "staging_plan_features.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
RISK_SCORES_PATH = OUTPUT_DIR / "risk_scores.csv"
HIGH_RISK_JSON_PATH = OUTPUT_DIR / "high_risk_jobplans.json"
HIGH_RISK_CSV_PATH = OUTPUT_DIR / "high_risk_jobplans.csv"

sys.path.insert(0, str(ROOT / "src"))


TRUST_CONTEXT = {
    "TLC-ONC": ("Medicine", "Oncology"),
    "TLC-CARD": ("Medicine", "Cardiology"),
    "TLC-GASTRO": ("Medicine", "Gastroenterology"),
    "TLC-RESP": ("Medicine", "Respiratory"),
    "TLC-UROL": ("Surgery", "Urology"),
    "TLC-ENT": ("Surgery", "ENT"),
    "TLC-TNO": ("Surgery", "Trauma & Orthopaedics"),
    "TLC-RAD": ("Diagnostics", "Radiology"),
    "TLC-PAED": ("Women & Children", "Paediatrics"),
    "TLC-ED": ("Emergency", "Emergency Medicine"),
}


MODEL_DISPLAY_NAMES = {
    "hist_gradient_boosting": "Histogram-based Gradient Boosting",
    "logistic_regression": "Logistic Regression",
    "isolation_forest": "Isolation Forest",
}


st.set_page_config(
    page_title="JobPlan ML Risk Layer",
    page_icon="🧠",
    layout="wide",
)


st.markdown(
    """
    <style>
        .stApp {
            background: #F5FAF7;
            color: #000000 !important;
        }

        h1, h2, h3, h4, h5, h6, p, span, div, label {
            color: #000000;
        }

        .main-title {
            font-size: 36px;
            font-weight: 950;
            color: #063B2E !important;
            margin-bottom: 4px;
        }

        .main-subtitle {
            font-size: 17px;
            color: #102A21 !important;
            margin-bottom: 18px;
        }

        .hero-box {
            background: #FFFFFF;
            border: 2px solid #CFE7DC;
            border-left: 8px solid #0B6F4F;
            border-radius: 18px;
            padding: 18px 20px;
            margin: 12px 0 18px 0;
            box-shadow: 0 6px 18px rgba(6, 59, 46, 0.08);
        }

        .hero-title {
            font-size: 22px;
            font-weight: 900;
            color: #063B2E !important;
            margin-bottom: 8px;
        }

        .hero-text {
            font-size: 15px;
            line-height: 1.55;
            color: #102A21 !important;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 14px 0 20px 0;
        }

        .metric-card {
            background: #FFFFFF;
            border: 1px solid #D8E7DF;
            border-radius: 16px;
            padding: 16px;
            box-shadow: 0 4px 14px rgba(6, 59, 46, 0.06);
        }

        .metric-label {
            font-size: 12px;
            font-weight: 850;
            color: #0B6F4F !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 6px;
        }

        .metric-value {
            font-size: 26px;
            font-weight: 950;
            color: #000000 !important;
        }

        .metric-help {
            font-size: 12px;
            color: #102A21 !important;
            margin-top: 4px;
        }

        .risk-card {
            background: #FFFFFF;
            border: 2px solid #CFE7DC;
            border-radius: 18px;
            padding: 16px 18px;
            margin-bottom: 14px;
            box-shadow: 0 6px 18px rgba(6, 59, 46, 0.08);
            min-height: 285px;
        }

        .risk-card-high {
            border-left: 8px solid #B42318;
        }

        .risk-card-medium {
            border-left: 8px solid #F97316;
        }

        .risk-card-low {
            border-left: 8px solid #EAB308;
        }

        .risk-title {
            font-size: 19px;
            font-weight: 950;
            color: #063B2E !important;
            margin-bottom: 4px;
        }

        .risk-context {
            font-size: 13px;
            color: #102A21 !important;
            margin-bottom: 4px;
        }

        .risk-score {
            font-size: 34px;
            font-weight: 950;
            margin: 8px 0;
        }

        .risk-pill {
            display: inline-block;
            border-radius: 999px;
            padding: 5px 10px;
            font-size: 12px;
            font-weight: 850;
            color: #FFFFFF !important;
            background: #B42318;
        }

        .driver-list {
            margin-top: 10px;
            padding-left: 18px;
            font-size: 13px;
            line-height: 1.45;
        }

        .action-box {
            background: #F7FCFA;
            border: 1px solid #D8E7DF;
            border-radius: 12px;
            padding: 10px;
            margin-top: 10px;
            font-size: 13px;
            font-weight: 700;
            color: #102A21 !important;
        }

        .light-box {
            background: #FFFFFF;
            border: 2px solid #CFE7DC;
            border-left: 8px solid #0B6F4F;
            border-radius: 18px;
            padding: 16px 18px;
            margin: 12px 0 18px 0;
            color: #000000 !important;
            box-shadow: 0 6px 18px rgba(6, 59, 46, 0.08);
        }

        .light-line {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 14px;
            background: #F7FCFA;
            border: 1px solid #D8E7DF;
            border-radius: 10px;
            padding: 8px 10px;
            margin: 6px 0;
            color: #000000 !important;
        }

        .chip {
            display: inline-block;
            background: #F3FAF6;
            border: 1px solid #CFE7DC;
            color: #000000 !important;
            border-radius: 999px;
            padding: 7px 12px;
            margin: 4px 6px 4px 0;
            font-size: 13px;
            font-weight: 750;
        }

        .flow-step {
            background: #F3FAF6;
            border: 1px solid #CFE7DC;
            border-radius: 14px;
            padding: 12px;
            margin-bottom: 10px;
            font-weight: 800;
            color: #063B2E !important;
            text-align: center;
        }

        .flow-arrow {
            text-align: center;
            font-size: 22px;
            font-weight: 900;
            color: #0B6F4F !important;
            margin: 2px 0 8px 0;
        }

        .endpoint-row {
            display: flex;
            gap: 10px;
            align-items: center;
            background: #F7FCFA;
            border: 1px solid #D8E7DF;
            border-radius: 10px;
            padding: 8px 10px;
            margin: 6px 0;
            color: #000000 !important;
        }

        .method-badge {
            min-width: 54px;
            text-align: center;
            padding: 4px 8px;
            border-radius: 999px;
            background: #0B6F4F;
            color: #FFFFFF !important;
            font-size: 12px;
            font-weight: 850;
        }

        .endpoint-path {
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            color: #000000 !important;
            font-size: 14px;
        }

        textarea,
        div[data-testid="stTextArea"] textarea,
        div[data-testid="stTextArea"] textarea:disabled,
        div[data-testid="stTextArea"] textarea[disabled] {
            background-color: #FFFFFF !important;
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            opacity: 1 !important;
            caret-color: #000000 !important;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace !important;
            font-size: 13px !important;
            line-height: 1.45 !important;
            border: 2px solid #CFE7DC !important;
            border-radius: 12px !important;
        }

        div[data-testid="stTextArea"] label,
        div[data-testid="stTextArea"] label p {
            color: #063B2E !important;
            font-weight: 850 !important;
            font-size: 16px !important;
        }
    
        .risk-score-high {
            color: #B42318 !important;
        }

        .risk-score-medium {
            color: #F97316 !important;
        }

        .risk-score-low {
            color: #EAB308 !important;
        }

        .risk-pill-high {
            background: #B42318 !important;
            color: #FFFFFF !important;
        }

        .risk-pill-medium {
            background: #F97316 !important;
            color: #000000 !important;
        }

        .risk-pill-low {
            background: #EAB308 !important;
            color: #000000 !important;
        }

    </style>
    """,
    unsafe_allow_html=True,
)


def is_missing(value: Any) -> bool:
    if value is None:
        return True

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.size == 0

        if isinstance(value, np.generic):
            value = value.item()
    except Exception:
        pass

    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0

    try:
        result = pd.isna(value)

        if isinstance(result, bool):
            return result

        # If pandas/numpy returns an array-like result, this is not a scalar missing value.
        if hasattr(result, "size") and result.size != 1:
            return False

        return bool(result)
    except Exception:
        pass

    return str(value).strip() in {"", "N/A", "nan", "None", "null"}


def first_present(row: pd.Series, names: list[str]) -> Any:
    for name in names:
        if name in row.index:
            value = row.get(name)
            if not is_missing(value):
                return value
    return None


def risk_category_from_score(score: float) -> str:
    if score >= 75:
        return "High"
    if score >= 50:
        return "Medium"
    return "Low"


def parse_drivers(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if not is_missing(v)]

    if is_missing(value):
        return []

    if isinstance(value, str):
        value = value.strip()

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if not is_missing(v)]
        except Exception:
            pass

        if " | " in value:
            return [v.strip() for v in value.split(" | ") if v.strip()]

        if ";" in value:
            return [v.strip() for v in value.split(";") if v.strip()]

        return [value]

    return [str(value)]


def derive_context(row: pd.Series) -> tuple[str, str, str]:
    trust = first_present(
        row,
        [
            "trustLevelCode",
            "trust_level_code",
            "TrustLevelCode",
            "target_department",
            "departmentCode",
        ],
    )

    department = first_present(
        row,
        [
            "department",
            "Department",
            "department_name",
            "departmentName",
            "target_department_name",
        ],
    )

    specialty = first_present(
        row,
        [
            "specialty",
            "Specialty",
            "speciality",
            "Speciality",
            "specialty_name",
            "specialtyName",
        ],
    )

    if not is_missing(trust) and trust in TRUST_CONTEXT:
        mapped_department, mapped_specialty = TRUST_CONTEXT[trust]

        if is_missing(department):
            department = mapped_department

        if is_missing(specialty):
            specialty = mapped_specialty

    if is_missing(trust):
        trust = "Unknown Trust"

    if is_missing(department):
        department = "Unknown Department"

    if is_missing(specialty):
        specialty = "Unknown Specialty"

    return str(trust), str(department), str(specialty)


def normalise_scored_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "riskScore" not in df.columns:
        raise ValueError("riskScore column is missing from scored data.")

    df["riskScore"] = pd.to_numeric(df["riskScore"], errors="coerce").fillna(0.0)

    if "riskCategory" not in df.columns:
        df["riskCategory"] = df["riskScore"].apply(risk_category_from_score)

    contexts = df.apply(derive_context, axis=1, result_type="expand")
    df["trustLevelCode"] = contexts[0]
    df["trust_level_code"] = contexts[0]
    df["department"] = contexts[1]
    df["specialty"] = contexts[2]

    if "mainDrivers" in df.columns:
        df["mainDrivers"] = df["mainDrivers"].apply(parse_drivers)
    elif "mainDriversText" in df.columns:
        df["mainDrivers"] = df["mainDriversText"].apply(parse_drivers)
    else:
        df["mainDrivers"] = [[] for _ in range(len(df))]

    df["mainDriversText"] = df["mainDrivers"].apply(lambda drivers: " | ".join(drivers))

    if "recommendedAction" not in df.columns:
        df["recommendedAction"] = df["riskCategory"].map(
            {
                "High": "Immediate Clinical Director review",
                "Medium": "Review during next job planning checkpoint",
                "Low": "No immediate action required",
            }
        )

    if "riskInterpretation" not in df.columns:
        df["riskInterpretation"] = df["riskCategory"].map(
            {
                "High": "This JobPlan should be prioritised for review.",
                "Medium": "This JobPlan has moderate risk signals and should be monitored.",
                "Low": "This JobPlan appears stable based on the current POC signals.",
            }
        )

    if "dataConfidence" not in df.columns:
        df["dataConfidence"] = "High"

    return df


def load_metadata() -> dict:
    metadata_path = MODEL_DIR / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def load_evaluation_report() -> dict:
    evaluation_path = OUTPUT_DIR / "evaluation_report.json"
    if evaluation_path.exists():
        return json.loads(evaluation_path.read_text(encoding="utf-8"))
    return {}


def load_scored_data() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(exist_ok=True)

    if RISK_SCORES_PATH.exists():
        df = pd.read_csv(RISK_SCORES_PATH)
        return normalise_scored_df(df)

    if not DATA_PATH.exists():
        st.error("No scored output or input dataset found.")
        st.stop()

    try:
        from jobplan_risk.score import score

        raw_df = pd.read_csv(DATA_PATH)
        scored = score(raw_df, MODEL_DIR)
        scored_df = pd.DataFrame(scored)
        scored_df = normalise_scored_df(scored_df)

        scored_df.to_csv(RISK_SCORES_PATH, index=False)
        (OUTPUT_DIR / "risk_scores.json").write_text(
            scored_df.to_json(orient="records", indent=2),
            encoding="utf-8",
        )

        return scored_df

    except Exception as exc:
        st.error(f"Could not score data: {exc}")
        st.stop()


def active_model_name(metadata: dict) -> tuple[str, str]:
    raw = (
        metadata.get("model_mode")
        or metadata.get("active_model")
        or metadata.get("model_name")
        or "hist_gradient_boosting"
    )
    return str(raw), MODEL_DISPLAY_NAMES.get(str(raw), str(raw))


def metrics_for_active_model(report: dict, active_raw: str) -> dict:
    if not isinstance(report, dict):
        return {}

    candidate = None

    if active_raw in report:
        candidate = report[active_raw]

    elif "models" in report and isinstance(report["models"], dict):
        candidate = report["models"].get(active_raw)

    elif "model_results" in report and isinstance(report["model_results"], dict):
        candidate = report["model_results"].get(active_raw)

    elif "metrics" in report and isinstance(report["metrics"], dict):
        candidate = report["metrics"]

    else:
        candidate = report

    if isinstance(candidate, dict) and "test" in candidate and isinstance(candidate["test"], dict):
        candidate = candidate["test"]

    if isinstance(candidate, dict) and "metrics" in candidate and isinstance(candidate["metrics"], dict):
        candidate = candidate["metrics"]

    return candidate if isinstance(candidate, dict) else {}


def metric_value(metrics: dict, *names: str) -> Any:
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def format_number(value: Any, digits: int = 1) -> str:
    if is_missing(value):
        return "N/A"

    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def metric_cards(cards: list[tuple[str, str, str]]) -> None:
    html_cards = ""
    for label, value, help_text in cards:
        html_cards += f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(label)}</div>
            <div class="metric-value">{html.escape(str(value))}</div>
            <div class="metric-help">{html.escape(help_text)}</div>
        </div>
        """

    st.markdown(
        f"""
        <div class="metric-grid">
            {html_cards}
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_card(row: pd.Series) -> None:
    job_plan = row.get("jobPlanCode", row.get("job_plan_code", "N/A"))
    trust = row.get("trustLevelCode", row.get("trust_level_code", "Unknown Trust"))
    department = row.get("department", "Unknown Department")
    specialty = row.get("specialty", "Unknown Specialty")
    score = format_number(row.get("riskScore"), 1)

    raw_category = str(row.get("riskCategory", "High")).strip().lower()

    category = {
        "high": "High",
        "medium": "Medium",
        "low": "Low",
    }.get(raw_category, "High")

    category_key = category.lower()

    action = row.get("recommendedAction", "Immediate Clinical Director review")
    drivers = parse_drivers(row.get("mainDrivers", row.get("mainDriversText", [])))[:3]

    category_class = f"risk-card-{category_key}"
    score_class = f"risk-score-{category_key}"
    pill_class = f"risk-pill-{category_key}"

    driver_items = "".join(
        f"<li>{html.escape(str(driver))}</li>" for driver in drivers
    )

    if not driver_items:
        driver_items = "<li>No driver details available.</li>"

    st.markdown(
        f"""
        <div class="risk-card {category_class}">
            <div class="risk-title">{html.escape(str(job_plan))}</div>
            <div class="risk-context">{html.escape(str(department))} / {html.escape(str(specialty))}</div>
            <div class="risk-context">Trust: {html.escape(str(trust))}</div>
            <div class="risk-score {score_class}">{html.escape(score)}</div>
            <span class="risk-pill {pill_class}">{html.escape(category)}</span>
            <div style="font-weight:850; margin-top:12px;">Main Risk Drivers</div>
            <ul class="driver-list">{driver_items}</ul>
            <div class="action-box">Action: {html.escape(str(action))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clean_nulls(obj: Any) -> Any:
    def to_json_safe(value: Any) -> Any:
        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                return value.tolist()

            if isinstance(value, np.generic):
                return value.item()
        except Exception:
            pass

        if isinstance(value, pd.Timestamp):
            return value.isoformat()

        return value

    def is_empty_cleaned(value: Any) -> bool:
        if value is None:
            return True

        if value == "":
            return True

        try:
            import numpy as np

            if isinstance(value, np.ndarray):
                return value.size == 0
        except Exception:
            pass

        if isinstance(value, (list, tuple, set, dict)):
            return len(value) == 0

        return False

    obj = to_json_safe(obj)

    if isinstance(obj, dict):
        cleaned = {}

        for key, value in obj.items():
            cleaned_value = clean_nulls(value)

            if not is_empty_cleaned(cleaned_value):
                cleaned[key] = cleaned_value

        return cleaned

    if isinstance(obj, (list, tuple, set)):
        cleaned_list = []

        for item in obj:
            cleaned_item = clean_nulls(item)

            if not is_empty_cleaned(cleaned_item):
                cleaned_list.append(cleaned_item)

        return cleaned_list

    if is_missing(obj):
        return None

    return to_json_safe(obj)


def build_handoff_df(scored_df: pd.DataFrame) -> pd.DataFrame:
    high_risk_df = scored_df[scored_df["riskCategory"] == "High"].copy()
    high_risk_df = high_risk_df.sort_values("riskScore", ascending=False)

    base_columns = [
        "jobPlanCode",
        "trustLevelCode",
        "department",
        "specialty",
        "riskScore",
        "riskCategory",
        "mainDrivers",
        "mainDriversText",
        "recommendedAction",
        "riskInterpretation",
        "dataConfidence",
        "modelRiskComponent",
        "ruleRiskComponent",
    ]

    optional_evidence_columns = [
        "totalPAs",
        "dccPAs",
        "spaPAs",
        "priorDccPAs",
        "priorSpaPAs",
        "teamPlanAlignmentScore",
        "teamDemandPAs",
        "teamPlannedCapacityPAs",
        "historyChangeCount",
        "isLocum",
    ]

    selected_columns = [col for col in base_columns if col in high_risk_df.columns]

    for col in optional_evidence_columns:
        if col in high_risk_df.columns and not high_risk_df[col].isna().all():
            selected_columns.append(col)

    return high_risk_df[selected_columns].copy()


def save_exports(scored_df: pd.DataFrame) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(exist_ok=True)

    handoff_df = build_handoff_df(scored_df)

    scored_export_df = scored_df.copy()
    scored_export_df.to_csv(OUTPUT_DIR / "risk_scores.csv", index=False)
    (OUTPUT_DIR / "risk_scores.json").write_text(
        scored_export_df.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    handoff_df.to_csv(HIGH_RISK_CSV_PATH, index=False)
    HIGH_RISK_JSON_PATH.write_text(
        handoff_df.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    return handoff_df


def example_handoff_json(row: pd.Series) -> dict:
    drivers = parse_drivers(row.get("mainDrivers", row.get("mainDriversText", [])))[:3]

    payload = {
        "source": "analyser-ml",
        "purpose": "JobPlan review prioritisation",
        "jobPlanCode": row.get("jobPlanCode"),
        "trustLevelCode": row.get("trustLevelCode"),
        "department": row.get("department"),
        "specialty": row.get("specialty"),
        "riskScore": row.get("riskScore"),
        "riskCategory": row.get("riskCategory"),
        "mainDrivers": drivers,
        "recommendedAction": row.get("recommendedAction"),
        "riskComponents": {
            "modelRiskComponent": row.get("modelRiskComponent"),
            "ruleRiskComponent": row.get("ruleRiskComponent"),
            "dataConfidence": row.get("dataConfidence"),
        },
        "decisionPolicy": "Review prioritisation only. Not an automated approval or rejection decision.",
    }

    return clean_nulls(payload)


scored_df = load_scored_data()
metadata = load_metadata()
evaluation_report = load_evaluation_report()
active_raw_model, active_display_model = active_model_name(metadata)
active_metrics = metrics_for_active_model(evaluation_report, active_raw_model)
handoff_df = save_exports(scored_df)

high_risk_df = scored_df[scored_df["riskCategory"] == "High"].copy()
high_risk_df = high_risk_df.sort_values("riskScore", ascending=False).reset_index(drop=True)

medium_risk_df = scored_df[scored_df["riskCategory"] == "Medium"].copy()
low_risk_df = scored_df[scored_df["riskCategory"] == "Low"].copy()

st.markdown(
    """
    <div class="main-title">JobPlan ML Risk Layer POC</div>
    <div class="main-subtitle">
        Review prioritisation for JobPlans using ML risk scoring, explainable drivers, and structured integration output.
    </div>
    """,
    unsafe_allow_html=True,
)


tab_overview, tab_scores, tab_high_risk, tab_drivers, tab_model, tab_export = st.tabs(
    [
        "POC Overview",
        "Risk Scores",
        "JobPlan Review Queue",
        "Drivers & Hotspots",
        "Model & Training",
        "Export & Integration",
    ]
)


with tab_overview:
    st.header("POC Overview")

    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">What this POC demonstrates</div>
            <div class="hero-text">
                This POC shows how a machine-learning risk layer can help prioritise JobPlans for review.
                Each JobPlan receives a 0–100 risk score, a risk category, main risk drivers, and a recommended review action.
                The goal is not to replace Clinical Director judgement, but to help product and engineering teams understand how ML could support review prioritisation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cards(
        [
            ("Total JobPlans", f"{len(scored_df):,}", "All scored JobPlans in the current dataset"),
            ("High Risk", f"{len(high_risk_df):,}", "JobPlans prioritised for immediate review"),
            ("Average Score", f"{scored_df['riskScore'].mean():.1f}", "Mean risk score across all JobPlans"),
            ("Active Model", active_display_model, "Single saved model used by dashboard and API"),
        ]
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="light-box">
                <div class="hero-title">What it does</div>
                <span class="chip">Scores JobPlans 0–100</span>
                <span class="chip">Finds highest-risk plans</span>
                <span class="chip">Explains main drivers</span>
                <span class="chip">Exports JSON / CSV</span>
                <span class="chip">Supports API integration</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="light-box">
                <div class="hero-title">What it does not do</div>
                <span class="chip">Does not approve JobPlans</span>
                <span class="chip">Does not reject JobPlans</span>
                <span class="chip">Does not replace clinical judgement</span>
                <span class="chip">Does not create compliance facts by itself</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="light-box">
            <div class="hero-title">Current POC status</div>
            <div class="hero-text">
                The current dataset is synthetic / staging-shaped and uses pseudo-risk labels.
                This validates the pipeline, model output, dashboard visibility, API shape, and integration contract.
                The next step is to replace the dataset with a real staging extract and validate the highest-risk plans with domain experts.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with tab_scores:
    st.header("Risk Scores")

    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">JobPlan-level risk scoring</div>
            <div class="hero-text">
                Risk score is assigned to the JobPlan, not to workflow stage.
                This tab provides the score distribution and a searchable scored output view.
                Detailed review cards are kept separately in the Highest-Risk JobPlans tab.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    risk_counts = scored_df["riskCategory"].value_counts()

    metric_cards(
        [
            ("High", f"{int(risk_counts.get('High', 0)):,}", "Prioritise first"),
            ("Medium", f"{int(risk_counts.get('Medium', 0)):,}", "Monitor / review checkpoint"),
            ("Low", f"{int(risk_counts.get('Low', 0)):,}", "No immediate action"),
            ("Max Score", f"{scored_df['riskScore'].max():.1f}", "Highest current risk score"),
        ]
    )

    distribution_df = pd.DataFrame(
        [
            {"Risk Category": "High", "Count": int(risk_counts.get("High", 0))},
            {"Risk Category": "Medium", "Count": int(risk_counts.get("Medium", 0))},
            {"Risk Category": "Low", "Count": int(risk_counts.get("Low", 0))},
        ]
    )

    st.subheader("Risk Category Distribution")
    st.dataframe(distribution_df, use_container_width=True, hide_index=True)

    st.subheader("Scored JobPlans")
    search = st.text_input("Filter by JobPlan, Trust, Department or Specialty", "")

    score_columns = [
        "jobPlanCode",
        "trustLevelCode",
        "department",
        "specialty",
        "riskScore",
        "riskCategory",
        "mainDriversText",
        "recommendedAction",
    ]

    available_score_columns = [col for col in score_columns if col in scored_df.columns]
    visible_scores = scored_df[available_score_columns].sort_values("riskScore", ascending=False)

    if search.strip():
        search_lower = search.strip().lower()
        visible_scores = visible_scores[
            visible_scores.astype(str).apply(
                lambda row: row.str.lower().str.contains(search_lower, na=False).any(),
                axis=1,
            )
        ]

    st.caption("Showing top 200 rows for dashboard performance. Full output is available in Export / Integration.")
    st.dataframe(visible_scores.head(200), use_container_width=True, hide_index=True)


with tab_high_risk:
    st.header("JobPlan Review Queue")

    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">Review queue</div>
            <div class="hero-text">
                This tab shows JobPlans as review cards. By default it focuses on High-Risk JobPlans,
                but Medium and Low risk cards can also be added for comparison during product or engineering discussions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    high_risk_sorted = high_risk_df.sort_values("riskScore", ascending=False).reset_index(drop=True)
    medium_risk_sorted = medium_risk_df.sort_values("riskScore", ascending=False).reset_index(drop=True)
    low_risk_sorted = low_risk_df.sort_values("riskScore", ascending=False).reset_index(drop=True)

    high_total = int(len(high_risk_sorted))
    medium_total = int(len(medium_risk_sorted))
    low_total = int(len(low_risk_sorted))

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    def card_slider(label, total, default_value, key):
        if total <= 0:
            st.caption(f"{label}: no JobPlans available")
            return 0

        return int(
            st.slider(
                label,
                min_value=0,
                max_value=total,
                value=min(default_value, total),
                step=1,
                key=key,
            )
        )

    with filter_col1:
        high_visible = card_slider(
            "High-risk cards to show",
            high_total,
            12,
            "high_risk_cards_to_show",
        )

    with filter_col2:
        medium_visible = card_slider(
            "Medium-risk cards to show",
            medium_total,
            0,
            "medium_risk_cards_to_show",
        )

    with filter_col3:
        low_visible = card_slider(
            "Low-risk cards to show",
            low_total,
            0,
            "low_risk_cards_to_show",
        )

    selected_high = high_risk_sorted.head(high_visible)
    selected_medium = medium_risk_sorted.head(medium_visible)
    selected_low = low_risk_sorted.head(low_visible)

    total_selected = len(selected_high) + len(selected_medium) + len(selected_low)

    st.caption(
        f"Showing {len(selected_high):,} High-risk, {len(selected_medium):,} Medium-risk, "
        f"and {len(selected_low):,} Low-risk JobPlan cards "
        f"({total_selected:,} total visible). "
        f"Available: {high_total:,} High, {medium_total:,} Medium, {low_total:,} Low."
    )

    if total_selected == 0:
        st.info("No cards selected. Increase one of the filters above to display JobPlans.")
    else:
        sections = [
            ("High", selected_high),
            ("Medium", selected_medium),
            ("Low", selected_low),
        ]

        for section_label, section_df in sections:
            if len(section_df) == 0:
                continue

            st.subheader(f"{section_label}-Risk JobPlans")

            rows = list(section_df.iterrows())

            for i in range(0, len(rows), 3):
                cols = st.columns(3, gap="medium")
                for j in range(3):
                    if i + j < len(rows):
                        _, row = rows[i + j]
                        with cols[j]:
                            risk_card(row)


with tab_drivers:
    st.header("Drivers & Hotspots")

    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">Why JobPlans are being prioritised</div>
            <div class="hero-text">
                This tab helps product and engineering teams understand which risk signals are appearing most often
                and where high-risk plans are concentrated.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    driver_rows = []
    for _, row in scored_df.iterrows():
        for driver in parse_drivers(row.get("mainDrivers", [])):
            driver_rows.append(
                {
                    "Driver": driver,
                    "Risk Category": row.get("riskCategory"),
                    "Risk Score": row.get("riskScore"),
                    "Trust": row.get("trustLevelCode"),
                    "Department": row.get("department"),
                    "Specialty": row.get("specialty"),
                }
            )

    driver_df = pd.DataFrame(driver_rows)

    if len(driver_df) == 0:
        st.info("No risk driver information available.")
    else:
        st.subheader("Most Frequent Risk Drivers")

        driver_frequency = (
            driver_df.groupby("Driver")
            .size()
            .reset_index(name="Count")
            .sort_values("Count", ascending=False)
        )

        st.dataframe(driver_frequency.head(15), use_container_width=True, hide_index=True)

    st.subheader("High-Risk Hotspots")

    hotspot_columns = ["trustLevelCode", "department", "specialty"]

    hotspot_df = (
        high_risk_df.groupby(hotspot_columns)
        .agg(
            HighRiskJobPlans=("jobPlanCode", "count"),
            AverageRiskScore=("riskScore", "mean"),
            MaxRiskScore=("riskScore", "max"),
        )
        .reset_index()
        .sort_values(["HighRiskJobPlans", "AverageRiskScore"], ascending=False)
    )

    hotspot_df["AverageRiskScore"] = hotspot_df["AverageRiskScore"].round(1)
    hotspot_df["MaxRiskScore"] = hotspot_df["MaxRiskScore"].round(1)

    st.dataframe(hotspot_df.head(50), use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="light-box">
            <div class="hero-title">Product / Engineering interpretation</div>
            <div class="hero-text">
                Hotspots can help decide where to focus validation with domain experts.
                Driver frequency can also help engineering confirm which features and rule-engine findings should be prioritised for real-data integration.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with tab_model:
    st.header("Model & Training")

    st.markdown(
        f"""
        <div class="hero-box">
            <div class="hero-title">Active Model: {html.escape(active_display_model)}</div>
            <div class="hero-text">
                The dashboard and API use the single trained model saved in <b>models/risk_model.joblib</b>.
                Histogram-based Gradient Boosting is a good fit for this POC because it handles mixed operational signals well
                and can capture non-linear risk patterns without requiring a large deep-learning setup.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown(
        """
        <div class="light-box">
            <div class="hero-title">Where ML and rule-based risk live</div>
            <div class="hero-text">
                The trained ML model is stored in <b>models/risk_model.joblib</b>.
                Rule-based risk evidence is not stored inside the model file; it is implemented in the scoring
                and explanation code, mainly <b>src/jobplan_risk/explain.py</b>.
                The final risk score is produced in <b>src/jobplan_risk/score.py</b> by combining the ML prediction
                with rule-based evidence such as PA limit breach, SPA increase, DCC reduction, low team-plan alignment,
                missing team-plan link, and workflow instability.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    training_info = metadata.get("training_info", metadata.get("training", {}))
    feature_columns = (
        metadata.get("feature_columns")
        or metadata.get("features")
        or metadata.get("input_features")
        or []
    )

    feature_count = len(feature_columns) if feature_columns else "N/A"
    training_rows = (
        training_info.get("training_rows")
        or training_info.get("rows")
        or metadata.get("training_rows")
        or len(scored_df)
    )
    positive_rate = training_info.get("positive_rate") or metadata.get("positive_rate") or "N/A"

    if isinstance(positive_rate, float):
        positive_rate_display = f"{positive_rate:.2%}"
    else:
        positive_rate_display = str(positive_rate)

    metric_cards(
        [
            ("Training / Scoring Rows", f"{training_rows}", "Current dataset size used by the POC"),
            ("Feature Count", f"{feature_count}", "Model input signals where available"),
            ("Training Risk Label Rate", positive_rate_display, "Share of POC training rows labelled as review-risk"),
            ("Model Artifact", "risk_model.joblib", "Single model used by dashboard and API"),
        ]
    )

    st.subheader("Evaluation Metrics")

    metric_labels = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "roc_auc": "ROC-AUC",
        "average_precision": "Average Precision",
    }

    eval_rows = []
    for key, label in metric_labels.items():
        value = metric_value(active_metrics, key)
        if value is not None:
            eval_rows.append(
                {
                    "Metric": label,
                    "Value": round(float(value), 4) if isinstance(value, (int, float)) else value,
                }
            )

    if eval_rows:
        st.dataframe(pd.DataFrame(eval_rows), use_container_width=True, hide_index=True)
    else:
        st.warning("No active-model evaluation metrics found. Regenerate outputs/evaluation_report.json if needed.")

    st.subheader("What Determines the JobPlan Risk Score")

    st.markdown(
        """
        <div class="light-box">
            <div class="hero-title">Risk scoring logic</div>
            <div class="hero-text">
                The JobPlan risk score is produced from a hybrid approach. The trained ML model is stored in
                <b>models/risk_model.joblib</b> and predicts review risk from patterns across PA allocation,
                SPA/DCC movement, peer deviation, team-plan alignment, capacity-gap indicators, workflow instability,
                and locum/contract signals.
                <br><br>
                Rule-based risk evidence is not stored inside the model file. It is implemented in the scoring and
                explanation code, mainly <b>src/jobplan_risk/explain.py</b>. The final risk score is produced in
                <b>src/jobplan_risk/score.py</b> by combining the ML prediction with transparent rule-based evidence
                such as PA limit breach, SPA increase, DCC reduction, low team-plan alignment, missing team-plan link,
                and workflow instability.
                <br><br>
                The score is used only to prioritise review, not to approve or reject a JobPlan.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="light-box">
            <span class="chip">SPA / DCC drift</span>
            <span class="chip">Prior-year PA changes</span>
            <span class="chip">PA limit breach signals</span>
            <span class="chip">Peer deviation</span>
            <span class="chip">Team-plan alignment</span>
            <span class="chip">Capacity-gap indicators</span>
            <span class="chip">Workflow instability</span>
            <span class="chip">Locum / contract flags</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("POC Limitation")

    st.markdown(
        """
        <div class="light-box">
            <div class="hero-text">
                The current model uses synthetic / staging-shaped data and pseudo-risk labels.
                The evaluation validates the ML pipeline and dashboard behaviour, not production accuracy.
                The next step is to use a real staging extract and validate the highest-risk plans with domain experts.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Artifacts")

    st.markdown(
        """
        <div class="light-box">
            <div class="light-line">models/risk_model.joblib — active trained model</div>
            <div class="light-line">models/metadata.json — feature schema and training metadata</div>
            <div class="light-line">outputs/evaluation_report.json — active model evaluation</div>
            <div class="light-line">outputs/risk_scores.json — full scored output</div>
            <div class="light-line">outputs/high_risk_jobplans.json — focused integration handoff</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with tab_export:
    st.header("Export & Integration")

    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-title">Structured ML output for integration</div>
            <div class="hero-text">
                This section provides the JSON/CSV handoff for the wider analyser workflow.
                The payload is intentionally focused: JobPlan identity, Trust context, Department/Specialty,
                risk score, risk category, drivers, recommended action, and risk components.
                Stage is intentionally excluded.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cards(
        [
            ("Full Scored Output", f"{len(scored_df):,}", "All scored JobPlans"),
            ("High-Risk Handoff", f"{len(handoff_df):,}", "Focused review-prioritisation payload"),
            ("Export Format", "JSON / CSV", "Ready for demo or downstream service"),
            ("Decision Policy", "Review Only", "No automated approval or rejection"),
        ]
    )

    st.subheader("Recommended Export Files")

    st.markdown(
        """
        <div class="light-box">
            <div class="light-line">outputs/high_risk_jobplans.json</div>
            <div class="light-line">outputs/high_risk_jobplans.csv</div>
            <div class="light-line">outputs/risk_scores.json</div>
            <div class="light-line">outputs/risk_scores.csv</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dl1, dl2 = st.columns(2)

    with dl1:
        st.download_button(
            label="Download high_risk_jobplans.json",
            data=handoff_df.to_json(orient="records", indent=2).encode("utf-8"),
            file_name="high_risk_jobplans.json",
            mime="application/json",
        )

    with dl2:
        st.download_button(
            label="Download high_risk_jobplans.csv",
            data=handoff_df.to_csv(index=False).encode("utf-8"),
            file_name="high_risk_jobplans.csv",
            mime="text/csv",
        )

    st.subheader("Integration Flow")

    st.markdown(
        """
        <div class="light-box">
            <div class="flow-step">Rule engine findings</div>
            <div class="flow-arrow">+</div>
            <div class="flow-step">ML risk output</div>
            <div class="flow-arrow">↓</div>
            <div class="flow-step">LLM explanation</div>
            <div class="flow-arrow">↓</div>
            <div class="flow-step">Dashboard / stakeholder narrative</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "The LLM should explain computed facts from the rule engine and ML layer. "
        "It should not invent compliance numbers or create risk scores by itself."
    )

    st.subheader("Demo API Endpoints")

    st.markdown(
        """
        <div class="light-box">
            <div class="endpoint-row"><span class="method-badge">GET</span><span class="endpoint-path">/health</span></div>
            <div class="endpoint-row"><span class="method-badge">GET</span><span class="endpoint-path">/api/v1/jobplans/{jobPlanCode}/analysis</span></div>
            <div class="endpoint-row"><span class="method-badge">GET</span><span class="endpoint-path">/api/v1/departments/{trustLevelCode}/summary</span></div>
            <div class="endpoint-row"><span class="method-badge">POST</span><span class="endpoint-path">/api/v1/analysis/batch</span></div>
            <div class="endpoint-row"><span class="method-badge">POST</span><span class="endpoint-path">/api/v1/scenarios/simulate</span></div>
            <div class="endpoint-row"><span class="method-badge">POST</span><span class="endpoint-path">/score</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Example Handoff JSON")

    if len(handoff_df) > 0:
        sample = handoff_df.iloc[0]
        example = example_handoff_json(sample)
        example_json = json.dumps(example, indent=2, default=str)

        st.text_area(
            "Example Handoff JSON",
            value=example_json,
            height=420,
        )
    else:
        st.warning("No High-Risk JobPlans are currently available for export.")

    st.subheader("What this output is for")

    st.markdown(
        """
        <div class="light-box">
            <span class="chip">Product and stakeholder demo</span>
            <span class="chip">Rule-engine + ML integration</span>
            <span class="chip">LLM explanation input</span>
            <span class="chip">Highest-risk review queue</span>
            <span class="chip">Evidence-based prioritisation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
