from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from jobplan_risk.features import MODEL_FEATURES, prepare_features
from jobplan_risk.labels import build_pseudo_labels
from jobplan_risk.score import score
from jobplan_risk.train import build_preprocessor, train


DATA_PATH = Path("data/staging_plan_features.csv")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

GREEN_SEQUENCE = [
    "#063B2E",
    "#0B6F4F",
    "#009B77",
    "#59C29E",
    "#A6E3CC",
    "#DDF4EA",
]

RISK_COLOR_MAP = {
    "High": "#B3261E",
    "Medium": "#B06000",
    "Low": "#0B6F4F",
}


def apply_plotly_theme(fig):
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#102A21", size=13),
        title=dict(font=dict(color="#063B2E", size=18)),
        legend=dict(
            font=dict(color="#102A21"),
            bgcolor="rgba(255,255,255,0)",
        ),
        margin=dict(l=20, r=20, t=55, b=30),
        colorway=GREEN_SEQUENCE,
    )
    fig.update_xaxes(
        color="#102A21",
        gridcolor="#E6F4EE",
        linecolor="#CFE7DC",
        zerolinecolor="#CFE7DC",
    )
    fig.update_yaxes(
        color="#102A21",
        gridcolor="#E6F4EE",
        linecolor="#CFE7DC",
        zerolinecolor="#CFE7DC",
    )
    return fig


def plot_green_bar(series, title, value_label="Value", horizontal=False):
    data = series.reset_index()
    data.columns = ["Category", value_label]

    fig = px.bar(
        data,
        x=value_label if horizontal else "Category",
        y="Category" if horizontal else value_label,
        orientation="h" if horizontal else "v",
        color_discrete_sequence=["#0B6F4F"],
        text=value_label,
    )
    fig.update_traces(
        marker_color="#0B6F4F",
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>" if horizontal else "<b>%{x}</b><br>%{y}<extra></extra>",
    )
    fig.update_layout(title=title, showlegend=False)
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


def plot_risk_distribution(series, title="Risk Category Distribution"):
    order = ["High", "Medium", "Low"]
    data = series.reindex(order).fillna(0).reset_index()
    data.columns = ["Risk Category", "Plans"]

    fig = px.bar(
        data,
        x="Risk Category",
        y="Plans",
        color="Risk Category",
        color_discrete_map=RISK_COLOR_MAP,
        text="Plans",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(title=title, showlegend=False)
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


def plot_green_pie(series, title):
    data = series.reset_index()
    data.columns = ["Category", "Value"]

    fig = px.pie(
        data,
        names="Category",
        values="Value",
        color_discrete_sequence=GREEN_SEQUENCE,
        hole=0.45,
    )
    fig.update_traces(
        textinfo="label+percent",
        textfont_color="#102A21",
        marker=dict(line=dict(color="#FFFFFF", width=2)),
    )
    fig.update_layout(title=title)
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)



st.set_page_config(
    page_title="AI JobPlan Risk Dashboard",
    page_icon="🚦",
    layout="wide",
)


st.markdown(
    """
    <style>
    :root {
        --rld-green-dark: #063B2E;
        --rld-green: #0B6F4F;
        --rld-green-bright: #009B77;
        --rld-green-soft: #E6F4EE;
        --rld-green-pale: #F6FAF8;
        --rld-border: #CFE7DC;
        --rld-text: #102A21;
        --rld-muted: #52635C;
        --rld-white: #FFFFFF;
        --rld-red: #B3261E;
        --rld-amber: #B06000;
        --rld-shadow: rgba(6, 59, 46, 0.12);
    }

    html, body, .stApp {
        background: var(--rld-green-pale) !important;
        color: var(--rld-text) !important;
    }

    .block-container {
        padding-top: 1.4rem !important;
        padding-bottom: 3rem !important;
    }

    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: var(--rld-text);
    }

    .brand-header {
        padding: 22px 26px;
        border-radius: 22px;
        background: linear-gradient(135deg, var(--rld-green-dark) 0%, var(--rld-green) 52%, var(--rld-green-bright) 100%);
        color: var(--rld-white) !important;
        margin-bottom: 22px;
        box-shadow: 0 10px 28px var(--rld-shadow);
    }

    .brand-header * {
        color: var(--rld-white) !important;
    }

    .brand-logo {
        font-size: 18px;
        font-weight: 850;
        letter-spacing: .02em;
        margin-bottom: 8px;
    }

    .brand-title {
        font-size: 36px;
        font-weight: 900;
        line-height: 1.1;
        margin-bottom: 6px;
    }

    .brand-subtitle {
        font-size: 17px;
        opacity: 0.94;
    }

    .hero-box {
        padding: 22px;
        border-radius: 20px;
        background: var(--rld-white);
        color: var(--rld-text) !important;
        border: 1px solid var(--rld-border);
        margin-bottom: 18px;
        box-shadow: 0 5px 18px var(--rld-shadow);
    }

    .hero-box * {
        color: var(--rld-text) !important;
    }

    .section-box {
        padding: 18px;
        border-radius: 18px;
        background: var(--rld-white);
        color: var(--rld-text) !important;
        border: 1px solid var(--rld-border);
        box-shadow: 0 4px 14px var(--rld-shadow);
        margin-bottom: 16px;
    }

    .risk-card {
        padding: 20px;
        border-radius: 18px;
        border: 1px solid var(--rld-border);
        background: var(--rld-white);
        color: var(--rld-text) !important;
        box-shadow: 0 5px 18px var(--rld-shadow);
        margin-bottom: 16px;
    }

    .risk-card * {
        color: var(--rld-text) !important;
    }

    .risk-high {
        border-left: 12px solid var(--rld-red);
    }

    .risk-medium {
        border-left: 12px solid var(--rld-amber);
    }

    .risk-low {
        border-left: 12px solid var(--rld-green-bright);
    }

    .big-number {
        font-size: 36px;
        font-weight: 900;
        color: var(--rld-green-dark) !important;
    }

    .small-label {
        font-size: 12px;
        color: var(--rld-muted) !important;
        text-transform: uppercase;
        letter-spacing: .05em;
        font-weight: 700;
    }

    .driver {
        font-size: 14px;
        margin-top: 6px;
        color: var(--rld-text) !important;
    }

    .nhs-action {
        font-size: 15px;
        font-weight: 750;
        margin-top: 10px;
        color: var(--rld-green-dark) !important;
    }

    .urgent {
        color: var(--rld-red) !important;
        font-weight: 900;
    }

    .medium {
        color: var(--rld-amber) !important;
        font-weight: 900;
    }

    .good {
        color: var(--rld-green) !important;
        font-weight: 900;
    }

    .note-box {
        padding: 16px;
        border-radius: 16px;
        background: var(--rld-green-soft);
        color: var(--rld-text) !important;
        border: 1px solid var(--rld-border);
    }

    .note-box * {
        color: var(--rld-text) !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: var(--rld-white);
        border: 1px solid var(--rld-border);
        border-radius: 18px;
        padding: 16px;
        box-shadow: 0 4px 14px var(--rld-shadow);
    }

    [data-testid="stMetric"] * {
        color: var(--rld-text) !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--rld-green-dark) !important;
        font-weight: 900 !important;
    }

    [data-testid="stMetricDelta"] {
        color: var(--rld-green) !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        background: var(--rld-white) !important;
        border-radius: 14px 14px 0 0 !important;
        border: 1px solid var(--rld-border) !important;
        margin-right: 4px !important;
        padding: 10px 16px !important;
    }

    button[data-baseweb="tab"] p {
        color: var(--rld-green-dark) !important;
        font-weight: 800 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: var(--rld-green) !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] p {
        color: var(--rld-white) !important;
    }

    /* Sidebar and filters */
    section[data-testid="stSidebar"] {
        background: var(--rld-green-soft) !important;
        color: var(--rld-text) !important;
    }

    section[data-testid="stSidebar"] * {
        color: var(--rld-text) !important;
    }

    /* Inputs/selects */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    textarea {
        background: var(--rld-white) !important;
        color: var(--rld-text) !important;
        border-color: var(--rld-border) !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div,
    input,
    textarea {
        color: var(--rld-text) !important;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        background: var(--rld-white) !important;
        border: 1px solid var(--rld-border) !important;
        border-radius: 14px !important;
        padding: 6px !important;
    }

    /* Download buttons and standard buttons */
    .stDownloadButton button,
    .stButton button {
        background: var(--rld-green) !important;
        color: var(--rld-white) !important;
        border-radius: 12px !important;
        border: 1px solid var(--rld-green-dark) !important;
        font-weight: 800 !important;
    }

    .stDownloadButton button:hover,
    .stButton button:hover {
        background: var(--rld-green-dark) !important;
        color: var(--rld-white) !important;
    }

    /* Info/success/warning boxes readability */
    [data-testid="stAlert"] {
        background: var(--rld-white) !important;
        color: var(--rld-text) !important;
        border: 1px solid var(--rld-border) !important;
        border-radius: 14px !important;
    }

    [data-testid="stAlert"] * {
        color: var(--rld-text) !important;
    }

    hr {
        border-color: var(--rld-border) !important;
    }
    
    /* Force filters/multiselect chips into RLDatix green instead of red */
    [data-baseweb="tag"] {
        background-color: #0B6F4F !important;
        border: 1px solid #063B2E !important;
        color: #FFFFFF !important;
    }

    [data-baseweb="tag"] * {
        color: #FFFFFF !important;
    }

    [data-baseweb="tag"] svg {
        fill: #FFFFFF !important;
    }

    /* Dropdown selected values */
    div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        color: #102A21 !important;
    }

    div[data-baseweb="select"] * {
        color: #102A21 !important;
    }

    /* Checkbox and slider accent colours */
    input[type="checkbox"]:checked {
        accent-color: #0B6F4F !important;
    }

    .stSlider [data-baseweb="slider"] div {
        color: #0B6F4F !important;
    }

    .stSlider [role="slider"] {
        background-color: #0B6F4F !important;
        border-color: #063B2E !important;
    }

    /* Plotly chart surface */
    .js-plotly-plot,
    .plot-container,
    .svg-container {
        background: #FFFFFF !important;
        color: #102A21 !important;
        border-radius: 16px !important;
    }

    .js-plotly-plot text {
        fill: #102A21 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


def safe_metric(fn, y_true, y_pred_or_score):
    try:
        return float(fn(y_true, y_pred_or_score))
    except Exception:
        return None


def as_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def as_num_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def evaluate_models_cached(df: pd.DataFrame) -> dict:
    features = prepare_features(df)
    y = build_pseudo_labels(features)
    X = features[MODEL_FEATURES]

    if "planning_year" in df.columns and df["planning_year"].nunique() >= 2:
        max_year = df["planning_year"].max()
        train_idx = df["planning_year"] < max_year
        test_idx = df["planning_year"] == max_year
        split_strategy = f"time_based_train_before_{max_year}_test_{max_year}"
    else:
        train_positions, test_positions = train_test_split(
            np.arange(len(df)),
            test_size=0.25,
            random_state=42,
            stratify=y if y.nunique() == 2 else None,
        )
        train_idx = df.index.isin(train_positions)
        test_idx = df.index.isin(test_positions)
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

    report = {
        "splitStrategy": split_strategy,
        "rowsTotal": int(len(df)),
        "rowsTrain": int(len(X_train)),
        "rowsTest": int(len(X_test)),
        "positiveRateTrain": float(y_train.mean()),
        "positiveRateTest": float(y_test.mean()),
        "models": {},
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)

        report["models"][name] = {
            "accuracy": safe_metric(accuracy_score, y_test, y_pred),
            "precision": safe_metric(lambda a, b: precision_score(a, b, zero_division=0), y_test, y_pred),
            "recall": safe_metric(lambda a, b: recall_score(a, b, zero_division=0), y_test, y_pred),
            "f1": safe_metric(lambda a, b: f1_score(a, b, zero_division=0), y_test, y_pred),
            "rocAuc": safe_metric(roc_auc_score, y_test, y_score),
            "averagePrecision": safe_metric(average_precision_score, y_test, y_score),
            "confusionMatrix": confusion_matrix(y_test, y_pred).tolist(),
            "classificationReport": classification_report(
                y_test,
                y_pred,
                zero_division=0,
                output_dict=True,
            ),
        }

    return report


@st.cache_data(show_spinner=False)
def score_cached(df: pd.DataFrame) -> pd.DataFrame:
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
        "cpPAs",
        "otherPAs",
        "priorTotalPAs",
        "priorDccPAs",
        "priorSpaPAs",
        "peerMedianSpaShare",
        "peerMedianDccShare",
        "paLimitBreach",
        "spaAbovePeerThreshold",
        "hasTeamPlanLink",
        "missingTeamPlanLink",
        "teamPlanAlignmentScore",
        "teamDemandPAs",
        "teamPlannedCapacityPAs",
        "historyChangeCount",
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
            lambda x: " | ".join(x) if isinstance(x, list) else str(x)
        )

    scored_df["riskBandSort"] = scored_df["riskCategory"].map(
        {"High": 3, "Medium": 2, "Low": 1}
    ).fillna(0)

    return scored_df


def risk_badge(category: str) -> str:
    if category == "High":
        return "🔴 High"
    if category == "Medium":
        return "🟠 Medium"
    return "🟢 Low"


def action_recommendation(row: pd.Series) -> str:
    drivers = row.get("mainDriversText", "")
    score_value = float(row.get("riskScore", 0))

    if score_value >= 85:
        return "Immediate Clinical Director review"
    if "team-plan alignment" in str(drivers).lower() or "alignment" in str(drivers).lower():
        return "Review service demand/capacity alignment"
    if "spa" in str(drivers).lower():
        return "Review SPA/DCC balance with clinician"
    if "history" in str(drivers).lower() or "returned" in str(drivers).lower():
        return "Review workflow instability and plan changes"
    if score_value >= 70:
        return "Prioritise in next job-planning review"
    return "Monitor"


def risk_card(row: pd.Series) -> None:
    category = row.get("riskCategory", "Low")
    css = "risk-high" if category == "High" else "risk-medium" if category == "Medium" else "risk-low"

    drivers = row.get("mainDrivers", [])
    if not isinstance(drivers, list):
        drivers = [str(drivers)]

    driver_html = "".join([f"<div class='driver'>• {d}</div>" for d in drivers[:3]])

    st.markdown(
        f"""
        <div class="risk-card {css}">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div class="small-label">JobPlan</div>
                    <div style="font-size:22px; font-weight:800;">{row.get("jobPlanCode", "")}</div>
                    <div style="margin-top:6px;">{row.get("department", "N/A")} / {row.get("specialty", "N/A")}</div>
                    <div style="color:#666;">Stage: {row.get("plan_stage", "N/A")} | Trust: {row.get("trust_level_code", "N/A")}</div>
                </div>
                <div style="text-align:right;">
                    <div class="small-label">Risk Score</div>
                    <div class="big-number">{row.get("riskScore", 0):.1f}</div>
                    <div>{risk_badge(category)}</div>
                </div>
            </div>
            <hr/>
            <div class="small-label">Main Risk Drivers</div>
            {driver_html}
            <div class="nhs-action">Recommended action: {action_recommendation(row)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def driver_summary(scored_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in scored_df.iterrows():
        drivers = row.get("mainDrivers", [])
        if not isinstance(drivers, list):
            continue
        for driver in drivers:
            rows.append(
                {
                    "driver": driver,
                    "riskCategory": row.get("riskCategory"),
                    "riskScore": row.get("riskScore"),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["driver", "count", "avgRiskScore", "highRiskCount"])

    d = pd.DataFrame(rows)
    return (
        d.groupby("driver")
        .agg(
            count=("driver", "size"),
            avgRiskScore=("riskScore", "mean"),
            highRiskCount=("riskCategory", lambda x: int((x == "High").sum())),
        )
        .reset_index()
        .sort_values(["highRiskCount", "count", "avgRiskScore"], ascending=False)
    )


def build_hotspots(scored_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in ["trust_level_code", "department", "specialty"] if c in scored_df.columns]

    if not group_cols:
        return pd.DataFrame()

    grouped = (
        scored_df.groupby(group_cols)
        .agg(
            totalPlans=("jobPlanCode", "count"),
            highRiskPlans=("riskCategory", lambda x: int((x == "High").sum())),
            mediumRiskPlans=("riskCategory", lambda x: int((x == "Medium").sum())),
            avgRiskScore=("riskScore", "mean"),
            maxRiskScore=("riskScore", "max"),
            avgTeamAlignment=("teamPlanAlignmentScore", "mean") if "teamPlanAlignmentScore" in scored_df.columns else ("riskScore", "mean"),
            avgTotalPAs=("totalPAs", "mean") if "totalPAs" in scored_df.columns else ("riskScore", "mean"),
        )
        .reset_index()
    )

    grouped["highRiskRate"] = grouped["highRiskPlans"] / grouped["totalPlans"]
    grouped["operationalPriorityScore"] = (
        grouped["highRiskPlans"] * 3
        + grouped["mediumRiskPlans"] * 1.5
        + grouped["avgRiskScore"] / 20
    )

    return grouped.sort_values(
        ["operationalPriorityScore", "highRiskPlans", "avgRiskScore"],
        ascending=False,
    )


def make_action_queue(scored_df: pd.DataFrame) -> pd.DataFrame:
    queue = scored_df.copy()
    queue["recommendedAction"] = queue.apply(action_recommendation, axis=1)

    if "teamPlanAlignmentScore" in queue.columns:
        queue["teamPlanAlignmentScore"] = pd.to_numeric(queue["teamPlanAlignmentScore"], errors="coerce")

    columns = [
        "jobPlanCode",
        "riskScore",
        "riskCategory",
        "recommendedAction",
        "trust_level_code",
        "department",
        "specialty",
        "plan_stage",
        "totalPAs",
        "dccPAs",
        "spaPAs",
        "priorSpaPAs",
        "teamPlanAlignmentScore",
        "historyChangeCount",
        "mainDriversText",
    ]

    available = [c for c in columns if c in queue.columns]

    return queue.sort_values(
        ["riskBandSort", "riskScore"],
        ascending=False,
    )[available]


st.markdown(
    """
    <div class="brand-header">
        <div class="brand-logo">RLDatix • Workforce Intelligence</div>
        <div class="brand-title">AI JobPlan Risk Dashboard</div>
        <div class="brand-subtitle">
            Green-theme stakeholder view for NHS-style job-planning risk prioritisation
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='subtitle'>Stakeholder view for NHS-style job-planning risk prioritisation</div>",
    unsafe_allow_html=True,
)

if not DATA_PATH.exists():
    st.error("Dataset not found. Expected file: data/staging_plan_features.csv")
    st.stop()

OUTPUT_DIR.mkdir(exist_ok=True)

with st.spinner("Loading dataset..."):
    df = load_data(str(DATA_PATH))

with st.spinner("Training model..."):
    training_metadata = train(DATA_PATH, MODEL_DIR)

with st.spinner("Evaluating model..."):
    evaluation_report = evaluate_models_cached(df)

with st.spinner("Scoring JobPlans..."):
    scored_df = score_cached(df)

training_metadata_path = OUTPUT_DIR / "training_metadata.json"
evaluation_report_path = OUTPUT_DIR / "evaluation_report.json"
risk_scores_csv_path = OUTPUT_DIR / "risk_scores.csv"
risk_scores_json_path = OUTPUT_DIR / "risk_scores.json"

training_metadata_path.write_text(json.dumps(training_metadata, indent=2), encoding="utf-8")
evaluation_report_path.write_text(json.dumps(evaluation_report, indent=2), encoding="utf-8")
scored_df.to_csv(risk_scores_csv_path, index=False)
risk_scores_json_path.write_text(scored_df.to_json(orient="records", indent=2), encoding="utf-8")

tab_risk, tab_high_risk, tab_eval, tab_train, tab_data, tab_export = st.tabs(
    [
        "🚦 Risk Scores",
        "🔥 Highest-Risk JobPlans",
        "📈 Evaluation",
        "🧠 Training Details",
        "📄 Dataset",
        "🔌 Export / Integration",
    ]
)


with tab_risk:
    st.header("🚦 Executive JobPlan Risk View")

    total_plans = len(scored_df)
    high_count = int((scored_df["riskCategory"] == "High").sum())
    medium_count = int((scored_df["riskCategory"] == "Medium").sum())
    low_count = int((scored_df["riskCategory"] == "Low").sum())
    high_rate = high_count / total_plans if total_plans else 0
    avg_score = float(scored_df["riskScore"].mean())
    max_score = float(scored_df["riskScore"].max())

    hotspots = build_hotspots(scored_df)
    top_hotspot = hotspots.iloc[0] if len(hotspots) > 0 else None

    if top_hotspot is not None:
        hotspot_text = (
            f"{top_hotspot.get('specialty', 'N/A')} "
            f"({top_hotspot.get('trust_level_code', 'N/A')})"
        )
    else:
        hotspot_text = "N/A"

    st.markdown(
        f"""
        <div class="hero-box">
            <div style="font-size:22px; font-weight:800; margin-bottom:8px;">
                Executive Summary
            </div>
            <div style="font-size:16px;">
                The ML layer scored <b>{total_plans:,}</b> JobPlans.
                <span class="urgent">{high_count:,}</span> are currently prioritised as High Risk
                and <span class="medium">{medium_count:,}</span> as Medium Risk.
                The main operational hotspot is <b>{hotspot_text}</b>.
            </div>
            <div style="margin-top:10px; color:#555;">
                This view is designed for Clinical Directors, Medical Staffing, Workforce Planning and Product stakeholders.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total JobPlans Scored", f"{total_plans:,}")
    k2.metric("High Risk Plans", f"{high_count:,}", f"{high_rate:.1%}")
    k3.metric("Medium Risk Plans", f"{medium_count:,}")
    k4.metric("Average Risk Score", f"{avg_score:.1f}")
    k5.metric("Highest Risk Score", f"{max_score:.1f}")

    st.divider()

    st.subheader("NHS Operational Risk Signals")

    signal_cols = st.columns(4)

    pa_breach_count = int(as_bool_series(scored_df["paLimitBreach"]).sum()) if "paLimitBreach" in scored_df.columns else 0

    if "teamPlanAlignmentScore" in scored_df.columns:
        alignment = as_num_series(scored_df["teamPlanAlignmentScore"])
        demand_capacity_count = int((alignment < 0.8).sum())
    else:
        demand_capacity_count = 0

    if {"spaPAs", "priorSpaPAs"}.issubset(scored_df.columns):
        spa_delta = as_num_series(scored_df["spaPAs"]) - as_num_series(scored_df["priorSpaPAs"])
        spa_drift_count = int((spa_delta > 0.5).sum())
    else:
        spa_drift_count = 0

    workflow_count = 0
    if "historyChangeCount" in scored_df.columns:
        workflow_count += int((as_num_series(scored_df["historyChangeCount"]) > 14).sum())
    if "planReturnedToDiscussionAfterSignoff" in scored_df.columns:
        workflow_count += int(as_bool_series(scored_df["planReturnedToDiscussionAfterSignoff"]).sum())

    signal_cols[0].metric("Compliance / PA Limit Risk", f"{pa_breach_count:,}")
    signal_cols[1].metric("Demand-Capacity Alignment Risk", f"{demand_capacity_count:,}")
    signal_cols[2].metric("SPA / DCC Balance Drift", f"{spa_drift_count:,}")
    signal_cols[3].metric("Workflow Instability Signals", f"{workflow_count:,}")

    st.divider()

    filter_col, main_col = st.columns([1, 3])

    with filter_col:
        st.markdown("### Filters")

        categories = st.multiselect(
            "Risk Category",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )

        if "department" in scored_df.columns:
            departments = sorted(scored_df["department"].dropna().unique().tolist())
            selected_departments = st.multiselect(
                "Department",
                departments,
                default=departments,
            )
        else:
            selected_departments = []

        if "specialty" in scored_df.columns:
            specialties = sorted(scored_df["specialty"].dropna().unique().tolist())
            selected_specialties = st.multiselect(
                "Specialty",
                specialties,
                default=specialties,
            )
        else:
            selected_specialties = []

        min_score = st.slider(
            "Minimum Risk Score",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
        )

    filtered = scored_df.copy()

    if categories:
        filtered = filtered[filtered["riskCategory"].isin(categories)]

    if selected_departments and "department" in filtered.columns:
        filtered = filtered[filtered["department"].isin(selected_departments)]

    if selected_specialties and "specialty" in filtered.columns:
        filtered = filtered[filtered["specialty"].isin(selected_specialties)]

    filtered = filtered[filtered["riskScore"] >= min_score]

    with main_col:
        st.markdown("### Risk Distribution and Service Hotspots")

        dist_col, specialty_col = st.columns(2)

        with dist_col:
            st.markdown("#### Risk Category Distribution")
            if len(filtered) > 0:
                risk_order = ["High", "Medium", "Low"]
                risk_counts = filtered["riskCategory"].value_counts().reindex(risk_order).fillna(0)
                plot_risk_distribution(risk_counts)
            else:
                st.info("No data for selected filters.")

        with specialty_col:
            st.markdown("#### Average Risk by Specialty")
            if "specialty" in filtered.columns and len(filtered) > 0:
                avg_by_specialty = (
                    filtered.groupby("specialty")["riskScore"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(10)
                )
                plot_green_bar(avg_by_specialty, "Average Risk by Specialty", "Average risk score", horizontal=True)
            else:
                st.info("No specialty data available.")

    st.divider()

    st.subheader("Top 5 Trust / Specialty Hotspots")

    filtered_hotspots = build_hotspots(filtered)

    if len(filtered_hotspots) > 0:
        top5_hotspots = filtered_hotspots.head(5).copy()

        visible_hotspot_cols = [
            "trust_level_code",
            "department",
            "specialty",
            "totalPlans",
            "highRiskPlans",
            "mediumRiskPlans",
            "highRiskRate",
            "avgRiskScore",
            "maxRiskScore",
            "avgTeamAlignment",
            "operationalPriorityScore",
        ]

        visible_hotspot_cols = [c for c in visible_hotspot_cols if c in top5_hotspots.columns]

        top5_hotspots["highRiskRate"] = top5_hotspots["highRiskRate"].round(3)
        top5_hotspots["avgRiskScore"] = top5_hotspots["avgRiskScore"].round(1)
        top5_hotspots["maxRiskScore"] = top5_hotspots["maxRiskScore"].round(1)
        top5_hotspots["avgTeamAlignment"] = top5_hotspots["avgTeamAlignment"].round(3)
        top5_hotspots["operationalPriorityScore"] = top5_hotspots["operationalPriorityScore"].round(1)

        st.dataframe(
            top5_hotspots[visible_hotspot_cols],
            use_container_width=True,
            height=245,
        )

        chart_hotspots = top5_hotspots.copy()
        chart_hotspots["group"] = (
            chart_hotspots["specialty"].astype(str)
            + " / "
            + chart_hotspots["trust_level_code"].astype(str)
        )

        st.markdown("#### High-Risk Plan Count by Hotspot")
        plot_green_bar(chart_hotspots.set_index("group")["highRiskPlans"], "High-Risk Plan Count by Hotspot", "High risk plans", horizontal=True)
    else:
        st.info("No hotspot data available for selected filters.")

    st.divider()

    st.subheader("Top Risk Drivers Across Selected JobPlans")

    drivers = driver_summary(filtered)

    if len(drivers) > 0:
        driver_col1, driver_col2 = st.columns([2, 1])

        with driver_col1:
            st.dataframe(
                drivers.head(10),
                use_container_width=True,
                height=320,
            )

        with driver_col2:
            st.markdown("#### Driver Frequency")
            driver_chart = drivers.head(8).set_index("driver")["count"]
            plot_green_bar(driver_chart, "Driver Frequency", "Count", horizontal=True)
    else:
        st.info("No risk drivers available.")

    st.divider()

    st.subheader("Immediate Review Queue")

    review_queue = make_action_queue(filtered).head(25)

    st.dataframe(
        review_queue,
        use_container_width=True,
        height=420,
    )

    st.divider()

    st.subheader("Top 5 Highest-Risk JobPlans")

    top5 = filtered.sort_values("riskScore", ascending=False).head(5)

    if len(top5) == 0:
        st.warning("No JobPlans match the selected filters.")
    else:
        for _, row in top5.iterrows():
            risk_card(row)

    st.divider()

    st.subheader("All Risk Scores")

    display_columns = [
        "jobPlanCode",
        "riskScore",
        "riskCategory",
        "trust_level_code",
        "department",
        "specialty",
        "planning_year",
        "plan_stage",
        "totalPAs",
        "dccPAs",
        "spaPAs",
        "priorSpaPAs",
        "teamPlanAlignmentScore",
        "historyChangeCount",
        "dataConfidence",
        "modelMode",
        "mainDriversText",
    ]

    available_display_columns = [c for c in display_columns if c in filtered.columns]

    st.dataframe(
        filtered.sort_values("riskScore", ascending=False)[available_display_columns],
        use_container_width=True,
        height=520,
    )

    st.download_button(
        label="Download filtered risk scores CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_risk_scores.csv",
        mime="text/csv",
    )


with tab_high_risk:
    st.header("🔥 Highest-Risk JobPlans")

    high_risk_df = scored_df[scored_df["riskCategory"] == "High"].copy()
    high_risk_df = high_risk_df.sort_values("riskScore", ascending=False).reset_index(drop=True)

    OUTPUT_DIR.mkdir(exist_ok=True)
    high_risk_df.to_csv(OUTPUT_DIR / "high_risk_jobplans.csv", index=False)
    (OUTPUT_DIR / "high_risk_jobplans.json").write_text(
        high_risk_df.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    if len(high_risk_df) == 0:
        st.success("No High Risk JobPlans found.")
    else:
        for _, row in high_risk_df.iterrows():
            risk_card(row)


with tab_eval:
    st.header("📈 Model Evaluation")

    st.info(
        "Evaluation proves that the ML pipeline can learn the pseudo-risk labels. "
        "For production, the same evaluation must be repeated on real staging/historical data."
    )

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Rows Total", f"{evaluation_report['rowsTotal']:,}")
    e2.metric("Train Rows", f"{evaluation_report['rowsTrain']:,}")
    e3.metric("Test Rows", f"{evaluation_report['rowsTest']:,}")
    e4.metric("Test Positive Rate", f"{evaluation_report['positiveRateTest']:.3f}")

    st.write(f"Split strategy: `{evaluation_report['splitStrategy']}`")

    model_name = st.selectbox(
        "Select model",
        list(evaluation_report["models"].keys()),
        key="evaluation_model_select",
    )

    metrics = evaluation_report["models"][model_name]

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    m2.metric("Precision", f"{metrics['precision']:.3f}")
    m3.metric("Recall", f"{metrics['recall']:.3f}")
    m4.metric("F1", f"{metrics['f1']:.3f}")
    m5.metric("ROC-AUC", f"{metrics['rocAuc']:.3f}")
    m6.metric("Avg Precision", f"{metrics['averagePrecision']:.3f}")

    st.subheader("Confusion Matrix")

    cm = pd.DataFrame(
        metrics["confusionMatrix"],
        index=["Actual Low/Normal", "Actual At-Risk"],
        columns=["Predicted Low/Normal", "Predicted At-Risk"],
    )

    st.dataframe(cm, use_container_width=True)

    st.subheader("Metric Interpretation")

    st.markdown(
        """
        - **Accuracy**: overall correctness.
        - **Precision**: when the model flags risk, how often it is correct.
        - **Recall**: how many risky plans it catches.
        - **F1**: balance between precision and recall.
        - **ROC-AUC**: how well the model ranks risky plans above normal plans.
        - **Average Precision**: useful when risk cases are fewer than normal cases.
        """
    )

    with st.expander("Full evaluation JSON"):
        st.json(evaluation_report)


with tab_train:
    st.header("🧠 Training Details")

    st.markdown(
        """
        This tab is mainly for engineering/data science review.
        For stakeholders, the Risk Scores tab is the main value.
        """
    )

    st.json(training_metadata)

    st.subheader("Training Files Created")

    for file in [
        "models/risk_model.joblib",
        "models/metadata.json",
        "outputs/training_metadata.json",
    ]:
        path = Path(file)
        if path.exists():
            st.success(f"{file} exists")
        else:
            st.warning(f"{file} not found")

    st.subheader("POC Model Logic")

    st.markdown(
        """
        The model combines:
        1. **ML component** — learned pattern from PA mix, trends, peer context, workflow and TJP signals.
        2. **Rule component** — deterministic risk factors such as PA limit breach, SPA increase, team-plan misalignment, mediation/appeal and returned-to-discussion signals.
        3. **Final risk score** — blended into a 0–100 score with High / Medium / Low category.
        """
    )


with tab_data:
    st.header("📄 Dataset Overview")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Rows", f"{len(df):,}")
    d2.metric("Columns", f"{len(df.columns):,}")
    d3.metric("Planning Years", df["planning_year"].nunique() if "planning_year" in df.columns else "N/A")
    d4.metric("Specialties", df["specialty"].nunique() if "specialty" in df.columns else "N/A")

    left_data, right_data = st.columns(2)

    with left_data:
        st.subheader("Plans by Planning Year")
        if "planning_year" in df.columns:
            plot_green_bar(df["planning_year"].value_counts().sort_index(), "Plans by Planning Year", "Plans")

    with right_data:
        st.subheader("Plans by Specialty")
        if "specialty" in df.columns:
            plot_green_bar(df["specialty"].value_counts().head(20), "Plans by Specialty", "Plans", horizontal=True)

    st.subheader("Input Feature Table Preview")
    st.dataframe(df.head(100), use_container_width=True, height=500)

    with st.expander("Dataset columns"):
        st.write(list(df.columns))


with tab_export:
    st.header("🔌 Export / Integration")

    st.markdown(
        """
        The risk scores are structured outputs and can be consumed by:
        - Bedrock narrative layer
        - n8n workflow
        - rule engine
        - demo UI
        - future JobPlan analyser service
        """
    )

    st.subheader("Generated Output Files")

    for path in [
        risk_scores_csv_path,
        risk_scores_json_path,
        evaluation_report_path,
        training_metadata_path,
    ]:
        if path.exists():
            st.success(str(path))
        else:
            st.warning(f"Missing: {path}")

    st.subheader("Integration Contract")

    st.code(
        """
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
        """,
        language="json",
    )

    st.download_button(
        label="Download all risk scores CSV",
        data=scored_df.to_csv(index=False).encode("utf-8"),
        file_name="risk_scores.csv",
        mime="text/csv",
    )

    st.download_button(
        label="Download evaluation report JSON",
        data=json.dumps(evaluation_report, indent=2).encode("utf-8"),
        file_name="evaluation_report.json",
        mime="application/json",
    )


st.markdown("---")
st.caption(
    "POC note: current data is synthetic/pseudo-labelled. "
    "This validates the ML pipeline, scoring flow and stakeholder dashboard. "
    "Real staging data is required before claiming production accuracy."
)
