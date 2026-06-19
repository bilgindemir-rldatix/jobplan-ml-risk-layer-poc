from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "pAsPerWeek",
    "paidPA",
    "minsPerWeek",
    "ntPerPA",
    "currentStageOrdinal",
    "daysInCurrentStage",
    "totalPAs",
    "dccPAs",
    "spaPAs",
    "cpPAs",
    "otherPAs",
    "dccShare",
    "spaShare",
    "deltaSpaPAs",
    "deltaDccPAs",
    "deltaTotalPAs",
    "deltaSpaShare",
    "yearsSinceLastPlan",
    "peerMedianSpaShare",
    "peerMedianDccShare",
    "peerGroupSize",
    "spaShareZScore",
    "dccShareZScore",
    "totalPAsZScore",
    "deviationFromPeerMedianSpa",
    "teamPlanAlignmentScore",
    "teamCapacityGapPAs",
    "teamCapacityGapShare",
    "historyChangeCount",
]

BINARY_FEATURES = [
    "isSignedUpToNewContract",
    "isAnnualised",
    "uses5thWeekPerQuarter",
    "hasTeamPlanLink",
    "isOnCall",
    "isShift",
    "isCPD",
    "paLimitBreach",
    "spaAbovePeerThreshold",
    "missingTeamPlanLink",
    "hasMediationOrAppeal",
    "hasNewManagerChanges",
    "planReturnedToDiscussionAfterSignoff",
    "isLocum",
    "isConsultant",
]

CATEGORICAL_FEATURES = [
    "workType",
    "doctorClassification",
    "gradeCode",
    "plan_stage",
]

MODEL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

CRITICAL_COLUMNS = [
    "job_plan_code",
    "totalPAs",
    "dccPAs",
    "spaPAs",
    "priorDccPAs",
    "priorSpaPAs",
    "hasTeamPlanLink",
    "paLimitBreach",
]


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace({0: np.nan})
    return numerator / denominator


def _to_bool_int(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"]).astype(int)


def prepare_features(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy()

    numeric_defaults = {
        "pAsPerWeek": np.nan,
        "paidPA": np.nan,
        "minsPerWeek": np.nan,
        "ntPerPA": np.nan,
        "currentStageOrdinal": np.nan,
        "daysInCurrentStage": 0,
        "totalPAs": np.nan,
        "dccPAs": np.nan,
        "spaPAs": np.nan,
        "cpPAs": 0,
        "otherPAs": 0,
        "priorTotalPAs": np.nan,
        "priorDccPAs": np.nan,
        "priorSpaPAs": np.nan,
        "yearsSinceLastPlan": np.nan,
        "peerMedianSpaShare": np.nan,
        "peerMedianDccShare": np.nan,
        "peerGroupSize": 0,
        "teamPlanAlignmentScore": np.nan,
        "teamDemandPAs": np.nan,
        "teamPlannedCapacityPAs": np.nan,
        "historyChangeCount": 0,
    }

    for col, default in numeric_defaults.items():
        if col not in data.columns:
            data[col] = default
        data[col] = pd.to_numeric(data[col], errors="coerce")

    for col in BINARY_FEATURES:
        if col not in data.columns:
            data[col] = False
        data[col] = _to_bool_int(data[col])

    for col in CATEGORICAL_FEATURES:
        if col not in data.columns:
            data[col] = "Unknown"
        data[col] = data[col].fillna("Unknown").astype(str)

    data["dccShare"] = _safe_div(data["dccPAs"], data["totalPAs"])
    data["spaShare"] = _safe_div(data["spaPAs"], data["totalPAs"])

    data["deltaSpaPAs"] = data["spaPAs"] - data["priorSpaPAs"]
    data["deltaDccPAs"] = data["dccPAs"] - data["priorDccPAs"]
    data["deltaTotalPAs"] = data["totalPAs"] - data["priorTotalPAs"]

    data["priorSpaShare"] = _safe_div(data["priorSpaPAs"], data["priorTotalPAs"])
    data["deltaSpaShare"] = data["spaShare"] - data["priorSpaShare"]

    data["deviationFromPeerMedianSpa"] = data["spaShare"] - data["peerMedianSpaShare"]

    if "trust_level_code" not in data.columns:
        data["trust_level_code"] = "Unknown"

    for source_col, z_col in [
        ("spaShare", "spaShareZScore"),
        ("dccShare", "dccShareZScore"),
        ("totalPAs", "totalPAsZScore"),
    ]:
        mean = data.groupby("trust_level_code")[source_col].transform("mean")
        std = data.groupby("trust_level_code")[source_col].transform("std").replace({0: np.nan})
        data[z_col] = ((data[source_col] - mean) / std).fillna(0)

    data["teamCapacityGapPAs"] = data["teamDemandPAs"] - data["teamPlannedCapacityPAs"]
    data["teamCapacityGapShare"] = _safe_div(data["teamCapacityGapPAs"], data["teamDemandPAs"])

    if "missingTeamPlanLink" not in raw.columns:
        data["missingTeamPlanLink"] = (data["hasTeamPlanLink"] == 0).astype(int)

    return data


def data_confidence(row: pd.Series) -> tuple[str, list[str]]:
    missing_critical = [c for c in CRITICAL_COLUMNS if c not in row.index or pd.isna(row[c])]

    optional = [
        "teamPlanAlignmentScore",
        "peerMedianSpaShare",
        "peerMedianDccShare",
        "historyChangeCount",
    ]

    missing_optional = [c for c in optional if c not in row.index or pd.isna(row[c])]

    if missing_critical:
        return "Low", missing_critical + missing_optional

    if len(missing_optional) >= 2:
        return "Medium", missing_optional

    return "High", missing_optional
