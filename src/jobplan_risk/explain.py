from __future__ import annotations

import pandas as pd


def rule_risk_component(row: pd.Series) -> float:
    score = 0.0

    if row.get("paLimitBreach", 0) == 1:
        score += 25

    if row.get("deltaSpaPAs", 0) > 0.5:
        score += min(20, row["deltaSpaPAs"] * 16)

    if row.get("deltaDccPAs", 0) < -0.5:
        score += min(15, abs(row["deltaDccPAs"]) * 12)

    if row.get("teamPlanAlignmentScore", 1) < 0.8:
        score += min(20, (0.8 - row["teamPlanAlignmentScore"]) * 100)

    if row.get("historyChangeCount", 0) >= 10:
        score += min(12, row["historyChangeCount"] * 0.75)

    if row.get("hasMediationOrAppeal", 0) == 1:
        score += 18

    if row.get("planReturnedToDiscussionAfterSignoff", 0) == 1:
        score += 15

    if row.get("isLocum", 0) == 1:
        score += 5

    if row.get("missingTeamPlanLink", 0) == 1:
        score += 8

    return float(max(0, min(100, score)))


def risk_category(score: float) -> str:
    if score >= 75:
        return "High"
    if score >= 50:
        return "Medium"
    return "Low"


def top_drivers(row: pd.Series) -> list[str]:
    drivers = []

    if row.get("paLimitBreach", 0) == 1:
        drivers.append("PA limit breach detected against configured trust limits.")

    if row.get("deltaSpaPAs", 0) > 0.5:
        drivers.append(f"SPA increased by {row['deltaSpaPAs']:.2f} PA compared with prior plan.")

    if row.get("deltaDccPAs", 0) < -0.5:
        drivers.append(f"DCC reduced by {abs(row['deltaDccPAs']):.2f} PA compared with prior plan.")

    if row.get("teamPlanAlignmentScore", 1) < 0.8:
        drivers.append(f"Team-plan alignment score is low ({row['teamPlanAlignmentScore']:.2f}).")

    if row.get("historyChangeCount", 0) >= 10:
        drivers.append(f"High history change count detected ({int(row['historyChangeCount'])}).")

    if row.get("hasMediationOrAppeal", 0) == 1:
        drivers.append("Historical mediation or appeal signal exists.")

    if row.get("planReturnedToDiscussionAfterSignoff", 0) == 1:
        drivers.append("Plan returned to Discussion after reaching Sign-off.")

    if row.get("deviationFromPeerMedianSpa", 0) > 0.05:
        drivers.append(
            f"SPA share is {row['deviationFromPeerMedianSpa'] * 100:.1f} percentage points above peer median."
        )

    if row.get("missingTeamPlanLink", 0) == 1:
        drivers.append("Missing linked team plan.")

    return drivers[:3] if drivers else ["No major risk driver detected from available POC features."]
