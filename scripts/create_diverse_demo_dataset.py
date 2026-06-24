from pathlib import Path
import hashlib
import random

import numpy as np
import pandas as pd


random.seed(42)
np.random.seed(42)

OUTPUT_PATH = Path("data/staging_plan_features.csv")
BACKUP_PATH = Path("data/backups/staging_plan_features_before_12000_demo.csv")

base = pd.read_csv(BACKUP_PATH)

DEPARTMENTS = [
    ("Medicine", "Oncology", "TLC-ONC", "TLC-MED"),
    ("Medicine", "Cardiology", "TLC-CARD", "TLC-MED"),
    ("Medicine", "Gastroenterology", "TLC-GASTRO", "TLC-MED"),
    ("Medicine", "Respiratory", "TLC-RESP", "TLC-MED"),
    ("Surgery", "Urology", "TLC-UROL", "TLC-SURG"),
    ("Surgery", "ENT", "TLC-ENT", "TLC-SURG"),
    ("Surgery", "Trauma & Orthopaedics", "TLC-TNO", "TLC-SURG"),
    ("Diagnostics", "Radiology", "TLC-RAD", "TLC-DIAG"),
    ("Women & Children", "Paediatrics", "TLC-PAED", "TLC-WC"),
    ("Emergency", "Emergency Medicine", "TLC-ED", "TLC-EMERG"),
]

STAGES = [
    ("Draft", 1),
    ("Discussion", 2),
    ("SignOff", 3),
    ("Approved", 4),
]

GRADES = ["CONS", "ASSOCSPEC", "SAS"]
WORK_TYPES = ["FullTime", "PartTime"]
DOCTOR_CLASSES = ["Consultant", "SAS", "Associate Specialist"]


def user_hash(i: int) -> str:
    return hashlib.sha256(f"demo-user-{i}".encode()).hexdigest()[:16].upper()


def clamp(value, low, high):
    return max(low, min(high, value))


def make_row(i: int, profile: str) -> dict:
    department, specialty, trust, parent = random.choice(DEPARTMENTS)
    stage, stage_ord = random.choice(STAGES)
    year = random.choice([2023, 2024, 2025, 2026])
    grade = random.choice(GRADES)
    work_type = random.choice(WORK_TYPES)
    doctor_class = "Consultant" if grade == "CONS" else random.choice(DOCTOR_CLASSES)

    # Keep total PAs realistic for consultant job planning.
    total = round(float(np.random.choice([8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5])), 2)
    prior_total = round(total + float(np.random.choice([-0.5, -0.25, 0, 0.25, 0.5])), 2)

    if profile == "Low":
        # Stable, balanced, good alignment, low workflow friction.
        prior_spa = round(float(np.random.uniform(1.6, 2.2)), 2)
        spa_delta = round(float(np.random.uniform(-0.20, 0.25)), 2)
        spa = clamp(round(prior_spa + spa_delta, 2), 1.2, 2.4)

        prior_dcc = round(float(np.random.uniform(6.8, 8.2)), 2)
        dcc_delta = round(float(np.random.uniform(-0.20, 0.25)), 2)
        dcc = clamp(round(prior_dcc + dcc_delta, 2), 6.4, 8.6)

        team_alignment = round(float(np.random.uniform(0.90, 0.99)), 2)
        pa_limit_breach = False
        spa_above_peer = False
        missing_team = False
        history_count = int(np.random.randint(0, 5))
        mediation = False
        returned = False
        locum = False
        days_stage = int(np.random.randint(1, 22))
        target_score = int(np.random.randint(12, 42))
        at_risk = 0

    elif profile == "Medium":
        # Moderate drift or alignment issue, but not severe.
        prior_spa = round(float(np.random.uniform(1.8, 2.4)), 2)
        spa_delta = round(float(np.random.uniform(0.30, 0.70)), 2)
        spa = clamp(round(prior_spa + spa_delta, 2), 2.2, 3.1)

        prior_dcc = round(float(np.random.uniform(6.8, 8.1)), 2)
        dcc_delta = round(float(np.random.uniform(-0.65, -0.20)), 2)
        dcc = clamp(round(prior_dcc + dcc_delta, 2), 5.9, 7.7)

        team_alignment = round(float(np.random.uniform(0.77, 0.88)), 2)
        pa_limit_breach = bool(np.random.choice([False, False, False, True]))
        spa_above_peer = bool(np.random.choice([False, True]))
        missing_team = bool(np.random.choice([False, False, True]))
        history_count = int(np.random.randint(5, 11))
        mediation = False
        returned = bool(np.random.choice([False, False, True]))
        locum = bool(np.random.choice([False, False, True]))
        days_stage = int(np.random.randint(18, 55))
        target_score = int(np.random.randint(50, 74))
        # Make some medium cases non-risk and some risk so the binary model does not collapse.
        at_risk = int(np.random.choice([0, 1], p=[0.45, 0.55]))

    else:
        # Strong operational signals: breach, SPA creep, reduced DCC, poor alignment.
        prior_spa = round(float(np.random.uniform(1.8, 2.5)), 2)
        spa_delta = round(float(np.random.uniform(0.85, 1.80)), 2)
        spa = clamp(round(prior_spa + spa_delta, 2), 3.0, 4.4)

        prior_dcc = round(float(np.random.uniform(7.0, 8.4)), 2)
        dcc_delta = round(float(np.random.uniform(-1.60, -0.75)), 2)
        dcc = clamp(round(prior_dcc + dcc_delta, 2), 5.0, 7.1)

        team_alignment = round(float(np.random.uniform(0.55, 0.75)), 2)
        pa_limit_breach = True
        spa_above_peer = True
        missing_team = bool(np.random.choice([False, True], p=[0.45, 0.55]))
        history_count = int(np.random.randint(11, 24))
        mediation = bool(np.random.choice([False, True], p=[0.55, 0.45]))
        returned = bool(np.random.choice([False, True], p=[0.45, 0.55]))
        locum = bool(np.random.choice([False, True], p=[0.50, 0.50]))
        days_stage = int(np.random.randint(45, 120))
        target_score = int(np.random.randint(78, 96))
        at_risk = 1

    cp = round(float(np.random.choice([0.0, 0.25, 0.5, 0.75])), 2)

    # Adjust total if needed so values stay coherent.
    minimum_total = dcc + spa + cp
    if minimum_total > total:
        total = round(minimum_total + float(np.random.choice([0.0, 0.25, 0.5])), 2)

    other = round(max(0.0, total - dcc - spa - cp), 2)

    prior_total = round(max(prior_total, prior_dcc + prior_spa + cp), 2)

    peer_spa_share = round(float(np.random.uniform(0.18, 0.25)), 2)
    peer_dcc_share = round(float(np.random.uniform(0.66, 0.78)), 2)

    demand = round(float(np.random.uniform(32, 78)), 1)
    planned_capacity = round(demand * team_alignment, 1)

    has_team_link = not missing_team

    return {
        "job_plan_code": f"JP-{year}-{i:05d}",
        "user_code_hash": user_hash(i),
        "trust_level_code": trust,
        "parent_trust_level_code": parent,
        "department": department,
        "specialty": specialty,
        "planning_year": year,
        "plan_stage": stage,
        "currentStageOrdinal": stage_ord,
        "daysInCurrentStage": days_stage,
        "pAsPerWeek": total,
        "paidPA": total,
        "minsPerWeek": int(total * 240),
        "ntPerPA": 240,
        "workType": work_type,
        "doctorClassification": doctor_class,
        "isSignedUpToNewContract": bool(np.random.choice([True, False], p=[0.82, 0.18])),
        "isAnnualised": bool(np.random.choice([True, False], p=[0.25, 0.75])),
        "uses5thWeekPerQuarter": bool(np.random.choice([True, False], p=[0.20, 0.80])),
        "hasTeamPlanLink": bool(has_team_link),
        "totalPAs": round(total, 2),
        "dccPAs": round(max(dcc, 0), 2),
        "spaPAs": round(max(spa, 0), 2),
        "cpPAs": round(cp, 2),
        "otherPAs": round(other, 2),
        "isOnCall": bool(np.random.choice([True, False], p=[0.28, 0.72])),
        "isShift": bool(np.random.choice([True, False], p=[0.17, 0.83])),
        "isCPD": bool(np.random.choice([True, False], p=[0.22, 0.78])),
        "priorTotalPAs": round(prior_total, 2),
        "priorDccPAs": round(max(prior_dcc, 0), 2),
        "priorSpaPAs": round(max(prior_spa, 0), 2),
        "yearsSinceLastPlan": int(np.random.choice([1, 1, 1, 2])),
        "peerMedianSpaShare": peer_spa_share,
        "peerMedianDccShare": peer_dcc_share,
        "peerGroupSize": int(np.random.randint(8, 38)),
        "paLimitBreach": bool(pa_limit_breach),
        "spaAbovePeerThreshold": bool(spa_above_peer),
        "missingTeamPlanLink": bool(missing_team),
        "teamPlanAlignmentScore": team_alignment,
        "teamDemandPAs": demand,
        "teamPlannedCapacityPAs": planned_capacity,
        "historyChangeCount": history_count,
        "hasMediationOrAppeal": bool(mediation),
        "hasNewManagerChanges": bool(np.random.choice([True, False], p=[0.20, 0.80])),
        "planReturnedToDiscussionAfterSignoff": bool(returned),
        "isLocum": bool(locum),
        "gradeCode": grade,
        "isConsultant": bool(doctor_class == "Consultant"),
        # Demo-only columns.
        "demoRiskProfile": profile,
        "targetRiskScore": target_score,
        "atRisk": at_risk,
    }


rows = []
i = 1

# More than 10,000 JobPlans with deliberate diversity.
# This gives a visible and credible Low / Medium / High spread.
profile_counts = {
    "Low": 4200,
    "Medium": 4200,
    "High": 3600,
}

for profile, count in profile_counts.items():
    for _ in range(count):
        rows.append(make_row(i, profile))
        i += 1

df = pd.DataFrame(rows)

# Preserve original column ordering where possible, append demo columns at the end.
original_cols = [c for c in base.columns if c in df.columns]
extra_cols = [c for c in df.columns if c not in original_cols]
df = df[original_cols + extra_cols]

df.to_csv(OUTPUT_PATH, index=False)

print(f"Created diverse demo dataset: {OUTPUT_PATH}")
print(f"Rows: {len(df):,}")
print("\nDemo risk profile distribution:")
print(df["demoRiskProfile"].value_counts())
print("\nAt-risk binary label distribution:")
print(df["atRisk"].value_counts())
