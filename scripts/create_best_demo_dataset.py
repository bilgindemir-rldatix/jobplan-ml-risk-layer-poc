from pathlib import Path
import hashlib
import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

OUTPUT_PATH = Path("data/staging_plan_features.csv")

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
    return hashlib.sha256(f"best-demo-user-{i}".encode()).hexdigest()[:16].upper()


def clamp(value, low, high):
    return max(low, min(high, value))


def make_row(i: int, profile: str) -> dict:
    department, specialty, trust, parent = random.choice(DEPARTMENTS)
    stage, stage_ord = random.choice(STAGES)
    year = random.choice([2023, 2024, 2025, 2026])
    grade = random.choice(GRADES)
    work_type = random.choice(WORK_TYPES)
    doctor_class = "Consultant" if grade == "CONS" else random.choice(DOCTOR_CLASSES)

    total = round(float(np.random.choice([8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0])), 2)

    if profile == "Low":
        prior_spa = round(float(np.random.uniform(1.6, 2.2)), 2)
        spa = clamp(round(prior_spa + float(np.random.uniform(-0.15, 0.25)), 2), 1.4, 2.4)

        prior_dcc = round(float(np.random.uniform(6.7, 8.2)), 2)
        dcc = clamp(round(prior_dcc + float(np.random.uniform(-0.20, 0.25)), 2), 6.4, 8.5)

        team_alignment = round(float(np.random.uniform(0.91, 0.99)), 2)
        pa_limit_breach = False
        spa_above_peer = False
        missing_team = False
        history_count = int(np.random.randint(0, 5))
        mediation = False
        returned = False
        locum = False
        days_stage = int(np.random.randint(2, 24))
        target_score = round(float(np.random.uniform(12, 44)), 1)
        at_risk = 0

    elif profile == "Medium":
        prior_spa = round(float(np.random.uniform(1.8, 2.4)), 2)
        spa = clamp(round(prior_spa + float(np.random.uniform(0.25, 0.70)), 2), 2.2, 3.1)

        prior_dcc = round(float(np.random.uniform(6.7, 8.1)), 2)
        dcc = clamp(round(prior_dcc + float(np.random.uniform(-0.65, -0.20)), 2), 5.9, 7.7)

        team_alignment = round(float(np.random.uniform(0.78, 0.88)), 2)
        pa_limit_breach = bool(np.random.choice([False, False, False, True]))
        spa_above_peer = bool(np.random.choice([False, True], p=[0.55, 0.45]))
        missing_team = bool(np.random.choice([False, False, True]))
        history_count = int(np.random.randint(5, 11))
        mediation = False
        returned = bool(np.random.choice([False, False, True]))
        locum = bool(np.random.choice([False, False, True]))
        days_stage = int(np.random.randint(18, 55))
        target_score = round(float(np.random.uniform(50, 74)), 1)
        at_risk = int(np.random.choice([0, 1], p=[0.35, 0.65]))

    else:
        prior_spa = round(float(np.random.uniform(1.8, 2.5)), 2)
        spa = clamp(round(prior_spa + float(np.random.uniform(0.75, 1.45)), 2), 2.8, 4.0)

        prior_dcc = round(float(np.random.uniform(6.8, 8.3)), 2)
        dcc = clamp(round(prior_dcc + float(np.random.uniform(-1.25, -0.55)), 2), 5.3, 7.2)

        team_alignment = round(float(np.random.uniform(0.60, 0.76)), 2)

        # Important: not every High case has every severe flag.
        # This prevents everything becoming 99+.
        pa_limit_breach = bool(np.random.choice([True, False], p=[0.55, 0.45]))
        spa_above_peer = bool(np.random.choice([True, False], p=[0.75, 0.25]))
        missing_team = bool(np.random.choice([True, False], p=[0.30, 0.70]))
        history_count = int(np.random.randint(9, 19))
        mediation = bool(np.random.choice([True, False], p=[0.20, 0.80]))
        returned = bool(np.random.choice([True, False], p=[0.30, 0.70]))
        locum = bool(np.random.choice([True, False], p=[0.35, 0.65]))
        days_stage = int(np.random.randint(35, 95))
        target_score = round(float(np.random.uniform(76, 94)), 1)
        at_risk = 1

    cp = round(float(np.random.choice([0.0, 0.25, 0.5, 0.75])), 2)

    minimum_total = dcc + spa + cp
    if minimum_total > total:
        total = round(minimum_total + float(np.random.choice([0.0, 0.25, 0.5])), 2)

    other = round(max(0.0, total - dcc - spa - cp), 2)
    prior_total = round(max(total + float(np.random.choice([-0.5, -0.25, 0, 0.25, 0.5])), prior_dcc + prior_spa + cp), 2)

    peer_spa_share = round(float(np.random.uniform(0.18, 0.25)), 2)
    peer_dcc_share = round(float(np.random.uniform(0.66, 0.78)), 2)

    demand = round(float(np.random.uniform(32, 78)), 1)
    planned_capacity = round(demand * team_alignment, 1)

    has_team_link = not missing_team

    return {
        "job_plan_code": f"JP-{year}-{i:05d}",
        "user_code_hash": user_hash(i),
        "trust_level_code": trust,
        "trustLevelCode": trust,
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
        "demoRiskProfile": profile,
        "targetRiskScore": target_score,
        "atRisk": at_risk,
    }


rows = []
i = 1

profile_counts = {
    "Low": 4800,
    "Medium": 4200,
    "High": 3000,
}

for profile, count in profile_counts.items():
    for _ in range(count):
        rows.append(make_row(i, profile))
        i += 1

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Created best demo dataset: {OUTPUT_PATH}")
print(f"Rows: {len(df):,}")
print("\nDemo risk profile distribution:")
print(df["demoRiskProfile"].value_counts())
print("\nTrust N/A count:")
print(df["trust_level_code"].isna().sum())
