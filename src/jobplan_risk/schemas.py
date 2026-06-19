from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class JobPlanFeatureInput(BaseModel):
    job_plan_code: str
    user_code_hash: Optional[str] = None
    trust_level_code: Optional[str] = None
    parent_trust_level_code: Optional[str] = None
    department: Optional[str] = None
    specialty: Optional[str] = None
    planning_year: Optional[int] = None
    plan_stage: Optional[str] = "Unknown"
    currentStageOrdinal: Optional[float] = None
    daysInCurrentStage: Optional[float] = 0

    pAsPerWeek: Optional[float] = None
    paidPA: Optional[float] = None
    minsPerWeek: Optional[float] = None
    ntPerPA: Optional[float] = None
    workType: Optional[str] = "Unknown"
    doctorClassification: Optional[str] = "Unknown"

    isSignedUpToNewContract: Optional[bool] = False
    isAnnualised: Optional[bool] = False
    uses5thWeekPerQuarter: Optional[bool] = False
    hasTeamPlanLink: Optional[bool] = False

    totalPAs: Optional[float] = None
    dccPAs: Optional[float] = None
    spaPAs: Optional[float] = None
    cpPAs: Optional[float] = 0
    otherPAs: Optional[float] = 0

    isOnCall: Optional[bool] = False
    isShift: Optional[bool] = False
    isCPD: Optional[bool] = False

    priorTotalPAs: Optional[float] = None
    priorDccPAs: Optional[float] = None
    priorSpaPAs: Optional[float] = None
    yearsSinceLastPlan: Optional[float] = None

    peerMedianSpaShare: Optional[float] = None
    peerMedianDccShare: Optional[float] = None
    peerGroupSize: Optional[float] = 0

    paLimitBreach: Optional[bool] = False
    spaAbovePeerThreshold: Optional[bool] = False
    missingTeamPlanLink: Optional[bool] = False
    teamPlanAlignmentScore: Optional[float] = None
    teamDemandPAs: Optional[float] = None
    teamPlannedCapacityPAs: Optional[float] = None

    historyChangeCount: Optional[float] = 0
    hasMediationOrAppeal: Optional[bool] = False
    hasNewManagerChanges: Optional[bool] = False
    planReturnedToDiscussionAfterSignoff: Optional[bool] = False

    isLocum: Optional[bool] = False
    gradeCode: Optional[str] = "Unknown"
    isConsultant: Optional[bool] = True
