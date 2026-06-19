-- Conceptual only. Adapt names/joins to staging.
-- Goal: one row per active job plan.
-- Do not recalculate PA totals from raw WorkEpisode rows for the POC.

SELECT
    jp.JP_Code AS job_plan_code,
    jp.JP_TL_Code AS trust_level_code,
    jp.JP_DoctorClassification AS doctorClassification,
    jp.JP_IsSignedUpToNewContract AS isSignedUpToNewContract,
    jp.JP_PAsPerWeek AS pAsPerWeek,
    jp.JP_PaidPA AS paidPA,
    jp.JP_WorkType AS workType,
    CASE WHEN jp.JP_TeamPlanId IS NULL THEN 0 ELSE 1 END AS hasTeamPlanLink,
    jp.JP_CurrentStageRef AS currentStageOrdinal,
    pac.PD_TotalPAs AS totalPAs,
    pac.PD_isOnCall AS isOnCall,
    pac.PD_isShift AS isShift,
    pac.PD_isCPD AS isCPD
FROM dbo.JobPlan jp
JOIN dbo.PAAnalysisCache pac
    ON pac.PD_JP_Code = jp.JP_Code
LEFT JOIN dbo.UserPreferences up
    ON up.UP_US_Code = jp.JP_US_Code
WHERE (up.UP_IsTestUser = 0 OR up.UP_US_Code IS NULL)
  AND pac.PD_TotalPAs IS NOT NULL
  AND pac.PD_TotalPAs > 0;

-- Add later:
-- 1. Pivot PAValueCache to dccPAs, spaPAs, cpPAs.
-- 2. Join prior JobPlanHistory for YoY deltas.
-- 3. Join Core Doctor / TrustLevel for locum, grade, peer grouping.
-- 4. Join PALimits/rule outputs.
-- 5. Join optional TJP teamPlanAlignmentScore.
