# JobPlan ML Risk Layer POC

This repo implements the ML/risk scoring layer for the AI-Driven Job Plan Analyser POC.

## Dataset principle

Use one row per job plan.

Recommended sources:

- PAAnalysisCache
- Pivoted PAValueCache
- JobPlan
- JobPlanHistory
- JobPlanHistoryChangesCache
- StageDate
- PALimits
- Core Doctor / TrustLevel
- Optional TeamJobPlanning alignment

Do not use raw WorkEpisode rows for this POC.

## Setup
