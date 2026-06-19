from __future__ import annotations

import numpy as np
import pandas as pd


def build_pseudo_labels(features: pd.DataFrame) -> pd.Series:
    """
    POC atRisk pseudo-label.

    Meaning:
    plans likely to need manager attention or post-sign-off adjustment.

    Not meaning:
    definitive NHS contract violation.
    """
    if "atRisk" in features.columns:
        return features["atRisk"].astype(int)

    history_threshold = np.nanpercentile(features["historyChangeCount"].fillna(0), 90)

    label = (
        (features["paLimitBreach"].fillna(0).astype(int) == 1)
        | (features["deltaSpaPAs"].fillna(0) > 0.5)
        | (features["teamPlanAlignmentScore"].fillna(1) < 0.8)
        | (features["historyChangeCount"].fillna(0) > history_threshold)
        | (features["hasMediationOrAppeal"].fillna(0).astype(int) == 1)
        | (features["planReturnedToDiscussionAfterSignoff"].fillna(0).astype(int) == 1)
    )

    return label.astype(int)
