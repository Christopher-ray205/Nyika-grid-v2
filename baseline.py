"""
Nyika-Grid Predict AI — Naive Baseline
------------------------------------------
A simple "hourly average" baseline model: predicts load as the historical
average for that hour of day, with no weather or day-type awareness.

This exists to answer a question judges and operators will reasonably
ask: "why use ML instead of just averaging past readings?" — by measuring
the actual RMSE improvement the trained model provides over this naive
approach, rather than just asserting the model is better.
"""

import numpy as np
import pandas as pd
from typing import Dict


def build_hourly_baseline(df: pd.DataFrame) -> Dict[int, float]:
    """Returns {hour: mean_load_mw} computed from training data."""
    return df.groupby("hour")["grid_load_mw"].mean().to_dict()


def predict_baseline(baseline_lookup: Dict[int, float], hour: int, overall_mean: float) -> float:
    """Predict using the hourly baseline, falling back to the overall mean for unseen hours."""
    return baseline_lookup.get(hour, overall_mean)


def evaluate_baseline(df_test: pd.DataFrame, baseline_lookup: Dict[int, float], overall_mean: float) -> float:
    """RMSE of the naive baseline on a test set."""
    preds = df_test["hour"].apply(lambda h: predict_baseline(baseline_lookup, h, overall_mean))
    errors = df_test["grid_load_mw"].values - preds.values
    return float(np.sqrt(np.mean(errors ** 2)))
