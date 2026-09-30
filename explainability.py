"""
Nyika-Grid Predict AI — Per-Prediction Explainability
--------------------------------------------------------------
Explains an INDIVIDUAL prediction — "this forecast is high because of X" —
rather than only the model's overall (global) feature importance.

METHOD: this uses feature ablation (also called a "leave-one-out" or
marginal-contribution explanation), not the SHAP library. For each feature,
it re-runs the prediction with that one feature replaced by a typical
baseline value, and reports how much the prediction changes as a result.
This is a standard, simple local-explanation technique — but it is an
approximation, not exact Shapley values, and contributions may not sum
exactly to (prediction - baseline) when the model has feature interactions.
That trade-off is worth it here: it requires no extra library, runs
instantly, and is easy to explain to a non-technical judge or operator.
"""

import pandas as pd
from typing import Dict

# Typical/"nothing unusual" conditions, used as the reference point a
# feature is compared against. Chosen as reasonable midpoints, not derived
# from real data statistics (a real deployment could compute these from
# historical averages instead).
LOAD_BASELINE = {"hour": 12, "temperature_c": 25.0, "is_weekend": 0}
SOLAR_BASELINE = {"hour": 12, "cloud_cover_pct": 20.0, "month": 6}


def explain_load_prediction(model, hour: int, temperature_c: float, is_weekend: int) -> Dict:
    actual_input = {"hour": hour, "temperature_c": temperature_c, "is_weekend": is_weekend}
    return _explain(model, actual_input, LOAD_BASELINE)


def explain_solar_prediction(model, hour: int, cloud_cover_pct: float, month: int) -> Dict:
    actual_input = {"hour": hour, "cloud_cover_pct": cloud_cover_pct, "month": month}
    return _explain(model, actual_input, SOLAR_BASELINE)


def _explain(model, actual_input: dict, baseline_input: dict) -> Dict:
    features = list(actual_input.keys())

    full_pred = float(model.predict(pd.DataFrame([actual_input]))[0])
    baseline_pred = float(model.predict(pd.DataFrame([baseline_input]))[0])

    contributions = {}
    for feature in features:
        # Prediction with every feature at the actual value EXCEPT this one, held at baseline
        ablated_input = dict(actual_input)
        ablated_input[feature] = baseline_input[feature]
        ablated_pred = float(model.predict(pd.DataFrame([ablated_input]))[0])

        # How much does restoring this feature to its actual value change the prediction?
        contributions[feature] = round(full_pred - ablated_pred, 2)

    return {
        "prediction": round(full_pred, 2),
        "baseline_prediction": round(baseline_pred, 2),
        "baseline_inputs": baseline_input,
        "actual_inputs": actual_input,
        "feature_contributions_mw": contributions,
    }
