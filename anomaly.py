"""
Nyika-Grid Predict AI — Anomaly Detection
------------------------------------------------
Flags when a logged *actual* reading deviates unusually far from what the
model predicted. This isn't about the model being wrong on average — it's
about catching individual readings that don't fit the pattern at all,
which in a real deployment could mean a faulty meter, power theft,
an unplanned outage, or a genuine one-off event worth investigating.
"""

from typing import Optional, Dict


def check_anomaly(
    predicted_mw: float,
    actual_mw: float,
    baseline_rmse_mw: float,
    z_threshold: float = 3.0,
) -> Dict:
    """
    Flags an anomaly if the prediction error is more than `z_threshold`
    times the model's typical RMSE — i.e. this reading is far outside
    the model's normal error range, not just "a bit off."
    """
    if baseline_rmse_mw <= 0:
        baseline_rmse_mw = 1.0  # avoid divide-by-zero if no RMSE history yet

    error = actual_mw - predicted_mw
    z_score = error / baseline_rmse_mw
    is_anomaly = abs(z_score) >= z_threshold

    if not is_anomaly:
        return {
            "is_anomaly": False,
            "z_score": round(z_score, 2),
            "message": "Reading is within the model's normal error range.",
        }

    direction = "higher" if error > 0 else "lower"
    return {
        "is_anomaly": True,
        "z_score": round(z_score, 2),
        "message": (
            f"Actual reading ({actual_mw:.1f} MW) is {abs(error):.1f} MW {direction} than predicted "
            f"({predicted_mw:.1f} MW) — {abs(z_score):.1f}x the model's typical error. "
            f"Worth checking for a meter fault, unplanned event, or data entry error."
        ),
    }
