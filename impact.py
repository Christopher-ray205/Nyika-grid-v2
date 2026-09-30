"""
Nyika-Grid Predict AI — Cost & Emissions Impact Estimator
--------------------------------------------------------------------
Translates MWh of solar generation + battery discharge into terms non-technical
stakeholders care about: money and emissions avoided.

IMPORTANT — these are order-of-magnitude estimates using typical published
figures, not site-specific engineering numbers. Zambia's national grid is
majority hydro, which is already low-carbon — so the honest environmental
story here is NOT "greening the grid," it's "avoiding diesel backup
generation during shortfalls." That distinction is kept explicit throughout
rather than implying a bigger climate claim than the data supports.

Assumptions (deliberately exposed as constants so they can be replaced with
real site figures rather than buried in a calculation):
  - Diesel backup generation cost:  ~12,000 ZMW per MWh
    (typical backup generator running at ~0.3-0.4 L diesel per kWh at
    recent Zambian pump prices — a rough planning figure, not a quote)
  - Diesel backup emissions:        ~0.9 tonnes CO2 per MWh generated
    (typical for a diesel genset; grid-average figures would be much lower
    here specifically because Zambia's grid is hydro-dominant)
"""

from typing import List, Dict

DIESEL_COST_PER_MWH_ZMW = 12000.0
DIESEL_CO2_TONNES_PER_MWH = 0.9


def estimate_impact(avoided_diesel_mwh: float) -> Dict:
    """Core conversion: MWh of avoided diesel backup -> cost + CO2 avoided."""
    avoided = max(avoided_diesel_mwh, 0)
    return {
        "avoided_diesel_mwh": round(avoided, 2),
        "cost_saved_zmw": round(avoided * DIESEL_COST_PER_MWH_ZMW, 0),
        "co2_avoided_tonnes": round(avoided * DIESEL_CO2_TONNES_PER_MWH, 2),
    }


def estimate_daily_impact(solar_mwh_series: List[float], battery_discharge_mwh_series: List[float]) -> Dict:
    """
    Daily impact from a 24-hour schedule: every MWh of solar generated, and every
    MWh discharged from the battery, is treated as one MWh that didn't have to
    come from diesel backup during that hour.

    This is a simplification (see module docstring) — it assumes all solar and
    battery output displaces diesel rather than displacing hydro/grid power,
    which is the realistic case specifically during high-demand or deficit
    hours, not necessarily every hour of the day.
    """
    total_solar = sum(solar_mwh_series)
    total_discharge = sum(battery_discharge_mwh_series)
    avoided = total_solar + total_discharge

    result = estimate_impact(avoided)
    result["solar_contribution_mwh"] = round(total_solar, 2)
    result["battery_contribution_mwh"] = round(total_discharge, 2)
    return result


def project_annual_impact(daily_impact: Dict) -> Dict:
    """Rough annualized projection from a single representative day. Clearly
    labeled as a projection, not a measured annual figure."""
    return {
        "projected_annual_cost_saved_zmw": round(daily_impact["cost_saved_zmw"] * 365, 0),
        "projected_annual_co2_avoided_tonnes": round(daily_impact["co2_avoided_tonnes"] * 365, 1),
        "note": "Rough projection assuming today's conditions repeat daily — actual results will vary with season and weather.",
    }
