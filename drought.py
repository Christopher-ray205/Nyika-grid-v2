"""
Nyika-Grid Predict AI — Drought Stress-Test Scenarios
------------------------------------------------------------
Lets the shedding-schedule optimizer be replayed against real points in
Zambia's 2024 hydropower crisis, instead of only hypothetical conditions.

Grounded in reported events (not invented figures):
  - 11 March 2024: ZESCO began 8-hour daily load-shedding after assessing
    depressed water levels in the Kafue and Zambezi basins.
  - May 2024: daily load-shedding rose to 12 hours as the drought deepened.
  - September 2024: load-shedding reached roughly 20 hours a day; Lake
    Kariba's usable storage fell to about 8% (from 26% a year earlier),
    and Zambia halted generation at Kariba North Bank Power Station for
    the first time in the plant's history (14 Sept 2024).
  Sources: Wikipedia "2024 Zambian drought"; Lusaka Times; African Arguments;
  Down To Earth — reporting from March-September 2024.

METHOD, STATED PLAINLY: this module does NOT have ZESCO's actual internal
generation-capacity-loss figures for each date. It uses the REPORTED daily
load-shedding HOURS (a real, published fact) as a proxy for the fraction of
a day the grid was capacity-constrained, and reduces the simulated
available capacity by that same fraction. This is a transparent
simplification for stress-testing the shedding optimizer against realistic
severity levels — not a claim of precise historical capacity data.
"""

from typing import Dict

DROUGHT_SCENARIOS = {
    "none": {
        "label": "Normal conditions",
        "reported_shedding_hours": 0,
        "note": "No drought stress applied — capacity as configured for the region.",
    },
    "2024_march": {
        "label": "March 2024 — drought response begins",
        "reported_shedding_hours": 8,
        "note": ("ZESCO introduced 8-hour daily load-shedding on 11 March 2024, "
                  "after assessing depressed water levels in the Kafue and Zambezi basins."),
    },
    "2024_may": {
        "label": "May 2024 — shortfall deepens",
        "reported_shedding_hours": 12,
        "note": "Daily load-shedding rose to 12 hours by May 2024 as hydro output kept falling.",
    },
    "2024_september": {
        "label": "September 2024 — crisis peak",
        "reported_shedding_hours": 20,
        "note": ("Load-shedding reached roughly 20 hours a day; Lake Kariba's usable storage fell "
                  "to about 8%, and Zambia halted generation at Kariba North Bank Power Station "
                  "entirely for the first time in its history."),
    },
}


def apply_drought_scenario(base_capacity_mw: float, scenario_key: str) -> Dict:
    """Returns the drought-adjusted capacity plus the scenario's context, for display."""
    if scenario_key not in DROUGHT_SCENARIOS:
        scenario_key = "none"

    scenario = DROUGHT_SCENARIOS[scenario_key]
    reduction_fraction = scenario["reported_shedding_hours"] / 24.0
    adjusted_capacity = base_capacity_mw * (1 - reduction_fraction)

    return {
        "scenario_key": scenario_key,
        "label": scenario["label"],
        "note": scenario["note"],
        "reduction_pct": round(reduction_fraction * 100, 1),
        "base_capacity_mw": round(base_capacity_mw, 1),
        "adjusted_capacity_mw": round(adjusted_capacity, 1),
    }
