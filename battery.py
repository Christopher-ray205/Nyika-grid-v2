"""
Nyika-Grid Predict AI — Battery Storage Planner
------------------------------------------------------
For solar-hybrid mini-grids: given an hourly load and solar forecast,
simulates a battery's state of charge across the day and recommends
when to charge (solar surplus) vs discharge (solar deficit), and flags
any hours where even the battery can't cover the shortfall.
"""

from typing import List, Dict


def simulate_battery_schedule(
    hourly_load: List[float],
    hourly_solar: List[float],
    battery_capacity_mwh: float,
    initial_soc_frac: float = 0.5,
    max_charge_rate_mw: float = None,
) -> List[Dict]:
    """
    hourly_load / hourly_solar: 24 values (MW), same order (hour 0-23).
    battery_capacity_mwh: usable battery capacity.
    initial_soc_frac: starting state of charge as a fraction (0-1) of capacity.
    max_charge_rate_mw: optional cap on charge/discharge rate per hour;
        defaults to the full battery capacity (i.e. no rate limit).

    Returns a list of hourly entries with the recommended action, resulting
    state of charge, and any unmet deficit that the battery couldn't cover.
    """
    if battery_capacity_mwh <= 0:
        # No battery installed at this site — nothing to schedule.
        return [
            {"hour": h, "load_mw": round(hourly_load[h], 1), "solar_mw": round(hourly_solar[h], 1),
             "net_mw": round(hourly_solar[h] - hourly_load[h], 1), "action": "no_battery",
             "charge_mwh": 0.0, "discharge_mwh": 0.0,
             "soc_mwh": 0.0, "soc_pct": 0.0, "unmet_deficit_mw": round(max(hourly_load[h] - hourly_solar[h], 0), 1)}
            for h in range(len(hourly_load))
        ]

    rate_cap = max_charge_rate_mw or battery_capacity_mwh
    soc = battery_capacity_mwh * initial_soc_frac  # current stored energy, MWh

    schedule = []
    for h in range(len(hourly_load)):
        load = hourly_load[h]
        solar = hourly_solar[h]
        net = solar - load  # positive = surplus, negative = deficit

        unmet_deficit = 0.0
        charge_amount = 0.0
        discharge_amount = 0.0

        if net > 0:
            # Surplus solar — charge the battery
            charge_amount = min(net, rate_cap, battery_capacity_mwh - soc)
            soc += charge_amount
            action = "charge" if charge_amount > 0.01 else "idle"
        elif net < 0:
            # Deficit — discharge the battery to cover it
            needed = -net
            discharge_amount = min(needed, rate_cap, soc)
            soc -= discharge_amount
            unmet_deficit = round(needed - discharge_amount, 2)
            action = "discharge" if discharge_amount > 0.01 else "idle"
        else:
            action = "idle"

        schedule.append({
            "hour": h,
            "load_mw": round(load, 1),
            "solar_mw": round(solar, 1),
            "net_mw": round(net, 1),
            "action": action,
            "charge_mwh": round(charge_amount, 2),
            "discharge_mwh": round(discharge_amount, 2),
            "soc_mwh": round(soc, 2),
            "soc_pct": round(100 * soc / battery_capacity_mwh, 1),
            "unmet_deficit_mw": round(max(unmet_deficit, 0), 1),
        })

    return schedule


def summarize_battery_schedule(schedule: List[Dict]) -> Dict:
    hours_with_unmet = [s for s in schedule if s["unmet_deficit_mw"] > 0]
    return {
        "hours_battery_could_not_cover": len(hours_with_unmet),
        "max_unmet_deficit_mw": round(max((s["unmet_deficit_mw"] for s in schedule), default=0), 1),
        "min_soc_pct": round(min((s["soc_pct"] for s in schedule), default=0), 1),
        "max_soc_pct": round(max((s["soc_pct"] for s in schedule), default=0), 1),
    }
