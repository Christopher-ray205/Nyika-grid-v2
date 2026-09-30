"""
Nyika-Grid Predict AI — Load-Shedding Schedule Optimizer
----------------------------------------------------------------
Turns a raw load forecast into an actionable shedding plan: for every hour
where predicted load exceeds available generation capacity, decides which
feeder zone(s) should be shed, rotating fairly across zones over the day
so no single zone bears a disproportionate share of outages.

This is the difference between "the model says load will be high" and
"here is what operators should actually do about it."
"""

from typing import List, Dict


def generate_shedding_schedule(
    hourly_forecast: List[Dict],
    capacity_mw: float,
    num_zones: int,
) -> List[Dict]:
    """
    hourly_forecast: list of {"hour": int, "predicted_load_mw": float}, 24 entries expected.
    capacity_mw: total available generation for the region.
    num_zones: number of feeder zones that can be individually shed.

    Returns a list of {"hour", "predicted_load_mw", "deficit_mw", "zones_to_shed": [...]}
    Zones are labeled "A", "B", "C", ... and rotated round-robin across
    deficit hours so shedding burden is spread fairly rather than always
    hitting the same zone first.
    """
    zone_labels = [chr(ord("A") + i) for i in range(num_zones)]
    load_per_zone = capacity_mw / num_zones if num_zones else capacity_mw

    schedule = []
    zone_pointer = 0  # rotates which zone gets shed first, for fairness

    for entry in hourly_forecast:
        predicted_load = entry["predicted_load_mw"]
        deficit = predicted_load - capacity_mw

        zones_to_shed = []
        if deficit > 0:
            # How many zones' worth of load needs to be shed to close the gap
            zones_needed = min(num_zones, max(1, int(-(-deficit // load_per_zone))))  # ceil division
            for i in range(zones_needed):
                zones_to_shed.append(zone_labels[(zone_pointer + i) % num_zones])
            zone_pointer = (zone_pointer + zones_needed) % num_zones  # rotate starting point next time

        schedule.append({
            "hour": entry["hour"],
            "predicted_load_mw": round(predicted_load, 1),
            "deficit_mw": round(max(deficit, 0), 1),
            "zones_to_shed": zones_to_shed,
        })

    return schedule


def summarize_schedule(schedule: List[Dict]) -> Dict:
    """Quick summary stats for a generated schedule: how many hours are affected, and per-zone shed counts."""
    affected_hours = [s for s in schedule if s["zones_to_shed"]]
    zone_counts: Dict[str, int] = {}
    for s in affected_hours:
        for z in s["zones_to_shed"]:
            zone_counts[z] = zone_counts.get(z, 0) + 1

    return {
        "total_hours_with_shedding": len(affected_hours),
        "peak_deficit_mw": round(max((s["deficit_mw"] for s in schedule), default=0), 1),
        "shed_hours_per_zone": zone_counts,
    }
