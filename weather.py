"""
Nyika-Grid Predict AI — Live Weather Integration
------------------------------------------------------
Pulls current temperature and cloud cover from Open-Meteo (free, no API
key required) for a given region's coordinates. Falls back gracefully —
callers should always check `success` and fall back to manual input if
False (e.g. no internet access, or the API is temporarily unreachable).

NOTE: this has NOT been live-tested in the development sandbox this
prototype was built in, because that environment's network access is
restricted to a fixed allow-list of package-registry domains and does
not include api.open-meteo.com. The manual-entry fallback path IS
tested. Test the live call in an environment with normal internet
access before relying on it.
"""

import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_live_weather(lat: float, lon: float, timeout: int = 6) -> dict:
    """
    Fetch current temperature (°C) and cloud cover (%) for a location.
    Returns {"success": True, "temperature_c": ..., "cloud_cover_pct": ...}
    or {"success": False, "error": "..."} on any failure.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,cloud_cover",
    }
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        current = data.get("current", {})
        temperature_c = current.get("temperature_2m")
        cloud_cover_pct = current.get("cloud_cover")

        if temperature_c is None or cloud_cover_pct is None:
            return {"success": False, "error": "Unexpected response format from weather API."}

        return {
            "success": True,
            "temperature_c": float(temperature_c),
            "cloud_cover_pct": float(cloud_cover_pct),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
