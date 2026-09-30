"""
Nyika-Grid Predict AI — Data Utilities
------------------------------------------
Shared simulated-data generation for multiple regions, plus the solar
generation curve used by both the training script and live predictions.

Replace these generators with real data ingestion (ZESCO load data,
Zambia Met Dept / solar irradiance data) when available.
"""

import numpy as np
import pandas as pd

# Region configuration: each region gets its own trained load + solar model,
# demonstrating multi-region / mini-grid support (documentation Section 15/16).
REGIONS = {
    "lusaka_urban": {
        "label": "Lusaka Urban Grid",
        "base_load": 150.0,
        "peak_effect": 80.0,
        "temp_sensitivity": 2.5,
        "panel_capacity_mw": 40.0,   # solar farm capacity feeding this grid
        "lat": -15.4167,
        "lon": 28.2833,
        "alert_threshold_mw": 220.0,
        "available_capacity_mw": 235.0,   # total dispatchable generation (grid + solar) before shedding is needed
        "num_zones": 4,                    # feeder zones that can be individually shed
        "battery_capacity_mwh": 0.0,       # no battery storage on the main urban grid
    },
    "eastern_minigrid": {
        "label": "Eastern Mini-Grid (Chipata, Solar-Hybrid)",
        "base_load": 20.0,
        "peak_effect": 10.0,
        "temp_sensitivity": 0.8,
        "panel_capacity_mw": 35.0,   # sized so midday solar can exceed daytime load — genuine surplus for battery charging
        "lat": -13.6333,
        "lon": 32.6500,
        "alert_threshold_mw": 32.0,
        "available_capacity_mw": 28.0,    # non-solar dispatchable capacity — deficit is measured on NET load (load minus solar)
        "num_zones": 3,
        "battery_capacity_mwh": 15.0,      # battery storage available to smooth solar/demand mismatch
    },
}


def solar_curve_mw(hour, cloud_cover_pct, panel_capacity_mw):
    """
    Simple physics-inspired solar generation curve:
    zero at night, bell-shaped between 06:00-18:00, reduced by cloud cover.
    Works for both scalar and array inputs.
    """
    hour = np.asarray(hour, dtype=float)
    cloud_cover_pct = np.asarray(cloud_cover_pct, dtype=float)

    daylight = (hour >= 6) & (hour <= 18)
    daylight_frac = np.clip((hour - 6) / 12.0, 0, 1)
    base = np.where(daylight, np.sin(np.pi * daylight_frac) * panel_capacity_mw, 0.0)

    cloud_factor = 1 - (cloud_cover_pct / 100.0) * 0.75
    output = base * cloud_factor
    return np.clip(output, 0, None)


def generate_regional_data(region_key: str, data_size: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Simulated hourly load + solar dataset for a single region."""
    cfg = REGIONS[region_key]
    np.random.seed(seed)

    hours = np.random.randint(0, 24, data_size)
    temperature = np.random.uniform(15, 38, data_size)
    is_weekend = np.random.choice([0, 1], p=[0.71, 0.29], size=data_size)
    cloud_cover_pct = np.random.uniform(0, 100, data_size)
    month = np.random.randint(1, 13, data_size)

    peak_hour_effect = np.where((hours >= 18) & (hours <= 21), cfg["peak_effect"], 0)
    temp_effect = (temperature - 20) * cfg["temp_sensitivity"]
    grid_load_mw = (
        cfg["base_load"] + peak_hour_effect + temp_effect
        + np.random.normal(0, cfg["base_load"] * 0.1, data_size)
    )
    grid_load_mw = np.clip(grid_load_mw, 0, None)

    solar_generation_mw = solar_curve_mw(hours, cloud_cover_pct, cfg["panel_capacity_mw"])
    solar_generation_mw = solar_generation_mw + np.random.normal(0, 1.0, data_size)
    solar_generation_mw = np.clip(solar_generation_mw, 0, None)

    return pd.DataFrame({
        "hour": hours,
        "temperature_c": temperature,
        "is_weekend": is_weekend,
        "cloud_cover_pct": cloud_cover_pct,
        "month": month,
        "grid_load_mw": grid_load_mw,
        "solar_generation_mw": solar_generation_mw,
    })


def generate_multiday_series(region_key: str, days: int = 90, seed: int = 42) -> pd.DataFrame:
    """
    Longer hourly time series with daily + weekly + seasonal structure,
    formatted for Prophet ('ds', 'y' columns) to support multi-day
    forecasting with confidence intervals.
    """
    cfg = REGIONS[region_key]
    np.random.seed(seed)

    periods = days * 24
    timestamps = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=periods, freq="h")
    hours = timestamps.hour.values
    dow = timestamps.dayofweek.values
    is_weekend = (dow >= 5).astype(int)
    day_of_year = timestamps.dayofyear.values

    seasonal_temp = 20 + 8 * np.sin(2 * np.pi * (day_of_year / 365))
    daily_temp_swing = -0.3 * (hours - 14)
    temperature = seasonal_temp + daily_temp_swing + np.random.normal(0, 3, periods)

    peak_hour_effect = np.where((hours >= 18) & (hours <= 21), cfg["peak_effect"], 0)
    temp_effect = (temperature - 20) * cfg["temp_sensitivity"]
    weekend_effect = np.where(is_weekend == 1, -0.05 * cfg["base_load"], 0)

    grid_load_mw = (
        cfg["base_load"] + peak_hour_effect + temp_effect + weekend_effect
        + np.random.normal(0, cfg["base_load"] * 0.08, periods)
    )
    grid_load_mw = np.clip(grid_load_mw, 0, None)

    return pd.DataFrame({"ds": timestamps, "y": grid_load_mw})
