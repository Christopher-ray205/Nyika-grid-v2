"""
Nyika-Grid Predict AI — Model Training (v2)
--------------------------------------------------
Trains, per region:
  - A Random Forest load forecasting model  -> models/{region}_load.pkl
  - A Random Forest solar generation model  -> models/{region}_solar.pkl
  - A Prophet model for 7-day forecasts with confidence intervals
    -> models/{region}_prophet.pkl

Usage:
    python train_model.py
"""

import logging
import warnings

# Prophet/cmdstanpy are very chatty by default; quiet them down for clean output
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

import json
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from prophet import Prophet

from data_utils import REGIONS, generate_regional_data, generate_multiday_series
from baseline import build_hourly_baseline, evaluate_baseline

MODELS_DIR = "models"


def train_load_model(df):
    X = df[["hour", "temperature_c", "is_weekend"]]
    y = df["grid_load_mw"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return model, rmse, r2


def train_solar_model(df):
    X = df[["hour", "cloud_cover_pct", "month"]]
    y = df["solar_generation_mw"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return model, rmse, r2


def train_prophet_model(region_key):
    df = generate_multiday_series(region_key, days=90)
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    model.fit(df)
    return model


def main():
    import os
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("--- Nyika-Grid Predict AI: Training (v2, multi-region + solar) ---\n")

    for region_key, cfg in REGIONS.items():
        print(f"=== Region: {cfg['label']} ({region_key}) ===")

        df = generate_regional_data(region_key)

        load_model, load_rmse, load_r2 = train_load_model(df)
        print(f"  Load model    — R^2: {load_r2:.2f} | RMSE: {load_rmse:.2f} MW")
        joblib.dump(load_model, f"{MODELS_DIR}/{region_key}_load.pkl")

        solar_model, solar_rmse, solar_r2 = train_solar_model(df)
        print(f"  Solar model   — R^2: {solar_r2:.2f} | RMSE: {solar_rmse:.2f} MW")
        joblib.dump(solar_model, f"{MODELS_DIR}/{region_key}_solar.pkl")

        # Naive baseline comparison — proves the ML model earns its complexity
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        baseline_lookup = build_hourly_baseline(train_df)
        overall_mean = float(train_df["grid_load_mw"].mean())
        baseline_rmse = evaluate_baseline(test_df, baseline_lookup, overall_mean)
        improvement_pct = 100 * (baseline_rmse - load_rmse) / baseline_rmse
        print(f"  Naive hourly-average baseline RMSE: {baseline_rmse:.2f} MW "
              f"(model is {improvement_pct:.0f}% better)")

        feature_importances = dict(zip(
            ["hour", "temperature_c", "is_weekend"],
            [round(float(x), 3) for x in load_model.feature_importances_]
        ))

        metrics = {
            "load_model_rmse": round(load_rmse, 2),
            "load_model_r2": round(load_r2, 2),
            "solar_model_rmse": round(solar_rmse, 2),
            "solar_model_r2": round(solar_r2, 2),
            "baseline_rmse": round(baseline_rmse, 2),
            "improvement_over_baseline_pct": round(improvement_pct, 1),
            "load_feature_importances": feature_importances,
        }
        with open(f"{MODELS_DIR}/{region_key}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        joblib.dump({"lookup": baseline_lookup, "overall_mean": overall_mean}, f"{MODELS_DIR}/{region_key}_baseline.pkl")

        print("  Training Prophet 7-day forecast model (this takes a few seconds)...")
        prophet_model = train_prophet_model(region_key)
        joblib.dump(prophet_model, f"{MODELS_DIR}/{region_key}_prophet.pkl")
        print("  Prophet model saved.\n")

    print("All models trained and saved to ./models/")


if __name__ == "__main__":
    main()
