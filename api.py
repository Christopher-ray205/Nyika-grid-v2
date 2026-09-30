"""
Nyika-Grid Predict AI — API (v2)
--------------------------------------
Endpoints:
  GET  /regions                    List available regions
  POST /predict/load               Predict grid load for a region
  POST /predict/solar              Predict solar generation for a region
  POST /predict/net-load           Combined load, solar, and net load
  GET  /forecast/7day?region=...   7-day hourly forecast with confidence interval
  POST /alerts/check               Check a load value against the region's threshold
  POST /accuracy/log-actual        Record the actual outcome for a logged prediction
  GET  /accuracy/summary?region=.. RMSE/MAE over logged predictions with actuals
  GET  /weather/live?region=...    Attempt a live weather pull for the region

Usage:
    1. Run `python train_model.py` first to produce the model files.
    2. uvicorn api:app --reload
    3. Visit http://127.0.0.1:8000/docs to test interactively.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import joblib
import pandas as pd
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional

from data_utils import REGIONS, solar_curve_mw
import db
import alerts
import weather
import shedding
import battery
import anomaly
import sms_interface
import impact
import data_ingestion
import drought
import explainability
import holidays_zm
import execution_tracking
import notice
import map_view
import report
import webhooks
from fastapi import UploadFile, File, Form
from fastapi.responses import StreamingResponse
import io

app = FastAPI(
    title="Nyika-Grid Predict AI",
    description="Multi-region grid load + solar generation forecasting for Zambia.",
    version="0.2.0"
)

# Allow the standalone frontend (served from any local port, or opened as a file)
# to call this API during local development. Tighten this before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = "models"
_models = {}  # cache: {(region, use_custom): {"load": model, "solar": model, "prophet": model, "used_custom": {...}}}


def _load_region_models(region_key: str, use_custom: bool = False):
    cache_key = (region_key, use_custom)
    if cache_key in _models:
        return _models[cache_key]

    # Prophet isn't retrained from a CSV upload (needs a real timestamp series),
    # so it always comes from the original trained model.
    prophet_path = f"{MODELS_DIR}/{region_key}_prophet.pkl"
    if not os.path.exists(prophet_path):
        raise HTTPException(status_code=503, detail=f"Model file {prophet_path} not found. Run `python train_model.py` first.")

    used_custom = {"load": False, "solar": False}
    loaded = {"prophet": joblib.load(prophet_path)}

    for sub in ("load", "solar"):
        custom_path = f"{MODELS_DIR}/{region_key}_{sub}_custom.pkl"
        default_path = f"{MODELS_DIR}/{region_key}_{sub}.pkl"

        if use_custom and os.path.exists(custom_path):
            loaded[sub] = joblib.load(custom_path)
            used_custom[sub] = True
        elif os.path.exists(default_path):
            loaded[sub] = joblib.load(default_path)
        else:
            raise HTTPException(
                status_code=503,
                detail=f"Model file {default_path} not found. Run `python train_model.py` first."
            )

    loaded["used_custom"] = used_custom
    _models[cache_key] = loaded
    return loaded


def _validate_region(region_key: str):
    if region_key not in REGIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown region '{region_key}'. Valid options: {list(REGIONS.keys())}"
        )


# ---------- Schemas ----------

class LoadRequest(BaseModel):
    region: str = Field(..., description="Region key, e.g. 'lusaka_urban'")
    hour: int = Field(..., ge=0, le=23)
    temperature_c: float = Field(..., ge=-10, le=55)
    is_weekend: int = Field(..., ge=0, le=1)


class SolarRequest(BaseModel):
    region: str
    hour: int = Field(..., ge=0, le=23)
    cloud_cover_pct: float = Field(..., ge=0, le=100)
    month: int = Field(..., ge=1, le=12)


class NetLoadRequest(BaseModel):
    region: str
    hour: int = Field(..., ge=0, le=23)
    temperature_c: float = Field(..., ge=-10, le=55)
    is_weekend: int = Field(..., ge=0, le=1)
    cloud_cover_pct: float = Field(..., ge=0, le=100)
    month: int = Field(..., ge=1, le=12)
    log_prediction: bool = Field(False, description="If true, saves this prediction to the database")
    use_custom: bool = Field(False, description="Use the custom-retrained model for this region, if one exists")


class AlertCheckRequest(BaseModel):
    region: str
    predicted_load_mw: float


class ActualOutcomeRequest(BaseModel):
    log_id: int
    actual_load_mw: Optional[float] = None
    actual_solar_mw: Optional[float] = None


class ShedRequest(BaseModel):
    region: str
    temperature_c: float = Field(30.0, description="Assumed temperature held constant across the day")
    is_weekend: int = Field(0, ge=0, le=1)
    use_custom: bool = Field(False)
    drought_scenario: str = Field("none", description="One of: none, 2024_march, 2024_may, 2024_september")


class BatteryRequest(BaseModel):
    region: str
    temperature_c: float = Field(30.0)
    is_weekend: int = Field(0, ge=0, le=1)
    cloud_cover_pct: float = Field(20.0, ge=0, le=100)
    month: int = Field(9, ge=1, le=12)
    initial_soc_frac: float = Field(0.5, ge=0, le=1)
    use_custom: bool = Field(False)


class SmsRequest(BaseModel):
    text: str
    lang: str = Field("EN", description="Reply language: EN, NY, or BE")


class LogPlanRequest(BaseModel):
    region: str
    shed_date: str = Field(..., description="YYYY-MM-DD")
    schedule: list  # the "schedule" list from a /shedding/schedule response


class MarkExecutionRequest(BaseModel):
    entry_id: int
    status: str = Field(..., description="'executed' or 'skipped'")
    notes: Optional[str] = None


class NoticeRequest(BaseModel):
    region: str
    schedule: list  # the "schedule" list from a /shedding/schedule response
    notice_date: Optional[str] = Field(None, description="YYYY-MM-DD, defaults to today")


class ReportRequest(BaseModel):
    region: str
    temperature_c: float = Field(30.0)
    is_weekend: int = Field(0, ge=0, le=1)
    cloud_cover_pct: float = Field(20.0, ge=0, le=100)
    month: int = Field(9, ge=1, le=12)
    drought_scenario: str = Field("none")
    report_date: Optional[str] = Field(None, description="YYYY-MM-DD, defaults to today")


class WebhookRegisterRequest(BaseModel):
    region: str = Field(..., description="A region key, or '*' for all regions")
    url: str


class WebhookDispatchRequest(BaseModel):
    region: str
    predicted_load_mw: float
    message: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=12)
    region: str
    hour: int = Field(..., ge=0, le=23)
    temperature_c: float = Field(..., ge=-10, le=55)
    is_weekend: int = Field(..., ge=0, le=1)
    cloud_cover_pct: float = Field(..., ge=0, le=100)
    month: int = Field(..., ge=1, le=12)
    initial_soc_frac: float = Field(0.5, ge=0, le=1)
    use_custom: bool = False


def _build_24h_predictions(region_key: str, temperature_c: float, is_weekend: int,
                            cloud_cover_pct: float = 20.0, month: int = 9, use_custom: bool = False):
    models = _load_region_models(region_key, use_custom)
    hours = list(range(24))

    load_df = pd.DataFrame([{"hour": h, "temperature_c": temperature_c, "is_weekend": is_weekend} for h in hours])
    loads = models["load"].predict(load_df)

    solar_df = pd.DataFrame([{"hour": h, "cloud_cover_pct": cloud_cover_pct, "month": month} for h in hours])
    solars = np.clip(models["solar"].predict(solar_df), 0, None)

    return hours, list(loads), list(solars)


# ---------- Endpoints ----------

@app.get("/")
def root():
    return {"message": "Nyika-Grid Predict AI v2 is running.", "docs": "/docs"}


@app.get("/regions")
def list_regions():
    return {
        key: {"label": cfg["label"], "alert_threshold_mw": cfg["alert_threshold_mw"]}
        for key, cfg in REGIONS.items()
    }


@app.post("/predict/load")
def predict_load(req: LoadRequest):
    _validate_region(req.region)
    models = _load_region_models(req.region)
    input_df = pd.DataFrame([{"hour": req.hour, "temperature_c": req.temperature_c, "is_weekend": req.is_weekend}])
    prediction = float(models["load"].predict(input_df)[0])
    return {"region": req.region, "predicted_load_mw": round(prediction, 2)}


@app.post("/predict/solar")
def predict_solar(req: SolarRequest):
    _validate_region(req.region)
    models = _load_region_models(req.region)
    input_df = pd.DataFrame([{"hour": req.hour, "cloud_cover_pct": req.cloud_cover_pct, "month": req.month}])
    prediction = float(models["solar"].predict(input_df)[0])
    return {"region": req.region, "predicted_solar_mw": round(max(prediction, 0), 2)}


@app.post("/predict/net-load")
def predict_net_load(req: NetLoadRequest):
    _validate_region(req.region)
    models = _load_region_models(req.region, req.use_custom)

    load_input = pd.DataFrame([{"hour": req.hour, "temperature_c": req.temperature_c, "is_weekend": req.is_weekend}])
    predicted_load = float(models["load"].predict(load_input)[0])

    solar_input = pd.DataFrame([{"hour": req.hour, "cloud_cover_pct": req.cloud_cover_pct, "month": req.month}])
    predicted_solar = max(float(models["solar"].predict(solar_input)[0]), 0)

    net_load = predicted_load - predicted_solar

    log_id = None
    if req.log_prediction:
        log_id = db.log_prediction(
            region=req.region, hour=req.hour, temperature_c=req.temperature_c,
            is_weekend=req.is_weekend, predicted_load_mw=predicted_load,
            predicted_solar_mw=predicted_solar
        )

    return {
        "region": req.region,
        "predicted_load_mw": round(predicted_load, 2),
        "predicted_solar_mw": round(predicted_solar, 2),
        "net_load_mw": round(net_load, 2),
        "log_id": log_id,
        "used_custom_model": models["used_custom"],
    }


@app.get("/forecast/24h")
def forecast_24h(region: str, temperature_c: float = 30.0, is_weekend: int = 0,
                  cloud_cover_pct: float = 20.0, month: int = 9, use_custom: bool = False):
    """Convenience endpoint for a frontend: full 24-hour load + solar + net-load
    curve in one call, instead of 24 separate /predict requests."""
    _validate_region(region)
    cfg = REGIONS[region]
    hours, loads, solars = _build_24h_predictions(region, temperature_c, is_weekend, cloud_cover_pct, month, use_custom)
    net_loads = [l - s for l, s in zip(loads, solars)]

    return {
        "region": region,
        "alert_threshold_mw": cfg["alert_threshold_mw"],
        "hours": hours,
        "load_mw": [round(v, 2) for v in loads],
        "solar_mw": [round(v, 2) for v in solars],
        "net_load_mw": [round(v, 2) for v in net_loads],
    }


@app.get("/forecast/7day")
def forecast_7day(region: str):
    _validate_region(region)
    models = _load_region_models(region)

    future = models["prophet"].make_future_dataframe(periods=24 * 7, freq="h")
    forecast = models["prophet"].predict(future)
    forecast_tail = forecast.tail(24 * 7)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    return {
        "region": region,
        "forecast": [
            {
                "timestamp": row["ds"].isoformat(),
                "predicted_load_mw": round(row["yhat"], 2),
                "lower_bound_mw": round(row["yhat_lower"], 2),
                "upper_bound_mw": round(row["yhat_upper"], 2),
            }
            for _, row in forecast_tail.iterrows()
        ]
    }


@app.post("/alerts/check")
def check_alert(req: AlertCheckRequest):
    _validate_region(req.region)
    threshold = REGIONS[req.region]["alert_threshold_mw"]
    result = alerts.check_load_alert(req.predicted_load_mw, threshold)
    if result["triggered"]:
        alerts.send_alert(result["message"], method="log", recipient=f"{req.region}-operations")
        # Also deliver to any registered webhooks, so external systems get a real
        # notification rather than only a log line. Results are returned as-is.
        result["webhook_deliveries"] = webhooks.dispatch_alert_to_webhooks(req.region, {
            "region": req.region,
            "predicted_load_mw": req.predicted_load_mw,
            "threshold_mw": threshold,
            "level": result["level"],
            "message": result["message"],
        })
    return result


@app.get("/weather/live")
def live_weather(region: str):
    _validate_region(region)
    cfg = REGIONS[region]
    result = weather.fetch_live_weather(cfg["lat"], cfg["lon"])
    return result


@app.post("/accuracy/log-actual")
def log_actual_outcome(req: ActualOutcomeRequest):
    logs = db.get_recent_logs(limit=200)
    entry = next((l for l in logs if l.id == req.log_id), None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Log entry {req.log_id} not found.")

    ok = db.log_actual(req.log_id, req.actual_load_mw, req.actual_solar_mw)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Log entry {req.log_id} not found.")

    result = {"success": True, "anomaly": None}
    if req.actual_load_mw is not None:
        metrics = _get_metrics(entry.region)
        baseline_rmse = metrics["load_model_rmse"] if metrics else 15.0
        result["anomaly"] = anomaly.check_anomaly(entry.predicted_load_mw, req.actual_load_mw, baseline_rmse)

    return result


@app.get("/accuracy/summary")
def accuracy_summary(region: Optional[str] = None):
    stats = db.get_accuracy_stats(region)
    if stats is None:
        return {"message": "No logged predictions with actual outcomes yet."}
    return stats


def _get_metrics(region_key: str) -> Optional[dict]:
    path = f"{MODELS_DIR}/{region_key}_metrics.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@app.get("/model/comparison")
def model_comparison(region: str):
    _validate_region(region)
    metrics = _get_metrics(region)
    if metrics is None:
        raise HTTPException(status_code=503, detail="Metrics file not found. Run train_model.py first.")
    return metrics


@app.post("/shedding/schedule")
def shedding_schedule(req: ShedRequest):
    _validate_region(req.region)
    cfg = REGIONS[req.region]
    hours, loads, solars = _build_24h_predictions(req.region, req.temperature_c, req.is_weekend, use_custom=req.use_custom)

    drought_info = drought.apply_drought_scenario(cfg["available_capacity_mw"], req.drought_scenario)
    effective_capacity = drought_info["adjusted_capacity_mw"]

    # Deficit is measured on NET load (load minus solar), since solar generation
    # directly reduces what the grid/dispatchable capacity needs to cover.
    net_loads = [l - s for l, s in zip(loads, solars)]
    forecast = [{"hour": h, "predicted_load_mw": nl} for h, nl in zip(hours, net_loads)]
    schedule = shedding.generate_shedding_schedule(forecast, effective_capacity, cfg["num_zones"])
    summary = shedding.summarize_schedule(schedule)

    return {"region": req.region, "capacity_mw": effective_capacity,
            "num_zones": cfg["num_zones"], "schedule": schedule, "summary": summary,
            "drought_scenario": drought_info,
            "note": "Deficit is based on net load (grid load minus solar generation)."}


@app.get("/drought/scenarios")
def drought_scenarios():
    return drought.DROUGHT_SCENARIOS


@app.post("/explain/load")
def explain_load(req: LoadRequest):
    _validate_region(req.region)
    models = _load_region_models(req.region, False)
    return explainability.explain_load_prediction(models["load"], req.hour, req.temperature_c, req.is_weekend)


@app.post("/battery/schedule")
def battery_schedule(req: BatteryRequest):
    _validate_region(req.region)
    cfg = REGIONS[req.region]
    hours, loads, solars = _build_24h_predictions(
        req.region, req.temperature_c, req.is_weekend, req.cloud_cover_pct, req.month, req.use_custom
    )

    schedule = battery.simulate_battery_schedule(
        loads, solars, cfg["battery_capacity_mwh"], req.initial_soc_frac
    )
    summary = battery.summarize_battery_schedule(schedule)

    solar_series = [s["solar_mw"] for s in schedule]
    discharge_series = [s["discharge_mwh"] for s in schedule]
    daily_impact = impact.estimate_daily_impact(solar_series, discharge_series)
    annual_impact = impact.project_annual_impact(daily_impact)

    return {"region": req.region, "battery_capacity_mwh": cfg["battery_capacity_mwh"],
            "schedule": schedule, "summary": summary,
            "impact": {**daily_impact, **annual_impact}}


def _safe_load_region_models(region_key: str):
    """Like _load_region_models but returns None instead of raising, for the SMS interface
    which needs to reply with plain text even when something's missing."""
    try:
        return _load_region_models(region_key)
    except HTTPException:
        return None


@app.post("/sms/query")
def sms_query(req: SmsRequest):
    result = sms_interface.handle_sms_command(req.text, _safe_load_region_models, req.lang)
    return result


@app.get("/model/custom-status")
def custom_model_status(region: str):
    _validate_region(region)
    return data_ingestion.custom_model_exists(region)


@app.post("/retrain")
async def retrain(region: str = Form(...), file: UploadFile = File(...)):
    _validate_region(region)

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    validation = data_ingestion.validate_and_clean(df)
    if not validation["success"]:
        return {"success": False, "issues": validation["issues"],
                "rows_before": validation["rows_before"], "rows_after": validation["rows_after"]}

    train_result = data_ingestion.retrain_from_dataframe(
        validation["clean_df"], region, validation["has_solar_columns"]
    )

    # Invalidate any cached custom models for this region so the next prediction picks up the new ones
    for key in list(_models.keys()):
        if key[0] == region and key[1] is True:
            del _models[key]

    return {"success": True, "issues": validation["issues"], **train_result}


@app.post("/impact/daily")
def impact_daily(req: BatteryRequest):
    _validate_region(req.region)
    cfg = REGIONS[req.region]
    hours, loads, solars = _build_24h_predictions(
        req.region, req.temperature_c, req.is_weekend, req.cloud_cover_pct, req.month, req.use_custom
    )
    schedule = battery.simulate_battery_schedule(loads, solars, cfg["battery_capacity_mwh"], req.initial_soc_frac)
    solar_series = [s["solar_mw"] for s in schedule]
    discharge_series = [s["discharge_mwh"] for s in schedule]
    daily_impact = impact.estimate_daily_impact(solar_series, discharge_series)
    annual_impact = impact.project_annual_impact(daily_impact)
    return {"region": req.region, **daily_impact, **annual_impact}


# ---------------- Zambian holiday calendar ----------------

@app.get("/holidays/upcoming")
def holidays_upcoming(count: int = 5):
    import datetime
    results = holidays_zm.upcoming_holidays(datetime.date.today(), count)
    return [{"date": d.isoformat(), "name": name} for d, name in results]


@app.get("/holidays/check")
def holidays_check(date: str):
    import datetime
    try:
        d = datetime.date.fromisoformat(date)
    except ValueError:
        raise HTTPException(status_code=400, detail="date must be in YYYY-MM-DD format")
    name = holidays_zm.is_public_holiday(d)
    return {"date": date, "is_holiday": name is not None, "holiday_name": name,
            "effective_is_weekend": holidays_zm.effective_is_weekend(d)}


# ---------------- Shedding execution tracking (plan vs. reality) ----------------

@app.post("/shedding/log-plan")
def log_shedding_plan(req: LogPlanRequest):
    import datetime
    _validate_region(req.region)
    try:
        shed_date = datetime.date.fromisoformat(req.shed_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="shed_date must be in YYYY-MM-DD format")

    logged_ids = []
    for entry in req.schedule:
        for zone in entry.get("zones_to_shed", []):
            entry_id = execution_tracking.log_planned_shedding(req.region, shed_date, entry["hour"], zone)
            logged_ids.append({"entry_id": entry_id, "hour": entry["hour"], "zone": zone})

    return {"success": True, "logged": logged_ids}


@app.post("/shedding/mark-execution")
def mark_shedding_execution(req: MarkExecutionRequest):
    if req.status not in ("executed", "skipped"):
        raise HTTPException(status_code=400, detail="status must be 'executed' or 'skipped'")
    ok = execution_tracking.mark_execution_status(req.entry_id, req.status, req.notes)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Entry {req.entry_id} not found.")
    return {"success": True}


@app.get("/shedding/planned-entries")
def get_shedding_planned_entries(region: str, shed_date: str):
    import datetime
    _validate_region(region)
    try:
        d = datetime.date.fromisoformat(shed_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="shed_date must be in YYYY-MM-DD format")
    entries = execution_tracking.get_planned_entries(region, d)
    return [{"id": e.id, "hour": e.hour, "zone": e.zone, "status": e.status, "notes": e.notes} for e in entries]


@app.get("/shedding/compliance-summary")
def shedding_compliance_summary(region: Optional[str] = None, days_back: int = 30):
    return execution_tracking.get_compliance_summary(region, days_back)


# ---------------- Community notice ----------------

@app.post("/notice/text")
def notice_text(req: NoticeRequest):
    import datetime
    _validate_region(req.region)
    cfg = REGIONS[req.region]
    notice_date = datetime.date.fromisoformat(req.notice_date) if req.notice_date else None
    text = notice.generate_notice_text(cfg["label"], req.schedule, notice_date)
    return {"text": text}


@app.post("/notice/image")
def notice_image(req: NoticeRequest):
    import datetime
    _validate_region(req.region)
    cfg = REGIONS[req.region]
    notice_date = datetime.date.fromisoformat(req.notice_date) if req.notice_date else None
    img = notice.generate_notice_image(cfg["label"], req.schedule, notice_date)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# ---------------- Regional map ----------------

@app.get("/map/regions")
def map_regions(temperature_c: float = 30.0, is_weekend: int = 0, cloud_cover_pct: float = 20.0, month: int = 9):
    """Returns per-region risk status + coordinates, ready for the map view."""
    points = []
    for region_key, cfg in REGIONS.items():
        models = _load_region_models(region_key, False)
        load_input = pd.DataFrame([{"hour": 19, "temperature_c": temperature_c, "is_weekend": is_weekend}])
        predicted_load = float(models["load"].predict(load_input)[0])
        solar_input = pd.DataFrame([{"hour": 19, "cloud_cover_pct": cloud_cover_pct, "month": month}])
        predicted_solar = max(float(models["solar"].predict(solar_input)[0]), 0)
        net_load = predicted_load - predicted_solar

        points.append({
            "region_key": region_key,
            "label": cfg["label"],
            "lat": cfg["lat"],
            "lon": cfg["lon"],
            "predicted_load_mw": round(predicted_load, 2),
            "net_load_mw": round(net_load, 2),
            "at_risk": predicted_load >= cfg["alert_threshold_mw"],
        })
    return {"regions": points}


# ---------------- Stakeholder PDF report ----------------

@app.post("/report/pdf")
def report_pdf(req: ReportRequest):
    import datetime
    _validate_region(req.region)
    cfg = REGIONS[req.region]
    report_date = datetime.date.fromisoformat(req.report_date) if req.report_date else datetime.date.today()

    hours, loads, solars = _build_24h_predictions(req.region, req.temperature_c, req.is_weekend, req.cloud_cover_pct, req.month)

    drought_info = drought.apply_drought_scenario(cfg["available_capacity_mw"], req.drought_scenario)
    net_loads = [l - s for l, s in zip(loads, solars)]
    forecast = [{"hour": h, "predicted_load_mw": nl} for h, nl in zip(hours, net_loads)]
    schedule = shedding.generate_shedding_schedule(forecast, drought_info["adjusted_capacity_mw"], cfg["num_zones"])
    shed_summary = shedding.summarize_schedule(schedule)

    battery_schedule = battery.simulate_battery_schedule(loads, solars, cfg["battery_capacity_mwh"], 0.5)
    solar_series = [s["solar_mw"] for s in battery_schedule]
    discharge_series = [s["discharge_mwh"] for s in battery_schedule]
    daily_impact = impact.estimate_daily_impact(solar_series, discharge_series)
    annual_impact = impact.project_annual_impact(daily_impact)
    impact_combined = {**daily_impact, **annual_impact}

    notice_text = notice.generate_notice_text(cfg["label"], schedule, report_date)

    forecast_summary = {
        "peak_load_mw": max(loads),
        "peak_hour": hours[loads.index(max(loads))],
        "avg_load_mw": sum(loads) / len(loads),
    }

    drought_note = drought_info["note"] if req.drought_scenario != "none" else None

    pdf_bytes = report.generate_stakeholder_report(
        cfg["label"], report_date, forecast_summary, shed_summary,
        impact_combined, notice_text, drought_note
    )

    buf = io.BytesIO(pdf_bytes)
    return StreamingResponse(buf, media_type="application/pdf",
                              headers={"Content-Disposition": f"attachment; filename={req.region}_report_{report_date}.pdf"})


# ---------------- Webhook alerts ----------------

@app.post("/webhooks/register")
def webhook_register(req: WebhookRegisterRequest):
    if req.region != "*":
        _validate_region(req.region)
    webhook_id = webhooks.register_webhook(req.region, req.url)
    return {"success": True, "webhook_id": webhook_id}


@app.get("/webhooks/list")
def webhook_list(region: Optional[str] = None):
    subs = webhooks.list_webhooks(region)
    return [{"id": s.id, "region": s.region, "url": s.url} for s in subs]


@app.post("/webhooks/deactivate/{webhook_id}")
def webhook_deactivate(webhook_id: int):
    ok = webhooks.deactivate_webhook(webhook_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Webhook {webhook_id} not found.")
    return {"success": True}


@app.post("/webhooks/dispatch")
def webhook_dispatch(req: WebhookDispatchRequest):
    _validate_region(req.region)
    message = req.message or f"Predicted load {req.predicted_load_mw:.1f} MW for {req.region}"
    payload = {"region": req.region, "predicted_load_mw": req.predicted_load_mw, "message": message}
    results = webhooks.dispatch_alert_to_webhooks(req.region, payload)
    return {"dispatched_to": len(results), "results": results}


def _chat_provider_status():
    requested_provider = os.getenv("NYIKA_CHAT_PROVIDER", "auto").strip().lower()
    api_key = os.getenv("OPENAI_API_KEY")

    if requested_provider not in ("auto", "openai", "ollama"):
        return {"available": False, "provider": requested_provider,
                "model": None, "message": "NYIKA_CHAT_PROVIDER must be auto, openai, or ollama."}

    if requested_provider in ("auto", "openai") and api_key:
        return {
            "available": True,
            "provider": "openai-compatible",
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            "api_key": api_key,
            "message": "Generative assistant ready.",
        }

    if requested_provider == "openai":
        return {"available": False, "provider": "openai-compatible", "model": None,
                "message": "Set OPENAI_API_KEY in the API server environment to enable hosted chat."}

    ollama_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=1.5)
        response.raise_for_status()
        installed_models = {model.get("name") for model in response.json().get("models", [])}
        if ollama_model not in installed_models:
            return {"available": False, "provider": "ollama", "model": ollama_model,
                    "message": f"Run `ollama pull {ollama_model}` to install the configured local chat model."}
        return {"available": True, "provider": "ollama", "model": ollama_model,
                "host": ollama_host, "message": "Generative assistant ready."}
    except Exception:
        return {"available": False, "provider": "ollama", "model": ollama_model,
                "message": "Start Ollama locally, or configure OPENAI_API_KEY for hosted chat."}


@app.get("/chat/status")
def chat_status():
    status = _chat_provider_status()
    return {key: value for key, value in status.items() if key not in ("api_key", "host", "base_url")}


def _build_chat_context(req: ChatRequest) -> dict:
    _validate_region(req.region)
    cfg = REGIONS[req.region]
    hours, loads, solars = _build_24h_predictions(
        req.region, req.temperature_c, req.is_weekend, req.cloud_cover_pct, req.month, req.use_custom
    )
    net_loads = [float(load - solar) for load, solar in zip(loads, solars)]
    current_load = float(loads[req.hour])
    current_solar = float(solars[req.hour])
    current_net_load = net_loads[req.hour]

    shed_forecast = [
        {"hour": hour, "predicted_load_mw": net_load}
        for hour, net_load in zip(hours, net_loads)
    ]
    shed_schedule = shedding.generate_shedding_schedule(
        shed_forecast, cfg["available_capacity_mw"], cfg["num_zones"]
    )
    shed_summary = shedding.summarize_schedule(shed_schedule)

    battery_schedule = battery.simulate_battery_schedule(
        loads, solars, cfg["battery_capacity_mwh"], req.initial_soc_frac
    )
    battery_summary = battery.summarize_battery_schedule(battery_schedule)
    selected_battery_hour = battery_schedule[req.hour]

    return {
        "region": cfg["label"],
        "as_of_date": __import__("datetime").date.today().isoformat(),
        "conditions": {
            "selected_hour": req.hour,
            "temperature_c": req.temperature_c,
            "cloud_cover_pct": req.cloud_cover_pct,
            "is_weekend": bool(req.is_weekend),
            "month": req.month,
        },
        "forecast_at_selected_hour": {
            "load_mw": round(current_load, 1),
            "solar_mw": round(current_solar, 1),
            "net_load_mw": round(current_net_load, 1),
            "alert_threshold_mw": cfg["alert_threshold_mw"],
            "risk": "elevated" if current_load >= cfg["alert_threshold_mw"] else "stable",
        },
        "forecast_24h": {
            "peak_load_mw": round(max(loads), 1),
            "peak_load_hour": hours[int(np.argmax(loads))],
            "peak_net_load_mw": round(max(net_loads), 1),
        },
        "load_shedding": {
            "daily_summary": shed_summary,
            "selected_hour": shed_schedule[req.hour],
        },
        "battery": {
            "capacity_mwh": cfg["battery_capacity_mwh"],
            "daily_summary": battery_summary,
            "selected_hour": selected_battery_hour,
        },
        "model_data_note": "Forecasts are illustrative and based on simulated training data unless a custom model is selected.",
    }


@app.post("/chat")
def chat(req: ChatRequest):
    if req.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="The latest chat message must be from the user.")

    provider = _chat_provider_status()
    if not provider["available"]:
        raise HTTPException(status_code=503, detail=provider["message"])

    context = _build_chat_context(req)
    system_message = (
        "You are Nyika-Grid Assistant, a concise assistant for explaining Zambia grid forecasts. "
        "Use LIVE_CONTEXT as the sole source for current system values. Do not invent measurements, "
        "claim an alert was delivered, or say you controlled the grid. Clearly describe outputs as "
        "forecasts, mention the simulated-data limitation when relevant, and say when context does "
        "not contain an answer. You may explain schedules but cannot execute operational actions.\n\n"
        "LIVE_CONTEXT:\n" + json.dumps(context, ensure_ascii=True)
    )
    messages = [{"role": "system", "content": system_message}]
    messages.extend({"role": item.role, "content": item.content} for item in req.messages[-10:])

    try:
        if provider["provider"] == "ollama":
            response = requests.post(
                f"{provider['host']}/api/chat",
                json={"model": provider["model"], "messages": messages, "stream": False,
                      "options": {"temperature": 0.2}},
                timeout=90,
            )
            response.raise_for_status()
            reply = response.json().get("message", {}).get("content", "").strip()
        else:
            response = requests.post(
                f"{provider['base_url']}/chat/completions",
                headers={"Authorization": f"Bearer {provider['api_key']}"},
                json={"model": provider["model"], "messages": messages, "temperature": 0.2},
                timeout=90,
            )
            response.raise_for_status()
            reply = response.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        raise HTTPException(status_code=502, detail="The generative model request failed. Check the configured provider and model.") from exc

    if not reply:
        raise HTTPException(status_code=502, detail="The configured model returned an empty response.")
    return {"reply": reply, "provider": provider["provider"], "model": provider["model"]}
