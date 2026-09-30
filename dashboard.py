"""
Nyika-Grid Predict AI — Dashboard (v2)
-------------------------------------------
Loads trained models directly (no need for the API to be running).

Usage:
    1. Run train_model.py first to produce the model files in ./models/
    2. streamlit run dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import datetime
import io
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from data_utils import REGIONS
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
import json

st.set_page_config(page_title="Nyika-Grid Predict AI", page_icon="⚡", layout="wide")

MODELS_DIR = "models"


@st.cache_data
def load_metrics(region_key):
    path = f"{MODELS_DIR}/{region_key}_metrics.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_resource
def load_models(region_key, use_custom=False):
    prophet_path = f"{MODELS_DIR}/{region_key}_prophet.pkl"
    if not os.path.exists(prophet_path):
        return None

    loaded = {"prophet": joblib.load(prophet_path)}
    used_custom = {"load": False, "solar": False}

    for sub in ("load", "solar"):
        custom_path = f"{MODELS_DIR}/{region_key}_{sub}_custom.pkl"
        default_path = f"{MODELS_DIR}/{region_key}_{sub}.pkl"
        if use_custom and os.path.exists(custom_path):
            loaded[sub] = joblib.load(custom_path)
            used_custom[sub] = True
        elif os.path.exists(default_path):
            loaded[sub] = joblib.load(default_path)
        else:
            return None

    loaded["used_custom"] = used_custom
    return loaded


def custom_model_status(region_key):
    return data_ingestion.custom_model_exists(region_key)


st.title("⚡ Nyika-Grid Predict AI")
st.caption("Multi-region electricity load + solar generation forecaster for Zambia — prototype")

# ---------------- Session state defaults (must exist before widgets are created) ----------------
if "temperature" not in st.session_state:
    st.session_state.temperature = 30
if "cloud_cover" not in st.session_state:
    st.session_state.cloud_cover = 20
if "weather_status" not in st.session_state:
    st.session_state.weather_status = None


# ---------------- Callbacks (run BEFORE widgets re-render, so session_state updates are safe) ----------------
def _apply_heatwave():
    st.session_state.temperature = min(42, st.session_state.temperature + 5)


def _apply_cloudy_day():
    st.session_state.cloud_cover = min(100, st.session_state.cloud_cover + 60)


def _apply_live_weather():
    cfg = REGIONS[st.session_state.region_key]
    result = weather.fetch_live_weather(cfg["lat"], cfg["lon"])
    if result["success"]:
        st.session_state.temperature = round(result["temperature_c"])
        st.session_state.cloud_cover = round(result["cloud_cover_pct"])
        st.session_state.weather_status = ("success", None)
    else:
        st.session_state.weather_status = ("error", result["error"])


# ---------------- Sidebar: region + conditions ----------------
with st.sidebar:
    st.header("Settings")

    region_key = st.selectbox(
        "Region",
        options=list(REGIONS.keys()),
        format_func=lambda k: REGIONS[k]["label"],
        key="region_key"
    )
    cfg = REGIONS[region_key]

    custom_status = custom_model_status(region_key)
    use_custom_model = False
    if custom_status["load"] or custom_status["solar"]:
        use_custom_model = st.checkbox(
            "🧪 Use my custom-retrained model",
            value=False,
            help="Uses the model retrained on your uploaded CSV (see the Upload & Retrain tab) instead of the demo model."
        )
        which = [k for k, v in custom_status.items() if v]
        st.caption(f"Custom model available for: {', '.join(which)}")

    models = load_models(region_key, use_custom_model)
    if models is None:
        st.error("Model files not found for this region. Run `python train_model.py` first.")
        st.stop()

    if use_custom_model and (models["used_custom"]["load"] or models["used_custom"]["solar"]):
        used = [k for k, v in models["used_custom"].items() if v]
        st.success(f"Using custom model for: {', '.join(used)}")

    st.divider()
    st.subheader("Conditions")

    hour = st.slider("Hour of day", 0, 23, 19)
    temperature = st.slider("Temperature (°C)", 15, 42, key="temperature")
    cloud_cover = st.slider("Cloud cover (%)", 0, 100, key="cloud_cover")

    plan_date = st.date_input("Date (for weekend/holiday detection)", value=datetime.date.today(), key="plan_date")
    holiday_name = holidays_zm.is_public_holiday(plan_date)
    auto_is_weekend = bool(holidays_zm.effective_is_weekend(plan_date))
    if holiday_name:
        st.caption(f"📅 {plan_date.strftime('%d %b')} is **{holiday_name}** — treated as weekend-like demand (see MODEL_CARD.md for why).")
    is_weekend = st.checkbox("Weekend?", value=auto_is_weekend, key="is_weekend_check")
    month = st.slider("Month", 1, 12, 9)

    st.divider()
    st.subheader("What-if scenarios")
    col_a, col_b = st.columns(2)
    with col_a:
        st.button("🔥 Heatwave (+5°C)", on_click=_apply_heatwave)
    with col_b:
        st.button("☁️ Cloudy day (+60%)", on_click=_apply_cloudy_day)

    st.divider()
    st.subheader("Live weather")
    st.button("🌐 Fetch live weather for region", on_click=_apply_live_weather)
    if st.session_state.weather_status:
        status, error = st.session_state.weather_status
        if status == "success":
            st.success("Live weather applied.")
        else:
            st.warning(f"Could not fetch live weather ({error}). Using manual values instead.")

# ---------------- Predictions ----------------
load_input = pd.DataFrame([{"hour": hour, "temperature_c": float(temperature), "is_weekend": int(is_weekend)}])
predicted_load = float(models["load"].predict(load_input)[0])

solar_input = pd.DataFrame([{"hour": hour, "cloud_cover_pct": float(cloud_cover), "month": month}])
predicted_solar = max(float(models["solar"].predict(solar_input)[0]), 0)

net_load = predicted_load - predicted_solar

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 Live Prediction", "📅 7-Day Forecast", "🗺️ Regional Overview", "📈 Historical Accuracy",
    "🧠 Smart Recommendations", "📱 SMS Simulator", "📤 Upload & Retrain", "📋 Plan vs. Reality",
    "🔔 Integrations"
])

# ---------------- Tab 1: Live prediction ----------------
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Grid Load", f"{predicted_load:.1f} MW")
    col2.metric("Predicted Solar Generation", f"{predicted_solar:.1f} MW")
    col3.metric("Net Load (Load − Solar)", f"{net_load:.1f} MW")

    alert_result = alerts.check_load_alert(predicted_load, cfg["alert_threshold_mw"])
    if alert_result["triggered"]:
        st.error(f"🚨 {alert_result['message']}")
        if st.button("Send alert to operations team"):
            alerts.send_alert(alert_result["message"], method="log", recipient=f"{region_key}-operations")
            st.info("Alert logged to alerts.log (SMS/email dispatch itself is simulated).")

            deliveries = webhooks.dispatch_alert_to_webhooks(region_key, {
                "region": region_key,
                "predicted_load_mw": predicted_load,
                "threshold_mw": cfg["alert_threshold_mw"],
                "level": alert_result["level"],
                "message": alert_result["message"],
            })
            if deliveries:
                ok = sum(1 for d in deliveries if d["success"])
                st.caption(f"🔔 Webhooks: {ok} of {len(deliveries)} delivered. See the Integrations tab for details.")
                for d in deliveries:
                    if not d["success"]:
                        st.warning(f"Webhook #{d['webhook_id']} failed: {d.get('error', 'HTTP ' + str(d.get('status_code')))}")
    else:
        st.success(f"✅ {alert_result['message']}")

    if st.button("💾 Log this prediction"):
        log_id = db.log_prediction(
            region=region_key, hour=hour, temperature_c=float(temperature),
            is_weekend=int(is_weekend), predicted_load_mw=predicted_load,
            predicted_solar_mw=predicted_solar
        )
        st.success(f"Logged as entry #{log_id}. Add the actual outcome later in the Historical Accuracy tab.")

    st.divider()
    st.subheader("24-hour outlook")
    st.caption(f"Holding temperature at {temperature}°C, cloud cover at {cloud_cover}%, weekend={is_weekend}")

    hours_range = list(range(24))
    load_df = pd.DataFrame([{"hour": h, "temperature_c": float(temperature), "is_weekend": int(is_weekend)} for h in hours_range])
    solar_df = pd.DataFrame([{"hour": h, "cloud_cover_pct": float(cloud_cover), "month": month} for h in hours_range])

    load_series = models["load"].predict(load_df)
    solar_series = np.clip(models["solar"].predict(solar_df), 0, None)
    net_series = load_series - solar_series

    chart_df = pd.DataFrame({
        "hour": hours_range, "load_mw": load_series, "solar_mw": solar_series, "net_load_mw": net_series
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_df["hour"], y=chart_df["load_mw"], name="Grid Load", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=chart_df["hour"], y=chart_df["solar_mw"], name="Solar Generation", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=chart_df["hour"], y=chart_df["net_load_mw"], name="Net Load", mode="lines+markers", line=dict(dash="dot")))
    fig.add_hline(y=cfg["alert_threshold_mw"], line_dash="dash", line_color="red",
                  annotation_text="Alert threshold", annotation_position="top left")
    fig.update_layout(xaxis_title="Hour of day", yaxis_title="MW", xaxis=dict(dtick=1), height=420)
    st.plotly_chart(fig, width='stretch')

    csv_data = chart_df.to_csv(index=False)
    st.download_button("⬇️ Export 24-hour forecast as CSV", data=csv_data,
                        file_name=f"{region_key}_24h_forecast.csv", mime="text/csv")

    st.divider()
    st.subheader("Why trust this model?")
    metrics = load_metrics(region_key)
    if metrics:
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Model RMSE", f"{metrics['load_model_rmse']} MW")
        mcol2.metric("Naive baseline RMSE", f"{metrics['baseline_rmse']} MW")
        mcol3.metric("Improvement", f"{metrics['improvement_over_baseline_pct']}%",
                     help="How much more accurate the trained model is vs. simply predicting the historical average for that hour.")

        st.caption("What drives the model's predictions (feature importance):")
        importances = metrics["load_feature_importances"]
        imp_fig = go.Figure(go.Bar(
            x=list(importances.values()), y=list(importances.keys()), orientation="h"
        ))
        imp_fig.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=10),
                               xaxis_title="Relative importance")
        st.plotly_chart(imp_fig, width='stretch')
    else:
        st.info("Model metrics not found — retrain with train_model.py to see this comparison.")

    st.divider()
    st.subheader("Why is THIS forecast what it is?")
    st.caption(
        "Global feature importance (above) shows what matters on average. This shows what's "
        "driving *this specific* prediction, compared to typical conditions (midday, 25°C, weekday). "
        "Method: feature ablation — not the SHAP library — see explainability.py for what that means."
    )
    explanation = explainability.explain_load_prediction(models["load"], hour, float(temperature), int(is_weekend))
    st.caption(f"Typical/baseline conditions predict {explanation['baseline_prediction']} MW. "
               f"Your selected conditions predict {explanation['prediction']} MW. Difference driven by:")

    contrib = explanation["feature_contributions_mw"]
    exp_fig = go.Figure(go.Bar(
        x=list(contrib.values()), y=list(contrib.keys()), orientation="h",
        marker_color=["#B8703A" if v >= 0 else "#5C7A52" for v in contrib.values()]
    ))
    exp_fig.update_layout(height=180, margin=dict(l=0, r=0, t=10, b=10), xaxis_title="Contribution (MW)")
    st.plotly_chart(exp_fig, width='stretch')

# ---------------- Tab 2: 7-day forecast ----------------
with tab2:
    st.subheader(f"7-day hourly forecast — {cfg['label']}")
    st.caption("Generated with Prophet, including confidence intervals (shaded band).")

    future = models["prophet"].make_future_dataframe(periods=24 * 7, freq="h")
    forecast = models["prophet"].predict(future)
    forecast_tail = forecast.tail(24 * 7)

    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(
        x=forecast_tail["ds"], y=forecast_tail["yhat_upper"],
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    fig7.add_trace(go.Scatter(
        x=forecast_tail["ds"], y=forecast_tail["yhat_lower"],
        fill="tonexty", fillcolor="rgba(99,110,250,0.15)", line=dict(width=0),
        name="Confidence interval"
    ))
    fig7.add_trace(go.Scatter(
        x=forecast_tail["ds"], y=forecast_tail["yhat"],
        line=dict(color="rgb(99,110,250)"), name="Predicted load"
    ))
    fig7.update_layout(xaxis_title="Date", yaxis_title="Predicted Load (MW)", height=450)
    st.plotly_chart(fig7, width='stretch')

    csv7 = forecast_tail[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
        columns={"ds": "timestamp", "yhat": "predicted_load_mw",
                 "yhat_lower": "lower_bound_mw", "yhat_upper": "upper_bound_mw"}
    ).to_csv(index=False)
    st.download_button("⬇️ Export 7-day forecast as CSV", data=csv7,
                        file_name=f"{region_key}_7day_forecast.csv", mime="text/csv")

# ---------------- Tab 3: Regional overview ----------------
with tab3:
    st.subheader("Regional risk overview")
    st.caption(f"All regions evaluated at hour={hour}, temperature={temperature}°C, cloud cover={cloud_cover}%, weekend={is_weekend}")

    region_points = []
    region_cards = []
    for rkey, rcfg in REGIONS.items():
        rmodels = load_models(rkey)
        if rmodels is None:
            region_cards.append({"label": rcfg["label"], "missing": True})
            continue

        r_load_input = pd.DataFrame([{"hour": hour, "temperature_c": float(temperature), "is_weekend": int(is_weekend)}])
        r_load = float(rmodels["load"].predict(r_load_input)[0])

        r_solar_input = pd.DataFrame([{"hour": hour, "cloud_cover_pct": float(cloud_cover), "month": month}])
        r_solar = max(float(rmodels["solar"].predict(r_solar_input)[0]), 0)

        r_net = r_load - r_solar
        r_alert = alerts.check_load_alert(r_load, rcfg["alert_threshold_mw"])

        region_points.append({
            "label": rcfg["label"], "lat": rcfg["lat"], "lon": rcfg["lon"],
            "predicted_load_mw": r_load, "net_load_mw": r_net, "at_risk": r_alert["triggered"],
        })
        region_cards.append({
            "label": rcfg["label"], "missing": False, "load": r_load, "solar": r_solar,
            "net": r_net, "triggered": r_alert["triggered"],
        })

    if region_points:
        st.caption("🗺️ Map view — marker color shows current risk status (red = elevated, green = stable). "
                   "Positions are approximate region centers, not individual sites.")
        st.plotly_chart(map_view.build_region_map(region_points), width='stretch')

    cols = st.columns(len(region_cards))
    for i, card in enumerate(region_cards):
        with cols[i]:
            if card["missing"]:
                st.warning(f"{card['label']}: models not found.")
                continue
            st.markdown(f"**{card['label']}**")
            st.metric("Load", f"{card['load']:.1f} MW")
            st.metric("Solar", f"{card['solar']:.1f} MW")
            st.metric("Net Load", f"{card['net']:.1f} MW")
            if card["triggered"]:
                st.error("🔴 Elevated risk")
            else:
                st.success("🟢 Stable")

# ---------------- Tab 4: Historical accuracy ----------------
with tab4:
    st.subheader("Historical accuracy tracker")
    st.caption("Log predictions in the Live Prediction tab, then record actual outcomes here to build an accuracy history.")

    stats = db.get_accuracy_stats(region_key)
    if stats:
        c1, c2, c3 = st.columns(3)
        c1.metric("Logged comparisons", stats["count"])
        c2.metric("RMSE", f"{stats['rmse']:.2f} MW")
        c3.metric("MAE", f"{stats['mae']:.2f} MW")
    else:
        st.info("No actual outcomes logged yet for this region.")

    st.divider()
    st.subheader("Recent logged predictions")
    logs = db.get_recent_logs(limit=10, region=region_key)

    if not logs:
        st.write("No predictions logged yet. Use the 'Log this prediction' button in the Live Prediction tab.")
    else:
        for log in logs:
            with st.expander(f"#{log.id} — {log.timestamp.strftime('%Y-%m-%d %H:%M')} — predicted {log.predicted_load_mw:.1f} MW"):
                st.write(f"Hour: {log.hour}, Temp: {log.temperature_c}°C, Weekend: {bool(log.is_weekend)}")
                st.write(f"Predicted load: {log.predicted_load_mw:.2f} MW | Predicted solar: {log.predicted_solar_mw}")
                if log.actual_load_mw is not None:
                    st.write(f"Actual load: {log.actual_load_mw:.2f} MW")
                else:
                    actual_val = st.number_input(f"Enter actual load (MW) for #{log.id}", min_value=0.0, key=f"actual_{log.id}")
                    if st.button(f"Save actual for #{log.id}", key=f"save_{log.id}"):
                        db.log_actual(log.id, actual_load_mw=actual_val)

                        region_metrics = load_metrics(log.region)
                        baseline_rmse = region_metrics["load_model_rmse"] if region_metrics else 15.0
                        anomaly_result = anomaly.check_anomaly(log.predicted_load_mw, actual_val, baseline_rmse)

                        if anomaly_result["is_anomaly"]:
                            st.warning(f"⚠️ Anomaly detected: {anomaly_result['message']}")
                        else:
                            st.success("Saved. Reading looks normal — no anomaly detected.")
                        st.rerun()

# ---------------- Tab 5: Smart recommendations (shedding + battery) ----------------
with tab5:
    st.subheader(f"Smart recommendations — {cfg['label']}")
    st.caption("Turns the forecast into an actionable plan, not just a number.")

    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        rec_temperature = st.slider("Assumed temperature (°C) for the day", 15, 42, int(temperature), key="rec_temp")
    with rec_col2:
        rec_weekend = st.checkbox("Assume weekend", value=bool(is_weekend), key="rec_weekend")

    st.markdown("#### 🔌 Load-shedding schedule")

    drought_key = st.selectbox(
        "Drought stress-test scenario",
        options=list(drought.DROUGHT_SCENARIOS.keys()),
        format_func=lambda k: drought.DROUGHT_SCENARIOS[k]["label"],
        key="drought_scenario",
        help="Replays this region's shedding plan against real points in Zambia's 2024 hydropower crisis timeline."
    )
    drought_info = drought.apply_drought_scenario(cfg["available_capacity_mw"], drought_key)
    if drought_key != "none":
        st.warning(f"📉 {drought_info['note']} (~{drought_info['reduction_pct']}% capacity reduction applied — see drought.py for method/sourcing)")
    effective_capacity = drought_info["adjusted_capacity_mw"]

    st.caption(f"Available capacity: {effective_capacity} MW (of {cfg['available_capacity_mw']} MW normal) "
               f"across {cfg['num_zones']} zones. Deficit is based on net load (grid load minus solar).")

    hours_range2 = list(range(24))
    shed_load_df = pd.DataFrame([{"hour": h, "temperature_c": float(rec_temperature), "is_weekend": int(rec_weekend)} for h in hours_range2])
    shed_solar_df = pd.DataFrame([{"hour": h, "cloud_cover_pct": float(cloud_cover), "month": month} for h in hours_range2])

    shed_loads = models["load"].predict(shed_load_df)
    shed_solars = np.clip(models["solar"].predict(shed_solar_df), 0, None)
    net_loads = shed_loads - shed_solars

    forecast_entries = [{"hour": h, "predicted_load_mw": float(nl)} for h, nl in zip(hours_range2, net_loads)]
    shed_schedule = shedding.generate_shedding_schedule(forecast_entries, effective_capacity, cfg["num_zones"])
    shed_summary = shedding.summarize_schedule(shed_schedule)

    if shed_summary["total_hours_with_shedding"] == 0:
        st.success("✅ No shedding needed under these conditions — net load stays within capacity all day.")
    else:
        st.error(f"⚠️ Shedding needed in {shed_summary['total_hours_with_shedding']} of 24 hours. "
                 f"Peak deficit: {shed_summary['peak_deficit_mw']} MW.")
        st.caption(f"Shed hours per zone (fairness check): {shed_summary['shed_hours_per_zone']}")

        shed_display_df = pd.DataFrame([
            {"Hour": s["hour"], "Net Load (MW)": s["predicted_load_mw"], "Deficit (MW)": s["deficit_mw"],
             "Zones to shed": ", ".join(s["zones_to_shed"]) if s["zones_to_shed"] else "—"}
            for s in shed_schedule
        ])
        st.dataframe(shed_display_df, hide_index=True, height=300)

        st.markdown("###### 📋 Turn this into action")
        ncol1, ncol2 = st.columns(2)
        with ncol1:
            if st.button("📝 Log this plan for execution tracking"):
                logged = []
                for s in shed_schedule:
                    for zone in s["zones_to_shed"]:
                        entry_id = execution_tracking.log_planned_shedding(region_key, plan_date, s["hour"], zone)
                        logged.append(entry_id)
                st.success(f"Logged {len(logged)} planned shedding actions for {plan_date}. "
                           f"Review and confirm them in the 'Plan vs. Reality' tab.")
        with ncol2:
            notice_text = notice.generate_notice_text(cfg["label"], shed_schedule, plan_date)
            st.download_button("⬇️ Download community notice (text)", data=notice_text,
                               file_name=f"{region_key}_notice_{plan_date}.txt", mime="text/plain")

        with st.expander("Preview community notice"):
            st.code(notice_text, language=None)
            notice_img = notice.generate_notice_image(cfg["label"], shed_schedule, plan_date)
            img_buf = io.BytesIO()
            notice_img.save(img_buf, format="PNG")
            st.image(img_buf.getvalue())
            st.download_button("⬇️ Download community notice (image)", data=img_buf.getvalue(),
                               file_name=f"{region_key}_notice_{plan_date}.png", mime="image/png")

    st.markdown("###### 📄 Stakeholder report")
    st.caption("One PDF combining today's forecast, shedding plan, impact estimate, and community notice — "
               "for someone who won't click through every tab.")
    if st.button("Generate stakeholder PDF report"):
        report_battery_schedule = battery.simulate_battery_schedule(
            list(shed_loads), list(shed_solars), cfg["battery_capacity_mwh"], initial_soc_frac=0.5
        )
        report_impact = impact.estimate_daily_impact(
            [s["solar_mw"] for s in report_battery_schedule],
            [s["discharge_mwh"] for s in report_battery_schedule],
        )
        report_impact.update(impact.project_annual_impact(report_impact))

        loads_list = list(shed_loads)
        report_forecast_summary = {
            "peak_load_mw": float(max(loads_list)),
            "peak_hour": int(loads_list.index(max(loads_list))),
            "avg_load_mw": float(np.mean(loads_list)),
        }
        report_notice_text = notice.generate_notice_text(cfg["label"], shed_schedule, plan_date)
        report_drought_note = drought_info["note"] if drought_key != "none" else None

        pdf_bytes = report.generate_stakeholder_report(
            cfg["label"], plan_date, report_forecast_summary, shed_summary,
            report_impact, report_notice_text, report_drought_note
        )
        st.session_state["last_report"] = {
            "key": (region_key, str(plan_date), drought_key, int(rec_temperature), bool(rec_weekend)),
            "pdf": pdf_bytes,
        }
        st.success("Report generated — download it below.")

    current_report_key = (region_key, str(plan_date), drought_key, int(rec_temperature), bool(rec_weekend))
    stored_report = st.session_state.get("last_report")
    if stored_report and stored_report["key"] == current_report_key:
        st.download_button("⬇️ Download stakeholder report (PDF)", data=stored_report["pdf"],
                           file_name=f"{region_key}_report_{plan_date}.pdf", mime="application/pdf")
    elif stored_report:
        st.caption("A previously generated report exists for different settings — click Generate to refresh it for the current selection.")

    st.divider()
    st.markdown("#### 🔋 Battery charge/discharge plan")

    if cfg["battery_capacity_mwh"] <= 0:
        st.info(f"{cfg['label']} has no battery storage configured — this feature applies to solar-hybrid mini-grids.")

        no_batt_schedule = battery.simulate_battery_schedule(list(shed_loads), list(shed_solars), 0)
        solar_only_series = [s["solar_mw"] for s in no_batt_schedule]
        daily_impact_solar_only = impact.estimate_daily_impact(solar_only_series, [0.0] * 24)

        st.markdown("###### 💰 Estimated impact of today's solar generation alone")
        icol1, icol2, icol3 = st.columns(3)
        icol1.metric("Diesel avoided today", f"{daily_impact_solar_only['avoided_diesel_mwh']} MWh")
        icol2.metric("Cost saved today", f"K{daily_impact_solar_only['cost_saved_zmw']:,.0f}")
        icol3.metric("CO₂ avoided today", f"{daily_impact_solar_only['co2_avoided_tonnes']} t")
    else:
        battery_schedule = battery.simulate_battery_schedule(
            list(shed_loads), list(shed_solars), cfg["battery_capacity_mwh"], initial_soc_frac=0.5
        )
        battery_summary = battery.summarize_battery_schedule(battery_schedule)

        bcol1, bcol2, bcol3 = st.columns(3)
        bcol1.metric("Battery capacity", f"{cfg['battery_capacity_mwh']} MWh")
        bcol2.metric("Hours battery couldn't fully cover", battery_summary["hours_battery_could_not_cover"])
        bcol3.metric("Max unmet deficit", f"{battery_summary['max_unmet_deficit_mw']} MW")

        st.markdown("###### 💰 Estimated impact of today's solar + battery use")
        st.caption(
            "Order-of-magnitude estimate: treats solar generated and battery energy discharged as diesel-backup "
            "avoided. Zambia's grid is majority hydro (already low-carbon) — this is about avoiding diesel "
            "generators during shortfalls, not a broader grid-decarbonization claim. Figures scale with this "
            "region's configured solar/battery capacity (a demo parameter, not a rated real-site capacity)."
        )
        solar_series_imp = [s["solar_mw"] for s in battery_schedule]
        discharge_series_imp = [s["discharge_mwh"] for s in battery_schedule]
        daily_impact = impact.estimate_daily_impact(solar_series_imp, discharge_series_imp)
        annual_impact = impact.project_annual_impact(daily_impact)

        icol1, icol2, icol3 = st.columns(3)
        icol1.metric("Diesel avoided today", f"{daily_impact['avoided_diesel_mwh']} MWh")
        icol2.metric("Cost saved today", f"K{daily_impact['cost_saved_zmw']:,.0f}")
        icol3.metric("CO₂ avoided today", f"{daily_impact['co2_avoided_tonnes']} t")
        st.caption(
            f"Projected annually (if today repeats): K{annual_impact['projected_annual_cost_saved_zmw']:,.0f} "
            f"and {annual_impact['projected_annual_co2_avoided_tonnes']:,.0f} t CO₂ — {annual_impact['note']}"
        )

        batt_fig = go.Figure()
        batt_fig.add_trace(go.Scatter(x=[s["hour"] for s in battery_schedule],
                                       y=[s["soc_pct"] for s in battery_schedule],
                                       name="Battery SOC (%)", fill="tozeroy"))
        batt_fig.update_layout(xaxis_title="Hour", yaxis_title="State of charge (%)",
                                xaxis=dict(dtick=1), height=300)
        st.plotly_chart(batt_fig, width='stretch')

        action_colors = {"charge": "🟢", "discharge": "🔵", "idle": "⚪", "no_battery": "—"}
        batt_display_df = pd.DataFrame([
            {"Hour": s["hour"], "Action": f"{action_colors.get(s['action'],'')} {s['action']}",
             "Load (MW)": s["load_mw"], "Solar (MW)": s["solar_mw"], "SOC (%)": s["soc_pct"],
             "Unmet (MW)": s["unmet_deficit_mw"]}
            for s in battery_schedule
        ])
        st.dataframe(batt_display_df, hide_index=True, height=300)

# ---------------- Tab 6: SMS simulator ----------------
with tab6:
    st.subheader("📱 SMS-style access simulator")
    st.caption(
        "Many people in Zambia reach digital services over basic phones via SMS, not apps. "
        "This simulates the same forecasting engine behind a plain-text interface — the kind "
        "that could sit behind a real SMS gateway with no change to the underlying logic."
    )

    sms_lang = st.selectbox(
        "Reply language", options=["EN", "NY", "BE"],
        format_func=lambda x: {"EN": "English", "NY": "Nyanja", "BE": "Bemba"}[x],
        key="sms_lang",
        help="Nyanja/Bemba text is best-effort and has not been reviewed by a native speaker — see sms_interface.py."
    )
    if sms_lang != "EN":
        st.caption("⚠️ Translations here are a starting point, not yet verified by a native speaker.")

    st.code(sms_interface._t("help", sms_lang) + " " + sms_interface._region_help_text(), language=None)

    sms_text = st.text_input("Type an SMS command", value="STATUS LSK", key="sms_input")
    if st.button("Send"):
        def _sms_models_loader(region_key):
            return load_models(region_key)

        sms_result = sms_interface.handle_sms_command(sms_text, _sms_models_loader, sms_lang)
        if sms_result["success"]:
            st.success(f"📩 Reply: {sms_result['reply']}")
        else:
            st.warning(f"📩 Reply: {sms_result['reply']}")

    st.divider()
    st.caption("Try: `STATUS EMG`, `LOAD LSK 19`, `LOAD EMG 6`, `HELP`, `LANG NY`")

# ---------------- Tab 7: Upload & Retrain ----------------
with tab7:
    st.subheader(f"📤 Upload real data & retrain — {cfg['label']}")
    st.caption(
        "The rest of this dashboard runs on simulated data. Upload a real CSV here to retrain "
        "the model on it — closing the biggest gap between this prototype and a real deployment."
    )

    with st.expander("Required CSV format"):
        st.markdown(
            "**Required columns:** `hour` (0-23), `temperature_c`, `is_weekend` (0 or 1), `grid_load_mw`\n\n"
            "**Optional** (include all three to also retrain the solar model): "
            "`cloud_cover_pct` (0-100), `month` (1-12), `solar_generation_mw`\n\n"
            "Rows with missing values, duplicates, or out-of-range values are automatically "
            "detected and dropped before training — you'll see exactly what was removed and why."
        )

    uploaded_file = st.file_uploader(f"Upload CSV for {cfg['label']}", type="csv", key="csv_uploader")

    if uploaded_file is not None:
        try:
            upload_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read this file as a CSV: {e}")
            upload_df = None

        if upload_df is not None:
            validation = data_ingestion.validate_and_clean(upload_df)

            st.markdown("###### Data quality check")
            for issue in validation["issues"]:
                if "Dropped" in issue or "Only" in issue:
                    st.warning(issue)
                else:
                    st.success(issue)
            st.caption(f"Rows before cleaning: {validation['rows_before']} → after cleaning: {validation['rows_after']}")

            if validation["has_solar_columns"]:
                st.info("Solar columns detected — the solar model will be retrained too.")

            if validation["success"]:
                st.markdown("###### Preview of cleaned data")
                st.dataframe(validation["clean_df"].head(10), hide_index=True)

                if st.button("🚀 Retrain model on this data", type="primary"):
                    with st.spinner("Training..."):
                        train_result = data_ingestion.retrain_from_dataframe(
                            validation["clean_df"], region_key, validation["has_solar_columns"]
                        )
                    st.success(f"Retrained on {train_result['rows_used']} rows.")

                    original_metrics = load_metrics(region_key)
                    rcol1, rcol2 = st.columns(2)
                    with rcol1:
                        st.metric("New load model RMSE", f"{train_result['load_rmse']} MW",
                                   delta=(f"{train_result['load_rmse'] - original_metrics['load_model_rmse']:.2f} vs demo model"
                                          if original_metrics else None), delta_color="inverse")
                        st.caption(f"R² = {train_result['load_r2']}")
                    if train_result["trained_solar"]:
                        with rcol2:
                            st.metric("New solar model RMSE", f"{train_result['solar_rmse']} MW")
                            st.caption(f"R² = {train_result['solar_r2']}")

                    st.info(
                        "Custom model saved. Enable **'Use my custom-retrained model'** in the sidebar "
                        "to use it throughout the dashboard instead of the demo model."
                    )
                    load_models.clear()  # bust the cache so the sidebar toggle picks up the new file immediately
            else:
                st.error("Cannot train on this data yet — see the issues above.")

    st.divider()
    status = custom_model_status(region_key)
    if status["load"] or status["solar"]:
        st.caption(f"✅ Custom model currently saved for this region — load: {status['load']}, solar: {status['solar']}")
    else:
        st.caption("No custom model trained yet for this region.")

# ---------------- Tab 8: Plan vs. Reality & Holidays ----------------
with tab8:
    st.subheader(f"📋 Plan vs. Reality — {cfg['label']}")
    st.caption(
        "A forecast can be accurate and the shedding plan can still not get carried out as scheduled. "
        "This tracks that separately: log a plan (Smart Recommendations tab), then confirm here what "
        "actually happened."
    )

    review_date = st.date_input("Review plans for date", value=datetime.date.today(), key="review_date")
    planned_entries = execution_tracking.get_planned_entries(region_key, review_date)

    if not planned_entries:
        st.info(f"No plan logged for {region_key} on {review_date}. "
                f"Go to Smart Recommendations → 'Log this plan for execution tracking'.")
    else:
        pending = [e for e in planned_entries if e.status == "planned"]
        st.caption(f"{len(planned_entries)} planned actions, {len(pending)} awaiting confirmation.")

        for entry in planned_entries:
            cols = st.columns([2, 2, 2, 3])
            cols[0].write(f"**{entry.hour}:00**")
            cols[1].write(f"Zone {entry.zone}")
            if entry.status == "planned":
                cols[2].write("⏳ awaiting")
                if cols[3].button("Confirm executed", key=f"exec_{entry.id}"):
                    execution_tracking.mark_execution_status(entry.id, "executed")
                    st.rerun()
                if cols[3].button("Mark skipped", key=f"skip_{entry.id}"):
                    execution_tracking.mark_execution_status(entry.id, "skipped", notes="Marked via dashboard")
                    st.rerun()
            elif entry.status == "executed":
                cols[2].write("✅ executed")
                cols[3].write(entry.notes or "")
            else:
                cols[2].write("⛔ skipped")
                cols[3].write(entry.notes or "")

    st.divider()
    st.subheader("Compliance summary (last 30 days)")
    summary = execution_tracking.get_compliance_summary(region_key, days_back=30)
    if summary["total_logged"] == 0:
        st.info("No logged plans in the last 30 days yet.")
    else:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total planned actions", summary["total_logged"])
        s2.metric("Executed", summary["executed"])
        s3.metric("Skipped", summary["skipped"])
        s4.metric("Compliance rate", f"{summary['compliance_pct']}%" if summary["compliance_pct"] is not None else "—")

    st.divider()
    st.subheader("🗓️ Zambian public holiday calendar")
    st.caption(
        "Holidays are treated as weekend-like demand in predictions (see the sidebar date picker and "
        "MODEL_CARD.md for the reasoning). Dates are computed, not hardcoded to one year — verified "
        "against published sources while building this."
    )
    upcoming = holidays_zm.upcoming_holidays(datetime.date.today(), 6)
    holiday_df = pd.DataFrame([{"Date": d.strftime("%a, %d %B %Y"), "Holiday": name} for d, name in upcoming])
    st.dataframe(holiday_df, hide_index=True)

# ---------------- Tab 9: Integrations (webhooks) ----------------
with tab9:
    st.subheader("🔔 Webhook alerts")
    st.caption(
        "The alerts elsewhere in this app are logged to a file — enough to show an alert *would* fire, "
        "but nothing downstream can react to it. Register a URL here and it receives a real HTTP POST "
        "(JSON) when an alert is dispatched, so another system — an ops dashboard, a chatbot, a "
        "notification service — can actually respond."
    )

    with st.form("register_webhook_form", clear_on_submit=True):
        wh_region = st.selectbox(
            "Region to subscribe to",
            options=["*"] + list(REGIONS.keys()),
            format_func=lambda k: "All regions" if k == "*" else REGIONS[k]["label"],
        )
        wh_url = st.text_input("Webhook URL", placeholder="https://example.com/hooks/nyika")
        wh_submit = st.form_submit_button("Register webhook")

    if wh_submit:
        if not wh_url.strip().lower().startswith(("http://", "https://")):
            st.error("URL must start with http:// or https://")
        else:
            new_id = webhooks.register_webhook(wh_region, wh_url.strip())
            st.success(f"Registered webhook #{new_id}.")

    st.markdown("###### Active webhooks")
    active_hooks = webhooks.list_webhooks()
    if not active_hooks:
        st.info("No webhooks registered yet.")
    else:
        for hook in active_hooks:
            hcols = st.columns([1, 2, 5, 2])
            hcols[0].write(f"#{hook.id}")
            hcols[1].write("All regions" if hook.region == "*" else REGIONS.get(hook.region, {}).get("label", hook.region))
            hcols[2].write(hook.url)
            if hcols[3].button("Deactivate", key=f"deact_{hook.id}"):
                webhooks.deactivate_webhook(hook.id)
                st.rerun()

    st.divider()
    st.markdown("###### Send a test alert")
    st.caption("Dispatches to every webhook subscribed to the selected region. Delivery results are shown as-is — "
               "successes and failures both — rather than assuming it worked.")
    test_region = st.selectbox("Region", options=list(REGIONS.keys()),
                                format_func=lambda k: REGIONS[k]["label"], key="wh_test_region")
    test_load = st.number_input("Predicted load to report (MW)", min_value=0.0, value=250.0, key="wh_test_load")

    if st.button("Send test alert"):
        payload = {
            "region": test_region,
            "predicted_load_mw": float(test_load),
            "message": f"TEST ALERT: predicted load {test_load:.1f} MW for {REGIONS[test_region]['label']}",
        }
        results = webhooks.dispatch_alert_to_webhooks(test_region, payload)
        if not results:
            st.info("No webhooks are subscribed to this region.")
        else:
            for r in results:
                if r["success"]:
                    st.success(f"✅ Webhook #{r['webhook_id']} → {r['url']} (HTTP {r['status_code']})")
                else:
                    st.error(f"❌ Webhook #{r['webhook_id']} → {r['url']}: {r.get('error', 'HTTP ' + str(r.get('status_code')))}")

st.divider()
st.caption(
    "⚠️ This prototype is trained on **simulated data**. Predictions are illustrative only. "
    "Live weather fetch requires normal internet access — it will fail gracefully and fall back "
    "to manual input if unavailable."
)
