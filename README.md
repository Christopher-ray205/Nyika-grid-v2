# Nyika-Grid Predict AI — Prototype (v6)

A working, fully-tested prototype forecasting electricity grid load **and solar generation** in Zambia, across multiple regions, with alerting (including real webhook delivery), accuracy tracking, real-data retraining, drought stress-testing, multi-language SMS access, plan-vs-reality execution tracking, a regional map, one-click stakeholder PDF reports, and a live dashboard plus standalone web frontend.

> ⚠️ Trained on **simulated data by default** — see "Upload & Retrain." See `MODEL_CARD.md` for a full, honest account of this project's limitations.

---

## What's new in v6

| Feature | Where |
| --- | --- |
| **Regional map view** | The Regional Overview tab now shows an actual map, with each region's marker colored by current risk (red = elevated, green = stable). Marker positions are approximate region centers, not individual sites — a per-site map would need real multi-site GPS data (`map_view.py`) |
| **Stakeholder PDF report** | One click produces a single PDF combining today's forecast, shedding plan, impact estimate, drought-scenario note (if applied), and community notice — for someone who won't click through every tab. Unicode-safe (no dropped characters) (`report.py`) |
| **Webhook alerts (real delivery)** | Register a URL and it receives an actual HTTP POST (JSON) whenever an alert fires — from both the API's `/alerts/check` and the dashboard's alert button. Delivery results (successes *and* failures) are reported as-is, never assumed. Subscriptions can be per-region or `*` for all, and can be deactivated (`webhooks.py`) |

> **Note on alerts:** webhook delivery is real. SMS/email dispatch (`alerts.py`) is still simulated — logged to `alerts.log`, with commented Twilio/SMTP hooks — so don't describe it as sending real texts or emails.

## What's new in v5

| Feature | Where |
| --- | --- |
| **Zambian public holiday calendar** | Computes real Zambian public holidays for any year (fixed dates, Easter-based, and rule-based like "first Monday of July"), verified against published sources for 2024 and 2026. Holidays are treated as weekend-like demand for prediction purposes — an explicit, stated approximation (`holidays_zm.py`) |
| **Plan vs. reality execution tracking** | A shedding *forecast* can be accurate while the *plan* still doesn't get carried out — this tracks that separately. Log a plan, confirm per-hour-per-zone whether it was executed or skipped, see a compliance rate over time (`execution_tracking.py`) |
| **Printable/shareable community notice** | Turns a technical shedding schedule into a plain-language notice — as text (for WhatsApp/SMS) or a postable branded PNG image — for the many people who will never open this dashboard (`notice.py`) |

## What's new in v4

| Feature | Where |
| --- | --- |
| **Drought stress-test scenarios** | Replay the load-shedding optimizer against real points in Zambia's 2024 hydropower crisis (March/May/September 2024), using ZESCO's actually-reported load-shedding hours as a transparent proxy for capacity loss (`drought.py`) |
| **Per-prediction explainability** | "This forecast is high *because of X*" for an individual prediction, using a feature-ablation method (not the SHAP library — clearly labeled as such) (`explainability.py`) |
| **Nyanja & Bemba SMS support** | The SMS interface now replies in English, Nyanja, or Bemba (`LANG NY`/`LANG BE`) — translations are best-effort and explicitly flagged as not yet native-speaker-reviewed |
| **Model card** | `MODEL_CARD.md` — an honest, judge-facing account of training data, validation method, and known limitations |

## What's new in v3

| Feature | Where |
| --- | --- |
| **CSV upload + retrain** | Upload a real load-data CSV (with automatic data-quality cleaning) and retrain the model on it, without touching code. Saved separately from the demo models so nothing is overwritten (`data_ingestion.py`) |
| **"Use my custom-retrained model" toggle** | Sidebar switch to use your retrained model instead of the demo one — tested to confirm it genuinely changes predictions, not just a label |
| **Cost & CO₂ impact estimator** | Converts solar generation + battery use into Kwacha saved and tonnes of CO₂ avoided, with an honest framing: Zambia's grid is majority hydro (already low-carbon), so this is about diesel-backup avoided, not grid decarbonization (`impact.py`) |
| **Battery charge/discharge tracking** | The battery planner now reports exact MWh charged/discharged per hour, feeding directly into the impact estimator |

## What's new in v2

| Feature | Where |
| --- | --- |
| **Multi-region support** | Two regions modeled independently: Lusaka Urban Grid and Eastern Mini-Grid (Chipata, solar-hybrid) |
| **Solar generation forecasting** | Separate solar model per region, physics-inspired training curve (zero at night, bell curve through the day, reduced by cloud cover) |
| **Net load calculation** | `net_load = predicted_load − predicted_solar`, shown throughout |
| **7-day forecast with confidence intervals** | Prophet model per region, `yhat_lower`/`yhat_upper` bounds shown as a shaded band |
| **Alerts** | Threshold-based check per region; SMS/email dispatch is simulated (logged to `alerts.log`, with commented Twilio/SMTP hooks in `alerts.py`), while webhook delivery to registered URLs is real (v6) |
| **Historical accuracy tracker** | SQLite database logs every prediction; log the real outcome later and the dashboard computes RMSE/MAE automatically |
| **Live weather integration** | Pulls current temperature + cloud cover from the free Open-Meteo API, with a tested graceful fallback to manual input if unavailable |
| **What-if scenarios** | One-click "Heatwave (+5°C)" and "Cloudy day (+60%)" buttons |
| **Regional risk overview** | Side-by-side comparison card for every region at the current conditions |
| **CSV export** | Download the 24-hour or 7-day forecast as CSV directly from the dashboard |
| **Load-shedding schedule optimizer** | Turns the net-load forecast into a fair, rotating shed schedule across feeder zones (`shedding.py`) |
| **Battery charge/discharge planner** | For the solar-hybrid mini-grid: simulates state of charge across the day, recommending charge/discharge timing (`battery.py`) |
| **Anomaly detection** | Flags logged actual readings that deviate far from prediction — a possible meter fault, theft, or genuine event worth investigating (`anomaly.py`) |
| **Model vs. naive baseline comparison** | Every trained model is benchmarked against a simple hourly-average baseline, so the dashboard shows *how much* the ML approach actually helps (`baseline.py`) |
| **Feature importance / explainability** | Bar chart showing what's driving each prediction (hour vs. temperature vs. weekend) |
| **SMS-style text interface** | Plain-text `STATUS`/`LOAD`/`HELP` commands simulate access over a basic phone/SMS gateway, not just a smartphone app (`sms_interface.py`) |
| **Standalone web frontend** | A branded, public-facing HTML/CSS/JS page (`frontend/index.html`) as an alternative to the internal Streamlit ops dashboard — talks to the same FastAPI backend over CORS |

---

## What's included

```text
data_utils.py     # Region config, simulated data generators, solar curve
train_model.py    # Trains load + solar (Random Forest), baseline, and Prophet models per region
baseline.py       # Naive hourly-average model, used to benchmark the ML model
db.py             # SQLite logging: predictions, actuals, accuracy stats
alerts.py         # Threshold checks + simulated alert dispatch
weather.py        # Live weather fetch (Open-Meteo) with graceful fallback
shedding.py       # Load-shedding schedule optimizer (fair rotation across zones)
battery.py        # Battery charge/discharge simulator for solar-hybrid sites
anomaly.py        # Flags actual readings that deviate far from prediction
sms_interface.py  # Plain-text command interface (STATUS / LOAD / HELP)
impact.py         # Cost (ZMW) and CO2 impact estimator for solar + battery use
data_ingestion.py # CSV validation/cleaning + retraining on real uploaded data
drought.py        # Drought stress-test scenarios grounded in Zambia's 2024 crisis timeline
explainability.py # Per-prediction "why is this forecast high" via feature ablation
MODEL_CARD.md     # Honest account of training data, validation, and known limitations
holidays_zm.py    # Computed Zambian public holiday calendar (any year)
execution_tracking.py  # Plan-vs-reality tracking for shedding schedules
notice.py         # Printable/shareable community shedding notice (text + PNG)
map_view.py       # Regional risk map (Plotly, no API token needed)
report.py         # One-click stakeholder PDF report
webhooks.py       # Webhook subscriptions + real HTTP alert delivery
api.py            # FastAPI backend — all endpoints listed below (CORS-enabled for the frontend)
dashboard.py      # Streamlit dashboard — 9 tabs, described below
frontend/         # Standalone HTML/CSS/JS web frontend (index.html) + its test harness
fonts/            # Bundled DejaVu fonts (Bitstream Vera license, redistribution permitted) for Unicode-safe PDF/image generation, with system-font fallback if absent
requirements.txt
models/           # Pre-trained model files (included, ready to run)
```

---

## Step 1 — Train the models

```bash
python train_model.py
```

Trains, for **each** region: a load model, a solar model, and a Prophet forecasting model — 6 files total in `./models/`. Takes under a minute. Pre-trained models are already included, so this step is optional unless you want to retrain.

Expected output:

```text
=== Region: Lusaka Urban Grid (lusaka_urban) ===
  Load model    — R^2: 0.83 | RMSE: 15.86 MW
  Solar model   — R^2: 0.99 | RMSE: 0.97 MW
  Training Prophet 7-day forecast model...
  Prophet model saved.

=== Region: Eastern Mini-Grid (Chipata, Solar-Hybrid) (eastern_minigrid) ===
  Load model    — R^2: 0.82 | RMSE: 4.25 MW
  Solar model   — R^2: 0.98 | RMSE: 0.93 MW
  ...
```

## Step 2 — Run the API (optional, for backend testing)

```bash
uvicorn api:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to test endpoints interactively.

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/regions` | GET | List available regions |
| `/predict/load` | POST | Predict grid load |
| `/predict/solar` | POST | Predict solar generation |
| `/predict/net-load` | POST | Combined load, solar, net load (optionally logs to DB) |
| `/forecast/24h?region=...` | GET | Full 24-hour load + solar + net-load curve in one call (used by the frontend) |
| `/forecast/7day?region=...` | GET | 7-day hourly forecast with confidence interval |
| `/alerts/check` | POST | Check a load value against the region's threshold |
| `/weather/live?region=...` | GET | Attempt a live weather pull |
| `/accuracy/log-actual` | POST | Record the real outcome for a logged prediction (returns anomaly check result) |
| `/accuracy/summary?region=...` | GET | RMSE/MAE over logged predictions |
| `/model/comparison?region=...` | GET | Model RMSE vs. naive baseline RMSE, plus feature importances |
| `/shedding/schedule` | POST | Fair, rotating load-shedding plan across zones for the day (accepts `drought_scenario` to stress-test against Zambia's 2024 crisis timeline) |
| `/battery/schedule` | POST | Battery charge/discharge plan for solar-hybrid sites |
| `/sms/query` | POST | Plain-text command interface (`{"text": "STATUS LSK"}`) |
| `/retrain` | POST | Upload a CSV (multipart form: `region`, `file`) to retrain the model on real data |
| `/model/custom-status?region=...` | GET | Whether a custom-retrained model exists for a region |
| `/impact/daily` | POST | Cost (ZMW) + CO2 impact estimate from today's solar/battery use |
| `/drought/scenarios` | GET | List the available drought stress-test scenarios |
| `/explain/load` | POST | Per-prediction feature-ablation explanation for a single load forecast |
| `/holidays/upcoming?count=...` | GET | Next N Zambian public holidays from today |
| `/holidays/check?date=...` | GET | Whether a given date is a holiday, and its effective weekend status |
| `/shedding/log-plan` | POST | Log a generated shedding schedule for later execution confirmation |
| `/shedding/mark-execution` | POST | Confirm a planned shed action as executed or skipped |
| `/shedding/planned-entries?region=...&shed_date=...` | GET | List planned actions for a region/date |
| `/shedding/compliance-summary?region=...` | GET | Plan-vs-reality compliance rate over the last N days |
| `/notice/text` | POST | Plain-language shedding notice, from a schedule |
| `/notice/image` | POST | Same notice as a postable PNG image |
| `/map/regions` | GET | Per-region coordinates + current risk status for the map |
| `/report/pdf` | POST | Stakeholder PDF report (forecast, shedding, impact, notice; supports `drought_scenario`) |
| `/webhooks/register` | POST | Subscribe a URL to alerts for one region or `*` |
| `/webhooks/list?region=...` | GET | Active webhook subscriptions |
| `/webhooks/deactivate/{id}` | POST | Stop a webhook receiving alerts |
| `/webhooks/dispatch` | POST | Manually dispatch an alert payload to subscribers |
| `/chat/status` | GET | Check whether a local or hosted generative model is configured |
| `/chat` | POST | Explain current forecasts and recommendations using a configured model |

`/alerts/check` now also delivers to registered webhooks (returned under `webhook_deliveries`) when the threshold is crossed.

## Project purpose and current status

Nyika-Grid Predict AI is a Zambia-focused prototype for anticipating electricity demand and solar generation, then turning those forecasts into practical planning information. It is designed to help grid and mini-grid operators see likely peak demand, understand the effect of solar on net load, compare forecast accuracy, plan fairer load shedding, and estimate when battery storage may help. It also provides plain-language information for operations teams and communities.

**What works today:** the project runs locally with trained demo models, forecasts for two example regions, a Streamlit operations dashboard, a FastAPI backend, and a standalone browser frontend. It includes scenario planning, weather lookup through Open-Meteo, CSV-based model retraining, SQLite-based prediction tracking, webhook alert delivery, and recommendations derived from its forecast. Its English/Nyanja/Bemba SMS-style interface is a simulator, not a connected SMS service.

**What is not connected yet:** the default load and solar models use synthetic data. There is no direct connection to ZESCO, a substation, smart meters, solar inverters, a battery management system, or a field sensor network. Weather lookup is an external API request, not a reading from a locally installed weather station. Consequently, this is not yet a real-time monitoring system and its recommendations must not be treated as dispatch instructions. The model card documents the limitations in more detail.

### Intended completed system

With permissioned operational data, validated integrations, and field testing, the project is intended to grow into a continuously refreshed decision-support system that can:

- Ingest time-stamped load, solar, battery, and weather telemetry for individual sites, feeders, or substations.
- Show observed conditions alongside forecasts, with data-quality and stale-data indicators.
- Forecast demand and renewable generation at useful operational horizons, and quantify uncertainty against real held-out data.
- Generate alerts and explainable load-shedding or battery recommendations using site-specific capacities and constraints.
- Let operators ask questions in natural language about forecasts and recommendations, with answers grounded in the latest available system data.
- Improve its models from reviewed outcomes and new data, with model versions, evaluation reports, and operator approval before a model is used.

These are development goals, not claims about the current prototype. Any future connection that can issue control commands would require a separate security, reliability, regulatory, and human-approval design. The present system is read-only with respect to grid equipment.

### Sensors and data needed for real-time operation

The exact equipment depends on the utility or mini-grid design. Sensors should be installed and integrated by qualified personnel, with approval from the asset owner. The project does not currently ingest any of the following telemetry directly; the table describes a practical target data set for a future connector.

| Data source | Measurements to collect | Why it is needed |
| --- | --- | --- |
| Feeder/substation or site meter (smart meter / power-quality meter, using suitable CTs and PTs) | Active power and energy, voltage, current, frequency, power factor; reactive power where available | Actual demand, net import/export, peak detection, and forecast evaluation |
| Solar inverter telemetry | AC output power and energy, inverter state, faults/availability; DC voltage/current if exposed | Actual generation and equipment availability |
| Solar irradiance and panel sensors | Plane-of-array irradiance and module temperature | Better site-specific PV forecasts and detection of underperformance |
| Battery management system (BMS) | State of charge (SOC), state of health (SOH), charge/discharge power, voltage, current, temperature, alarms | Safe, capacity-aware storage recommendations and battery performance tracking |
| Local weather station or trusted forecast feed | Ambient temperature, cloud/irradiance, humidity, wind, and rainfall as available | Weather-aware demand and renewable generation forecasts |
| Site and asset registry | Stable site/feeder IDs, GPS coordinates, timezone, rated limits, sensor scaling and units | Correctly associating readings with the right model, map location, and operating limits |

### Telemetry integration requirements

For a field deployment, meters and controllers commonly expose protocols such as Modbus RTU/TCP or DLMS/COSEM; an edge gateway can normalize those readings and publish them to a secured service, often using MQTT over TLS or an authenticated HTTPS API. This repository does not yet implement those protocol drivers or an MQTT consumer. A production ingestion path should provide:

- UTC timestamps, explicit units, stable asset identifiers, quality flags, and a documented sampling interval (for example, 1- to 15-minute readings aggregated into hourly model inputs).
- Local buffering for network outages, duplicate/out-of-order handling, missing-data rules, sensor calibration checks, and stale-data alerts.
- Authentication, encrypted transport, least-privilege access, network segmentation, audit logs, and a retention policy. Do not expose meters or control networks directly to the public internet.
- A mapping from raw telemetry to model inputs. The current point models expect hourly `hour`, `temperature_c`, and `is_weekend` for load, plus `cloud_cover_pct` and `month` for solar; real-data CSV retraining is documented in `data_ingestion.py`.

### How the system can improve over time

1. **Establish trustworthy data:** agree on data access with asset owners, install or connect approved meters, validate units/timestamps, and collect enough representative seasonal history. Start with read-only telemetry.
2. **Prove forecast value:** train and compare against simple baselines using chronological, held-out real data; report errors by site, season, and horizon. Do not rely on random train/test splits alone for time-series evaluation.
3. **Adapt to each site:** add site metadata and operational constraints, tune models for local load and solar patterns, and use operator-reviewed actual outcomes to detect drift and retrain deliberately.
4. **Harden operations:** deploy authenticated ingestion, resilient storage, backups, monitoring, access control, and reviewed alert integrations; verify fail-safe behavior before operational use.
5. **Expand carefully:** add more sites and regional views, review local-language notices with native speakers, and evaluate any future automation separately from forecasting and chatbot features.

## Local setup

Use Python 3.11 or newer in a virtual environment. From the repository root, run the following in PowerShell on Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, use `\.venv\Scripts\python.exe -m pip install -r requirements.txt` and run project commands with `\.venv\Scripts\python.exe`. On macOS/Linux, activate with `source .venv/bin/activate` instead.

Pre-trained models are included. To regenerate them, run `python train_model.py`. To start the API, run `uvicorn api:app --reload`; the API documentation is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). The Streamlit dashboard can be started separately with `streamlit run dashboard.py`.

### Generative Assistant

The standalone frontend includes a generative chatbot for questions about the selected region's forecast, shedding outlook, and battery plan. The API computes those values and sends them as context to the language model. The model generates an explanation; it does not make the underlying forecast, read sensors, send operational commands, or control the grid. The assistant is available only in the standalone frontend and requires a reachable model provider.

#### Run with Ollama locally

1. Install Ollama for your operating system from [ollama.com/download](https://ollama.com/download). Start the Ollama application/service. Its default API address is `http://127.0.0.1:11434`.
2. In a terminal, download the default model configured by this project:

```bash
ollama pull llama3.2:3b
```

1. Start the FastAPI server in the same terminal environment where you set the provider configuration. For PowerShell:

```powershell
$env:NYIKA_CHAT_PROVIDER = "ollama"
$env:OLLAMA_HOST = "http://127.0.0.1:11434"
$env:OLLAMA_MODEL = "llama3.2:3b"
uvicorn api:app --reload
```

The equivalent macOS/Linux environment setup is:

```bash
export NYIKA_CHAT_PROVIDER=ollama
export OLLAMA_HOST=http://127.0.0.1:11434
export OLLAMA_MODEL=llama3.2:3b
uvicorn api:app --reload
```

1. Check [http://127.0.0.1:8000/chat/status](http://127.0.0.1:8000/chat/status). It should return `"available": true`. If it says the model is missing, run `ollama pull` for the exact value of `OLLAMA_MODEL`. If Ollama is running on another machine or port, set `OLLAMA_HOST` to an address reachable **from the API process**. Keep Ollama on a trusted network; do not expose its unauthenticated local API publicly.
1. In a second terminal, serve the frontend:

```powershell
cd frontend
python -m http.server 5500
```

Open [http://127.0.0.1:5500](http://127.0.0.1:5500) and use the Nyika-Grid Assistant section. Keep Ollama, the API, and the frontend server running while using the chat. The frontend's API base should point to the running API (default `http://127.0.0.1:8000`).

#### Configuration reference

| Variable | Default | Purpose |
| --- | --- | --- |
| `NYIKA_CHAT_PROVIDER` | `auto` | `ollama`, `openai`, or `auto`. `auto` uses OpenAI-compatible chat when `OPENAI_API_KEY` is set; otherwise it checks local Ollama. Set `ollama` to explicitly require the local provider. |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama API base URL reachable by FastAPI. |
| `OLLAMA_MODEL` | `llama3.2:3b` | Exact installed Ollama model tag; pull it before starting chat. |
| `OPENAI_API_KEY` | unset | Optional key for hosted OpenAI-compatible chat. Keep it in the API server environment, never in frontend code or source control. |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Optional base URL for the OpenAI-compatible provider. |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model name used by the OpenAI-compatible provider. |

Ollama runs inference on the local machine, so model speed depends on available memory and CPU/GPU. After the model is downloaded, chat inference can run locally; other project features such as Open-Meteo weather lookup still need network access. The API sends forecast context to the configured provider, so using hosted chat sends that context outside the machine. The assistant is read-only and its generated explanations can be incorrect; verify operational decisions independently.

For a different OpenAI-compatible provider, set `NYIKA_CHAT_PROVIDER=openai`, `OPENAI_API_KEY`, and optionally `OPENAI_BASE_URL` and `OPENAI_MODEL` in the environment used to start Uvicorn. Never commit API keys to the repository.

## Step 3 — Run the dashboard

```bash
streamlit run dashboard.py
```

Nine tabs:

1. **Live Prediction** — sliders for hour/temperature/cloud cover/weekend, a date picker with automatic Zambian holiday detection, what-if buttons, live weather fetch, alert banner, 24-hour load+solar+net-load chart, CSV export, a "why trust this model" panel (baseline comparison + feature importance), and per-prediction explainability (why *this* forecast is what it is).
2. **7-Day Forecast** — Prophet forecast with confidence interval band, CSV export.
3. **Regional Overview** — an actual map with risk-colored region markers, plus both regions compared side-by-side at the current conditions.
4. **Historical Accuracy** — log predictions, enter actual outcomes later, see RMSE/MAE build up over time, with automatic anomaly flagging on entry.
5. **Smart Recommendations** — the load-shedding schedule (with a drought stress-test selector replaying Zambia's real 2024 crisis timeline), the battery charge/discharge plan (mini-grid) or solar-only figures (urban grid), the Kwacha/CO2 impact of today's solar+battery use, a button to log the plan for execution tracking, and a downloadable/previewable community notice (text + image), and a one-click stakeholder PDF report.
6. **SMS Simulator** — type plain-text commands (`STATUS LSK`, `LOAD EMG 13`, `HELP`, `LANG NY`) in English, Nyanja, or Bemba to see how the same forecasting engine could serve people over a basic phone.
7. **Upload & Retrain** — upload a real load-data CSV, see the automatic data-quality report, and retrain the model on it. A sidebar toggle ("Use my custom-retrained model") lets you switch predictions across the whole dashboard to the retrained model once it exists.
8. **Plan vs. Reality** — review a logged shedding plan and confirm per-hour-per-zone whether it was executed or skipped, see a rolling compliance rate, and browse the computed Zambian public holiday calendar.
9. **Integrations** — register/deactivate webhook URLs, and send a test alert to see per-webhook delivery results (successes and failures shown as-is).

---

## Step 4 — Run the standalone web frontend

A polished, branded HTML/CSS/JS frontend is included at `frontend/index.html` — a public-facing alternative to the Streamlit dashboard, calling the same FastAPI backend.

```bash
# with the API already running (uvicorn api:app --reload), in another terminal:
cd frontend
python -m http.server 5500
```

Then open [http://127.0.0.1:5500](http://127.0.0.1:5500) in a browser. (Serving it this way avoids browser restrictions on `file://` pages making network requests — it's the recommended way to open it, rather than double-clicking the HTML file directly.)

If your API runs on a different port, change the "API base" field in the page footer.

**What it includes:**

- Live control-room-style readout of current predicted load, solar generation, and net load, with a stable/at-risk status pill
- Hour / temperature / cloud cover / weekend controls, plus one-click "Heatwave" and "Cloudy day" what-if buttons
- 24-hour forecast chart (Chart.js)
- Region switcher (Lusaka Urban Grid / Eastern Mini-Grid)
- Load-shedding and battery recommendation panels, pulled live from `/shedding/schedule` and `/battery/schedule`
- An SMS-command simulator box, hitting `/sms/query`

**How it was tested:** since this sandbox can't render a real browser, the frontend was tested with a headless DOM (`jsdom`) driving the *actual* `index.html` and its real inline JavaScript — not a rewritten test version — against the live FastAPI backend. The test script simulated real user actions (moving the hour slider, clicking What-If buttons, switching regions, sending an SMS command) and checked the resulting DOM content and network calls. All passed with zero JavaScript errors or unhandled promise rejections. It's included as `frontend/test_frontend.js` if you want to rerun it yourself:

```bash
cd frontend
npm install jsdom
node test_frontend.js
```

What this **couldn't** verify: actual visual appearance/layout in a real browser, since no visual renderer was available in this environment. Open it in a real browser to confirm the visual design before presenting it — the logic is confirmed correct, but I haven't seen it rendered.

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## What was actually tested

Every module here was executed and verified, not just syntax-checked:

- `train_model.py` — run end-to-end, all model + baseline + metrics files produced and reloaded successfully
- `api.py` — live server started, every endpoint hit with real HTTP requests, including the new `/shedding/schedule`, `/battery/schedule`, `/model/comparison`, and `/sms/query` endpoints
- `shedding.py` — tested with a deficit scenario; confirmed the shed burden rotates fairly across zones rather than always hitting the same one
- `battery.py` — **caught and fixed a real modeling flaw here**: the initial region parameters made it physically impossible for solar to ever exceed daytime load, so the battery could never charge under realistic conditions. Fixed by resizing the mini-grid's solar capacity relative to its baseline load, then re-verified a full charge/discharge cycle (charges to 100% at midday, smooths the afternoon drop-off, and honestly still shows an unmet deficit during the evening peak — a battery this size can't cover everything, and the tool says so rather than hiding it).
- `anomaly.py` — tested via the real `/accuracy/log-actual` flow: a wildly off actual reading correctly triggers the anomaly flag, a close one doesn't.
- `sms_interface.py` — tested all commands (`STATUS`, `LOAD`, `HELP`) plus edge cases (empty input, unknown command, invalid hour, unknown region code) — all fail gracefully with a helpful reply rather than crashing.
- `dashboard.py` — run headlessly via Streamlit's `AppTest` framework, not just a syntax check: initial render (all 9 tabs, as of v6), region switching, both what-if buttons, the live-weather-fetch fallback, logging a prediction, saving an actual outcome (with anomaly check), the Smart Recommendations tab for both regions (including the "no battery" branch for the urban grid), and the SMS simulator's Send button — all exercised and confirmed to run without exceptions.
- `weather.py`'s live API call could not be tested end-to-end in this development sandbox (its network access is restricted to package registries only) — but the failure path was tested and confirmed to fall back gracefully rather than crash. Test the live call yourself once deployed somewhere with normal internet access.
- `frontend/index.html` — tested with a headless DOM (`jsdom`) executing the real inline JavaScript against the live API, simulating real interactions (slider moves, button clicks, region switch, SMS send). Confirmed correct arithmetic and zero JS errors. Visual layout could not be confirmed in this sandbox (no browser renderer available) — open it in a real browser to check appearance.
- `data_ingestion.py` — tested with an intentionally messy CSV (missing values, duplicate rows, out-of-range hour/temperature/load values); confirmed every issue was correctly detected, counted, and dropped. Tested the "too few clean rows" rejection path, the missing-required-column rejection path, and a full retrain producing a realistic R²/RMSE on data with genuine signal.
- `/retrain` endpoint — tested via a real multipart HTTP upload (not just calling the Python function directly), confirmed the resulting custom model is saved and, once the "use custom model" toggle is enabled, genuinely changes predictions (verified two different numbers from the same inputs, before vs. after retraining).
- `data_ingestion.py`'s dashboard integration — tested through Streamlit's actual `file_uploader` widget via `AppTest` (uploading real file bytes, not mocked), then clicking the real "Retrain" button and confirming the model file appears and metrics render.
- `impact.py` — sanity-checked against real trained-model output rather than only hand-picked test numbers, to confirm the scale of results was consistent with the region's configured (demo) solar capacity rather than being an arithmetic bug.

---

- `drought.py` — verified against real search-confirmed facts about Zambia's 2024 drought (ZESCO's reported 8h→12h→20h load-shedding escalation), not invented figures; tested that the September 2024 scenario correctly drives shedding-affected hours from 4/24 (normal) to 24/24 (crisis).
- `explainability.py` — tested against real trained models with physically sensible scenarios (evening-peak + hot day correctly attributes ~70 MW to hour-of-day and ~28 MW to temperature); verified via both direct function calls and the live `/explain/load` API endpoint, and confirmed the dashboard panel shows matching numbers.
- `sms_interface.py`'s language support — tested all three languages (English/Nyanja/Bemba) for both STATUS and LOAD commands, the LANG switch command, and its error/usage paths, plus the dashboard's language selector via `AppTest` with a real button click producing a real Nyanja reply through the live model.

- `holidays_zm.py` — cross-checked against three independent published sources (Wikipedia "Public holidays in Zambia", officeholidays.com, nationaltoday.com) for both 2024 and 2026; every computed date matched exactly, including catching that one lower-quality aggregator site had the wrong Heroes Day/Unity Day dates for 2026 (I verified the actual rule — first Monday of July — against multiple sources rather than trusting a single listing).
- `execution_tracking.py` and its dashboard tab — tested the full loop through real button clicks via `AppTest`: logging a plan, confirming one entry executed and another skipped, and verifying the compliance-rate metric recalculates correctly (1 executed / 1 skipped / 16 pending → 5.6%). Also tested via a real HTTP request sequence against the live API.
- `notice.py` — generated and visually inspected the rendered PNG output (not just checked that a file was created) for three cases: a normal partial-day shedding schedule, a full no-shedding day, and a severe multi-hour deficit scenario with merged hour ranges — all rendered correctly and matched the project's established visual identity.

- `map_view.py` — the figure object was verified (correct trace type, risk colors, computed center), and the Regional Overview tab was exercised through `AppTest`. **Not verified:** how the map actually looks rendered in a browser — no headless browser was available in the build environment, so static image export failed. Also caught and fixed a real bug on the way: the installed Plotly version uses the newer `Scattermap` API, not `Scattermapbox`, which would have crashed on first use.
- `report.py` — caught a real bug: fpdf2's built-in fonts only support latin-1, so an em-dash in the impact text crashed PDF generation. Fixed by switching to the Unicode-capable DejaVu font. Then converted the resulting PDFs to images and visually inspected them (both a direct call and one produced via the API with a drought scenario applied), confirming every figure matched what was computed. Also guarded against a stale-report bug in the dashboard (a previously generated PDF being offered for download after switching regions) and tested that guard.
- `webhooks.py` — tested against a real local HTTP receiver, not mocks of the requests library: registration, region filtering (region-specific plus `*`), delivery to a reachable URL (confirmed by the receiver's own log), graceful failure on an unreachable URL, and deactivation actually stopping delivery. Also caught that webhooks originally fired only from a manual test button and not from the real alert paths, then wired `/alerts/check` and the dashboard alert button to deliver and verified delivery happens exactly when the threshold is crossed and not otherwise.
- **Not tested:** webhook delivery to real third-party services (only a local receiver), retry/backoff behavior on failure (there is none — a failed delivery is reported once, not retried), and webhook authentication/signing (none is implemented, so receivers cannot verify a request came from this system).

## Next steps

- Replace simulated data in `data_utils.py` with real ZESCO load, weather, and solar irradiance data (or use the new Upload & Retrain tab to do this without touching code)
- Wire `alerts.py`'s commented Twilio/SMTP examples to real credentials
- Move `db.py` from SQLite to PostgreSQL for multi-user/production use
- Upgrade the map from approximate region centers to real per-site markers once GPS-tagged multi-site data is available
- Wire `alerts.py`'s SMS/email dispatch to a real provider (webhook delivery is already real; SMS/email is still simulated)
- Replace `impact.py`'s estimated diesel cost/CO2 constants with real site-specific figures once available
