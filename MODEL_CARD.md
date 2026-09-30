# Model Card — Nyika-Grid Predict AI

A short, honest account of what this system does, how it was validated, and — most importantly — what it does **not** yet know how to do. Written for judges, reviewers, and anyone deciding whether to trust or extend this prototype.

---

## What this is

A prototype forecasting system for electricity grid load and solar generation in Zambia, covering two example regions (Lusaka Urban Grid, Eastern Mini-Grid), with tools that turn forecasts into action: load-shedding schedules, battery charge/discharge plans, drought stress-testing, and cost/CO2 impact estimates.

## Training data

**Simulated by default.** The demo models (`*_load.pkl`, `*_solar.pkl`) are trained on synthetically generated data (see `data_utils.py`) designed to resemble plausible daily/seasonal patterns — not on ZESCO's actual historical readings, which were not available while building this.

The **Upload & Retrain** feature (`data_ingestion.py`) lets a real user replace this with genuine load data. Any model retrained this way is only as good as the data uploaded — the built-in validation catches obvious quality problems (missing values, duplicates, out-of-range readings) but cannot verify the data is *representative* of a full year's conditions.

## Models used

| Task | Model | Why |
|---|---|---|
| Load forecast (point) | Random Forest | Simple, fast, works well on tabular features like hour/temperature/weekend |
| Solar forecast (point) | Random Forest | Same reasoning; trained against a physics-inspired daylight curve |
| 7-day forecast + confidence interval | Prophet | Built for exactly this: daily/weekly seasonality with uncertainty bounds |
| Naive baseline (for comparison) | Hourly average lookup | Answers "does the ML model actually help?" — see Section 5 metrics files |

## What's been validated, and how

- **Backend logic**: every API endpoint was hit with real HTTP requests during development, not just called as Python functions.
- **Dashboard**: exercised through Streamlit's `AppTest` framework — real button clicks, slider moves, file uploads — not just a syntax check. This caught at least one real bug (a what-if button crash) and one real modeling flaw (a mini-grid's solar capacity was sized so it could never physically exceed its load, meaning the battery could never charge) before delivery.
- **Frontend**: exercised with a headless DOM (`jsdom`) running the actual JavaScript against the live API. Visual appearance was **not** verified — no browser renderer was available in the build environment.
- **Explainability**: verified the per-prediction ablation method produces physically sensible attributions (e.g., evening-peak-hour and high-temperature scenarios correctly attribute most of the deviation to those two features).

## Known limitations (please read before presenting or deploying this)

1. **No real historical validation.** Every RMSE/R² figure in this project reflects performance on synthetic data (or, once retrained, on whatever was uploaded). None of it has been checked against a real held-out year of Zambian grid data.
2. **No substation-level granularity.** Each region is one aggregate model. Real load-shedding decisions in practice happen at a feeder/substation level with more local detail than this prototype has.
3. **Weather inputs are simplified.** Live weather integration exists (`weather.py`) but was never live-tested in this build environment (network-restricted sandbox) — only its failure/fallback path was confirmed. Temperature and cloud cover are also each single numbers per hour, not full meteorological forecasts.
4. **The drought scenarios are a proxy, not real capacity data.** `drought.py` uses *reported load-shedding hours* (a real, published fact from Zambia's 2024 crisis) as a stand-in for *capacity reduction*, because ZESCO's actual internal generation-loss figures for each date were not available. This is stated explicitly in the module — it's a reasonable stress-test, not a claim to official data.
5. **Nyanja and Bemba SMS translations are best-effort, not native-speaker-reviewed.** See `sms_interface.py`'s docstring. Common greetings/status words were checked against reference sources; technical vocabulary may be imperfect. This should be reviewed by a fluent speaker before real use.
6. **Cost/CO2 impact figures are order-of-magnitude estimates.** `impact.py`'s diesel cost and emissions constants are typical planning figures, not a quote for any specific site. They also scale directly with each region's configured solar/battery capacity, which is itself a demo parameter, not a rated real-site capacity.
7. **No adversarial or stress testing of the ML models themselves.** Testing in this project focused on "does the code run correctly and produce internally consistent results," not on formal model robustness, fairness, or security review.
8. **Single-point-in-time snapshot.** This system has no live connection to ZESCO or any real utility system. Every number shown is either from the built-in synthetic generator or from whatever was manually uploaded/entered.

9. **The map shows approximate region centers, not sites.** Each marker sits at a single representative coordinate for its region (near Lusaka and near Chipata), so it should not be read as the location of any actual substation, feeder, or mini-grid. A genuine per-site map needs real GPS-tagged multi-site data. The map's rendering was also not visually verified in a browser during the build.
10. **Webhook alerts are real but minimal.** Delivery to registered URLs is genuine (verified against a local receiver), but there is no retry on failure, no request signing (a receiver cannot verify a message came from this system), and no authentication on the registration endpoints — anyone who can reach the API could register a webhook. SMS/email alerts remain simulated (logged to a file), not actually sent.
11. **Generated reports reflect the same simulated foundation.** The stakeholder PDF faithfully summarizes whatever the models currently predict, which — unless retrained on real data — is synthetic. It carries a footer saying so, but it is a document that can be forwarded without its context, so treat it as a demonstration artifact, not an operational report.

## What would change first before any real deployment

1. Real historical load, weather, and solar data — replacing the entire simulated foundation.
2. Native-speaker review of the Nyanja/Bemba text.
3. A named institutional partner (ZESCO, a mini-grid operator, or the Met Department) providing both data and a way to validate forecasts against real outcomes.
4. A security/access review before connecting this to anything resembling live grid infrastructure.

---

*This document exists because a system that hides its limitations is less trustworthy than one that states them plainly — and because being specific about what's still missing is usually a stronger signal of engineering maturity than omitting it.*
