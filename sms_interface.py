"""
Nyika-Grid Predict AI — SMS-Style Text Interface
------------------------------------------------------
Many people in Zambia access digital services over basic phones via SMS
or USSD, not smartphones/apps. This module parses short plain-text
commands and returns short plain-text responses (kept under ~160
characters, standard SMS length) so the same forecasting engine could
sit behind a real SMS gateway (e.g. Africa's Talking) with no changes
to the core logic — only the transport layer would need to change.

Supported commands (case-insensitive), in English, Nyanja, or Bemba:
    STATUS <region_code>            e.g. "STATUS LSK"
    LOAD <region_code> <hour>       e.g. "LOAD LSK 19"
    HELP
    LANG <EN|NY|BE>                 switch reply language, e.g. "LANG NY"

LANGUAGE NOTE, STATED PLAINLY: the Nyanja and Bemba text below uses only
vocabulary that was checked against reference sources while building
this (common greetings, "bwino"=well/fine, "magetsi"=electricity in
Nyanja/Chichewa, "vuto"=problem). It has NOT been reviewed by a native
speaker of either language, and utility-specific terms in Bemba in
particular are commonly code-switched with English in real usage rather
than translated — which this module does deliberately rather than
guessing at technical vocabulary it can't verify. Have a native speaker
review and correct this before any real deployment.
"""

import pandas as pd
from typing import Dict

from data_utils import REGIONS

# Short SMS-friendly codes so users don't have to type full region keys
REGION_CODES = {
    "LSK": "lusaka_urban",
    "EMG": "eastern_minigrid",
}

SUPPORTED_LANGUAGES = ["EN", "NY", "BE"]

# Best-effort, NOT native-speaker-reviewed — see module docstring.
_TEXT = {
    "help": {
        "EN": "Commands: STATUS <code>, LOAD <code> <hour>, LANG <EN|NY|BE>, HELP.",
        "NY": "Ma command: STATUS <code>, LOAD <code> <hour>, LANG <EN|NY|BE>, HELP. (Nyanja - not yet reviewed by a fluent speaker)",
        "BE": "Ama command: STATUS <code>, LOAD <code> <hour>, LANG <EN|NY|BE>, HELP. (Bemba - not yet reviewed by a fluent speaker)",
    },
    "unknown_command": {
        "EN": "Unknown command '{cmd}'.",
        "NY": "Sindikudziwa command '{cmd}'.",
        "BE": "Nshaishiba command '{cmd}'.",
    },
    "unknown_region": {
        "EN": "Unknown code '{code}'.",
        "NY": "Sindikudziwa code '{code}'.",
        "BE": "Nshaishiba code '{code}'.",
    },
    "status_usage": {
        "EN": "Usage: STATUS <code>.",
        "NY": "Gwiritsani ntchito: STATUS <code>.",
        "BE": "Bomfyeni: STATUS <code>.",
    },
    "load_usage": {
        "EN": "Usage: LOAD <code> <hour>.",
        "NY": "Gwiritsani ntchito: LOAD <code> <hour>.",
        "BE": "Bomfyeni: LOAD <code> <hour>.",
    },
    "status_high": {
        "EN": "{region}: HIGH RISK. Load now ~{load:.0f}MW (threshold {threshold:.0f}MW).",
        "NY": "{region}: PANGOZI (magetsi ochuluka). Pano ~{load:.0f}MW (malire {threshold:.0f}MW).",
        "BE": "{region}: HIGH RISK. Load nomba ~{load:.0f}MW (threshold {threshold:.0f}MW).",
    },
    "status_stable": {
        "EN": "{region}: STABLE. Load now ~{load:.0f}MW (threshold {threshold:.0f}MW).",
        "NY": "{region}: BWINO. Pano ~{load:.0f}MW (malire {threshold:.0f}MW).",
        "BE": "{region}: BWINO. Load nomba ~{load:.0f}MW (threshold {threshold:.0f}MW).",
    },
    "load_reply": {
        "EN": "{region} at {hour}:00 - predicted load ~{load:.0f}MW.",
        "NY": "{region} pa {hour}:00 - magetsi oyembekezeredwa ~{load:.0f}MW.",
        "BE": "{region} pa {hour}:00 - load iyo twaipilibula ~{load:.0f}MW.",
    },
    "unavailable": {
        "EN": "Forecast unavailable right now. Try again later.",
        "NY": "Sizinatheke pano. Yesaninso pambuyo pake.",
        "BE": "Tapali kwati nomba. Eseshenipo mu kabili.",
    },
    "bad_hour": {
        "EN": "Hour must be 0-23, e.g. LOAD LSK 19",
        "NY": "Hour iyenera kukhala 0-23, mwachitsanzo LOAD LSK 19",
        "BE": "Hour ilekabila 0-23, ku muampele LOAD LSK 19",
    },
    "lang_set": {
        "EN": "Language set to English.",
        "NY": "Chinenero chasinthidwa ku Chinyanja.",
        "BE": "Ululimi lwasangulwa ku Cibemba.",
    },
    "lang_usage": {
        "EN": "Usage: LANG EN, LANG NY, or LANG BE.",
        "NY": "Gwiritsani ntchito: LANG EN, LANG NY, kapena LANG BE.",
        "BE": "Bomfyeni: LANG EN, LANG NY, nangu LANG BE.",
    },
}


def _t(key: str, lang: str, **kwargs) -> str:
    lang = lang if lang in SUPPORTED_LANGUAGES else "EN"
    template = _TEXT[key].get(lang, _TEXT[key]["EN"])
    return template.format(**kwargs) if kwargs else template


def _region_help_text() -> str:
    codes = ", ".join(f"{code}={REGIONS[key]['label']}" for code, key in REGION_CODES.items())
    return f"Codes: {codes}"


def handle_sms_command(text: str, models_loader, lang: str = "EN") -> Dict:
    """
    models_loader: a function(region_key) -> dict with "load" model,
    so this module doesn't need to know how models are stored/cached.
    lang: "EN", "NY", or "BE" — can also be changed mid-conversation with LANG <code>.

    Returns {"reply": "<text>", "command": "...", "success": bool, "lang": "..."}
    """
    if not text or not text.strip():
        return {"reply": _t("help", lang) + " " + _region_help_text(), "command": None, "success": False, "lang": lang}

    parts = text.strip().upper().split()
    command = parts[0]

    if command == "LANG":
        if len(parts) < 2 or parts[1] not in SUPPORTED_LANGUAGES:
            return {"reply": _t("lang_usage", lang), "command": "LANG", "success": False, "lang": lang}
        new_lang = parts[1]
        return {"reply": _t("lang_set", new_lang), "command": "LANG", "success": True, "lang": new_lang}

    if command == "HELP":
        return {"reply": _t("help", lang) + " " + _region_help_text(), "command": "HELP", "success": True, "lang": lang}

    if command == "STATUS":
        if len(parts) < 2:
            return {"reply": _t("status_usage", lang) + " " + _region_help_text(), "command": "STATUS", "success": False, "lang": lang}
        result = _handle_status(parts[1], models_loader, lang)
        result["lang"] = lang
        return result

    if command == "LOAD":
        if len(parts) < 3:
            return {"reply": _t("load_usage", lang) + " " + _region_help_text(), "command": "LOAD", "success": False, "lang": lang}
        result = _handle_load(parts[1], parts[2], models_loader, lang)
        result["lang"] = lang
        return result

    return {"reply": _t("unknown_command", lang, cmd=command) + " " + _t("help", lang),
            "command": None, "success": False, "lang": lang}


def _resolve_region(code: str):
    region_key = REGION_CODES.get(code.upper())
    if region_key is None:
        return None
    return region_key


def _handle_status(code: str, models_loader, lang: str = "EN") -> Dict:
    region_key = _resolve_region(code)
    if region_key is None:
        return {"reply": _t("unknown_region", lang, code=code) + " " + _region_help_text(), "command": "STATUS", "success": False}

    cfg = REGIONS[region_key]
    models = models_loader(region_key)
    if models is None:
        return {"reply": _t("unavailable", lang), "command": "STATUS", "success": False}

    import datetime
    current_hour = datetime.datetime.now().hour
    input_df = pd.DataFrame([{"hour": current_hour, "temperature_c": 28.0, "is_weekend": 0}])
    predicted = float(models["load"].predict(input_df)[0])

    key = "status_high" if predicted >= cfg["alert_threshold_mw"] else "status_stable"
    reply = _t(key, lang, region=cfg["label"], load=predicted, threshold=cfg["alert_threshold_mw"])
    return {"reply": reply[:160], "command": "STATUS", "success": True}


def _handle_load(code: str, hour_str: str, models_loader, lang: str = "EN") -> Dict:
    region_key = _resolve_region(code)
    if region_key is None:
        return {"reply": _t("unknown_region", lang, code=code) + " " + _region_help_text(), "command": "LOAD", "success": False}

    try:
        hour = int(hour_str)
        if not (0 <= hour <= 23):
            raise ValueError
    except ValueError:
        return {"reply": _t("bad_hour", lang), "command": "LOAD", "success": False}

    cfg = REGIONS[region_key]
    models = models_loader(region_key)
    if models is None:
        return {"reply": _t("unavailable", lang), "command": "LOAD", "success": False}

    input_df = pd.DataFrame([{"hour": hour, "temperature_c": 28.0, "is_weekend": 0}])
    predicted = float(models["load"].predict(input_df)[0])

    reply = _t("load_reply", lang, region=cfg["label"], hour=hour, load=predicted)
    return {"reply": reply[:160], "command": "LOAD", "success": True}
