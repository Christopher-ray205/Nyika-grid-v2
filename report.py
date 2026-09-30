"""
Nyika-Grid Predict AI — Stakeholder PDF Report
------------------------------------------------------
A one-click leave-behind document combining what's otherwise scattered
across dashboard tabs: today's forecast, the shedding plan, the
cost/CO2 impact estimate, and the plain-language community notice —
for a non-technical stakeholder (investor, judge, partner) who won't
click through six tabs but will read one PDF.
"""

import os
import datetime
from typing import List, Dict, Optional
from fpdf import FPDF

COPPER = (184, 112, 58)
INK = (18, 24, 43)
EARTH = (168, 68, 50)
SAVANNA = (107, 133, 88)
GREY = (90, 90, 90)


def _resolve_font_dir():
    """Prefer the bundled fonts/ directory (works on any machine); fall
    back to the system DejaVu install if the bundle is missing."""
    bundled = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
    if os.path.exists(os.path.join(bundled, "DejaVuSans.ttf")):
        return bundled + os.sep
    return "/usr/share/fonts/truetype/dejavu/"


_FONT_DIR = _resolve_font_dir()


class _ReportPDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use DejaVu (Unicode-capable) rather than the core Helvetica/Courier
        # fonts, which only support latin-1 and break on characters like
        # em-dashes that appear in this project's generated text elsewhere.
        self.add_font("DejaVu", "", _FONT_DIR + "DejaVuSans.ttf")
        self.add_font("DejaVu", "B", _FONT_DIR + "DejaVuSans-Bold.ttf")
        self.add_font("DejaVu", "I", _FONT_DIR + "DejaVuSans-Oblique.ttf")
        self.add_font("DejaVuMono", "", _FONT_DIR + "DejaVuSansMono.ttf")

    def header(self):
        self.set_font("DejaVu", "B", 16)
        self.set_text_color(*COPPER)
        self.cell(0, 10, "Nyika-Grid Predict AI", ln=True)
        self.set_font("DejaVu", "", 10)
        self.set_text_color(*GREY)
        self.cell(0, 6, "Daily Stakeholder Report", ln=True)
        self.ln(4)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(*GREY)
        self.cell(0, 10, "Trained on simulated data unless retrained on real data. See MODEL_CARD.md for limitations.", align="C")


def generate_stakeholder_report(
    region_label: str,
    report_date: datetime.date,
    forecast_summary: Dict,
    shed_summary: Dict,
    impact: Optional[Dict],
    notice_text: str,
    drought_note: Optional[str] = None,
) -> bytes:
    """
    forecast_summary: {"peak_load_mw": float, "peak_hour": int, "avg_load_mw": float}
    shed_summary: output of shedding.summarize_schedule()
    impact: output of impact.estimate_daily_impact() merged with project_annual_impact(), or None
    notice_text: output of notice.generate_notice_text()
    """
    pdf = _ReportPDF()
    pdf.add_page()

    pdf.set_font("DejaVu", "B", 13)
    pdf.set_text_color(*INK)
    pdf.cell(0, 8, f"{region_label}", ln=True)
    pdf.set_font("DejaVu", "", 11)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 7, report_date.strftime("%A, %d %B %Y"), ln=True)
    pdf.ln(4)

    # --- Forecast summary ---
    pdf.set_font("DejaVu", "B", 12)
    pdf.set_text_color(*INK)
    pdf.cell(0, 8, "Today's Forecast", ln=True)
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 6, f"Peak load: {forecast_summary['peak_load_mw']:.1f} MW at {forecast_summary['peak_hour']}:00", ln=True)
    pdf.cell(0, 6, f"Average load across the day: {forecast_summary['avg_load_mw']:.1f} MW", ln=True)
    pdf.ln(4)

    # --- Drought stress note, if applicable ---
    if drought_note:
        pdf.set_font("DejaVu", "B", 12)
        pdf.set_text_color(*EARTH)
        pdf.cell(0, 8, "Drought Stress-Test Applied", ln=True)
        pdf.set_font("DejaVu", "", 10)
        pdf.set_text_color(50, 50, 50)
        pdf.multi_cell(0, 6, drought_note)
        pdf.ln(4)

    # --- Shedding summary ---
    pdf.set_font("DejaVu", "B", 12)
    pdf.set_text_color(*INK)
    pdf.cell(0, 8, "Load-Shedding Plan", ln=True)
    pdf.set_font("DejaVu", "", 10)
    pdf.set_text_color(50, 50, 50)
    if shed_summary["total_hours_with_shedding"] == 0:
        pdf.set_text_color(*SAVANNA)
        pdf.cell(0, 6, "No shedding needed under current conditions.", ln=True)
    else:
        pdf.set_text_color(*EARTH)
        pdf.cell(0, 6, f"Shedding needed in {shed_summary['total_hours_with_shedding']} of 24 hours.", ln=True)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(0, 6, f"Peak deficit: {shed_summary['peak_deficit_mw']} MW", ln=True)
        zones = ", ".join(f"Zone {z}: {n}h" for z, n in shed_summary["shed_hours_per_zone"].items())
        pdf.cell(0, 6, f"Shed hours per zone (fairness check): {zones}", ln=True)
    pdf.ln(4)

    # --- Impact ---
    if impact:
        pdf.set_font("DejaVu", "B", 12)
        pdf.set_text_color(*INK)
        pdf.cell(0, 8, "Estimated Impact of Solar + Battery Use", ln=True)
        pdf.set_font("DejaVu", "", 10)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(0, 6, f"Diesel backup avoided today: {impact['avoided_diesel_mwh']} MWh", ln=True)
        pdf.cell(0, 6, f"Estimated cost saved today: K{impact['cost_saved_zmw']:,.0f}", ln=True)
        pdf.cell(0, 6, f"CO2 avoided today: {impact['co2_avoided_tonnes']} tonnes", ln=True)
        if "projected_annual_cost_saved_zmw" in impact:
            pdf.set_font("DejaVu", "I", 9)
            pdf.set_text_color(*GREY)
            pdf.multi_cell(0, 5, f"Projected annually (if today repeats): "
                                  f"K{impact['projected_annual_cost_saved_zmw']:,.0f}, "
                                  f"{impact['projected_annual_co2_avoided_tonnes']:,.0f} tonnes CO2. "
                                  f"{impact.get('note', '')}")
        pdf.ln(4)

    # --- Community notice ---
    pdf.set_font("DejaVu", "B", 12)
    pdf.set_text_color(*INK)
    pdf.cell(0, 8, "Community Notice (as would be shared publicly)", ln=True)
    pdf.set_font("DejaVuMono", "", 9)
    pdf.set_fill_color(245, 245, 245)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(0, 5, notice_text, fill=True)

    return bytes(pdf.output())
