"""
Nyika-Grid Predict AI — Community Shedding Notice
------------------------------------------------------------
Most people affected by load-shedding will never open this dashboard.
This turns a technical shedding schedule into something postable on a
community noticeboard or shareable in a WhatsApp group: plain language,
no jargon, no MW figures — just which zone, and roughly when.
"""

import os
import datetime
from typing import List, Dict, Optional
from PIL import Image, ImageDraw, ImageFont

# Colors matching the frontend's copper/navy identity, kept simple for a
# printable/shareable graphic rather than an interactive UI.
INK = (18, 24, 43)
COPPER = (184, 112, 58)
PAPER = (246, 241, 231)
EARTH = (168, 68, 50)
SAVANNA = (107, 133, 88)


def _merge_shed_hours_into_ranges(shed_schedule: List[Dict]) -> Dict[str, List[str]]:
    """Groups consecutive shed hours per zone into readable ranges, e.g. 18-21h instead of 18h,19h,20h,21h."""
    zone_hours: Dict[str, List[int]] = {}
    for entry in shed_schedule:
        for zone in entry["zones_to_shed"]:
            zone_hours.setdefault(zone, []).append(entry["hour"])

    zone_ranges: Dict[str, List[str]] = {}
    for zone, hours in zone_hours.items():
        hours = sorted(hours)
        ranges = []
        start = prev = hours[0]
        for h in hours[1:]:
            if h == prev + 1:
                prev = h
                continue
            ranges.append(f"{start:02d}:00–{prev + 1:02d}:00" if start != prev else f"{start:02d}:00")
            start = prev = h
        ranges.append(f"{start:02d}:00–{prev + 1:02d}:00" if start != prev else f"{start:02d}:00")
        zone_ranges[zone] = ranges

    return zone_ranges


def generate_notice_text(region_label: str, shed_schedule: List[Dict], notice_date: Optional[datetime.date] = None) -> str:
    """Plain-language text notice — postable to WhatsApp, SMS, or read aloud."""
    notice_date = notice_date or datetime.date.today()
    zone_ranges = _merge_shed_hours_into_ranges(shed_schedule)

    lines = [
        f"⚡ POWER NOTICE — {region_label}",
        f"{notice_date.strftime('%A, %d %B %Y')}",
        "",
    ]

    if not zone_ranges:
        lines.append("No power interruptions expected today.")
    else:
        lines.append("Expected power interruptions today:")
        for zone in sorted(zone_ranges.keys()):
            times = ", ".join(zone_ranges[zone])
            lines.append(f"  Zone {zone}: {times}")
        lines.append("")
        lines.append("Please plan accordingly. Times are estimates and may vary.")

    lines.append("")
    lines.append("— Nyika-Grid Predict AI (community forecast, not an official ZESCO notice)")

    return "\n".join(lines)


def generate_notice_image(region_label: str, shed_schedule: List[Dict],
                           notice_date: Optional[datetime.date] = None,
                           width: int = 900, height: int = 700) -> Image.Image:
    """Renders the same notice as a postable PNG image."""
    notice_date = notice_date or datetime.date.today()
    zone_ranges = _merge_shed_hours_into_ranges(shed_schedule)

    img = Image.new("RGB", (width, height), INK)
    draw = ImageDraw.Draw(img)

    def _font(size, bold=False):
        # Bundled fonts (fonts/) are preferred so this works on any machine;
        # fall back to the system DejaVu install if the bundle is missing.
        bundled_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
        system_dir = "/usr/share/fonts/truetype/dejavu/"
        filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"

        for base in (bundled_dir, system_dir):
            path = os.path.join(base, filename)
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
        return ImageFont.load_default(size=size)

    title_font = _font(38, bold=True)
    date_font = _font(24)
    body_font = _font(28, bold=True)
    zone_font = _font(30)
    footer_font = _font(18)

    y = 40
    draw.text((40, y), "⚡ POWER NOTICE", font=title_font, fill=COPPER)
    y += 55
    draw.text((40, y), region_label, font=date_font, fill=PAPER)
    y += 34
    draw.text((40, y), notice_date.strftime("%A, %d %B %Y"), font=date_font, fill=(185, 192, 214))
    y += 55
    draw.line([(40, y), (width - 40, y)], fill=(70, 78, 105), width=2)
    y += 35

    if not zone_ranges:
        draw.text((40, y), "No power interruptions expected today.", font=body_font, fill=SAVANNA)
    else:
        draw.text((40, y), "Expected power interruptions today:", font=body_font, fill=PAPER)
        y += 55
        for zone in sorted(zone_ranges.keys()):
            times = ", ".join(zone_ranges[zone])
            draw.ellipse([(40, y + 6), (58, y + 24)], fill=EARTH)
            draw.text((72, y), f"Zone {zone}:  {times}", font=zone_font, fill=PAPER)
            y += 48
        y += 20
        draw.text((40, y), "Please plan accordingly. Times are estimates and may vary.",
                   font=footer_font, fill=(185, 192, 214))

    draw.text((40, height - 45), "Nyika-Grid Predict AI — community forecast, not an official ZESCO notice",
               font=footer_font, fill=(120, 128, 150))

    return img
