"""
Nyika-Grid Predict AI — Zambian Public Holiday Calendar
------------------------------------------------------------
Real Zambian public holidays, computed for any year (not hardcoded to one
year) using the actual rules that determine each date:
  - Fixed-date holidays (New Year's Day, Youth Day, Labour Day, Africa
    Freedom Day, Independence Day, Christmas)
  - Easter-based holidays (Good Friday, Easter Monday) via the standard
    Gregorian Easter algorithm
  - Rule-based holidays (Heroes Day = first Monday of July; Unity Day =
    the day after; Farmers' Day = first Monday of August)

Verified against published 2026 Zambian holiday listings (officeholidays.com,
worldtravelguide.net) while building this.

WHY THIS MATTERS FOR FORECASTING: demand on a public holiday typically
behaves more like a weekend than a normal weekday, but the trained load
model only knows about `is_weekend` — it has no dedicated holiday feature
(that would require retraining on data that includes it). As an interim,
honest improvement without needing a full retrain, this module lets a
caller treat a public holiday AS a weekend for prediction purposes — an
approximation, clearly labeled as such below.
"""

import datetime
from typing import List, Optional, Tuple

FIXED_HOLIDAYS = {
    (1, 1): "New Year's Day",
    (3, 12): "Youth Day",
    (5, 1): "Labour Day",
    (5, 25): "Africa Freedom Day",
    (10, 24): "Independence Day",
    (12, 25): "Christmas Day",
}


def _easter_sunday(year: int) -> datetime.date:
    """Standard Gregorian Easter algorithm (Anonymous/Meeus/Gauss)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return datetime.date(year, month, day)


def _first_monday(year: int, month: int) -> datetime.date:
    d = datetime.date(year, month, 1)
    offset = (7 - d.weekday()) % 7  # weekday(): Monday=0
    return d + datetime.timedelta(days=offset)


def get_holidays_for_year(year: int) -> List[Tuple[datetime.date, str]]:
    holidays = []

    for (month, day), name in FIXED_HOLIDAYS.items():
        holidays.append((datetime.date(year, month, day), name))

    easter = _easter_sunday(year)
    holidays.append((easter - datetime.timedelta(days=2), "Good Friday"))
    holidays.append((easter + datetime.timedelta(days=1), "Easter Monday"))

    heroes_day = _first_monday(year, 7)
    holidays.append((heroes_day, "Heroes Day"))
    holidays.append((heroes_day + datetime.timedelta(days=1), "Unity Day"))

    holidays.append((_first_monday(year, 8), "Farmers' Day"))

    return sorted(holidays, key=lambda h: h[0])


def is_public_holiday(date: datetime.date) -> Optional[str]:
    """Returns the holiday name if `date` is a Zambian public holiday, else None."""
    for holiday_date, name in get_holidays_for_year(date.year):
        if holiday_date == date:
            return name
    return None


def upcoming_holidays(from_date: datetime.date, count: int = 5) -> List[Tuple[datetime.date, str]]:
    """Next `count` holidays on or after from_date (spans into next year if needed)."""
    candidates = [h for h in get_holidays_for_year(from_date.year) if h[0] >= from_date]
    candidates += get_holidays_for_year(from_date.year + 1)
    return sorted(candidates, key=lambda h: h[0])[:count]


def effective_is_weekend(date: datetime.date) -> int:
    """
    The forecasting-relevant flag: 1 if this date is a weekend OR a public
    holiday (treating holidays as weekend-like demand, per the module
    docstring's stated approximation), else 0.
    """
    is_weekend = date.weekday() >= 5
    is_holiday = is_public_holiday(date) is not None
    return int(is_weekend or is_holiday)
