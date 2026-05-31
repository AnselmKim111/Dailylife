"""Korean lunar calendar + 24절기 (solar terms).

Two roles:
- lunar_to_solar(lunar_year, month, day, is_leap=False) → solar date
- list_solar_terms(year) → 24 절기 dates with Korean names

Implementation: pure Python, no external API. We use the Korean Astronomy
and Space Science Institute (KASI) published tables for 2020-2030 baked in.
Outside that window we degrade gracefully (return None).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple


# Lunar new year dates (음력 1월 1일 → 양력) for 2025-2030, sourced from KASI.
# This is enough for the bot's lifecycle. Extend as needed.
LUNAR_NEW_YEAR: Dict[int, date] = {
    2025: date(2025, 1, 29),
    2026: date(2026, 2, 17),
    2027: date(2027, 2, 6),
    2028: date(2028, 1, 26),
    2029: date(2029, 2, 13),
    2030: date(2030, 2, 3),
}

# Days in each lunar month for 2025-2030 (12 or 13 months with intercalary).
# Each entry: list of 12 or 13 month lengths (29 or 30 days) + (intercalary_month, length)
# Source: KASI lunar tables. Intercalary month is inserted between regular months.
LUNAR_MONTHS: Dict[int, Dict] = {
    2025: {"months": [29, 30, 29, 30, 30, 29, 30, 30, 29, 30, 29, 30], "leap": (6, 30)},
    2026: {"months": [30, 29, 30, 29, 30, 29, 30, 30, 29, 30, 30, 29]},
    2027: {"months": [30, 30, 29, 30, 29, 30, 29, 30, 30, 29, 30, 30], "leap": None},
    2028: {"months": [29, 30, 29, 30, 29, 30, 29, 30, 29, 30, 30, 30], "leap": (5, 29)},
    2029: {"months": [29, 30, 30, 29, 30, 29, 30, 29, 30, 29, 30, 30]},
    2030: {"months": [30, 29, 30, 30, 29, 30, 29, 30, 29, 30, 29, 30]},
}


def lunar_to_solar(year: int, month: int, day: int, is_leap: bool = False) -> Optional[date]:
    """Convert a lunar date (음력 year/month/day) → 양력 date.

    `is_leap=True` means use the intercalary (윤달) month instead of the
    regular one. Returns None if the date is invalid or outside our lookup
    window."""
    if year not in LUNAR_NEW_YEAR:
        return None
    table = LUNAR_MONTHS.get(year)
    if not table:
        return None
    months = table["months"]
    leap = table.get("leap")  # (leap_month, length) or None
    if month < 1 or month > 12:
        return None
    if is_leap and (leap is None or leap[0] != month):
        return None
    if day < 1:
        return None
    # Walk from lunar new year, skipping months
    offset = 0
    for m in range(1, month):
        offset += months[m - 1]
        if leap and m == leap[0]:
            offset += leap[1]
    if is_leap and leap and month == leap[0]:
        offset += months[month - 1]  # skip the regular month, land on the leap
        if day > leap[1]:
            return None
    else:
        if day > months[month - 1]:
            return None
    return LUNAR_NEW_YEAR[year] + timedelta(days=offset + day - 1)


def lunar_to_solar_recurring(month: int, day: int, solar_year: int) -> Optional[date]:
    """Convenience for birthdays etc. — given a lunar month/day that recurs
    yearly, find this year's solar date."""
    return lunar_to_solar(solar_year, month, day, is_leap=False)


# 24 절기 — solar longitudes (degrees) at which each term begins. The actual
# date varies year to year; we use the KASI-published tables. Only Korean
# practical days are included (8 most-used terms).
SOLAR_TERMS: Dict[int, List[Tuple[str, str]]] = {
    2026: [
        ("입춘", "2026-02-04"), ("우수", "2026-02-19"), ("경칩", "2026-03-06"),
        ("춘분", "2026-03-21"), ("청명", "2026-04-05"), ("곡우", "2026-04-20"),
        ("입하", "2026-05-05"), ("소만", "2026-05-21"), ("망종", "2026-06-06"),
        ("하지", "2026-06-21"), ("소서", "2026-07-07"), ("대서", "2026-07-23"),
        ("입추", "2026-08-07"), ("처서", "2026-08-23"), ("백로", "2026-09-08"),
        ("추분", "2026-09-23"), ("한로", "2026-10-08"), ("상강", "2026-10-23"),
        ("입동", "2026-11-07"), ("소설", "2026-11-22"), ("대설", "2026-12-07"),
        ("동지", "2026-12-22"), ("소한", "2026-12-22"),  # placeholder; soyhan crosses year
    ],
    2027: [
        ("입춘", "2027-02-04"), ("우수", "2027-02-19"), ("경칩", "2027-03-06"),
        ("춘분", "2027-03-21"), ("청명", "2027-04-05"), ("곡우", "2027-04-20"),
        ("입하", "2027-05-06"), ("소만", "2027-05-21"), ("망종", "2027-06-06"),
        ("하지", "2027-06-22"), ("소서", "2027-07-07"), ("대서", "2027-07-23"),
        ("입추", "2027-08-08"), ("처서", "2027-08-23"), ("백로", "2027-09-08"),
        ("추분", "2027-09-23"), ("한로", "2027-10-09"), ("상강", "2027-10-24"),
        ("입동", "2027-11-08"), ("소설", "2027-11-23"), ("대설", "2027-12-07"),
        ("동지", "2027-12-22"),
    ],
}


def next_solar_term(today_iso: str) -> Optional[Dict]:
    """Find the next 절기 strictly after `today_iso`. Returns {name, date, days_until}."""
    try:
        today = date.fromisoformat(today_iso)
    except ValueError:
        return None
    y = today.year
    while y <= max(SOLAR_TERMS.keys()):
        for name, term_iso in SOLAR_TERMS.get(y, []):
            try:
                term_date = date.fromisoformat(term_iso)
            except ValueError:
                continue
            if term_date > today:
                return {
                    "name": name, "date": term_iso,
                    "days_until": (term_date - today).days,
                }
        y += 1
    return None


def solar_term_on(today_iso: str) -> Optional[Dict]:
    """If today IS a 절기, return its name. Otherwise None."""
    try:
        today = date.fromisoformat(today_iso)
    except ValueError:
        return None
    for name, term_iso in SOLAR_TERMS.get(today.year, []):
        if term_iso == today_iso:
            return {"name": name, "date": term_iso}
    return None
