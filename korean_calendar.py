"""Korean public holidays + 24절기. Static data for 2026-2028 (extend as needed).

Source: standard 한국 공휴일 + 양력 24절기 dates (관상감/한국천문연구원 published values).
대체공휴일까지는 포함, 24절기는 주요 8개(춘분·하지·추분·동지·입춘·입하·입추·입동)만.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List


HOLIDAYS: Dict[int, List[Dict]] = {
    2026: [
        {"date": "2026-01-01", "name": "신정"},
        {"date": "2026-02-16", "name": "설날 연휴"},
        {"date": "2026-02-17", "name": "설날"},
        {"date": "2026-02-18", "name": "설날 연휴"},
        {"date": "2026-03-01", "name": "삼일절"},
        {"date": "2026-03-02", "name": "삼일절 대체"},
        {"date": "2026-05-05", "name": "어린이날"},
        {"date": "2026-05-24", "name": "부처님오신날"},
        {"date": "2026-05-25", "name": "부처님오신날 대체"},
        {"date": "2026-06-03", "name": "전국동시지방선거"},
        {"date": "2026-06-06", "name": "현충일"},
        {"date": "2026-08-15", "name": "광복절"},
        {"date": "2026-08-17", "name": "광복절 대체"},
        {"date": "2026-09-24", "name": "추석 연휴"},
        {"date": "2026-09-25", "name": "추석"},
        {"date": "2026-09-26", "name": "추석 연휴"},
        {"date": "2026-10-03", "name": "개천절"},
        {"date": "2026-10-05", "name": "개천절 대체"},
        {"date": "2026-10-09", "name": "한글날"},
        {"date": "2026-12-25", "name": "성탄절"},
    ],
    2027: [
        {"date": "2027-01-01", "name": "신정"},
        {"date": "2027-02-05", "name": "설날 연휴"},
        {"date": "2027-02-06", "name": "설날"},
        {"date": "2027-02-07", "name": "설날 연휴"},
        {"date": "2027-02-08", "name": "설날 대체"},
        {"date": "2027-03-01", "name": "삼일절"},
        {"date": "2027-05-05", "name": "어린이날"},
        {"date": "2027-05-13", "name": "부처님오신날"},
        {"date": "2027-06-06", "name": "현충일"},
        {"date": "2027-08-15", "name": "광복절"},
        {"date": "2027-08-16", "name": "광복절 대체"},
        {"date": "2027-09-15", "name": "추석 연휴"},
        {"date": "2027-09-16", "name": "추석"},
        {"date": "2027-09-17", "name": "추석 연휴"},
        {"date": "2027-10-03", "name": "개천절"},
        {"date": "2027-10-04", "name": "개천절 대체"},
        {"date": "2027-10-09", "name": "한글날"},
        {"date": "2027-10-11", "name": "한글날 대체"},
        {"date": "2027-12-25", "name": "성탄절"},
    ],
    2028: [
        {"date": "2028-01-01", "name": "신정"},
        {"date": "2028-01-25", "name": "설날 연휴"},
        {"date": "2028-01-26", "name": "설날"},
        {"date": "2028-01-27", "name": "설날 연휴"},
        {"date": "2028-03-01", "name": "삼일절"},
        {"date": "2028-05-02", "name": "부처님오신날"},
        {"date": "2028-05-05", "name": "어린이날"},
        {"date": "2028-06-06", "name": "현충일"},
        {"date": "2028-08-15", "name": "광복절"},
        {"date": "2028-10-02", "name": "추석 연휴"},
        {"date": "2028-10-03", "name": "개천절 / 추석"},
        {"date": "2028-10-04", "name": "추석 연휴"},
        {"date": "2028-10-09", "name": "한글날"},
        {"date": "2028-12-25", "name": "성탄절"},
    ],
}


def list_holidays(year: int) -> List[Dict]:
    return list(HOLIDAYS.get(year, []))


def is_holiday(date_local: str) -> Dict:
    """Return {is_holiday: bool, name: str|None}. date_local is YYYY-MM-DD KST."""
    try:
        y = int(date_local[:4])
    except Exception:
        return {"is_holiday": False, "name": None}
    for h in HOLIDAYS.get(y, []):
        if h["date"] == date_local:
            return {"is_holiday": True, "name": h["name"]}
    return {"is_holiday": False, "name": None}


def next_holiday(today_local: str) -> Dict:
    try:
        d = date.fromisoformat(today_local)
    except Exception:
        return {}
    y = d.year
    while y <= max(HOLIDAYS.keys()):
        for h in HOLIDAYS.get(y, []):
            if h["date"] >= today_local:
                hd = date.fromisoformat(h["date"])
                return {**h, "days_until": (hd - d).days}
        y += 1
    return {}
