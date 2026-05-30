"""Lightweight weather + courier wrappers — no extra API keys required.

- weather: open-meteo.com (free, no key, JSON). Resolves Korean address → coord via Kakao Local.
- courier: scrapes 스마트택배 mobile page via fetch_url as a stopgap.
- transit_text_query: pure fetch_url fallback for ODsay-less transit answers.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

import external

logger = logging.getLogger(__name__)

OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODE_KR: Dict[int, str] = {
    0: "맑음", 1: "대체로 맑음", 2: "부분적 구름", 3: "흐림",
    45: "안개", 48: "착빙성 안개",
    51: "약한 이슬비", 53: "이슬비", 55: "강한 이슬비",
    61: "약한 비", 63: "비", 65: "강한 비",
    71: "약한 눈", 73: "눈", 75: "강한 눈",
    77: "싸락눈",
    80: "소나기 약함", 81: "소나기", 82: "강한 소나기",
    85: "눈 소나기 약함", 86: "눈 소나기",
    95: "뇌우", 96: "뇌우(우박 약)", 99: "뇌우(우박 강)",
}


async def weather(location: str, days: int = 1) -> Dict[str, Any]:
    """Get weather forecast for a Korean address/place. Geocodes via Kakao first.

    Returns {ok, location, hourly: [{time, temp, code, code_kr, rain%}], summary}."""
    coord = await external.kakao_address_to_coord(location)
    if not coord.get("ok") or not coord.get("match"):
        # try keyword search as fallback
        kw = await external.kakao_local_keyword(location, size=1)
        if kw.get("ok") and kw.get("places"):
            p = kw["places"][0]
            x, y = p["x"], p["y"]
            resolved = p["name"]
        else:
            return {"ok": False, "error": f"좌표를 찾지 못함: {location}"}
    else:
        x = coord["match"]["x"]
        y = coord["match"]["y"]
        resolved = coord["match"].get("address") or location

    params = {
        "latitude": y, "longitude": x,
        "hourly": "temperature_2m,weather_code,precipitation_probability",
        "timezone": "Asia/Seoul",
        "forecast_days": max(1, min(int(days), 7)),
    }
    async with httpx.AsyncClient(timeout=20.0) as c:
        r = await c.get(OPENMETEO_URL, params=params)
        if r.status_code >= 400:
            return {"ok": False, "error": f"open-meteo {r.status_code}"}
        data = r.json()
    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    temps = hourly.get("temperature_2m") or []
    codes = hourly.get("weather_code") or []
    rains = hourly.get("precipitation_probability") or []
    out_hours: List[Dict] = []
    # condense to 6h windows for a digest
    step = max(1, len(times) // 8)
    for i in range(0, len(times), step):
        if i >= len(codes):
            break
        out_hours.append({
            "time": times[i],
            "temp": temps[i] if i < len(temps) else None,
            "code": codes[i],
            "code_kr": WEATHER_CODE_KR.get(int(codes[i]), str(codes[i])),
            "rain_pct": rains[i] if i < len(rains) else None,
        })
    if temps:
        lo, hi = min(temps), max(temps)
        any_rain = any(p and p >= 50 for p in rains)
        summary = f"{resolved}: {lo:.0f}~{hi:.0f}℃"
        if any_rain:
            summary += ", 비 가능성 ↑"
    else:
        summary = resolved
    return {"ok": True, "location": resolved, "hourly": out_hours, "summary": summary}


async def track_parcel(tracking_no: str, carrier: Optional[str] = None) -> Dict[str, Any]:
    """Best-effort 택배 추적. Uses the mobile 스마트택배 fallback page via fetch_url.
    Returns raw page text; let the LLM extract structured status."""
    # Trackingmore or 스마트택배 mobile pages without auth keys
    url = f"https://www.doortodoor.co.kr/parcel/doortodoor.do?fsp_action=PARCEL_Action&fsp_cmd=retrieveInvNoACT&invc_no={tracking_no}"
    if carrier and "cj" not in carrier.lower():
        # generic search via 알리·CJ 외 운송사 — fallback to smart-tracker
        url = f"https://tracker.delivery/#/{carrier}/{tracking_no}"
    res = await external.fetch_url(url, max_chars=4000)
    return {"ok": res.get("ok", False), "tracking_no": tracking_no,
            "carrier": carrier, "raw": res.get("text", "")[:4000],
            "url": res.get("url_final")}


async def transit_text_query(origin: str, destination: str,
                             arrive_by_iso: Optional[str] = None) -> Dict[str, Any]:
    """Stopgap until ODsay key arrives. Asks for a fresh public-transit page and
    returns the raw text — the LLM then parses it. Quality depends on caller."""
    q = f"{origin} {destination} 대중교통".replace(" ", "+")
    url = f"https://m.search.naver.com/search.naver?query={q}"
    res = await external.fetch_url(url, max_chars=8000)
    notes = ""
    if arrive_by_iso:
        notes = f"\n*arrival target*: {arrive_by_iso}"
    return {
        "ok": res.get("ok", False),
        "origin": origin, "destination": destination,
        "raw": res.get("text", "")[:6000] + notes,
        "url": res.get("url_final"),
    }
