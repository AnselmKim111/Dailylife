"""Wrappers around Naver Search, Kakao Local/Mobility, and generic URL fetch."""

from __future__ import annotations

import logging
import os
import re
from html import unescape
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

NAVER_CLIENT_ID = os.environ.get("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET = os.environ.get("NAVER_CLIENT_SECRET", "")
KAKAO_REST_API_KEY = os.environ.get("KAKAO_REST_API_KEY", "")

NAVER_SEARCH_URL = "https://openapi.naver.com/v1/search/{type}.json"
KAKAO_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
KAKAO_ADDRESS_URL = "https://dapi.kakao.com/v2/local/search/address.json"
KAKAO_DIRECTIONS_URL = "https://apis-navi.kakaomobility.com/v1/directions"

DEFAULT_HTTP_TIMEOUT = 30.0


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    return unescape(text).strip()


async def naver_search(query: str, kind: str = "blog", display: int = 5) -> Dict[str, Any]:
    """Naver Search API. kind: blog | news | webkr | encyc | local."""
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        return {"ok": False, "error": "NAVER_CLIENT_ID/SECRET not configured"}
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {"query": query, "display": min(max(display, 1), 10)}
    url = NAVER_SEARCH_URL.format(type=kind)
    async with httpx.AsyncClient(timeout=DEFAULT_HTTP_TIMEOUT) as c:
        r = await c.get(url, headers=headers, params=params)
        if r.status_code >= 400:
            return {"ok": False, "error": f"naver {r.status_code}: {r.text[:200]}"}
        data = r.json()
    items = []
    for it in data.get("items", []):
        items.append(
            {
                "title": _strip_html(it.get("title", "")),
                "link": it.get("link"),
                "snippet": _strip_html(it.get("description", "")),
                "extra": {k: v for k, v in it.items() if k not in {"title", "link", "description"}},
            }
        )
    return {"ok": True, "items": items}


async def kakao_local_keyword(
    query: str,
    x: Optional[float] = None,
    y: Optional[float] = None,
    radius_m: Optional[int] = None,
    size: int = 5,
) -> Dict[str, Any]:
    if not KAKAO_REST_API_KEY:
        return {"ok": False, "error": "KAKAO_REST_API_KEY not configured"}
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    params: Dict[str, Any] = {"query": query, "size": min(max(size, 1), 15)}
    if x is not None and y is not None:
        params.update({"x": x, "y": y})
        if radius_m:
            params["radius"] = min(max(radius_m, 0), 20000)
    async with httpx.AsyncClient(timeout=DEFAULT_HTTP_TIMEOUT) as c:
        r = await c.get(KAKAO_KEYWORD_URL, headers=headers, params=params)
        if r.status_code >= 400:
            return {"ok": False, "error": f"kakao {r.status_code}: {r.text[:200]}"}
        data = r.json()
    docs = []
    for d in data.get("documents", []):
        docs.append(
            {
                "name": d.get("place_name"),
                "category": d.get("category_name"),
                "address": d.get("address_name"),
                "road_address": d.get("road_address_name"),
                "phone": d.get("phone"),
                "x": float(d["x"]) if d.get("x") else None,
                "y": float(d["y"]) if d.get("y") else None,
                "url": d.get("place_url"),
                "distance_m": int(d["distance"]) if d.get("distance") else None,
            }
        )
    return {"ok": True, "places": docs}


async def kakao_address_to_coord(address: str) -> Dict[str, Any]:
    if not KAKAO_REST_API_KEY:
        return {"ok": False, "error": "KAKAO_REST_API_KEY not configured"}
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    async with httpx.AsyncClient(timeout=DEFAULT_HTTP_TIMEOUT) as c:
        r = await c.get(KAKAO_ADDRESS_URL, headers=headers, params={"query": address})
        if r.status_code >= 400:
            return {"ok": False, "error": f"kakao {r.status_code}: {r.text[:200]}"}
        data = r.json()
    docs = data.get("documents", [])
    if not docs:
        return {"ok": True, "match": None}
    d = docs[0]
    return {
        "ok": True,
        "match": {
            "address": d.get("address_name"),
            "road_address": (d.get("road_address") or {}).get("address_name") if d.get("road_address") else None,
            "x": float(d["x"]),
            "y": float(d["y"]),
        },
    }


async def kakao_directions(
    origin_x: float,
    origin_y: float,
    dest_x: float,
    dest_y: float,
) -> Dict[str, Any]:
    """Driving directions only (Kakao Mobility free tier doesn't include transit)."""
    if not KAKAO_REST_API_KEY:
        return {"ok": False, "error": "KAKAO_REST_API_KEY not configured"}
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    params = {
        "origin": f"{origin_x},{origin_y}",
        "destination": f"{dest_x},{dest_y}",
        "priority": "RECOMMEND",
    }
    async with httpx.AsyncClient(timeout=DEFAULT_HTTP_TIMEOUT) as c:
        r = await c.get(KAKAO_DIRECTIONS_URL, headers=headers, params=params)
        if r.status_code >= 400:
            return {"ok": False, "error": f"kakao {r.status_code}: {r.text[:200]}"}
        data = r.json()
    routes = data.get("routes", [])
    if not routes:
        return {"ok": True, "summary": None}
    s = routes[0].get("summary", {})
    return {
        "ok": True,
        "summary": {
            "distance_m": s.get("distance"),
            "duration_s": s.get("duration"),
            "toll_won": s.get("fare", {}).get("toll"),
            "taxi_won": s.get("fare", {}).get("taxi"),
        },
    }


async def fetch_url(url: str, max_chars: int = 8000) -> Dict[str, Any]:
    """GET a URL and return text content (HTML stripped)."""
    if not (url.startswith("http://") or url.startswith("https://")):
        return {"ok": False, "error": "url must start with http(s)://"}
    async with httpx.AsyncClient(
        timeout=DEFAULT_HTTP_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (Dailylife bot)"},
    ) as c:
        try:
            r = await c.get(url)
        except httpx.HTTPError as e:
            return {"ok": False, "error": f"fetch failed: {e}"}
    ctype = r.headers.get("content-type", "")
    body = r.text
    if "html" in ctype or "<html" in body[:500].lower():
        # Drop scripts/styles, then strip tags.
        body = re.sub(r"<script[\s\S]*?</script>", " ", body, flags=re.I)
        body = re.sub(r"<style[\s\S]*?</style>", " ", body, flags=re.I)
        body = _strip_html(body)
        body = re.sub(r"\s+", " ", body).strip()
    return {
        "ok": True,
        "status": r.status_code,
        "content_type": ctype,
        "url_final": str(r.url),
        "text": body[:max_chars],
        "truncated": len(body) > max_chars,
    }
