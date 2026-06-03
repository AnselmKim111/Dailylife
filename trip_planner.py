"""Trip planning v2 — mission 확장.

/plan_trip 명령이 trip row + autonomous mission 생성.
mission은 v5 인프라 그대로 사용 — max_hops=40, kind='mission' (Opus).
완료 시 itinerary_md + GCal 일괄 등록 카드 (사용자 동의 시).
v12 W6.
"""

from __future__ import annotations

import logging
from typing import Optional

import db

logger = logging.getLogger(__name__)


TRIP_PLANNING_PROMPT = (
    "[TRIP PLANNING v2] 당신은 사용자 *통합 여행 플래너*. 한 mission 안에서 "
    "다음을 모두 완성:\n"
    "1. **항공편** — 출국·귀국 후보 3개 비교 (가격·시간대·경유 횟수). "
    "fetch_url + web_search로 실제 가격 인용. [출처](url) inline.\n"
    "2. **숙소** — 후보 3개 비교 (1박 가격·평점·위치). 사용자 *조용·로컬·여유* "
    "선호 (persona) 반영.\n"
    "3. **날별 동선** — 도착 → 출국까지 시간 단위 동선. 이동 시간·식당·관광지.\n"
    "4. **식당** — 끼니별 추천 (현지 음식 우선). 점심 ≤2만, 저녁 ≤5만 권장.\n"
    "5. **예산 추정** — 항공/숙소/식/교통/관광/기타 표.\n"
    "6. **위험** — 비자·날씨·환율·치안 1-2줄.\n"
    "7. **체크리스트** — 출발 전 준비 ≤5개.\n\n"
    "최종 결과 마크다운 헤더:\n"
    "# 🛫 {destination} ({start_date}~{end_date}) {party_size}인\n\n"
    "각 사실 주장에 [출처](url). 추측은 '추측이지만' 접두. 모르면 '모름'.\n\n"
    "여행 정보:\n"
    "- destination: {destination}\n"
    "- 날짜: {start_date} ~ {end_date}\n"
    "- 인원: {party_size}\n"
    "- 예산: {budget_text}\n"
)


def start_trip(
    chat_id: int, destination: str, start_date: str, end_date: str,
    party_size: int = 1, budget_total: Optional[int] = None,
) -> int:
    """trip row 생성 + mission 시작 (mission_id로 mission_pump가 처리)."""
    budget_text = (f"{budget_total:,}원" if budget_total else "자유")
    goal = TRIP_PLANNING_PROMPT.format(
        destination=destination,
        start_date=start_date,
        end_date=end_date,
        party_size=party_size,
        budget_text=budget_text,
    )
    title = f"{destination} {start_date}~{end_date}"[:80]
    mission_id = db.add_mission(chat_id, title, goal, max_hops=40)
    trip_id = db.create_trip(
        chat_id, destination, start_date, end_date,
        party_size=party_size, budget_total=budget_total,
        mission_id=mission_id)
    return trip_id


def link_mission_result(trip_id: int, mission_id: int) -> bool:
    """mission 완료 시 mission.result_md → trip.itinerary_md로 복사."""
    mission = db.get_mission(mission_id)
    if not mission or not mission.get("result_md") if hasattr(mission, "get") else not mission["result_md"]:
        return False
    result = mission["result_md"]
    return db.update_trip(trip_id, itinerary_md=result, status="done")
