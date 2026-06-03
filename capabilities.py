"""봇 능력 사전 — 카테고리별 자연어 설명.

`/can` 명령이 사용 + system_prompt가 능력 질문 시 참고.
명령어 노출 X — '자연어로 부탁하면 다 됨' 강조.
v13 W2.
"""

from __future__ import annotations

from typing import Dict, List

CAPABILITIES: Dict[str, List[str]] = {
    "일정": [
        "오늘·이번주·앞 60일 일정 (로컬+구글 캘린더 통합)",
        "이벤트 추가·수정·삭제",
        "출발 시간 자동 계산 (Leave-by, 카카오 길찾기)",
        "회의 초대 자동 RSVP (auto_rule 설정 시)",
        "보호 시간 침범 미팅 자동 거절 메일",
        "여행 모드 자동 감지 (날씨·시간대 destination 기준)",
    ],
    "메모리": [
        "사람·연락처·중요한 날짜 (음력 자동 변환)",
        "노트 자유 기록 + FTS 검색",
        "facts 키-값 기억",
        "장기 기억 의미 검색 (lifelog RAG, 수개월 전 발화도 회수)",
        "엔티티 관계 그래프 (X와 관련된 모든 것 한 번에)",
        "주간 인물 요약 (persona)",
    ],
    "결정": [
        "여러 옵션 비교 → 추천 1개 + 이유",
        "큰 결정 multi-factor 시뮬레이션 (예: 이직·이사·결혼)",
        "여행 통째 계획 (항공·숙소·동선·식당·예산)",
        "전문가 시점 시뮬레이션 (변호사·회계사·디자이너·엔지니어·마케터 등)",
    ],
    "외부 행동": [
        "Gmail 메일 전송·답장 초안",
        "Google Calendar RSVP",
        "회의 시간 자동 협상 (외부인과 메일 왕복)",
        "택배 추적",
        "전화 통화 (Twilio 설정 시 — 식당 예약·문의 등 봇이 직접)",
        "웹 폼 자동 채우기 (Playwright + vault 설정 시)",
    ],
    "글쓰기": [
        "Long-form 보고서·연설문·논문 multi-pass 협업 (사용자 톤 학습)",
        "외국어 ↔ 한국어 양방향 통역",
        "문서 위험 분석 (계약서·약관)",
    ],
    "재정": [
        "구독 자동 추적 (Netflix, ChatGPT 등 가격 인상 감지)",
        "이상 결제 알림",
        "지출 카테고리별 요약",
        "예산 가드 (월별 예산 초과 시 알림)",
    ],
    "능동 알림": [
        "아침 한 통 brief (오늘 일정 + 잊을 만한 것 + 임박 골 + 메일 요약)",
        "출발 시간 자동 알림 (location 있는 일정)",
        "결혼 D-day, 골 마일스톤 D-30/14/3/1",
        "관계 식어가는 친구 1줄 알림 (CRM)",
        "조용 시간 23:00-07:00 (모든 알림 자동 묵음)",
    ],
    "자율 미션": [
        "/mission — 밤사이 여러 시간 자율 프로젝트 (예: 부산 여행 풀패키지)",
        "/agent — 다단계 자율 에이전트 (web_search + fetch_url + 도구 조합)",
        "장기 watch (예: 부동산 가격 11억 이하면 알려줘)",
    ],
}


def render_overview() -> str:
    """7개 카테고리 메뉴 + 각 카테고리에 항목 개수."""
    lines = ["📚 가능한 일 (자연어로 부탁하면 다 됨)"]
    for cat, items in CAPABILITIES.items():
        lines.append(f"  • {cat} ({len(items)})")
    lines.append("")
    lines.append("자세히: /can <카테고리> (예: /can 일정)")
    lines.append("또는 그냥 자연어로 부탁 — 명령 외울 필요 X")
    return "\n".join(lines)


def render_category(cat: str) -> str:
    """특정 카테고리 항목 전체."""
    items = CAPABILITIES.get(cat)
    if not items:
        # 부분 매칭
        for k in CAPABILITIES:
            if cat in k or k in cat:
                items = CAPABILITIES[k]
                cat = k
                break
    if not items:
        return f"모르는 카테고리. /can으로 목록 보기."
    lines = [f"📚 {cat}"]
    for item in items:
        lines.append(f"  · {item}")
    return "\n".join(lines)


def render_for_prompt() -> str:
    """system_prompt 주입용 — 1-2줄 요약."""
    categories = " · ".join(CAPABILITIES.keys())
    return (
        f"Capabilities (자연어로 부탁 가능): {categories}. "
        "능력 질문에 명령어 나열 금지 — 자연어로 안내."
    )
