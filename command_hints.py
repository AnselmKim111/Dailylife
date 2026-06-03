"""Slash hint — 자연어 → 명령 매칭 정규식.

답변 끝에 '💡 다음엔 /command' 1줄. 같은 응답에서 1회만.
사용자가 같은 패턴 3회 이상 시 매크로 등록 제안.
v13 W3.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# (정규식, 추천 명령 템플릿, 사람 명 캡처 그룹 인덱스|None)
HINTS: List[Tuple[str, str, Optional[int]]] = [
    # 일정
    (r"오늘\s*(?:뭐|어떤|일정|있)", "/today", None),
    (r"이번\s*주\s*(?:뭐|일정)", "/week", None),
    (r"앞으로\s*\d*\s*(?:일|주|개월)?\s*(?:일정|뭐)", "/agenda", None),

    # 회상
    (r"(\S+)\s*(?:관련된|관련|에 대해)\s*(?:거|것)\s*(?:다|모두)\s*(?:알려|보여)", "/recall {0}", 1),
    (r"그\s*(영화|책|노래|카페|식당)\s*(?:뭐였|어디였|이름)", "/ask {0}", None),

    # 사람
    (r"(?:사람|연락처|인맥)\s*목록", "/people", None),

    # 골
    (r"(?:장기\s*)?골\s*목록|진행\s*중인\s*골", "/goals", None),

    # 지출
    (r"이번\s*달\s*지출|지출\s*요약|얼마\s*썼", "/spending", None),

    # 습관
    (r"내\s*습관|습관\s*기록", "/habits", None),

    # 메모
    (r"메모\s*(?:목록|뭐\s*있)", "/notes", None),

    # 여행 계획
    (r"(\S+)\s*(?:여행|출장)\s*(?:계획|짜|만들|풀패키지)", "/plan_trip <목적지> <시작> <끝>", None),

    # 결정
    (r"(.{2,30})\s*vs\s*(.{2,30})\s*(?:결정|선택|어느|뭐가)", "/decide <상황>", None),

    # 미션
    (r"밤사이\s*|자율\s*프로젝트|풀패키지|혼자\s*해줘", "/mission <목표>", None),

    # 자동 액션 룰
    (r"(?:자동|룰)\s*(?:추가|등록|만들)", "/rules add <자연어>", None),

    # 전문가 모드
    (r"변호사|법률\s*(?:자문|상담)", "/expert legal", None),
    (r"회계사|세무\s*(?:상담|자문)", "/expert finance", None),
    (r"커리어\s*코치|이직\s*상담", "/expert career", None),

    # 통역
    (r"번역해|통역해", "/translate <텍스트>", None),

    # 구독
    (r"구독\s*(?:확인|목록|뭐\s*있)", "/charges", None),

    # 관계
    (r"(?:식어가는|소원해진)\s*(?:사람|친구)", "/relationships", None),
    (r"인간관계\s*(?:정리|상태)", "/relationships", None),

    # 글쓰기
    (r"(?:긴\s*글|보고서|연설문)\s*(?:써|작성)", "/write <목적>", None),

    # 알림 끄기
    (r"(?:알림|nudge)\s*(?:끄|꺼)|시끄러", "/nudges", None),

    # 조용 시간
    (r"조용\s*(?:시간|모드)", "/quiet", None),

    # 음성
    (r"음성으로\s*(?:말|답)|TTS", "/say <텍스트>", None),

    # 그래프 회상
    (r"(\S+)\s*랑\s*(?:.{0,10}\s*)?(?:갔|봤|만난)\s*(?:거|것|곳)", "/ask <질문>", None),

    # 메일 협상
    (r"(?:회의|미팅)\s*시간\s*(?:잡|조율)", "/negotiate <email> <주제>", None),

    # 능력 디스커버리
    (r"(?:뭐|무엇)\s*(?:할\s*수|가능|돼)", "/can", None),
    (r"도와줄\s*수\s*있", "/can", None),
]


def suggest_command(user_text: str) -> Optional[str]:
    """user_text가 어떤 hint 패턴에 매칭되면 추천 명령 1개 반환."""
    if not user_text or len(user_text) > 200:
        return None
    text = user_text.strip()
    for pattern, command_template, capture_idx in HINTS:
        m = re.search(pattern, text)
        if not m:
            continue
        if capture_idx is not None:
            try:
                target = m.group(capture_idx).strip()
                if target:
                    return command_template.replace("{0}", target)
            except IndexError:
                pass
        return command_template
    return None
