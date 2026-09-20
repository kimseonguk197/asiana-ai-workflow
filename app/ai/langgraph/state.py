# 그래프 전체(main_graph, sql_graph)가 공유하는 상태 정의

# LangGraph의 각 노드는 dict를 입력받아 dict(갱신분)를 반환하는 함수이며,
# 여러 노드가 반환한 dict는 아래 State 정의를 기준으로 병합(merge) 
from typing import Any, Optional, TypedDict


class ChatGraphState(TypedDict, total=False):
    # ── 입력값 ──────────────────────────────────────────────
    message: str
    db: Any            # sqlalchemy.orm.Session
    member: Any        # models.Member (get_my_profile 분기에서 사용)
    member_id: int

    # ── 캐시(semantic_cache) ───────────────────────────────
    cache_hit: bool

    # ── 1차 분류: get_api / get_my_profile / 그 외(정책 RAG) ─
    classification: str
    
    # ── 2차 분류(action == "get_api"인 경우만): QUERY/ACTION/GENERAL
    intent: str

    # ── Text-to-SQL 서브그래프(sql_graph) 상태 ───────────────
    current_sql: str
    corrected_sql: Optional[str]
    validation_error: Optional[str]
    execution_error: Optional[str]
    retry_count: int  # 재시도 카운트
    query_results: list[dict[str, Any]]

    # ── Action 파이프라인 상태 (action_graph.py 전용) ───────
    category: Optional[str]
    selected_action_name: Optional[str]
    selected_action_args: dict
    action_error: Optional[str]  # execute_action 실패 시 에러 메시지 (에러 시 select_action으로 재시도)

    # ── 최종 응답 ───────────────────────────────────────────
    response: str
