# 그래프 전체(main_graph, sql_graph)가 공유하는 상태 정의

# LangGraph의 각 노드는 dict를 입력받아 dict(갱신분)를 반환하는 함수이며,
# 여러 노드가 반환한 dict는 아래 State 정의를 기준으로 병합(merge) 
from typing import Any, Optional, TypedDict


class ChatGraphState(TypedDict, total=False):
    message: str
    db: Any            
    member: Any        
    member_id: int

    # 캐시여부
    cache_hit: bool

    # 1차 분류: get_api / get_my_profile 등
    classification: str
    
    # 2차 분류: QUERY/ACTION/GENERAL
    intent: str

    # 하위 그래프인 sql_graph/action_graph에서 반복작업실패(3회)한 경우
    escalate: bool#  escalate:True로 상위 그래프로 전파
    reclassify: bool  # 1차 재분류가 다시 됐다면 True
    reclassify_count: int  # 1차 분류로 되돌아 온 횟수(1번만 추가 재분류처리)

    # HITL 을 위한 변수
    pending_confirm: Optional[dict]

    # # sql_graph 서브그래프() 상태
    # current_sql: str
    # corrected_sql: Optional[str]
    # validation_error: Optional[str]
    # execution_error: Optional[str]
    # retry_count: int  # 재시도 카운트
    # query_results: list[dict[str, Any]]

    # # action_graph 상태
    # category: Optional[str]
    # selected_action_name: Optional[str]
    # selected_action_args: dict
    # action_error: Optional[str]


    # 최종 응답
    response: str


# sql_graph 전용 State 
class SqlGraphState(TypedDict, total=False):
    message: str
    db: Any
    member_id: int

    # sql_graph 진행 상태
    current_sql: str
    corrected_sql: Optional[str]
    validation_error: Optional[str]
    execution_error: Optional[str]
    retry_count: int  # 재시도 카운트
    query_results: list[dict[str, Any]]

    # 부모 그래프(get_api_graph/main_graph)로 return할 값
    response: str
    escalate: bool  


# action_graph 전용 State
class ActionGraphState(TypedDict, total=False):
    message: str
    db: Any
    member_id: int

    # action_graph 진행 상태
    category: Optional[str]
    selected_action_name: Optional[str]
    selected_action_args: dict
    action_error: Optional[str]  
    retry_count: int

    # 부모 그래프(get_api_graph/main_graph)로 return할 값
    response: str
    escalate: bool 
