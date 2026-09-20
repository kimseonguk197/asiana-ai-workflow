# Text-to-SQL 파이프라인의 LangGraph 버전
# 명시적인 그래프(사이클)로 표현
#   generate_sql ─▶ validate_sql ─▶ execute_sql ─▶ format_success ─▶ END
#                        │  ▲            │
#                    (invalid)      (실행 실패)
#                        │  │            │
#                        ▼  └────────────┘
#                     fix_sql  ──▶ (재시도 한도 초과 시) format_failure ─▶ END
from langgraph.graph import StateGraph, START, END

from app.ai.api_use.query.generator import generate_sql, fix_sql
from app.ai.api_use.query.validator import validate_and_correct
from app.ai.api_use.query.executor import execute_sql
from app.ai.api_use.query.response import format_sql_result, format_error_response
# from app.ai.langgraph.state import ChatGraphState
# 기존에는 main_graph와 공유하는 ChatGraphState를 그대로 썼지만, 분리
from app.ai.langgraph.state import SqlGraphState

# 기존 sql_pipeline.py의 `for attempt in range(3)`와 동일한 총 시도 횟수
MAX_ATTEMPTS = 3


# def generate_sql_node(state: ChatGraphState) -> dict:
# 나머지 node와 edge도 모두 SqlGraphState로 변경
def generate_sql_node(state: SqlGraphState) -> dict:
    print("[LangGraph][SQL] SQL 생성 시도 #1")
    sql = generate_sql(state["message"])
    return {"current_sql": sql, "retry_count": 0}


def fix_sql_node(state: SqlGraphState) -> dict:
    error = state.get("execution_error") or state.get("validation_error")
    retry_count = state.get("retry_count", 0) + 1
    print(f"[LangGraph][SQL] SQL 수정 시도 #{retry_count + 1} | 이전 오류: {error}")
    fixed_sql = fix_sql(state["current_sql"], error)
    return {
        "current_sql": fixed_sql,
        "retry_count": retry_count,
        "validation_error": None,
        "execution_error": None,
    }


def validate_sql_node(state: SqlGraphState) -> dict:
    result = validate_and_correct(state["current_sql"])
    if not result.is_valid:
        print(f"[LangGraph][SQL] 검증 실패: {result.error_message}")
        return {"validation_error": result.error_message, "corrected_sql": None}
    print("[LangGraph][SQL] 검증 통과")
    return {"corrected_sql": result.corrected_sql, "validation_error": None}

# validate_sql 이후 라우팅: 통과 -> execute_sql / 실패+재시도가능 -> fix_sql / 재시도소진 -> format_failure
def route_after_validate(state: SqlGraphState) -> str:
    if state.get("validation_error"):
        decision = "retry" if state.get("retry_count", 0) < MAX_ATTEMPTS - 1 else "fail"
    else:
        decision = "execute"
    print(f"[LangGraph][SQL] route_after_validate -> {decision}")
    return decision



def execute_sql_node(state: SqlGraphState) -> dict:
    try:
        results = execute_sql(
            state["db"],
            state["corrected_sql"],
            {"current_member_id": state["member_id"]},
        )
        print(f"[LangGraph][SQL] 성공 | {len(results)}건 조회 | 재시도={state.get('retry_count', 0)}회")
        return {"query_results": results, "execution_error": None}
    except Exception as e:
        print(f"[LangGraph][SQL] 실행 실패: {e}")
        return {"execution_error": str(e)}

# execute_sql 이후 라우팅: 성공 -> format_success / 실패+재시도가능 -> fix_sql / 재시도소진 -> format_failure
def route_after_execute(state: SqlGraphState) -> str:
    if state.get("execution_error"):
        decision = "retry" if state.get("retry_count", 0) < MAX_ATTEMPTS - 1 else "fail"
    else:
        decision = "success"
    print(f"[LangGraph][SQL] route_after_execute -> {decision}")
    return decision

# 성공응답
def format_success_node(state: SqlGraphState) -> dict:
    print("[LangGraph][SQL] format_success 진입")
    response = format_sql_result(state["message"], state["query_results"])
    return {"response": response}


# 실패응답
def format_failure_node(state: SqlGraphState) -> dict:
    error = state.get("execution_error") or state.get("validation_error")
    print(f"[LangGraph][SQL] format_failure 진입 | 최종 오류: {error}")
    response = format_error_response(state["message"], error)
    # return {"response": response}
    # 재시도(MAX_ATTEMPTS)까지 다 쓰고도 실패 → main_graph에서 1차 분류부터 재시도할 수 있도록 신호 전달
    return {"response": response, "escalate": True}



def build_sql_graph():
    graph = StateGraph(SqlGraphState)

    graph.add_node("generate_sql", generate_sql_node)
    graph.add_node("fix_sql", fix_sql_node)
    graph.add_node("validate_sql", validate_sql_node)
    graph.add_node("execute_sql", execute_sql_node)
    graph.add_node("format_success", format_success_node)
    graph.add_node("format_failure", format_failure_node)

    graph.add_edge(START, "generate_sql")
    # generate_sql 노드가 실행된 다음 validate_sql 노드를 실행
    graph.add_edge("generate_sql", "validate_sql")
    graph.add_conditional_edges(
        "validate_sql",
        route_after_validate,
        {"execute": "execute_sql", "retry": "fix_sql", "fail": "format_failure"},
    )
    graph.add_conditional_edges(
        "execute_sql",
        route_after_execute,
        {"success": "format_success", "retry": "fix_sql", "fail": "format_failure"},
    )
    # fix_sql실행후 다시 validate_sql노드로 이동
    graph.add_edge("fix_sql", "validate_sql")
    graph.add_edge("format_success", END)
    graph.add_edge("format_failure", END)

    return graph.compile()


# 모듈 로드 시 1회 컴파일하여 재사용 (기존 llm_sql 등 모듈 레벨 인스턴스와 동일한 패턴)
sql_graph = build_sql_graph()

# 그래프 구조를 서버 기동 시 1회 콘솔에 출력 (사이클 포함 노드/엣지 전체를 한눈에 확인용)
print("[LangGraph][SQL] sql_graph 구조 ↓↓↓")
print(sql_graph.get_graph().draw_mermaid())
