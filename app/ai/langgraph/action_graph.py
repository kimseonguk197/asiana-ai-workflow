# Action(주문/상품 등록·수정 등 DB를 변경하는 API 호출) 파이프라인의 LangGraph 버전.
#
#   classify_category ─(카테고리 없음)─▶ cannot_process ─▶ END
#         │(카테고리 있음)
#         ▼
#   select_action ─(tool 선택 실패: 정보 부족)─▶ END (안내 메시지)
#         │(tool 선택됨)
#         ▼
#   execute_action ─(성공)─────────────────────────────────▶ END
#         │(실행 오류, 재시도 가능)──▶ select_action (재시도)
#         └(실행 오류, 재시도 소진)──▶ action_failed ─▶ END
from langgraph.graph import StateGraph, START, END

# action_pipeline.py의 카테고리 분류 함수(_classify_category)와 registry.py의 action 선택 함수
# (get_action_list_by_category, get_action_name)를 그대로 재사용합니다 (로직 복제 방지). _classify_category는
# 모듈 내부용(_ prefix)이지만 같은 프로젝트 내에서 재사용 목적으로 import합니다.
from app.ai.api_use.action.action_pipeline import _classify_category
from app.ai.api_use.action.registry import get_action_list_by_category, get_action_name, execute_action
# from app.ai.langgraph.state import ChatGraphState
# 기존에는 main_graph와 공유하는 ChatGraphState를 그대로 썼지만, 분리
from app.ai.langgraph.state import ActionGraphState

import os
from langgraph.types import interrupt
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.runnables import RunnableConfig

# def classify_category_node(state: ChatGraphState) -> dict:
# 나머지 node와 edge도 모두 ActionGraphState로 변경
def classify_category_node(state: ActionGraphState) -> dict:
    category = _classify_category(state["message"])
    return {"category": category}

def select_action_node(state: ActionGraphState) -> dict:
    action_list = get_action_list_by_category(state["category"])
    action_name, args = get_action_name(state["message"], action_list)

    if action_name is None:
        return {"selected_action_name": None, "response": args}

    return {"selected_action_name": action_name, "selected_action_args": args}


# 기존: 실행 실패 시 바로 에러 메시지로 END (재시도 없음)
# def execute_action_node(state: ChatGraphState) -> dict:
#     try:
#         response = execute_action(
#             state["selected_action_name"],
#             state["selected_action_args"],
#             state["db"],
#             state["member_id"],
#         )
#     except Exception as e:
#         print(f"[LangGraph][Action] 실행 실패 | error={e}")
#         response = f"요청 처리 중 오류가 발생했습니다: {str(e)}"
#     return {"response": response}
MAX_ATTEMPTS = 2  # execute_action 실패 시 select_action으로 되돌아가 재시도할 최대 횟수
def execute_action_node(state: ActionGraphState) -> dict:
    try:
        response = execute_action(
            state["selected_action_name"],
            state["selected_action_args"],
            state["db"],
            state["member_id"],
        )
        return {"response": response, "action_error": None}
    except Exception as e:
        retry_count = state.get("retry_count", 0) + 1
        print(f"[LangGraph][Action] 실행 실패 (시도 #{retry_count}) | error={e}")
        return {"action_error": str(e), "retry_count": retry_count}


def execute_action_node_hitl(state: ActionGraphState, config: RunnableConfig) -> dict:
    # HITL 순서1.execute_action 직전 정지
    # DB에 checkpoint상태저장(현재 노드에서 사용된 ActionGraphState의 변수, 노드명 등 여러 상태값 저장)
    # {"__interrupt__": (value=아래 dict, ...),)}형태의 값 상위로 전달
    decision = interrupt({
    # HITL 순서4. 이후 Command(resume=decision)으로 이 위치로 다시 호출되면, interrupt()는 approve를 반환
        "action_name": state["selected_action_name"],
        "args": state["selected_action_args"],
    })
    if decision != "approve":
        print(f"[LangGraph][Action] 사용자가 실행을 취소함 | action={state['selected_action_name']}")
        return {"response": "요청하신 작업을 취소했습니다.", "action_error": None}

    # db는 run_action_node(get_api_graph.py)에서 주입하여, config(configurable)로 전달
    db = config["configurable"]["db"]
    try:
        response = execute_action(
            state["selected_action_name"],
            state["selected_action_args"],
            db,
            state["member_id"],
        )
        return {"response": response, "action_error": None}
    except Exception as e:
        retry_count = state.get("retry_count", 0) + 1
        print(f"[LangGraph][Action] 실행 실패 (시도 #{retry_count}) | error={e}")
        return {"action_error": str(e), "retry_count": retry_count}

def action_failed_node(state: ActionGraphState) -> dict:
    print(f"[LangGraph][Action] 재시도 소진 | 최종 오류: {state.get('action_error')}")
    response = f"요청 처리 중 오류가 발생했습니다: {state.get('action_error')}"
    # return {"response": response}
    # 재시도(MAX_ATTEMPTS)까지 다 쓰고도 실패 → main_graph에서 1차 분류부터 재시도할 수 있도록 신호 전달
    return {"response": response, "escalate": True}


def cannot_process_node(state: ActionGraphState) -> dict:
    return {"response": "처리할 수 없는 요청입니다."}


def route_after_classify(state: ActionGraphState) -> str:
    return "select" if state.get("category") else "cannot_process"


def route_after_select(state: ActionGraphState) -> str:
    return "execute" if state.get("selected_action_name") else "no_action"


# execute_action 이후 라우팅: 성공 -> END / 실패+재시도가능 -> select_action(재시도) / 재시도소진 -> action_failed
def route_after_execute(state: ActionGraphState) -> str:
    if state.get("action_error"):
        decision = "retry" if state.get("retry_count", 0) < MAX_ATTEMPTS else "fail"
    else:
        decision = "success"
    print(f"[LangGraph][Action] route_after_execute -> {decision}")
    return decision


def build_action_graph():
    graph = StateGraph(ActionGraphState)

    graph.add_node("classify_category", classify_category_node)
    graph.add_node("select_action", select_action_node)
    # graph.add_node("execute_action", execute_action_node)
    # HITL 적용
    graph.add_node("execute_action", execute_action_node_hitl)
    graph.add_node("cannot_process", cannot_process_node)
    graph.add_node("action_failed", action_failed_node)

    graph.add_edge(START, "classify_category")
    graph.add_conditional_edges(
        "classify_category",
        route_after_classify,
        {"select": "select_action", "cannot_process": "cannot_process"},
    )
    graph.add_conditional_edges(
        "select_action",
        route_after_select,
        {"execute": "execute_action", "no_action": END},
    )
    # 실행 실패 시 select_action으로 되돌아가 재시도
    graph.add_conditional_edges(
        "execute_action",
        route_after_execute,
        {"success": END, "retry": "select_action", "fail": "action_failed"},
    )
    graph.add_edge("cannot_process", END)
    graph.add_edge("action_failed", END)

    # return graph.compile()
    # execute_action 안의 interrupt()가 동작하려면 checkpointer가 필수.
    return graph.compile(checkpointer=_checkpointer)


# PostgresSaver는 (autocommit=True, row_factory=dict_row) 커넥션이 필요.
_checkpointer_pool = ConnectionPool(
    conninfo=os.getenv("DATABASE_URL"),
    max_size=10,
    kwargs={"autocommit": True, "row_factory": dict_row}, #조회 결과를 dict 형태로 반환
)
_checkpointer = PostgresSaver(_checkpointer_pool)
_checkpointer.setup()  # checkpoints 관련 테이블이 없다면 자동 생성

action_graph = build_action_graph()

# 그래프 구조를 서버 기동 시 1회 콘솔에 출력 (노드/엣지 전체를 한눈에 확인용)
print("[LangGraph][Action] action_graph 구조 (mermaid) ↓↓↓")
print(action_graph.get_graph().draw_mermaid())
