# Action(주문/상품 등록·수정 등 DB를 변경하는 API 호출) 파이프라인의 LangGraph 버전.
#
#   classify_category ─(카테고리 없음)─▶ cannot_process ─▶ END
#         │(카테고리 있음)
#         ▼
#   select_action ─(tool 선택 실패: 정보 부족)─▶ END (안내 메시지)
#         │(tool 선택됨)
#         ▼
#   execute_action ─▶ END
from langgraph.graph import StateGraph, START, END

# action_pipeline.py의 카테고리 분류 함수(_classify_category)와 registry.py의 action 선택 함수
# (get_action_list_by_category, get_action_name)를 그대로 재사용합니다 (로직 복제 방지). _classify_category는
# 모듈 내부용(_ prefix)이지만 같은 프로젝트 내에서 재사용 목적으로 import합니다.
from app.ai.api_use.action.action_pipeline import _classify_category
from app.ai.api_use.action.registry import get_action_list_by_category, get_action_name, execute_action
from app.ai.langgraph.state import ChatGraphState


def classify_category_node(state: ChatGraphState) -> dict:
    category = _classify_category(state["message"])
    return {"category": category}


def select_action_node(state: ChatGraphState) -> dict:
    action_list = get_action_list_by_category(state["category"])
    action_name, args = get_action_name(state["message"], action_list)

    if action_name is None:
        return {"selected_action_name": None, "response": args}

    return {"selected_action_name": action_name, "selected_action_args": args}


def execute_action_node(state: ChatGraphState) -> dict:
    try:
        response = execute_action(
            state["selected_action_name"],
            state["selected_action_args"],
            state["db"],
            state["member_id"],
        )
    except Exception as e:
        print(f"[LangGraph][Action] 실행 실패 | error={e}")
        response = f"요청 처리 중 오류가 발생했습니다: {str(e)}"
    return {"response": response}


def cannot_process_node(state: ChatGraphState) -> dict:
    return {"response": "처리할 수 없는 요청입니다."}


def route_after_classify(state: ChatGraphState) -> str:
    return "select" if state.get("category") else "cannot_process"


def route_after_select(state: ChatGraphState) -> str:
    return "execute" if state.get("selected_action_name") else "no_action"


def build_action_graph():
    graph = StateGraph(ChatGraphState)

    graph.add_node("classify_category", classify_category_node)
    graph.add_node("select_action", select_action_node)
    graph.add_node("execute_action", execute_action_node)
    graph.add_node("cannot_process", cannot_process_node)

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
    graph.add_edge("execute_action", END)
    graph.add_edge("cannot_process", END)

    return graph.compile()


action_graph = build_action_graph()

# 그래프 구조를 서버 기동 시 1회 콘솔에 출력 (노드/엣지 전체를 한눈에 확인용)
print("[LangGraph][Action] action_graph 구조 (mermaid) ↓↓↓")
print(action_graph.get_graph().draw_mermaid())
