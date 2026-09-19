# get_api(1차 분류) 이후 서브그래프: 2차 분류(QUERY/ACTION/GENERAL)와 그 라우팅
# main_graph.py가 떠안고 있던 이 부분을 분리해, main_graph는 get_api_graph를 한 노드로만 호출한다.

from langgraph.graph import END, StateGraph, START

from app.ai.langgraph.state import ChatGraphState
from app.ai.langgraph.sql_graph import sql_graph

from app.ai.langgraph.action_graph import action_graph
from app.ai.api_use.chat_classify import classify_intent
from app.ai.llm_use.llm_calling_langchain import generate_general_response


# 2차분류 : QUERY, ACTION, GENERAL
def classify_intent_node(state: ChatGraphState) -> dict:
    intent = classify_intent(state["message"])
    print(f"[LangGraph][GetApi] 2차 분류 결과: {intent}")
    return {"intent": intent}


def run_sql_node(state: ChatGraphState) -> dict:
    print("[LangGraph][GetApi] run_sql 진입 (sql_graph 서브그래프 실행)")
    result = sql_graph.invoke({
        "message": state["message"],
        "db": state["db"],
        "member_id": state["member_id"],
        "retry_count": 0,
    })
    return {"response": result["response"]}


def run_action_node(state: ChatGraphState) -> dict:
    print("[LangGraph][GetApi] run_action 진입 (action_graph 서브그래프 실행)")
    result = action_graph.invoke({
        "message": state["message"],
        "db": state["db"],
        "member_id": state["member_id"],
    })
    return {"response": result["response"]}


def run_general_node(state: ChatGraphState) -> dict:
    print("[LangGraph][GetApi] run_general 진입")
    response = generate_general_response(state["message"])
    return {"response": response}


def route_after_intent(state: ChatGraphState) -> str:
    intent = state.get("intent")
    if intent == "QUERY":
        decision = "query"
    elif intent == "ACTION":
        decision = "action"
    else:
        decision = "general"
    print(f"[LangGraph][GetApi] route_after_intent -> {decision}")
    return decision


def build_get_api_graph():
    graph = StateGraph(ChatGraphState)

    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("run_sql", run_sql_node)
    graph.add_node("run_action", run_action_node)
    graph.add_node("run_general", run_general_node)

    # START : 그래프를 실행할 때의 진입점
#   classify_intent ──(QUERY)───▶ run_sql ──┐
#         │──────────(ACTION)───▶ run_action┤
#         └──────────(GENERAL)──▶ run_general┤
#                                             ▼
#                                            END

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {"query": "run_sql", "action": "run_action", "general": "run_general"},
    )
    graph.add_edge("run_sql", END)
    graph.add_edge("run_action", END)
    graph.add_edge("run_general", END)
    return graph.compile()


get_api_graph = build_get_api_graph()

# 그래프 구조를 서버 기동 시 1회 콘솔에 출력 (노드/엣지 전체를 한눈에 확인용)
print("[LangGraph][GetApi] get_api_graph 구조 (mermaid) ↓↓↓")
print(get_api_graph.get_graph().draw_mermaid())
