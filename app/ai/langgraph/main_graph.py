# 전체 챗 파이프라인의 LangGraph 버전 (최상위 오케스트레이션 그래프)

from langgraph.graph import StateGraph, START, END

from app.ai.langgraph.state import ChatGraphState
from app.ai.langgraph.get_api_graph import get_api_graph

from app.ai.llm_use.llm_calling_langchain import classify_message_langchain
from app.ai.rag.retriever import search_policy
from app.ai.llm_use.llm_calling_langchain import (
    generate_response_langchain,
    generate_response_langchain_sllm,
    generate_general_response,
)
from app.ai.rag.semantic_cache import semantic_cache

# LangSmith 연동은 .env(LANGSMITH_TRACING/LANGSMITH_API_KEY/LANGSMITH_PROJECT)에서 설정.
# 이 그래프 전용 설정이 아니라 앱 전역 트레이싱 스위치라 여기 두지 않음.

# 캐시 확인 및 state갱신
def check_cache_node(state: ChatGraphState) -> dict:
    cached_response = semantic_cache.search(state["message"], state["member_id"])
    if cached_response:
        print(f"[LangGraph] 캐시 조회 결과: {cached_response}")
        return {"response": cached_response, "cache_hit": True}
    print("[LangGraph] 캐시 조회 결과: MISS")
    return {"cache_hit": False}

def route_after_cache(state: ChatGraphState) -> str:
    decision = "hit" if state.get("cache_hit") else "miss"
    print(f"[LangGraph] route_after_cache -> {decision}")
    return decision


# 1차분류 : get_api, get_my_profile, get_policy
def classify_message_node(state: ChatGraphState) -> dict:
    classification = classify_message_langchain(state["message"])
    print(f"[LangGraph] 1차 분류 결과: {classification}")
    return {"classification": classification}


def route_after_classify_message(state: ChatGraphState) -> str:
    classification = state.get("classification")
    if classification == "get_api":
        decision = "api"
    elif classification == "get_my_profile":
        decision = "profile"
    elif classification == "get_policy":
        decision = "policy"
    else:
        decision = "general"
    print(f"[LangGraph] route_after_classify_message -> {decision}")
    return decision


# get_api(1차 분류) 이후의 2차 분류(QUERY/ACTION/GENERAL)와 그 라우팅은 get_api_graph.py로 분리됨
def run_get_api_node(state: ChatGraphState) -> dict:
    print("[LangGraph] run_get_api 진입 (get_api_graph 서브그래프 실행)")
    result = get_api_graph.invoke({
        "message": state["message"],
        "db": state["db"],
        "member_id": state["member_id"],
    })
    return {"response": result["response"]}


def run_profile_node(state: ChatGraphState) -> dict:
    print("[LangGraph] run_profile 진입")
    member = state["member"]
    data = f"- 회원번호: {member.id} / email: {member.email} / 회원명: {member.name} / age: {member.age} "
    response = generate_response_langchain_sllm(state["message"], data)
    return {"response": response}


def run_policy_node(state: ChatGraphState) -> dict:
    print("[LangGraph] run_policy 진입")
    context = search_policy(state["message"])
    response = generate_response_langchain(state["message"], context)
    return {"response": response}

def run_general_node(state: ChatGraphState) -> dict:
    print("[LangGraph] run_general 진입")
    response = generate_general_response(state["message"])
    return {"response": response}


def store_cache_node(state: ChatGraphState) -> dict:
    # 캐시 히트로 종료된 경우 check_cache 이후 바로 END로 빠지므로 이 노드는 호출되지 않음
    print("[LangGraph] store_cache 진입")
    semantic_cache.store(state["message"], state["response"], state["member_id"])
    return {}


def build_chat_graph():
    # Node는 처리 과정, State는 데이터 묶음으로 노드들에게 전달되고 공유
    graph = StateGraph(ChatGraphState)

    graph.add_node("check_cache", check_cache_node)
    graph.add_node("classify_message", classify_message_node)
    graph.add_node("run_get_api", run_get_api_node)
    graph.add_node("run_profile", run_profile_node)
    graph.add_node("run_policy", run_policy_node)
    graph.add_node("run_general", run_general_node)
    graph.add_node("store_cache", store_cache_node)

    # START : 그래프를 실행할 때의 진입점
#   check_cache ──(hit)──────────────────────────────────────────────▶ END
#       │(miss)
#       ▼
#   classify_message ──(get_api)──▶ run_get_api (get_api_graph 서브그래프: QUERY/ACTION/GENERAL)─┐
#       │(get_my_profile)──▶ run_profile ─────────────────────────────────────────────────────┤
#       │(get_policy)──────▶ run_policy ───────────────────────────────────────────────────────┤
#       └(그 외)───────────▶ run_general ───────────────────────────────────────────────────────┤
#                                                                                      ▼
#                                                                                  store_cache ─▶ END

    # Node = 행동, Edge = 이동 규칙
    # check_cache노드부터 이동하여 실행
    graph.add_edge(START, "check_cache")
    graph.add_conditional_edges(
        "check_cache",  # 1. 분기할 기준 노드
        route_after_cache, # 2. 분기값을 결정하는 함수
        {"hit": END, "miss": "classify_message"}  # 3. 분기값 → 다음 노드
    )
    graph.add_conditional_edges(
        "classify_message",   #classify_message는 앞의 분기에서 선택됐을 때만 실행
        route_after_classify_message,
        {
            "api": "run_get_api", #api값이 선택된 경우, get_api_graph 서브그래프 실행
            "profile": "run_profile",
            "policy": "run_policy",
            "general": "run_general",
        },
    )

    for node in ("run_get_api", "run_profile", "run_policy", "run_general", ):
        graph.add_edge(node, "store_cache")

    graph.add_edge("store_cache", END)

    return graph.compile()


chat_graph = build_chat_graph()

# 그래프 구조를 서버 기동 시 1회 콘솔에 출력 (노드/엣지 전체를 한눈에 확인용)
print("[LangGraph] chat_graph 구조 (mermaid) ↓↓↓")
print(chat_graph.get_graph().draw_mermaid())


# 기존 chat.py의 create_chat() 내부 분기(캐시 조회 ~ 응답 생성) 진입 함수
def run_chat_graph(message: str, db, member) -> str:
    # Graph 실행을 시작할 때 StateGraph객체의 invoke함수를 실행하여 초기 State 주입
    final_state = chat_graph.invoke({
        "message": message,
        "db": db,
        "member": member,
        "member_id": member.id,
    })
    return final_state["response"]
