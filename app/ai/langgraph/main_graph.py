# 전체 챗 파이프라인의 LangGraph 버전 (최상위 오케스트레이션 그래프)

from langgraph.graph import StateGraph, START, END
#  대기 중인 chat_graph(내부 action_graph)를 사용자의 답변으로 재개(resume)하기 위해 필요.
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig

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
from app.ai.langgraph.check_pointer import main_checkpointer

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
        result = "api"
    elif classification == "get_my_profile":
        result = "profile"
    elif classification == "get_policy":
        result = "policy"
    else:
        result = "general"
    print(f"[LangGraph] route_after_classify_message -> {result}")
    return result


MAX_RECLASSIFY = 1  # sql_graph/action_graph가 재시도까지 소진하고도 실패(escalate)했을 때 1차 분류로 돌아갈 최대 횟수
def run_get_api_node(state: ChatGraphState) -> dict:
    print("[LangGraph] run_get_api 진입 (get_api_graph 서브그래프 실행 결과 조회)")
    # 1차에서 재분류작업 : 2차 분류 이후의 작업에서 처리 실패 날경우 1차에서부터 재작업
    reclassify_count = state.get("reclassify_count", 0)
    # sql_graph/action_graph가 내부 재시도를 다 쓰고도 실패해서 올려보낸 escalate 신호 확인
    if state.get("escalate") and reclassify_count < MAX_RECLASSIFY:
        print(f"[LangGraph] get_api 처리 실패(escalate) → 1차 분류부터 재시도 ({reclassify_count + 1}/{MAX_RECLASSIFY})")
        # state가 재분류 이후에도 그대로 이어지므로, 이전 실패값(escalate/retry_count)은 초기화
        return {"reclassify": True, "reclassify_count": reclassify_count + 1, "escalate": False, "retry_count": 0}
    return {"reclassify": False}

def route_after_get_api(state: ChatGraphState) -> str:
    if state.get("reclassify"):
        decision = "reclassify"
    else:
        decision = "done"
    print(f"[LangGraph] route_after_get_api -> {decision}")
    return decision


def run_profile_node(state: ChatGraphState, config: RunnableConfig) -> dict:
    print("[LangGraph] run_profile 진입")
    member = config["configurable"]["member"]
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
    # Node는 처리 작업, State는 데이터 묶음으로 노드들에게 전달되고, 변경되고, 공유
    graph = StateGraph(ChatGraphState)

    graph.add_node("check_cache", check_cache_node)
    graph.add_node("classify_message", classify_message_node)
    # 컴파일된 get_api_graph를 노드로 직접 추가
    graph.add_node("get_api_graph", get_api_graph)
    graph.add_node("run_get_api", run_get_api_node)
    graph.add_node("run_profile", run_profile_node)
    graph.add_node("run_policy", run_policy_node)
    graph.add_node("run_general", run_general_node)
    graph.add_node("store_cache", store_cache_node)

    # START : 그래프를 실행할 때의 진입점
#   check_cache ──(hit)──────────────────────────────────────────────▶ END
#       │(miss)
#       ▼
#   classify_message ◀────────────────────────────┐(sql_graph/action_graph 재시도까지 소진 후 실패: escalate, 최대 MAX_RECLASSIFY회)
#       │(get_api)──▶ get_api_graph (서브그래프: QUERY/ACTION/GENERAL) ──▶ run_get_api ─┘
#       │(get_my_profile)──▶ run_profile ─────────────────────────────────────────────────────┐
#       │(get_policy)──────▶ run_policy ───────────────────────────────────────────────────────┤
#       └(그 외)───────────▶ run_general ───────────────────────────────────────────────────────┤
#                                                                                      ▼

    # Node = 행동 및 상태값 결정, Edge = 이동 규칙
    # check_cache노드부터 이동하여 실행
    graph.add_edge(START, "check_cache")
    graph.add_conditional_edges(
        "check_cache",  # 1. 노드에서 작업 후 State값 결정
        route_after_cache, # 2. State값을 통해 분기값을 결정하는 함수
        {"hit": END, "miss": "classify_message"}  # 3. 분기값 → 다음 노드
    )
    graph.add_conditional_edges(
        "classify_message",   #classify_message는 앞의 분기에서 선택됐을 때만 실행
        route_after_classify_message,
        {
            "api": "get_api_graph", #api값이 선택된 경우, get_api_graph 서브그래프 실행.
            "profile": "run_profile",
            "policy": "run_policy",
            "general": "run_general",
        },
    )
    # get_api_graph 실행 후 run_get_api에서 escalate 확인
    graph.add_edge("get_api_graph", "run_get_api")

    # sql_graph/action_graph가 재시도까지 소진하고도 실패(escalate)했으면 classify_message로 되돌아가
    # 1차 분류부터 재시도(최대 MAX_RECLASSIFY회)
    graph.add_conditional_edges(
        "run_get_api",
        route_after_get_api,
        {"reclassify": "classify_message", "done": "store_cache"},
    )
    # 나머지 분기는 항상 store_cache로 수렴
    for node in ("run_profile", "run_policy", "run_general"):
            graph.add_edge(node, "store_cache")

    graph.add_edge("store_cache", END)
    return graph.compile()

    # checkpointer 주입
    # HITL 적용
    # return graph.compile(checkpointer=main_checkpointer)

chat_graph = build_chat_graph()




# 기존 chat.py의 create_chat() 내부 분기(캐시 조회 ~ 응답 생성) 진입 함수
def run_chat_graph(message: str, db, member) -> str:
    # Graph 실행을 시작할 때 StateGraph객체의 invoke함수를 실행하여 초기 State 주입
    thread_id = f"chat-member-{member.id}"
    final_state = chat_graph.invoke(
        {
            "message": message,
            "member_id": member.id,
            "retry_count": 0,
            "reclassify_count": 0,
            "escalate": False,
        },
        # db/member는 checkpoint에 저장되면 안 되므로 state가 아닌 config로 전달
        config={"configurable": {
            # "thread_id": thread_id, # checkpointer가 붙은 그래프는 thread_id 필수
            "db": db, 
            "member": member}},
    )
    return final_state["response"]





# HITL을 위한 함수
def run_chat_graph_hitl(message: str, db, member) -> str:
    # HITL 순서3. 중단여부 확인 및 재개 요청
    # get_state함수를 통해 thread_id로 DB 조회
    # 실행완료되지 않은 checkpoint가 있다면(next가 남아있다면), 사용자의 응답에 따라 멈춘 지점부터 재실행
    thread_id = f"chat-member-{member.id}"
    if chat_graph.get_state({"configurable": {"thread_id": thread_id}}).next:
        # 사용자의 답변에 따라 decision을 approve로 세팅
        decision = "approve" if message.strip() in ("예", "네", "y", "yes", "승인") else "deny"
        print(f"[LangGraph] action 실행 확인 재개 | member_id={member.id} | 답변={message} -> {decision}")
        # 재개 지시
        # 1.checkpoint에 저장된 멈춘 지점(get_api_graph > action_graph > execute_action)으로 직행
        # 2.execute_action의 interrupt()에 decision 문자열 전달
        final_state = chat_graph.invoke(
            Command(resume=decision),
            config={"configurable": {"thread_id": thread_id, "db": db, "member": member}},
        )
    else:
        final_state = chat_graph.invoke(
            {
                "message": message,
                "member_id": member.id,
                "retry_count": 0,
                "reclassify_count": 0,
                "escalate": False,
            },
            config={"configurable": {"thread_id": thread_id, "db": db, "member": member}},
        )

    # HITL 순서2. 사용자에게 확인 문구 전달
    # action 실행 전 확인 대기(interrupt) 상태면, 최종 응답 대신 확인 메시지를 반환
    # 다음 메시지가 들어오면 위 resume 분기에서 이어서 처리
    if final_state.get("__interrupt__"):
        payload = final_state["__interrupt__"][0].value
        return (
            f"다음 작업을 진행할까요?\n"
            f"- 작업: {payload.get('action_name')}\n"
            f"- 내용: {payload.get('args')}\n"
            f"(예/아니오로 답해주세요)"
        )
    return final_state["response"]
