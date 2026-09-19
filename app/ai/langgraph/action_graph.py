# Action(주문/상품 등록·수정 등 DB를 변경하는 API 호출) 파이프라인의 LangGraph 버전.
#
# 기존 action_pipeline.py는 "카테고리 분류 -> action(tool) 선택 -> 즉시 실행"까지 한 번에 처리합니다.
# 하지만 place_order/cancel_order/register_product/update_product는 전부 DB를 변경(insert/update)하는
# 되돌리기 어려운 작업이라, 실행 전에 "이대로 진행할지" 사용자에게 먼저 확인받는 시나리오가 필요합니다.
# 이런 확인(Human-in-the-loop) 단계는 LangGraph의 interrupt()/Command(resume=...)로 표현합니다.
# 그래프 실행을 그 자리에서 멈추고 질문을 반환한 뒤, 사용자의 다음 메시지("네"/"아니오")로 이어서 실행합니다.
#
#   classify_category ─(카테고리 없음)─▶ cannot_process ─▶ END
#         │(카테고리 있음)
#         ▼
#   select_action ─(tool 선택 실패: 정보 부족)─▶ END (안내 메시지)
#         │(tool 선택됨)
#         ▼
#   confirm  ── interrupt("정말 진행할까요?") ── 여기서 1턴 정지, 사용자 응답 대기 ──
#         │
#   route_after_confirm
#     ┌───┴────┐
#  (승인)     (거절/그 외 응답 = 안전 기본값)
#     ▼           ▼
# execute_action  cancelled
#     └────▶ END ◀┘
#
# ⚠ db 세션 관련 주의사항
# FastAPI의 db: Session은 요청(request) 단위로 열리고 요청이 끝나면 닫힙니다. confirm 노드에서
# interrupt로 멈춰 있는 동안(=사용자가 다음 메시지를 보내기 전까지)은 이미 그 요청이 끝나
# db 세션도 닫힌 상태이므로 그대로 재사용하면 안 됩니다. 그래서 재개(resume) 시에는
# `Command(resume=답변, update={"db": 새_요청의_db_session})` 형태로 "새 요청의 db 세션"을
# 함께 주입해서 이어가도록 구성했습니다. (아래 resume_action_pipeline 참고)
#
# ⚠ thread_id / checkpointer 관련 주의사항
# - thread_id: 그래프의 정지 상태를 구분하는 키입니다. 이 프로젝트에는 아직 "대화 세션" 개념이 없어
#   데모 목적으로 f"member:{member_id}"를 사용했습니다. 회원이 여러 대화를 동시에 진행할 수 있다면
#   실제 적용 시 세션/스레드 단위 id로 교체해야 합니다.
# - checkpointer: 데모이므로 프로세스 메모리에만 저장되는 InMemorySaver를 사용했습니다. 서버가
#   재시작되면 확인 대기 중이던 상태는 모두 사라집니다. 실제 적용 시에는 이미 사용 중인 Postgres에
#   연결하는 langgraph-checkpoint-postgres(PostgresSaver)로 교체하는 것을 권장합니다.
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command

# action_pipeline.py의 카테고리 분류 함수(_classify_category)와 action 선택용 LLM 인스턴스(_llm)를
# 그대로 재사용합니다 (로직 복제 방지). 두 이름 모두 모듈 내부용(_ prefix)이라 공개 API는 아니지만,
# 같은 프로젝트 내에서 재사용 목적으로 import합니다.
from app.ai.api_use.action.action_pipeline import _llm, _classify_category
from app.ai.api_use.action.registry import get_action_by_category, execute_action
from app.ai.langgraph.state import ChatGraphState

# 승인으로 간주할 사용자 응답. 그 외 모든 응답은 거절/취소로 처리합니다 (안전을 위한 기본값=거절).
_CONFIRM_YES = {"네", "예", "응", "그래", "진행", "y", "yes", "ok", "확인"}

_SELECT_ACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """당신은 사용자의 요청을 분석하여 적절한 함수를 호출하는 도우미입니다.
                사용자의 요청에서 필요한 정보를 추출하여 함수를 호출하세요.
                함수 호출에 필요한 파라미터가 불명확한 경우, 함수를 호출하지 말고 어떤 정보가 필요한지 안내하세요."""),
    ("user", "{user_message}"),
])


def classify_category_node(state: ChatGraphState) -> dict:
    category = _classify_category(state["message"])
    return {"category": category}


def select_action_node(state: ChatGraphState) -> dict:
    action_tools = get_action_by_category(state["category"])
    llm_with_actions = _llm.bind_tools(action_tools)
    chain = _SELECT_ACTION_PROMPT | llm_with_actions
    response = chain.invoke({"user_message": state["message"]})

    if not response.tool_calls:
        print("[LangGraph][Action] action 미선택 → LLM 직접 응답 반환")
        return {
            "selected_action_name": None,
            "response": response.content or "요청을 처리하려면 더 구체적인 정보가 필요합니다.",
        }

    action_call = response.tool_calls[0]
    print(f"[LangGraph][Action] action 선택 | action={action_call['name']} | args={action_call['args']}")
    return {
        "selected_action_name": action_call["name"],
        "selected_action_args": action_call["args"],
    }


def confirm_node(state: ChatGraphState) -> dict:
    args_text = ", ".join(f"{k}={v}" for k, v in state["selected_action_args"].items())
    question = (
        f"다음 요청을 실행합니다.\n"
        f"- 작업: {state['selected_action_name']}\n"
        f"- 내용: {args_text}\n\n"
        f"진행할까요? ('네' 입력 시 실행, 그 외 입력 시 취소됩니다)"
    )
    # 여기서 그래프가 일시정지되고 question이 그대로 사용자에게 응답으로 전달됨.
    # 사용자가 다음 메시지를 보내면 그 값이 answer로 반환되며 그래프가 이어서 실행됨.
    answer = interrupt({
        "question": question,
        "action_name": state["selected_action_name"],
        "args": state["selected_action_args"],
    })
    return {"confirmation_answer": str(answer)}


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


def cancelled_node(state: ChatGraphState) -> dict:
    return {"response": "요청이 취소되었습니다."}


def cannot_process_node(state: ChatGraphState) -> dict:
    return {"response": "처리할 수 없는 요청입니다."}


def route_after_classify(state: ChatGraphState) -> str:
    return "select" if state.get("category") else "cannot_process"


def route_after_select(state: ChatGraphState) -> str:
    return "confirm" if state.get("selected_action_name") else "no_action"


def route_after_confirm(state: ChatGraphState) -> str:
    answer = (state.get("confirmation_answer") or "").strip().lower()
    return "approved" if answer in _CONFIRM_YES else "rejected"


def build_api_graph():
    graph = StateGraph(ChatGraphState)

    graph.add_node("classify_category", classify_category_node)
    graph.add_node("select_action", select_action_node)
    graph.add_node("confirm", confirm_node)
    graph.add_node("execute_action", execute_action_node)
    graph.add_node("cancelled", cancelled_node)
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
        {"confirm": "confirm", "no_action": END},
    )
    graph.add_conditional_edges(
        "confirm",
        route_after_confirm,
        {"approved": "execute_action", "rejected": "cancelled"},
    )
    graph.add_edge("execute_action", END)
    graph.add_edge("cancelled", END)
    graph.add_edge("cannot_process", END)

    # interrupt()를 사용하는 그래프는 checkpointer가 반드시 필요함 (없으면 실행 시 에러)
    return graph.compile(checkpointer=InMemorySaver())


api_graph = build_api_graph()


def _thread_config(member_id: int) -> dict:
    # TODO(적용 시): 회원별 단일 스레드 대신 대화 세션 단위 id로 교체 권장
    return {"configurable": {"thread_id": f"member:{member_id}"}}


def has_pending_confirmation(member_id: int) -> bool:
    """이 회원에 대해 확인 대기 중인 action이 있는지 여부.
    호출 측(라우터)에서 새 메시지가 들어왔을 때 start_action_pipeline과
    resume_action_pipeline 중 무엇을 호출할지 판단하는 데 사용."""
    snapshot = api_graph.get_state(_thread_config(member_id))
    return bool(snapshot.next)


def _extract_response(result: dict) -> str:
    if "__interrupt__" in result:
        return result["__interrupt__"][0].value["question"]
    return result["response"]


def start_action_pipeline(user_message: str, db, member_id: int) -> str:
    """기존 call_action_pipeline()과 동일한 시그니처의 대체 함수.
    확인이 필요한 action이면 바로 실행하지 않고 확인 질문을 반환한다
    (실제 실행은 사용자가 승인한 뒤 resume_action_pipeline 호출 시점에 일어남)."""
    result = api_graph.invoke(
        {"message": user_message, "db": db, "member_id": member_id},
        config=_thread_config(member_id),
    )
    return _extract_response(result)


def resume_action_pipeline(user_answer: str, db, member_id: int) -> str:
    """confirm 단계에서 정지된 그래프를 사용자의 답변으로 재개.
    db는 반드시 '현재 요청'에서 새로 얻은 세션으로 주입한다 (정지 중이던 세션은 이미 닫혀 있음)."""
    result = api_graph.invoke(
        Command(resume=user_answer, update={"db": db, "member_id": member_id}),
        config=_thread_config(member_id),
    )
    return _extract_response(result)
