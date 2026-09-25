# 사용자 메시지의 의도를 3가지로 분류하여 각 파이프라인으로 라우팅
import os
from sqlalchemy.orm import Session
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .intent_list import INTENT_LIST

from app.ai.api_use.query.sql_pipeline import call_sql_pipeline
from app.ai.api_use.action.action_pipeline import call_action_pipeline
from app.ai.llm_use.llm_calling_langchain import generate_general_response


# 의도 분류 후 QUERY/ACTION/GENERAL 파이프라인으로 라우팅 (chat.py의 get_api 분기에서도 재사용)
def process_api_request(message: str, db: Session, member_id: int) -> str:
    intent = classify_intent(message)
    print(f"[파이프라인] 의도 분류 결과: {intent}")

    # 1)기존 조회 함수 호출 작업(목록/내역 조회 등)
    if intent == "QUERY":
        return call_sql_pipeline(message, db, member_id)
    # 2)기존API활용 작업(insert, update 등)
    elif intent == "ACTION":
        return call_action_pipeline(message, db, member_id)
    # 3)DB 작업 없는 일반 LLM응답
    else:
        return generate_general_response(message)

#  의도 분류 LLM. 셋 중 하나의 도구를 반드시 호출하도록 강제.
_llm_classify = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,  # 분류 목적의 낮은 temperature값 설정.
).bind_tools(INTENT_LIST, tool_choice="required")

_classify_prompt = ChatPromptTemplate.from_messages([
    ("system", """사용자의 메시지를 아래 3가지 중 하나로 분류하세요."""),
    ("user", "{user_message}"),
])

# 사용자 메시지 의도 분류: "QUERY" | "ACTION" | "GENERAL"
def classify_intent(user_message: str) -> str:
    chain = _classify_prompt | _llm_classify
    response = chain.invoke({"user_message": user_message})

    if response.tool_calls:
        intent = response.tool_calls[0]["name"].strip().upper()
        if intent in ("QUERY", "ACTION", "GENERAL"):
            return intent

    # 예외: 예상치 못한 출력은 안전하게 QUERY로 처리
    print(f"[의도 분류] 예상치 못한 응답: '{response}' → QUERY로 fallback")
    return "QUERY"



