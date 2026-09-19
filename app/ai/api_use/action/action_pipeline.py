
import os
from sqlalchemy.orm import Session
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.ai.api_use.action.registry import get_category_descriptions, get_action_list_by_category, get_action_name, execute_action, ACTION_CATEGORIES


# 사용자 메시지로부터 적절한 action(api)을 선택하고 실행
def call_action_pipeline(user_message: str, db: Session, member_id: int) -> str:
    # 1.함수 카테고리 분류(order, product 등)
    category = _classify_category(user_message)
    if category is None:
        return "처리할 수 없는 요청입니다."

    action_list = get_action_list_by_category(category)
    
    # 2.카테고리 내 action 중 사용자 요청에 맞는 action 선택 (파라미터 추출 포함)
    action_name, args = get_action_name(user_message, action_list)
    if action_name is None:
        # 파라미터 부족 등으로 LLM이 action을 선택하지 못한 경우, args에 담긴 안내 메시지를 그대로 반환
        return args

    try:
        return execute_action(action_name, args, db, member_id)
    except Exception as e:
        print(f"[action_pipeline] 실행 실패 | action={action_name} | error={e}")
        return f"요청 처리 중 오류가 발생했습니다: {str(e)}"


# 1단계: 카테고리 분류용 LLM
# 파라미터 추출은 창의성 불필요하므로, temperature=0. 카테고리 이름만 출력하면 되므로 max_tokens 최소화
_llm_classify = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
    max_tokens=20,
)

# 카테고리만 분류 : 카테고리 이름과 설명을 LLM에 전달
def _classify_category(user_message: str) -> str:

    prompt = ChatPromptTemplate.from_messages([
    ("system",
        """사용자의 요청이 아래 카테고리 중 어디에 해당하는지 분류하세요.
        {category_descriptions}
        반드시 카테고리 이름만 출력하세요.""",),
    ("user", "{user_message}"),
    ])
    chain = prompt | _llm_classify | StrOutputParser()
    result = chain.invoke({
        # 카테고리 : order, product 등
        "category_descriptions": get_category_descriptions(),
        "user_message": user_message,
    })
    category = result.strip().lower()
    if category not in ACTION_CATEGORIES:
        print(f"[action_pipeline] 알 수 없는 카테고리: '{category}' → 전체 fallback")
        return None
    print(f"[action_pipeline] 카테고리 분류: {category}")
    return category
