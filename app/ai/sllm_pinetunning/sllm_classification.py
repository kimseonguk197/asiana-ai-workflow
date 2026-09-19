from langchain_ollama import ChatOllama
from app.ai.llm_use.classification_list import TOOLS

llm_classify = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

VALID_LABELS = ["get_my_orders", "get_my_profile", "get_policy"]

def classify_message_sllm(message: str) -> str:
    prompt = (
        "아래 질문을 읽고 반드시 다음 세 가지 중 하나만 출력해.\n"
        "- get_my_orders\n"
        "- get_my_profile\n"
        "- get_policy\n\n"
        f"질문: {message}\n"
        "출력:"
    )
    response = llm_classify.invoke(prompt)
    raw = response.content.strip()
    print(raw)
    # 후처리: 유효한 레이블 포함 여부 확인
    for label in VALID_LABELS:
        if label in raw:
            return label
    return "응답불가" 