
import os
import json
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#  SQL 조회후 응답 생성용 LLM
#  temperature=0.3: 약간의 자연스러움을 허용하되 일관성 유지
llm_format = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.3,
    max_tokens=1500,
)

# SQL 조회후 응답 생성용 LLM
def format_sql_result(
    user_message: str,
    query_results: list[dict[str, Any]],
) -> str:
    # db 조회 결과값이 없다면, 아래 문구로 사용자에게 최종응답
    if not query_results:
        return "조회된 데이터가 없습니다. 검색 조건을 다시 확인해 주세요."

    total_rows = len(query_results)

    # 결과 크기 제한 : 결과가 많을경우 50줄 까지 앞부분만 전달
    truncated = query_results[:50]
    shown_rows = len(truncated)

    # 결과를 JSON 형태로 직렬화하여 프롬프트에 포함
    results_json = json.dumps(truncated, ensure_ascii=False, indent=2)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "사용자의 질문에 대해 데이터베이스 조회 결과를 바탕으로 자연스러운 한국어로 답변하세요.\n"
            "## 답변 규칙\n"
            "1. 결과 데이터의 id값을 포함하여 모든 컬럼을 정확하게 반영해서 답변하세요.\n"
            "2. 결과가 여러 행이면 목록 형태로 정리하세요.\n"
            "3. 결과가 없으면 \"조회된 데이터가 없습니다.\"라고 친절하게 안내하세요.\n"
            "4. 전체 결과가 {total_rows}건이고 {shown_rows}건만 보여주는 경우 그 사실을 언급하세요.\n"
            "## 조회 결과 ({shown_rows}건)\n"
            "{results}",
        ),
        ("user", "{user_message}"),
    ])
    chain = prompt | llm_format | StrOutputParser()
    response = chain.invoke({
        "user_message": user_message,
        "results": results_json,
        "total_rows": total_rows,
        "shown_rows": shown_rows,
    })

    print(f"response: {response}")
    return response


# 모든 재시도 실패 후 사용자 친화적인 오류 메시지를 반환
def format_error_response(user_message: str, error_detail: str) -> str:
    print(f"[포매팅] 오류 응답 생성 | 질문: {user_message[:50]} | 원인: {error_detail}")
    return (
        "죄송합니다. 요청하신 내용을 처리하는 중 문제가 발생했습니다. "
        "질문을 더 구체적으로 다시 입력해 주세요"
    )
