# SQL 생성
import os
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

from app.ai.rag.vector_store import vector_store
from app.ai.api_use.query.schema_context import (
    get_schema_context,
    get_schema_context_by_tables,
    TABLE_DESCRIPTIONS,
    ALLOWED_TABLES,
    PERSONAL_TABLES,
)

#  LLM 인스턴스 (SQL 생성 전용)
#  temperature=0: 재현 가능한 결정론적 SQL 생성
llm_sql = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,          # SQL은 창의성 불필요 → 0 고정
    max_tokens=1000,        # SQL이 지나치게 길어지는 것을 방지
)


_SQL_RULES = (
    "## 반드시 지켜야 할 규칙\n"
    "1. SELECT 문만 생성하세요.\n"
    f"2. 개인 데이터({', '.join(sorted(PERSONAL_TABLES))}) 조회 시 WHERE member_id = :current_member_id 포함.\n"
    "3. 집계 쿼리가 아닌 경우 LIMIT 100 포함.\n"
    "4. SQL 코드만 출력하세요. 설명이나 마크다운 블록 없이.\n"
    "5. 모든 테이블 조회 시 del_yn = 'N' 조건을 반드시 포함하세요."
)

#  SQL 생성 프롬프트 템플릿
SQL_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
(
    "system",
    "사용자의 자연어 질문을 PostgreSQL SELECT 문으로 변환하세요.\n\n"
    "## 데이터베이스 스키마\n{schema}\n\n"
    + _SQL_RULES + "\n\n"
    "## 참고 예시\n{examples}\n\n"
    "이제 아래 질문에 대한 SQL을 생성하세요.",
),
("user", "{user_message}"),
])

#  SQL 수정 프롬프트 (자가 교정 / Self-Correction) : LLM이 잘못된 SQL을 생성했을 때, 오류 메시지를 다시 LLM에 전달해 스스로 수정하도록 요청하는 기법
SQL_FIX_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "아래 SQL 쿼리를 실행했을 때 오류가 발생했습니다. 오류를 분석하고 올바른 SQL로 수정하세요.\n\n"
        "## 데이터베이스 스키마\n{schema}\n\n"
        + _SQL_RULES + "\n\n"
        "## 원본 SQL\n{original_sql}\n\n"
        "## 발생한 오류 메시지\n{error_message}\n\n"
        "위 오류를 수정한 올바른 SQL을 출력하세요.",
    ),
    ("user", "수정된 SQL을 생성해주세요."),
])

from langchain_ollama import ChatOllama

llm_classify = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

# 자연어 질문을 PostgreSQL SELECT 문으로 변환
def generate_sql(user_message: str) -> str:
    schema = get_schema_context()
    # relevant_tables = _select_relevant_tables(user_message)
    # relevant_tables = _select_vector(user_message)
    # schema = get_schema_context_by_tables(relevant_tables)
    chain = SQL_GENERATION_PROMPT | llm_sql
    
    response = chain.invoke({
        "schema": schema,
        "examples": FEW_SHOT_EXAMPLES,
        "user_message": user_message,
    })
    print("입력 토큰사용량:", response.usage_metadata["input_tokens"])
    print("출력 토큰사용량:", response.usage_metadata["output_tokens"])
    print("전체 토큰사용량:", response.usage_metadata["total_tokens"])
    print("토큰 상세:", response.usage_metadata.get("input_token_details"))

    return response.content

# 사용자 질문에서 관련 테이블을 LLM으로 선택
def _select_relevant_tables(user_message: str) -> list[str]:
    desc_text = "\n".join(
        f"- {name}: {desc}" for name, desc in TABLE_DESCRIPTIONS.items()
    )
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "아래 테이블 목록에서 사용자 질문과 관련된 테이블을 모두 선택하세요.\n\n"
            "[테이블 목록]\n{table_descriptions}\n\n"
            "관련 테이블명을 콤마로 구분하여 출력하세요. 예: orders, products",
        ),
        ("user", "{user_message}"),
    ])
    chain = prompt | llm_sql | StrOutputParser()
    result = chain.invoke({
        "table_descriptions": desc_text, 
        "user_message": user_message})
    selected = [t.strip().lower() for t in result.split(",") if t.strip().lower() in ALLOWED_TABLES]
    print(f"[테이블 선택] 질문: {user_message[:40]}... → {selected}")
    return selected

def _select_vector(user_message: str) -> list[str]:
    results = vector_store.similarity_search_with_relevance_scores(user_message, k=top_k)
    for doc, score in results:
        print(f"{score:.4f} | {doc.page_content}")
    selected = [table for doc, _ in results if (table := doc.page_content.split(":")[0]) in ALLOWED_TABLES]
    print(f"[벡터 테이블 선택] 질문: {user_message[:40]}... → {selected}")
    return selected


# 실행 실패한 SQL을 LLM을 통해 자가 교정(Self-Correction)
def fix_sql(original_sql: str, error_message: str) -> str:
    # schema = get_schema_context()
    relevant_tables = _tables_from_sql(original_sql)
    schema = get_schema_context_by_tables(relevant_tables)
    chain = SQL_FIX_PROMPT | llm_sql | StrOutputParser()
    fixed_sql = chain.invoke({
        "schema": schema,
        "original_sql": original_sql,
        "error_message": error_message,
    })

    print(f"[SQL 수정] 원본 오류: {error_message[:80]}... → 수정 SQL: {fixed_sql[:100]}...")
    return fixed_sql


# 이미 한번 만들어진 SQL에서 테이블명을 추출 :  from 하고 join 뒤에 테이블명을 추출
def _tables_from_sql(sql: str) -> list[str]:
    pattern = re.compile(r'\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)', re.IGNORECASE)
    found = {t.lower() for t in pattern.findall(sql)}
    return list(found & ALLOWED_TABLES)




#  Few-shot 예시 (SQL 생성 정확도를 높이는 핵심 기법)
#    1. del_yn, 정렬, limit 등 기본 조회 예시
#    2. join, 집계함수 등 복잡한 쿼리 예시
FEW_SHOT_EXAMPLES = """
[예시 1 - 단순 조건 조회]
질문: 나이가 40살 이상인 회원 목록을 알려줘
SQL:
SELECT id, email, name, age, created_at
FROM members
WHERE age >= 40
  AND del_yn = 'N'
ORDER BY age DESC
LIMIT 100;

[예시 2 - 개인 데이터 조회 (:current_member_id 필수)]
질문: 내가 주문한 상품 목록을 보여줘
SQL:
SELECT o.id, p.name AS product_name, p.category, p.price, o.quantity,
       (p.price * o.quantity) AS total_price, o.created_at
FROM orders o
JOIN products p ON o.product_id = p.id
WHERE o.member_id = :current_member_id
  AND o.del_yn = 'N'
  AND p.del_yn = 'N'
ORDER BY o.created_at DESC
LIMIT 100;

[예시 3 - 집계 함수 + 날짜 필터 + 개인 데이터]
질문: 내가 최근 1달 동안 주문한 총금액이 얼마야?
SQL:
SELECT ROUND(SUM(p.price * o.quantity)) AS total_amount
FROM orders o
JOIN products p ON o.product_id = p.id
WHERE o.member_id = :current_member_id
  AND o.del_yn = 'N'
  AND p.del_yn = 'N'
  AND o.created_at >= NOW() - INTERVAL '1 month';
"""