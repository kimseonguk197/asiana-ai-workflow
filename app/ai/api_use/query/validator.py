#SQL 검증 & 보안 (SQL Validation)
import re
from dataclasses import dataclass
from typing import Optional

from app.ai.api_use.query.schema_context import ALLOWED_TABLES, PERSONAL_TABLES

#  보안: 절대 허용하지 않는 SQL 키워드 블랙리스트
# ─────────────────────────────────────────────────────────────
DANGEROUS_KEYWORDS = frozenset({
    # DML (데이터 조작)
    "INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT",
    # DDL (데이터 정의)
    "DROP", "CREATE", "ALTER", "TRUNCATE", "RENAME",
    # DCL (데이터 제어)
    "GRANT", "REVOKE",
    # 트랜잭션
    "COMMIT", "ROLLBACK", "SAVEPOINT",
    # 시스템 명령 (PostgreSQL 특화)
    "COPY", "EXECUTE",  "PERFORM",  "pg_read_file",   "pg_sleep",   "lo_import",  "lo_export",
})

#  결과 건수 제한
MAX_ROWS = 100


@dataclass
class ValidationResult:
    is_valid: bool # 유효성 판단
    corrected_sql: Optional[str] = None   # 검증 통과 후 수정된 SQL
    error_message: Optional[str] = None   # 검증 실패 시 사유

def validate_and_correct(sql: str) -> ValidationResult:
    print(f"[SQL 검증] 입력 쿼리:\n{sql}")

    # 1. 빈 SQL 체크
    if not sql or not sql.strip():
        return ValidationResult(
            is_valid=False,
            corrected_sql=None,
            error_message="SQL이 비어 있습니다.",
        )

    # 2. SQL 주석 제거 
    #주석 공격 예시)SELECT * FROM member WHERE name = '{user_name}' AND member_id = :current_member_id;
    #user_input에 admin--을 입력하여, AND 이후 부분 무력시도
    sql = _remove_sql_comments(sql).strip()

    #  3.SELECT ONLY 정책
    #  3-1) 첫구문 SELECT 문 여부 확인
    first_token = sql.split()[0].upper()
    if first_token != "SELECT":
        return ValidationResult(
            is_valid=False,
            corrected_sql=None,
            error_message=f"SELECT 문만 허용됩니다. (감지된 키워드: {first_token})",
        )

    # 3-2) 다중 SQL 문 차단 (SQL Stacking 방지)
    #   SQL Stacking 공격 예시: SELECT * FROM members; DROP TABLE members;
    sql_without_final_semicolon = sql.rstrip(";").strip() #문장 맨 오른쪽 끝에 있는 세미콜론(;)과 공백제거
    # 그래도 쿼리에 ;이 남아있다면 다중 SQL문
    if ";" in sql_without_final_semicolon:
        return ValidationResult(
            is_valid=False,
            corrected_sql=None,
            error_message="다중 SQL 문은 허용되지 않습니다.",
        )

    # 3-3)쿼리 내 SELECT 문 아닌 위험 키워드 검사
    sql_upper = sql.upper()
    for keyword in DANGEROUS_KEYWORDS:
        # 특정 문자에 끼워져 있는것이 아닌, 독립된 하나의 단어로 쓰인 위험한 키워드 검출
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, sql_upper):
            print(f"[SQL 검증] 위험 키워드 감지: {keyword}")
            return ValidationResult(
                is_valid=False,
                corrected_sql=None,
                error_message=f"허용되지 않는 SQL 키워드가 포함되어 있습니다: {keyword}",
            )

    # 4. 조회가 허용된 테이블 화이트리스트 검사
    mentioned_tables = _extract_table_names(sql) #쿼리에서 테이블 목록 추출 
    unknown_tables = mentioned_tables - ALLOWED_TABLES #조회 허용 테이블 목록
    if unknown_tables:
        print(f"[SQL 검증] 허용되지 않은 테이블: {unknown_tables}")
        return ValidationResult(
            is_valid=False,
            corrected_sql=None,
            error_message=f"허용되지 않은 테이블 접근: {', '.join(unknown_tables)}",
        )

    # 5. Row-Level Security (RLS) 검사 -> 타인데이터 조회
    #   개인 데이터 테이블(orders, chats)을 사용하는 쿼리에 반드시 :current_member_id 필터가 포함되어야함(_SQL_RULES 참고)
    personal_tables_in_sql = mentioned_tables & PERSONAL_TABLES
    if personal_tables_in_sql:
        if ":current_member_id" not in sql:
            print(f"[SQL 검증] RLS 위반: {personal_tables_in_sql} 테이블에 current_member_id 필터 없음")
            return ValidationResult(
                is_valid=False,
                corrected_sql=None,
                error_message=(
                    f"개인 데이터 테이블({', '.join(personal_tables_in_sql)})을 조회할 때는 ':current_member_id' 필터가 필요합니다."
                ),
            )
        
    # 6. LIMIT 자동 주입
    sql_upper = sql.upper()
    if re.search(r'\bLIMIT\b', sql_upper):
        # 이미 LIMIT이 있으면 건드리지 않음
        corrected_sql = sql
    else:
        # LIMIT 주입: 마지막 세미콜론 앞에 삽입
        sql_stripped = sql.rstrip().rstrip(";").rstrip()
        corrected_sql = f"{sql_stripped}\nLIMIT {MAX_ROWS};"

    print(f"[SQL 검증] 통과: {corrected_sql[:80]}...")
    return ValidationResult(
        is_valid=True,
        corrected_sql=corrected_sql,
        error_message=None,
    )


#  내부 유틸리티 함수들
# SQL에서 주석을 제거
def _remove_sql_comments(sql: str) -> str:
    # 블록 주석 제거: /* ... */ 
    sql = re.sub(r'/\*.*?\*/', ' ', sql, flags=re.DOTALL)
    # 한 줄 주석 제거: -- 이후 줄 끝까지
    sql = re.sub(r'--[^\n]*', ' ', sql)
    # 연속 공백 정리
    sql = re.sub(r'\s+', ' ', sql).strip()
    return sql

#  SQL 문에서 테이블 이름을 추출
def _extract_table_names(sql: str) -> set[str]:
    sql_upper = sql.upper()
    # 정규식 기반으로 FROM 또는 JOIN 다음 테이블명 추출
    table_pattern = re.compile(
        r'\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)',
        re.IGNORECASE,
    )
    found = table_pattern.findall(sql_upper)
    # 다시 소문자로 반환(테이블명은 일반적으로 소문자)
    return {t.lower() for t in found}
