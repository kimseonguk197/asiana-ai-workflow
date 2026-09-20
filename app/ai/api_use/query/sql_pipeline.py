from sqlalchemy.orm import Session

from app.ai.api_use.query.generator import generate_sql, fix_sql
from app.ai.api_use.query.validator import validate_and_correct
from app.ai.api_use.query.executor import execute_sql
from app.ai.api_use.query.response import format_sql_result, format_error_response


def call_sql_pipeline(
    user_message: str,
    db: Session,
    current_member_id: int,
) -> str:
    current_sql = ""
    last_error = ""
    retry_count = 0
    # 검증(validation) 실패 또는 쿼리실행실패시 재시도(Self-Healing)
    for attempt in range(3):
        # 1.SQL 생성 또는 수정 
        if attempt == 0:
            print(f"[Text-to-SQL] SQL 생성 시도 #{attempt + 1}")
            current_sql = generate_sql(user_message)
        else:
            # 4.이전 오류를 기반으로 SQL 수정 (Self-Correction)
            print(
                f"[Text-to-SQL] SQL 수정 시도 #{attempt + 1} | "
                f"이전 오류: {last_error}"
            )
            current_sql = fix_sql(current_sql, last_error)
            retry_count = attempt

        # 2.SQL 검증 -> 성공시 : SQL실행, 실패시 : SQL 재생성
        validation = validate_and_correct(current_sql)
        if not validation.is_valid:
            last_error = validation.error_message
            continue

        # 3.SQL 실행 -> 성공시 : 응답 반환, 실패시 : SQL 재생성
        try:
            execution_params = {"current_member_id": current_member_id,}
            corrected_sql = validation.corrected_sql
            results = execute_sql(db, corrected_sql, execution_params)

            print(
                f"[Text-to-SQL] 성공 | {len(results)}건 조회 | 재시도={retry_count}회"
            )
            response_text = format_sql_result(user_message, results)
            return response_text

        except Exception as e:
            # SQL 실행 실패 → 오류 메시지를 다음 재시도에 전달
            last_error = str(e)
            print(
                f"[Text-to-SQL] 실행 실패 (시도 #{attempt + 1}): {last_error}"
            )
            continue

    # 모든 재시도 실패시
    return format_error_response(user_message, last_error)
