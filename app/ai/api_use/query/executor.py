# SQL실행 및 예외처리
from typing import Any
from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.exc import ( OperationalError, ProgrammingError,  DataError, StatementError, )


def execute_sql(
    db: Session,
    sql: str,
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    print(f"[SQL 실행] 쿼리: {sql}")

    try:
        result = db.execute(text(sql), params)

        columns = list(result.keys())
        rows = result.fetchall()

        # [{"컬럼명": 값, ...}, {...} ...] 형태로 변환
        result_list = [
            {col: _serialize_value(val) for col, val in zip(columns, row)}
            for row in rows
        ]
        return result_list

    except ProgrammingError as e:
        error_msg = f"SQL 문법 또는 스키마 오류: {str(e.orig or e)}"
        print(f"[SQL 실행] ProgrammingError: {error_msg}")
        raise Exception(error_msg) from e

    except DataError as e:
        error_msg = f"데이터 타입 오류: {str(e.orig or e)}"
        print(f"[SQL 실행] DataError: {error_msg}")
        raise Exception(error_msg) from e

    except OperationalError as e:
        error_msg = f"데이터베이스 운영 오류: {str(e.orig or e)}"
        print(f"[SQL 실행] OperationalError: {error_msg}")
        raise Exception(error_msg) from e

    except StatementError as e:
        error_msg = f"파라미터 바인딩 오류: {str(e)}"
        print(f"[SQL 실행] StatementError: {error_msg}")
        raise Exception(error_msg) from e

    except Exception as e:
        error_msg = f"예상치 못한 오류: {str(e)}"
        print(f"[SQL 실행] 알 수 없는 오류: {error_msg}")
        raise Exception(error_msg) from e

# DB 결과값과 타입을 JSON 직렬화 가능한 형태로 변환
def _serialize_value(value: Any) -> Any:
    from datetime import datetime, date
    from decimal import Decimal
    if value is None:
        return None
    if isinstance(value, Decimal):
        # Decimal → float (JSON 직렬화를 위해)
        return float(value)
    if isinstance(value, (datetime, date)):
        # datetime → ISO 8601 문자열
        return value.isoformat()
    return value
