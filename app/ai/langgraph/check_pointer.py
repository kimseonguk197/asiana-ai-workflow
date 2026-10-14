
# checkpointer를 위한 코드
import os
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver

main_checkpointer_connector = ConnectionPool(
    conninfo=os.getenv("DATABASE_URL"),
    max_size=10,
    kwargs={"autocommit": True, "row_factory": dict_row}, #dict_row : 조회 결과를 dict 형태로 반환
)
main_checkpointer = PostgresSaver(main_checkpointer_connector)
main_checkpointer.setup()  # checkpoints 관련 테이블이 없다면 자동 생성

