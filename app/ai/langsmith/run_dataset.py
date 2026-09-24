"""
LangSmith dataset을 이용해 '로컬' 챗봇 파이프라인을 실행/기록
동작 방식:
    1. LangSmith의 dataset 예제(inputs)를 읽어 run_chat_graph_hitl을 실행
    2. 실행 결과/트레이스를 LangSmith로 업로드해서 웹 UI의 Experiment 화면에서 확인

사용법: python -m app.ai.langsmith.run_dataset
"""
from dotenv import load_dotenv

load_dotenv()

from langsmith import Client, evaluate

from app import models
from app.database import SessionLocal
from app.ai.langgraph.main_graph import run_chat_graph_hitl

DATASET_NAME = "test-data"

# 테스트에 쓸 회원 id
TEST_MEMBER_ID = 1


def _get_test_member(db):
    member = db.query(models.Member).filter(models.Member.id == TEST_MEMBER_ID).first()
    if not member:
        raise ValueError(f"TEST_MEMBER_ID={TEST_MEMBER_ID} 인 회원을 찾을 수 없습니다.")
    return member


# LangSmith가 dataset의 각 example.inputs를 이 함수에 그대로 넘겨준다.
# 여기서 실제 로컬 챗봇 파이프라인(run_chat_graph_hitl)을 호출하는 게 이 스크립트의 핵심.
def target(inputs: dict) -> dict:
    db = SessionLocal()
    try:
        member = _get_test_member(db)
        response = run_chat_graph_hitl(inputs["message"], db, member)
        return {"response": response}
    finally:
        db.close()


def main():
    client = Client()  # LANGSMITH_API_KEY를 환경변수에서 자동으로 읽음
    if not client.has_dataset(dataset_name=DATASET_NAME):
        raise ValueError(f"'{DATASET_NAME}' dataset이 LangSmith에 없습니다. DATASET_NAME을 확인하세요.")

    # evaluators 없이 실행 -> 자동 채점은 안 하고, 실제 응답/트레이스/에러만 LangSmith에 기록.
    evaluate(
        target,
        data=DATASET_NAME,
        experiment_prefix="local-dev-run",
    )


if __name__ == "__main__":
    main()
