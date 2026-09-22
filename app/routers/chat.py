from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app import models, schemas
from app.dependencies import get_db, get_current_member
from app.ai.llm_use.llm_calling_simple import classify_message, generate_response
from app.ai.rag.retriever import search_policy
from app.ai.llm_use.llm_calling_langchain import classify_message_langchain
from app.ai.llm_use.llm_calling_langchain import generate_response_langchain_memory
from app.ai.llm_use.llm_calling_langchain import generate_response_langchain
from app.ai.llm_use.llm_calling_langchain import generate_response_langchain_sllm
from app.ai.llm_use.llm_calling_langchain import generate_general_response
from app.ai.rag.memory import load_chat_history
from app.routers.order import my_orders
from app.routers.member import my_page
# from app.ai.api_use.chat_classify import process_api_request

from app.ai.rag.semantic_cache import semantic_cache
from app.ai.langgraph.main_graph import run_chat_graph
from app.ai.langgraph.main_graph import run_chat_graph_hitl

router = APIRouter(prefix="/chats", tags=["chat"])
@router.post("", response_model=schemas.ChatResponse, status_code=status.HTTP_201_CREATED)
def create_chat(
    body: schemas.ChatRequest,
    db: Session = Depends(get_db),
    current_member: models.Member = Depends(get_current_member),
):

    # [LangGraph 적용시] 캐시 조회부터 if/elif 분기 전체를 run_chat_graph줄로 대체
    response_text = run_chat_graph(body.message, db, current_member)
    # response_text = run_chat_graph_hitl(body.message, db, current_member)
        
    # # 같은질문에 대한 캐싱 : redis stack에 같은 질문이 이력이 있는지 검색
    # cached_response = semantic_cache.search(body.message, current_member.id)

    # # 히트 시: redis에 저장된 값으로 즉시 응답
    # # 미스 시: 아래 else 분기 처리로 진행
    # if cached_response:
    #     response_text = cached_response

    # else:
    #     classification = classify_message(body.message)
    #     # action = classify_message_langchain(body.message)
    #     print(classification)
    #     if classification == "get_api":
    #         response_text = process_api_request(body.message, db, current_member.id)
    #     # 민감정보의 경우 sLLM을 통해 응답생성
    #     elif classification == "get_my_profile":
    #         member = my_page(current_member=current_member)
    #         print(member)
    #         data = _format_profile(member)
    #         print(data)
    #         response_text = generate_response_langchain_sllm(body.message, data)
    #     elif classification == "get_policy":
    #         context = search_policy(body.message)
    #         # response_text = generate_response(body.message, context)
    #         response_text = generate_response_langchain(body.message, context)
    #         # # 최근대화고려 작업(Window Memory): 응답시 최근 5턴 대화 기록을 함께 전달
    #         # history = load_chat_history(current_member.id, db)
    #         # response_text = generate_response_langchain_memory(body.message, context, history)
    #     else:
    #         response_text = generate_general_response(body.message)

    #     # redis stack에 질문/응답을 저장
    #     # store: member_id 포함 (flush_by_member로 사용자별 선택 삭제 가능)
    #     semantic_cache.store(body.message, response_text, current_member.id)


    chat_record = models.Chat(
        member_id=current_member.id,
        request=body.message,
        response=response_text,
    )

    db.add(chat_record)
    db.commit()
    db.refresh(chat_record)
    return chat_record


def _format_orders(orders: list) -> str:
    if not orders:
        return "주문 내역이 없습니다."
    lines = [
        f"- 주문번호: {o.id} / 상품ID: {o.product_id} / 수량: {o.quantity} / 주문일: {o.created_at.strftime('%Y-%m-%d')}"
        for o in orders
    ]
    return "\n".join(lines)


def _format_profile(member: list) -> str:
    if not member:
        return "회원정보가 없습니다."
    return f"- 회원번호: {member.id} / email: {member.email} / 회원명: {member.name} / age: {member.age} "




