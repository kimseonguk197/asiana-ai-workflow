from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text as sql_text
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.ai.rag.semantic_cache import semantic_cache

from app import schemas
from app.dependencies import get_db
from app.ai.rag.vector_store import vector_store

router = APIRouter(prefix="/documents", tags=["documents"])



@router.post("", response_model=schemas.DocumentResponse, status_code=status.HTTP_201_CREATED)
def add_documents(body: schemas.DocumentCreate, db: Session = Depends(get_db)):
    # 문서단위, pdf 등 문장 전체일경우
    # 입력된 텍스트들을 청크 단위로 분할
    # chunks = splitter.create_documents(body.texts)
    # chunk_texts = [chunk.page_content for chunk in chunks]
    chunk_texts = [chunk for chunk in body.texts]

    # 가장일반적인 문장저장 : vector_store.py의 vector_store를 활용하여 저장
    # add_texts()는 임베딩 생성 + langchain_pg_embedding 테이블에 저장
    vector_store.add_texts(chunk_texts)

    return {"added": len(body.texts)}

# 테스트 데이터)
# {"texts": [
#     "[교환 정책]\n교환 신청의 책임부서는 제품지원팀입니다.\n이 쇼핑몰의 교환 가능 기간은 상품 수령 후 7일 이내입니다.\n\n[환불 정책]\n환불 책임부서는 고객지원팀입니다.\n개봉, 사용흔적, 제품손상이 있는 경우 환불이 제한될 수 있습니다.\n결제 수단 기준으로 3영업일 이내 환불이 처리됩니다.\n\n[배송 정책]\n기본 배송비는 3,000원이며 5만원 이상 구매 시 무료배송입니다.\n전자제품의 경우는 배송비는 5,000원이며 10만원 이상 구매 시 무료배송입니다.\n도서산간 지역은 추가 배송비가 발생할 수 있습니다.\n배송은 결제 완료 후 영업일 기준 1~3일 이내 출고됩니다.\n천재지변 또는 물류사 사정에 따라 배송이 지연될 수 있습니다."
#   ]}
# 문서단위로 입력될 경우 : 청킹 설정
# ["\n\n", "\n", ". ", " ", ""] 이런 내부 매커니즘을 우선순위로 문장 split
splitter = RecursiveCharacterTextSplitter(
    # chunk_size: 청크 당 최대 글자 수
    chunk_size=150,
    chunk_overlap=20,
)



@router.get("", response_model=list[schemas.DocumentChunk])
def list_documents(db: Session = Depends(get_db)):
    # langchain_pg_embedding 테이블에서 id(UUID)와 document(청크 텍스트) 전체 조회
    # Admin UI에서 청크 목록을 확인하고 수정 대상 UUID를 선택하기 위한 용도
    rows = db.execute(
        sql_text("SELECT id::text, document FROM langchain_pg_embedding ORDER BY id")
    ).fetchall()
    return [{"id": row[0], "content": row[1]} for row in rows]


@router.put("/{chunk_id}", response_model=schemas.DocumentChunk)
def upsert_document(chunk_id: str, body: schemas.DocumentChunkUpdate, db: Session = Depends(get_db)):
    # 1. UUID로 기존 청크 존재 여부 확인
    existing = db.execute(
        sql_text("SELECT id FROM langchain_pg_embedding WHERE id = :id"),
        {"id": chunk_id}
    ).fetchone()

    if not existing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="해당 청크를 찾을 수 없습니다.")

    # 2. 기존 청크 삭제 (UUID 기반)
    # PGVector.delete()는 내부적으로 langchain_pg_embedding에서 UUID로 행 삭제
    vector_store.delete(ids=[chunk_id])

    # 3. 수정된 텍스트를 재임베딩 후 새 청크로 저장
    # 새 UUID가 자동 부여됨 → 반환값으로 새 UUID를 확인 가능
    new_ids = vector_store.add_texts([body.text])

    # 4. Redis Semantic Cache flush
    # 청크 내용이 바뀌면 기존 캐시 응답이 outdated 될 수 있으므로 전체 삭제
    semantic_cache.flush()

    return {"id": new_ids[0], "content": body.text}
