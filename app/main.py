from dotenv import load_dotenv
load_dotenv()  # 모든 모듈 임포트 전에 .env 로드

from fastapi import FastAPI
from app.database import engine, Base
from app.routers import member, product, order, chat
from app.routers import document

# LangSmith 연동은 .env에서 설정된 환경변수값(API_KEY 등)만 존재하면 자동으로 읽어 적용

# 재시작시 매번 테이블 재생성을 하려면 drop_all 주석해제
# Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Agent API")

app.include_router(member.router)
app.include_router(product.router)
app.include_router(order.router)
app.include_router(chat.router)
app.include_router(document.router)
