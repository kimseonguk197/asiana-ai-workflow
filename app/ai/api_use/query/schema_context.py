"""
스키마 컨텍스트 (Schema Context) : LLM 프롬프트에 DB 구조를 주입해 LLM이 올바른 SQL을 생성할 수 있도록 돕는 기법
기본 제공 방식
    - 방법1 : DDL 방식(CREATE TABLE 문 그대로 제공)
    - 방법2 : 자연어 방식(테이블 구조를 자연어로 설명)
    - 방법3 : 하이브리드 방식( DDL + 자연어 설명 혼합) -> 우리 수업의 방식
"""

#  허용된 테이블 화이트리스트 : SQL 검증 단계에서 이 목록에 없는 테이블 접근을 차단
ALLOWED_TABLES = frozenset({"members", "products", "orders", "chats"})

#  RLS 대상 테이블 : jwt토큰의 member_id 필터가 자동 적용
PERSONAL_TABLES = frozenset({"orders", "chats", "members"})

#  테이블 자연어 설명
TABLE_DESCRIPTIONS: dict[str, str] = {
    "members":  "서비스를 이용하는 사용자 정보를 저장하는 회원 테이블",
    "products": "판매 중인 상품 목록 테이블",
    "orders":   "회원의 상품 구매 이력을 저장하는 주문 테이블",
    "chats":    "챗봇과의 대화 내용을 저장하는 채팅 이력 테이블",
}

#  테이블별 DDL 
#  민감 컬럼(password 등)은 스키마 제공에서 제외
_TABLE_DDL: dict[str, str] = {
    "members": """\
CREATE TABLE members (
    id          INTEGER     PRIMARY KEY,
    email       VARCHAR     NOT NULL,       -- 이메일 (로그인 ID)
    name        VARCHAR,                    -- 회원 이름
    age         INTEGER,                    -- 나이
    del_yn      VARCHAR(1)  NOT NULL,       -- 삭제 여부 (N: 정상, Y: 삭제)
    created_at  TIMESTAMP                   -- 가입 일시 (UTC)
);""",
    "products": """\
CREATE TABLE products (
    id          INTEGER     PRIMARY KEY,
    name        VARCHAR     NOT NULL,       -- 상품명
    category    VARCHAR     NOT NULL,       -- 카테고리 (예: 전자제품, 의류, 식품)
    price       FLOAT       NOT NULL,       -- 상품 가격 (원 단위)
    stock       INTEGER     NOT NULL,       -- 현재 재고 수량
    member_id   INTEGER,                    -- 상품 등록자 회원 ID (FK → members.id)
    del_yn      VARCHAR(1)  NOT NULL,       -- 삭제 여부 (N: 정상, Y: 삭제)
    created_at  TIMESTAMP                   -- 등록 일시 (UTC)
);""",
    "orders": """\
CREATE TABLE orders (
    id          INTEGER     PRIMARY KEY,
    member_id   INTEGER     NOT NULL,       -- 주문한 회원 ID (FK → members.id)
    product_id  INTEGER     NOT NULL,       -- 주문한 상품 ID (FK → products.id)
    quantity    INTEGER     NOT NULL,       -- 주문 수량
    del_yn      VARCHAR(1)  NOT NULL,       -- 삭제 여부 (N: 정상, Y: 취소)
    created_at  TIMESTAMP                   -- 주문 일시 (UTC)
);""",
    "chats": """\
CREATE TABLE chats (
    id          INTEGER     PRIMARY KEY,
    member_id   INTEGER     NOT NULL,       -- 질의한 회원 ID (FK → members.id)
    request     VARCHAR     NOT NULL,       -- 사용자 질문
    response    VARCHAR     NOT NULL,       -- AI 응답
    del_yn      VARCHAR(1)  NOT NULL,       -- 삭제 여부 (N: 정상, Y: 삭제)
    created_at  TIMESTAMP                   -- 질의 일시 (UTC)
);""",
}

# 테이블 관계도
_RELATIONSHIPS: list[tuple[str, str, str]] = [
    ("members",  "orders",   "- members(id) ↔ orders(member_id)   : 1:N — 한 회원이 여러 주문 가능"),
    ("products", "orders",   "- products(id) ↔ orders(product_id) : 1:N — 한 상품이 여러 주문에 포함 가능"),
    ("members",  "products", "- members(id) ↔ products(member_id) : 1:N — 한 회원이 여러 상품 등록 가능"),
    ("members",  "chats",    "- members(id) ↔ chats(member_id)    : 1:N — 한 회원이 여러 채팅 가능"),
]


# TABLE_DESCRIPTIONS + _TABLE_DDL 을 합쳐 단일 DDL 블록을 생성
def _build_ddl_block(table: str) -> str:
    desc = TABLE_DESCRIPTIONS[table]
    return f"-- [{table}] {desc}\n{_TABLE_DDL[table]}"

# 전체스키마 반환 함수
def get_schema_context() -> str:
    ddl_blocks = "\n\n".join(_build_ddl_block(t) for t in _TABLE_DDL)
    rel_section = "[테이블 관계]\n" + "\n".join(line for _, _, line in _RELATIONSHIPS)
    return ddl_blocks + "\n\n" + rel_section

# 필터링된 스키마 조회 함수 
# relevant_tables과 관계된(FK)테이블의 스키마 반환
def get_schema_context_by_tables(relevant_tables: list[str]) -> str:
    tables: set[str] = {t.lower().strip() for t in relevant_tables} & ALLOWED_TABLES
    if not tables:
        # 입력 테이블이 허용된 테이블에 없으면 전체 스키마를 반환
        return get_schema_context()

    # FK 부모 테이블 자동 포함 (예: orders → members, products 자동 추가)
    for parent, child, _ in _RELATIONSHIPS:
        if child in tables:
            tables.add(parent)

    ddl_blocks = "\n\n".join(_build_ddl_block(t) for t in _TABLE_DDL if t in tables)
    # 부모테이블, 자식테이블이 모두 대상일경우 그 관계도 설명을 포함
    rel_lines  = [desc for parent, child, desc in _RELATIONSHIPS if parent in tables and child in tables]
    rel_section  = "[테이블 관계]\n" + "\n".join(rel_lines) if rel_lines else ""

    return (ddl_blocks + ("\n\n" + rel_section if rel_section else "")).strip()
