INTENT_LIST = [
    {
        "type": "function",
        "function": {
            "name": "QUERY",
            "description": "데이터 조회/통계/검색 요청 예) 주문/상품 목록조회",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ACTION",
            "description": "데이터 생성/변경/삭제 요청 예)상품 주문해줘, 새 상품 등록해줘, 주문 취소해줘, 재고 수정해줘",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "GENERAL",
            "description": "그 외 일반 대화 (인사, 시스템 사용법, 기타) 예) 안녕하세요, 이 앱은 뭐야?, 도움말",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
