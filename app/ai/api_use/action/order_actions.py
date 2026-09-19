
from sqlalchemy.orm import Session

from app import models
from app.services import order_service

#  Action 스키마 (OpenAI function calling 형식)
ACTION_LIST = [
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": "특정 상품을 주문합니다. 사용자가 상품 ID와 수량을 지정하여 주문을 요청할 때 사용합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "description": "주문할 상품의 ID",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "주문할 수량 (1 이상)",
                    },
                },
                "required": ["product_id", "quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "기존 주문을 취소합니다. 취소 시 해당 수량만큼 재고가 복구됩니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "취소할 주문의 ID",
                    },
                },
                "required": ["order_id"],
            },
        },
    },
]


def _cancel_order(args: dict, db: Session, member_id: int) -> str:
    try:
        order_id, product_name, quantity = order_service.cancel_order(
            db, member_id, args["order_id"]
        )
    except ValueError as e:
        return str(e)

    return (
        f"주문이 취소되었습니다.\n"
        f"- 주문번호: {order_id}\n"
        f"- 상품명: {product_name}\n"
        f"- 수량: {quantity}개 (재고 복구 완료)"
    )


def _place_order(args: dict, db: Session, member_id: int) -> str:
    try:
        order, product = order_service.place_order(
            db, member_id, args["product_id"], args["quantity"]
        )
    except ValueError as e:
        return str(e)

    return (
        f"주문이 완료되었습니다!\n"
        f"- 주문번호: {order.id}\n"
        f"- 상품명: {product.name}\n"
        f"- 수량: {order.quantity}개\n"
        f"- 총 금액: {product.price * order.quantity:,.0f}원"
    )


# "action 이름 : handler 함수" 매핑
HANDLERS = {
    "place_order": _place_order,
    "cancel_order": _cancel_order,
}
