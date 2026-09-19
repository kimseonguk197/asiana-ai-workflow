from sqlalchemy.orm import Session

from app import models

def list_my_orders(db: Session, member_id: int):
    return db.query(models.Order).filter(models.Order.member_id == member_id).all()


def place_order(db: Session, member_id: int, product_id: int, quantity: int):
    product = db.query(models.Product).filter(models.Product.id == product_id).first()
    if not product:
        raise ValueError(f"상품 ID {product_id}를 찾을 수 없습니다. 올바른 상품 ID를 확인해 주세요.")
    if product.stock < quantity:
        raise ValueError(
            f"재고가 부족합니다.\n"
            f"- 요청 수량: {quantity}개\n"
            f"- 현재 재고: {product.stock}개"
        )

    product.stock -= quantity
    order = models.Order(member_id=member_id, product_id=product_id, quantity=quantity)
    db.add(order)
    db.commit()
    db.refresh(order)

    print(f"[place_order] 완료 | 회원={member_id}, 상품={product.name}, 수량={quantity}")
    return order, product


def cancel_order(db: Session, member_id: int, order_id: int):
    order = db.query(models.Order).filter(
        models.Order.id == order_id,
        models.Order.del_yn == 'N',
    ).first()
    if not order:
        raise ValueError(f"주문 ID {order_id}를 찾을 수 없거나 이미 취소된 주문입니다.")
    if order.member_id != member_id:
        raise ValueError("본인의 주문만 취소할 수 있습니다.")

    product = db.query(models.Product).filter(models.Product.id == order.product_id).first()
    product_name = product.name if product else f"상품 ID {order.product_id}"
    if product:
        product.stock += order.quantity

    quantity = order.quantity
    order.del_yn = 'Y'
    db.commit()

    print(f"[cancel_order] 완료 | 회원={member_id}, 주문={order_id}")
    return order_id, product_name, quantity
