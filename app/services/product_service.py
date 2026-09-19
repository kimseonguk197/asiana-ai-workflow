from typing import Optional

from sqlalchemy.orm import Session

from app import models


def list_products(db: Session, name: Optional[str] = None, category: Optional[str] = None):
    query = db.query(models.Product)
    if name:
        query = query.filter(models.Product.name.ilike(f"%{name}%"))
    if category:
        query = query.filter(models.Product.category.ilike(f"%{category}%"))
    return query.all()


def register_product(db: Session, member_id: int, name: str, category: str, price: float, stock: int):
    product = models.Product(name=name, category=category, price=price, stock=stock, member_id=member_id)
    db.add(product)
    db.commit()
    db.refresh(product)

    print(f"[register_product] 완료 | 회원={member_id}, 상품={product.name}")
    return product


def update_product(db: Session, member_id: int, product_id: int, **fields):
    product = db.query(models.Product).filter(models.Product.id == product_id).first()
    if not product:
        raise ValueError(f"상품 ID {product_id}를 찾을 수 없습니다.")
    if product.member_id != member_id:
        raise ValueError("본인이 등록한 상품만 수정할 수 있습니다.")

    for field, value in fields.items():
        setattr(product, field, value)

    db.commit()
    db.refresh(product)

    print(f"[update_product] 완료 | 회원={member_id}, 상품={product_id}")
    return product
