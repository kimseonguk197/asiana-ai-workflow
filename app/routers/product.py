from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app import models, schemas
from app.dependencies import get_db, get_current_member
from app.services import product_service

router = APIRouter(prefix="/products", tags=["products"])


@router.post("", response_model=schemas.ProductResponse, status_code=status.HTTP_201_CREATED)
def create_product(
    body: schemas.ProductCreate,
    db: Session = Depends(get_db),
    current_member: models.Member = Depends(get_current_member),
):
    product = product_service.register_product(
        db, current_member.id, body.name, body.category, body.price, body.stock
    )
    return product

@router.get("", response_model=List[schemas.ProductResponse])
def list_products(
    name: Optional[str] = None,
    category: Optional[str] = None,
    db: Session = Depends(get_db),
):
    return product_service.list_products(db, name, category)

@router.put("/{product_id}", response_model=schemas.ProductResponse)
def update_product(
    product_id: int,
    body: schemas.ProductUpdate,
    db: Session = Depends(get_db),
    current_member: models.Member = Depends(get_current_member),
):
    
    product = product_service.update_product(
        db, current_member.id, product_id, **body.model_dump(exclude_unset=True)
    )
    return product

