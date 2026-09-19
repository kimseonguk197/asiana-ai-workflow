from sqlalchemy.orm import Session

from app import models


def list_my_chats(db: Session, member_id: int, limit: int = 5):
    return (
        db.query(models.Chat)
        .filter(models.Chat.member_id == member_id, models.Chat.del_yn == 'N')
        .order_by(models.Chat.created_at.desc())
        .limit(limit)
        .all()
    )
