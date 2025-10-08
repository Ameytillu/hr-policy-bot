from typing import Iterable
from sqlalchemy import select
from src.storage.session import get_session
from src.storage.models import Message

class ChatRepo:
    @staticmethod
    def save_message(session_id: int, role: str, content: str) -> None:
        with get_session() as db:
            db.add(Message(session_id=session_id, role=role, content=content))
            db.commit()

    @staticmethod
    def last_messages(session_id: int, limit: int = 10) -> Iterable[Message]:
        with get_session() as db:
            return db.execute(
                select(Message).where(Message.session_id == session_id).order_by(Message.id.desc()).limit(limit)
            ).scalars().all()
