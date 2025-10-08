from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, DateTime, Text, Integer, ForeignKey, func

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    role: Mapped[str] = mapped_column(String(64))
    region: Mapped[str] = mapped_column(String(64))
    start_date: Mapped[DateTime | None]
    created_at: Mapped[DateTime] = mapped_column(server_default=func.now())

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    started_at: Mapped[DateTime] = mapped_column(server_default=func.now())

class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"))
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text())
    created_at: Mapped[DateTime] = mapped_column(server_default=func.now())

class PolicyChunk(Base):
    __tablename__ = "policy_chunks"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    policy_id: Mapped[str] = mapped_column(String(128))
    section: Mapped[str] = mapped_column(String(128))
    text: Mapped[str] = mapped_column(Text())
    region: Mapped[str] = mapped_column(String(64))
    effective_from: Mapped[DateTime | None]
    sunset_on: Mapped[DateTime | None]
    source_uri: Mapped[str] = mapped_column(String(512))
