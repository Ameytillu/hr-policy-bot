from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.core.config import settings

_engine = create_engine(settings.DATABASE_URL or "sqlite:///smarthr_dev.db", echo=False, future=True)
SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)

def get_session():
    return SessionLocal()
