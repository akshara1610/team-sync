from sqlalchemy import create_engine, Column, String, DateTime, JSON, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from src.config import settings
from src.models.schemas import MeetingStatus

Base = declarative_base()


class Meeting(Base):
    """SQLAlchemy model for meetings."""
    __tablename__ = "meetings"

    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    status = Column(SQLEnum(MeetingStatus), default=MeetingStatus.SCHEDULED)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=True)
    transcript_path = Column(String, nullable=True)
    summary_path = Column(String, nullable=True)
    participants = Column(JSON, default=[])
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Database engine and session
engine = create_engine(settings.DATABASE_URL, echo=settings.DEBUG)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for getting database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
