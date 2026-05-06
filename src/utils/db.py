"""
db.py
─────
SQLite database helpers via SQLAlchemy.

Tables
------
students        — student_id, name, roll_no, enrolled_at
attendance      — id, student_id, timestamp, confidence, crop_path
crop_buffer     — id, student_id, crop_path, verified, created_at
"""

from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String,
    create_engine, text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.utils.config_loader import cfg

_DB_PATH = cfg["database"]["path"]
Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{_DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass


class Student(Base):
    __tablename__ = "students"
    student_id  = Column(String, primary_key=True)
    name        = Column(String, nullable=False)
    roll_no     = Column(String, unique=True, nullable=False)
    enrolled_at = Column(DateTime, default=datetime.utcnow)


class AttendanceRecord(Base):
    __tablename__ = "attendance"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    student_id  = Column(String, nullable=False)
    timestamp   = Column(DateTime, default=datetime.utcnow)
    confidence  = Column(Float)
    crop_path   = Column(String)


class CropBuffer(Base):
    __tablename__ = "crop_buffer"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    student_id  = Column(String, nullable=False)
    crop_path   = Column(String, nullable=False)
    verified    = Column(Boolean, default=False)
    created_at  = Column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    """Create all tables (idempotent)."""
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return SessionLocal()


# ── Convenience helpers ──────────────────────────────────────────────

def add_student(student_id: str, name: str, roll_no: str) -> None:
    with get_session() as s:
        s.add(Student(student_id=student_id, name=name, roll_no=roll_no))
        s.commit()


def log_attendance(student_id: str, confidence: float, crop_path: str = "") -> None:
    with get_session() as s:
        s.add(AttendanceRecord(
            student_id=student_id,
            confidence=confidence,
            crop_path=crop_path,
        ))
        s.commit()


def add_crop_to_buffer(student_id: str, crop_path: str, verified: bool = False) -> int:
    with get_session() as s:
        rec = CropBuffer(student_id=student_id, crop_path=crop_path, verified=verified)
        s.add(rec)
        s.commit()
        return rec.id


def mark_crop_verified(crop_id: int) -> None:
    with get_session() as s:
        s.execute(
            text("UPDATE crop_buffer SET verified=1 WHERE id=:id"),
            {"id": crop_id},
        )
        s.commit()


def count_verified_crops(student_id: str) -> int:
    with get_session() as s:
        return s.query(CropBuffer).filter_by(
            student_id=student_id, verified=True
        ).count()


if __name__ == "__main__":
    init_db()
    print("Database initialised at", _DB_PATH)
