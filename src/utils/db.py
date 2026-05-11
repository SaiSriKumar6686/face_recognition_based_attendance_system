"""
db.py
─────
SQLite database helpers via SQLAlchemy.

Tables
------
students           — student_id, name, roll_no, enrolled_at, consent, photo
attendance         — id, student_id, timestamp, confidence, crop_path, classroom_id, liveness_score
crop_buffer        — id, student_id, crop_path, verified, created_at
admin_users        — id, username, password_hash, role
audit_log          — id, user, action, target, details, timestamp
classrooms         — id, name, camera_url, created_at
system_health_log  — id, timestamp, endpoint, inference_ms, faces_detected, success
"""

import csv
import io
from datetime import datetime, date, timedelta
from pathlib import Path

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text,
    create_engine, text, func, and_,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from werkzeug.security import generate_password_hash, check_password_hash

from src.utils.config_loader import cfg

_DB_PATH = cfg["database"]["path"]
Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{_DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)

# Dedup window: don't log the same student within this many seconds
_DEDUP_WINDOW_SECONDS = 300  # 5 minutes


class Base(DeclarativeBase):
    pass


# ── Core Models ──────────────────────────────────────────────────────

class Student(Base):
    __tablename__ = "students"
    student_id    = Column(String, primary_key=True)
    name          = Column(String, nullable=False)
    roll_no       = Column(String, unique=True, nullable=False)
    enrolled_at   = Column(DateTime, default=datetime.utcnow)
    consent_given = Column(Boolean, default=False)
    consent_date  = Column(DateTime, nullable=True)
    photo_path    = Column(String, nullable=True)


class AttendanceRecord(Base):
    __tablename__ = "attendance"
    id              = Column(Integer, primary_key=True, autoincrement=True)
    student_id      = Column(String, nullable=False)
    timestamp       = Column(DateTime, default=datetime.utcnow)
    confidence      = Column(Float)
    crop_path       = Column(String)
    classroom_id    = Column(Integer, nullable=True)
    liveness_score  = Column(Float, nullable=True)


class CropBuffer(Base):
    __tablename__ = "crop_buffer"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    student_id  = Column(String, nullable=False)
    crop_path   = Column(String, nullable=False)
    verified    = Column(Boolean, default=False)
    created_at  = Column(DateTime, default=datetime.utcnow)


class AdminUser(Base):
    __tablename__ = "admin_users"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    username      = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role          = Column(String, default="admin")  # admin | viewer

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id        = Column(Integer, primary_key=True, autoincrement=True)
    user      = Column(String, nullable=False)
    action    = Column(String, nullable=False)
    target    = Column(String, nullable=True)
    details   = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)


class Classroom(Base):
    __tablename__ = "classrooms"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    name        = Column(String, nullable=False, unique=True)
    camera_url  = Column(String, nullable=True)
    created_at  = Column(DateTime, default=datetime.utcnow)


class SystemHealthLog(Base):
    __tablename__ = "system_health_log"
    id             = Column(Integer, primary_key=True, autoincrement=True)
    timestamp      = Column(DateTime, default=datetime.utcnow)
    endpoint       = Column(String, nullable=False)
    inference_ms   = Column(Float)
    faces_detected = Column(Integer)
    success        = Column(Boolean, default=True)


# ── Initialization ───────────────────────────────────────────────────

def _migrate_columns():
    """Add missing columns to existing tables (SQLite doesn't auto-add via create_all)."""
    migrations = [
        ("students", "consent_given", "BOOLEAN DEFAULT 0"),
        ("students", "consent_date", "DATETIME"),
        ("students", "photo_path", "VARCHAR"),
        ("attendance", "classroom_id", "INTEGER"),
        ("attendance", "liveness_score", "FLOAT"),
    ]
    with engine.connect() as conn:
        for table, col, col_type in migrations:
            try:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"))
                conn.commit()
            except Exception:
                # Column already exists — ignore
                pass


def init_db() -> None:
    """Create all tables (idempotent) and ensure default admin exists."""
    Base.metadata.create_all(engine)
    _migrate_columns()
    _ensure_default_admin()


def _ensure_default_admin():
    """Create or reset default admin account.

    The password hash is always regenerated on startup to ensure
    compatibility when the DB is moved between different Python/Werkzeug
    versions (e.g. local dev → Docker container).
    """
    with get_session() as s:
        existing = s.query(AdminUser).filter_by(username="admin").first()
        if not existing:
            admin = AdminUser(username="admin", role="admin")
            admin.set_password("admin123")
            s.add(admin)
        else:
            # Re-hash to match the current Werkzeug version
            existing.set_password("admin123")
        s.commit()


def get_session() -> Session:
    return SessionLocal()


# ── Core helpers ─────────────────────────────────────────────────────

def add_student(student_id: str, name: str, roll_no: str, photo_path: str = None) -> None:
    with get_session() as s:
        s.add(Student(student_id=student_id, name=name, roll_no=roll_no, photo_path=photo_path))
        s.commit()


def log_attendance(student_id: str, confidence: float, crop_path: str = "",
                   classroom_id: int = None, liveness_score: float = None) -> None:
    with get_session() as s:
        s.add(AttendanceRecord(
            student_id=student_id,
            confidence=confidence,
            crop_path=crop_path,
            classroom_id=classroom_id,
            liveness_score=liveness_score,
        ))
        s.commit()


def log_attendance_dedup(student_id: str, confidence: float, crop_path: str = "",
                         classroom_id: int = None, liveness_score: float = None) -> bool:
    """
    Log attendance only if the same student_id was NOT logged
    within the last _DEDUP_WINDOW_SECONDS. Returns True if logged.
    """
    cutoff = datetime.utcnow() - timedelta(seconds=_DEDUP_WINDOW_SECONDS)
    with get_session() as s:
        recent = (
            s.query(AttendanceRecord)
             .filter(
                 AttendanceRecord.student_id == student_id,
                 AttendanceRecord.timestamp >= cutoff,
             )
             .first()
        )
        if recent:
            return False
        s.add(AttendanceRecord(
            student_id=student_id,
            confidence=confidence,
            crop_path=crop_path,
            classroom_id=classroom_id,
            liveness_score=liveness_score,
        ))
        s.commit()
        return True


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


# ── Audit Log helpers ────────────────────────────────────────────────

def log_audit(user: str, action: str, target: str = None, details: str = None) -> None:
    with get_session() as s:
        s.add(AuditLog(user=user, action=action, target=target, details=details))
        s.commit()


def get_audit_logs(limit: int = 100) -> list:
    with get_session() as s:
        return s.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit).all()


# ── Classroom helpers ────────────────────────────────────────────────

def add_classroom(name: str, camera_url: str = None) -> int:
    with get_session() as s:
        c = Classroom(name=name, camera_url=camera_url)
        s.add(c)
        s.commit()
        return c.id


def get_all_classrooms() -> list:
    with get_session() as s:
        return s.query(Classroom).order_by(Classroom.name).all()


def delete_classroom(classroom_id: int) -> None:
    with get_session() as s:
        s.query(Classroom).filter_by(id=classroom_id).delete()
        s.commit()


# ── System Health helpers ────────────────────────────────────────────

def log_health(endpoint: str, inference_ms: float, faces_detected: int, success: bool = True) -> None:
    with get_session() as s:
        s.add(SystemHealthLog(
            endpoint=endpoint,
            inference_ms=inference_ms,
            faces_detected=faces_detected,
            success=success,
        ))
        s.commit()


def get_health_stats(hours: int = 24) -> dict:
    """Get health statistics for the last N hours."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    with get_session() as s:
        logs = s.query(SystemHealthLog).filter(SystemHealthLog.timestamp >= cutoff).all()
        if not logs:
            return {"total_requests": 0, "avg_inference_ms": 0, "success_rate": 100,
                    "total_faces": 0, "inference_times": [], "hourly_requests": []}

        total = len(logs)
        successes = sum(1 for l in logs if l.success)
        avg_ms = sum(l.inference_ms for l in logs) / total
        total_faces = sum(l.faces_detected for l in logs)

        # Inference time distribution (for histogram)
        inference_times = [l.inference_ms for l in logs]

        # Hourly request counts
        hourly = {}
        for l in logs:
            h = l.timestamp.strftime("%H:00")
            hourly[h] = hourly.get(h, 0) + 1

        return {
            "total_requests": total,
            "avg_inference_ms": round(avg_ms, 1),
            "success_rate": round(successes / total * 100, 1),
            "total_faces": total_faces,
            "inference_times": inference_times,
            "hourly_requests": [{"hour": k, "count": v} for k, v in sorted(hourly.items())],
        }


# ── Attendance query helpers ─────────────────────────────────────────

def get_attendance_by_date(target_date: date) -> list[dict]:
    """Return all attendance records for a specific date, joined with student names."""
    with get_session() as s:
        records = (
            s.query(AttendanceRecord, Student.name, Student.roll_no)
             .outerjoin(Student, AttendanceRecord.student_id == Student.student_id)
             .filter(func.date(AttendanceRecord.timestamp) == target_date)
             .order_by(AttendanceRecord.timestamp.desc())
             .all()
        )
        return [
            {
                "id": r.AttendanceRecord.id,
                "student_id": r.AttendanceRecord.student_id,
                "name": r.name or r.AttendanceRecord.student_id,
                "roll_no": r.roll_no or "",
                "timestamp": r.AttendanceRecord.timestamp,
                "confidence": r.AttendanceRecord.confidence,
                "liveness_score": r.AttendanceRecord.liveness_score,
                "classroom_id": r.AttendanceRecord.classroom_id,
                "status": "Verified" if r.AttendanceRecord.confidence and r.AttendanceRecord.confidence >= 0.45 else "Review",
            }
            for r in records
        ]


def get_attendance_filtered(
    date_from: date | None = None,
    date_to: date | None = None,
    student_id: str | None = None,
    status: str | None = None,
    classroom_id: int | None = None,
) -> list[dict]:
    """
    Query attendance with optional filters.
    If no filters are set, returns ALL attendance records.
    """
    with get_session() as s:
        q = (
            s.query(AttendanceRecord, Student.name, Student.roll_no)
             .outerjoin(Student, AttendanceRecord.student_id == Student.student_id)
        )

        if date_from:
            q = q.filter(func.date(AttendanceRecord.timestamp) >= date_from)
        if date_to:
            q = q.filter(func.date(AttendanceRecord.timestamp) <= date_to)
        if student_id:
            q = q.filter(AttendanceRecord.student_id == student_id)
        if classroom_id:
            q = q.filter(AttendanceRecord.classroom_id == classroom_id)

        records = q.order_by(AttendanceRecord.timestamp.desc()).all()

        result = []
        for r in records:
            conf = r.AttendanceRecord.confidence or 0
            rec_status = "Verified" if conf >= 0.45 else "Review"
            if status and rec_status.lower() != status.lower():
                continue
            result.append({
                "id": r.AttendanceRecord.id,
                "student_id": r.AttendanceRecord.student_id,
                "name": r.name or r.AttendanceRecord.student_id,
                "roll_no": r.roll_no or "",
                "timestamp": r.AttendanceRecord.timestamp,
                "confidence": conf,
                "liveness_score": r.AttendanceRecord.liveness_score,
                "classroom_id": r.AttendanceRecord.classroom_id,
                "status": rec_status,
            })
        return result


def export_attendance_csv(
    date_from: date | None = None,
    date_to: date | None = None,
    student_id: str | None = None,
    status: str | None = None,
    classroom_id: int | None = None,
) -> str:
    """Export filtered (or all) attendance to CSV string."""
    records = get_attendance_filtered(date_from, date_to, student_id, status, classroom_id)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["#", "Student ID", "Name", "Roll No", "Date", "Time", "Confidence", "Liveness", "Status"])
    for i, r in enumerate(records, 1):
        writer.writerow([
            i,
            r["student_id"],
            r["name"],
            r["roll_no"],
            r["timestamp"].strftime("%Y-%m-%d"),
            r["timestamp"].strftime("%H:%M:%S"),
            f"{r['confidence'] * 100:.1f}%",
            f"{(r['liveness_score'] or 0) * 100:.0f}%" if r.get('liveness_score') else "N/A",
            r["status"],
        ])
    return output.getvalue()


def get_weekly_summary(end_date: date | None = None) -> list[dict]:
    """Return daily attendance counts for the last 7 days."""
    if end_date is None:
        end_date = date.today()
    start_date = end_date - timedelta(days=6)

    with get_session() as s:
        results = []
        for i in range(7):
            d = start_date + timedelta(days=i)
            unique = (
                s.query(AttendanceRecord.student_id)
                 .filter(func.date(AttendanceRecord.timestamp) == d)
                 .distinct()
                 .count()
            )
            results.append({
                "date": d.isoformat(),
                "day": d.strftime("%a"),
                "count": unique,
            })
    return results


def get_monthly_heatmap(year: int = None, month: int = None) -> list[dict]:
    """Return daily attendance counts for a full month (for heatmap)."""
    today = date.today()
    if year is None:
        year = today.year
    if month is None:
        month = today.month

    # Get first and last day of month
    first_day = date(year, month, 1)
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)

    with get_session() as s:
        results = []
        d = first_day
        while d <= last_day:
            unique = (
                s.query(AttendanceRecord.student_id)
                 .filter(func.date(AttendanceRecord.timestamp) == d)
                 .distinct()
                 .count()
            )
            results.append({
                "date": d.isoformat(),
                "day": d.day,
                "weekday": d.strftime("%a"),
                "count": unique,
            })
            d += timedelta(days=1)
    return results


def get_student_attendance_history(student_id: str) -> list[dict]:
    """Return all attendance records for a specific student."""
    with get_session() as s:
        records = (
            s.query(AttendanceRecord)
             .filter(AttendanceRecord.student_id == student_id)
             .order_by(AttendanceRecord.timestamp.desc())
             .all()
        )
        return [
            {
                "date": r.timestamp.strftime("%Y-%m-%d"),
                "time": r.timestamp.strftime("%H:%M:%S"),
                "confidence": r.confidence,
                "liveness_score": r.liveness_score,
            }
            for r in records
        ]


def get_student_attendance_percentage(student_id: str) -> float:
    """Calculate attendance percentage: days present / total unique days."""
    with get_session() as s:
        total_days = (
            s.query(func.date(AttendanceRecord.timestamp))
             .distinct()
             .count()
        )
        if total_days == 0:
            return 0.0
        student_days = (
            s.query(func.date(AttendanceRecord.timestamp))
             .filter(AttendanceRecord.student_id == student_id)
             .distinct()
             .count()
        )
        return round((student_days / total_days) * 100, 1)


def delete_student_completely(student_id: str) -> None:
    """Delete student + all their attendance records, crops, and embeddings."""
    with get_session() as s:
        s.query(AttendanceRecord).filter_by(student_id=student_id).delete()
        s.query(CropBuffer).filter_by(student_id=student_id).delete()
        s.query(Student).filter_by(student_id=student_id).delete()
        s.commit()


def record_consent(student_id: str, given: bool = True) -> None:
    with get_session() as s:
        student = s.query(Student).filter_by(student_id=student_id).first()
        if student:
            student.consent_given = given
            student.consent_date = datetime.utcnow() if given else None
            s.commit()


if __name__ == "__main__":
    init_db()
    print("Database initialised at", _DB_PATH)
