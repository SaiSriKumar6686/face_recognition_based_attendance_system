"""
web/app.py
───────────
Flask web admin dashboard + standalone demo application.

Routes
──────
GET  /                     — dashboard (attendance today)
GET  /demo                 — interactive demo: upload image → see recognition results
POST /demo/analyze         — process uploaded image and return annotated results
GET  /review               — review queue (low-confidence crops)
POST /review/<id>/confirm  — mark a crop as verified
POST /review/<id>/reject   — discard a crop
GET  /students             — list enrolled students
POST /enroll               — enroll new student (form submission)
GET  /api/attendance       — JSON API for today's attendance records
GET  /crops/<path>         — serve saved face crop images
"""

import io
import base64
import time
import uuid
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, redirect, url_for, send_from_directory

from sqlalchemy import func

from src.utils.db import (
    init_db, get_session, Student, AttendanceRecord,
    CropBuffer, mark_crop_verified,
)
from src.utils.config_loader import cfg
from src.utils.logger import log

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_UPLOAD_DIR = _PROJECT_ROOT / "data" / "demo_uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def create_app() -> Flask:
    template_dir = Path(__file__).parent / "templates"
    static_dir   = Path(__file__).parent / "static"
    app = Flask(__name__,
                template_folder=str(template_dir),
                static_folder=str(static_dir))

    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

    init_db()

    # ── Dashboard ─────────────────────────────────────────────────────

    @app.route("/")
    def dashboard():
        today = date.today()
        with get_session() as s:
            records = (
                s.query(AttendanceRecord)
                 .filter(func.date(AttendanceRecord.timestamp) == today)
                 .order_by(AttendanceRecord.timestamp.desc())
                 .limit(100)
                 .all()
            )
            total_students = s.query(Student).count()
            # Count unique students detected today
            unique_today = (
                s.query(AttendanceRecord.student_id)
                 .filter(func.date(AttendanceRecord.timestamp) == today)
                 .distinct()
                 .count()
            )
        return render_template(
            "dashboard.html",
            records=records,
            today=today,
            total_students=total_students,
            unique_today=unique_today,
        )

    # ── Demo (Interactive Upload) ─────────────────────────────────────

    @app.route("/demo")
    def demo():
        with get_session() as s:
            total_students = s.query(Student).count()
        return render_template("demo.html", total_students=total_students)

    @app.route("/demo/analyze", methods=["POST"])
    def demo_analyze():
        """Process an uploaded image: detect faces → embed → match → return annotated result."""
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # Read image from upload
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image file"}), 400

        # Save original
        uid = str(uuid.uuid4())[:8]
        save_path = _UPLOAD_DIR / f"{uid}_{file.filename}"
        cv2.imwrite(str(save_path), frame)

        # Run the recognition pipeline
        from src.inference.embedder import get_embedder
        from src.inference.matcher import get_matcher

        embedder = get_embedder()
        matcher  = get_matcher()

        t0 = time.perf_counter()

        results = []
        annotated = frame.copy()

        if hasattr(embedder, 'embed_from_frame'):
            # InsightFace path — unified detection + recognition
            face_results = embedder.embed_from_frame(frame)
            # Also get bounding boxes from the embedder's app
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces_meta = embedder.app.get(rgb)

            for i, (crop, emb) in enumerate(face_results):
                pred_id, confidence, decision = matcher.match(emb)

                # Get student name from DB
                student_name = pred_id or "Unknown"
                if pred_id:
                    with get_session() as s:
                        student = s.query(Student).filter_by(student_id=pred_id).first()
                        if student:
                            student_name = student.name

                results.append({
                    "student_id": pred_id or "Unknown",
                    "name": student_name,
                    "confidence": round(float(confidence) * 100, 1),
                    "decision": decision,
                })

                # Annotate
                if i < len(faces_meta):
                    box = faces_meta[i].bbox.astype(int)
                    if decision == "high":
                        color = (0, 255, 0)
                    elif decision == "soft":
                        color = (0, 200, 255)
                    else:
                        color = (0, 0, 255)

                    cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, 2)

                    label = f"{pred_id or '?'} ({confidence*100:.0f}%)"
                    # Background for text
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (box[0], box[1]-th-10), (box[0]+tw+4, box[1]), color, -1)
                    cv2.putText(annotated, label, (box[0]+2, box[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:
            # ONNX fallback
            from src.inference.face_detector import detect_faces
            crops = detect_faces(frame)
            if crops:
                embs = embedder.embed_batch(crops)
                for emb in embs:
                    pred_id, confidence, decision = matcher.match(emb)
                    results.append({
                        "student_id": pred_id or "Unknown",
                        "name": pred_id or "Unknown",
                        "confidence": round(float(confidence) * 100, 1),
                        "decision": decision,
                    })

        dt = time.perf_counter() - t0

        # Encode annotated image to base64
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_b64 = base64.b64encode(buf).decode("utf-8")

        # Save annotated image
        cv2.imwrite(str(_UPLOAD_DIR / f"{uid}_annotated.jpg"), annotated)

        return jsonify({
            "faces_detected": len(results),
            "processing_time_ms": round(dt * 1000, 1),
            "results": results,
            "annotated_image": f"data:image/jpeg;base64,{img_b64}",
        })

    # ── Review queue ──────────────────────────────────────────────────

    @app.route("/review")
    def review():
        with get_session() as s:
            pending = (
                s.query(CropBuffer)
                 .filter_by(verified=False)
                 .order_by(CropBuffer.created_at.desc())
                 .limit(50)
                 .all()
            )
        return render_template("review.html", pending=pending)

    @app.route("/review/<int:crop_id>/confirm", methods=["POST"])
    def confirm_crop(crop_id: int):
        mark_crop_verified(crop_id)
        log.info(f"Admin confirmed crop_id={crop_id}")
        return redirect(url_for("review"))

    @app.route("/review/<int:crop_id>/reject", methods=["POST"])
    def reject_crop(crop_id: int):
        with get_session() as s:
            rec = s.query(CropBuffer).get(crop_id)
            if rec:
                try:
                    Path(rec.crop_path).unlink(missing_ok=True)
                except Exception:
                    pass
                s.delete(rec)
                s.commit()
        log.info(f"Admin rejected crop_id={crop_id}")
        return redirect(url_for("review"))

    # ── Students ──────────────────────────────────────────────────────

    @app.route("/students")
    def students():
        with get_session() as s:
            all_students = s.query(Student).order_by(Student.name).all()
        return render_template("students.html", students=all_students)

    @app.route("/enroll", methods=["POST"])
    def enroll():
        student_id = request.form["student_id"].strip()
        name       = request.form["name"].strip()
        roll_no    = request.form["roll_no"].strip()
        from src.enrollment.enroll_student import enroll as do_enroll
        success = do_enroll(student_id, name, roll_no)
        msg = "Enrolled successfully." if success else "Enrollment failed — check seed images."
        return render_template("enroll_result.html", message=msg, success=success)

    # ── JSON API ──────────────────────────────────────────────────────

    @app.route("/api/attendance")
    def api_attendance():
        today = date.today()
        with get_session() as s:
            records = (
                s.query(AttendanceRecord)
                 .filter(func.date(AttendanceRecord.timestamp) == today)
                 .all()
            )
            data = [
                {
                    "id":         r.id,
                    "student_id": r.student_id,
                    "timestamp":  r.timestamp.isoformat(),
                    "confidence": r.confidence,
                }
                for r in records
            ]
        return jsonify({"date": str(today), "count": len(data), "records": data})

    # ── Static file serving for crops ─────────────────────────────────

    @app.route("/crops/<path:filepath>")
    def serve_crop(filepath):
        crops_dir = _PROJECT_ROOT / "data"
        return send_from_directory(str(crops_dir), filepath)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
