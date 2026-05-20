"""
web/app.py
───────────
Flask web application — all routes for the Attendance System.

Routes (grouped by feature):
────────────────────────────
Auth:
  GET/POST /login              — admin login
  GET      /logout             — admin logout

Dashboard:
  GET  /                       — analytics dashboard with weekly trend + heatmap

Attendance:
  GET  /attendance             — attendance records with filters
  GET  /attendance/export      — CSV download (respects filters)

Testing (side feature):
  GET  /testing                — upload image for recognition testing
  POST /testing/analyze        — process image → recognition + attendance logging

Students:
  GET  /students               — enrolled students list
  GET  /students/<id>          — per-student profile
  POST /enroll                 — enroll new student (with optional photo upload)
  POST /students/<id>/delete   — delete student + all data
  POST /students/<id>/consent  — record biometric consent

Review:
  GET  /review                 — review queue (low-confidence crops)
  POST /review/<id>/confirm    — verify a crop
  POST /review/<id>/reject     — discard a crop

Classrooms:
  GET  /classrooms             — manage classrooms
  POST /classrooms/add         — add new classroom
  POST /classrooms/<id>/delete — remove classroom

Metrics & Monitoring:
  GET  /metrics                — model architecture & config
  GET  /benchmarks             — benchmark comparison page
  GET  /health                 — system health monitoring
  GET  /audit-log              — audit trail

Live Feed:
  GET  /live                   — live CCTV feed page

APIs:
  GET  /api/attendance         — today's attendance JSON
  GET  /api/weekly-trend       — weekly chart data
  GET  /api/health-stats       — health metrics JSON
  GET  /api/heatmap            — monthly heatmap data

Static:
  GET  /crops/<path>           — serve face crop images
  GET  /ref-images/<path>      — serve reference images
"""

import base64
import os
import shutil
import time
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from flask import (
    Flask, jsonify, render_template, request,
    redirect, url_for, send_from_directory, Response,
)
from flask_login import login_required, current_user
from sqlalchemy import func

from src.utils.db import (
    init_db, get_session, Student, AttendanceRecord, CropBuffer, Classroom,
    AdminUser, AuditLog, SystemHealthLog,
    mark_crop_verified, log_attendance_dedup,
    get_attendance_by_date, get_attendance_filtered, export_attendance_csv,
    get_weekly_summary, get_monthly_heatmap, get_student_attendance_history,
    get_student_attendance_percentage, get_all_classrooms,
    add_classroom, delete_classroom, delete_student_completely,
    record_consent, log_audit, get_audit_logs,
    log_health, get_health_stats,
)
from src.utils.config_loader import cfg
from src.utils.logger import log
from src.utils.notifications import is_email_configured

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_UPLOAD_DIR = _PROJECT_ROOT / "data" / "demo_uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_SEED_DIR = _PROJECT_ROOT / "data" / "seed_images"
_SEED_DIR.mkdir(parents=True, exist_ok=True)

# Anti-spoof threshold
_LIVENESS_THRESHOLD = cfg.get("anti_spoof", {}).get("threshold", 0.4)
_LIVENESS_ENABLED = cfg.get("anti_spoof", {}).get("enabled", True)


def create_app() -> Flask:
    template_dir = Path(__file__).parent / "templates"
    static_dir   = Path(__file__).parent / "static"
    app = Flask(__name__,
                template_folder=str(template_dir),
                static_folder=str(static_dir))

    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'faceguard-ai-secret-key-change-in-production')

    # ── Session cookie settings for reverse-proxy environments (HF Spaces) ──
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    # HF Spaces embeds the app in a cross-origin iframe (huggingface.co → hf.space).
    # SameSite=Lax blocks cookies in cross-origin iframes, so we must use None.
    if os.environ.get('SPACE_ID') or os.environ.get('FORCE_HTTPS'):
        app.config['SESSION_COOKIE_SAMESITE'] = 'None'   # Required for cross-origin iframe
        app.config['SESSION_COOKIE_SECURE'] = True        # SameSite=None requires Secure
        app.config['PREFERRED_URL_SCHEME'] = 'https'
    else:
        app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'     # Local dev — standard setting

    # ── Reverse-proxy support (trust X-Forwarded-* headers) ──────────────
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

    init_db()

    # ── Flask-Login Setup ─────────────────────────────────────────────
    from web.auth import login_manager, auth_bp
    login_manager.init_app(app)
    app.register_blueprint(auth_bp)

    # ══════════════════════════════════════════════════════════════════
    #  DASHBOARD
    # ══════════════════════════════════════════════════════════════════

    @app.route("/")
    @login_required
    def dashboard():
        today = date.today()
        with get_session() as s:
            records = (
                s.query(AttendanceRecord)
                 .filter(func.date(AttendanceRecord.timestamp) == today)
                 .order_by(AttendanceRecord.timestamp.desc())
                 .limit(10)
                 .all()
            )
            total_students = s.query(Student).count()
            unique_today = (
                s.query(AttendanceRecord.student_id)
                 .filter(func.date(AttendanceRecord.timestamp) == today)
                 .distinct()
                 .count()
            )
            total_records = (
                s.query(AttendanceRecord)
                 .filter(func.date(AttendanceRecord.timestamp) == today)
                 .count()
            )
        weekly = get_weekly_summary()
        heatmap = get_monthly_heatmap()
        attendance_rate = round((unique_today / total_students * 100), 1) if total_students else 0

        recent_activity = []
        with get_session() as s:
            recent = (
                s.query(AttendanceRecord, Student.name)
                 .outerjoin(Student, AttendanceRecord.student_id == Student.student_id)
                 .order_by(AttendanceRecord.timestamp.desc())
                 .limit(10)
                 .all()
            )
            for r in recent:
                recent_activity.append({
                    "student_id": r.AttendanceRecord.student_id,
                    "name": r.name or r.AttendanceRecord.student_id,
                    "timestamp": r.AttendanceRecord.timestamp,
                    "confidence": r.AttendanceRecord.confidence,
                })

        return render_template(
            "dashboard.html",
            records=records, today=today, total_students=total_students,
            unique_today=unique_today, total_records=total_records,
            weekly=weekly, heatmap=heatmap, attendance_rate=attendance_rate,
            recent_activity=recent_activity,
        )

    # ══════════════════════════════════════════════════════════════════
    #  ATTENDANCE
    # ══════════════════════════════════════════════════════════════════

    @app.route("/attendance")
    @login_required
    def attendance():
        date_from_str  = request.args.get("date_from", "")
        date_to_str    = request.args.get("date_to", "")
        student_filter = request.args.get("student_id", "").strip()
        status_filter  = request.args.get("status", "").strip()
        classroom_filter = request.args.get("classroom_id", "").strip()

        date_from = date_to = None
        if date_from_str:
            try: date_from = datetime.strptime(date_from_str, "%Y-%m-%d").date()
            except ValueError: pass
        if date_to_str:
            try: date_to = datetime.strptime(date_to_str, "%Y-%m-%d").date()
            except ValueError: pass

        classroom_id = int(classroom_filter) if classroom_filter else None

        records = get_attendance_filtered(
            date_from=date_from, date_to=date_to,
            student_id=student_filter or None,
            status=status_filter or None,
            classroom_id=classroom_id,
        )

        with get_session() as s:
            all_students = s.query(Student).order_by(Student.name).all()
            total_students = s.query(Student).count()

        classrooms = get_all_classrooms()
        unique_ids = set(r["student_id"] for r in records)
        present_count = len(unique_ids)
        absent_count = total_students - present_count if not date_from and not date_to else 0

        return render_template(
            "attendance.html",
            records=records, all_students=all_students, classrooms=classrooms,
            present_count=present_count, absent_count=absent_count,
            total_records=len(records),
            date_from=date_from_str, date_to=date_to_str,
            student_filter=student_filter, status_filter=status_filter,
            classroom_filter=classroom_filter,
        )

    @app.route("/attendance/export")
    @login_required
    def attendance_export():
        date_from_str  = request.args.get("date_from", "")
        date_to_str    = request.args.get("date_to", "")
        student_filter = request.args.get("student_id", "").strip()
        status_filter  = request.args.get("status", "").strip()
        classroom_filter = request.args.get("classroom_id", "").strip()

        date_from = date_to = None
        if date_from_str:
            try: date_from = datetime.strptime(date_from_str, "%Y-%m-%d").date()
            except ValueError: pass
        if date_to_str:
            try: date_to = datetime.strptime(date_to_str, "%Y-%m-%d").date()
            except ValueError: pass

        classroom_id = int(classroom_filter) if classroom_filter else None

        csv_data = export_attendance_csv(
            date_from=date_from, date_to=date_to,
            student_id=student_filter or None,
            status=status_filter or None,
            classroom_id=classroom_id,
        )

        parts = ["attendance"]
        if date_from_str: parts.append(f"from_{date_from_str}")
        if date_to_str: parts.append(f"to_{date_to_str}")
        if not date_from_str and not date_to_str: parts.append("complete")
        filename = "_".join(parts) + ".csv"

        log_audit(current_user.username, "EXPORT_CSV", details=filename)

        return Response(
            csv_data, mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    # ══════════════════════════════════════════════════════════════════
    #  TESTING (Interactive Recognition — Side Feature)
    # ══════════════════════════════════════════════════════════════════

    @app.route("/testing")
    @login_required
    def testing():
        with get_session() as s:
            total_students = s.query(Student).count()
        return render_template("testing.html",
                               total_students=total_students,
                               liveness_enabled=_LIVENESS_ENABLED)

    @app.route("/testing/analyze", methods=["POST"])
    @login_required
    def testing_analyze():
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image file"}), 400

        uid = str(uuid.uuid4())[:8]
        save_path = _UPLOAD_DIR / f"{uid}_{file.filename}"
        cv2.imwrite(str(save_path), frame)

        from src.inference.embedder import get_embedder
        from src.inference.matcher import get_matcher
        from src.inference.anti_spoof import check_liveness

        embedder = get_embedder()
        matcher  = get_matcher()

        t0 = time.perf_counter()
        results = []
        annotated = frame.copy()
        attendance_logged = []
        liveness_results = []

        if hasattr(embedder, 'embed_from_frame'):
            face_results = embedder.embed_from_frame(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces_meta = embedder.app.get(rgb)

            for i, (crop, emb) in enumerate(face_results):
                pred_id, confidence, decision = matcher.match(emb)

                # Anti-spoofing check (doesn't modify recognition)
                liveness_score = check_liveness(crop) if _LIVENESS_ENABLED else 1.0
                liveness_pass = liveness_score >= _LIVENESS_THRESHOLD

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
                    "liveness_score": round(liveness_score, 3),
                    "liveness_pass": liveness_pass,
                })

                # Only log attendance if BOTH high-confidence AND liveness passes
                if decision == "high" and pred_id and liveness_pass:
                    logged = log_attendance_dedup(pred_id, confidence, str(save_path),
                                                  liveness_score=liveness_score)
                    if logged:
                        attendance_logged.append(pred_id)

                # Annotate
                if i < len(faces_meta):
                    box = faces_meta[i].bbox.astype(int)
                    if decision == "high" and liveness_pass:
                        color = (0, 255, 0)
                    elif decision == "high" and not liveness_pass:
                        color = (0, 0, 255)  # red = spoof suspected
                    elif decision == "soft":
                        color = (0, 200, 255)
                    else:
                        color = (0, 0, 255)

                    cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, 2)
                    label = f"{pred_id or '?'} ({confidence*100:.0f}%)"
                    if not liveness_pass:
                        label += " SPOOF?"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (box[0], box[1]-th-10), (box[0]+tw+4, box[1]), color, -1)
                    cv2.putText(annotated, label, (box[0]+2, box[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:
            from src.inference.face_detector import detect_faces
            crops = detect_faces(frame)
            if crops:
                embs = embedder.embed_batch(crops)
                for crop_item, emb in zip(crops, embs):
                    pred_id, confidence, decision = matcher.match(emb)
                    liveness_score = check_liveness(crop_item) if _LIVENESS_ENABLED else 1.0
                    liveness_pass = liveness_score >= _LIVENESS_THRESHOLD
                    results.append({
                        "student_id": pred_id or "Unknown",
                        "name": pred_id or "Unknown",
                        "confidence": round(float(confidence) * 100, 1),
                        "decision": decision,
                        "liveness_score": round(liveness_score, 3),
                        "liveness_pass": liveness_pass,
                    })
                    if decision == "high" and pred_id and liveness_pass:
                        logged = log_attendance_dedup(pred_id, confidence,
                                                      liveness_score=liveness_score)
                        if logged:
                            attendance_logged.append(pred_id)

        dt = time.perf_counter() - t0

        # Log system health
        log_health("testing_analyze", round(dt * 1000, 1), len(results), success=True)
        log_audit(current_user.username, "ANALYZE_IMAGE",
                  details=f"{len(results)} faces, {len(attendance_logged)} logged")

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        img_b64 = base64.b64encode(buf).decode("utf-8")

        return jsonify({
            "faces_detected": len(results),
            "processing_time_ms": round(dt * 1000, 1),
            "results": results,
            "annotated_image": f"data:image/jpeg;base64,{img_b64}",
            "attendance_logged": attendance_logged,
            "liveness_enabled": _LIVENESS_ENABLED,
        })

    # ══════════════════════════════════════════════════════════════════
    #  REVIEW QUEUE
    # ══════════════════════════════════════════════════════════════════

    @app.route("/review")
    @login_required
    def review():
        with get_session() as s:
            pending = (
                s.query(CropBuffer).filter_by(verified=False)
                 .order_by(CropBuffer.created_at.desc()).limit(50).all()
            )
        return render_template("review.html", pending=pending)

    @app.route("/review/<int:crop_id>/confirm", methods=["POST"])
    @login_required
    def confirm_crop(crop_id: int):
        mark_crop_verified(crop_id)
        log_audit(current_user.username, "CONFIRM_CROP", target=str(crop_id))
        return redirect(url_for("review"))

    @app.route("/review/<int:crop_id>/reject", methods=["POST"])
    @login_required
    def reject_crop(crop_id: int):
        with get_session() as s:
            rec = s.query(CropBuffer).get(crop_id)
            if rec:
                try: Path(rec.crop_path).unlink(missing_ok=True)
                except Exception: pass
                s.delete(rec)
                s.commit()
        log_audit(current_user.username, "REJECT_CROP", target=str(crop_id))
        return redirect(url_for("review"))

    # ══════════════════════════════════════════════════════════════════
    #  STUDENTS
    # ══════════════════════════════════════════════════════════════════

    @app.route("/students")
    @login_required
    def students():
        with get_session() as s:
            all_students = s.query(Student).order_by(Student.name).all()
        return render_template("students.html", students=all_students)

    @app.route("/students/<student_id>")
    @login_required
    def student_profile(student_id: str):
        with get_session() as s:
            student = s.query(Student).filter_by(student_id=student_id).first()
        if not student:
            return redirect(url_for("students"))
        history = get_student_attendance_history(student_id)
        percentage = get_student_attendance_percentage(student_id)

        ref_path = _PROJECT_ROOT / "data" / "reference_images" / f"{student_id}.jpeg"
        has_photo = ref_path.exists()
        if not has_photo:
            ref_path = _PROJECT_ROOT / "data" / "reference_images" / f"{student_id}.jpg"
            has_photo = ref_path.exists()
        # Also check seed images
        if not has_photo:
            seed_dir = _SEED_DIR / student_id
            if seed_dir.exists():
                imgs = list(seed_dir.glob("*.jpg")) + list(seed_dir.glob("*.jpeg")) + list(seed_dir.glob("*.png"))
                has_photo = len(imgs) > 0

        return render_template(
            "student_profile.html",
            student=student, history=history, percentage=percentage,
            has_photo=has_photo, total_days=len(set(h["date"] for h in history)),
        )

    @app.route("/enroll", methods=["POST"])
    @login_required
    def enroll():
        student_id = request.form["student_id"].strip()
        name       = request.form["name"].strip()
        roll_no    = request.form["roll_no"].strip()

        # Handle photo upload (new feature)
        photo_files = request.files.getlist("photos")
        seed_dir = _SEED_DIR / student_id
        seed_dir.mkdir(parents=True, exist_ok=True)

        photo_saved = 0
        for pf in photo_files:
            if pf and pf.filename:
                ext = Path(pf.filename).suffix.lower()
                if ext in {".jpg", ".jpeg", ".png", ".bmp"}:
                    save_name = f"{student_id}_{photo_saved}{ext}"
                    pf.save(str(seed_dir / save_name))
                    photo_saved += 1

        from src.enrollment.enroll_student import enroll as do_enroll
        success = do_enroll(student_id, name, roll_no)
        log_audit(current_user.username, "ENROLL_STUDENT",
                  target=student_id, details=f"{photo_saved} photos uploaded")

        msg = f"Enrolled {name} successfully with {photo_saved} photo(s)." if success else "Enrollment failed — check seed images."
        return render_template("enroll_result.html", message=msg, success=success)

    @app.route("/students/<student_id>/delete", methods=["POST"])
    @login_required
    def delete_student(student_id: str):
        # Remove from FAISS
        # Note: FAISS FlatIP doesn't support deletion. We remove from DB only.
        # A full rebuild would be needed for production.
        delete_student_completely(student_id)

        # Remove seed images
        seed_dir = _SEED_DIR / student_id
        if seed_dir.exists():
            shutil.rmtree(seed_dir, ignore_errors=True)

        log_audit(current_user.username, "DELETE_STUDENT",
                  target=student_id, details="All data removed")
        return redirect(url_for("students"))

    @app.route("/students/<student_id>/consent", methods=["POST"])
    @login_required
    def update_consent(student_id: str):
        given = request.form.get("consent") == "true"
        record_consent(student_id, given)
        log_audit(current_user.username, "UPDATE_CONSENT",
                  target=student_id, details=f"Consent {'granted' if given else 'revoked'}")
        return redirect(url_for("student_profile", student_id=student_id))

    # ══════════════════════════════════════════════════════════════════
    #  CLASSROOMS
    # ══════════════════════════════════════════════════════════════════

    @app.route("/classrooms")
    @login_required
    def classrooms():
        rooms = get_all_classrooms()
        return render_template("classrooms.html", classrooms=rooms)

    @app.route("/classrooms/add", methods=["POST"])
    @login_required
    def add_classroom_route():
        name = request.form.get("name", "").strip()
        camera_url = request.form.get("camera_url", "").strip()
        if name:
            add_classroom(name, camera_url or None)
            log_audit(current_user.username, "ADD_CLASSROOM", target=name)
        return redirect(url_for("classrooms"))

    @app.route("/classrooms/<int:classroom_id>/delete", methods=["POST"])
    @login_required
    def delete_classroom_route(classroom_id: int):
        delete_classroom(classroom_id)
        log_audit(current_user.username, "DELETE_CLASSROOM", target=str(classroom_id))
        return redirect(url_for("classrooms"))

    # ══════════════════════════════════════════════════════════════════
    #  METRICS, BENCHMARKS, HEALTH
    # ══════════════════════════════════════════════════════════════════

    @app.route("/metrics")
    @login_required
    def metrics():
        from src.inference.matcher import get_matcher
        matcher = get_matcher()
        gallery_size = matcher.index.ntotal

        with get_session() as s:
            total_students = s.query(Student).count()
            total_attendance = s.query(AttendanceRecord).count()

        return render_template(
            "metrics.html", gallery_size=gallery_size,
            total_students=total_students, total_attendance=total_attendance,
            liveness_enabled=_LIVENESS_ENABLED,
            liveness_threshold=_LIVENESS_THRESHOLD,
        )

    @app.route("/benchmarks")
    @login_required
    def benchmarks():
        from src.inference.matcher import get_matcher
        matcher = get_matcher()
        gallery_size = matcher.index.ntotal
        return render_template("benchmarks.html", gallery_size=gallery_size)

    @app.route("/health")
    @login_required
    def health():
        stats = get_health_stats(hours=24)
        return render_template("health.html", stats=stats)

    @app.route("/audit-log")
    @login_required
    def audit_log_page():
        logs = get_audit_logs(limit=200)
        return render_template("audit_log.html", logs=logs)

    # ══════════════════════════════════════════════════════════════════
    #  LIVE FEED
    # ══════════════════════════════════════════════════════════════════

    @app.route("/live")
    @login_required
    def live():
        classrooms_list = get_all_classrooms()
        return render_template("live.html", classrooms=classrooms_list,
                               liveness_enabled=_LIVENESS_ENABLED)

    @app.route("/api/live/frame", methods=["POST"])
    @login_required
    def api_live_frame():
        """
        Accept a webcam frame as base64 JPEG, run the full
        detection → embedding → matching → anti-spoof pipeline,
        and return JSON with results + bounding boxes.
        """
        data = request.get_json(silent=True)
        if not data or "frame" not in data:
            return jsonify({"error": "No frame data"}), 400

        # Decode base64 → OpenCV BGR frame
        frame_b64 = data["frame"]
        # Strip optional data-URI prefix
        if "," in frame_b64:
            frame_b64 = frame_b64.split(",", 1)[1]

        try:
            img_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            return jsonify({"error": "Invalid frame data"}), 400

        if frame is None:
            return jsonify({"error": "Could not decode frame"}), 400

        from src.inference.embedder import get_embedder
        from src.inference.matcher import get_matcher
        from src.inference.anti_spoof import check_liveness

        embedder = get_embedder()
        matcher  = get_matcher()

        t0 = time.perf_counter()
        results = []
        attendance_logged = []

        if hasattr(embedder, 'embed_from_frame'):
            face_results = embedder.embed_from_frame(frame)
            # Get face metadata (bounding boxes) from InsightFace
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces_meta = embedder.app.get(rgb)

            for i, (crop, emb) in enumerate(face_results):
                pred_id, confidence, decision = matcher.match(emb)

                liveness_score = check_liveness(crop) if _LIVENESS_ENABLED else 1.0
                liveness_pass = liveness_score >= _LIVENESS_THRESHOLD

                student_name = pred_id or "Unknown"
                if pred_id:
                    with get_session() as s:
                        student = s.query(Student).filter_by(student_id=pred_id).first()
                        if student:
                            student_name = student.name

                # Extract bounding box
                bbox = None
                if i < len(faces_meta):
                    box = faces_meta[i].bbox.astype(int).tolist()
                    bbox = box  # [x1, y1, x2, y2]

                results.append({
                    "student_id": pred_id or "Unknown",
                    "name": student_name,
                    "confidence": round(float(confidence) * 100, 1),
                    "decision": decision,
                    "liveness_score": round(liveness_score, 3),
                    "liveness_pass": liveness_pass,
                    "bbox": bbox,
                })

                if decision == "high" and pred_id and liveness_pass:
                    logged = log_attendance_dedup(pred_id, confidence,
                                                  liveness_score=liveness_score)
                    if logged:
                        attendance_logged.append(pred_id)
        else:
            from src.inference.face_detector import detect_faces
            crops = detect_faces(frame)
            if crops:
                embs = embedder.embed_batch(crops)
                for crop_item, emb in zip(crops, embs):
                    pred_id, confidence, decision = matcher.match(emb)
                    liveness_score = check_liveness(crop_item) if _LIVENESS_ENABLED else 1.0
                    liveness_pass = liveness_score >= _LIVENESS_THRESHOLD
                    results.append({
                        "student_id": pred_id or "Unknown",
                        "name": pred_id or "Unknown",
                        "confidence": round(float(confidence) * 100, 1),
                        "decision": decision,
                        "liveness_score": round(liveness_score, 3),
                        "liveness_pass": liveness_pass,
                        "bbox": None,
                    })
                    if decision == "high" and pred_id and liveness_pass:
                        logged = log_attendance_dedup(pred_id, confidence,
                                                      liveness_score=liveness_score)
                        if logged:
                            attendance_logged.append(pred_id)

        dt = time.perf_counter() - t0
        log_health("live_frame", round(dt * 1000, 1), len(results), success=True)

        return jsonify({
            "faces_detected": len(results),
            "processing_time_ms": round(dt * 1000, 1),
            "results": results,
            "attendance_logged": attendance_logged,
            "liveness_enabled": _LIVENESS_ENABLED,
            "frame_width": frame.shape[1],
            "frame_height": frame.shape[0],
        })

    # ══════════════════════════════════════════════════════════════════
    #  JSON APIs
    # ══════════════════════════════════════════════════════════════════

    @app.route("/api/attendance")
    @login_required
    def api_attendance():
        today = date.today()
        with get_session() as s:
            records = (
                s.query(AttendanceRecord)
                 .filter(func.date(AttendanceRecord.timestamp) == today).all()
            )
            data = [{"id": r.id, "student_id": r.student_id,
                     "timestamp": r.timestamp.isoformat(), "confidence": r.confidence}
                    for r in records]
        return jsonify({"date": str(today), "count": len(data), "records": data})

    @app.route("/api/weekly-trend")
    @login_required
    def api_weekly_trend():
        return jsonify(get_weekly_summary())

    @app.route("/api/health-stats")
    @login_required
    def api_health_stats():
        return jsonify(get_health_stats(hours=24))

    @app.route("/api/heatmap")
    @login_required
    def api_heatmap():
        year = request.args.get("year", type=int)
        month = request.args.get("month", type=int)
        return jsonify(get_monthly_heatmap(year, month))

    # ══════════════════════════════════════════════════════════════════
    #  STATIC FILE SERVING
    # ══════════════════════════════════════════════════════════════════

    @app.route("/crops/<path:filepath>")
    @login_required
    def serve_crop(filepath):
        return send_from_directory(str(_PROJECT_ROOT / "data"), filepath)

    @app.route("/ref-images/<path:filepath>")
    @login_required
    def serve_ref_image(filepath):
        return send_from_directory(str(_PROJECT_ROOT / "data" / "reference_images"), filepath)

    return app


if __name__ == "__main__":
    """
    Standalone launcher — run `python web/app.py` to start the full system.
    Automatically initialises:
      1. SQLite database (with schema migrations)
      2. InsightFace buffalo_l recognition model (ResNet-50)
      3. FAISS IndexFlatIP gallery of enrolled embeddings
      4. Flask web application on http://localhost:5000
    """
    import sys
    import argparse

    # Ensure project root is importable
    sys.path.insert(0, str(_PROJECT_ROOT))

    parser = argparse.ArgumentParser(description="Face Recognition Based Attendance System")
    parser.add_argument("--port", type=int, default=5000, help="Web server port (default: 5000)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    args = parser.parse_args()

    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║   Attendance System                              ║")
    print("  ╠══════════════════════════════════════════════════╣")
    print("  ║  Model:     InsightFace buffalo_l (ResNet-50)    ║")
    print("  ║  Detector:  RetinaFace (det_10g)                 ║")
    print("  ║  Matching:  FAISS IndexFlatIP (cosine)           ║")
    print("  ║  Pipeline:  CLAHE → Denoise → Embed → Match     ║")
    print("  ║  Anti-Spoof: LBP + FFT + Color + Moiré          ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()

    # 1. Database
    log.info("Initialising database (with schema migrations)...")
    init_db()

    # 2. Recognition Model
    log.info("Loading recognition model (this may take a few seconds)...")
    from src.inference.embedder import get_embedder
    embedder = get_embedder()
    log.info(f"Embedder ready: {type(embedder).__name__}")

    # 3. FAISS Gallery
    log.info("Loading FAISS index...")
    from src.inference.matcher import get_matcher
    matcher = get_matcher()
    log.info(f"Gallery: {matcher.index.ntotal} enrolled embeddings")

    print()
    print(f"  ✓ System ready!")
    print(f"  ✓ Open your browser at: http://localhost:{args.port}")
    print(f"  ✓ Default login: admin / admin123")
    print(f"  ✓ Press Ctrl+C to stop")
    print()

    # 4. Start Flask
    app = create_app()
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
