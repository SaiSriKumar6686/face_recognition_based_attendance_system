"""
enroll_student.py
──────────────────
Enroll a new student by:
  1. Reading seed images from data/seed_images/<student_id>/
  2. Running face detection + alignment on each
  3. Embedding each crop and computing the mean embedding
  4. Storing the mean embedding in FAISS
  5. Writing student metadata to SQLite

After enrollment the student is immediately recognisable for attendance,
even before any real CCTV data is collected (Phase 1 → Phase 2 transition).

Usage
─────
    python -m src.enrollment.enroll_student \
        --student_id  s001 \
        --name        "Sai Kumar" \
        --roll_no     21CS001

Or import and call enroll() from the web admin API.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.utils.config_loader import cfg
from src.utils.logger import log
from src.utils.db import init_db, add_student, get_session, Student
from src.inference.face_detector import detect_faces
from src.inference.embedder import get_embedder
from src.inference.matcher import get_matcher

_SEED_DIR = Path(cfg["paths"]["seed_images"])
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def enroll(student_id: str, name: str, roll_no: str) -> bool:
    """
    Enroll a student.

    Parameters
    ----------
    student_id : unique identifier (used as folder name under seed_images/).
    name       : display name.
    roll_no    : college roll number (must be unique).

    Returns
    -------
    True on success, False if no faces were detected in seed images.
    """
    init_db()

    seed_dir = _SEED_DIR / student_id
    if not seed_dir.exists():
        log.error(f"Seed folder not found: {seed_dir}")
        return False

    img_paths = [p for p in seed_dir.iterdir() if p.suffix.lower() in _IMG_EXTS]
    if not img_paths:
        log.error(f"No images found in {seed_dir}")
        return False

    log.info(f"Enrolling '{name}' ({student_id}) with {len(img_paths)} seed image(s)…")

    embedder = get_embedder()
    embeddings = []

    # Check if we have InsightFaceEmbedder (has embed_from_frame)
    _has_frame_embed = hasattr(embedder, 'embed_from_frame')

    for img_path in img_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            log.warning(f"  Could not read: {img_path.name}")
            continue

        if _has_frame_embed:
            # InsightFace path: detect + embed in one shot on full frame
            results = embedder.embed_from_frame(frame)
            if not results:
                log.warning(f"  No face detected in: {img_path.name}")
                continue
            # take the best face (largest crop)
            crop, emb = max(results, key=lambda r: r[0].shape[0] * r[0].shape[1])
            embeddings.append(emb)
        else:
            # ONNX fallback path: detect → crop → embed
            crops = detect_faces(frame)
            if not crops:
                log.warning(f"  No face detected in: {img_path.name}")
                continue
            crop = max(crops, key=lambda c: c.shape[0] * c.shape[1])
            emb = embedder.embed(crop)
            embeddings.append(emb)

        log.debug(f"  ✓ {img_path.name}")

    if not embeddings:
        log.error("No valid face embeddings produced — enrollment failed.")
        return False

    # mean embedding (robust to small pose variations across seed images)
    mean_emb = np.mean(embeddings, axis=0).astype(np.float32)
    norm     = np.linalg.norm(mean_emb)
    mean_emb = mean_emb / (norm + 1e-8)

    # check for existing student
    with get_session() as s:
        existing = s.query(Student).filter_by(student_id=student_id).first()
        if existing:
            log.info(f"  Student {student_id} already in DB — updating embedding only.")
        else:
            add_student(student_id, name, roll_no)

    # add to FAISS
    matcher = get_matcher()
    matcher.add_embedding(student_id, mean_emb)
    matcher.save()

    log.info(f"✓ Enrolled {name} ({student_id}) | {len(embeddings)} embedding(s) used.")
    return True


def enroll_all_from_seed_dir() -> None:
    """
    Batch-enroll all students found as subdirectories under data/seed_images/.
    Each folder must be named <student_id> and the student must already exist
    in the DB, OR you can extend this to read a CSV manifest.
    """
    if not _SEED_DIR.exists():
        log.error(f"Seed directory not found: {_SEED_DIR}")
        return

    student_dirs = [d for d in _SEED_DIR.iterdir() if d.is_dir()]
    log.info(f"Found {len(student_dirs)} student folder(s) to enroll.")

    for d in student_dirs:
        # For batch enroll without a manifest, student name = folder name
        enroll(student_id=d.name, name=d.name, roll_no=d.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll a student into the attendance system.")
    parser.add_argument("--student_id", required=True)
    parser.add_argument("--name",       required=True)
    parser.add_argument("--roll_no",    required=True)
    parser.add_argument("--all",        action="store_true",
                        help="Enroll all students found in seed_images/")
    args = parser.parse_args()

    if args.all:
        enroll_all_from_seed_dir()
    else:
        success = enroll(args.student_id, args.name, args.roll_no)
        exit(0 if success else 1)
