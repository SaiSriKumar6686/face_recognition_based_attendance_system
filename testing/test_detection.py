"""
test_detection.py
─────────────────
Script 1: Face Detection Quality Test

Tests the RetinaFace detector on:
  1. Every reference image  → expects exactly 1 face per image.
  2. Every group test image → counts detected faces.
  3. Saves annotated images with bounding boxes to testing/results/detection/.

Metrics reported:
  • Detection rate (% of reference images where a face was found)
  • False-negative list (reference images with 0 detections)
  • Per-group-image face count

Usage:
    python testing/test_detection.py
"""

import sys
import time
from pathlib import Path

# Ensure project root is on PYTHONPATH
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
from src.inference.face_detector import detect_faces, get_detector
from src.utils.logger import log

_REF_DIR    = _PROJECT_ROOT / "data" / "reference_images"
_TEST_DIR   = _PROJECT_ROOT / "data" / "test_inputs"
_RESULT_DIR = _PROJECT_ROOT / "testing" / "results" / "detection"
_RESULT_DIR.mkdir(parents=True, exist_ok=True)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _annotate_detections(frame: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Run detection, draw bounding boxes on a copy of the frame, return (annotated, crops)."""
    detector = get_detector()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if detector._use_insightface:
        faces = detector.app.get(rgb)
        annotated = frame.copy()
        crops = []
        for face in faces:
            box = face.bbox.astype(int)
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            score = face.det_score
            cv2.putText(annotated, f"{score:.2f}", (box[0], box[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            from insightface.utils import face_align
            kps = face.kps.astype(np.float32)
            aligned = face_align.norm_crop(rgb, kps, image_size=112)
            crops.append(cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
        return annotated, crops
    else:
        crops = detect_faces(frame)
        return frame, crops


def test_reference_images():
    """Test detection on single-face reference images."""
    print("\n" + "=" * 60)
    print("  FACE DETECTION TEST — Reference Images (1 face expected)")
    print("=" * 60)

    if not _REF_DIR.exists():
        print(f"ERROR: Reference dir not found: {_REF_DIR}")
        return

    img_paths = sorted([p for p in _REF_DIR.iterdir() if p.suffix.lower() in _IMG_EXTS])
    total = len(img_paths)
    detected = 0
    failed = []
    times = []

    for p in img_paths:
        frame = cv2.imread(str(p))
        if frame is None:
            failed.append((p.name, "unreadable"))
            continue

        t0 = time.perf_counter()
        annotated, crops = _annotate_detections(frame)
        dt = time.perf_counter() - t0
        times.append(dt)

        n_faces = len(crops)
        if n_faces >= 1:
            detected += 1
            # save the first crop
            cv2.imwrite(str(_RESULT_DIR / f"ref_crop_{p.stem}.jpg"), crops[0])
        else:
            failed.append((p.name, "no_face"))

        cv2.imwrite(str(_RESULT_DIR / f"ref_annotated_{p.stem}.jpg"), annotated)

    avg_ms = (sum(times) / len(times) * 1000) if times else 0

    print(f"\n  Results:")
    print(f"    Total images:     {total}")
    print(f"    Detected (≥1):    {detected}  ({detected/total*100:.1f}%)")
    print(f"    Failed:           {len(failed)}")
    print(f"    Avg latency:      {avg_ms:.1f} ms / image")

    if failed:
        print(f"\n  Failed images:")
        for name, reason in failed:
            print(f"    ✗ {name} — {reason}")

    print(f"\n  Annotated images saved to: {_RESULT_DIR}")
    return detected, total, failed


def test_group_images():
    """Test detection on group/classroom images."""
    print("\n" + "=" * 60)
    print("  FACE DETECTION TEST — Group / Test Images")
    print("=" * 60)

    if not _TEST_DIR.exists():
        print(f"ERROR: Test dir not found: {_TEST_DIR}")
        return

    img_paths = sorted([p for p in _TEST_DIR.iterdir() if p.suffix.lower() in _IMG_EXTS])
    results = []

    for p in img_paths:
        frame = cv2.imread(str(p))
        if frame is None:
            continue

        t0 = time.perf_counter()
        annotated, crops = _annotate_detections(frame)
        dt = time.perf_counter() - t0

        n_faces = len(crops)
        results.append((p.name, n_faces, dt * 1000))

        cv2.imwrite(str(_RESULT_DIR / f"test_annotated_{p.stem}.jpg"), annotated)

    print(f"\n  {'Image':<55} {'Faces':>6} {'Time (ms)':>10}")
    print(f"  {'─'*55} {'─'*6} {'─'*10}")
    for name, n, ms in results:
        short = name[:52] + "..." if len(name) > 55 else name
        print(f"  {short:<55} {n:>6} {ms:>10.1f}")

    total_faces = sum(r[1] for r in results)
    print(f"\n  Total faces detected across all test images: {total_faces}")
    print(f"  Annotated images saved to: {_RESULT_DIR}")
    return results


if __name__ == "__main__":
    test_reference_images()
    test_group_images()
    print("\n✓ Detection testing complete.\n")
