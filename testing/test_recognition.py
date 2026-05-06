"""
test_recognition.py
────────────────────
Script 2: Face Recognition Accuracy Benchmark

Two evaluation modes:

A) Reference-vs-Reference (Closed-Set Identification)
   For every reference image, extract embedding → query the FAISS index
   → check if the top-1 match equals the ground-truth student ID.
   This measures Rank-1 identification accuracy.

B) Group Image Recognition
   For every test group image, detect all faces → match each against
   the enrolled gallery → report who was identified and at what confidence.

Both modes use the currently active model (ONNX placeholder or AdaFace).
This lets you directly compare accuracy before vs after real training.

Metrics reported:
  • Rank-1 accuracy (closed-set)
  • Mean / median / min confidence on correct matches
  • Confusion pairs (top wrong-match cases)
  • Per-group identification results

Usage:
    python testing/test_recognition.py
    python testing/test_recognition.py --mode gallery   # ref-vs-ref only
    python testing/test_recognition.py --mode group     # group images only
"""

import sys
import argparse
import time
from pathlib import Path
from collections import defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
from src.inference.face_detector import detect_faces
from src.inference.embedder import get_embedder
from src.inference.matcher import get_matcher
from src.utils.logger import log

_REF_DIR    = _PROJECT_ROOT / "data" / "reference_images"
_TEST_DIR   = _PROJECT_ROOT / "data" / "test_inputs"
_RESULT_DIR = _PROJECT_ROOT / "testing" / "results" / "recognition"
_RESULT_DIR.mkdir(parents=True, exist_ok=True)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def test_gallery_accuracy():
    """
    Closed-set Rank-1 identification test.
    Each reference photo is treated as a probe: detect face → embed → match.
    Ground truth = filename stem (e.g. 23C11A0565.jpeg → student_id=23C11A0565).
    """
    print("\n" + "=" * 70)
    print("  RECOGNITION TEST — Gallery Rank-1 Identification (Reference Images)")
    print("=" * 70)

    embedder = get_embedder()
    matcher  = get_matcher()

    img_paths = sorted([p for p in _REF_DIR.iterdir() if p.suffix.lower() in _IMG_EXTS])
    total = len(img_paths)

    correct = 0
    wrong_matches = []
    no_face = []
    confidences_correct = []
    confidences_wrong = []

    print(f"\n  Testing {total} reference images against FAISS index...\n")

    for p in img_paths:
        gt_id = p.stem   # ground-truth student ID
        frame = cv2.imread(str(p))
        if frame is None:
            no_face.append((gt_id, "unreadable"))
            continue

        crops = detect_faces(frame)
        if not crops:
            no_face.append((gt_id, "no_face_detected"))
            continue

        # Take the largest crop (should be only face in reference image)
        crop = max(crops, key=lambda c: c.shape[0] * c.shape[1])
        emb = embedder.embed(crop)
        pred_id, confidence, decision = matcher.match(emb)

        if pred_id == gt_id:
            correct += 1
            confidences_correct.append(confidence)
        else:
            wrong_matches.append((gt_id, pred_id, confidence, decision))
            confidences_wrong.append(confidence)

    rank1 = correct / total * 100 if total else 0

    print(f"  ┌─────────────────────────────────────────────────┐")
    print(f"  │  Rank-1 Accuracy:  {correct}/{total}  ({rank1:.1f}%)             │")
    print(f"  └─────────────────────────────────────────────────┘")

    if confidences_correct:
        print(f"\n  Correct Match Confidence Distribution:")
        print(f"    Mean:   {np.mean(confidences_correct):.4f}")
        print(f"    Median: {np.median(confidences_correct):.4f}")
        print(f"    Min:    {np.min(confidences_correct):.4f}")
        print(f"    Max:    {np.max(confidences_correct):.4f}")

    if wrong_matches:
        print(f"\n  ✗ Wrong Matches ({len(wrong_matches)}):")
        print(f"    {'Ground Truth':<18} {'Predicted':<18} {'Conf':>8} {'Decision':>10}")
        print(f"    {'─'*18} {'─'*18} {'─'*8} {'─'*10}")
        for gt, pred, conf, dec in wrong_matches[:20]:
            print(f"    {gt:<18} {pred or 'None':<18} {conf:>8.4f} {dec:>10}")

    if no_face:
        print(f"\n  ✗ No Face Detected ({len(no_face)}):")
        for sid, reason in no_face:
            print(f"    {sid} — {reason}")

    # Save detailed results to file
    with open(_RESULT_DIR / "gallery_results.txt", "w") as f:
        f.write(f"Rank-1 Accuracy: {correct}/{total} ({rank1:.1f}%)\n\n")
        f.write("Wrong matches:\n")
        for gt, pred, conf, dec in wrong_matches:
            f.write(f"  {gt} -> {pred} (conf={conf:.4f}, decision={dec})\n")
        f.write(f"\nNo face detected:\n")
        for sid, reason in no_face:
            f.write(f"  {sid} — {reason}\n")

    return correct, total, wrong_matches


def test_group_recognition():
    """
    Detect all faces in group images and attempt to identify each.
    """
    print("\n" + "=" * 70)
    print("  RECOGNITION TEST — Group Image Identification")
    print("=" * 70)

    embedder = get_embedder()
    matcher  = get_matcher()

    img_paths = sorted([p for p in _TEST_DIR.iterdir() if p.suffix.lower() in _IMG_EXTS])

    for p in img_paths:
        frame = cv2.imread(str(p))
        if frame is None:
            continue

        t0 = time.perf_counter()
        crops = detect_faces(frame)
        dt_detect = time.perf_counter() - t0

        if not crops:
            print(f"\n  {p.name}: No faces detected.")
            continue

        t1 = time.perf_counter()
        embeddings = embedder.embed_batch(crops)
        dt_embed = time.perf_counter() - t1

        results = []
        high_count, soft_count, unknown_count = 0, 0, 0
        for emb in embeddings:
            pred_id, confidence, decision = matcher.match(emb)
            results.append((pred_id, confidence, decision))
            if decision == "high":
                high_count += 1
            elif decision == "soft":
                soft_count += 1
            else:
                unknown_count += 1

        print(f"\n  {p.name}:")
        print(f"    Faces detected: {len(crops)}")
        print(f"    Detection time: {dt_detect*1000:.1f} ms")
        print(f"    Embedding time: {dt_embed*1000:.1f} ms")
        print(f"    High confidence: {high_count}  |  Soft: {soft_count}  |  Unknown: {unknown_count}")

        # Print per-face details for smaller images
        if len(results) <= 20:
            print(f"    {'Student ID':<18} {'Confidence':>10} {'Decision':>10}")
            print(f"    {'─'*18} {'─'*10} {'─'*10}")
            for pred_id, conf, dec in sorted(results, key=lambda x: -x[1]):
                print(f"    {pred_id or 'unknown':<18} {conf:>10.4f} {dec:>10}")

        # Save annotated image
        annotated = frame.copy()
        from src.inference.face_detector import get_detector
        detector = get_detector()
        if detector._use_insightface:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.app.get(rgb)
            for i, face in enumerate(faces):
                box = face.bbox.astype(int)
                if i < len(results):
                    pred_id, conf, dec = results[i]
                    color = (0, 255, 0) if dec == "high" else (0, 255, 255) if dec == "soft" else (0, 0, 255)
                    cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, 2)
                    label = f"{pred_id or '?'} {conf:.2f}"
                    cv2.putText(annotated, label, (box[0], box[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.imwrite(str(_RESULT_DIR / f"group_annotated_{p.stem}.jpg"), annotated)

    print(f"\n  Annotated group images saved to: {_RESULT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "gallery", "group"], default="all")
    args = parser.parse_args()

    if args.mode in ("all", "gallery"):
        test_gallery_accuracy()
    if args.mode in ("all", "group"):
        test_group_recognition()

    print("\n✓ Recognition testing complete.\n")
