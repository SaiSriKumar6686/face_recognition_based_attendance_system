"""
test_e2e_pipeline.py
─────────────────────
Script 4: End-to-End Pipeline Test

Simulates the full production workflow on a batch of test images:
  1. Load image → Detect faces → Embed → Match → Decision
  2. Measures latency at each stage
  3. Generates a comprehensive summary report with:
     • Throughput (faces/sec)
     • Latency breakdown (detection / embedding / matching)
     • Confidence distribution histogram (text-based)
     • Per-image results log

This is the script to run before deployment to verify the entire
pipeline works correctly end-to-end.

Usage:
    python testing/test_e2e_pipeline.py
    python testing/test_e2e_pipeline.py --source data/test_inputs/group1.jpeg
"""

import sys
import argparse
import time
from pathlib import Path
from collections import Counter

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
from src.inference.face_detector import detect_faces
from src.inference.embedder import get_embedder
from src.inference.matcher import get_matcher
from src.utils.logger import log

_TEST_DIR   = _PROJECT_ROOT / "data" / "test_inputs"
_RESULT_DIR = _PROJECT_ROOT / "testing" / "results" / "e2e"
_RESULT_DIR.mkdir(parents=True, exist_ok=True)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _text_histogram(values, bins=10, width=40, title=""):
    """Print a simple text-based histogram."""
    if not values:
        return
    counts, edges = np.histogram(values, bins=bins)
    max_count = max(counts) if max(counts) > 0 else 1
    print(f"\n    {title}")
    for i in range(len(counts)):
        bar_len = int(counts[i] / max_count * width)
        bar = "█" * bar_len
        print(f"    [{edges[i]:>5.2f} - {edges[i+1]:>5.2f}] {bar} {counts[i]}")


def run_e2e_test(source: str = None):
    """Run full pipeline on test images."""
    print("\n" + "=" * 70)
    print("  END-TO-END PIPELINE TEST")
    print("=" * 70)

    embedder = get_embedder()
    matcher  = get_matcher()

    if source and Path(source).is_file():
        img_paths = [Path(source)]
    else:
        img_paths = sorted([p for p in _TEST_DIR.iterdir() if p.suffix.lower() in _IMG_EXTS])

    all_confidences = []
    all_decisions = Counter()
    all_latencies = {"detect": [], "embed": [], "match": []}
    total_faces = 0
    per_image_results = []

    for p in img_paths:
        frame = cv2.imread(str(p))
        if frame is None:
            continue

        # Stage 1: Detection
        t0 = time.perf_counter()
        crops = detect_faces(frame)
        dt_detect = time.perf_counter() - t0
        all_latencies["detect"].append(dt_detect)

        if not crops:
            per_image_results.append({
                "image": p.name, "faces": 0,
                "detect_ms": dt_detect * 1000,
                "results": []
            })
            continue

        # Stage 2: Embedding
        t1 = time.perf_counter()
        embeddings = embedder.embed_batch(crops)
        dt_embed = time.perf_counter() - t1
        all_latencies["embed"].append(dt_embed)

        # Stage 3: Matching
        results = []
        t2 = time.perf_counter()
        for emb in embeddings:
            pred_id, confidence, decision = matcher.match(emb)
            results.append((pred_id, confidence, decision))
            all_confidences.append(confidence)
            all_decisions[decision] += 1
        dt_match = time.perf_counter() - t2
        all_latencies["match"].append(dt_match)

        total_faces += len(crops)

        per_image_results.append({
            "image": p.name,
            "faces": len(crops),
            "detect_ms": dt_detect * 1000,
            "embed_ms": dt_embed * 1000,
            "match_ms": dt_match * 1000,
            "results": results
        })

    # ── Summary Report ────────────────────────────────
    print(f"\n  {'Image':<50} {'Faces':>6} {'Det(ms)':>8} {'Emb(ms)':>8} {'Match(ms)':>9}")
    print(f"  {'─'*50} {'─'*6} {'─'*8} {'─'*8} {'─'*9}")
    for r in per_image_results:
        short = r["image"][:47] + "..." if len(r["image"]) > 50 else r["image"]
        print(f"  {short:<50} {r['faces']:>6} {r['detect_ms']:>8.1f} "
              f"{r.get('embed_ms', 0):>8.1f} {r.get('match_ms', 0):>9.1f}")

    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │  PIPELINE SUMMARY                               │")
    print(f"  ├─────────────────────────────────────────────────┤")
    print(f"  │  Images processed:  {len(img_paths):<28}│")
    print(f"  │  Total faces:       {total_faces:<28}│")
    print(f"  │  High confidence:   {all_decisions.get('high', 0):<28}│")
    print(f"  │  Soft (review):     {all_decisions.get('soft', 0):<28}│")
    print(f"  │  Unknown:           {all_decisions.get('unknown', 0):<28}│")
    print(f"  └─────────────────────────────────────────────────┘")

    if all_latencies["detect"]:
        avg_det  = np.mean(all_latencies["detect"]) * 1000
        avg_emb  = np.mean(all_latencies["embed"]) * 1000 if all_latencies["embed"] else 0
        avg_mat  = np.mean(all_latencies["match"]) * 1000 if all_latencies["match"] else 0
        total_t  = sum(sum(v) for v in all_latencies.values())
        fps = total_faces / total_t if total_t > 0 else 0

        print(f"\n  Latency Breakdown (avg per image):")
        print(f"    Detection:   {avg_det:.1f} ms")
        print(f"    Embedding:   {avg_emb:.1f} ms")
        print(f"    Matching:    {avg_mat:.1f} ms")
        print(f"    Throughput:  {fps:.1f} faces/sec")

    if all_confidences:
        _text_histogram(all_confidences, bins=10, width=30,
                       title="Confidence Score Distribution:")

    # Save full report
    report_path = _RESULT_DIR / "e2e_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("End-to-End Pipeline Test Report\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Images: {len(img_paths)} | Faces: {total_faces}\n")
        f.write(f"High: {all_decisions.get('high',0)} | Soft: {all_decisions.get('soft',0)} | Unknown: {all_decisions.get('unknown',0)}\n\n")
        for r in per_image_results:
            f.write(f"\n--- {r['image']} ({r['faces']} faces) ---\n")
            for pred_id, conf, dec in r.get("results", []):
                f.write(f"  {pred_id or 'unknown':<18} conf={conf:.4f}  [{dec}]\n")
    log.info(f"Full report saved: {report_path}")

    print(f"\n  Reports saved to: {_RESULT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="Specific image to test")
    args = parser.parse_args()

    run_e2e_test(source=args.source)
    print("\n✓ E2E pipeline test complete.\n")
