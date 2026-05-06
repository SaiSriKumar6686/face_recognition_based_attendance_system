"""
rebuild_buffalo_l.py
─────────────────────
Rebuild FAISS index using InsightFace buffalo_l (ResNet-50 backbone)
which is significantly more accurate than buffalo_sc (MobileFaceNet).

Also applies image preprocessing to reference crops before enrollment,
and tests group image recognition with the improved pipeline.

Usage:
    python testing/rebuild_buffalo_l.py
"""

import sys
import time
from pathlib import Path
from collections import defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
import faiss

from insightface.app import FaceAnalysis
from insightface.utils import face_align

from src.utils.config_loader import cfg
from src.utils.logger import log
from testing.preprocess import preprocess_crop

_REF_DIR    = _PROJECT_ROOT / "data" / "reference_images"
_TEST_DIR   = _PROJECT_ROOT / "data" / "test_inputs"
_EMB_DIR    = _PROJECT_ROOT / "data" / "embeddings"
_RESULT_DIR = _PROJECT_ROOT / "testing" / "results" / "buffalo_l"
_RESULT_DIR.mkdir(parents=True, exist_ok=True)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

EMB_DIM     = 512
HIGH_THRESH = cfg["inference"]["high_confidence_threshold"]
SOFT_THRESH = cfg["inference"]["soft_threshold"]


class BuffaloLEmbedder:
    """InsightFace buffalo_l (ResNet-50, WebFace600K) embedder."""

    def __init__(self):
        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        log.info("BuffaloL embedder loaded (w600k_r50)")

    def embed_from_frame(self, frame_bgr, preprocess=False):
        """Detect + embed all faces in a frame."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb)

        results = []
        for face in faces:
            kps = face.kps.astype(np.float32)
            aligned = face_align.norm_crop(rgb, kps, image_size=112)
            crop_bgr = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)

            if preprocess:
                crop_bgr = preprocess_crop(crop_bgr)

            emb = face.embedding
            norm = np.linalg.norm(emb)
            emb = (emb / (norm + 1e-8)).astype(np.float32)

            results.append((crop_bgr, emb, face))
        return results


def main():
    t_start = time.time()
    embedder = BuffaloLEmbedder()

    # ── Step 1: Build gallery ─────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 1: Building Gallery with buffalo_l (ResNet-50)")
    print("=" * 70)

    img_paths = sorted([p for p in _REF_DIR.iterdir() if p.suffix.lower() in _IMG_EXTS])
    index = faiss.IndexFlatIP(EMB_DIM)
    id_map = {}
    enrolled = 0

    for p in img_paths:
        student_id = p.stem
        frame = cv2.imread(str(p))
        if frame is None:
            continue

        results = embedder.embed_from_frame(frame, preprocess=True)
        if not results:
            print(f"  ✗ {student_id}: no face detected")
            continue

        crop, emb, face = max(results, key=lambda r: r[0].shape[0] * r[0].shape[1])
        vec = emb.reshape(1, -1)
        id_map[index.ntotal] = student_id
        index.add(vec)
        enrolled += 1

        # Save preprocessed crop
        cv2.imwrite(str(_RESULT_DIR / f"enrolled_{student_id}.jpg"), crop)

    faiss.write_index(index, str(_EMB_DIR / "faiss_buffalo_l.index"))
    np.save(str(_EMB_DIR / "id_map_buffalo_l.npy"),
            np.array([id_map[i] for i in range(len(id_map))]))
    print(f"\n  Enrolled: {enrolled}/{len(img_paths)} students")

    # ── Step 2: Gallery Rank-1 ────────────────────────
    print("\n" + "=" * 70)
    print("  STEP 2: Rank-1 Gallery Accuracy (buffalo_l)")
    print("=" * 70)

    correct = 0
    total = 0
    confs = []

    for p in img_paths:
        gt_id = p.stem
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        total += 1

        results = embedder.embed_from_frame(frame)
        if not results:
            continue

        crop, emb, face = max(results, key=lambda r: r[0].shape[0] * r[0].shape[1])
        vec = emb.reshape(1, -1)
        scores, indices = index.search(vec, k=1)
        pred_id = id_map.get(int(indices[0][0]))

        if pred_id == gt_id:
            correct += 1
            confs.append(float(scores[0][0]))

    rank1 = correct / total * 100 if total else 0
    print(f"\n  Rank-1: {correct}/{total} ({rank1:.1f}%)")
    if confs:
        print(f"  Confidence: mean={np.mean(confs):.4f}, min={np.min(confs):.4f}")

    # ── Step 3: Group image recognition ───────────────
    print("\n" + "=" * 70)
    print("  STEP 3: Group Image Recognition (buffalo_l)")
    print("=" * 70)

    test_paths = sorted([p for p in _TEST_DIR.iterdir() if p.suffix.lower() in _IMG_EXTS])
    overall = defaultdict(int)

    for p in test_paths:
        frame = cv2.imread(str(p))
        if frame is None:
            continue

        t0 = time.perf_counter()
        results = embedder.embed_from_frame(frame)
        dt = time.perf_counter() - t0

        if not results:
            print(f"\n  {p.name}: No faces")
            continue

        matches = []
        annotated = frame.copy()

        for crop, emb, face in results:
            vec = emb.reshape(1, -1)
            scores, indices = index.search(vec, k=1)
            score = float(scores[0][0])
            idx = int(indices[0][0])
            pred_id = id_map.get(idx)

            if score >= HIGH_THRESH:
                decision = "high"
            elif score >= SOFT_THRESH:
                decision = "soft"
            else:
                decision = "unknown"
                pred_id = None

            matches.append((pred_id, score, decision))
            overall[decision] += 1

            # Annotate image
            box = face.bbox.astype(int)
            color = (0,255,0) if decision == "high" else (0,255,255) if decision == "soft" else (0,0,255)
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, 2)
            label = f"{pred_id or '?'} {score:.2f}"
            cv2.putText(annotated, label, (box[0], box[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        cv2.imwrite(str(_RESULT_DIR / f"group_{p.stem}.jpg"), annotated)

        identified = set(m[0] for m in matches if m[2] == "high" and m[0])
        high = sum(1 for m in matches if m[2] == "high")
        soft = sum(1 for m in matches if m[2] == "soft")
        unk  = sum(1 for m in matches if m[2] == "unknown")

        print(f"\n  {p.name} ({len(results)} faces, {dt*1000:.0f}ms):")
        print(f"    High: {high} | Soft: {soft} | Unknown: {unk} | Unique IDs: {len(identified)}")

        if len(matches) <= 15:
            for pid, conf, dec in sorted(matches, key=lambda x: -x[1]):
                sym = "✓" if dec == "high" else "?" if dec == "soft" else "✗"
                print(f"      {sym} {pid or 'unknown':<18} {conf:.4f}")

    print(f"\n  ─── OVERALL ───")
    print(f"  High: {overall['high']} | Soft: {overall['soft']} | Unknown: {overall['unknown']}")
    total_faces = sum(overall.values())
    if total_faces:
        print(f"  High-confidence rate: {overall['high']/total_faces*100:.1f}%")

    elapsed = time.time() - t_start
    print(f"\n  Annotated images: {_RESULT_DIR}")
    print(f"  Total time: {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
