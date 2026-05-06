"""
dataset_builder.py
───────────────────
Phase 1 — Synthetic Dataset Builder

For every student in data/seed_images/<student_id>/ this script:
  1. Detects + aligns face crops from each seed image.
  2. Runs all 4 augmentation branches (geometric, photometric, CCTV sim, occlusion).
  3. Filters outputs with BRISQUE + face-detector quality gate.
  4. Saves final images to data/synthetic/<student_id>/aug_<N>.jpg.

Target: ~200–500 training samples per identity from 5–10 seed photos.

Run this FIRST before any training.

Usage
─────
    python scripts/dataset_builder.py
    python scripts/dataset_builder.py --student_id s001  # single student
    python scripts/dataset_builder.py --target 400       # override target samples
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.utils.config_loader import cfg
from src.utils.logger import log
from src.inference.face_detector import detect_faces
from src.augmentation.geometric    import apply_geometric
from src.augmentation.photometric  import apply_photometric
from src.augmentation.cctv_sim     import apply_cctv_sim
from src.augmentation.occlusion    import apply_occlusion
from src.augmentation.quality_filter import filter_synthetics

_SEED_DIR = Path(cfg["paths"]["seed_images"])
_SYN_DIR  = Path(cfg["paths"]["synthetic"])
_TARGET   = cfg["augmentation"]["target_samples_per_identity"]
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Variants per branch per seed image
# 5 branches × ~10 variants × 8 seeds ≈ 400 before filtering
_VARIANTS_PER_BRANCH = {
    "geometric":   10,
    "photometric": 10,
    "cctv_sim":    12,   # highest ROI — more variants
    "occlusion":    8,
}


def _load_seed_crops(student_dir: Path) -> list[np.ndarray]:
    """Load + detect + align all seed images for one student."""
    crops = []
    img_paths = [p for p in student_dir.iterdir() if p.suffix.lower() in _IMG_EXTS]
    for p in img_paths:
        frame = cv2.imread(str(p))
        if frame is None:
            continue
        detected = detect_faces(frame)
        if detected:
            # take largest crop
            crops.append(max(detected, key=lambda c: c.shape[0] * c.shape[1]))
    return crops


def build_for_student(student_id: str, target: int = _TARGET) -> int:
    """
    Build synthetic dataset for one student.

    Returns number of synthetic images saved.
    """
    student_seed = _SEED_DIR / student_id
    student_syn  = _SYN_DIR  / student_id

    if not student_seed.exists():
        log.warning(f"Seed folder missing: {student_seed}")
        return 0

    student_syn.mkdir(parents=True, exist_ok=True)

    crops = _load_seed_crops(student_seed)
    if not crops:
        log.error(f"No face crops detected for {student_id} — skipping.")
        return 0

    log.info(f"[{student_id}] {len(crops)} seed crop(s) detected.")

    all_synthetics: list[np.ndarray] = []

    for crop in crops:
        # Branch 1 — Geometric (always first for diversity)
        all_synthetics.extend(apply_geometric(crop,    _VARIANTS_PER_BRANCH["geometric"]))

        # Branch 2 — Photometric
        all_synthetics.extend(apply_photometric(crop,  _VARIANTS_PER_BRANCH["photometric"]))

        # Branch 3 — CCTV sim (highest ROI — apply to each seed + its geometric variants)
        geo_variants = apply_geometric(crop, 3)
        for gv in [crop] + geo_variants:
            all_synthetics.extend(apply_cctv_sim(gv,  _VARIANTS_PER_BRANCH["cctv_sim"]))

        # Branch 4 — Occlusion
        all_synthetics.extend(apply_occlusion(crop,   _VARIANTS_PER_BRANCH["occlusion"]))

    log.info(f"[{student_id}] {len(all_synthetics)} raw synthetics → quality filter…")

    # Quality gate
    filtered = filter_synthetics(all_synthetics, verbose=False)
    log.info(f"[{student_id}] {len(filtered)} passed quality filter.")

    # Sub-sample to target
    import random
    if len(filtered) > target:
        filtered = random.sample(filtered, target)

    # Save
    for idx, img in enumerate(filtered):
        out_path = student_syn / f"aug_{idx:04d}.jpg"
        cv2.imwrite(str(out_path), img)

    saved = len(filtered)
    log.info(f"[{student_id}] Saved {saved} synthetic images → {student_syn}")
    return saved


def build_all(target: int = _TARGET) -> None:
    """Build synthetic dataset for every student in seed_images/."""
    if not _SEED_DIR.exists():
        log.error(f"Seed directory not found: {_SEED_DIR}")
        return

    student_dirs = [d for d in sorted(_SEED_DIR.iterdir()) if d.is_dir()]
    if not student_dirs:
        log.error("No student subdirectories found in seed_images/. "
                  "Add photos first: data/seed_images/<student_id>/photo.jpg")
        return

    log.info(f"Building synthetic dataset for {len(student_dirs)} student(s)…")
    total_saved = 0
    for d in student_dirs:
        total_saved += build_for_student(d.name, target)

    log.info(f"═══ Dataset build complete: {total_saved} total synthetic images ═══")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build synthetic augmented dataset.")
    parser.add_argument("--student_id", default=None,
                        help="Build for a single student (default: all)")
    parser.add_argument("--target", type=int, default=_TARGET,
                        help=f"Target samples per identity (default: {_TARGET})")
    args = parser.parse_args()

    if args.student_id:
        build_for_student(args.student_id, args.target)
    else:
        build_all(args.target)
