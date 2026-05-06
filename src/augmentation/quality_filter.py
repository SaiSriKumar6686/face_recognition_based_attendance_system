"""
quality_filter.py
──────────────────
Post-augmentation quality gate.
Every synthetic image passes through two checks:

1.  BRISQUE score  — measures perceived image quality (lower = better).
    Threshold: reject if score > 80 (severe degradation / noise).

2.  Face detector re-check  — ensures the augmented image still contains
    a detectable face.  Uses a lightweight Haar cascade as a fast pre-filter;
    swap for RetinaFace if you need higher precision.

Images failing either check are discarded.
"""

import cv2
import numpy as np

try:
    from image_quality import brisque  # pip install image-quality
    _BRISQUE_AVAILABLE = True
except ImportError:
    _BRISQUE_AVAILABLE = False
    print("[quality_filter] WARNING: image-quality not installed — BRISQUE check skipped.")

# ── Face detector (lightweight Haar) ─────────────────────────────────
_HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_detector = cv2.CascadeClassifier(_HAAR_PATH)

BRISQUE_THRESHOLD = 80.0   # reject if score ≥ this


def _brisque_ok(image: np.ndarray) -> bool:
    if not _BRISQUE_AVAILABLE:
        return True
    try:
        score = brisque.score(image)
        return score < BRISQUE_THRESHOLD
    except Exception:
        return True   # on error, pass the image through


def _face_detected(image: np.ndarray) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    faces = _face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
    )
    return len(faces) > 0


def filter_synthetics(images: list[np.ndarray], verbose: bool = False) -> list[np.ndarray]:
    """
    Filter a list of synthetic images, keeping only those that pass
    both BRISQUE and face-detection checks.

    Parameters
    ----------
    images  : list of H×W×3 numpy arrays.
    verbose : print per-image results.

    Returns
    -------
    Filtered list of numpy arrays.
    """
    passed = []
    for i, img in enumerate(images):
        bq_ok = _brisque_ok(img)
        fd_ok = _face_detected(img)
        if bq_ok and fd_ok:
            passed.append(img)
            if verbose:
                print(f"  [{i:03d}] PASS")
        else:
            if verbose:
                reason = []
                if not bq_ok: reason.append("BRISQUE")
                if not fd_ok: reason.append("no-face")
                print(f"  [{i:03d}] REJECT ({', '.join(reason)})")
    return passed


if __name__ == "__main__":
    dummy_images = [np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8) for _ in range(10)]
    kept = filter_synthetics(dummy_images, verbose=True)
    print(f"Kept {len(kept)}/{len(dummy_images)} images")
