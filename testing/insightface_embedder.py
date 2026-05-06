"""
insightface_embedder.py
────────────────────────
Drop-in InsightFace recognition embedder for testing.

Uses the buffalo_sc recognition model (w600k_mbf.onnx — MobileFaceNet
trained on WebFace600K) that is already downloaded on this machine.
This replaces the placeholder ONNX backbone with a real, production-grade
face recognition model producing genuinely discriminative 512-d embeddings.

This file lives in testing/ so the main codebase remains untouched.
When you're ready to integrate, you can port this logic into
src/inference/embedder.py.

Usage:
    from testing.insightface_embedder import InsightFaceEmbedder
    emb = InsightFaceEmbedder()
    vector = emb.embed(crop_bgr)   # → np.ndarray shape (512,)
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
    _INSIGHTFACE_OK = True
except ImportError:
    _INSIGHTFACE_OK = False

from src.utils.logger import log


class InsightFaceEmbedder:
    """
    Uses InsightFace's built-in recognition model for embedding extraction.
    
    The buffalo_sc pack includes:
      - det_500m.onnx  → face detection (RetinaFace)
      - w600k_mbf.onnx → face recognition (MobileFaceNet, WebFace600K)
    
    By enabling the 'recognition' module, we get real 512-d embeddings
    that are properly trained to separate identities.
    """

    def __init__(self, model_name: str = "buffalo_sc"):
        if not _INSIGHTFACE_OK:
            raise ImportError("insightface is required. pip install insightface")

        self.app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        log.info(f"InsightFaceEmbedder: loaded {model_name} (detection + recognition)")

    def embed_from_frame(self, frame_bgr: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Detect + embed all faces in a full frame.

        Returns
        -------
        List of (aligned_crop_bgr, embedding_512d) tuples.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb)

        results = []
        for face in faces:
            kps = face.kps.astype(np.float32)
            aligned = face_align.norm_crop(rgb, kps, image_size=112)
            crop_bgr = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)

            emb = face.embedding  # 512-d from w600k_mbf
            norm = np.linalg.norm(emb)
            emb = emb / (norm + 1e-8)

            results.append((crop_bgr, emb.astype(np.float32)))

        return results

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Extract embedding from a pre-cropped 112×112 face.
        
        NOTE: For best results, pass the original full frame to
        embed_from_frame() instead. This method re-detects the face
        in the crop, which can fail if the crop is too tight.
        """
        # Pad the crop slightly to help detection on tight crops
        h, w = crop_bgr.shape[:2]
        pad = int(0.15 * max(h, w))
        padded = cv2.copyMakeBorder(crop_bgr, pad, pad, pad, pad,
                                     cv2.BORDER_REFLECT_101)

        results = self.embed_from_frame(padded)
        if results:
            return results[0][1]

        # Fallback: if detection fails on tight crop, try without padding
        results = self.embed_from_frame(crop_bgr)
        if results:
            return results[0][1]

        log.warning("InsightFaceEmbedder: no face detected in crop, returning zeros")
        return np.zeros(512, dtype=np.float32)

    def embed_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        """Embed a list of pre-cropped faces."""
        embeddings = [self.embed(c) for c in crops]
        return np.array(embeddings, dtype=np.float32)


# ── Singleton helper ──────────────────────────────────────────────────

_singleton = None


def get_insightface_embedder() -> InsightFaceEmbedder:
    global _singleton
    if _singleton is None:
        _singleton = InsightFaceEmbedder()
    return _singleton


if __name__ == "__main__":
    import time
    emb = InsightFaceEmbedder()

    test_img = _PROJECT_ROOT / "data" / "reference_images" / "23C11A0565.jpeg"
    frame = cv2.imread(str(test_img))
    if frame is not None:
        t0 = time.perf_counter()
        results = emb.embed_from_frame(frame)
        dt = time.perf_counter() - t0
        print(f"Detected {len(results)} face(s) in {dt*1000:.1f} ms")
        if results:
            crop, vec = results[0]
            print(f"Embedding shape: {vec.shape}, norm: {np.linalg.norm(vec):.4f}")
    else:
        print(f"Could not read: {test_img}")
