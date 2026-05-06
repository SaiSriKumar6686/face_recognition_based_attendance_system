"""
embedder.py
───────────
Face embedding extractor using InsightFace's buffalo_l recognition model
(ResNet-50 trained on WebFace600K — w600k_r50.onnx).

Produces 512-dimensional L2-normalised embeddings with genuine
discriminative power for identity separation.

Falls back to ONNX AdaFace model if InsightFace is not available.

Usage
-----
    from src.inference.embedder import get_embedder
    emb = get_embedder()
    vector = emb.embed(crop_bgr)   # → np.ndarray shape (512,)
"""

from pathlib import Path

import cv2
import numpy as np

from src.utils.config_loader import cfg
from src.utils.logger import log

_ONNX_DIR   = Path(cfg["paths"]["onnx"])
_INPUT_SIZE = tuple(cfg["inference"]["input_size"])   # (112, 112)

# ── Image preprocessing for quality enhancement ─────────────────────

def _preprocess_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Enhance a face crop before embedding extraction.
    Applies CLAHE + bilateral denoise + unsharp mask.
    """
    result = crop_bgr.copy()

    # CLAHE on luminance channel
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Bilateral denoise (preserve edges)
    result = cv2.bilateralFilter(result, 5, 40, 40)

    # Unsharp mask
    blurred = cv2.GaussianBlur(result, (0, 0), 1.0)
    result = cv2.addWeighted(result, 2.0, blurred, -1.0, 0)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


# ── InsightFace-based Embedder (primary) ─────────────────────────────

try:
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
    _INSIGHTFACE_OK = True
except ImportError:
    _INSIGHTFACE_OK = False


class InsightFaceEmbedder:
    """
    Uses InsightFace buffalo_l (ResNet-50, WebFace600K) for production-grade
    512-d face embeddings.
    """

    def __init__(self, model_name: str = "buffalo_l"):
        if not _INSIGHTFACE_OK:
            raise ImportError("insightface is required: pip install insightface")

        self.app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self._model_name = model_name
        log.info(f"Embedder: InsightFace {model_name} loaded (detection + recognition)")

    def embed_from_frame(self, frame_bgr: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Detect + embed all faces in a full frame.

        Returns list of (aligned_crop_bgr, embedding_512d) tuples.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb)

        results = []
        for face in faces:
            kps = face.kps.astype(np.float32)
            aligned = face_align.norm_crop(rgb, kps, image_size=112)
            crop_bgr = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)

            emb = face.embedding
            norm = np.linalg.norm(emb)
            emb = (emb / (norm + 1e-8)).astype(np.float32)

            results.append((crop_bgr, emb))
        return results

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Extract L2-normalised 512-d embedding from a single face crop.
        Adds padding to help detection on tight crops, then preprocesses.
        """
        enhanced = _preprocess_crop(crop_bgr)

        # Pad the crop to help detection
        h, w = enhanced.shape[:2]
        pad = int(0.15 * max(h, w))
        padded = cv2.copyMakeBorder(enhanced, pad, pad, pad, pad,
                                     cv2.BORDER_REFLECT_101)

        results = self.embed_from_frame(padded)
        if results:
            return results[0][1]

        # Fallback without padding
        results = self.embed_from_frame(enhanced)
        if results:
            return results[0][1]

        log.warning("Embedder: no face detected in crop — returning zero vector")
        return np.zeros(512, dtype=np.float32)

    def embed_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        """Embed a list of pre-cropped faces."""
        return np.array([self.embed(c) for c in crops], dtype=np.float32)


# ── ONNX-based Embedder (fallback for custom-trained AdaFace) ────────

class OnnxEmbedder:
    """
    Wraps a custom ONNX model (e.g. trained AdaFace IR-18) for inference.
    Use this after training your own model via train_bootstrap.py + export_onnx.py.
    """

    def __init__(self, model_path: str | Path = None):
        import onnxruntime as ort

        if model_path is None:
            model_path = self._find_latest_onnx()

        log.info(f"Embedder: loading ONNX model from {model_path}")
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @staticmethod
    def _find_latest_onnx() -> Path:
        onnx_files = sorted(_ONNX_DIR.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(
                f"No ONNX model found in {_ONNX_DIR}. "
                "Export the trained model first via scripts/export_onnx.py"
            )
        return onnx_files[-1]

    def _to_tensor(self, crop_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(crop_bgr, _INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img / 255.0 - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        enhanced = _preprocess_crop(crop_bgr)
        tensor = self._to_tensor(enhanced)
        output = self.session.run([self.output_name], {self.input_name: tensor})
        emb = output[0][0]
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-8)

    def embed_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        tensors = np.concatenate([self._to_tensor(_preprocess_crop(c)) for c in crops], axis=0)
        output = self.session.run([self.output_name], {self.input_name: tensors})
        embs = output[0]
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / (norms + 1e-8)


# ── Singleton helper ──────────────────────────────────────────────────
# Uses InsightFace by default; falls back to ONNX if unavailable.

_embedder_singleton = None


def get_embedder():
    global _embedder_singleton
    if _embedder_singleton is None:
        if _INSIGHTFACE_OK:
            try:
                _embedder_singleton = InsightFaceEmbedder("buffalo_l")
            except Exception as e:
                log.warning(f"buffalo_l failed ({e}), trying buffalo_sc...")
                try:
                    _embedder_singleton = InsightFaceEmbedder("buffalo_sc")
                except Exception:
                    log.warning("InsightFace failed — falling back to ONNX embedder")
                    _embedder_singleton = OnnxEmbedder()
        else:
            _embedder_singleton = OnnxEmbedder()
    return _embedder_singleton
