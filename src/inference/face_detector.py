"""
face_detector.py
─────────────────
RetinaFace-based face detection + alignment.

Pipeline per CCTV frame:
  Frame → RetinaFace detect → 5-point landmark align → 112×112 crop

Alignment follows the standard ArcFace / AdaFace protocol:
align to mean face landmarks using similarity transform.

Dependencies:
    pip install insightface onnxruntime

RetinaFace model is downloaded automatically by insightface on first run
and cached in ~/.insightface/models/.
"""

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
    _INSIGHTFACE_OK = True
except ImportError:
    _INSIGHTFACE_OK = False
    print("[face_detector] insightface not installed — using fallback Haar detector.")

from src.utils.config_loader import cfg
from src.utils.logger import log

_INFER_CFG   = cfg["inference"]
_INPUT_SIZE  = tuple(_INFER_CFG["input_size"])   # (112, 112)
_DETECTOR_BE = _INFER_CFG["face_detector"]       # "retinaface"


class FaceDetector:
    """
    Wrapper around InsightFace RetinaFace detection + ArcFace alignment.
    Falls back to OpenCV Haar cascade if InsightFace is not installed.
    """

    def __init__(self):
        if _INSIGHTFACE_OK:
            self.app = FaceAnalysis(
                name="buffalo_sc",          # lightweight model pack
                allowed_modules=["detection"],
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self._use_insightface = True
            log.info("FaceDetector: InsightFace RetinaFace loaded.")
        else:
            _haar = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(_haar)
            self._use_insightface = False
            log.warning("FaceDetector: Using fallback Haar cascade (lower accuracy).")

    def detect_and_align(self, frame: np.ndarray) -> list[np.ndarray]:
        """
        Detect all faces in `frame` and return aligned 112×112 crops.

        Parameters
        ----------
        frame : H×W×3 BGR numpy array (raw CCTV frame).

        Returns
        -------
        List of aligned face crops (112×112×3 BGR).
        Empty list if no faces detected.
        """
        if self._use_insightface:
            return self._detect_insightface(frame)
        return self._detect_haar(frame)

    def _detect_insightface(self, frame: np.ndarray) -> list[np.ndarray]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.app.get(rgb)
        crops = []
        for face in faces:
            kps = face.kps.astype(np.float32)    # 5-point landmarks
            aligned = face_align.norm_crop(rgb, kps, image_size=_INPUT_SIZE[0])
            crops.append(cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR))
        return crops

    def _detect_haar(self, frame: np.ndarray) -> list[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self._haar.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        crops = []
        h_frame, w_frame = frame.shape[:2]
        for (x, y, w, h) in detections:
            # add small padding
            pad = int(0.1 * min(w, h))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_frame, x + w + pad)
            y2 = min(h_frame, y + h + pad)
            crop = frame[y1:y2, x1:x2]
            crop = cv2.resize(crop, _INPUT_SIZE)
            crops.append(crop)
        return crops


# ── Convenience function ──────────────────────────────────────────────

_detector_singleton: FaceDetector = None


def get_detector() -> FaceDetector:
    global _detector_singleton
    if _detector_singleton is None:
        _detector_singleton = FaceDetector()
    return _detector_singleton


def detect_faces(frame: np.ndarray) -> list[np.ndarray]:
    """Module-level shortcut: detect + align faces in one call."""
    return get_detector().detect_and_align(frame)


if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    if ret:
        crops = detect_faces(frame)
        print(f"Detected {len(crops)} face(s)")
    cam.release()
