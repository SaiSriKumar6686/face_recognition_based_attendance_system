"""
live_inference.py
──────────────────
Real-time CCTV inference loop.

Pipeline per frame
──────────────────
CCTV frame
  → FaceDetector (RetinaFace) → aligned 112×112 crop(s)
  → Embedder (ONNX AdaFace)   → 512-d embedding
  → Matcher (FAISS)           → confidence split
        ┌── High confidence ──→ log_attendance() + save crop
        └── Low confidence  ──→ save to buffer for admin review

Runs in a background thread so the Flask web server stays responsive.
"""

import time
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.utils.config_loader import cfg
from src.utils.logger import log
from src.utils.db import log_attendance, add_crop_to_buffer, init_db
from src.inference.face_detector import detect_faces
from src.inference.embedder import get_embedder
from src.inference.matcher import get_matcher

_RAW_DIR = Path(cfg["paths"]["cctv_crops_raw"])
_RAW_DIR.mkdir(parents=True, exist_ok=True)

_HIGH_THRESH = cfg["inference"]["high_confidence_threshold"]


class LiveInference:
    """
    Runs the CCTV inference loop.

    Parameters
    ----------
    source : int or str — OpenCV VideoCapture source (0 = default webcam,
             or RTSP URL for CCTV stream).
    fps_limit : max frames to process per second (set low to save CPU).
    """

    def __init__(self, source: int | str = 0, fps_limit: float = 2.0):
        self.source    = source
        self.fps_limit = fps_limit
        self._running  = False
        self._thread: threading.Thread = None

        # lazy-init singletons
        self.embedder = None
        self.matcher  = None

        init_db()

    def start(self) -> None:
        if self._running:
            log.warning("LiveInference already running.")
            return
        self.embedder = get_embedder()
        self.matcher  = get_matcher()
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info(f"LiveInference started (source={self.source}, fps_limit={self.fps_limit})")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        log.info("LiveInference stopped.")

    # ── Main loop ─────────────────────────────────────────────────────

    def _loop(self) -> None:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            log.error(f"Cannot open video source: {self.source}")
            self._running = False
            return
            
        # check if it's a static image
        is_static_image = isinstance(self.source, str) and self.source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        last_frame = None

        frame_interval = 1.0 / self.fps_limit
        while self._running:
            t0 = time.monotonic()
            ret, frame = cap.read()
            
            if not ret:
                if is_static_image and last_frame is not None:
                    frame = last_frame
                else:
                    log.warning("Frame grab failed — retrying in 1s.")
                    time.sleep(1)
                    continue
            else:
                last_frame = frame

            self._process_frame(frame)

            elapsed = time.monotonic() - t0
            sleep_t = max(0, frame_interval - elapsed)
            time.sleep(sleep_t)

        cap.release()

    def _process_frame(self, frame: np.ndarray) -> None:
        crops = detect_faces(frame)
        if not crops:
            return

        embeddings = self.embedder.embed_batch(crops)

        for crop, embedding in zip(crops, embeddings):
            student_id, confidence, decision = self.matcher.match(embedding)

            if decision == "high":
                self._handle_high_confidence(student_id, confidence, crop)
            elif decision == "soft":
                self._handle_low_confidence(student_id, confidence, crop)
            # "unknown" — discard silently

    def _save_crop(self, filename: str, crop: np.ndarray) -> str:
        path = _RAW_DIR / filename
        cv2.imwrite(str(path), crop)
        return str(path)

    def _handle_high_confidence(self, student_id: str, confidence: float,
                                 crop: np.ndarray) -> None:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = f"{student_id}_{ts}.jpg"
        path = self._save_crop(name, crop)
        log_attendance(student_id, confidence, path)
        add_crop_to_buffer(student_id, path, verified=True)
        log.info(f"✓ Attendance: {student_id} (confidence={confidence:.3f})")

    def _handle_low_confidence(self, student_id: str | None, confidence: float,
                                crop: np.ndarray) -> None:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        sid  = student_id or "unknown"
        name = f"review_{sid}_{ts}.jpg"
        path = self._save_crop(name, crop)
        add_crop_to_buffer(sid, path, verified=False)
        log.debug(f"? Low confidence: candidate={sid} ({confidence:.3f}) — queued for review")


if __name__ == "__main__":
    engine = LiveInference(source=0, fps_limit=1.0)
    engine.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        engine.stop()
