"""
matcher.py
──────────
FAISS flat index for fast nearest-neighbour identity matching.

Workflow
--------
1.  On startup: load FAISS index + SQLite student map.
2.  Per-frame: embed face crop → query index → confidence gate.
3.  High confidence (≥ threshold)  → mark attendance, save crop.
4.  Low confidence  (≥ soft)       → flag for manual review.
5.  Below soft threshold           → unknown face, skip.

FAISS index type: IndexFlatIP (inner product on L2-normalised vectors = cosine sim).
"""

from pathlib import Path

import faiss
import numpy as np

from src.utils.config_loader import cfg
from src.utils.logger import log
from src.utils.db import get_session, Student

_EMB_DIR    = Path(cfg["paths"]["embeddings"])
_INDEX_PATH = cfg["database"]["faiss_index"]
_DB_PATH    = cfg["database"]["path"]
_INFER_CFG  = cfg["inference"]

HIGH_THRESH = _INFER_CFG["high_confidence_threshold"]
SOFT_THRESH = _INFER_CFG["soft_threshold"]
EMB_DIM     = cfg["training"]["embedding_dim"]   # 512


class Matcher:
    """
    FAISS-backed cosine similarity matcher.
    Maps embedding → student_id with confidence score.
    """

    def __init__(self):
        self.index: faiss.IndexFlatIP = None
        self.id_to_student: dict[int, str] = {}   # faiss index pos → student_id
        self._load_or_create_index()

    # ── Index management ─────────────────────────────────────────────

    def _load_or_create_index(self) -> None:
        index_path = Path(_INDEX_PATH)
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
            log.info(f"FAISS index loaded: {self.index.ntotal} vectors")
        else:
            self.index = faiss.IndexFlatIP(EMB_DIM)
            log.info("New FAISS index created (empty).")
        self._rebuild_id_map()

    def _rebuild_id_map(self) -> None:
        """Rebuild mapping from FAISS positional index → student_id from DB."""
        # Convention: we store embeddings in insertion order matching DB student rows
        # sorted by enrolled_at.  For robustness, store the map in a separate .npy file.
        map_path = _EMB_DIR / "id_map.npy"
        if map_path.exists():
            arr = np.load(str(map_path), allow_pickle=True)
            self.id_to_student = {int(k): v for k, v in enumerate(arr)}
        else:
            self.id_to_student = {}

    def save(self) -> None:
        Path(_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, _INDEX_PATH)
        map_path = _EMB_DIR / "id_map.npy"
        arr = np.array([self.id_to_student[i] for i in range(len(self.id_to_student))])
        np.save(str(map_path), arr)
        log.info(f"FAISS index saved ({self.index.ntotal} vectors)")

    def add_embedding(self, student_id: str, embedding: np.ndarray) -> None:
        """Add a single embedding to the index (e.g., after enrollment)."""
        vec = embedding.astype(np.float32).reshape(1, -1)
        self.id_to_student[self.index.ntotal] = student_id
        self.index.add(vec)

    def update_embeddings(self, student_id: str, embeddings: np.ndarray) -> None:
        """
        Replace all embeddings for a student (used during continual learning).
        NOTE: FAISS FlatIP does not support deletion; simplest approach is
        to rebuild the index from scratch when updating.
        Call rebuild_index_from_db() for a full rebuild.
        """
        log.warning(
            f"update_embeddings called for {student_id}. "
            "Trigger rebuild_index_from_db() for a full index refresh."
        )
        for emb in embeddings:
            self.add_embedding(student_id, emb)

    def rebuild_index_from_db(self, embeddings_by_student: dict[str, np.ndarray]) -> None:
        """
        Full rebuild.  Pass a dict of student_id → mean_embedding (shape (512,)).
        """
        self.index = faiss.IndexFlatIP(EMB_DIM)
        self.id_to_student = {}
        for student_id, emb in embeddings_by_student.items():
            self.add_embedding(student_id, emb)
        self.save()
        log.info(f"Index rebuilt with {len(embeddings_by_student)} identities.")

    # ── Matching ─────────────────────────────────────────────────────

    def match(self, embedding: np.ndarray) -> tuple[str | None, float, str]:
        """
        Match an embedding against the index.

        Returns
        -------
        (student_id | None, confidence, decision)
        decision ∈ {"high", "soft", "unknown"}
        """
        if self.index.ntotal == 0:
            return None, 0.0, "unknown"

        vec = embedding.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(vec, k=1)
        score = float(scores[0][0])
        idx   = int(indices[0][0])

        if score >= HIGH_THRESH:
            return self.id_to_student.get(idx), score, "high"
        elif score >= SOFT_THRESH:
            return self.id_to_student.get(idx), score, "soft"
        else:
            return None, score, "unknown"

    def get_all_embeddings(self) -> np.ndarray:
        """Return all embeddings stored in the FAISS index."""
        if self.index.ntotal == 0:
            return np.empty((0, EMB_DIM), dtype=np.float32)
        # For IndexFlatIP, we can directly reconstruct the vectors
        return np.array([self.index.reconstruct(i) for i in range(self.index.ntotal)])

    def get_all_ids(self) -> list[str]:
        """Return all student IDs in the same order as get_all_embeddings()."""
        return [self.id_to_student.get(i) for i in range(self.index.ntotal)]


# ── Singleton helper ──────────────────────────────────────────────────

_matcher_singleton: Matcher = None


def get_matcher() -> Matcher:
    global _matcher_singleton
    if _matcher_singleton is None:
        _matcher_singleton = Matcher()
    return _matcher_singleton
