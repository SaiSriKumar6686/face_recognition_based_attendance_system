"""
retrain_trigger.py
───────────────────
Monitors the crop buffer and fires incremental fine-tuning when either:

  Condition A — N new verified samples have accumulated since last retrain.
  Condition B — Nightly scheduled time has been reached.

After fine-tuning:
  1. Exports new ONNX model.
  2. Runs shadow A/B evaluation (shadow_eval).
  3. If evaluation passes → hot-swaps the live model.

Run this as a background thread alongside live_inference.py.
"""

import threading
import time
from datetime import datetime, date
from pathlib import Path

from src.utils.config_loader import cfg
from src.utils.logger import log
from src.utils.db import get_session, CropBuffer

_CL_CFG      = cfg["continual_learning"]
_TRIGGER_N   = _CL_CFG["retrain_trigger_n"]          # new verified samples
_SCHED_TIME  = _CL_CFG["scheduled_retrain"]           # "HH:MM"
_ONNX_DIR    = Path(cfg["paths"]["onnx"])
_CKPT_DIR    = Path(cfg["paths"]["checkpoints"])

# track state
_last_retrain_count: int  = 0
_last_retrain_date:  date = None


def _count_new_verified_since_last(last_count: int) -> int:
    with get_session() as s:
        total = s.query(CropBuffer).filter_by(verified=True).count()
    return total - last_count


def _is_scheduled_time() -> bool:
    now       = datetime.now()
    hh, mm    = map(int, _SCHED_TIME.split(":"))
    target    = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    in_window = abs((now - target).total_seconds()) < 60   # ±60s window
    return in_window and now.date() != _last_retrain_date


def _run_retrain() -> None:
    """Execute one full incremental fine-tune + export + shadow eval cycle."""
    log.info("═" * 60)
    log.info("RETRAIN TRIGGERED")
    log.info("═" * 60)

    try:
        # 1. Compute EWC fisher on existing model + dataset
        from src.training.dataset import SyntheticFaceDataset
        from torch.utils.data import DataLoader
        from src.training.adaface_model import build_model, load_checkpoint
        from src.continual_learning.ewc import EWC

        dataset = SyntheticFaceDataset()
        loader  = DataLoader(dataset, batch_size=32, shuffle=True)
        model   = build_model(num_classes=dataset.num_classes)
        ckpt    = _CKPT_DIR / "bootstrap_best.pt"
        if ckpt.exists():
            load_checkpoint(model, ckpt)
        ewc = EWC(model, loader, device="cpu", n_batches=30)

        # 2. Fine-tune with EWC penalty
        from src.training.trainer import fine_tune_incremental
        fine_tune_incremental(ewc_penalty=ewc.penalty)

        # 3. Export new ONNX
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        onnx_path = _ONNX_DIR / f"model_{ts}.onnx"
        from src.training.adaface_model import load_checkpoint, export_onnx
        ft_ckpt   = _CKPT_DIR / "finetune_latest.pt"
        load_checkpoint(model, ft_ckpt)
        export_onnx(model, onnx_path)

        # 4. Shadow A/B evaluation
        promoted = _shadow_eval(onnx_path)
        if promoted:
            log.info(f"✓ New model promoted: {onnx_path.name}")
        else:
            log.warning("New model did NOT pass shadow eval — keeping old model.")
            onnx_path.unlink(missing_ok=True)

    except Exception as e:
        log.error(f"Retrain failed: {e}", exc_info=True)


def _shadow_eval(new_onnx: Path) -> bool:
    """
    Placeholder shadow A/B evaluation.
    Compare new model vs current live model on a held-out seed subset.

    TODO: implement proper evaluation metrics (rank-1 accuracy, TAR@FAR).
    Returns True to promote, False to discard.
    """
    if not cfg["continual_learning"]["shadow_eval_before_promote"]:
        return True

    log.info(f"Shadow eval: {new_onnx.name} vs current live model — (stub: auto-promote)")
    # TODO: load both models, run inference on held-out crops, compare TAR@FAR=0.01
    return True


# ── Background thread ─────────────────────────────────────────────────

class RetrainTrigger:
    def __init__(self, poll_interval_sec: int = 60):
        self.poll_interval = poll_interval_sec
        self._running      = False
        self._thread: threading.Thread = None

    def start(self) -> None:
        global _last_retrain_count, _last_retrain_date
        with get_session() as s:
            _last_retrain_count = s.query(CropBuffer).filter_by(verified=True).count()
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info(f"RetrainTrigger started (poll every {self.poll_interval}s)")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    def _loop(self) -> None:
        global _last_retrain_count, _last_retrain_date
        while self._running:
            new = _count_new_verified_since_last(_last_retrain_count)
            if new >= _TRIGGER_N or _is_scheduled_time():
                reason = "N samples" if new >= _TRIGGER_N else "scheduled"
                log.info(f"Retrain condition met ({reason}: {new} new samples)")
                _run_retrain()
                with get_session() as s:
                    _last_retrain_count = s.query(CropBuffer).filter_by(verified=True).count()
                _last_retrain_date = date.today()

            time.sleep(self.poll_interval)


if __name__ == "__main__":
    trigger = RetrainTrigger(poll_interval_sec=30)
    trigger.start()
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        trigger.stop()
