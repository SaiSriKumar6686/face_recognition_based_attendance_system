"""
export_onnx.py
───────────────
Export the best trained PyTorch checkpoint to ONNX for CPU inference.

Usage
─────
    python scripts/export_onnx.py
    python scripts/export_onnx.py --checkpoint models/checkpoints/finetune_latest.pt
    python scripts/export_onnx.py --out models/onnx/model_v2.onnx
"""

import argparse
from pathlib import Path
from datetime import datetime

from src.training.adaface_model import build_model, load_checkpoint, export_onnx
from src.training.dataset import SyntheticFaceDataset
from src.utils.config_loader import cfg
from src.utils.logger import log

_CKPT_DIR = Path(cfg["paths"]["checkpoints"])
_ONNX_DIR = Path(cfg["paths"]["onnx"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint to export (default: bootstrap_best.pt)")
    parser.add_argument("--out", default=None,
                        help="Output ONNX path (default: auto-timestamped)")
    args = parser.parse_args()

    ckpt_path = args.checkpoint or str(_CKPT_DIR / "bootstrap_best.pt")
    if not Path(ckpt_path).exists():
        log.error(f"Checkpoint not found: {ckpt_path}")
        exit(1)

    # infer num_classes from synthetic dataset
    try:
        dataset     = SyntheticFaceDataset()
        num_classes = dataset.num_classes
    except Exception:
        num_classes = 100   # fallback
        log.warning("Could not read dataset — using num_classes=100 placeholder.")

    model = build_model(num_classes=num_classes)
    load_checkpoint(model, ckpt_path)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or str(_ONNX_DIR / f"model_{ts}.onnx")
    _ONNX_DIR.mkdir(parents=True, exist_ok=True)

    export_onnx(model, out_path, device="cpu")
    log.info(f"Export complete → {out_path}")
