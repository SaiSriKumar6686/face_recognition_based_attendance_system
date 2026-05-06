"""
adaface_model.py
────────────────
AdaFace IR-18 wrapper.

AdaFace paper: Kim et al., 2022 — "AdaFace: Quality Adaptive Margin for
Face Recognition" (CVPR 2022).

This module:
1.  Builds the IR-18 backbone + AdaFace head.
2.  Provides save / load / ONNX export helpers.
3.  Keeps backbone frozen except the last `n` layers during fine-tuning.

NOTE ─ You need the AdaFace repo cloned into third_party/adaface/ :
    git clone https://github.com/mk-minchul/AdaFace.git third_party/adaface
Then add third_party/adaface to sys.path (done automatically below).
"""

import sys
from pathlib import Path

# ── Add AdaFace to path ───────────────────────────────────────────────
_ADAFACE_PATH = Path(__file__).parents[2] / "third_party" / "adaface"
if _ADAFACE_PATH.exists():
    sys.path.insert(0, str(_ADAFACE_PATH))
else:
    print("[adaface_model] WARNING: third_party/adaface not found. "
          "Clone AdaFace repo there before training.")

import torch
import torch.nn as nn
from src.utils.config_loader import cfg
from src.utils.logger import log

_TCFG = cfg["training"]


def build_model(num_classes: int, device: str = None) -> nn.Module:
    """
    Build AdaFace IR-18 model.

    Parameters
    ----------
    num_classes : number of student identities.
    device      : 'cuda' or 'cpu' (falls back to CPU if CUDA unavailable).

    Returns
    -------
    Compiled nn.Module ready for training or inference.
    """
    # TODO: import net from AdaFace repo once cloned
    # from net import build_model as _build
    # model = _build(model_name="ir_18")

    # ── Placeholder stub ─────────────────────────────────────────────
    # Replace the lines below with the actual AdaFace build call
    # after cloning third_party/adaface.
    log.warning("Using placeholder backbone — replace with real AdaFace IR-18 build.")
    model = _PlaceholderBackbone(embedding_dim=_TCFG["embedding_dim"],
                                  num_classes=num_classes)

    dev = _resolve_device(device)
    model = model.to(dev)
    log.info(f"Model built on device={dev}, classes={num_classes}")
    return model


def freeze_except_last_n(model: nn.Module, n: int = 2) -> None:
    """
    Freeze all parameters except the last `n` layer groups.
    Used during incremental fine-tuning to prevent catastrophic forgetting
    of early feature layers.
    """
    layers = list(model.named_parameters())
    freeze_up_to = len(layers) - n * 10   # approximate; adjust per architecture
    for i, (name, param) in enumerate(layers):
        param.requires_grad = i >= freeze_up_to
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Frozen all but last {n} layer groups | trainable params: {n_trainable:,}")


def save_checkpoint(model: nn.Module, path: str | Path, epoch: int = 0) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "state_dict": model.state_dict()}, path)
    log.info(f"Checkpoint saved → {path}")


def load_checkpoint(model: nn.Module, path: str | Path) -> int:
    """Load weights into model; returns epoch number stored in checkpoint."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    log.info(f"Loaded checkpoint from {path} (epoch {ckpt.get('epoch', '?')})")
    return ckpt.get("epoch", 0)


def export_onnx(model: nn.Module, out_path: str | Path, device: str = "cpu") -> None:
    """
    Export model to ONNX format for CPU inference with onnxruntime.
    Input shape: (batch, 3, 112, 112)
    Output shape: (batch, 512) — L2-normalised embeddings
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval().to(device)
    dummy = torch.zeros(1, 3, 112, 112).to(device)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["face_crop"],
        output_names=["embedding"],
        dynamic_axes={"face_crop": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )
    log.info(f"ONNX model exported → {out_path}")


def _resolve_device(device: str = None) -> str:
    if device is None:
        device = _TCFG.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available — falling back to CPU.")
        device = "cpu"
    return device


# ── Placeholder backbone (remove after AdaFace is installed) ─────────

class _PlaceholderBackbone(nn.Module):
    """Toy model to allow the rest of the codebase to run without AdaFace."""

    def __init__(self, embedding_dim: int = 512, num_classes: int = 100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(32 * 4 * 4, embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x).flatten(1)
        emb  = torch.nn.functional.normalize(self.fc(feat), dim=1)
        return emb
