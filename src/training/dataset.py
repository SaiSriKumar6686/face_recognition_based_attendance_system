"""
dataset.py
──────────
PyTorch Dataset classes for:

1.  SyntheticFaceDataset  — reads pre-generated synthetic images from
    data/synthetic/<student_id>/<image>.jpg.
    Used during Phase 1 bootstrap training.

2.  MixedFaceDataset      — combines synthetic + real verified crops for
    continual learning fine-tune passes (70/30 or configurable ratio).

Expected directory layout
─────────────────────────
data/synthetic/
    student_001/
        aug_000.jpg
        aug_001.jpg
        ...
    student_002/
        ...
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.utils.config_loader import cfg

_SYNTHETIC_DIR = Path(cfg["paths"]["synthetic"])
_VERIFIED_DIR  = Path(cfg["paths"]["cctv_crops_verified"])
_INPUT_SIZE    = tuple(cfg["inference"]["input_size"])   # (112, 112)

# Normalisation values used by AdaFace (ImageNet means)
_NORMALIZE = A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

_BASE_TRANSFORM = A.Compose([
    A.Resize(*_INPUT_SIZE),
    _NORMALIZE,
    ToTensorV2(),
])


def _build_label_map(root: Path) -> dict[str, int]:
    """Assign integer label to each student subdirectory found under `root`."""
    dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    return {d.name: idx for idx, d in enumerate(dirs)}


def _collect_samples(root: Path, label_map: dict[str, int]) -> list[tuple[str, int]]:
    """Return (image_path, label) pairs for all images under `root`."""
    exts = {".jpg", ".jpeg", ".png"}
    samples = []
    for student_id, label in label_map.items():
        student_dir = root / student_id
        if not student_dir.exists():
            continue
        for img_path in student_dir.iterdir():
            if img_path.suffix.lower() in exts:
                samples.append((str(img_path), label))
    return samples


class SyntheticFaceDataset(Dataset):
    """Dataset built entirely from pre-generated synthetic augmented images."""

    def __init__(self, synthetic_dir: Path = _SYNTHETIC_DIR, transform=None):
        self.label_map = _build_label_map(synthetic_dir)
        self.samples   = _collect_samples(synthetic_dir, self.label_map)
        self.transform = transform or _BASE_TRANSFORM
        self.num_classes = len(self.label_map)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {synthetic_dir}. "
                "Run scripts/dataset_builder.py first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label

    def get_label_map(self) -> dict[str, int]:
        return self.label_map


class MixedFaceDataset(Dataset):
    """
    Blended dataset for continual fine-tuning.
    Combines real verified crops with synthetic replay samples
    at a configurable ratio.
    """

    def __init__(
        self,
        synthetic_dir: Path = _SYNTHETIC_DIR,
        verified_dir:  Path = _VERIFIED_DIR,
        synthetic_ratio: float = None,
        transform=None,
    ):
        if synthetic_ratio is None:
            synthetic_ratio = cfg["augmentation"]["synthetic_ratio"]
        self.synthetic_ratio = synthetic_ratio

        # build shared label map from synthetic (source of truth for IDs)
        self.label_map  = _build_label_map(synthetic_dir)
        self.num_classes = len(self.label_map)
        self.transform  = transform or _BASE_TRANSFORM

        syn_samples  = _collect_samples(synthetic_dir, self.label_map)
        real_samples = _collect_samples(verified_dir,  self.label_map)

        # sub-sample synthetics to keep the target ratio
        if real_samples:
            n_real = len(real_samples)
            n_syn  = int(n_real * synthetic_ratio / (1 - synthetic_ratio))
            import random
            syn_samples = random.sample(syn_samples, min(n_syn, len(syn_samples)))

        self.samples = real_samples + syn_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label
