"""
geometric.py  —  Branch 1: Geometric transforms
─────────────────────────────────────────────────
Rotation, flip, crop, affine shear.
All transforms are implemented via albumentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
from src.utils.config_loader import cfg

_gcfg = cfg["augmentation"]["branches"]["geometric"]


def build_geometric_transform(p: float = 1.0) -> A.Compose:
    """Return an albumentations Compose pipeline for geometric augmentation."""
    return A.Compose([
        A.Rotate(
            limit=_gcfg["rotation_limit"],
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7,
        ),
        A.HorizontalFlip(p=0.5 if _gcfg["flip"] else 0.0),
        A.RandomResizedCrop(
            size=(112, 112),
            scale=tuple(_gcfg["crop_scale"]),
            ratio=(0.9, 1.1),
            p=0.6,
        ),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.05,
            rotate_limit=10,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.4,
        ),
    ], p=p)


def apply_geometric(image: np.ndarray, n_variants: int = 10) -> list[np.ndarray]:
    """
    Generate `n_variants` geometrically-augmented copies of `image`.

    Parameters
    ----------
    image      : H×W×3 BGR or RGB numpy array, already cropped to face region.
    n_variants : number of augmented images to produce.

    Returns
    -------
    List of augmented numpy arrays (same shape as input).
    """
    transform = build_geometric_transform()
    results = []
    for _ in range(n_variants):
        augmented = transform(image=image)["image"]
        results.append(augmented)
    return results


if __name__ == "__main__":
    # Quick smoke-test
    dummy = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    variants = apply_geometric(dummy, n_variants=5)
    print(f"Geometric: produced {len(variants)} variants, shape={variants[0].shape}")
