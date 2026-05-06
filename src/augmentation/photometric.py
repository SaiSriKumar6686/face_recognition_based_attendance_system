"""
photometric.py  —  Branch 2: Photometric transforms
─────────────────────────────────────────────────────
Brightness, contrast, saturation, grayscale, gamma.
Designed to simulate indoor fluorescent / mixed lighting conditions.
"""

import albumentations as A
import numpy as np
from src.utils.config_loader import cfg

_pcfg = cfg["augmentation"]["branches"]["photometric"]


def build_photometric_transform(p: float = 1.0) -> A.Compose:
    return A.Compose([
        A.ColorJitter(
            brightness=_pcfg["brightness_limit"],
            contrast=_pcfg["contrast_limit"],
            saturation=_pcfg["saturation_limit"],
            hue=0.05,
            p=0.8,
        ),
        A.ToGray(p=_pcfg["grayscale_prob"]),      # CCTV is often near-monochrome
        A.RandomGamma(gamma_limit=(70, 130), p=0.4),
        A.CLAHE(clip_limit=2.0, p=0.3),            # simulate different exposure
        A.RandomToneCurve(scale=0.1, p=0.3),
    ], p=p)


def apply_photometric(image: np.ndarray, n_variants: int = 10) -> list[np.ndarray]:
    """
    Generate `n_variants` photometrically-augmented copies of `image`.

    Parameters
    ----------
    image      : H×W×3 numpy array (BGR or RGB).
    n_variants : number of output images.

    Returns
    -------
    List of augmented numpy arrays.  Grayscale outputs are converted back
    to 3-channel so downstream pipeline stays consistent.
    """
    transform = build_photometric_transform()
    results = []
    for _ in range(n_variants):
        aug = transform(image=image)["image"]
        # ensure 3-channel output
        if aug.ndim == 2:
            aug = np.stack([aug, aug, aug], axis=-1)
        results.append(aug)
    return results


if __name__ == "__main__":
    dummy = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    variants = apply_photometric(dummy, n_variants=5)
    print(f"Photometric: produced {len(variants)} variants, shape={variants[0].shape}")
