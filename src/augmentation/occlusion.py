"""
occlusion.py  —  Branch 5: Occlusion simulation
──────────────────────────────────────────────────
Random rectangular patches, glasses simulation, surgical mask simulation.
Critical for real-world robustness: students will wear glasses, walk through
doorways, or be partially occluded by other people.
"""

import random
import numpy as np
import cv2
from src.utils.config_loader import cfg

_ocfg = cfg["augmentation"]["branches"]["occlusion"]


def _random_patch(image: np.ndarray) -> np.ndarray:
    """Black rectangle at a random position covering 10–40% of face area."""
    img = image.copy()
    h, w = img.shape[:2]
    frac = random.uniform(_ocfg["min_patch_fraction"], _ocfg["max_patch_fraction"])
    ph = int(h * frac ** 0.5)
    pw = int(w * frac ** 0.5)
    y = random.randint(0, max(0, h - ph))
    x = random.randint(0, max(0, w - pw))
    # fill with random colour or black
    fill = random.choice([(0, 0, 0), tuple(np.random.randint(50, 200, 3).tolist())])
    img[y:y+ph, x:x+pw] = fill
    return img


def _glasses_sim(image: np.ndarray) -> np.ndarray:
    """Two dark horizontal bars over eye region."""
    img = image.copy()
    h, w = img.shape[:2]
    eye_top    = int(h * 0.28)
    eye_bottom = int(h * 0.48)
    bar_h      = max(4, int((eye_bottom - eye_top) * 0.30))
    # left lens bar
    img[eye_top:eye_top + bar_h, int(w*0.05):int(w*0.42)] = (20, 20, 20)
    # right lens bar
    img[eye_top:eye_top + bar_h, int(w*0.58):int(w*0.95)] = (20, 20, 20)
    # bridge
    img[eye_top:eye_top + bar_h//2, int(w*0.42):int(w*0.58)] = (20, 20, 20)
    return img


def _mask_sim(image: np.ndarray) -> np.ndarray:
    """White/light rectangle covering lower 40% of face (surgical mask)."""
    img = image.copy()
    h, w = img.shape[:2]
    mask_top = int(h * 0.55)
    # vary mask colour: white, light blue, black
    colour = random.choice([(230, 230, 230), (180, 200, 210), (30, 30, 30)])
    img[mask_top:, :] = colour
    return img


# ── Main entry point ─────────────────────────────────────────────────

def apply_occlusion(image: np.ndarray, n_variants: int = 10) -> list[np.ndarray]:
    """
    Generate `n_variants` occluded copies of `image`.
    Applies random patch, glasses, or mask simulation based on configured probs.

    Parameters
    ----------
    image      : H×W×3 numpy array.
    n_variants : number of output images.

    Returns
    -------
    List of occluded numpy arrays.
    """
    results = []
    for _ in range(n_variants):
        img = image.copy()
        r = random.random()
        if r < _ocfg["mask_prob"]:
            img = _mask_sim(img)
        elif r < _ocfg["mask_prob"] + _ocfg["glasses_prob"]:
            img = _glasses_sim(img)
        else:
            img = _random_patch(img)
        results.append(img)
    return results


if __name__ == "__main__":
    dummy = np.random.randint(100, 200, (112, 112, 3), dtype=np.uint8)
    variants = apply_occlusion(dummy, n_variants=5)
    print(f"Occlusion: produced {len(variants)} variants, shape={variants[0].shape}")
