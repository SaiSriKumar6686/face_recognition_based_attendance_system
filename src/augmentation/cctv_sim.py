"""
cctv_sim.py  —  Branch 3: CCTV degradation simulation
───────────────────────────────────────────────────────
Downscale → re-upscale  (compression artifact recreation)
Motion blur
JPEG compression artifact simulation
Gaussian noise

This branch is the HIGHEST ROI augmentation for ceiling-mounted CCTV systems
and should be applied before all other branches during dataset_builder runs.
"""

import random
import numpy as np
import cv2
import albumentations as A
from src.utils.config_loader import cfg

_ccfg = cfg["augmentation"]["branches"]["cctv_sim"]


def _downscale_upscale(image: np.ndarray) -> np.ndarray:
    """Compress face to simulate CCTV low-resolution capture then re-upscale."""
    h, w = image.shape[:2]
    small_h = random.randint(_ccfg["downscale_min"], _ccfg["downscale_max"])
    small_w = int(w * small_h / h)
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _jpeg_compress(image: np.ndarray) -> np.ndarray:
    """Encode→decode at low JPEG quality to add compression artifacts."""
    quality = random.randint(*_ccfg["jpeg_quality_range"])
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buf = cv2.imencode(".jpg", image, encode_param)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _motion_blur(image: np.ndarray) -> np.ndarray:
    """Simulate motion blur from a moving subject under the camera."""
    ksize = random.choice(range(
        _ccfg["blur_kernel_range"][0],
        _ccfg["blur_kernel_range"][1] + 1,
        2,   # odd sizes only
    ))
    kernel = np.zeros((ksize, ksize))
    direction = random.choice(["horizontal", "vertical", "diagonal"])
    mid = ksize // 2
    if direction == "horizontal":
        kernel[mid, :] = 1
    elif direction == "vertical":
        kernel[:, mid] = 1
    else:
        np.fill_diagonal(kernel, 1)
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)


def _add_gaussian_noise(image: np.ndarray) -> np.ndarray:
    sigma = random.uniform(*_ccfg["gaussian_noise_sigma"])
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _ceiling_angle_perspective(image: np.ndarray) -> np.ndarray:
    """
    Apply a mild top-down perspective warp to simulate a ceiling-mounted
    camera looking slightly down at the subject (20–30° pitch).
    """
    h, w = image.shape[:2]
    tilt = random.uniform(0.05, 0.15)  # fraction of height to warp
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [w * tilt, h * tilt],
        [w * (1 - tilt), h * tilt],
        [w, h],
        [0, h],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


# ── Main entry point ─────────────────────────────────────────────────

_DEGRADATIONS = [
    (_downscale_upscale,        0.80),   # (fn, probability)
    (_jpeg_compress,            0.70),
    (_motion_blur,              0.50),
    (_add_gaussian_noise,       0.60),
    (_ceiling_angle_perspective, 0.40),
]


def apply_cctv_sim(image: np.ndarray, n_variants: int = 10) -> list[np.ndarray]:
    """
    Generate `n_variants` CCTV-degraded copies of `image`.
    Each variant applies a random subset of degradations stochastically.

    Parameters
    ----------
    image      : H×W×3 BGR numpy array (112×112 recommended).
    n_variants : number of output images.

    Returns
    -------
    List of degraded numpy arrays.
    """
    results = []
    for _ in range(n_variants):
        img = image.copy()
        for fn, prob in _DEGRADATIONS:
            if random.random() < prob:
                img = fn(img)
        results.append(img)
    return results


if __name__ == "__main__":
    dummy = np.random.randint(100, 200, (112, 112, 3), dtype=np.uint8)
    variants = apply_cctv_sim(dummy, n_variants=5)
    print(f"CCTV sim: produced {len(variants)} variants, shape={variants[0].shape}")
