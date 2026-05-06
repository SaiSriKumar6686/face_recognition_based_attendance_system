"""
preprocess.py
──────────────
Image preprocessing pipeline for face crops before embedding.

Applies quality-improving transforms to 112×112 aligned face crops:
  1. CLAHE (Contrast Limited Adaptive Histogram Equalisation) — normalises
     lighting across different CCTV conditions (fluorescent, backlit, dark).
  2. Bilateral filter — removes sensor noise while preserving facial edges.
  3. Unsharp mask — recovers detail lost to CCTV compression / motion blur.
  4. White-balance correction — reduces colour cast from indoor lighting.

Each transform is independently toggleable. The pipeline is designed to be
inserted between face detection (crop) and embedding extraction.

Usage:
    from testing.preprocess import preprocess_crop
    enhanced = preprocess_crop(crop_bgr)
"""

import cv2
import numpy as np


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE to the luminance channel (LAB colour space).
    This normalises exposure without destroying colour information.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_bilateral_denoise(image: np.ndarray, d: int = 5,
                             sigma_colour: float = 50,
                             sigma_space: float = 50) -> np.ndarray:
    """
    Bilateral filter: smooths noise while keeping edges sharp.
    Good for removing CCTV sensor noise without blurring facial features.
    """
    return cv2.bilateralFilter(image, d, sigma_colour, sigma_space)


def apply_unsharp_mask(image: np.ndarray, sigma: float = 1.0,
                        strength: float = 1.5) -> np.ndarray:
    """
    Unsharp masking: sharpens the image by subtracting a blurred copy.
    Recovers detail lost to JPEG compression or motion blur.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_white_balance(image: np.ndarray) -> np.ndarray:
    """
    Simple grey-world white balance correction.
    Removes colour casts from fluorescent / tungsten indoor lighting.
    """
    result = image.copy().astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_all = (avg_b + avg_g + avg_r) / 3.0

    if avg_b > 0:
        result[:, :, 0] *= avg_all / avg_b
    if avg_g > 0:
        result[:, :, 1] *= avg_all / avg_g
    if avg_r > 0:
        result[:, :, 2] *= avg_all / avg_r

    return np.clip(result, 0, 255).astype(np.uint8)


def preprocess_crop(crop_bgr: np.ndarray,
                     clahe: bool = True,
                     denoise: bool = True,
                     sharpen: bool = True,
                     white_balance: bool = True) -> np.ndarray:
    """
    Full preprocessing pipeline for a single face crop.

    Parameters
    ----------
    crop_bgr       : H×W×3 BGR face crop (typically 112×112).
    clahe          : Apply CLAHE contrast normalisation.
    denoise        : Apply bilateral denoising.
    sharpen        : Apply unsharp mask sharpening.
    white_balance  : Apply grey-world white balance.

    Returns
    -------
    Enhanced BGR face crop, same shape as input.
    """
    result = crop_bgr.copy()

    if white_balance:
        result = apply_white_balance(result)

    if clahe:
        result = apply_clahe(result, clip_limit=2.0)

    if denoise:
        result = apply_bilateral_denoise(result, d=5, sigma_colour=40,
                                          sigma_space=40)

    if sharpen:
        result = apply_unsharp_mask(result, sigma=1.0, strength=1.0)

    return result


def preprocess_batch(crops: list[np.ndarray], **kwargs) -> list[np.ndarray]:
    """Apply preprocessing to a list of face crops."""
    return [preprocess_crop(c, **kwargs) for c in crops]


if __name__ == "__main__":
    # Quick visual test
    dummy = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
    enhanced = preprocess_crop(dummy)
    print(f"Input shape: {dummy.shape}, Output shape: {enhanced.shape}")
    print(f"Input range: [{dummy.min()}, {dummy.max()}]")
    print(f"Output range: [{enhanced.min()}, {enhanced.max()}]")
    print("Preprocessing pipeline OK.")
