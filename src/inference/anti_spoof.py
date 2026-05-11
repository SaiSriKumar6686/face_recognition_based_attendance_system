"""
anti_spoof.py
─────────────
Lightweight anti-spoofing / liveness detection using texture analysis.

This module does NOT modify the face detection or recognition pipeline.
It acts as a pre-gate: after a face is detected but before attendance is logged,
we check if the face appears to be from a live person vs a printed photo/screen.

Methods used:
  1. LBP (Local Binary Pattern) variance — real faces have richer micro-texture
  2. FFT (Fast Fourier Transform) high-freq energy — printed photos lack fine detail
  3. Color channel statistics — screens/prints have different color distributions

Returns a liveness_score between 0.0 (likely spoof) and 1.0 (likely live).
Threshold of 0.4 recommended for gating.

Usage:
    from src.inference.anti_spoof import check_liveness
    score = check_liveness(face_crop_bgr)
    if score > 0.4:
        # proceed with attendance logging
"""

import numpy as np
import cv2
from src.utils.logger import log


# ── LBP Texture Analysis ────────────────────────────────────────────

def _compute_lbp_score(gray: np.ndarray) -> float:
    """
    Compute Local Binary Pattern variance.
    Real faces have higher texture complexity than printed/screen faces.
    """
    h, w = gray.shape
    if h < 10 or w < 10:
        return 0.5

    # Simple LBP: compare each pixel with its 8 neighbors
    lbp = np.zeros_like(gray, dtype=np.uint8)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = int(gray[i, j])
            code = 0
            code |= (1 << 7) if gray[i-1, j-1] >= center else 0
            code |= (1 << 6) if gray[i-1, j]   >= center else 0
            code |= (1 << 5) if gray[i-1, j+1] >= center else 0
            code |= (1 << 4) if gray[i, j+1]   >= center else 0
            code |= (1 << 3) if gray[i+1, j+1] >= center else 0
            code |= (1 << 2) if gray[i+1, j]   >= center else 0
            code |= (1 << 1) if gray[i+1, j-1] >= center else 0
            code |= (1 << 0) if gray[i, j-1]   >= center else 0
            lbp[i, j] = code

    # Compute histogram entropy (higher = more texture variation = more likely real)
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64) / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))

    # Normalize: typical range 4-8 for real faces, 2-5 for printed
    score = np.clip((entropy - 3.0) / 4.0, 0.0, 1.0)
    return float(score)


def _compute_lbp_score_fast(gray: np.ndarray) -> float:
    """
    Fast vectorized LBP computation for production use.
    """
    h, w = gray.shape
    if h < 10 or w < 10:
        return 0.5

    # Resize to standard size for consistent analysis
    gray_resized = cv2.resize(gray, (128, 128))
    h, w = gray_resized.shape

    center = gray_resized[1:-1, 1:-1].astype(np.int16)
    lbp = np.zeros_like(center, dtype=np.uint8)

    neighbors = [
        gray_resized[0:-2, 0:-2],  # top-left
        gray_resized[0:-2, 1:-1],  # top
        gray_resized[0:-2, 2:],    # top-right
        gray_resized[1:-1, 2:],    # right
        gray_resized[2:,   2:],    # bottom-right
        gray_resized[2:,   1:-1],  # bottom
        gray_resized[2:,   0:-2],  # bottom-left
        gray_resized[1:-1, 0:-2],  # left
    ]

    for i, n in enumerate(neighbors):
        lbp |= ((n.astype(np.int16) >= center).astype(np.uint8) << (7 - i))

    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64) / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))

    score = np.clip((entropy - 3.0) / 4.0, 0.0, 1.0)
    return float(score)


# ── FFT Frequency Analysis ──────────────────────────────────────────

def _compute_fft_score(gray: np.ndarray) -> float:
    """
    Analyze frequency content via FFT.
    Real faces have more high-frequency detail; printed/screen images
    lose fine detail due to pixelation and moiré patterns.
    """
    h, w = gray.shape
    if h < 10 or w < 10:
        return 0.5

    # Resize for consistent analysis
    gray_resized = cv2.resize(gray, (128, 128)).astype(np.float64)

    # Apply windowing to reduce edge artifacts
    window = np.outer(np.hanning(128), np.hanning(128))
    gray_windowed = gray_resized * window

    # 2D FFT
    f_transform = np.fft.fft2(gray_windowed)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    # Split into low and high frequency regions
    cy, cx = 64, 64
    radius = 20  # low-freq region radius

    y, x = np.ogrid[:128, :128]
    low_mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    high_mask = ~low_mask

    low_energy = np.sum(magnitude[low_mask])
    high_energy = np.sum(magnitude[high_mask])
    total_energy = low_energy + high_energy

    if total_energy == 0:
        return 0.5

    # High-freq ratio: higher = more real detail
    high_ratio = high_energy / total_energy

    # Normalize: typical range 0.3-0.7 for real, 0.1-0.4 for printed
    score = np.clip((high_ratio - 0.15) / 0.45, 0.0, 1.0)
    return float(score)


# ── Color Distribution Analysis ─────────────────────────────────────

def _compute_color_score(crop_bgr: np.ndarray) -> float:
    """
    Analyze color channel statistics.
    Screens emit light differently; prints have narrower color gamut.
    """
    if crop_bgr.shape[0] < 10 or crop_bgr.shape[1] < 10:
        return 0.5

    # Convert to different color spaces
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)

    # Saturation variance (real skin has natural variation)
    sat_var = np.var(hsv[:, :, 1].astype(np.float64))

    # Cr/Cb skin color range check
    cr = ycrcb[:, :, 1].astype(np.float64)
    cb = ycrcb[:, :, 2].astype(np.float64)
    cr_std = np.std(cr)
    cb_std = np.std(cb)

    # Real faces: moderate saturation variance, typical skin Cr/Cb range
    sat_score = np.clip(sat_var / 2000.0, 0.0, 1.0)
    chroma_score = np.clip((cr_std + cb_std) / 30.0, 0.0, 1.0)

    return float((sat_score + chroma_score) / 2.0)


# ── Moiré Pattern Detection ─────────────────────────────────────────

def _compute_moire_score(gray: np.ndarray) -> float:
    """
    Detect moiré patterns that appear when photographing screens.
    Screens produce periodic interference patterns visible in FFT.
    """
    if gray.shape[0] < 32 or gray.shape[1] < 32:
        return 0.8  # assume real if too small to analyze

    gray_resized = cv2.resize(gray, (128, 128)).astype(np.float64)

    # Laplacian to emphasize periodic patterns
    laplacian = cv2.Laplacian(gray_resized, cv2.CV_64F)
    lap_var = np.var(laplacian)

    # FFT of laplacian — look for periodic spikes
    f = np.fft.fft2(laplacian)
    f_shift = np.fft.fftshift(f)
    magnitude = np.abs(f_shift)

    # Remove DC component
    cy, cx = 64, 64
    magnitude[cy-2:cy+3, cx-2:cx+3] = 0

    # Check for unusual peaks (moiré creates isolated spikes)
    mean_mag = np.mean(magnitude)
    max_mag = np.max(magnitude)

    if mean_mag == 0:
        return 0.8

    peak_ratio = max_mag / mean_mag

    # High peak ratio = likely moiré = likely screen capture
    # Normal range: 5-15 for real, 20+ for screen
    score = np.clip(1.0 - (peak_ratio - 10.0) / 30.0, 0.0, 1.0)
    return float(score)


# ── Main API ─────────────────────────────────────────────────────────

def check_liveness(face_crop_bgr: np.ndarray) -> float:
    """
    Perform anti-spoofing analysis on a face crop.

    Parameters
    ----------
    face_crop_bgr : BGR face crop from the detector.

    Returns
    -------
    liveness_score : float in [0, 1]. Higher = more likely live.
        Recommended threshold: 0.4
    """
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return 0.0

    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)

    # Compute individual scores
    lbp_score   = _compute_lbp_score_fast(gray)
    fft_score   = _compute_fft_score(gray)
    color_score = _compute_color_score(face_crop_bgr)
    moire_score = _compute_moire_score(gray)

    # Weighted combination
    # LBP and FFT are strongest signals; color and moiré are supplementary
    weights = {
        "lbp": 0.35,
        "fft": 0.30,
        "color": 0.15,
        "moire": 0.20,
    }

    final_score = (
        weights["lbp"]   * lbp_score +
        weights["fft"]   * fft_score +
        weights["color"] * color_score +
        weights["moire"] * moire_score
    )

    log.debug(
        f"Liveness: LBP={lbp_score:.2f} FFT={fft_score:.2f} "
        f"Color={color_score:.2f} Moiré={moire_score:.2f} → {final_score:.2f}"
    )

    return float(np.clip(final_score, 0.0, 1.0))


def get_liveness_details(face_crop_bgr: np.ndarray) -> dict:
    """Return detailed breakdown of liveness scores (for metrics page)."""
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return {"lbp": 0, "fft": 0, "color": 0, "moire": 0, "final": 0}

    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
    lbp = _compute_lbp_score_fast(gray)
    fft = _compute_fft_score(gray)
    color = _compute_color_score(face_crop_bgr)
    moire = _compute_moire_score(gray)
    final = 0.35 * lbp + 0.30 * fft + 0.15 * color + 0.20 * moire

    return {
        "lbp": round(lbp, 3),
        "fft": round(fft, 3),
        "color": round(color, 3),
        "moire": round(moire, 3),
        "final": round(final, 3),
    }
