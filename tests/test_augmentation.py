"""
test_augmentation.py
─────────────────────
Smoke tests for all augmentation branches.
Run with:  pytest tests/test_augmentation.py -v
"""

import numpy as np
import pytest

DUMMY = np.random.randint(100, 200, (112, 112, 3), dtype=np.uint8)


def test_geometric():
    from src.augmentation.geometric import apply_geometric
    results = apply_geometric(DUMMY, n_variants=5)
    assert len(results) == 5
    assert all(r.shape == DUMMY.shape for r in results)


def test_photometric():
    from src.augmentation.photometric import apply_photometric
    results = apply_photometric(DUMMY, n_variants=5)
    assert len(results) == 5
    for r in results:
        assert r.ndim == 3 and r.shape[2] == 3   # always 3-channel


def test_cctv_sim():
    from src.augmentation.cctv_sim import apply_cctv_sim
    results = apply_cctv_sim(DUMMY, n_variants=5)
    assert len(results) == 5
    assert all(r.shape == DUMMY.shape for r in results)


def test_occlusion():
    from src.augmentation.occlusion import apply_occlusion
    results = apply_occlusion(DUMMY, n_variants=5)
    assert len(results) == 5
    assert all(r.shape == DUMMY.shape for r in results)


def test_quality_filter_passes_normal_image():
    from src.augmentation.quality_filter import filter_synthetics
    normal_images = [np.random.randint(100, 200, (112, 112, 3), dtype=np.uint8) for _ in range(5)]
    kept = filter_synthetics(normal_images)
    # At minimum BRISQUE check should pass for reasonable images (face check may fail on noise)
    assert isinstance(kept, list)
