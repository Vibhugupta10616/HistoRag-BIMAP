"""Unit tests for Otsu tissue filter helper (no OpenSlide required)."""
from __future__ import annotations

import numpy as np
import pytest

from histoRAG.data.tiler import _otsu_threshold


def test_otsu_threshold_bimodal():
    """Otsu threshold on a bimodal distribution should fall between the peaks."""
    # 50% background (value=240) + 50% tissue (value=80) in HSV saturation
    channel = np.array([240] * 500 + [80] * 500, dtype=np.uint8).reshape(10, 100)
    thr = _otsu_threshold(channel)
    assert 80 <= thr < 240, f"Expected threshold at or between peaks, got {thr}"


def test_otsu_threshold_uniform():
    """Uniform input should return a threshold without crashing."""
    channel = np.full((100, 100), 128, dtype=np.uint8)
    thr = _otsu_threshold(channel)
    assert 0 <= thr <= 255


def test_otsu_threshold_all_zeros():
    channel = np.zeros((50, 50), dtype=np.uint8)
    thr = _otsu_threshold(channel)
    assert thr == 0
