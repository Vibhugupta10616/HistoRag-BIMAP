"""Unit tests for encoder abstraction and CLIP ViT-B/16 implementation."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from histoRAG.encoders.registry import get_encoder


@pytest.fixture
def dummy_images():
    """Four random 224×224 RGB images for encoder tests."""
    return [
        Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        for _ in range(4)
    ]


def test_clip_output_shape(dummy_images):
    enc = get_encoder("clip-vitb16", device="cpu")
    out = enc.encode(dummy_images)
    assert out.shape == (4, 512), f"Expected (4, 512), got {out.shape}"


def test_clip_output_dtype(dummy_images):
    enc = get_encoder("clip-vitb16", device="cpu")
    out = enc.encode(dummy_images)
    assert out.dtype == np.float32


def test_clip_l2_normalized(dummy_images):
    enc = get_encoder("clip-vitb16", device="cpu")
    out = enc.encode(dummy_images)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, np.ones(4), atol=1e-5)


def test_clip_deterministic(dummy_images):
    enc = get_encoder("clip-vitb16", device="cpu")
    out1 = enc.encode(dummy_images)
    out2 = enc.encode(dummy_images)
    np.testing.assert_allclose(out1, out2, atol=1e-6)


def test_clip_text_raises(dummy_images):
    enc = get_encoder("clip-vitb16", device="cpu")
    with pytest.raises(NotImplementedError):
        enc.encode_text(["test query"])


def test_uni_importable():
    """UNI2-h class should be instantiable without loading weights."""
    enc = get_encoder("uni2h", device="cpu")
    assert enc.name == "uni2h"
    assert enc.embed_dim == 1536
    assert enc._model is None  # Weights not loaded until encode() is called
