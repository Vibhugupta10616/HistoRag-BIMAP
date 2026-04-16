"""Shared pytest fixtures for the histoRAG test suite."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image


@pytest.fixture
def dummy_manifest(tmp_path):
    """Create 20 small PNG patches and a matching manifest DataFrame.

    10 patches from slide_001 with label 'classA',
    10 patches from slide_002 with label 'classB'.
    """
    records = []
    for i in range(20):
        label = "classA" if i < 10 else "classB"
        slide_id = "slide_001" if i < 10 else "slide_002"
        x = (i % 5) * 256
        y = (i // 5) * 256
        patch_id = f"{slide_id}__{x:07d}_{y:07d}"
        img_path = tmp_path / f"{patch_id}.png"
        arr = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_path)
        records.append({
            "patch_id": patch_id,
            "slide_id": slide_id,
            "x": x,
            "y": y,
            "level": 0,
            "magnification": 20.0,
            "label": label,
            "path": str(img_path),
        })
    return pd.DataFrame(records)


@pytest.fixture
def random_embeddings():
    """Return 20 random L2-normalized 512-d float32 embeddings and int64 IDs."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((20, 512)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ids = np.arange(20, dtype=np.int64)
    return emb, ids
