"""Unit tests for FaissFlatIP vector index."""
from __future__ import annotations

import numpy as np
import pytest

from histoRAG.index.faiss_index import FaissFlatIP


def test_recall_at_1(random_embeddings):
    """Each vector should retrieve itself as the nearest neighbour."""
    emb, ids = random_embeddings
    idx = FaissFlatIP(dim=512)
    idx.add(emb, ids)
    _, retrieved = idx.search(emb, k=1)
    np.testing.assert_array_equal(retrieved[:, 0], ids)


def test_ntotal_starts_zero():
    idx = FaissFlatIP(dim=512)
    assert idx.ntotal == 0


def test_ntotal_after_add(random_embeddings):
    emb, ids = random_embeddings
    idx = FaissFlatIP(dim=512)
    idx.add(emb, ids)
    assert idx.ntotal == 20


def test_similarities_bounded(random_embeddings):
    """Cosine similarities of L2-normalized vectors lie in [-1, 1]."""
    emb, ids = random_embeddings
    idx = FaissFlatIP(dim=512)
    idx.add(emb, ids)
    sims, _ = idx.search(emb[:5], k=5)
    assert sims.max() <= 1.0 + 1e-5
    assert sims.min() >= -1.0 - 1e-5


def test_save_load_roundtrip(random_embeddings, tmp_path):
    """Save then load should produce identical search results."""
    emb, ids = random_embeddings
    idx = FaissFlatIP(dim=512)
    idx.add(emb, ids)
    path = str(tmp_path / "test.faiss")
    idx.save(path)

    loaded = FaissFlatIP.load(path)
    assert loaded.ntotal == 20

    _, r1 = idx.search(emb[:3], k=3)
    _, r2 = loaded.search(emb[:3], k=3)
    np.testing.assert_array_equal(r1, r2)
