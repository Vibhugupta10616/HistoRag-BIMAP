"""FAISS flat inner-product index for exact cosine similarity retrieval."""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from histoRAG.index.base import VectorIndex


class FaissFlatIP(VectorIndex):
    """Exact cosine similarity search using FAISS IndexFlatIP + ID mapping.

    Expects L2-normalized embeddings: inner-product of two unit vectors equals
    their cosine similarity, so IndexFlatIP computes cosine similarity exactly.

    At MVP scale (≤100k patches) exact search is fast enough; IVF/HNSW
    approximate indexes are Phase-2 ablations.
    """

    def __init__(self, dim: int) -> None:
        """Initialise an empty index for *dim*-dimensional vectors."""
        self.dim = dim
        # IndexIDMap2 maps arbitrary int64 IDs to the flat inner-product index
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def add(self, embeddings: np.ndarray, ids: np.ndarray) -> None:
        """Add *embeddings* (N, D) float32 with int64 *ids* (N,) to the index."""
        emb = np.ascontiguousarray(embeddings, dtype=np.float32)
        ids_ = np.ascontiguousarray(ids, dtype=np.int64)
        self._index.add_with_ids(emb, ids_)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (similarities, ids) arrays each of shape (Q, k)."""
        q = np.ascontiguousarray(queries, dtype=np.float32)
        similarities, ids = self._index.search(q, k)
        return similarities, ids

    def save(self, path: str) -> None:
        """Write the index to *path* using faiss.write_index."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: str) -> "FaissFlatIP":
        """Load a FaissFlatIP from a previously saved file."""
        raw = faiss.read_index(str(path))
        obj = cls.__new__(cls)
        obj.dim = raw.d
        obj._index = raw
        return obj

    @property
    def ntotal(self) -> int:
        return self._index.ntotal
