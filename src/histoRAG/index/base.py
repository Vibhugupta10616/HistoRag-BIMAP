"""Abstract base class for vector similarity indexes."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VectorIndex(ABC):
    """Interface for approximate/exact nearest-neighbour indexes.

    All implementations store int64 IDs alongside embeddings so that
    search results map back to patch identifiers.
    """

    @abstractmethod
    def add(self, embeddings: np.ndarray, ids: np.ndarray) -> None:
        """Add *embeddings* (N, D) float32 with corresponding *ids* (N,) int64."""
        ...

    @abstractmethod
    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Find *k* nearest neighbours for each row in *queries* (Q, D).

        Returns
        -------
        similarities : (Q, k) float32 — higher is more similar
        ids          : (Q, k) int64  — patch ID for each result
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the index to disk at *path*."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "VectorIndex":
        """Load a previously saved index from *path*."""
        ...

    @property
    @abstractmethod
    def ntotal(self) -> int:
        """Number of vectors currently stored in the index."""
        ...
