"""Abstract base class for patch encoders."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image
from tqdm import tqdm


class Encoder(ABC):
    """Interface for all patch encoders used in HistoRAG.

    Subclasses must set class attributes *name* and *embed_dim*, and implement
    the *encode* method.
    """

    name: str
    """Short identifier used in experiment logs and registry lookups."""

    embed_dim: int
    """Dimensionality of the output embedding vectors."""

    @abstractmethod
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        """Encode a batch of PIL images into embedding vectors.

        Parameters
        ----------
        images:
            List of RGB PIL images. All must be the same size.

        Returns
        -------
        np.ndarray of shape (len(images), embed_dim), dtype float32, L2-normalized.
        """
        ...

    def encode_batched(
        self,
        images: list[Image.Image],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode *images* in mini-batches.

        Splits the image list into chunks of *batch_size*, calls *encode* on
        each, and concatenates the results.

        Returns concatenated (N, embed_dim) float32 array.
        """
        results = []
        indices = range(0, len(images), batch_size)
        if show_progress:
            indices = tqdm(indices, desc=f"Encoding [{self.name}]", unit="batch")
        for start in indices:
            batch = images[start : start + batch_size]
            results.append(self.encode(batch))
        return np.concatenate(results, axis=0)
