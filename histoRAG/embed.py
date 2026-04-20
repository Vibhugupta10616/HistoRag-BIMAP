"""CLIP ViT-B/16 encoder and FAISS flat index for patch embedding and retrieval."""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch
from PIL import Image
from tqdm import tqdm


class ClipEncoder:
    """Encodes image patches using CLIP ViT-B/16 (OpenAI weights).

    Returns L2-normalized 512-d float32 vectors so that inner product == cosine similarity.
    """

    def __init__(self, model_id: str = "ViT-B-16", pretrained: str = "openai", device: str = "auto") -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        model, _, preprocess = open_clip.create_model_and_transforms(model_id, pretrained=pretrained)
        self._model = model.to(device).eval()
        self._preprocess = preprocess

    @torch.inference_mode()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        """Encode a list of PIL images → (N, 512) L2-normalized float32 array."""
        tensors = torch.stack([self._preprocess(img) for img in images]).to(self.device)
        features = self._model.encode_image(tensors)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().float().numpy()

    def encode_batched(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Encode a large list of images in batches with a progress bar."""
        parts = []
        for i in tqdm(range(0, len(images), batch_size), desc="Encoding"):
            parts.append(self.encode(images[i : i + batch_size]))
        return np.concatenate(parts, axis=0)


class FaissFlatIP:
    """Exact cosine similarity search using FAISS IndexFlatIP.

    Expects L2-normalized embeddings: inner product of unit vectors == cosine similarity.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def add(self, embeddings: np.ndarray, ids: np.ndarray) -> None:
        self._index.add_with_ids(
            np.ascontiguousarray(embeddings, dtype=np.float32),
            np.ascontiguousarray(ids, dtype=np.int64),
        )

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (similarities, ids) arrays of shape (Q, k)."""
        return self._index.search(np.ascontiguousarray(queries, dtype=np.float32), k)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: str) -> "FaissFlatIP":
        raw = faiss.read_index(str(path))
        obj = cls.__new__(cls)
        obj.dim = raw.d
        obj._index = raw
        return obj

    @property
    def ntotal(self) -> int:
        return self._index.ntotal
