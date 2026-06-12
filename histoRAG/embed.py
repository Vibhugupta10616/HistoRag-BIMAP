"""Patch encoders (CLIP, CONCH, UNI2-h), FAISS index, and slide-level aggregation."""
from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class ClipEncoder:
    """Encodes patches using CLIP ViT-B/16 (OpenAI weights).

    Returns L2-normalized 512-d float32 vectors so that inner product == cosine similarity.
    """

    name = "clip-vitb16"
    output_dim = 512

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
        for i in tqdm(range(0, len(images), batch_size), desc=f"Encoding ({self.name})"):
            parts.append(self.encode(images[i : i + batch_size]))
        return np.concatenate(parts, axis=0)

    def encode_from_paths(self, image_paths: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode images directly from file paths, streaming only batch_size images at a time.
        
        Memory-efficient alternative to loading all images first.
        """
        parts = []
        n_batches = (len(image_paths) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(n_batches), desc=f"Encoding ({self.name})"):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(image_paths))
            images = [Image.open(p).convert("RGB") for p in image_paths[start:end]]
            parts.append(self.encode(images))
        return np.concatenate(parts, axis=0)


class CONCHEncoder:
    """Encodes patches using CONCH (histopathology vision-language, MahmoodLab).

    Returns L2-normalized 512-d float32 vectors.

    Installation:
        pip install git+https://github.com/mahmoodlab/CONCH
    HuggingFace:
        MahmoodLab/conch — may require a HuggingFace access request.
        Run `huggingface-cli login` before first use.
    """

    name = "conch"
    output_dim = 512

    def __init__(self, device: str = "auto") -> None:
        try:
            from conch.open_clip_custom import create_model_from_pretrained
        except ImportError as e:
            raise ImportError(
                "CONCH not installed. Run: pip install git+https://github.com/mahmoodlab/CONCH"
            ) from e

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        model, preprocess = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch")
        self._model = model.to(device).eval()
        self._preprocess = preprocess

    @torch.inference_mode()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        """Encode a list of PIL images → (N, 512) L2-normalized float32 array."""
        tensors = torch.stack([self._preprocess(img) for img in images]).to(self.device)
        # CONCH's encode_image returns normalized features when normalize=True
        features = self._model.encode_image(tensors, normalize=True)
        return features.cpu().float().numpy()

    def encode_batched(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Encode a large list of images in batches with a progress bar."""
        parts = []
        for i in tqdm(range(0, len(images), batch_size), desc=f"Encoding ({self.name})"):
            parts.append(self.encode(images[i : i + batch_size]))
        return np.concatenate(parts, axis=0)

    def encode_from_paths(self, image_paths: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode images directly from file paths, streaming only batch_size images at a time."""
        parts = []
        n_batches = (len(image_paths) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(n_batches), desc=f"Encoding ({self.name})"):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(image_paths))
            images = [Image.open(p).convert("RGB") for p in image_paths[start:end]]
            parts.append(self.encode(images))
        return np.concatenate(parts, axis=0)


class UNIEncoder:
    """Encodes patches using UNI2-h (histopathology vision-only SSL, MahmoodLab).

    Returns L2-normalized 1536-d float32 vectors.

    Requirements:
        pip install timm
        huggingface-cli login  (MahmoodLab/UNI2-h access required)
    Hardware note:
        ViT-H/14 requires ~10-14 GB VRAM at batch=32.
        Reduce batch_size or use device="cpu" if OOM on local GPU.
    """

    name = "uni2h"
    output_dim = 1536

    def __init__(self, device: str = "auto") -> None:
        try:
            import timm
            from timm.data import create_transform, resolve_model_data_config
        except ImportError as e:
            raise ImportError("timm not installed. Run: pip install timm") from e

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        model = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h",
            pretrained=True,
            init_values=1e-5,       # required by UNI2-h LayerScale initialization
            dynamic_img_size=True,  # allows non-square inputs at inference
        )
        self._model = model.to(device).eval()
        data_cfg = resolve_model_data_config(self._model)
        self._preprocess = create_transform(**data_cfg, is_training=False)

    @torch.inference_mode()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        """Encode a list of PIL images → (N, 1536) L2-normalized float32 array."""
        tensors = torch.stack([self._preprocess(img) for img in images]).to(self.device)
        features = self._model(tensors)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().float().numpy()

    def encode_batched(self, images: list[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Encode a large list of images in batches with a progress bar."""
        parts = []
        for i in tqdm(range(0, len(images), batch_size), desc=f"Encoding ({self.name})"):
            parts.append(self.encode(images[i : i + batch_size]))
        return np.concatenate(parts, axis=0)

    def encode_from_paths(self, image_paths: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode images directly from file paths, streaming only batch_size images at a time."""
        parts = []
        n_batches = (len(image_paths) + batch_size - 1) // batch_size
        for batch_idx in tqdm(range(n_batches), desc=f"Encoding ({self.name})"):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(image_paths))
            images = [Image.open(p).convert("RGB") for p in image_paths[start:end]]
            parts.append(self.encode(images))
        return np.concatenate(parts, axis=0)


# Registry: config name → encoder class
ENCODERS: dict[str, type] = {
    "clip-vitb16": ClipEncoder,
    "conch": CONCHEncoder,
    "uni2h": UNIEncoder,
}


def get_encoder(name: str, device: str = "auto"):
    """Instantiate an encoder by its config name (e.g. 'clip-vitb16', 'conch', 'uni2h')."""
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder '{name}'. Available: {sorted(ENCODERS)}")
    return ENCODERS[name](device=device)


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Slide-level aggregation
# ---------------------------------------------------------------------------

def aggregate_slide_embeddings(
    manifest: pd.DataFrame,
    embeddings: np.ndarray,
    method: str = "mean",
) -> tuple[np.ndarray, list[str]]:
    """Aggregate patch embeddings into one vector per slide via mean pooling.

    Mean pooling implicitly encodes tissue composition: a slide with mostly
    tumor-like patches produces a vector close to the 'tumor region' of the
    embedding space. Spatial arrangement is NOT preserved — that is a Phase 2
    extension (attention-weighted aggregation / graph models).

    Args:
        manifest:   Patch manifest DataFrame with a 'slide_id' column.
                    Row order must match the rows of `embeddings`.
        embeddings: (N_patches, dim) float32 array of L2-normalized patch vectors.
        method:     Aggregation method — only "mean" is implemented for Phase 1.

    Returns:
        slide_embeddings: (N_slides, dim) float32 array, L2-re-normalized after pooling.
        slide_ids:        List of slide_id strings in the same row order.
    """
    if method != "mean":
        raise NotImplementedError(f"Aggregation '{method}' not implemented. Use 'mean'.")

    # Preserve the order slides first appear in the manifest
    slide_ids = list(dict.fromkeys(manifest["slide_id"].tolist()))
    slide_id_array = manifest["slide_id"].to_numpy()

    slide_vecs = []
    for sid in slide_ids:
        mask = slide_id_array == sid
        mean_vec = embeddings[mask].mean(axis=0)
        # Re-normalize: mean of unit vectors is not necessarily unit length
        norm = np.linalg.norm(mean_vec)
        slide_vecs.append(mean_vec / norm if norm > 0 else mean_vec)

    slide_embeddings = np.stack(slide_vecs, axis=0).astype(np.float32)
    return slide_embeddings, slide_ids
