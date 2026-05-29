"""Patch encoders for HPC embedding pipeline.

Self-contained — no dependency on histoRAG/ or any other project folder.

Available encoders:
    ClipEncoder   CLIP ViT-B/16 (OpenAI) — general-purpose baseline, 512-d
    CONCHEncoder  CONCH ViT-B/16 (MahmoodLab) — histopathology vision-language, 512-d

CONCH setup (one-time, before running the pipeline):
    pip install git+https://github.com/mahmoodlab/CONCH
    huggingface-cli login          # needs MahmoodLab/conch access approved on HuggingFace
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


class ClipEncoder:
    """CLIP ViT-B/16 (OpenAI weights). Returns L2-normalised 512-d float32 vectors."""

    name = "clip"
    output_dim = 512

    def __init__(self, device: str = "auto") -> None:
        import open_clip
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        self._model = model.to(device).eval()
        self._preprocess = preprocess
        print(f"ClipEncoder loaded on {device}")

    @torch.inference_mode()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        tensors = torch.stack([self._preprocess(img) for img in images]).to(self.device)
        features = self._model.encode_image(tensors)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().float().numpy()

    def encode_batched(self, images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        parts = []
        for i in tqdm(range(0, len(images), batch_size), desc="Encoding (CLIP)"):
            parts.append(self.encode(images[i: i + batch_size]))
        return np.concatenate(parts, axis=0)


class CONCHEncoder:
    """CONCH ViT-B/16 (MahmoodLab). Returns L2-normalised 512-d float32 vectors.

    Requires:
        pip install git+https://github.com/mahmoodlab/CONCH
        huggingface-cli login   (MahmoodLab/conch access required on HuggingFace)
    """

    name = "conch"
    output_dim = 512

    def __init__(self, device: str = "auto") -> None:
        try:
            from conch.open_clip_custom import create_model_from_pretrained
        except ImportError as e:
            raise ImportError(
                "CONCH not installed.\n"
                "Run: pip install git+https://github.com/mahmoodlab/CONCH\n"
                "Then: huggingface-cli login"
            ) from e

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        model, preprocess = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/conch")
        self._model = model.to(device).eval()
        self._preprocess = preprocess
        print(f"CONCHEncoder loaded on {device}")

    @torch.inference_mode()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        tensors = torch.stack([self._preprocess(img) for img in images]).to(self.device)
        features = self._model.encode_image(tensors, normalize=True)
        return features.cpu().float().numpy()

    def encode_batched(self, images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        parts = []
        for i in tqdm(range(0, len(images), batch_size), desc="Encoding (CONCH)"):
            parts.append(self.encode(images[i: i + batch_size]))
        return np.concatenate(parts, axis=0)


# Registry — add new encoders here
ENCODERS: dict[str, type] = {
    "clip":  ClipEncoder,
    "conch": CONCHEncoder,
}


def get_encoder(name: str, device: str = "auto"):
    """Instantiate an encoder by name. Raises ValueError for unknown names."""
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder '{name}'. Available: {sorted(ENCODERS)}")
    return ENCODERS[name](device=device)
