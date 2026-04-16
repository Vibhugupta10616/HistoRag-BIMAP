"""CLIP ViT-B/16 encoder — vision tower for Phase 0, text tower ready for Phase 1."""
from __future__ import annotations

import numpy as np
import open_clip
import torch
from PIL import Image

from histoRAG.encoders.base import Encoder


class ClipEncoder(Encoder):
    """Encodes image patches using CLIP ViT-B/16 (OpenAI pretrained weights).

    Phase 0 uses only the vision encoder.
    The text encoder is loaded but gated behind encode_text(), which raises
    NotImplementedError until Phase 1 Pro-1 (text-based retrieval) is enabled.

    Parameters
    ----------
    model_id:
        open_clip model identifier, e.g. 'ViT-B-16'.
    pretrained:
        Pretrained weight tag, e.g. 'openai'.
    device:
        'auto' picks CUDA if available, else CPU.
    """

    name = "clip-vitb16"
    embed_dim = 512

    def __init__(
        self,
        model_id: str = "ViT-B-16",
        pretrained: str = "openai",
        device: str = "auto",
    ) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_id, pretrained=pretrained
        )
        self._model = model.to(device).eval()
        self._preprocess = preprocess
        # Tokenizer kept for Phase-1 text search hook
        self._tokenizer = open_clip.get_tokenizer(model_id)

    @torch.inference_mode()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        """Encode images via the CLIP vision tower.

        Applies CLIP preprocessing (resize to 224, normalize with OpenAI stats),
        then runs the vision transformer to produce one 512-d vector per image.
        Vectors are L2-normalized so inner-product == cosine similarity.

        Returns (B, 512) float32 numpy array.
        """
        tensors = torch.stack(
            [self._preprocess(img) for img in images]
        ).to(self.device)
        features = self._model.encode_image(tensors)
        # L2-normalize: required so FAISS IndexFlatIP computes cosine similarity
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().float().numpy()

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Encode text strings via the CLIP text tower.

        Disabled in Phase 0. Will be activated for Pro-1 text-based retrieval
        in Phase 1 without any changes to this class.

        Raises NotImplementedError in Phase 0.
        """
        raise NotImplementedError(
            "Text encoding is a Phase-1 Pro-1 feature; not active in Phase 0."
        )
