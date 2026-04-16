"""UNI2-h histopathology foundation model encoder — Phase 1 stub.

HuggingFace model: MahmoodLab/UNI2-h
Architecture: ViT-H/14 trained with DINOv2 SSL recipe on 100k+ WSIs.
Access has been granted. Weights not loaded in Phase 0 to avoid
4 GB VRAM contention during MVP runs.

To activate in Phase 1:
    1. Run: huggingface-cli login
    2. Set encoder.name: uni2h in config YAML
    3. If VRAM <8 GB, move runs to Colab T4 (set device: auto in config)
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from histoRAG.encoders.base import Encoder

# Official architecture kwargs from MahmoodLab documentation
_UNI2H_KWARGS: dict = {
    "img_size": 224,
    "patch_size": 14,
    "depth": 24,
    "num_heads": 24,
    "init_values": 1e-5,
    "embed_dim": 1536,
    "num_classes": 0,
    "no_embed_class": True,
    "reg_tokens": 8,
    "dynamic_img_size": True,
}

# Standard ImageNet normalization (used by UNI2-h)
_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


class UNIEncoder(Encoder):
    """UNI2-h encoder for histopathology patches (Phase 1).

    Lazy-loads model weights on first call to encode() to avoid
    unnecessary HuggingFace downloads during import.
    """

    name = "uni2h"
    embed_dim = 1536

    def __init__(self, device: str = "auto") -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._model = None  # Weights loaded on first encode() call

    def _load_model(self) -> None:
        """Download and load UNI2-h weights from HuggingFace hub."""
        import timm  # local import to avoid slow load at module level

        self._model = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h",
            pretrained=True,
            **_UNI2H_KWARGS,
        ).to(self.device).eval()

    @torch.inference_mode()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        """Encode images via UNI2-h ViT-H/14 vision transformer.

        Returns L2-normalized (B, 1536) float32 numpy array.
        """
        if self._model is None:
            self._load_model()
        tensors = torch.stack([_TRANSFORM(img) for img in images]).to(self.device)
        features = self._model(tensors)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().float().numpy()
