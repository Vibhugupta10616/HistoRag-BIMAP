"""PyTorch Dataset wrapping extracted patches and their manifest."""
from __future__ import annotations

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """Dataset of image patches loaded from disk.

    Reads patch file paths from a manifest DataFrame and applies an optional
    transform (e.g. the encoder's preprocessing pipeline).

    Parameters
    ----------
    manifest:
        DataFrame with at minimum columns 'patch_id' and 'path'.
    transform:
        Optional callable applied to each PIL image before returning.
    """

    def __init__(self, manifest: pd.DataFrame, transform=None) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        """Return (image, patch_id) for the patch at *idx*."""
        row = self.manifest.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, row["patch_id"]
