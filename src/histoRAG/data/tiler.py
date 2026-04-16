"""Patch extraction from WSIs with Otsu-HSV tissue filtering."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image

from histoRAG.data.wsi_loader import WSI


def _otsu_threshold(channel: np.ndarray) -> float:
    """Compute Otsu threshold for a 2-D uint8 array.

    Maximises inter-class variance between foreground (tissue) and background.
    Returns the threshold value in [0, 255].
    """
    hist, _ = np.histogram(channel.ravel(), bins=256, range=(0, 256))
    total = channel.size
    sum_all = float(np.dot(np.arange(256), hist))
    best_var, threshold = 0.0, 0
    sum_b, w_b = 0.0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > best_var:
            best_var = between
            threshold = t
    return float(threshold)


class Tiler:
    """Grid-based patch extractor with Otsu-HSV tissue masking.

    Parameters
    ----------
    patch_size:
        Pixel size of extracted patches (square) at the chosen WSI level.
    stride:
        Step between patch origins; set equal to patch_size for non-overlapping grid.
    target_magnification:
        Desired magnification level (e.g. 20.0 for 20×).
    thumb_downsample:
        Downsample factor for the thumbnail used in tissue detection.
    min_tissue_frac:
        Minimum fraction of tissue-positive pixels required to keep a patch.
    max_patches_per_slide:
        Hard cap on extracted patches per slide; sub-sampled with seeded RNG.
        Set to None for no cap.
    seed:
        RNG seed for patch sub-sampling (does not affect tissue filter).
    """

    def __init__(
        self,
        patch_size: int = 256,
        stride: int = 256,
        target_magnification: float = 20.0,
        thumb_downsample: int = 32,
        min_tissue_frac: float = 0.3,
        max_patches_per_slide: int | None = 5000,
        seed: int = 42,
    ) -> None:
        self.patch_size = patch_size
        self.stride = stride
        self.target_magnification = target_magnification
        self.thumb_downsample = thumb_downsample
        self.min_tissue_frac = min_tissue_frac
        self.max_patches_per_slide = max_patches_per_slide
        self.seed = seed

    def extract(
        self,
        wsi: WSI,
        slide_id: str,
        out_dir: str | Path,
        label: str = "unknown",
    ) -> list[dict]:
        """Extract tissue patches from *wsi*, save as PNG, return manifest rows.

        Algorithm:
        1. Select pyramid level closest to target_magnification.
        2. Build tissue mask from a small thumbnail via Otsu on HSV saturation.
        3. Iterate grid candidates; project each to thumbnail space; keep if tissue_frac >= min.
        4. Cap to max_patches_per_slide using seeded sub-sampling.
        5. Read and save each kept patch as PNG.

        Returns list of dicts with keys:
            patch_id, slide_id, x, y, level, magnification, label, path
        """
        out_dir = Path(out_dir) / slide_id
        out_dir.mkdir(parents=True, exist_ok=True)

        level = wsi.best_level_for_mag(self.target_magnification)
        ds = wsi.slide.level_downsamples[level]
        lv_w, lv_h = wsi.slide.level_dimensions[level]

        # Effective magnification at chosen level
        obj_p = wsi.slide.properties.get("openslide.objective-power")
        effective_mag = float(obj_p) / ds if obj_p else 0.0

        # Build tissue mask from thumbnail
        thumb_target = (
            max(1, wsi.dimensions[0] // self.thumb_downsample),
            max(1, wsi.dimensions[1] // self.thumb_downsample),
        )
        thumb = wsi.get_thumbnail(thumb_target)
        thumb_arr = np.array(thumb.convert("HSV"))
        sat = thumb_arr[:, :, 1]  # saturation channel: high = stained tissue
        otsu_thr = _otsu_threshold(sat)
        tissue_mask = sat > otsu_thr  # True where tissue

        thumb_w, thumb_h = thumb.size

        rows = []
        # Iterate non-overlapping grid at the chosen level
        for lv_x in range(0, lv_w - self.patch_size + 1, self.stride):
            for lv_y in range(0, lv_h - self.patch_size + 1, self.stride):
                # Project patch footprint onto thumbnail
                tx0 = int(lv_x * thumb_w / lv_w)
                ty0 = int(lv_y * thumb_h / lv_h)
                tx1 = min(int((lv_x + self.patch_size) * thumb_w / lv_w), thumb_w)
                ty1 = min(int((lv_y + self.patch_size) * thumb_h / lv_h), thumb_h)

                region = tissue_mask[ty0:ty1, tx0:tx1]
                if region.size == 0:
                    continue
                if region.mean() < self.min_tissue_frac:
                    continue

                # Convert level coordinates to level-0 coordinates for read_region
                x0 = int(lv_x * ds)
                y0 = int(lv_y * ds)
                patch_id = f"{slide_id}__{x0:07d}_{y0:07d}"
                rows.append({
                    "patch_id": patch_id,
                    "slide_id": slide_id,
                    "x": x0,
                    "y": y0,
                    "level": level,
                    "magnification": effective_mag,
                    "label": label,
                    "path": str(out_dir / f"{x0:07d}_{y0:07d}.png"),
                })

        # Cap with seeded sub-sampling
        if self.max_patches_per_slide and len(rows) > self.max_patches_per_slide:
            rng = random.Random(self.seed)
            rows = rng.sample(rows, self.max_patches_per_slide)

        # Read and save patches
        for row in rows:
            img = wsi.read_region_rgb((row["x"], row["y"]), level, (self.patch_size, self.patch_size))
            img.save(row["path"])

        return rows
