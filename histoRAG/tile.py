"""WSI loading and patch extraction with Otsu-HSV tissue filtering."""
from __future__ import annotations

import random
import warnings
from pathlib import Path

import numpy as np
import openslide
from PIL import Image


class WSI:
    """Thin wrapper around OpenSlide for reading WSI pyramids."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.slide = openslide.OpenSlide(str(self.path))

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.slide.dimensions

    def best_level_for_mag(self, target_mag: float) -> int:
        """Return the pyramid level closest to target_mag."""
        obj_power = self.slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        if obj_power is None:
            mpp_x = self.slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
            if mpp_x is not None:
                obj_power = 10.0 / float(mpp_x)
            else:
                warnings.warn(f"{self.path.name}: no magnification metadata, using level 0.")
                return 0
        obj_power = float(obj_power)
        best_level, best_diff = 0, float("inf")
        for lvl in range(self.slide.level_count):
            diff = abs(obj_power / self.slide.level_downsamples[lvl] - target_mag)
            if diff < best_diff:
                best_diff, best_level = diff, lvl
        return best_level

    def read_region_rgb(self, location, level, size):
        return self.slide.read_region(location, level, size).convert("RGB")

    def get_thumbnail(self, max_size=(1024, 1024)):
        return self.slide.get_thumbnail(max_size).convert("RGB")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.slide.close()


def _otsu_threshold(channel: np.ndarray) -> float:
    """Compute Otsu threshold for a 2-D uint8 array."""
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
            best_var, threshold = between, t
    return float(threshold)


class Tiler:
    """Grid-based patch extractor with Otsu-HSV tissue masking."""

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

    def extract(self, wsi: WSI, slide_id: str, out_dir: str | Path, label: str = "unknown") -> list[dict]:
        """Extract tissue patches, save as PNG, return manifest rows."""
        out_dir = Path(out_dir) / slide_id
        out_dir.mkdir(parents=True, exist_ok=True)

        level = wsi.best_level_for_mag(self.target_magnification)
        ds = wsi.slide.level_downsamples[level]
        lv_w, lv_h = wsi.slide.level_dimensions[level]
        obj_p = wsi.slide.properties.get("openslide.objective-power")
        effective_mag = float(obj_p) / ds if obj_p else 0.0

        thumb_target = (
            max(1, wsi.dimensions[0] // self.thumb_downsample),
            max(1, wsi.dimensions[1] // self.thumb_downsample),
        )
        thumb = wsi.get_thumbnail(thumb_target)
        sat = np.array(thumb.convert("HSV"))[:, :, 1]
        tissue_mask = sat > _otsu_threshold(sat)
        thumb_w, thumb_h = thumb.size

        rows = []
        for lv_x in range(0, lv_w - self.patch_size + 1, self.stride):
            for lv_y in range(0, lv_h - self.patch_size + 1, self.stride):
                tx0 = int(lv_x * thumb_w / lv_w)
                ty0 = int(lv_y * thumb_h / lv_h)
                tx1 = min(int((lv_x + self.patch_size) * thumb_w / lv_w), thumb_w)
                ty1 = min(int((lv_y + self.patch_size) * thumb_h / lv_h), thumb_h)
                region = tissue_mask[ty0:ty1, tx0:tx1]
                if region.size == 0 or region.mean() < self.min_tissue_frac:
                    continue
                x0, y0 = int(lv_x * ds), int(lv_y * ds)
                patch_id = f"{slide_id}__{x0:07d}_{y0:07d}"
                rows.append({
                    "patch_id": patch_id, "slide_id": slide_id,
                    "x": x0, "y": y0, "level": level,
                    "magnification": effective_mag, "label": label,
                    "path": str(out_dir / f"{x0:07d}_{y0:07d}.png"),
                })

        if self.max_patches_per_slide and len(rows) > self.max_patches_per_slide:
            rows = random.Random(self.seed).sample(rows, self.max_patches_per_slide)

        for row in rows:
            wsi.read_region_rgb((row["x"], row["y"]), level, (self.patch_size, self.patch_size)).save(row["path"])

        return rows
