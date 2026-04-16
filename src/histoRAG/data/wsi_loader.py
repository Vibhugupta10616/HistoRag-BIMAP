"""OpenSlide wrapper for whole slide image loading and magnification selection."""
from __future__ import annotations

import warnings
from pathlib import Path

import openslide


class WSI:
    """Thin wrapper around an OpenSlide object.

    Provides a consistent interface for selecting the appropriate pyramid
    level given a target magnification and reading RGB regions.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.slide = openslide.OpenSlide(str(self.path))

    @property
    def dimensions(self) -> tuple[int, int]:
        """Width × height of level 0 in pixels."""
        return self.slide.dimensions

    def best_level_for_mag(self, target_mag: float) -> int:
        """Return the pyramid level whose magnification is closest to *target_mag*.

        Strategy:
        1. Read objective-power slide property directly.
        2. Fall back to deriving magnification from MPP (microns per pixel).
        3. If no metadata: warn and return level 0.

        Effective magnification at level L = objective_power / level_downsample[L].
        """
        obj_power = self.slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        if obj_power is None:
            mpp_x = self.slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
            if mpp_x is not None:
                # Rough estimate: 10× ≈ 1 µm/px
                obj_power = 10.0 / float(mpp_x)
            else:
                warnings.warn(
                    f"{self.path.name}: magnification metadata absent; defaulting to level 0.",
                    stacklevel=2,
                )
                return 0

        obj_power = float(obj_power)
        best_level, best_diff = 0, float("inf")
        for lvl in range(self.slide.level_count):
            ds = self.slide.level_downsamples[lvl]
            diff = abs(obj_power / ds - target_mag)
            if diff < best_diff:
                best_diff = diff
                best_level = lvl
        return best_level

    def read_region_rgb(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ):
        """Read a region and return an RGB PIL image (drops alpha channel)."""
        return self.slide.read_region(location, level, size).convert("RGB")

    def get_thumbnail(self, max_size: tuple[int, int] = (1024, 1024)):
        """Return a downsampled RGB thumbnail PIL image."""
        return self.slide.get_thumbnail(max_size).convert("RGB")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.slide.close()

    def close(self):
        """Explicitly close the underlying OpenSlide handle."""
        self.slide.close()
