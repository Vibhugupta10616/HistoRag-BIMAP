"""Tile WSIs from data/raw/ into patches and save manifest.parquet.

Usage:
    python scripts/tile_wsis.py --config configs/phase0_mvp.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from histoRAG.data.tiler import Tiler
from histoRAG.data.wsi_loader import WSI
from histoRAG.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Tile WSIs into patches")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main(cfg: dict | None = None, args=None):
    if cfg is None:
        args = parse_args()
        cfg = load_config(args.config)

    data_cfg = cfg["data"]
    tile_cfg = cfg["tiling"]

    raw_dir = Path(data_cfg["raw_dir"])
    patches_dir = Path(tile_cfg["patches_dir"])
    patches_dir.mkdir(parents=True, exist_ok=True)

    # Load label map if provided
    label_map: dict[str, str] = {}
    if data_cfg.get("label_map"):
        with open(data_cfg["label_map"]) as f:
            label_map = json.load(f)

    slide_ids = data_cfg.get("slide_ids") or []
    tiler = Tiler(
        patch_size=tile_cfg["patch_size"],
        stride=tile_cfg["stride"],
        target_magnification=tile_cfg["magnification"],
        thumb_downsample=tile_cfg["tissue_filter"]["thumb_downsample"],
        min_tissue_frac=tile_cfg["tissue_filter"]["min_tissue_frac"],
        max_patches_per_slide=tile_cfg.get("max_patches_per_slide"),
        seed=cfg["run"]["seed"],
    )

    all_rows = []
    for slide_id in slide_ids:
        # Try common WSI extensions
        slide_path = None
        for ext in [".svs", ".tiff", ".tif", ".ndpi", ".mrxs", ".scn"]:
            p = raw_dir / f"{slide_id}{ext}"
            if p.exists():
                slide_path = p
                break
        if slide_path is None:
            print(f"WARNING: slide '{slide_id}' not found in {raw_dir}, skipping.")
            continue

        label = label_map.get(slide_id, "unknown")
        print(f"Tiling {slide_id} (label={label}) …")
        with WSI(slide_path) as wsi:
            rows = tiler.extract(wsi, slide_id=slide_id, out_dir=patches_dir, label=label)
        print(f"  → {len(rows)} patches")
        all_rows.extend(rows)

    if not all_rows:
        print("No patches extracted. Check slide_ids in config and data/raw/ contents.")
        return

    manifest = pd.DataFrame(all_rows)
    manifest_path = patches_dir / "manifest.parquet"
    manifest.to_parquet(manifest_path, index=False)
    print(f"\nManifest saved: {manifest_path}  ({len(manifest)} total patches)")


if __name__ == "__main__":
    main()
