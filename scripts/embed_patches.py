"""Embed patches from manifest into numpy arrays and build FAISS index.

Usage:
    python scripts/embed_patches.py --config configs/phase0_mvp.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from histoRAG.encoders.registry import get_encoder
from histoRAG.index.faiss_index import FaissFlatIP
from histoRAG.utils.config import hash_config, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Embed patches and build FAISS index")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg_hash = hash_config(cfg)

    patches_dir = Path(cfg["tiling"]["patches_dir"])
    manifest_path = patches_dir / "manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run tile_wsis.py first.")

    manifest = pd.read_parquet(manifest_path)
    print(f"Manifest: {len(manifest)} patches")

    enc_cfg = cfg["encoder"]
    device = cfg["run"].get("device", "auto")
    encoder = get_encoder(enc_cfg["name"], device=device)

    # Load images
    images = []
    for path in tqdm(manifest["path"], desc="Loading patches"):
        images.append(Image.open(path).convert("RGB"))

    # Encode
    t0 = time.time()
    embeddings = encoder.encode_batched(images, batch_size=enc_cfg.get("batch_size", 32))
    print(f"Encoded {len(embeddings)} patches in {time.time()-t0:.1f}s  shape={embeddings.shape}")

    # Save embeddings
    cache_dir = Path("data/indexes") / cfg_hash
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "embeddings.npy", embeddings)
    np.save(cache_dir / "patch_int_ids.npy", np.arange(len(manifest), dtype=np.int64))
    print(f"Embeddings cached → {cache_dir}")

    # Build and save FAISS index
    index = FaissFlatIP(dim=embeddings.shape[1])
    index.add(embeddings, np.arange(len(manifest), dtype=np.int64))
    index.save(cfg["index"]["save_path"])
    print(f"FAISS index saved → {cfg['index']['save_path']}")


if __name__ == "__main__":
    main()
