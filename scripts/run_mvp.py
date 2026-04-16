"""End-to-end MVP pipeline: embed patches → build FAISS index → evaluate → log.

Usage:
    python scripts/run_mvp.py --config configs/phase0_mvp.yaml --seed 42

The script assumes patches have already been tiled (run tile_wsis.py first)
and that data/patches/manifest.parquet exists.
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
from histoRAG.eval.metrics import mean_average_precision, random_baseline, top_k_accuracy
from histoRAG.eval.protocol import stratified_within_slide
from histoRAG.index.faiss_index import FaissFlatIP
from histoRAG.utils.config import load_config
from histoRAG.utils.logging import append_experiment_row
from histoRAG.utils.seeds import set_all_seeds


def parse_args():
    parser = argparse.ArgumentParser(description="HistoRAG Phase-0 MVP pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=None, help="Override seed in config")
    return parser.parse_args()


def load_manifest(patches_dir: str) -> pd.DataFrame:
    manifest_path = Path(patches_dir) / "manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            "Run scripts/tile_wsis.py first."
        )
    return pd.read_parquet(manifest_path)


def embed_patches(
    manifest: pd.DataFrame,
    encoder,
    batch_size: int,
    cache_dir: str,
    config_hash: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode all patches in *manifest*, with caching.

    Returns (embeddings, int_ids) arrays, both of length len(manifest).
    """
    cache_path = Path(cache_dir) / config_hash / "embeddings.npy"
    id_cache_path = Path(cache_dir) / config_hash / "patch_int_ids.npy"

    if cache_path.exists() and id_cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path), np.load(id_cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for path in tqdm(manifest["path"], desc="Loading patches"):
        images.append(Image.open(path).convert("RGB"))

    embeddings = encoder.encode_batched(images, batch_size=batch_size)
    int_ids = np.arange(len(manifest), dtype=np.int64)

    np.save(cache_path, embeddings)
    np.save(id_cache_path, int_ids)
    print(f"Cached embeddings to {cache_path}")
    return embeddings, int_ids


def build_index(embeddings: np.ndarray, int_ids: np.ndarray, save_path: str) -> FaissFlatIP:
    """Build and save a FaissFlatIP index from *embeddings*."""
    dim = embeddings.shape[1]
    idx = FaissFlatIP(dim=dim)
    idx.add(embeddings, int_ids)
    idx.save(save_path)
    print(f"FAISS index built ({idx.ntotal} vectors) → {save_path}")
    return idx


def evaluate(
    index: FaissFlatIP,
    embeddings: np.ndarray,
    manifest: pd.DataFrame,
    query_idx: pd.Index,
    gallery_idx: pd.Index,
    k_values: list[int],
    map_at_k: int,
) -> dict:
    """Run retrieval evaluation and return a metrics dict."""
    # Re-index embeddings; FAISS int IDs are positional (0..N-1)
    query_embs = embeddings[query_idx]
    gallery_labels = manifest.loc[gallery_idx, "label"].values
    query_labels = manifest.loc[query_idx, "label"].values

    max_k = max(max(k_values), map_at_k)
    # Search over entire index (query patches may self-match — handle below)
    sims, retrieved_ids = index.search(query_embs, k=max_k + 1)

    # Map retrieved int IDs → labels; skip self-matches
    retrieved_labels_list = []
    for q_pos, q_int_id in enumerate(query_idx):
        row_labels = []
        for ret_id in retrieved_ids[q_pos]:
            if ret_id == q_int_id:  # skip self
                continue
            if ret_id < 0 or ret_id >= len(manifest):
                continue
            row_labels.append(manifest.iloc[ret_id]["label"])
            if len(row_labels) == max_k:
                break
        # Pad with empty string if not enough results
        while len(row_labels) < max_k:
            row_labels.append("")
        retrieved_labels_list.append(row_labels)

    retrieved_labels = np.array(retrieved_labels_list)

    metrics = {}
    for k in k_values:
        metrics[f"top{k}"] = top_k_accuracy(retrieved_labels, query_labels, k=k)
    metrics["map_at_10"] = mean_average_precision(retrieved_labels, query_labels, k=map_at_k)
    n_classes = manifest["label"].nunique()
    metrics["random_baseline_top5"] = random_baseline(n_classes=n_classes, k=5)
    metrics["num_patches"] = len(manifest)
    metrics["num_query"] = len(query_idx)
    metrics["num_gallery"] = len(gallery_idx)
    return metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Override seed from CLI
    if args.seed is not None:
        cfg["run"]["seed"] = args.seed
    seed = cfg["run"]["seed"]
    set_all_seeds(seed)
    print(f"\n{'='*60}")
    print(f"HistoRAG Phase-0 MVP  |  seed={seed}  |  encoder={cfg['encoder']['name']}")
    print(f"{'='*60}")

    from histoRAG.utils.config import hash_config
    cfg_hash = hash_config(cfg)

    # Load manifest
    manifest = load_manifest(cfg["tiling"]["patches_dir"])
    print(f"Manifest loaded: {len(manifest)} patches across {manifest['slide_id'].nunique()} slides")

    # Encoder
    enc_cfg = cfg["encoder"]
    device = cfg["run"].get("device", "auto")
    encoder = get_encoder(enc_cfg["name"], device=device)

    # Embed
    t0 = time.time()
    embeddings, int_ids = embed_patches(
        manifest,
        encoder,
        batch_size=enc_cfg.get("batch_size", 32),
        cache_dir="data/indexes",
        config_hash=cfg_hash,
    )
    embed_time = time.time() - t0
    print(f"Embedding done in {embed_time:.1f}s  shape={embeddings.shape}")

    # Build index
    t0 = time.time()
    index_path = cfg["index"]["save_path"]
    index = build_index(embeddings, int_ids, save_path=index_path)
    index_time = time.time() - t0
    print(f"Index built in {index_time:.1f}s")

    # Eval split
    eval_cfg = cfg["eval"]
    split_cfg = eval_cfg["query_gallery_split"]
    query_idx, gallery_idx = stratified_within_slide(
        manifest,
        query_frac=split_cfg["query_frac"],
        seed=seed,
    )
    print(f"Split: {len(query_idx)} query  /  {len(gallery_idx)} gallery")

    # Evaluate
    t0 = time.time()
    metrics = evaluate(
        index, embeddings, manifest,
        query_idx, gallery_idx,
        k_values=eval_cfg["k_values"],
        map_at_k=eval_cfg["compute_map_at"],
    )
    query_time = time.time() - t0

    timings = {
        "embed_time_s": round(embed_time, 2),
        "index_time_s": round(index_time, 2),
        "query_time_s": round(query_time, 2),
    }

    print(f"\nResults (seed={seed}):")
    for k in eval_cfg["k_values"]:
        print(f"  top-{k:2d} accuracy : {metrics[f'top{k}']:.4f}")
    print(f"  mAP@{eval_cfg['compute_map_at']}          : {metrics['map_at_10']:.4f}")
    print(f"  random baseline (top-5): {metrics['random_baseline_top5']:.4f}")

    # Log
    uid = append_experiment_row(
        cfg, metrics, timings,
        notes=f"Phase 0 baseline, seed {seed}",
    )
    print(f"\nLogged as UID: {uid}")
    print(f"Config snapshot: configs/runs/{uid}.yaml")


if __name__ == "__main__":
    main()
