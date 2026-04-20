"""HistoRAG end-to-end pipeline: tile → embed → index → evaluate → log.

Usage:
    python pipeline.py --config configs/phase0_mvp.yaml --seed 42
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from histoRAG.embed import ClipEncoder, FaissFlatIP
from histoRAG.log import append_experiment_row, hash_config, load_config, set_all_seeds
from histoRAG.retrieve import mean_average_precision, random_baseline, stratified_within_slide, top_k_accuracy
from histoRAG.tile import Tiler, WSI


def tile(cfg: dict) -> pd.DataFrame:
    """Tile all slides listed in config → data/patches/manifest.parquet."""
    data_cfg, tile_cfg = cfg["data"], cfg["tiling"]
    patches_dir = Path(tile_cfg["patches_dir"])
    manifest_path = patches_dir / "manifest.parquet"

    if manifest_path.exists():
        print(f"Manifest already exists ({manifest_path}), skipping tiling.")
        return pd.read_parquet(manifest_path)

    patches_dir.mkdir(parents=True, exist_ok=True)
    label_map: dict[str, str] = {}
    if data_cfg.get("label_map"):
        with open(data_cfg["label_map"]) as f:
            label_map = json.load(f)

    tiler = Tiler(
        patch_size=tile_cfg["patch_size"],
        stride=tile_cfg["stride"],
        target_magnification=tile_cfg["magnification"],
        thumb_downsample=tile_cfg["tissue_filter"]["thumb_downsample"],
        min_tissue_frac=tile_cfg["tissue_filter"]["min_tissue_frac"],
        max_patches_per_slide=tile_cfg.get("max_patches_per_slide"),
        seed=cfg["run"]["seed"],
    )

    rows = []
    for slide_id in data_cfg.get("slide_ids", []):
        slide_path = None
        for ext in [".svs", ".tiff", ".tif", ".ndpi", ".mrxs"]:
            p = Path(data_cfg["raw_dir"]) / f"{slide_id}{ext}"
            if p.exists():
                slide_path = p
                break
        if slide_path is None:
            print(f"WARNING: {slide_id} not found in {data_cfg['raw_dir']}, skipping.")
            continue
        label = label_map.get(slide_id, "unknown")
        print(f"Tiling {slide_id} (label={label}) …")
        with WSI(slide_path) as wsi:
            r = tiler.extract(wsi, slide_id=slide_id, out_dir=patches_dir, label=label)
        print(f"  → {len(r)} patches")
        rows.extend(r)

    if not rows:
        raise RuntimeError("No patches extracted. Check slide_ids in config and data/raw/ contents.")

    manifest = pd.DataFrame(rows)
    manifest.to_parquet(manifest_path, index=False)
    print(f"Manifest saved: {manifest_path}  ({len(manifest)} patches total)\n")
    return manifest


def embed(manifest: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Encode all patches with CLIP, cache embeddings, build and save FAISS index."""
    enc_cfg, idx_cfg = cfg["encoder"], cfg["index"]
    cfg_hash = hash_config(cfg)
    cache_dir = Path("data/indexes") / cfg_hash
    emb_path = cache_dir / "embeddings.npy"
    ids_path = cache_dir / "patch_int_ids.npy"

    if emb_path.exists() and ids_path.exists():
        print(f"Loading cached embeddings from {cache_dir}")
        return np.load(emb_path), np.load(ids_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    encoder = ClipEncoder(device=cfg["run"].get("device", "auto"))
    images = [Image.open(p).convert("RGB") for p in tqdm(manifest["path"], desc="Loading patches")]
    embeddings = encoder.encode_batched(images, batch_size=enc_cfg.get("batch_size", 32))
    int_ids = np.arange(len(manifest), dtype=np.int64)
    np.save(emb_path, embeddings)
    np.save(ids_path, int_ids)
    print(f"Embeddings cached → {cache_dir}\n")
    return embeddings, int_ids


def build_index(embeddings: np.ndarray, int_ids: np.ndarray, save_path: str) -> FaissFlatIP:
    idx = FaissFlatIP(dim=embeddings.shape[1])
    idx.add(embeddings, int_ids)
    idx.save(save_path)
    print(f"FAISS index built ({idx.ntotal} vectors) → {save_path}\n")
    return idx


def evaluate(index: FaissFlatIP, embeddings: np.ndarray, manifest: pd.DataFrame, cfg: dict) -> dict:
    """Run retrieval evaluation and return metrics dict."""
    eval_cfg = cfg["eval"]
    split_cfg = eval_cfg["query_gallery_split"]
    seed = cfg["run"]["seed"]

    query_idx, gallery_idx = stratified_within_slide(manifest, query_frac=split_cfg["query_frac"], seed=seed)
    print(f"Split: {len(query_idx)} query  /  {len(gallery_idx)} gallery")

    query_embs = embeddings[query_idx]
    query_labels = manifest.loc[query_idx, "label"].values
    max_k = max(max(eval_cfg["k_values"]), eval_cfg["compute_map_at"])

    sims, retrieved_ids = index.search(query_embs, k=max_k + 1)

    retrieved_labels = []
    for q_pos, q_int_id in enumerate(query_idx):
        row_labels = []
        for ret_id in retrieved_ids[q_pos]:
            if ret_id == q_int_id or ret_id < 0 or ret_id >= len(manifest):
                continue
            row_labels.append(manifest.iloc[ret_id]["label"])
            if len(row_labels) == max_k:
                break
        while len(row_labels) < max_k:
            row_labels.append("")
        retrieved_labels.append(row_labels)

    retrieved_labels = np.array(retrieved_labels)
    metrics = {}
    for k in eval_cfg["k_values"]:
        metrics[f"top{k}"] = top_k_accuracy(retrieved_labels, query_labels, k=k)
    metrics["map_at_10"] = mean_average_precision(retrieved_labels, query_labels, k=eval_cfg["compute_map_at"])
    metrics["random_baseline_top5"] = random_baseline(n_classes=manifest["label"].nunique(), k=5)
    metrics["num_patches"] = len(manifest)
    metrics["num_query"] = len(query_idx)
    metrics["num_gallery"] = len(gallery_idx)
    return metrics, query_idx


def main():
    parser = argparse.ArgumentParser(description="HistoRAG pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["run"]["seed"] = args.seed
    seed = cfg["run"]["seed"]
    set_all_seeds(seed)

    print(f"\n{'='*60}")
    print(f"HistoRAG  |  seed={seed}  |  encoder={cfg['encoder']['name']}")
    print(f"{'='*60}\n")

    # Step 1: Tile
    manifest = tile(cfg)
    print(f"Manifest: {len(manifest)} patches across {manifest['slide_id'].nunique()} slides\n")

    # Step 2: Embed + cache
    t0 = time.time()
    embeddings, int_ids = embed(manifest, cfg)
    embed_time = time.time() - t0

    # Step 3: Build index
    t0 = time.time()
    index = build_index(embeddings, int_ids, save_path=cfg["index"]["save_path"])
    index_time = time.time() - t0

    # Step 4: Evaluate
    t0 = time.time()
    metrics, _ = evaluate(index, embeddings, manifest, cfg)
    query_time = time.time() - t0

    print("\nResults:")
    for k in cfg["eval"]["k_values"]:
        print(f"  top-{k:2d} accuracy : {metrics[f'top{k}']:.4f}")
    print(f"  mAP@{cfg['eval']['compute_map_at']}          : {metrics['map_at_10']:.4f}")
    print(f"  random baseline (top-5): {metrics['random_baseline_top5']:.4f}")

    # Step 5: Log
    timings = {"embed_time_s": round(embed_time, 2), "index_time_s": round(index_time, 2), "query_time_s": round(query_time, 2)}
    uid = append_experiment_row(cfg, metrics, timings, notes=f"Phase 0 baseline, seed {seed}")
    print(f"\nLogged as: {uid}")
    print(f"Config snapshot: configs/runs/{uid}.yaml")


if __name__ == "__main__":
    main()
