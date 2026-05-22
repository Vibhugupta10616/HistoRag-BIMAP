"""HistoRAG end-to-end pipeline: tile -> embed -> index -> evaluate -> log.

Supports two evaluation modes controlled by config key eval.level:
  "patch"  -- patch-level retrieval (Phase 0 baseline, default)
  "slide"  -- slide-level patient retrieval via mean-pooled embeddings (Phase 1)

Usage (from repo root):
    python MVP/pipeline.py --config MVP/configs/phase0_mvp.yaml --seed 42
    python MVP/pipeline.py --config MVP/configs/phase1_clip_slide.yaml --seeds 42 123 2024
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path so `import histoRAG` works regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Results and run-snapshot directories live inside MVP/.
_MVP_DIR = Path(__file__).resolve().parent

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from histoRAG.embed import FaissFlatIP, aggregate_slide_embeddings, get_encoder
from histoRAG.log import append_experiment_row, embed_cache_key, hash_config, load_config, set_all_seeds
from histoRAG.retrieve import (
    mean_average_precision,
    random_baseline,
    slide_query_gallery_split,
    stratified_within_slide,
    top_k_accuracy,
)
from histoRAG.tile import Tiler, WSI


# ---------------------------------------------------------------------------
# Step 1: Tile
# ---------------------------------------------------------------------------

def tile(cfg: dict) -> pd.DataFrame:
    """Tile all slides listed in config -> data/patches/manifest.parquet."""
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
        print(f"Tiling {slide_id} (label={label}) ...")
        with WSI(slide_path) as wsi:
            r = tiler.extract(wsi, slide_id=slide_id, out_dir=patches_dir, label=label)
        print(f"  -> {len(r)} patches")
        rows.extend(r)

    if not rows:
        raise RuntimeError("No patches extracted. Check slide_ids in config and data/raw/ contents.")

    manifest = pd.DataFrame(rows)
    manifest.to_parquet(manifest_path, index=False)
    print(f"Manifest saved: {manifest_path}  ({len(manifest)} patches total)\n")
    return manifest


# ---------------------------------------------------------------------------
# Step 2: Embed
# ---------------------------------------------------------------------------

def embed(manifest: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Encode all patches with the configured encoder and cache results.

    If encoder.precomputed_embeddings_path is set in the config, that .npy file
    is loaded directly -- useful when embeddings were computed on a remote GPU
    (e.g. FAU lab machine) and downloaded locally.

    Returns (embeddings, int_ids) where int_ids are sequential 0..N-1 indices.
    """
    enc_cfg = cfg["encoder"]
    cfg_hash = embed_cache_key(cfg)
    cache_dir = Path("data/indexes") / cfg_hash
    emb_path = cache_dir / "embeddings.npy"
    ids_path = cache_dir / "patch_int_ids.npy"

    # Allow manually placing a pre-computed .npy file from the FAU lab machine
    precomputed = enc_cfg.get("precomputed_embeddings_path")
    if precomputed and Path(precomputed).exists():
        print(f"Loading pre-computed embeddings from {precomputed}")
        embeddings = np.load(precomputed).astype(np.float32)
        int_ids = np.arange(len(manifest), dtype=np.int64)
        # Copy into the standard cache location so subsequent runs skip encoding
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, embeddings)
        np.save(ids_path, int_ids)
        print(f"Copied to cache -> {cache_dir}\n")
        return embeddings, int_ids

    if emb_path.exists() and ids_path.exists():
        print(f"Loading cached embeddings from {cache_dir}")
        return np.load(emb_path), np.load(ids_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    device = cfg["run"].get("device", "auto")
    encoder = get_encoder(enc_cfg["name"], device=device)

    # Track VRAM and throughput for the experiment log
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_encode_start = time.time()
    images = [Image.open(p).convert("RGB") for p in tqdm(manifest["path"], desc="Loading patches")]
    embeddings = encoder.encode_batched(images, batch_size=enc_cfg.get("batch_size", 32))
    encode_seconds = time.time() - t_encode_start

    int_ids = np.arange(len(manifest), dtype=np.int64)
    np.save(emb_path, embeddings)
    np.save(ids_path, int_ids)

    # Compute and print cost metrics (logged later in append_experiment_row)
    param_count = sum(p.numel() for p in encoder._model.parameters())
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024
        if torch.cuda.is_available()
        else None
    )
    throughput = len(manifest) / encode_seconds if encode_seconds > 0 else None
    cfg["_cost_metrics"] = {
        "param_count": param_count,
        "peak_vram_mb": round(peak_vram_mb, 1) if peak_vram_mb else "",
        "throughput_patches_per_sec": round(throughput, 1) if throughput else "",
    }

    print(f"Embeddings cached -> {cache_dir}")
    print(f"  Encoder params : {param_count:,}")
    print(f"  Peak VRAM      : {peak_vram_mb:.0f} MB" if peak_vram_mb else "  Device        : CPU")
    print(f"  Throughput     : {throughput:.1f} patches/sec\n" if throughput else "")
    return embeddings, int_ids


# ---------------------------------------------------------------------------
# Step 3: Build FAISS index
# ---------------------------------------------------------------------------

def build_index(embeddings: np.ndarray, int_ids: np.ndarray, save_path: str) -> FaissFlatIP:
    idx = FaissFlatIP(dim=embeddings.shape[1])
    idx.add(embeddings, int_ids)
    idx.save(save_path)
    print(f"FAISS index built ({idx.ntotal} vectors) -> {save_path}\n")
    return idx


# ---------------------------------------------------------------------------
# Step 4a: Patch-level evaluation (Phase 0)
# ---------------------------------------------------------------------------

def evaluate_patches(
    index: FaissFlatIP,
    embeddings: np.ndarray,
    manifest: pd.DataFrame,
    cfg: dict,
) -> dict:
    """Patch-level retrieval evaluation (Phase 0 protocol)."""
    eval_cfg = cfg["eval"]
    split_cfg = eval_cfg["query_gallery_split"]
    seed = cfg["run"]["seed"]

    query_idx, gallery_idx = stratified_within_slide(
        manifest, query_frac=split_cfg["query_frac"], seed=seed
    )
    print(f"Patch split: {len(query_idx)} query  /  {len(gallery_idx)} gallery")

    query_embs = embeddings[query_idx]
    query_labels = np.asarray(manifest.loc[query_idx, "label"])
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
    metrics: dict = {"eval_level": "patch"}
    for k in eval_cfg["k_values"]:
        metrics[f"top{k}"] = top_k_accuracy(retrieved_labels, query_labels, k=k)
    metrics["map_at_10"] = mean_average_precision(
        retrieved_labels, query_labels, k=eval_cfg["compute_map_at"]
    )
    metrics["random_baseline_top5"] = random_baseline(
        n_classes=manifest["label"].nunique(), k=5
    )
    metrics["num_patches"] = len(manifest)
    metrics["num_query"] = len(query_idx)
    metrics["num_gallery"] = len(gallery_idx)
    return metrics


# ---------------------------------------------------------------------------
# Step 4b: Slide-level evaluation (Phase 1)
# ---------------------------------------------------------------------------

def evaluate_slides(
    embeddings: np.ndarray,
    manifest: pd.DataFrame,
    cfg: dict,
) -> dict:
    """Slide-level retrieval evaluation (Phase 1 protocol).

    Aggregates patch embeddings to one vector per slide via mean pooling,
    then evaluates how well slide vectors retrieve slides with the same
    disease label across patients.
    """
    eval_cfg = cfg["eval"]
    seed = cfg["run"]["seed"]

    # Aggregate patches -> slide vectors
    agg_method = eval_cfg.get("aggregation", "mean")
    slide_embeddings, slide_ids = aggregate_slide_embeddings(manifest, embeddings, method=agg_method)

    # Build slide label list in same order as slide_ids
    slide_label_map = manifest.drop_duplicates("slide_id").set_index("slide_id")["label"].to_dict()
    slide_labels = [slide_label_map[sid] for sid in slide_ids]

    n_slides = len(slide_ids)
    print(f"Aggregated {len(manifest)} patches -> {n_slides} slide vectors ({agg_method} pool)")

    if n_slides < 2:
        raise RuntimeError("Need at least 2 slides for slide-level evaluation.")

    # Split slides into query / gallery
    query_frac = eval_cfg.get("query_gallery_split", {}).get("query_frac", 0.2)
    query_indices, gallery_indices = slide_query_gallery_split(
        slide_ids, slide_labels, query_frac=query_frac, seed=seed
    )

    if not query_indices:
        raise RuntimeError(
            "No query slides after split -- need more slides per label. "
            "Try fewer labels or more slides."
        )

    print(f"Slide split: {len(query_indices)} query  /  {len(gallery_indices)} gallery")

    # Build FAISS index from gallery slides only (avoids self-retrieval by design)
    gallery_embs = slide_embeddings[gallery_indices]
    gallery_labels = np.array([slide_labels[i] for i in gallery_indices])
    query_embs = slide_embeddings[query_indices]
    query_labels = np.array([slide_labels[i] for i in query_indices])

    k_values = eval_cfg["k_values"]
    map_k = eval_cfg["compute_map_at"]
    # Cap k at gallery size so FAISS doesn't error on small datasets
    effective_k = min(max(max(k_values), map_k), len(gallery_indices))

    gallery_index = FaissFlatIP(dim=gallery_embs.shape[1])
    gallery_index.add(gallery_embs, np.arange(len(gallery_indices), dtype=np.int64))

    _, retrieved_pos = gallery_index.search(query_embs, k=effective_k)

    # Map FAISS position indices back to labels
    retrieved_labels = np.array([
        [gallery_labels[pos] if 0 <= pos < len(gallery_labels) else "" for pos in row]
        for row in retrieved_pos
    ])
    # Pad to max_k if gallery was smaller
    max_k = max(max(k_values), map_k)
    if retrieved_labels.shape[1] < max_k:
        pad = np.full((len(query_indices), max_k - retrieved_labels.shape[1]), "")
        retrieved_labels = np.concatenate([retrieved_labels, pad], axis=1)

    metrics: dict = {"eval_level": "slide"}
    for k in k_values:
        pass
    metrics["slide_top1"] = top_k_accuracy(retrieved_labels, query_labels, k=1)
    metrics["slide_map_at_k"] = mean_average_precision(retrieved_labels, query_labels, k=map_k)
    metrics["random_baseline_top5"] = random_baseline(
        n_classes=len(np.unique(np.array(slide_labels))), k=5
    )
    for k in k_values:
        metrics[f"top{k}"] = top_k_accuracy(retrieved_labels, query_labels, k=min(k, effective_k))
    metrics["map_at_10"] = metrics["slide_map_at_k"]
    metrics["num_patches"] = len(manifest)
    metrics["num_slides"] = n_slides
    metrics["num_query_slides"] = len(query_indices)
    metrics["num_gallery_slides"] = len(gallery_indices)
    metrics["num_query"] = len(query_indices)
    metrics["num_gallery"] = len(gallery_indices)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HistoRAG pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None, help="Single seed (overrides config).")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Multiple seeds -- embeddings computed once, evaluation repeated per seed.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [cfg["run"]["seed"]]

    eval_level = cfg.get("eval", {}).get("level", "patch")

    print(f"\n{'='*60}")
    print(f"HistoRAG  |  seeds={seeds}  |  encoder={cfg['encoder']['name']}  |  eval={eval_level}")
    print(f"{'='*60}\n")

    # Step 1: Tile (seed-independent)
    set_all_seeds(seeds[0])
    manifest = tile(cfg)
    print(f"Manifest: {len(manifest)} patches across {manifest['slide_id'].nunique()} slides\n")

    # Step 2: Embed + cache (seed-independent -- encoder output is deterministic)
    t0 = time.time()
    embeddings, int_ids = embed(manifest, cfg)
    embed_time = time.time() - t0

    # Step 3: Build patch-level FAISS index (needed for patch eval; skipped for slide eval)
    if eval_level == "patch":
        t0 = time.time()
        index = build_index(embeddings, int_ids, save_path=cfg["index"]["save_path"])
        index_time = time.time() - t0
    else:
        index = None
        index_time = 0.0

    # Steps 4-5: Evaluate and log for each seed
    cost_metrics = cfg.pop("_cost_metrics", {})  # captured inside embed()

    for i, seed in enumerate(seeds):
        cfg["run"]["seed"] = seed
        set_all_seeds(seed)
        print(f"\n{'─'*40}")
        print(f"Seed {seed}  ({i+1}/{len(seeds)})")
        print(f"{'─'*40}")

        t0 = time.time()
        if eval_level == "slide":
            metrics = evaluate_slides(embeddings, manifest, cfg)
        else:
            metrics = evaluate_patches(index, embeddings, manifest, cfg)
        query_time = time.time() - t0

        # Merge encoder cost metrics (computed once during embed)
        metrics.update(cost_metrics)

        print("\nResults:")
        if eval_level == "slide":
            print(f"  slide top-1    : {metrics['slide_top1']:.4f}")
            print(f"  slide mAP@{cfg['eval']['compute_map_at']}    : {metrics['slide_map_at_k']:.4f}")
        else:
            for k in cfg["eval"]["k_values"]:
                print(f"  top-{k:2d} accuracy : {metrics[f'top{k}']:.4f}")
            print(f"  mAP@{cfg['eval']['compute_map_at']}          : {metrics['map_at_10']:.4f}")
        print(f"  random baseline (top-5): {metrics['random_baseline_top5']:.4f}")

        timings = {
            "embed_time_s": round(embed_time, 2) if i == 0 else 0.0,
            "index_time_s": round(index_time, 2) if i == 0 else 0.0,
            "query_time_s": round(query_time, 2),
        }
        phase_tag = "Phase 1" if eval_level == "slide" else "Phase 0"
        uid = append_experiment_row(
            cfg, metrics, timings,
            notes=f"{phase_tag} {eval_level}-level, {cfg['encoder']['name']}, seed {seed}",
            experiments_dir=_MVP_DIR / "results",
            runs_dir=_MVP_DIR / "configs" / "runs",
        )
        print(f"\nLogged as: {uid}")
        print(f"Config snapshot: MVP/configs/runs/{uid}.yaml")


if __name__ == "__main__":
    main()
