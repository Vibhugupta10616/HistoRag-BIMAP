"""Run retrieval evaluation on a built FAISS index and log results.

Usage:
    python scripts/evaluate.py --config configs/phase0_mvp.yaml --seed 42
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from histoRAG.eval.metrics import mean_average_precision, random_baseline, top_k_accuracy
from histoRAG.eval.protocol import stratified_within_slide
from histoRAG.index.faiss_index import FaissFlatIP
from histoRAG.utils.config import hash_config, load_config
from histoRAG.utils.logging import append_experiment_row
from histoRAG.utils.seeds import set_all_seeds


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval and log results")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["run"]["seed"] = args.seed
    seed = cfg["run"]["seed"]
    set_all_seeds(seed)

    cfg_hash = hash_config(cfg)
    cache_dir = f"data/indexes/{cfg_hash}"

    manifest = pd.read_parquet(f"{cfg['tiling']['patches_dir']}/manifest.parquet")
    embeddings = np.load(f"{cache_dir}/embeddings.npy")
    index = FaissFlatIP.load(cfg["index"]["save_path"])

    eval_cfg = cfg["eval"]
    query_idx, gallery_idx = stratified_within_slide(
        manifest, query_frac=eval_cfg["query_gallery_split"]["query_frac"], seed=seed
    )

    k_values = eval_cfg["k_values"]
    max_k = max(max(k_values), eval_cfg["compute_map_at"])
    query_embs = embeddings[query_idx]
    query_labels = manifest.loc[query_idx, "label"].values

    t0 = time.time()
    _, retrieved_ids = index.search(query_embs, k=max_k + 1)
    query_time = time.time() - t0

    # Build retrieved label matrix, skip self-matches
    retrieved_labels_list = []
    for q_pos, q_int_id in enumerate(query_idx):
        row_labels = []
        for ret_id in retrieved_ids[q_pos]:
            if ret_id == q_int_id:
                continue
            if ret_id < 0 or ret_id >= len(manifest):
                continue
            row_labels.append(manifest.iloc[ret_id]["label"])
            if len(row_labels) == max_k:
                break
        while len(row_labels) < max_k:
            row_labels.append("")
        retrieved_labels_list.append(row_labels)

    retrieved_labels = np.array(retrieved_labels_list)
    metrics = {}
    for k in k_values:
        metrics[f"top{k}"] = top_k_accuracy(retrieved_labels, query_labels, k=k)
    metrics["map_at_10"] = mean_average_precision(retrieved_labels, query_labels, k=eval_cfg["compute_map_at"])
    metrics["random_baseline_top5"] = random_baseline(manifest["label"].nunique(), k=5)
    metrics["num_patches"] = len(manifest)
    metrics["num_query"] = len(query_idx)
    metrics["num_gallery"] = len(gallery_idx)

    print(f"top-1: {metrics['top1']:.4f}  top-5: {metrics['top5']:.4f}  top-10: {metrics['top10']:.4f}")
    print(f"mAP@10: {metrics['map_at_10']:.4f}  random baseline (top-5): {metrics['random_baseline_top5']:.4f}")

    uid = append_experiment_row(
        cfg, metrics,
        timings={"embed_time_s": 0, "index_time_s": 0, "query_time_s": round(query_time, 2)},
        notes=f"eval-only run, seed {seed}",
    )
    print(f"Logged: {uid}")


if __name__ == "__main__":
    main()
