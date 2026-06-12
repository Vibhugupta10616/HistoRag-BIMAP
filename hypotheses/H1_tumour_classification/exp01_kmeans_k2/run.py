"""
H1 Experiment 01 — K-means k=2 (Dominant-Axis Test)

Pipeline:
    1. Load patch embeddings via histoRAG.loader (reads h5 files from embeddings_root)
    2. Derive ground-truth patch labels (tumour/other) from QuPath .geojson annotations
    3. Cluster all patches with K-means k=2 (no labels used during clustering)
    4. Map each cluster to tumour/other by majority vote against ground-truth labels
    5. Compute Accuracy, Precision, Recall per tissue and overall
    6. Save summary.json with per-tissue breakdowns to outputs/

Question answered:
    Is tumour tissue the DOMINANT axis of variation in the encoder's embedding space?

NOTE: Requires ~20GB RAM to load all embeddings at once. Run on HPC if local
      machine runs out of memory.

To run:
    python hypotheses/H1_tumour_classification/exp01_kmeans_k2/run.py
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import yaml

from histoRAG.loader import load_encoder
from histoRAG.labels import tumour_labels_from_geojson
from histoRAG.classify import (
    cluster_embeddings,
    match_clusters_to_labels,
    classification_metrics,
)


def run(config_path: str | Path) -> None:
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["outputs"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder      = cfg["encoder"]
    geojson_dir  = Path(cfg["inputs"]["geojson_dir"])
    n_clusters   = cfg["params"].get("n_clusters", 2)
    random_state = cfg["params"].get("random_state", 42)

    if not geojson_dir.exists():
        raise FileNotFoundError(
            f"geojson_dir not found: {geojson_dir}\n"
            "Download the HANCOCK .geojson annotation files and update config.yaml."
        )

    # ── Load embeddings ───────────────────────────────────────────────────────
    print(f"[H1 exp01] Loading embeddings for encoder '{encoder}'...")
    embeddings, manifest = load_encoder(encoder, cfg["inputs"]["embeddings_root"])
    manifest = manifest.reset_index(drop=True)
    print(f"[H1 exp01] {len(manifest):,} patches | embedding dim = {embeddings.shape[1]}")

    # ── Ground-truth labels from .geojson ─────────────────────────────────────
    print("[H1 exp01] Deriving ground-truth tumour labels from .geojson annotations...")
    tumour_labels = tumour_labels_from_geojson(manifest, geojson_dir)
    true_labels   = (tumour_labels == "tumour").astype(int).values

    n_tumour = int(true_labels.sum())
    n_other  = int((true_labels == 0).sum())
    print(f"[H1 exp01] Ground truth — Tumour: {n_tumour:,} | Other: {n_other:,}")

    # ── Unsupervised clustering ───────────────────────────────────────────────
    print(f"[H1 exp01] Clustering {len(manifest):,} patches with K-means k={n_clusters}...")
    cluster_ids = cluster_embeddings(embeddings, n_clusters=n_clusters, random_state=random_state)

    # ── Map clusters → tumour/other by majority vote ──────────────────────────
    print("[H1 exp01] Mapping clusters to tumour/other by majority vote...")
    predicted = match_clusters_to_labels(cluster_ids, true_labels)

    for cid in range(n_clusters):
        mask     = cluster_ids == cid
        majority = "tumour" if predicted[mask][0] == 1 else "other"
        gt_pct   = true_labels[mask].mean() * 100
        print(f"[H1 exp01]   Cluster {cid}: {mask.sum():>7,} patches | "
              f"{gt_pct:.1f}% GT tumour -> '{majority}'")

    # ── Overall metrics ───────────────────────────────────────────────────────
    metrics = classification_metrics(true_labels, predicted)
    print(f"\n[H1 exp01] Overall  accuracy={metrics['accuracy']:.3f}  "
          f"precision={metrics['precision']:.3f}  "
          f"recall={metrics['recall']:.3f}  "
          f"f1={metrics['f1']:.3f}")

    # ── Per-tissue ────────────────────────────────────────────────────────────
    print("\n[H1 exp01] Per-tissue results:")
    tissue_results = []

    for tissue in sorted(manifest["site"].unique()):
        tissue_mask    = manifest["site"] == tissue
        tissue_indices = np.where(tissue_mask)[0]

        tissue_true      = true_labels[tissue_indices]
        tissue_predicted = predicted[tissue_indices]
        tissue_cluster   = cluster_ids[tissue_indices]

        tissue_metrics = classification_metrics(tissue_true, tissue_predicted)

        cluster_breakdown = []
        for cid in range(n_clusters):
            mask = tissue_cluster == cid
            if mask.sum() > 0:
                cluster_breakdown.append({
                    "cluster_id":    cid,
                    "n_patches":     int(mask.sum()),
                    "gt_tumour_pct": round(float(tissue_true[mask].mean() * 100), 1),
                })

        nt = int(tissue_true.sum())
        no = int((tissue_true == 0).sum())
        tissue_results.append({
            "tissue":            tissue,
            "n_patches":         int(tissue_mask.sum()),
            "n_tumour":          nt,
            "n_other":           no,
            "tumour_pct":        round(nt / len(tissue_true) * 100, 1),
            "metrics":           tissue_metrics,
            "cluster_breakdown": cluster_breakdown,
        })

        print(f"\n  {tissue}")
        print(f"    Patches: {tissue_mask.sum():,} | Tumour: {nt:,} | Other: {no:,}")
        print(f"    Accuracy={tissue_metrics['accuracy']:.3f}  "
              f"Precision={tissue_metrics['precision']:.3f}  "
              f"Recall={tissue_metrics['recall']:.3f}")
        for cb in cluster_breakdown:
            print(f"      Cluster {cb['cluster_id']}: {cb['n_patches']:>6,} patches | "
                  f"{cb['gt_tumour_pct']}% GT tumour")

    # ── Save ──────────────────────────────────────────────────────────────────
    summary = {
        "experiment":      cfg["experiment"]["name"],
        "encoder":         encoder,
        "n_clusters":      n_clusters,
        "n_patches":       int(len(manifest)),
        "n_tumour_gt":     n_tumour,
        "n_other_gt":      n_other,
        "metrics_overall": metrics,
        "tissue_results":  tissue_results,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[H1 exp01] Done — summary saved -> {summary_path}")


if __name__ == "__main__":
    _config = Path(__file__).parent / "config.yaml"
    run(_config)
