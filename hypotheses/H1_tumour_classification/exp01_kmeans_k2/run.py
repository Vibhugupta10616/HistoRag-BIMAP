"""
H1 Experiment 01 — K-means k=2 (Dominant-Axis Test)

Pipeline:
    1. Load patch embeddings (provided by user, aligned to manifest)
    2. Derive ground-truth patch labels (tumour/other) from QuPath .geojson annotations
    3. Cluster all patches with K-means k=2 (no labels used during clustering)
    4. Map each cluster to tumour/other by majority vote against ground-truth labels
    5. Compute Accuracy, Precision, Recall
    6. Save summary.json to outputs/

Question answered:
    Is tumour tissue the DOMINANT axis of variation in the encoder's embedding space?
    If yes -> the two clusters should split cleanly along tumour vs other.
    If no  -> the dominant split is something else (stain, tissue type, scanner).
    Either result is informative; comparison across encoders reveals which embeddings
    have tumour as their primary discriminative signal.

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

from histoRAG.labels import tumour_labels_from_geojson
from histoRAG.classify import (
    cluster_embeddings,
    match_clusters_to_labels,
    classification_metrics,
)


def run(config_path: str | Path) -> None:
    # ------------------------------------------------------------------ setup
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["outputs"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------- load embeddings
    emb_path = Path(cfg["inputs"]["embeddings_path"])
    print(f"[H1 exp01] Loading embeddings: {emb_path}")
    embeddings = np.load(emb_path)                      # (N_patches, dim)
    manifest   = pd.read_parquet(cfg["inputs"]["manifest_path"])

    assert len(embeddings) == len(manifest), (
        f"Row count mismatch: embeddings={len(embeddings)}, manifest={len(manifest)}. "
        "Embeddings must be row-aligned to the manifest."
    )
    print(f"[H1 exp01] {len(manifest)} patches | embedding dim = {embeddings.shape[1]}")

    # ------------------------------------------------------- ground-truth labels
    print("[H1 exp01] Deriving ground-truth tumour labels from .geojson annotations...")
    tumour_labels = tumour_labels_from_geojson(manifest, cfg["inputs"]["geojson_dir"])
    true_labels   = (tumour_labels == "tumour").astype(int).values  # 1=tumour, 0=other

    n_tumour = int(true_labels.sum())
    n_other  = int((true_labels == 0).sum())
    print(f"[H1 exp01] Ground truth -> Tumour: {n_tumour} | Other: {n_other}")

    # ------------------------------------------------------ unsupervised clustering
    n_clusters   = cfg["params"].get("n_clusters", 2)
    random_state = cfg["params"].get("random_state", 42)

    print(f"[H1 exp01] Clustering with K-means k={n_clusters}...")
    cluster_ids = cluster_embeddings(embeddings, n_clusters=n_clusters, random_state=random_state)

    # ------------------------------------------ map clusters -> tumour/other labels
    print("[H1 exp01] Mapping clusters to tumour/other by majority vote...")
    predicted = match_clusters_to_labels(cluster_ids, true_labels)

    for cid in range(n_clusters):
        mask = cluster_ids == cid
        majority = "tumour" if predicted[mask][0] == 1 else "other"
        print(f"[H1 exp01]   Cluster {cid}: {mask.sum()} patches -> '{majority}'")

    # ------------------------------------------------------------ evaluate
    metrics = classification_metrics(true_labels, predicted)
    print(f"[H1 exp01] Metrics: {metrics}")

    # ------------------------------------------------------ save summary
    summary = {
        "experiment":    cfg["experiment"]["name"],
        "encoder":       cfg["encoder"],
        "n_clusters":    n_clusters,
        "n_patches":     int(len(manifest)),
        "n_tumour_gt":   n_tumour,
        "n_other_gt":    n_other,
        "metrics":       metrics,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[H1 exp01] Done -- summary saved -> {summary_path}")


if __name__ == "__main__":
    _config = Path(__file__).parent / "config.yaml"
    run(_config)
