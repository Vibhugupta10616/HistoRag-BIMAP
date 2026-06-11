"""
H1 Experiment 02 — Over-cluster then Assign (Buried-Signal Test)

Pipeline:
    1. Load patch embeddings (provided by user, aligned to manifest)
    2. Derive ground-truth patch labels (tumour/other) from QuPath .geojson annotations
    3. Cluster all patches with K-means k=8 (no labels used during clustering)
    4. Map each cluster to tumour/other by majority vote against ground-truth labels
    5. Compute Accuracy, Precision, Recall
    6. Save summary.json to outputs/

Question answered:
    Is tumour signal PRESENT in the encoder's embedding space, even if it is not the
    dominant axis of variation?
    Over-clustering (k=8) lets tumour patches form their own sub-cluster even when stain
    or tissue-type variation dominates the top split — giving tumour a chance to be
    recovered even in a weaker encoder.

    Compare with exp01 (k=2):
      - exp01 passes, exp02 passes   -> tumour is the dominant signal (strong encoder)
      - exp01 fails,  exp02 passes   -> tumour signal exists but is buried (weak encoder)
      - exp01 fails,  exp02 fails    -> encoder does not encode tumour-discriminative features

To run:
    python hypotheses/H1_tumour_classification/exp02_overcluster_assign/run.py
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import yaml

from histoRAG.loader import load_encoder
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

    encoder = cfg["encoder"]
    geojson_dir = Path(cfg["inputs"]["geojson_dir"])

    if not geojson_dir.exists():
        raise FileNotFoundError(
            f"geojson_dir not found: {geojson_dir}\n"
            "Download the HANCOCK .geojson annotation files and update config.yaml."
        )

    # -------------------------------------------------------- load embeddings
    print(f"[H1 exp02] Loading embeddings for encoder '{encoder}'...")
    embeddings, manifest = load_encoder(encoder, cfg["inputs"]["embeddings_root"])
    print(f"[H1 exp02] {len(manifest)} patches | embedding dim = {embeddings.shape[1]}")

    # ------------------------------------------------------- ground-truth labels
    print("[H1 exp02] Deriving ground-truth tumour labels from .geojson annotations...")
    tumour_labels = tumour_labels_from_geojson(manifest, cfg["inputs"]["geojson_dir"])
    true_labels   = (tumour_labels == "tumour").astype(int).values  # 1=tumour, 0=other

    n_tumour = int(true_labels.sum())
    n_other  = int((true_labels == 0).sum())
    print(f"[H1 exp02] Ground truth -> Tumour: {n_tumour} | Other: {n_other}")

    # ------------------------------------------------------ unsupervised clustering
    n_clusters   = cfg["params"].get("n_clusters", 8)
    random_state = cfg["params"].get("random_state", 42)

    print(f"[H1 exp02] Clustering with K-means k={n_clusters}...")
    cluster_ids = cluster_embeddings(embeddings, n_clusters=n_clusters, random_state=random_state)

    # ------------------------------------------ map clusters -> tumour/other labels
    print("[H1 exp02] Mapping clusters to tumour/other by majority vote...")
    predicted = match_clusters_to_labels(cluster_ids, true_labels)

    cluster_summary = []
    for cid in range(n_clusters):
        mask     = cluster_ids == cid
        majority = "tumour" if predicted[mask][0] == 1 else "other"
        gt_tumour_pct = float(true_labels[mask].mean() * 100)
        cluster_summary.append({
            "cluster_id":       cid,
            "n_patches":        int(mask.sum()),
            "gt_tumour_pct":    round(gt_tumour_pct, 1),
            "assigned_label":   majority,
        })
        print(
            f"[H1 exp02]   Cluster {cid}: {mask.sum()} patches, "
            f"{gt_tumour_pct:.1f}% tumour GT -> '{majority}'"
        )

    # ------------------------------------------------------------ evaluate
    metrics = classification_metrics(true_labels, predicted)
    print(f"[H1 exp02] Metrics: {metrics}")

    # ------------------------------------------------------ save summary
    summary = {
        "experiment":      cfg["experiment"]["name"],
        "encoder":         encoder,
        "n_clusters":      n_clusters,
        "n_patches":       int(len(manifest)),
        "n_tumour_gt":     n_tumour,
        "n_other_gt":      n_other,
        "metrics":         metrics,
        "cluster_summary": cluster_summary,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[H1 exp02] Done -- summary saved -> {summary_path}")


if __name__ == "__main__":
    _config = Path(__file__).parent / "config.yaml"
    run(_config)
