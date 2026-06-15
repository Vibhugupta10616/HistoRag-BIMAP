"""
H1 Experiment 01 — K-means k=2 (Dominant-Axis Test)

Pipeline:
    1. Load patch embeddings via histoRAG.loader (reads h5 files from embeddings_root)
    2. Derive ground-truth patch labels (tumour/other) from QuPath .geojson annotations
    3. Cluster all patches with K-means k=2 (no labels used during clustering)
    4. Map each cluster to tumour/other by majority vote against ground-truth labels
    5. Compute Accuracy, Precision, Recall per tissue and overall
    6. Save summary.json with per-tissue breakdowns to outputs/

Memory strategy (two-pass):
    Pre-scan  — read h5 shapes only (no data loaded) to get per-slide patch counts.
    Pass 1    — stream slides, sample proportionally up to subsample_target patches,
                fit KMeans on that subsample.
    Pass 2    — stream slides again, predict cluster IDs one slide at a time,
                accumulate results. Peak RAM = one slide at a time.
    ALL 8M+ patches are assigned a cluster in Pass 2 — nothing is skipped.

Question answered:
    Is tumour tissue the DOMINANT axis of variation in the encoder's embedding space?

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

from histoRAG.loader import count_encoder_patches, detect_patch_size, iter_encoder
from histoRAG.labels import tumour_labels_from_geojson
from histoRAG.classify import (
    fit_kmeans,
    match_clusters_to_labels,
    classification_metrics,
)


def run(config_path: str | Path, encoder_override: str | None = None) -> None:
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    encoder          = encoder_override or cfg["encoder"]
    out_dir          = Path(cfg["outputs"]["dir"]) / encoder
    out_dir.mkdir(parents=True, exist_ok=True)
    geojson_dir      = Path(cfg["inputs"]["geojson_dir"])
    emb_root         = cfg["inputs"]["embeddings_root"]
    n_clusters       = cfg["params"].get("n_clusters", 2)
    random_state     = cfg["params"].get("random_state", 42)
    subsample_target  = cfg["params"].get("subsample_target", 2_685_000)
    patch_size_cfg    = cfg["params"].get("patch_size", None)

    if not geojson_dir.exists():
        raise FileNotFoundError(
            f"geojson_dir not found: {geojson_dir}\n"
            "Download the HANCOCK .geojson annotation files and update config.yaml."
        )

    # ── Pre-scan: read shapes only, no embedding data loaded ─────────────────
    print(f"\n[H1 exp01] Pre-scan — counting patches per slide (encoder={encoder})...")
    slide_counts  = count_encoder_patches(encoder, emb_root)
    total_patches = sum(slide_counts.values())
    n_slides      = len(slide_counts)
    sample_frac   = min(1.0, subsample_target / total_patches)

    patch_size = patch_size_cfg if patch_size_cfg else detect_patch_size(encoder, emb_root)
    print(f"[H1 exp01] {n_slides} slides | {total_patches:,} total patches")
    print(f"[H1 exp01] Patch size (WSI level-0 px): {patch_size}"
          f"{' (from config)' if patch_size_cfg else ' (auto-detected)'}")
    print(f"[H1 exp01] Subsample target: {subsample_target:,} ({sample_frac*100:.1f}% of data)")

    # ── Pass 1: proportional subsample per slide → fit KMeans ────────────────
    print(f"\n[H1 exp01] Pass 1 — sampling embeddings for KMeans fit...")
    rng             = np.random.default_rng(random_state)
    subsample_parts = []

    for emb, slide_df in iter_encoder(encoder, emb_root):
        quota = max(10, int(len(emb) * sample_frac))
        idx   = rng.choice(len(emb), quota, replace=False)
        subsample_parts.append(emb[idx])

    sample_emb   = np.concatenate(subsample_parts, axis=0).astype(np.float32)
    actual_sample = len(sample_emb)
    emb_dim       = sample_emb.shape[1]
    print(f"[H1 exp01] Fitting KMeans k={n_clusters} on {actual_sample:,} patches "
          f"(dim={emb_dim}, ≈{actual_sample * emb_dim * 4 / 1e9:.2f} GB)")

    km = fit_kmeans(sample_emb, n_clusters=n_clusters, random_state=random_state, n_init=3)
    del sample_emb

    # ── Pass 2: predict ALL patches one slide at a time ───────────────────────
    print(f"\n[H1 exp01] Pass 2 — predicting cluster IDs for all {total_patches:,} patches...")
    all_cluster_ids = []
    all_true_labels = []
    manifest_parts  = []

    for emb, slide_df in iter_encoder(encoder, emb_root):
        slide_id = slide_df["slide_id"].iloc[0]

        tumour_lbl = tumour_labels_from_geojson(slide_df, geojson_dir, patch_size=patch_size)
        true_lbl   = (tumour_lbl == "tumour").astype(int).values

        cluster_ids_slide = km.predict(emb).astype(np.int32)

        all_cluster_ids.append(cluster_ids_slide)
        all_true_labels.append(true_lbl)
        manifest_parts.append(slide_df)

        print(f"  {slide_id}: {len(emb):>6,} patches | "
              f"tumour={true_lbl.sum():>5,} | other={(true_lbl==0).sum():>5,}")

    cluster_ids = np.concatenate(all_cluster_ids)
    true_labels = np.concatenate(all_true_labels)
    manifest    = pd.concat(manifest_parts, ignore_index=True)

    n_tumour = int(true_labels.sum())
    n_other  = int((true_labels == 0).sum())
    print(f"\n[H1 exp01] Ground truth — Tumour: {n_tumour:,} | Other: {n_other:,}")

    # ── Map clusters → tumour/other by majority vote ──────────────────────────
    print("[H1 exp01] Mapping clusters to tumour/other by majority vote...")
    predicted = match_clusters_to_labels(cluster_ids, true_labels)

    for cid in range(n_clusters):
        mask = cluster_ids == cid
        if mask.sum() == 0:
            print(f"[H1 exp01]   Cluster {cid}: empty")
            continue
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

    # ── Per-WSI metrics ───────────────────────────────────────────────────────
    wsi_results = []
    for slide_id in sorted(manifest["slide_id"].unique()):
        slide_mask    = manifest["slide_id"] == slide_id
        slide_indices = np.where(slide_mask)[0]
        slide_true    = true_labels[slide_indices]
        slide_pred    = predicted[slide_indices]
        slide_metrics = classification_metrics(slide_true, slide_pred)
        site          = manifest.loc[slide_mask, "site"].iloc[0]
        wsi_results.append({
            "slide_id":   slide_id,
            "site":       site,
            "n_patches":  int(slide_mask.sum()),
            "n_tumour":   int(slide_true.sum()),
            "tumour_pct": round(float(slide_true.mean() * 100), 1),
            "metrics":    slide_metrics,
        })

    # ── Save ──────────────────────────────────────────────────────────────────
    # NOTE: metrics are POST-HOC CLUSTER ALIGNMENT — the same GeoJSON labels
    # are used for both majority-vote assignment and evaluation. This is standard
    # for unsupervised clustering analysis but is not independent validation.
    summary = {
        "experiment":             cfg["experiment"]["name"],
        "encoder":                encoder,
        "patch_size":             patch_size,
        "n_clusters":             n_clusters,
        "n_patches":              int(len(manifest)),
        "n_tumour_gt":            n_tumour,
        "n_other_gt":             n_other,
        "subsample_used":         actual_sample,
        "embedding_dim":          emb_dim,
        "metrics_note":           "post-hoc cluster alignment — not independent validation",
        "metrics_overall":        metrics,
        "tissue_results":         tissue_results,
        "wsi_results":            wsi_results,
    }
    summary_path = out_dir / f"summary_{encoder}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[H1 exp01] Done — summary saved -> {summary_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--encoder", default=None, help="Override encoder (clip-vitb16 | conch | uni2h)")
    args = parser.parse_args()
    run(args.config, encoder_override=args.encoder)
