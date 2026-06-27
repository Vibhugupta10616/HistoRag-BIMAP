"""
H1 Experiment 01 — K-means k=2 Patch Clustering (Q1)

Pipeline:
    1. Pass 1: proportional subsample -> fit KMeans k=2
    2. Pass 2: stream all patches -> predict cluster IDs + collect UMAP subsample
    3. Majority-vote assignment: map clusters to tumour/other using global tumour rate
    4. Evaluate: overall accuracy/precision/recall/F1, per-tissue, per-WSI
    5. UMAP 2D + 3D on viz subsample, two colour schemes:
         (a) KMeans cluster label   (b) ground-truth tumour / other
    6. Cache arrays in outputs/{encoder}/cache/ + plots + summary.json

Usage:
    python run.py --encoder conch
    python run.py --encoder conch --skip-kmeans   # rerun UMAP + plots from KMeans cache
    python run.py --encoder conch --plots-only    # replot from UMAP cache only
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import yaml

from sklearn.cluster import MiniBatchKMeans

from histoRAG.loader import detect_patch_size, iter_encoder
from histoRAG.labels import tumour_labels_from_geojson
from histoRAG.classify import (
    match_clusters_to_labels,
    classification_metrics,
    cluster_summary,
    grouped_metrics,
)
from histoRAG.correlate import compute_umap, plot_umap, plot_umap_3d, plot_umap_3d_interactive


def _cluster_label_strings(cluster_ids: np.ndarray) -> np.ndarray:
    return np.array([f"cluster_{i:02d}" for i in cluster_ids], dtype=object)


def _plots(out_dir: Path, cache_dir: Path, encoder: str, tag: str) -> None:
    umap_2d      = np.load(cache_dir / "umap_coords_2d.npy")
    umap_3d      = np.load(cache_dir / "umap_coords_3d.npy")
    cluster_lbls = np.load(cache_dir / "viz_cluster_labels.npy", allow_pickle=True)
    gt_lbls      = np.load(cache_dir / "viz_gt_labels.npy",      allow_pickle=True)
    cluster_strs = _cluster_label_strings(cluster_lbls)

    plot_umap(umap_2d, labels=cluster_strs,
              out_path=out_dir / "umap_by_cluster.png",
              title=f"KMeans Clusters — {encoder}")
    plot_umap_3d(umap_3d, labels=cluster_strs,
                 out_path=out_dir / "umap_by_cluster_3d.png",
                 title=f"KMeans Clusters 3D — {encoder}")
    plot_umap_3d_interactive(umap_3d, labels=cluster_strs,
                             out_path=out_dir / "umap_by_cluster_3d.html",
                             title=f"KMeans Clusters 3D — {encoder}")
    plot_umap(umap_2d, labels=gt_lbls,
              out_path=out_dir / "umap_by_groundtruth.png",
              title=f"Ground-truth Labels — {encoder}")
    plot_umap_3d(umap_3d, labels=gt_lbls,
                 out_path=out_dir / "umap_by_groundtruth_3d.png",
                 title=f"Ground-truth Labels 3D — {encoder}")
    plot_umap_3d_interactive(umap_3d, labels=gt_lbls,
                             out_path=out_dir / "umap_by_groundtruth_3d.html",
                             title=f"Ground-truth Labels 3D — {encoder}")
    print(f"[{tag}] Plots saved -> {out_dir}")


def _umap_and_plots(
    out_dir: Path, cache_dir: Path,
    umap_emb: np.ndarray, viz_cluster_labels: np.ndarray, viz_gt_labels: np.ndarray,
    umap_kwargs: dict, encoder: str, tag: str,
) -> None:
    print(f"[{tag}] Computing 2D UMAP on {len(umap_emb):,} patches...")
    umap_2d = compute_umap(umap_emb, n_components=2, **umap_kwargs)
    print(f"[{tag}] Computing 3D UMAP on {len(umap_emb):,} patches...")
    umap_3d = compute_umap(umap_emb, n_components=3, **umap_kwargs)

    np.save(cache_dir / "umap_coords_2d.npy",     umap_2d)
    np.save(cache_dir / "umap_coords_3d.npy",     umap_3d)
    np.save(cache_dir / "viz_cluster_labels.npy", viz_cluster_labels)
    np.save(cache_dir / "viz_gt_labels.npy",      viz_gt_labels)
    print(f"[{tag}] UMAP cache saved.")
    _plots(out_dir, cache_dir, encoder, tag)


def run(
    config_path: str | Path,
    encoder_override: str | None = None,
    embeddings_root_override: str | None = None,
    geojson_dir_override: str | None = None,
    plots_only: bool = False,
    skip_kmeans: bool = False,
) -> None:
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    tag          = cfg["experiment"]["name"]
    encoder      = encoder_override or cfg["encoder"]
    geojson_dir  = Path(geojson_dir_override or cfg["inputs"]["geojson_dir"])
    emb_root     = embeddings_root_override or cfg["inputs"]["embeddings_root"]
    out_dir      = Path(cfg["outputs"]["dir"]) / encoder
    cache_dir    = out_dir / "cache"
    vis_dir      = out_dir / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    n_clusters     = cfg["params"].get("n_clusters", 2)
    random_state   = cfg["params"].get("random_state", 42)
    umap_max       = cfg["params"].get("umap_max_points", 50_000)
    umap_cfg         = cfg["params"].get("umap", {})
    patch_size_cfg   = cfg["params"].get("patch_size", None)
    umap_kwargs = dict(
        n_neighbors=umap_cfg.get("n_neighbors", 15),
        min_dist=umap_cfg.get("min_dist", 0.1),
        random_state=umap_cfg.get("random_state", random_state),
    )
    rng = np.random.default_rng(random_state)

    # ── plots-only: replot from UMAP cache ──────────────────────────────────
    if plots_only:
        required = ["umap_coords_2d.npy", "umap_coords_3d.npy",
                    "viz_cluster_labels.npy", "viz_gt_labels.npy"]
        missing = [f for f in required if not (cache_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"UMAP cache missing in {cache_dir}: {missing}\n"
                "Run without --plots-only first."
            )
        _plots(vis_dir, cache_dir, encoder, tag)
        return

    # ── skip-kmeans: rerun UMAP + plots from KMeans cache ───────────────────
    if skip_kmeans:
        required = ["umap_emb.npy", "viz_cluster_labels.npy", "viz_gt_labels.npy"]
        missing = [f for f in required if not (cache_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"KMeans cache missing in {cache_dir}: {missing}\n"
                "Run without --skip-kmeans first."
            )
        umap_emb           = np.load(cache_dir / "umap_emb.npy")
        viz_cluster_labels = np.load(cache_dir / "viz_cluster_labels.npy", allow_pickle=True)
        viz_gt_labels      = np.load(cache_dir / "viz_gt_labels.npy",      allow_pickle=True)
        _umap_and_plots(vis_dir, cache_dir, umap_emb, viz_cluster_labels, viz_gt_labels,
                        umap_kwargs, encoder, tag)
        return

    # ── full run ─────────────────────────────────────────────────────────────
    if not geojson_dir.exists():
        raise FileNotFoundError(
            f"geojson_dir not found: {geojson_dir}\n"
            "Download the HANCOCK .geojson annotation files and update config.yaml."
        )

    patch_size = patch_size_cfg or detect_patch_size(encoder, emb_root)
    print(f"\n[{tag}] encoder={encoder}  patch_size={patch_size}")

    # Pass 1: stream all patches -> MiniBatchKMeans.partial_fit
    print(f"[{tag}] Pass 1 — fitting MiniBatchKMeans k={n_clusters} on all patches...")
    mbk = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=random_state,
        n_init=3, batch_size=10_000, max_iter=300,
    )
    total_patches = 0
    emb_dim = None
    for emb, slide_df in iter_encoder(encoder, emb_root):
        emb = emb.astype(np.float32)
        mbk.partial_fit(emb)
        total_patches += len(emb)
        if emb_dim is None:
            emb_dim = emb.shape[1]
        print(f"  {slide_df['slide_id'].iloc[0]}: {len(emb):>6,} patches  total={total_patches:,}")
    print(f"[{tag}] Fit complete — {total_patches:,} patches  dim={emb_dim}")

    # Pass 2: predict cluster IDs for all patches + collect UMAP subsample per site
    print(f"[{tag}] Pass 2 — predicting + collecting UMAP subsample...")
    umap_frac       = min(1.0, umap_max / total_patches)
    all_cluster_ids = []
    all_true_labels = []
    manifest_parts  = []
    umap_by_site    = {}  # site -> {emb, cluster, gt}

    for emb, slide_df in iter_encoder(encoder, emb_root):
        slide_id   = slide_df["slide_id"].iloc[0]
        site       = slide_df["site"].iloc[0]
        tumour_lbl = tumour_labels_from_geojson(slide_df, geojson_dir, patch_size=patch_size)
        true_lbl   = (tumour_lbl == "tumour").astype(int).values
        cids       = mbk.predict(emb.astype(np.float32)).astype(np.int32)

        all_cluster_ids.append(cids)
        all_true_labels.append(true_lbl)
        manifest_parts.append(slide_df)

        quota   = min(len(emb), max(1, int(len(emb) * umap_frac)))
        viz_idx = rng.choice(len(emb), quota, replace=False)
        if site not in umap_by_site:
            umap_by_site[site] = {"emb": [], "cluster": [], "gt": []}
        umap_by_site[site]["emb"].append(emb[viz_idx].astype(np.float32))
        umap_by_site[site]["cluster"].append(cids[viz_idx])
        umap_by_site[site]["gt"].append(
            np.where(true_lbl[viz_idx] == 1, "tumour", "other").astype(object)
        )

        print(f"  {slide_id}: {len(emb):>6,} patches | "
              f"tumour={true_lbl.sum():>5,} | other={(true_lbl==0).sum():>5,}")

    cluster_ids = np.concatenate(all_cluster_ids)
    true_labels = np.concatenate(all_true_labels)
    manifest    = pd.concat(manifest_parts, ignore_index=True)

    # Stratified UMAP subsample: equal patches per site
    n_sites  = len(umap_by_site)
    per_site = umap_max // n_sites
    umap_emb_parts, umap_cluster_parts, umap_gt_parts = [], [], []
    for site in sorted(umap_by_site):
        s_emb     = np.concatenate(umap_by_site[site]["emb"],     axis=0)
        s_cluster = np.concatenate(umap_by_site[site]["cluster"])
        s_gt      = np.concatenate(umap_by_site[site]["gt"])
        n   = min(len(s_emb), per_site)
        idx = rng.choice(len(s_emb), n, replace=False)
        umap_emb_parts.append(s_emb[idx])
        umap_cluster_parts.append(s_cluster[idx])
        umap_gt_parts.append(s_gt[idx])
        print(f"[{tag}] UMAP site={site}: {n:,} / {len(s_emb):,} patches")

    umap_emb           = np.concatenate(umap_emb_parts, axis=0)
    viz_cluster_labels = np.concatenate(umap_cluster_parts)
    viz_gt_labels      = np.concatenate(umap_gt_parts)
    print(f"[{tag}] UMAP subsample: {len(umap_emb):,} patches across {n_sites} sites")

    n_tumour = int(true_labels.sum())
    n_other  = int((true_labels == 0).sum())
    print(f"[{tag}] Ground truth — Tumour: {n_tumour:,} | Other: {n_other:,}")

    # Majority-vote cluster assignment
    print(f"[{tag}] Mapping clusters to tumour/other by majority vote...")
    predicted = match_clusters_to_labels(cluster_ids, true_labels)

    cs_df = cluster_summary(cluster_ids, true_labels)
    print(cs_df.to_string(index=False))

    # Metrics
    metrics = classification_metrics(true_labels, predicted)
    print(f"\n[{tag}] Overall  accuracy={metrics['accuracy']:.3f}  "
          f"precision={metrics['precision']:.3f}  "
          f"recall={metrics['recall']:.3f}  "
          f"f1={metrics['f1']:.3f}")

    site_df = grouped_metrics(manifest, true_labels, predicted, group_col="site")
    print(f"\n[{tag}] Per-tissue results:")
    print(site_df.to_string(index=False))
    wsi_df = grouped_metrics(manifest, true_labels, predicted, group_col="slide_id")

    # Save KMeans cache
    np.save(cache_dir / "kmeans_cluster_ids.npy", cluster_ids)
    np.save(cache_dir / "kmeans_true_labels.npy",  true_labels)
    np.save(cache_dir / "kmeans_predicted.npy",    predicted)
    manifest.to_parquet(cache_dir / "manifest.parquet", index=False)
    np.save(cache_dir / "umap_emb.npy",            umap_emb)
    print(f"[{tag}] KMeans cache saved -> {cache_dir}")

    # UMAP + plots
    _umap_and_plots(vis_dir, cache_dir, umap_emb, viz_cluster_labels, viz_gt_labels,
                    umap_kwargs, encoder, tag)

    # Summary
    summary = {
        "experiment":      tag,
        "encoder":         encoder,
        "patch_size":      patch_size,
        "n_clusters":      n_clusters,
        "n_patches":       int(len(manifest)),
        "n_tumour_gt":     n_tumour,
        "n_other_gt":      n_other,
        "embedding_dim":   emb_dim,
        "metrics_note":    "post-hoc cluster alignment — not independent validation",
        "metrics_overall": metrics,
        "cluster_summary": cs_df.to_dict(orient="records"),
        "tissue_results":  site_df.to_dict(orient="records"),
        "wsi_results":     wsi_df.to_dict(orient="records"),
        "outputs": [
            "umap_by_cluster.png", "umap_by_cluster_3d.png", "umap_by_cluster_3d.html",
            "umap_by_groundtruth.png", "umap_by_groundtruth_3d.png",
            "umap_by_groundtruth_3d.html",
        ],
        "cache": [
            "cache/kmeans_cluster_ids.npy", "cache/kmeans_true_labels.npy",
            "cache/kmeans_predicted.npy", "cache/manifest.parquet",
            "cache/umap_emb.npy", "cache/umap_coords_2d.npy", "cache/umap_coords_3d.npy",
            "cache/viz_cluster_labels.npy", "cache/viz_gt_labels.npy",
        ],
    }
    summary_path = out_dir / f"summary_{encoder}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[{tag}] Done — summary saved -> {summary_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="H1 exp01 K-means k=2 patch clustering.")
    parser.add_argument("--config",          default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--encoder",         default=None, help="clip-vitb16 | conch | uni2h")
    parser.add_argument("--embeddings-root", default=None)
    parser.add_argument("--geojson-dir",     default=None)
    parser.add_argument("--skip-kmeans", action="store_true",
                        help="Load KMeans cache, rerun UMAP + plots only.")
    parser.add_argument("--plots-only",  action="store_true",
                        help="Replot from cached UMAP coords — no clustering or UMAP rerun.")
    args = parser.parse_args()
    run(args.config,
        encoder_override=args.encoder,
        embeddings_root_override=args.embeddings_root,
        geojson_dir_override=args.geojson_dir,
        plots_only=args.plots_only,
        skip_kmeans=args.skip_kmeans)
