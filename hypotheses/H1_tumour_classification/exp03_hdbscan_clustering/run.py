"""
H1 Experiment 03 — HDBSCAN Patch Clustering (Q3)

Pipeline:
    1. Pass 1: Fit IncrementalPCA (n_components_max=20) slide by slide
    2. Select n_components that retain variance_threshold (default 90%, cap 20)
    3. Pass 2: Transform all patches to PCA space; collect ground-truth tumour labels
    4. HDBSCAN on ALL PCA-reduced embeddings (~640 MB at 8M × 20d — tractable)
    5. UMAP 2D + 3D + interactive HTML, two colour schemes:
         (a) HDBSCAN cluster label   (b) ground-truth tumour / non-tumour
    6. Cache arrays + plots + summary.json

Usage:
    python run.py --encoder conch
    python run.py --encoder conch --plots-only      # regenerate plots from cache
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import HDBSCAN as hdbscan_lib
from sklearn.decomposition import IncrementalPCA

from histoRAG.labels import tumour_labels_from_geojson
from histoRAG.loader import detect_patch_size, iter_encoder
from histoRAG.correlate import (
    compute_umap,
    plot_umap,
    plot_umap_3d,
    plot_umap_3d_interactive,
)


def _cluster_label_strings(labels: np.ndarray) -> np.ndarray:
    """Convert integer HDBSCAN labels to strings; -1 becomes 'noise'."""
    return np.where(labels == -1, "noise",
                    np.array([f"cluster_{i:02d}" for i in labels], dtype=object))


def _plots(out_dir: Path, encoder: str) -> None:
    """Regenerate all six plots from cached arrays. No embedding scan needed."""
    umap_2d      = np.load(out_dir / "cache_umap_coords_2d.npy")
    umap_3d      = np.load(out_dir / "cache_umap_coords_3d.npy")
    cluster_lbls = np.load(out_dir / "cache_viz_cluster_labels.npy", allow_pickle=True)
    gt_lbls      = np.load(out_dir / "cache_viz_gt_labels.npy",      allow_pickle=True)

    cluster_strs = _cluster_label_strings(cluster_lbls)

    for coords, suffix, fn2d, fn3d, fnhtml in [
        (umap_2d, "2D", "umap_by_cluster.png",     "umap_by_cluster_3d.png",     "umap_by_cluster_3d.html"),
    ]:
        _ = suffix  # unused, kept for readability of the loop structure
        plot_umap(umap_2d, labels=cluster_strs,
                  out_path=out_dir / "umap_by_cluster.png",
                  title=f"HDBSCAN Clusters — {encoder}")
        plot_umap_3d(umap_3d, labels=cluster_strs,
                     out_path=out_dir / "umap_by_cluster_3d.png",
                     title=f"HDBSCAN Clusters 3D — {encoder}")
        plot_umap_3d_interactive(umap_3d, labels=cluster_strs,
                                 out_path=out_dir / "umap_by_cluster_3d.html",
                                 title=f"HDBSCAN Clusters 3D — {encoder}")
        break  # single iteration — avoids duplicating the block

    plot_umap(umap_2d, labels=gt_lbls,
              out_path=out_dir / "umap_by_groundtruth.png",
              title=f"Ground-truth Tumour Labels — {encoder}")
    plot_umap_3d(umap_3d, labels=gt_lbls,
                 out_path=out_dir / "umap_by_groundtruth_3d.png",
                 title=f"Ground-truth Tumour Labels 3D — {encoder}")
    plot_umap_3d_interactive(umap_3d, labels=gt_lbls,
                             out_path=out_dir / "umap_by_groundtruth_3d.html",
                             title=f"Ground-truth Tumour Labels 3D — {encoder}")
    print(f"[H1 exp03] Plots regenerated -> {out_dir}")


def run(
    config_path: str | Path,
    encoder_override: str | None = None,
    embeddings_root_override: str | None = None,
    geojson_dir_override: str | None = None,
    plots_only: bool = False,
) -> None:
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    encoder         = encoder_override or cfg["encoder"]
    geojson_dir     = Path(geojson_dir_override or cfg["inputs"]["geojson_dir"])
    embeddings_root = embeddings_root_override or cfg["inputs"]["embeddings_root"]
    out_dir         = Path(cfg["outputs"]["dir"]) / encoder
    out_dir.mkdir(parents=True, exist_ok=True)

    if plots_only:
        required = [
            "cache_umap_coords_2d.npy", "cache_umap_coords_3d.npy",
            "cache_viz_cluster_labels.npy", "cache_viz_gt_labels.npy",
        ]
        missing = [f for f in required if not (out_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Cache files missing in {out_dir}: {missing}\n"
                "Run without --plots-only first."
            )
        _plots(out_dir, encoder)
        return

    if not geojson_dir.exists():
        raise FileNotFoundError(
            f"geojson_dir not found: {geojson_dir}\n"
            "Download the HANCOCK .geojson annotation files and update config.yaml."
        )

    pca_cfg      = cfg["params"]["pca"]
    hdbscan_cfg  = cfg["params"]["hdbscan"]
    umap_cfg     = cfg["params"].get("umap", {})
    random_state = int(cfg["params"].get("random_state", 42))
    n_comp_max   = int(pca_cfg["n_components_max"])
    var_thresh   = float(pca_cfg["variance_threshold"])
    min_cls_size = int(hdbscan_cfg["min_cluster_size"])
    min_samples  = hdbscan_cfg.get("min_samples") or None
    umap_max     = int(umap_cfg.get("max_points", 50_000))

    patch_size = cfg["params"].get("patch_size") or detect_patch_size(encoder, embeddings_root)
    print(f"\n[H1 exp03] encoder={encoder}  patch_size={patch_size}")

    # ── Pass 1: fit IncrementalPCA ────────────────────────────────────────────
    print(f"[H1 exp03] Pass 1 — fitting IncrementalPCA (n_components={n_comp_max})...")
    pca = IncrementalPCA(n_components=n_comp_max)
    total_patches = 0

    for emb, slide_df in iter_encoder(encoder, embeddings_root):
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb   = emb / np.where(norms == 0, 1.0, norms)
        pca.partial_fit(emb)
        total_patches += len(emb)
        print(f"  PCA fit: {slide_df['slide_id'].iloc[0]}  total={total_patches:,}")

    cumvar     = np.cumsum(pca.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, var_thresh) + 1)
    n_components = min(n_components, n_comp_max)
    explained  = float(cumvar[n_components - 1])
    print(f"[H1 exp03] PCA: {n_components} components explain {explained*100:.1f}% variance")

    # ── Pass 2: transform all patches, collect ground-truth labels ────────────
    print("[H1 exp03] Pass 2 — transforming patches to PCA space...")
    pca_parts, gt_parts = [], []

    for emb, slide_df in iter_encoder(encoder, embeddings_root):
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb   = emb / np.where(norms == 0, 1.0, norms)
        pca_emb = pca.transform(emb)[:, :n_components].astype(np.float32)

        gt_lbl = tumour_labels_from_geojson(slide_df, geojson_dir, patch_size=patch_size)
        pca_parts.append(pca_emb)
        gt_parts.append(gt_lbl.values)

    all_pca = np.concatenate(pca_parts, axis=0)   # (N, n_components)  ~640 MB max
    all_gt  = np.concatenate(gt_parts,  axis=0)   # (N,) "tumour"/"other"
    print(f"[H1 exp03] PCA embeddings: {all_pca.shape}  ({all_pca.nbytes/1e9:.2f} GB)")

    # ── HDBSCAN clustering ────────────────────────────────────────────────────
    # PCA already reduced to ≤20d so running on all patches is tractable (~640 MB).
    rng = np.random.default_rng(random_state)
    print(f"[H1 exp03] HDBSCAN on all {len(all_pca):,} PCA-reduced patches...")

    clusterer = hdbscan_lib(
        min_cluster_size=min_cls_size,
        min_samples=min_samples,
        n_jobs=-1,
    )
    cluster_labels = clusterer.fit_predict(all_pca)

    n_clusters = int((np.unique(cluster_labels) > -1).sum())
    n_noise    = int((cluster_labels == -1).sum())
    print(f"[H1 exp03] HDBSCAN found {n_clusters} clusters, {n_noise:,} noise points")

    # ── UMAP visualisation subsample ─────────────────────────────────────────
    n_viz     = min(umap_max, len(all_pca))
    viz_idx   = rng.choice(len(all_pca), n_viz, replace=False)
    viz_emb   = all_pca[viz_idx]
    viz_clust = cluster_labels[viz_idx]
    viz_gt    = all_gt[viz_idx]

    umap_kwargs = dict(
        n_neighbors=umap_cfg.get("n_neighbors", 15),
        min_dist=umap_cfg.get("min_dist", 0.1),
        random_state=umap_cfg.get("random_state", 42),
    )
    print(f"[H1 exp03] Computing 2D UMAP on {n_viz:,} patches...")
    umap_2d = compute_umap(viz_emb, n_components=2, **umap_kwargs)

    print(f"[H1 exp03] Computing 3D UMAP on {n_viz:,} patches...")
    umap_3d = compute_umap(viz_emb, n_components=3, **umap_kwargs)

    # ── Cache ─────────────────────────────────────────────────────────────────
    np.save(out_dir / "cache_umap_coords_2d.npy",    umap_2d)
    np.save(out_dir / "cache_umap_coords_3d.npy",    umap_3d)
    np.save(out_dir / "cache_viz_cluster_labels.npy", viz_clust)
    np.save(out_dir / "cache_viz_gt_labels.npy",      viz_gt)
    print("[H1 exp03] Cache saved.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plots(out_dir, encoder)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "experiment":        cfg["experiment"]["name"],
        "encoder":           encoder,
        "patch_size":        patch_size,
        "n_patches_total":   total_patches,
        "pca_components":    n_components,
        "pca_variance_explained": round(explained, 4),
        "hdbscan_input_n":   int(len(all_pca)),
        "n_clusters":        n_clusters,
        "n_noise_patches":   n_noise,
        "min_cluster_size":  min_cls_size,
        "outputs": [
            "umap_by_cluster.png", "umap_by_cluster_3d.png", "umap_by_cluster_3d.html",
            "umap_by_groundtruth.png", "umap_by_groundtruth_3d.png", "umap_by_groundtruth_3d.html",
        ],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[H1 exp03] Done — outputs in {out_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--encoder", default=None, help="conch | uni2h")
    parser.add_argument("--embeddings-root", default=None)
    parser.add_argument("--geojson-dir", default=None)
    parser.add_argument("--plots-only", action="store_true",
                        help="Regenerate plots from cached arrays; skip embedding scan.")
    args = parser.parse_args()
    run(args.config,
        encoder_override=args.encoder,
        embeddings_root_override=args.embeddings_root,
        geojson_dir_override=args.geojson_dir,
        plots_only=args.plots_only)
