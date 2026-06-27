"""
H1 Experiment 03 — HDBSCAN Patch Clustering (Q3)

Pipeline:
    1. Pass 1: Fit IncrementalPCA (n_components_max=20) slide by slide
    2. Select n_components that retain variance_threshold (default 90%, cap 20)
    3. Pass 2: Transform all patches -> collect (pca_emb, gt_label, site)
    4. Stratified subsample: cap any site at site_cap (default 120k) patches
    5. HDBSCAN on subsampled PCA embeddings (~480k total)
    6. Evaluate vs ground truth: ARI, NMI, per-cluster purity, precision/recall/F1
    7. UMAP 2D + 3D on 50k viz subsample, two colour schemes:
         (a) HDBSCAN cluster label   (b) ground-truth tumour / non-tumour
    8. Cache arrays in outputs/{encoder}/cache/ + plots + summary.json

Usage:
    python run.py --encoder conch
    python run.py --encoder conch --skip-hdbscan   # rerun UMAP + plots from HDBSCAN cache
    python run.py --encoder conch --plots-only     # replot from UMAP cache only
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import yaml
from sklearn.cluster import HDBSCAN as hdbscan_lib
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
)

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
    return np.where(
        labels == -1,
        "noise",
        np.array([f"cluster_{i:02d}" for i in labels], dtype=object),
    )


def _compute_metrics(cluster_labels: np.ndarray, gt_labels: np.ndarray) -> dict:
    """Evaluate HDBSCAN clusters against ground-truth tumour annotations."""
    binary_gt = (gt_labels == "tumour").astype(int)

    ari = float(adjusted_rand_score(binary_gt, cluster_labels))
    nmi = float(normalized_mutual_info_score(binary_gt, cluster_labels))

    # Per-cluster purity
    per_cluster = {}
    for c in np.unique(cluster_labels):
        mask    = cluster_labels == c
        n_tot   = int(mask.sum())
        n_tumour = int(binary_gt[mask].sum())
        key     = "noise" if c == -1 else f"cluster_{c:02d}"
        per_cluster[key] = {
            "n_total":    n_tot,
            "n_tumour":   n_tumour,
            "tumour_pct": round(n_tumour / n_tot * 100, 2),
        }

    # Best tumour cluster by highest tumour purity (excluding noise)
    best_cluster, best_pct = None, 0.0
    for c in np.unique(cluster_labels):
        if c == -1:
            continue
        pct = float(binary_gt[cluster_labels == c].mean())
        if pct > best_pct:
            best_pct, best_cluster = pct, int(c)

    if best_cluster is not None:
        pred = (cluster_labels == best_cluster).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            binary_gt, pred, average="binary", zero_division=0
        )
    else:
        prec = rec = f1 = 0.0

    return {
        "ari": round(ari, 4),
        "nmi": round(nmi, 4),
        "best_tumour_cluster": f"cluster_{best_cluster:02d}" if best_cluster is not None else None,
        "best_tumour_cluster_purity_pct": round(best_pct * 100, 2),
        "tumour_precision": round(float(prec), 4),
        "tumour_recall":    round(float(rec),  4),
        "tumour_f1":        round(float(f1),   4),
        "per_cluster_purity": per_cluster,
    }


def _plots(out_dir: Path, cache_dir: Path, encoder: str) -> None:
    """Regenerate all six plots from cached UMAP arrays."""
    umap_2d      = np.load(cache_dir / "umap_coords_2d.npy")
    umap_3d      = np.load(cache_dir / "umap_coords_3d.npy")
    cluster_lbls = np.load(cache_dir / "viz_cluster_labels.npy", allow_pickle=True)
    gt_lbls      = np.load(cache_dir / "viz_gt_labels.npy",      allow_pickle=True)

    cluster_strs = _cluster_label_strings(cluster_lbls)

    plot_umap(umap_2d, labels=cluster_strs,
              out_path=out_dir / "umap_by_cluster.png",
              title=f"HDBSCAN Clusters — {encoder}")
    plot_umap_3d(umap_3d, labels=cluster_strs,
                 out_path=out_dir / "umap_by_cluster_3d.png",
                 title=f"HDBSCAN Clusters 3D — {encoder}")
    plot_umap_3d_interactive(umap_3d, labels=cluster_strs,
                             out_path=out_dir / "umap_by_cluster_3d.html",
                             title=f"HDBSCAN Clusters 3D — {encoder}")

    plot_umap(umap_2d, labels=gt_lbls,
              out_path=out_dir / "umap_by_groundtruth.png",
              title=f"Ground-truth Tumour Labels — {encoder}")
    plot_umap_3d(umap_3d, labels=gt_lbls,
                 out_path=out_dir / "umap_by_groundtruth_3d.png",
                 title=f"Ground-truth Tumour Labels 3D — {encoder}")
    plot_umap_3d_interactive(umap_3d, labels=gt_lbls,
                             out_path=out_dir / "umap_by_groundtruth_3d.html",
                             title=f"Ground-truth Tumour Labels 3D — {encoder}")
    print(f"[H1 exp03] Plots saved -> {out_dir}")


def _umap_and_plots(
    out_dir: Path, cache_dir: Path,
    hdbscan_pca: np.ndarray, cluster_labels: np.ndarray, hdbscan_gt: np.ndarray,
    umap_max: int, umap_kwargs: dict, rng: np.random.Generator, encoder: str,
) -> None:
    """Subsample, compute UMAP 2D + 3D, save cache, generate plots."""
    n_viz   = min(umap_max, len(hdbscan_pca))
    viz_idx = rng.choice(len(hdbscan_pca), n_viz, replace=False)

    viz_emb   = hdbscan_pca[viz_idx]
    viz_clust = cluster_labels[viz_idx]
    viz_gt    = hdbscan_gt[viz_idx]

    print(f"[H1 exp03] Computing 2D UMAP on {n_viz:,} patches...")
    umap_2d = compute_umap(viz_emb, n_components=2, **umap_kwargs)

    print(f"[H1 exp03] Computing 3D UMAP on {n_viz:,} patches...")
    umap_3d = compute_umap(viz_emb, n_components=3, **umap_kwargs)

    np.save(cache_dir / "umap_coords_2d.npy",     umap_2d)
    np.save(cache_dir / "umap_coords_3d.npy",     umap_3d)
    np.save(cache_dir / "viz_cluster_labels.npy", viz_clust)
    np.save(cache_dir / "viz_gt_labels.npy",      viz_gt)
    print("[H1 exp03] UMAP cache saved.")

    _plots(out_dir, cache_dir, encoder)


def run(
    config_path: str | Path,
    encoder_override: str | None = None,
    embeddings_root_override: str | None = None,
    geojson_dir_override: str | None = None,
    plots_only: bool = False,
    skip_hdbscan: bool = False,
) -> None:
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    encoder         = encoder_override or cfg["encoder"]
    geojson_dir     = Path(geojson_dir_override or cfg["inputs"]["geojson_dir"])
    embeddings_root = embeddings_root_override or cfg["inputs"]["embeddings_root"]
    out_dir         = Path(cfg["outputs"]["dir"]) / encoder
    cache_dir       = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    umap_cfg     = cfg["params"].get("umap", {})
    random_state = int(cfg["params"].get("random_state", 42))
    umap_max     = int(umap_cfg.get("max_points", 50_000))
    umap_kwargs  = dict(
        n_neighbors=umap_cfg.get("n_neighbors", 15),
        min_dist=umap_cfg.get("min_dist", 0.1),
        random_state=umap_cfg.get("random_state", 42),
    )
    rng = np.random.default_rng(random_state)

    # ── plots-only: replot from UMAP cache ───────────────────────────────────
    if plots_only:
        required = ["umap_coords_2d.npy", "umap_coords_3d.npy",
                    "viz_cluster_labels.npy", "viz_gt_labels.npy"]
        missing = [f for f in required if not (cache_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"UMAP cache missing in {cache_dir}: {missing}\n"
                "Run without --plots-only first."
            )
        _plots(out_dir, cache_dir, encoder)
        return

    # ── skip-hdbscan: rerun UMAP + plots from HDBSCAN cache ─────────────────
    if skip_hdbscan:
        required = ["hdbscan_pca.npy", "hdbscan_cluster_labels.npy", "hdbscan_gt_labels.npy"]
        missing = [f for f in required if not (cache_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"HDBSCAN cache missing in {cache_dir}: {missing}\n"
                "Run without --skip-hdbscan first."
            )
        hdbscan_pca    = np.load(cache_dir / "hdbscan_pca.npy")
        cluster_labels = np.load(cache_dir / "hdbscan_cluster_labels.npy")
        hdbscan_gt     = np.load(cache_dir / "hdbscan_gt_labels.npy", allow_pickle=True)
        _umap_and_plots(out_dir, cache_dir, hdbscan_pca, cluster_labels, hdbscan_gt,
                        umap_max, umap_kwargs, rng, encoder)
        return

    # ── full run ──────────────────────────────────────────────────────────────
    if not geojson_dir.exists():
        raise FileNotFoundError(
            f"geojson_dir not found: {geojson_dir}\n"
            "Download the HANCOCK .geojson annotation files and update config.yaml."
        )

    pca_cfg      = cfg["params"]["pca"]
    hdbscan_cfg  = cfg["params"]["hdbscan"]
    n_comp_max   = int(pca_cfg["n_components_max"])
    var_thresh   = float(pca_cfg["variance_threshold"])
    min_cls_size = int(hdbscan_cfg["min_cluster_size"])
    min_samples  = hdbscan_cfg.get("min_samples") or None
    site_cap     = int(hdbscan_cfg.get("site_cap", 120_000))

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

    cumvar       = np.cumsum(pca.explained_variance_ratio_)
    n_components = min(int(np.searchsorted(cumvar, var_thresh) + 1), n_comp_max)
    explained    = float(cumvar[n_components - 1])
    print(f"[H1 exp03] PCA: {n_components} components explain {explained*100:.1f}% variance")

    # ── Pass 2: transform + collect (pca_emb, gt_label, site) ────────────────
    print("[H1 exp03] Pass 2 — transforming patches + collecting labels...")
    pca_by_site: dict[str, list] = defaultdict(list)
    gt_by_site:  dict[str, list] = defaultdict(list)

    for emb, slide_df in iter_encoder(encoder, embeddings_root):
        norms   = np.linalg.norm(emb, axis=1, keepdims=True)
        emb     = emb / np.where(norms == 0, 1.0, norms)
        pca_emb = pca.transform(emb)[:, :n_components].astype(np.float32)
        gt_lbl  = tumour_labels_from_geojson(slide_df, geojson_dir, patch_size=patch_size)
        site    = slide_df["site"].iloc[0]
        pca_by_site[site].append(pca_emb)
        gt_by_site[site].append(gt_lbl.values)

    # ── Stratified subsample: cap any site at site_cap ───────────────────────
    print(f"[H1 exp03] Stratified sampling (cap={site_cap:,} per site)...")
    strat_pca, strat_gt, strat_sites = [], [], []
    for site in sorted(pca_by_site):
        s_pca = np.concatenate(pca_by_site[site], axis=0)
        s_gt  = np.concatenate(gt_by_site[site],  axis=0)
        if len(s_pca) > site_cap:
            idx   = rng.choice(len(s_pca), site_cap, replace=False)
            s_pca = s_pca[idx]
            s_gt  = s_gt[idx]
        strat_pca.append(s_pca)
        strat_gt.append(s_gt)
        strat_sites.extend([site] * len(s_pca))
        print(f"  site={site}  patches={len(s_pca):,}")

    hdbscan_pca   = np.concatenate(strat_pca, axis=0)
    hdbscan_gt    = np.concatenate(strat_gt,  axis=0)
    hdbscan_sites = np.array(strat_sites)
    print(f"[H1 exp03] HDBSCAN input: {len(hdbscan_pca):,} patches  ({hdbscan_pca.nbytes/1e9:.2f} GB)")

    # ── HDBSCAN ───────────────────────────────────────────────────────────────
    print("[H1 exp03] Running HDBSCAN...")
    clusterer      = hdbscan_lib(min_cluster_size=min_cls_size, min_samples=min_samples, n_jobs=-1)
    cluster_labels = clusterer.fit_predict(hdbscan_pca)
    n_clusters     = int((np.unique(cluster_labels) > -1).sum())
    n_noise        = int((cluster_labels == -1).sum())
    print(f"[H1 exp03] HDBSCAN: {n_clusters} clusters, {n_noise:,} noise points")

    # ── Save HDBSCAN cache ────────────────────────────────────────────────────
    np.save(cache_dir / "hdbscan_pca.npy",           hdbscan_pca)
    np.save(cache_dir / "hdbscan_cluster_labels.npy", cluster_labels)
    np.save(cache_dir / "hdbscan_gt_labels.npy",      hdbscan_gt)
    np.save(cache_dir / "hdbscan_sites.npy",          hdbscan_sites)
    print("[H1 exp03] HDBSCAN cache saved.")

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = _compute_metrics(cluster_labels, hdbscan_gt)
    print(f"[H1 exp03] ARI={metrics['ari']}  NMI={metrics['nmi']}  "
          f"F1={metrics['tumour_f1']}  best_cluster={metrics['best_tumour_cluster']}")

    # ── UMAP + plots ──────────────────────────────────────────────────────────
    _umap_and_plots(out_dir, cache_dir, hdbscan_pca, cluster_labels, hdbscan_gt,
                    umap_max, umap_kwargs, rng, encoder)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "experiment":             cfg["experiment"]["name"],
        "encoder":                encoder,
        "patch_size":             patch_size,
        "n_patches_total":        total_patches,
        "pca_components":         n_components,
        "pca_variance_explained": round(explained, 4),
        "site_cap":               site_cap,
        "hdbscan_input_n":        int(len(hdbscan_pca)),
        "n_clusters":             n_clusters,
        "n_noise_patches":        n_noise,
        "min_cluster_size":       min_cls_size,
        "metrics":                metrics,
        "outputs": [
            "umap_by_cluster.png", "umap_by_cluster_3d.png", "umap_by_cluster_3d.html",
            "umap_by_groundtruth.png", "umap_by_groundtruth_3d.png", "umap_by_groundtruth_3d.html",
        ],
        "cache": [
            "cache/hdbscan_pca.npy", "cache/hdbscan_cluster_labels.npy",
            "cache/hdbscan_gt_labels.npy", "cache/hdbscan_sites.npy",
            "cache/umap_coords_2d.npy", "cache/umap_coords_3d.npy",
            "cache/viz_cluster_labels.npy", "cache/viz_gt_labels.npy",
        ],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[H1 exp03] Done — outputs in {out_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--encoder",         default=None, help="conch | uni2h")
    parser.add_argument("--embeddings-root", default=None)
    parser.add_argument("--geojson-dir",     default=None)
    parser.add_argument("--plots-only",    action="store_true",
                        help="Replot from cached UMAP coords — no PCA/HDBSCAN/UMAP rerun.")
    parser.add_argument("--skip-hdbscan", action="store_true",
                        help="Reload HDBSCAN cache, rerun UMAP + plots only.")
    args = parser.parse_args()
    run(args.config,
        encoder_override=args.encoder,
        embeddings_root_override=args.embeddings_root,
        geojson_dir_override=args.geojson_dir,
        plots_only=args.plots_only,
        skip_hdbscan=args.skip_hdbscan)
