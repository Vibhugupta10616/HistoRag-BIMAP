"""
2D UMAP visualisation of K-means cluster assignments (k=2 and k=8).

Color  = KMeans cluster label
Marker = tumour (x) vs non-tumour (o) from ground-truth geojson annotations

This makes it visually evident that KMeans clusters do NOT align with
tumour/non-tumour boundaries — the two marker types are mixed within
every cluster.

Usage:
    python hypotheses/H1_tumour_classification/visualize_kmeans_umap.py --encoder conch
    python hypotheses/H1_tumour_classification/visualize_kmeans_umap.py --encoder uni2h
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.cluster import KMeans

from histoRAG.labels import tumour_labels_from_geojson
from histoRAG.loader import detect_patch_size, iter_encoder
from histoRAG.correlate import compute_umap

_HERE      = Path(__file__).parent
_CFG_EXP01 = _HERE / "exp01_kmeans_k2" / "config.yaml"
_OUT_VIS   = _HERE / "exp01_kmeans_k2" / "outputs" / "vis"
_N_SAMPLE  = 50_000
_RANDOM    = 42

# Tableau-10 palette for clusters
_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]


def _subsample_with_labels(
    encoder: str, embeddings_root: str, geojson_dir: Path,
    patch_size: int, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stream slides, proportionally sample n patches per slide,
    return (L2-normalised embeddings, tumour_labels).
    """
    # Estimate ~100 slides; adjust per-slide quota accordingly
    n_per_slide = max(200, n // 80)

    emb_parts, lbl_parts = [], []
    rng = np.random.default_rng(_RANDOM)

    for emb, slide_df in iter_encoder(encoder, embeddings_root):
        gt = tumour_labels_from_geojson(slide_df, geojson_dir, patch_size=patch_size)
        n_take = min(n_per_slide, len(emb))
        idx = rng.choice(len(emb), n_take, replace=False)
        emb_parts.append(emb[idx])
        lbl_parts.append(gt.values[idx])

    arr = np.concatenate(emb_parts, axis=0).astype(np.float32)
    lbl = np.concatenate(lbl_parts, axis=0)

    # Final subsample to exactly n
    if len(arr) > n:
        idx = rng.choice(len(arr), n, replace=False)
        arr, lbl = arr[idx], lbl[idx]

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.where(norms == 0, 1.0, norms)
    return arr, lbl


def _plot(
    coords: np.ndarray,
    cluster_labels: np.ndarray,
    gt_labels: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """
    Scatter plot: color = cluster, marker = tumour (x) vs non-tumour (o).
    """
    unique_clusters = sorted(set(cluster_labels))
    color_map = {c: _COLORS[i % len(_COLORS)] for i, c in enumerate(unique_clusters)}

    fig, ax = plt.subplots(figsize=(10, 7))

    marker_cfg = {
        "tumour": dict(marker="x", s=12, linewidths=0.8, alpha=0.8),
        "other":  dict(marker="o", s=6,  linewidths=0,   alpha=0.25),
    }

    for tissue, mkw in marker_cfg.items():
        mask_t = gt_labels == tissue
        for cl in unique_clusters:
            mask_c = cluster_labels == cl
            mask   = mask_t & mask_c
            if not mask.any():
                continue
            label = f"cluster_{cl} / {tissue}" if tissue == "tumour" else None
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                color=color_map[cl], label=label, **mkw,
            )

    # Legend: cluster colors (from tumour entries) + marker legend
    handles, lbls = ax.get_legend_handles_labels()
    # Append manual marker legend
    from matplotlib.lines import Line2D
    handles += [
        Line2D([0], [0], marker="x", color="grey", linestyle="None",
               markersize=6, label="tumour (×)"),
        Line2D([0], [0], marker="o", color="grey", linestyle="None",
               markersize=5, alpha=0.6, label="non-tumour (○)"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="best", ncol=2)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] Saved -> {out_path}")


def run(encoder: str) -> None:
    with open(_CFG_EXP01) as f:
        cfg = yaml.safe_load(f)

    embeddings_root = cfg["inputs"]["embeddings_root"]
    geojson_dir     = Path(cfg["inputs"]["geojson_dir"])
    patch_size      = detect_patch_size(encoder, embeddings_root)

    print(f"[viz] Subsampling {_N_SAMPLE:,} patches with tumour labels — encoder={encoder}...")
    emb, gt = _subsample_with_labels(encoder, embeddings_root, geojson_dir, patch_size, _N_SAMPLE)
    n_tumour = int((gt == "tumour").sum())
    print(f"[viz] Tumour patches in sample: {n_tumour:,} / {len(gt):,} ({n_tumour/len(gt)*100:.1f}%)")

    print("[viz] Fitting KMeans k=2 ...")
    labels_k2 = KMeans(n_clusters=2, random_state=_RANDOM, n_init=10).fit_predict(emb)

    print("[viz] Fitting KMeans k=8 ...")
    labels_k8 = KMeans(n_clusters=8, random_state=_RANDOM, n_init=10).fit_predict(emb)

    print("[viz] Computing 2D UMAP ...")
    coords = compute_umap(emb, n_components=2, random_state=_RANDOM)

    _OUT_VIS.mkdir(parents=True, exist_ok=True)

    _plot(coords, labels_k2, gt,
          out_path=_OUT_VIS / f"umap_kmeans_k2_{encoder}.png",
          title=f"KMeans k=2 — {encoder}  |  color=cluster  ×=tumour  ○=non-tumour")

    _plot(coords, labels_k8, gt,
          out_path=_OUT_VIS / f"umap_kmeans_k8_{encoder}.png",
          title=f"KMeans k=8 — {encoder}  |  color=cluster  ×=tumour  ○=non-tumour")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True, help="conch | uni2h")
    run(parser.parse_args().encoder)
