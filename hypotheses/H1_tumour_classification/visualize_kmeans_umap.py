"""
2D UMAP visualisation of K-means cluster assignments (k=2 and k=8), plus a
ground-truth tumour/other map over the same sampled points.

Subsamples 50k patches, fits both KMeans models, computes one shared UMAP
embedding, and saves PNGs into the existing exp01 / exp02 output dirs.

Usage:
    python hypotheses/H1_tumour_classification/visualize_kmeans_umap.py --encoder conch
    python hypotheses/H1_tumour_classification/visualize_kmeans_umap.py --encoder uni2h
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans

from histoRAG.loader import count_encoder_patches, detect_patch_size, iter_encoder
from histoRAG.labels import tumour_labels_from_geojson
from histoRAG.correlate import compute_umap, plot_umap

_HERE          = Path(__file__).parent
_CFG_EXP01     = _HERE / "exp01_kmeans_k2" / "config.yaml"
_OUT_VIS_EXP01 = _HERE / "exp01_kmeans_k2" / "outputs" / "vis"  # k=2 plot
_OUT_VIS_EXP02 = _HERE / "exp02_overcluster_assign" / "outputs" / "vis"  # k=8 plot
_N_SAMPLE      = 50_000
_RANDOM        = 42


def _subsample(encoder: str, embeddings_root: str, n: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Stream slides, proportionally sample n patches, return L2-normalised
    embeddings plus the aligned manifest rows (slide_id, site, x, y)."""
    rng = np.random.default_rng(_RANDOM)
    total = sum(count_encoder_patches(encoder, embeddings_root).values())
    emb_parts, manifest_parts = [], []
    for emb, slide_df in iter_encoder(encoder, embeddings_root):
        quota = max(1, round(n * len(emb) / total))
        idx = rng.choice(len(emb), min(quota, len(emb)), replace=False)
        emb_parts.append(emb[idx])
        manifest_parts.append(slide_df.iloc[idx])
    arr = np.concatenate(emb_parts, axis=0, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    manifest = pd.concat(manifest_parts, ignore_index=True)
    return arr / np.where(norms == 0, 1.0, norms), manifest


def run(encoder: str) -> None:
    with open(_CFG_EXP01) as f:
        cfg = yaml.safe_load(f)
    embeddings_root = cfg["inputs"]["embeddings_root"]
    geojson_dir     = cfg["inputs"]["geojson_dir"]

    print(f"[viz] Subsampling {_N_SAMPLE:,} patches — encoder={encoder}...")
    emb, manifest = _subsample(encoder, embeddings_root, _N_SAMPLE)

    print("[viz] Fitting KMeans k=2 ...")
    labels_k2 = KMeans(n_clusters=2, random_state=_RANDOM, n_init=10).fit_predict(emb)

    print("[viz] Fitting KMeans k=8 ...")
    labels_k8 = KMeans(n_clusters=8, random_state=_RANDOM, n_init=10).fit_predict(emb)

    print("[viz] Computing 2D UMAP ...")
    coords = compute_umap(emb, n_components=2, random_state=_RANDOM)

    print("[viz] Deriving ground-truth tumour/other labels from .geojson ...")
    patch_size = detect_patch_size(encoder, embeddings_root)
    gt_labels = tumour_labels_from_geojson(manifest, geojson_dir, patch_size=patch_size).values

    _OUT_VIS_EXP01.mkdir(parents=True, exist_ok=True)
    _OUT_VIS_EXP02.mkdir(parents=True, exist_ok=True)

    plot_umap(coords, labels=np.array([f"cluster_{i}" for i in labels_k2]),
              out_path=_OUT_VIS_EXP01 / f"umap_kmeans_k2_{encoder}.png",
              title=f"KMeans k=2 — {encoder} (n={_N_SAMPLE:,})")

    plot_umap(coords, labels=np.array([f"cluster_{i}" for i in labels_k8]),
              out_path=_OUT_VIS_EXP02 / f"umap_kmeans_k8_{encoder}.png",
              title=f"KMeans k=8 — {encoder} (n={_N_SAMPLE:,})")

    for out_dir in (_OUT_VIS_EXP01, _OUT_VIS_EXP02):
        plot_umap(coords, labels=gt_labels,
                  out_path=out_dir / f"umap_groundtruth_{encoder}.png",
                  title=f"Ground truth — {encoder} (n={_N_SAMPLE:,})")

    print(f"[viz] k=2 plot -> {_OUT_VIS_EXP01}")
    print(f"[viz] k=8 plot -> {_OUT_VIS_EXP02}")
    print(f"[viz] ground-truth plot -> {_OUT_VIS_EXP01} and {_OUT_VIS_EXP02}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True, help="conch | uni2h")
    run(parser.parse_args().encoder)
