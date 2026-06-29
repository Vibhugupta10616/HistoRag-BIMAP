"""
HackDay2 — Single-WSI HDBSCAN Pipeline

Steps:
  1. Patch the WSI (openslide, tissue-aware background removal)
  2. Embed patches with CONCH (frozen)
  3. Save embeddings -> outputs/<slide_id>/embeddings.h5
  4. PCA (optional) + HDBSCAN
  5. UMAP 2D -> two plots:
       (a) coloured by HDBSCAN cluster label
       (b) coloured by ground-truth tumour / other  [only if --geojson is given]

Usage:
  python pipeline.py --wsi path/to/slide.svs
  python pipeline.py --wsi path/to/slide.svs --geojson path/to/slide.geojson
  python pipeline.py --wsi path/to/slide.svs --skip-embed   # reuse saved embeddings.h5
  python pipeline.py --wsi path/to/slide.svs --plots-only   # replot from UMAP cache

Requirements (in addition to requirements.txt):
  pip install git+https://github.com/mahmoodlab/CONCH.git
  huggingface-cli login   # needed for hf_hub checkpoint; or set HF_TOKEN env var
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from sklearn.cluster import HDBSCAN as hdbscan_lib
from sklearn.decomposition import PCA
from tqdm import tqdm

from histoRAG.correlate import compute_umap, plot_umap
from histoRAG.labels import tumour_labels_from_geojson


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------

def _tissue_mask(slide, patch_size: int, level: int) -> np.ndarray:
    """Return bool mask [grid_h, grid_w]: True = patch has enough tissue."""
    w, h = slide.level_dimensions[level]
    grid_w = max(w // patch_size, 1)
    grid_h = max(h // patch_size, 1)

    # get_thumbnail preserves aspect ratio, so resize explicitly to exact grid dims
    thumb = slide.get_thumbnail((grid_w * 4, grid_h * 4))  # oversample to avoid aliasing
    thumb = thumb.convert("L").resize((grid_w, grid_h), Image.LANCZOS)
    thumb_arr = np.array(thumb, dtype=np.float32) / 255.0  # 0=black, 1=white
    return thumb_arr < 0.85   # tissue is darker than background


def patch_wsi(
    wsi_path: str | Path,
    patch_size: int,
    level: int,
    tissue_threshold: float,
) -> tuple[list[Image.Image], list[tuple[int, int]]]:
    """
    Tile a WSI and return (patches, coords) for tissue-containing patches.

    coords: list of (x, y) top-left corner positions at level-0 pixel space.
    """
    import openslide
    slide = openslide.OpenSlide(str(wsi_path))
    mask = _tissue_mask(slide, patch_size, level)

    # level-0 pixel size of one patch at the requested level
    ds = int(slide.level_downsamples[level])
    stride = patch_size * ds   # step in level-0 coordinates

    patches, coords = [], []
    grid_h, grid_w = mask.shape
    for ty in range(grid_h):
        for tx in range(grid_w):
            if not mask[ty, tx]:
                continue
            x0 = tx * stride
            y0 = ty * stride
            patch = slide.read_region((x0, y0), level, (patch_size, patch_size)).convert("RGB")
            arr = np.array(patch, dtype=np.float32) / 255.0
            # secondary check: discard nearly all-white patches
            if arr.mean() > (1.0 - tissue_threshold):
                continue
            patches.append(patch)
            coords.append((x0, y0))   # level-0 coords (what geojson uses)

    slide.close()
    return patches, coords


# ---------------------------------------------------------------------------
# CONCH embedding
# ---------------------------------------------------------------------------

def load_conch(checkpoint: str, device: torch.device):
    try:
        from conch.open_clip_custom import create_model_from_pretrained
    except ImportError:
        sys.exit(
            "CONCH not installed. Run:\n"
            "  pip install git+https://github.com/mahmoodlab/CONCH.git\n"
            "Then ensure you have HuggingFace access to MahmoodLab/conch."
        )
    model, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint)
    model = model.eval().to(device)
    return model, preprocess


@torch.inference_mode()
def embed_patches(
    patches: list[Image.Image],
    model,
    preprocess,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Return (N, dim) float32 embeddings for a list of PIL patches."""
    all_emb = []
    for i in tqdm(range(0, len(patches), batch_size), desc="Embedding"):
        batch = torch.stack([preprocess(p) for p in patches[i : i + batch_size]]).to(device)
        emb = model.encode_image(batch, normalize=True)
        all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# H5 I/O (same schema as CONCH files from loader.py)
# ---------------------------------------------------------------------------

def save_embeddings_h5(path: Path, embeddings: np.ndarray, xs: list, ys: list) -> None:
    with h5py.File(path, "w") as hf:
        hf.create_dataset("embeddings", data=embeddings)
        hf.create_dataset("x", data=np.array(xs, dtype=np.int32))
        hf.create_dataset("y", data=np.array(ys, dtype=np.int32))


def load_embeddings_h5(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as hf:
        emb = np.array(hf["embeddings"], dtype=np.float32)
        x   = np.array(hf["x"], dtype=np.int32)
        y   = np.array(hf["y"], dtype=np.int32)
    return emb, x, y


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(cluster_labels: np.ndarray, gt_labels: np.ndarray) -> dict:
    """Evaluate HDBSCAN clusters against ground-truth tumour annotations."""
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        precision_recall_fscore_support,
    )

    binary_gt = (gt_labels == "tumour").astype(int)

    ari = float(adjusted_rand_score(binary_gt, cluster_labels))
    nmi = float(normalized_mutual_info_score(binary_gt, cluster_labels))

    # per-cluster purity
    per_cluster = {}
    for c in np.unique(cluster_labels):
        mask     = cluster_labels == c
        n_tot    = int(mask.sum())
        n_tumour = int(binary_gt[mask].sum())
        key      = "noise" if c == -1 else f"cluster_{c:02d}"
        per_cluster[key] = {
            "n_total":    n_tot,
            "n_tumour":   n_tumour,
            "tumour_pct": round(n_tumour / n_tot * 100, 2),
        }

    # best tumour cluster = highest tumour purity (excluding noise)
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
        "ari":                          round(ari, 4),
        "nmi":                          round(nmi, 4),
        "best_tumour_cluster":          f"cluster_{best_cluster:02d}" if best_cluster is not None else None,
        "best_tumour_cluster_purity_%": round(best_pct * 100, 2),
        "tumour_precision":             round(float(prec), 4),
        "tumour_recall":                round(float(rec),  4),
        "tumour_f1":                    round(float(f1),   4),
        "per_cluster_purity":           per_cluster,
    }


# ---------------------------------------------------------------------------
# Results markdown
# ---------------------------------------------------------------------------

def _write_results_md(out_dir: Path, summary: dict) -> None:
    m = summary.get("metrics")
    lines = [
        f"# Results — {summary['slide_id']}",
        "",
        "## Run info",
        f"| | |",
        f"|---|---|",
        f"| Patches | {summary['n_patches']:,} |",
        f"| Embedding dim | {summary['embedding_dim']} |",
        f"| PCA components | {summary['pca_components']} |",
        f"| HDBSCAN clusters | {summary['n_clusters']} |",
        f"| Noise patches | {summary['n_noise_patches']:,} |",
        f"| UMAP points | {summary['umap_n_viz']:,} |",
        "",
    ]

    if m is None:
        lines += [
            "## Metrics",
            "_No annotations provided — run with a .geojson file for metrics._",
        ]
    else:
        lines += [
            "## Metrics vs ground-truth tumour annotations",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| ARI | {m['ari']} |",
            f"| NMI | {m['nmi']} |",
            f"| Best tumour cluster | {m['best_tumour_cluster']} |",
            f"| Best cluster purity | {m['best_tumour_cluster_purity_%']}% |",
            f"| Tumour precision | {m['tumour_precision']} |",
            f"| Tumour recall | {m['tumour_recall']} |",
            f"| Tumour F1 | {m['tumour_f1']} |",
            "",
            "## Per-cluster purity",
            "",
            "| Cluster | Total patches | Tumour patches | Tumour % |",
            "|---|---|---|---|",
        ]
        for cluster, stats in m["per_cluster_purity"].items():
            lines.append(
                f"| {cluster} | {stats['n_total']:,} | {stats['n_tumour']:,} | {stats['tumour_pct']}% |"
            )

    with open(out_dir / "results.md", "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Cluster-label helper
# ---------------------------------------------------------------------------

def _cluster_strings(labels: np.ndarray) -> np.ndarray:
    return np.array(["noise" if l == -1 else f"cluster_{l:02d}" for l in labels])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    cfg: dict,
    wsi_path: str | Path,
    geojson_path: str | Path | None,
    skip_embed: bool,
    plots_only: bool,
) -> None:
    wsi_path = Path(wsi_path)
    slide_id = wsi_path.stem

    out_dir   = Path(cfg["outputs_dir"]) / slide_id
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    h5_path      = cache_dir / "embeddings.h5"
    umap_2d_path = cache_dir / "umap_2d.npy"
    clust_path   = cache_dir / "cluster_labels.npy"
    gt_path      = cache_dir / "gt_labels.npy"

    umap_cfg = cfg["umap"]

    # ── plots-only: replot from cached UMAP arrays ───────────────────────────
    if plots_only:
        for p in [umap_2d_path, clust_path]:
            if not p.exists():
                sys.exit(f"Cache missing: {p}\nRun without --plots-only first.")
        umap_2d       = np.load(umap_2d_path)
        cluster_lbls  = np.load(clust_path, allow_pickle=True)
        cluster_strs  = _cluster_strings(cluster_lbls)
        plot_umap(umap_2d, labels=cluster_strs,
                  out_path=out_dir / "umap_by_cluster.png",
                  title=f"HDBSCAN Clusters — {slide_id}")
        if gt_path.exists():
            gt_lbls = np.load(gt_path, allow_pickle=True)
            plot_umap(umap_2d, labels=gt_lbls,
                      out_path=out_dir / "umap_by_groundtruth.png",
                      title=f"Ground-truth Labels — {slide_id}")
        print(f"Plots saved -> {out_dir}")
        return

    # ── step 1+2: patch + embed (or load cached H5) ──────────────────────────
    if skip_embed and h5_path.exists():
        print(f"[embed] Loading cached embeddings from {h5_path}")
        emb, xs, ys = load_embeddings_h5(h5_path)
    else:
        if skip_embed:
            print(f"[warn] --skip-embed set but no cache found at {h5_path}, running full embed")
        if not wsi_path.exists():
            sys.exit(f"WSI not found: {wsi_path}")

        patch_cfg = cfg["patching"]
        print(f"[patch] Tiling {wsi_path.name}  patch_size={patch_cfg['patch_size']}  level={patch_cfg['level']}")
        patches, coords = patch_wsi(
            wsi_path,
            patch_size=patch_cfg["patch_size"],
            level=patch_cfg["level"],
            tissue_threshold=patch_cfg["tissue_threshold"],
        )
        print(f"[patch] {len(patches):,} tissue patches extracted")
        if not patches:
            sys.exit("No tissue patches found — check tissue_threshold or WSI path.")

        max_p = patch_cfg.get("max_patches") or None
        if max_p and len(patches) > int(max_p):
            rng_p = np.random.default_rng(42)
            keep  = rng_p.choice(len(patches), int(max_p), replace=False)
            patches = [patches[i] for i in keep]
            coords  = [coords[i]  for i in keep]
            print(f"[patch] Subsampled to {len(patches):,} patches (max_patches={max_p})")

        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[embed] Device: {device}")
        print(f"[embed] Loading CONCH from {cfg['conch']['checkpoint']}")
        model, preprocess = load_conch(cfg["conch"]["checkpoint"], device)

        emb = embed_patches(patches, model, preprocess, device, patch_cfg["batch_size"])
        save_embeddings_h5(h5_path, emb, xs, ys)
        print(f"[embed] Saved -> {h5_path}  shape={emb.shape}")

        del patches  # free memory

    manifest = pd.DataFrame({"slide_id": slide_id, "x": xs, "y": ys})
    print(f"[embed] {len(emb):,} patches  dim={emb.shape[1]}")

    # ── step 3: optional PCA ─────────────────────────────────────────────────
    n_pca = int(cfg["pca"]["n_components"])
    if n_pca > 0 and n_pca < emb.shape[1]:
        print(f"[pca] Reducing {emb.shape[1]}D -> {n_pca}D")
        emb_fit = PCA(n_components=n_pca).fit_transform(emb).astype(np.float32)
    else:
        emb_fit = emb

    # ── step 4: HDBSCAN ──────────────────────────────────────────────────────
    hdb_cfg = cfg["hdbscan"]
    min_cls  = int(hdb_cfg["min_cluster_size"])
    min_samp = hdb_cfg.get("min_samples") or None
    print(f"[hdbscan] min_cluster_size={min_cls}  n_patches={len(emb_fit):,}")
    clusterer     = hdbscan_lib(min_cluster_size=min_cls, min_samples=min_samp, n_jobs=-1)
    cluster_labels = clusterer.fit_predict(emb_fit)
    n_clusters     = int((np.unique(cluster_labels) > -1).sum())
    n_noise        = int((cluster_labels == -1).sum())
    print(f"[hdbscan] {n_clusters} clusters  {n_noise:,} noise points")

    # ── step 5: ground-truth labels + metrics (optional) ────────────────────
    gt_labels = None
    metrics   = None
    if geojson_path is not None:
        geojson_path = Path(geojson_path)
        patch_size   = int(cfg["patching"]["patch_size"])
        gt_labels    = tumour_labels_from_geojson(manifest, geojson_path.parent, patch_size=patch_size)
        metrics = _compute_metrics(cluster_labels, gt_labels.values)
        print(f"[metrics] ARI={metrics['ari']}  NMI={metrics['nmi']}  "
              f"F1={metrics['tumour_f1']}  best_cluster={metrics['best_tumour_cluster']}  "
              f"purity={metrics['best_tumour_cluster_purity_%']}%")

    # ── step 6: UMAP 2D ──────────────────────────────────────────────────────
    max_pts = umap_cfg.get("max_points") if umap_cfg.get("max_points") is not None else len(emb_fit)
    n_viz   = min(int(max_pts), len(emb_fit))
    rng     = np.random.default_rng(umap_cfg["random_state"])
    idx     = rng.choice(len(emb_fit), n_viz, replace=False) if n_viz < len(emb_fit) else np.arange(n_viz)

    viz_emb   = emb_fit[idx]
    viz_clust = cluster_labels[idx]

    print(f"[umap] Computing 2D UMAP on {n_viz:,} patches...")
    umap_2d = compute_umap(
        viz_emb,
        n_components=2,
        n_neighbors=umap_cfg["n_neighbors"],
        min_dist=umap_cfg["min_dist"],
        random_state=umap_cfg["random_state"],
    )

    np.save(umap_2d_path, umap_2d)
    np.save(clust_path,   viz_clust)

    # ── step 7: plots ─────────────────────────────────────────────────────────
    cluster_strs = _cluster_strings(viz_clust)
    plot_umap(umap_2d, labels=cluster_strs,
              out_path=out_dir / "umap_by_cluster.png",
              title=f"HDBSCAN Clusters — {slide_id}")

    if gt_labels is not None:
        viz_gt = gt_labels.values[idx]
        np.save(gt_path, viz_gt)
        plot_umap(umap_2d, labels=viz_gt,
                  out_path=out_dir / "umap_by_groundtruth.png",
                  title=f"Ground-truth Labels — {slide_id}")

    # ── summary ───────────────────────────────────────────────────────────────
    summary = {
        "slide_id":        slide_id,
        "n_patches":       int(len(emb)),
        "embedding_dim":   int(emb.shape[1]),
        "pca_components":  n_pca if n_pca > 0 else "none",
        "n_clusters":      n_clusters,
        "n_noise_patches": n_noise,
        "umap_n_viz":      n_viz,
        "metrics":         metrics,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _write_results_md(out_dir, summary)

    print(f"\n[done] Outputs -> {out_dir}")
    print(f"       umap_by_cluster.png")
    if gt_labels is not None:
        print(f"       umap_by_groundtruth.png")
    print(f"       results.md")
    print(f"       summary.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HackDay2 single-WSI HDBSCAN pipeline")
    parser.add_argument("--config",      default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--wsi",         required=False, help="Path to WSI file (.svs/.tif/.ndpi)")
    parser.add_argument("--geojson",     default=None,   help="Path to .geojson annotation file for ground-truth overlay")
    parser.add_argument("--skip-embed",  action="store_true", help="Skip patching+embedding; load existing embeddings.h5")
    parser.add_argument("--plots-only",  action="store_true", help="Replot from cached UMAP coords only")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # resolve data_root + slide_id into concrete paths
    data_root = Path(cfg["data_root"])
    slide_id  = cfg["slide_id"]

    wsi_path = (
        args.wsi
        or cfg.get("wsi_path")
        or str(data_root / "wsi" / f"{slide_id}.svs")
    )
    geojson_path = (
        args.geojson
        or cfg.get("geojson_path")
        or (str(data_root / "annotations" / f"{slide_id}.geojson")
            if (data_root / "annotations" / f"{slide_id}.geojson").exists()
            else None)
    )
    # auto-set CONCH checkpoint from data_root if not explicitly configured
    if not cfg["conch"]["checkpoint"]:
        cfg["conch"]["checkpoint"] = str(data_root / "conch" / "pytorch_model.bin")

    # if pre-computed embeddings exist in Drive download location, copy to cache
    drive_emb = data_root / "embeddings" / slide_id / "cache" / "embeddings.h5"
    cache_emb = Path(cfg["outputs_dir"]) / slide_id / "cache" / "embeddings.h5"
    if drive_emb.exists() and not cache_emb.exists():
        cache_emb.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(drive_emb, cache_emb)
        print(f"[setup] Copied embeddings from {drive_emb}")

    run(
        cfg=cfg,
        wsi_path=wsi_path,
        geojson_path=geojson_path,
        skip_embed=args.skip_embed,
        plots_only=args.plots_only,
    )
