"""
H2 Experiment 02 — Tumour Patch Similarity (Q2)

Pipeline:
    1. Stream patch embeddings slide by slide via histoRAG.loader.iter_encoder
    2. Derive tumour labels per slide from QuPath .geojson annotations
    3. Keep only tumour patches for that slide; discard all others immediately
       Peak RAM = one slide + accumulated tumour patches across all slides (~5%)
    4. UMAP on a random subsample of tumour patches, coloured by anatomical site
       Expected: all sites are MIXED — cancer looks similar regardless of location
    5. KDE distribution plot: within-site vs cross-site pairwise similarities
       Uses ALL tumour patches (subsampled per site for tractability)
    8. Save outputs/

To run:
    python hypotheses/H2_patient_correlation/exp02_tumour_similarity/run.py
    python hypotheses/H2_patient_correlation/exp02_tumour_similarity/run.py --encoder conch
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import yaml

from histoRAG.loader import iter_encoder, detect_patch_size
from histoRAG.labels import tumour_labels_from_geojson
from histoRAG.correlate import (
    compute_umap,
    plot_umap,
    plot_umap_3d,
    plot_umap_3d_interactive,
    compute_similarity_pairs,
    plot_similarity_distribution,
)


def _plots(out_dir: Path, encoder: str, title_suffix: str) -> None:
    """Regenerate all plots from cached .npy files. Fast — no embedding scan."""
    umap_coords_2d = np.load(out_dir / "cache_umap_coords_2d.npy")
    umap_coords_3d = np.load(out_dir / "cache_umap_coords_3d.npy")
    umap_sites     = np.load(out_dir / "cache_umap_sites.npy", allow_pickle=True)
    within_sims    = np.load(out_dir / "cache_within_sims.npy")
    cross_sims     = np.load(out_dir / "cache_cross_sims.npy")

    plot_umap(umap_coords_2d, labels=umap_sites,
              out_path=out_dir / "umap_tumour_patches_by_site.png",
              title=f"Tumour Patch Embeddings ({encoder})  —  expected: sites mixed")
    plot_umap_3d(umap_coords_3d, labels=umap_sites,
                 out_path=out_dir / "umap_tumour_patches_by_site_3d.png",
                 title=f"Tumour Patch Embeddings 3D ({encoder})  —  expected: sites mixed")
    plot_umap_3d_interactive(umap_coords_3d, labels=umap_sites,
                             out_path=out_dir / "umap_tumour_patches_by_site_3d.html",
                             title=f"Tumour Patch Embeddings 3D ({encoder})  —  expected: sites mixed")
    plot_similarity_distribution(within_sims, cross_sims,
                                 out_path=out_dir / "similarity_distribution.png",
                                 title=f"Tumour Patch Similarity Distribution ({encoder})")
    print(f"[H2 exp02] Plots regenerated from cache -> {out_dir}")


def run(
    config_path: str | Path,
    encoder_override: str | None = None,
    embeddings_root_override: str | None = None,
    geojson_dir_override: str | None = None,
    plots_only: bool = False,
) -> None:
    # ------------------------------------------------------------------ setup
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    encoder         = encoder_override or cfg["encoder"]
    geojson_dir     = Path(geojson_dir_override or cfg["inputs"]["geojson_dir"])
    embeddings_root = embeddings_root_override or cfg["inputs"]["embeddings_root"]
    out_dir         = Path(cfg["outputs"]["dir"]) / encoder
    out_dir.mkdir(parents=True, exist_ok=True)

    # --plots-only: skip heavy computation, regenerate from cached arrays
    if plots_only:
        missing = [f for f in [
            "cache_umap_coords_2d.npy", "cache_umap_coords_3d.npy",
            "cache_umap_sites.npy", "cache_within_sims.npy", "cache_cross_sims.npy",
        ] if not (out_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Cache files missing in {out_dir}: {missing}\n"
                "Run without --plots-only first to build the cache."
            )
        _plots(out_dir, encoder, title_suffix="")
        return

    if not geojson_dir.exists():
        raise FileNotFoundError(
            f"geojson_dir not found: {geojson_dir}\n"
            "Download the HANCOCK .geojson annotation files and update config.yaml."
        )

    patch_size_cfg = cfg["params"].get("patch_size", None)
    patch_size = patch_size_cfg if patch_size_cfg else detect_patch_size(encoder, embeddings_root)
    print(f"\n[H2 exp02] encoder={encoder} | patch_size={patch_size}"
          f"{' (from config)' if patch_size_cfg else ' (auto-detected)'}")

    # ----------------------- stream slides, collect only tumour patches ------
    print("[H2 exp02] Streaming slides and collecting tumour patches...")

    tumour_emb_parts      = []
    tumour_manifest_parts = []
    total_patches         = 0

    for emb, slide_df in iter_encoder(encoder, embeddings_root):
        slide_id      = slide_df["slide_id"].iloc[0]
        n_slide       = len(emb)
        total_patches += n_slide

        tumour_lbl = tumour_labels_from_geojson(slide_df, geojson_dir, patch_size=patch_size)
        mask       = (tumour_lbl == "tumour").values
        n_tumour   = int(mask.sum())

        print(f"  {slide_id}: {n_slide:>6,} patches | "
              f"tumour={n_tumour:>4,} ({n_tumour/n_slide*100:.1f}%)")

        if n_tumour > 0:
            tumour_emb_parts.append(emb[mask])
            tumour_df = slide_df.copy().reset_index(drop=True)
            tumour_df["tumour_label"] = tumour_lbl.values
            tumour_manifest_parts.append(tumour_df[mask].reset_index(drop=True))

    if not tumour_emb_parts:
        raise ValueError(
            "No tumour patches found across all slides. "
            "Check that geojson_dir contains annotation files matching the slide IDs."
        )

    tumour_embeddings = np.concatenate(tumour_emb_parts, axis=0).astype(np.float32)
    tumour_manifest   = pd.concat(tumour_manifest_parts, ignore_index=True)
    n_tumour_total    = len(tumour_manifest)

    # L2-normalize so cosine similarity = dot product (CONCH is pre-normalized,
    # UNI h5 files store raw un-normalized vectors)
    norms = np.linalg.norm(tumour_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    tumour_embeddings = tumour_embeddings / norms

    print(f"\n[H2 exp02] Total patches scanned : {total_patches:,}")
    print(f"[H2 exp02] Tumour patches collected: {n_tumour_total:,} "
          f"({n_tumour_total/total_patches*100:.1f}% overall)")

    # ----------------------------------------------------------------- UMAP
    umap_params  = cfg["params"].get("umap", {})
    umap_max_pts = int(umap_params.get("max_points", 50_000))

    if umap_max_pts > 0 and n_tumour_total > umap_max_pts:
        rng      = np.random.default_rng(umap_params.get("random_state", 42))
        umap_idx = rng.choice(n_tumour_total, umap_max_pts, replace=False)
        umap_emb   = tumour_embeddings[umap_idx]
        umap_sites = tumour_manifest["site"].values[umap_idx]
        print(f"[H2 exp02] UMAP subsampled {umap_max_pts:,} / {n_tumour_total:,} tumour patches")
    else:
        umap_emb   = tumour_embeddings
        umap_sites = tumour_manifest["site"].values

    umap_kwargs = dict(
        n_neighbors=umap_params.get("n_neighbors", 15),
        min_dist=umap_params.get("min_dist", 0.1),
        random_state=umap_params.get("random_state", 42),
    )

    print("[H2 exp02] Computing 2D UMAP on tumour patches...")
    umap_coords_2d = compute_umap(umap_emb, n_components=2, **umap_kwargs)

    print("[H2 exp02] Computing 3D UMAP on tumour patches...")
    umap_coords_3d = compute_umap(umap_emb, n_components=3, **umap_kwargs)

    # ----------------------------------------- similarity pairs (for KDE plot)
    kde_sample = int(cfg["params"].get("kde_sample_per_site", 1000))
    print(f"[H2 exp02] Computing similarity pairs ({kde_sample}/site)...")
    within_sims, cross_sims = compute_similarity_pairs(
        tumour_embeddings,
        site_labels=tumour_manifest["site"].values,
        n_sample_per_site=kde_sample,
    )

    # ----------------------------------------------------------- save cache ---
    np.save(out_dir / "cache_umap_coords_2d.npy", umap_coords_2d)
    np.save(out_dir / "cache_umap_coords_3d.npy", umap_coords_3d)
    np.save(out_dir / "cache_umap_sites.npy",     umap_sites)
    np.save(out_dir / "cache_within_sims.npy",    within_sims)
    np.save(out_dir / "cache_cross_sims.npy",     cross_sims)
    print("[H2 exp02] Intermediate arrays cached.")

    # -------------------------------------------------------- generate plots --
    _plots(out_dir, encoder, title_suffix="")

    # --------------------------------------------------------------- summary
    summary = {
        "experiment":          cfg["experiment"]["name"],
        "encoder":             encoder,
        "patch_size":          patch_size,
        "n_patches_total":     total_patches,
        "n_tumour_patches":    n_tumour_total,
        "tumour_pct_overall":  round(n_tumour_total / total_patches * 100, 2),
        "kde_sample_per_site": kde_sample,
        "outputs": [
            "umap_tumour_patches_by_site.png",
            "umap_tumour_patches_by_site_3d.png",
            "umap_tumour_patches_by_site_3d.html",
            "similarity_distribution.png",
            "summary.json",
        ],
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[H2 exp02] Done — outputs in {out_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--encoder", default=None, help="Override encoder (conch | uni2h)")
    parser.add_argument("--embeddings-root", default=None,
                        help="Override embeddings_root from config (useful on HPC).")
    parser.add_argument("--geojson-dir", default=None,
                        help="Override geojson_dir from config (useful on HPC).")
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip embedding scan; regenerate plots from cached .npy files.")
    args = parser.parse_args()
    run(args.config, encoder_override=args.encoder,
        embeddings_root_override=args.embeddings_root,
        geojson_dir_override=args.geojson_dir,
        plots_only=args.plots_only)
