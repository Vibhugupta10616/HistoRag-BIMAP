"""
H2 Experiment 02 — Tumour Patch Similarity (Q2)

Pipeline:
    1. Stream patch embeddings slide by slide via histoRAG.loader.iter_encoder
    2. Derive tumour labels per slide from QuPath .geojson annotations
    3. Collect tumour patches AND non-tumour patches separately
       Peak RAM = one slide + accumulated patches across all slides
    4. UMAP on a random subsample of each class, coloured by anatomical site
       Tumour  expected: all sites are MIXED — cancer looks similar regardless of location
       Non-tumour expected: sites separate — normal tissue retains site-specific identity
    5. KDE distribution plot: within-site vs cross-site pairwise similarities
       4 curves on one axis: tumour within/cross (solid) + non-tumour within/cross (dashed)
    6. Save outputs/{encoder}/tumour/  and  outputs/{encoder}/nontumour/

To run:
    python hypotheses/H2_patient_correlation/exp02_tumour_similarity/run.py
    python hypotheses/H2_patient_correlation/exp02_tumour_similarity/run.py --encoder conch
"""

import json
import shutil
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


def _plots(tumour_dir: Path, nontumour_dir: Path, encoder: str) -> None:
    """Regenerate all plots from cached .npy files. Fast — no embedding scan."""
    for cls_dir, label in [(tumour_dir, "tumour"), (nontumour_dir, "nontumour")]:
        umap_coords_2d = np.load(cls_dir / "cache_umap_coords_2d.npy")
        umap_coords_3d = np.load(cls_dir / "cache_umap_coords_3d.npy")
        umap_sites     = np.load(cls_dir / "cache_umap_sites.npy", allow_pickle=True)

        plot_umap(umap_coords_2d, labels=umap_sites,
                  out_path=cls_dir / "umap_patches_by_site.png",
                  title=f"{label.capitalize()} Patch Embeddings ({encoder})  —  2D UMAP by site")
        plot_umap_3d(umap_coords_3d, labels=umap_sites,
                     out_path=cls_dir / "umap_patches_by_site_3d.png",
                     title=f"{label.capitalize()} Patch Embeddings 3D ({encoder})")
        plot_umap_3d_interactive(umap_coords_3d, labels=umap_sites,
                                 out_path=cls_dir / "umap_patches_by_site_3d.html",
                                 title=f"{label.capitalize()} Patch Embeddings 3D ({encoder})")

    # 4-curve similarity distribution (shared between both subfolders)
    within_t  = np.load(tumour_dir    / "cache_within_sims.npy")
    cross_t   = np.load(tumour_dir    / "cache_cross_sims.npy")
    within_nt = np.load(nontumour_dir / "cache_within_sims.npy")
    cross_nt  = np.load(nontumour_dir / "cache_cross_sims.npy")

    plot_similarity_distribution(
        within_t=within_t, cross_t=cross_t,
        within_nt=within_nt, cross_nt=cross_nt,
        out_path=tumour_dir.parent / "similarity_distribution.png",
        title=f"Patch Similarity: Tumour vs Non-tumour ({encoder})",
    )
    print(f"[H2 exp02] Plots regenerated from cache -> {tumour_dir.parent}")


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


def _umap_and_cache(embeddings: np.ndarray, sites: np.ndarray,
                    out_dir: Path, umap_params: dict, label: str) -> None:
    """Run 2D + 3D UMAP on a subsample, cache arrays, generate all three plots."""
    max_pts = int(umap_params.get("max_points", 50_000))
    n = len(embeddings)

    if max_pts > 0 and n > max_pts:
        rng = np.random.default_rng(umap_params.get("random_state", 42))
        idx = rng.choice(n, max_pts, replace=False)
        umap_emb   = embeddings[idx]
        umap_sites = sites[idx]
        print(f"[H2 exp02] {label} UMAP subsampled {max_pts:,} / {n:,} patches")
    else:
        umap_emb   = embeddings
        umap_sites = sites

    kwargs = dict(
        n_neighbors=umap_params.get("n_neighbors", 15),
        min_dist=umap_params.get("min_dist", 0.1),
        random_state=umap_params.get("random_state", 42),
    )

    print(f"[H2 exp02] Computing 2D UMAP on {label} patches...")
    coords_2d = compute_umap(umap_emb, n_components=2, **kwargs)

    print(f"[H2 exp02] Computing 3D UMAP on {label} patches...")
    coords_3d = compute_umap(umap_emb, n_components=3, **kwargs)

    np.save(out_dir / "cache_umap_coords_2d.npy", coords_2d)
    np.save(out_dir / "cache_umap_coords_3d.npy", coords_3d)
    np.save(out_dir / "cache_umap_sites.npy",     umap_sites)

    encoder_name = out_dir.parent.name
    plot_umap(coords_2d, labels=umap_sites,
              out_path=out_dir / "umap_patches_by_site.png",
              title=f"{label.capitalize()} Patch Embeddings ({encoder_name})  —  2D UMAP by site")
    plot_umap_3d(coords_3d, labels=umap_sites,
                 out_path=out_dir / "umap_patches_by_site_3d.png",
                 title=f"{label.capitalize()} Patch Embeddings 3D ({encoder_name})")
    plot_umap_3d_interactive(coords_3d, labels=umap_sites,
                             out_path=out_dir / "umap_patches_by_site_3d.html",
                             title=f"{label.capitalize()} Patch Embeddings 3D ({encoder_name})")


def run(
    config_path: str | Path,
    encoder_override: str | None = None,
    embeddings_root_override: str | None = None,
    geojson_dir_override: str | None = None,
    plots_only: bool = False,
    full: bool = False,
) -> None:
    # ------------------------------------------------------------------ setup
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    encoder         = encoder_override or cfg["encoder"]
    geojson_dir     = Path(geojson_dir_override or cfg["inputs"]["geojson_dir"])
    embeddings_root = embeddings_root_override or cfg["inputs"]["embeddings_root"]

    base_dir      = Path(cfg["outputs"]["dir"]) / encoder
    tumour_dir    = base_dir / "tumour"
    nontumour_dir = base_dir / "nontumour"
    tumour_dir.mkdir(parents=True, exist_ok=True)
    nontumour_dir.mkdir(parents=True, exist_ok=True)

    # --plots-only: skip heavy computation, regenerate from cached arrays
    if plots_only:
        required = [
            tumour_dir    / "cache_umap_coords_2d.npy",
            tumour_dir    / "cache_umap_coords_3d.npy",
            tumour_dir    / "cache_umap_sites.npy",
            tumour_dir    / "cache_within_sims.npy",
            tumour_dir    / "cache_cross_sims.npy",
            nontumour_dir / "cache_umap_coords_2d.npy",
            nontumour_dir / "cache_umap_coords_3d.npy",
            nontumour_dir / "cache_umap_sites.npy",
            nontumour_dir / "cache_within_sims.npy",
            nontumour_dir / "cache_cross_sims.npy",
        ]
        missing = [str(f) for f in required if not f.exists()]
        if missing:
            raise FileNotFoundError(
                f"Cache files missing:\n" + "\n".join(missing) +
                "\nRun without --plots-only first to build the cache."
            )
        _plots(tumour_dir, nontumour_dir, encoder)
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

    # ------------- stream slides, collect tumour AND non-tumour patches -------
    # Non-tumour patches are ~95% of total (~16 GB for CONCH, ~32 GB for UNI full set).
    # --full (HPC): collect all patches; downstream UMAP/KDE sampling handles tractability.
    # default (local): cap at 250k/site → ~1M total; safe for 16 GB RAM.
    umap_params  = cfg["params"].get("umap", {})
    umap_max_pts = int(umap_params.get("max_points", 50_000))
    kde_sample   = int(cfg["params"].get("kde_sample_per_site", 1000))
    nt_cap_per_site = None if full else 250_000

    if nt_cap_per_site is None:
        print("[H2 exp02] Non-tumour: collecting ALL patches (--full mode, HPC)")
    else:
        print(f"[H2 exp02] Non-tumour streaming cap: {nt_cap_per_site:,} patches/site")
    print("[H2 exp02] Streaming slides and collecting patches...")

    tumour_emb_parts      = []
    tumour_manifest_parts = []
    # per-site dicts for non-tumour (capped during streaming)
    nt_emb_by_site  = {}   # site -> list of arrays
    nt_rows_by_site = {}   # site -> list of DataFrames
    nt_seen_by_site = {}   # site -> total patches seen (for tracking)
    total_patches = 0

    for emb, slide_df in iter_encoder(encoder, embeddings_root):
        slide_id      = slide_df["slide_id"].iloc[0]
        n_slide       = len(emb)
        total_patches += n_slide

        tumour_lbl = tumour_labels_from_geojson(slide_df, geojson_dir, patch_size=patch_size)
        mask       = (tumour_lbl == "tumour").values
        n_tumour   = int(mask.sum())

        print(f"  {slide_id}: {n_slide:>6,} patches | "
              f"tumour={n_tumour:>4,} ({n_tumour/n_slide*100:.1f}%)")

        slide_df = slide_df.copy().reset_index(drop=True)
        slide_df["tumour_label"] = tumour_lbl.values

        if n_tumour > 0:
            tumour_emb_parts.append(emb[mask])
            tumour_manifest_parts.append(slide_df[mask].reset_index(drop=True))

        non_mask = ~mask
        if non_mask.any():
            site = slide_df["site"].iloc[0]
            nt_emb_slide = emb[non_mask]
            nt_df_slide  = slide_df[non_mask].reset_index(drop=True)

            nt_seen_by_site[site] = nt_seen_by_site.get(site, 0) + len(nt_emb_slide)

            already = sum(len(a) for a in nt_emb_by_site.get(site, []))
            if nt_cap_per_site is None or already < nt_cap_per_site:
                take = len(nt_emb_slide) if nt_cap_per_site is None else min(nt_cap_per_site - already, len(nt_emb_slide))
                nt_emb_by_site.setdefault(site, []).append(nt_emb_slide[:take])
                nt_rows_by_site.setdefault(site, []).append(nt_df_slide.iloc[:take])

    if not tumour_emb_parts:
        raise ValueError(
            "No tumour patches found across all slides. "
            "Check that geojson_dir contains annotation files matching the slide IDs."
        )

    for site, seen in nt_seen_by_site.items():
        kept = sum(len(a) for a in nt_emb_by_site.get(site, []))
        print(f"[H2 exp02] Non-tumour site={site}: kept {kept:,} / {seen:,} patches")

    tumour_embeddings = _normalize(np.concatenate(tumour_emb_parts, axis=0).astype(np.float32))
    del tumour_emb_parts
    tumour_manifest = pd.concat(tumour_manifest_parts, ignore_index=True)
    del tumour_manifest_parts

    nt_emb_flat  = [arr for parts in nt_emb_by_site.values()  for arr in parts]
    nt_rows_flat = [df  for parts in nt_rows_by_site.values() for df  in parts]
    del nt_emb_by_site, nt_rows_by_site

    nontumour_embeddings = _normalize(np.concatenate(nt_emb_flat, axis=0).astype(np.float32))
    del nt_emb_flat
    nontumour_manifest = pd.concat(nt_rows_flat, ignore_index=True)
    del nt_rows_flat

    n_tumour_total    = len(tumour_manifest)
    n_nontumour_total = len(nontumour_manifest)

    print(f"\n[H2 exp02] Total patches scanned    : {total_patches:,}")
    print(f"[H2 exp02] Tumour patches collected  : {n_tumour_total:,} "
          f"({n_tumour_total/total_patches*100:.1f}%)")
    print(f"[H2 exp02] Non-tumour patches collected: {n_nontumour_total:,} "
          f"({n_nontumour_total/total_patches*100:.1f}%)")

    # ----------------------------------------------------------------- UMAP
    _umap_and_cache(tumour_embeddings,    tumour_manifest["site"].values,
                    tumour_dir, umap_params, label="tumour")
    _umap_and_cache(nontumour_embeddings, nontumour_manifest["site"].values,
                    nontumour_dir, umap_params, label="nontumour")

    # ----------------------------------------- similarity pairs (for KDE plot)
    print(f"[H2 exp02] Computing tumour similarity pairs ({kde_sample}/site)...")
    within_t, cross_t = compute_similarity_pairs(
        tumour_embeddings,
        site_labels=tumour_manifest["site"].values,
        n_sample_per_site=kde_sample,
    )
    np.save(tumour_dir / "cache_within_sims.npy", within_t)
    np.save(tumour_dir / "cache_cross_sims.npy",  cross_t)

    print(f"[H2 exp02] Computing non-tumour similarity pairs ({kde_sample}/site)...")
    within_nt, cross_nt = compute_similarity_pairs(
        nontumour_embeddings,
        site_labels=nontumour_manifest["site"].values,
        n_sample_per_site=kde_sample,
    )
    np.save(nontumour_dir / "cache_within_sims.npy", within_nt)
    np.save(nontumour_dir / "cache_cross_sims.npy",  cross_nt)

    print("[H2 exp02] Intermediate arrays cached.")

    # ---------------------------------------------------- 4-curve KDE plot ---
    plot_similarity_distribution(
        within_t=within_t, cross_t=cross_t,
        within_nt=within_nt, cross_nt=cross_nt,
        out_path=base_dir / "similarity_distribution.png",
        title=f"Patch Similarity: Tumour vs Non-tumour ({encoder})",
    )

    # --------------------------------------------------------------- summary
    summary = {
        "experiment":             cfg["experiment"]["name"],
        "encoder":                encoder,
        "patch_size":             patch_size,
        "n_patches_total":        total_patches,
        "n_tumour_patches":       n_tumour_total,
        "tumour_pct_overall":     round(n_tumour_total    / total_patches * 100, 2),
        "n_nontumour_patches":    n_nontumour_total,
        "nontumour_pct_overall":  round(n_nontumour_total / total_patches * 100, 2),
        "kde_sample_per_site":    kde_sample,
        "outputs": {
            "similarity_distribution.png": "4-curve KDE (tumour + non-tumour)",
            "tumour": ["umap_patches_by_site.png", "umap_patches_by_site_3d.png", "umap_patches_by_site_3d.html"],
            "nontumour": ["umap_patches_by_site.png", "umap_patches_by_site_3d.png", "umap_patches_by_site_3d.html"],
        },
    }
    with open(base_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[H2 exp02] Done — outputs in {base_dir}")


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
    parser.add_argument("--full", action="store_true",
                        help="Collect ALL non-tumour patches (HPC use). "
                             "Default caps at 250k/site to stay within 16 GB RAM.")
    args = parser.parse_args()
    run(args.config, encoder_override=args.encoder,
        embeddings_root_override=args.embeddings_root,
        geojson_dir_override=args.geojson_dir,
        plots_only=args.plots_only,
        full=args.full)
