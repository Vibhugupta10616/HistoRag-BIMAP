"""
H2 Experiment 02 — Tumour Patch Similarity (Q2)

Pipeline:
    1. Stream patch embeddings slide by slide via histoRAG.loader.iter_encoder
    2. Derive tumour labels per slide from QuPath .geojson annotations
    3. Keep only tumour patches for that slide; discard all others immediately
       Peak RAM = one slide at a time + accumulated tumour patches across all slides
       (tumour prevalence varies per slide; overall ~5% of total patches)
    4. UMAP on a random subsample of tumour patches, coloured by anatomical site
       Expected: all sites are MIXED — cancer looks similar regardless of location
    5. Adaptive aggregation:
         median tumour patches/patient >= threshold -> aggregate per patient
         median < threshold -> keep individual patches
    6. Correlation matrix heatmap
       Expected: uniformly HIGH similarity — cancer histology shared across sites
    7. Save outputs/

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
    aggregate_by_patient,
    correlation_matrix,
    compute_umap,
    plot_umap,
    plot_heatmap,
    tumour_patch_counts,
    decide_aggregation,
)


def run(
    config_path: str | Path,
    encoder_override: str | None = None,
    embeddings_root_override: str | None = None,
    geojson_dir_override: str | None = None,
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
    # For each slide: load embeddings, derive tumour labels, keep tumour rows only.
    # Non-tumour patches (~95% on average, varies per slide) are discarded immediately.
    print(f"[H2 exp02] Streaming slides and collecting tumour patches...")

    tumour_emb_parts      = []
    tumour_manifest_parts = []
    total_patches         = 0

    for emb, slide_df in iter_encoder(encoder, embeddings_root):
        slide_id     = slide_df["slide_id"].iloc[0]
        n_slide      = len(emb)
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

    print(f"\n[H2 exp02] Total patches scanned : {total_patches:,}")
    print(f"[H2 exp02] Tumour patches collected: {n_tumour_total:,} "
          f"({n_tumour_total/total_patches*100:.1f}% overall)")

    # ----------------------------------------------------------------- UMAP
    # UMAP on individual tumour patches coloured by site.
    # Expected: all sites MIXED — cancer patches similar regardless of location.
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

    print("[H2 exp02] Computing UMAP on tumour patches...")
    umap_coords = compute_umap(
        umap_emb,
        n_neighbors=umap_params.get("n_neighbors", 15),
        min_dist=umap_params.get("min_dist", 0.1),
        random_state=umap_params.get("random_state", 42),
    )
    plot_umap(
        umap_coords,
        labels=umap_sites,
        out_path=out_dir / "umap_tumour_patches_by_site.png",
        title=f"Tumour Patch Embeddings ({encoder})  —  expected: sites mixed",
    )

    # ----------------------------------- adaptive aggregation + correlation map
    counts    = tumour_patch_counts(tumour_manifest, tumour_manifest["tumour_label"])
    threshold = cfg["params"].get("aggregation_threshold", 20)
    agg_mode  = decide_aggregation(counts, threshold=threshold)

    print(f"[H2 exp02] Building correlation matrix (mode: '{agg_mode}')...")

    if agg_mode == "patient":
        agg_embeddings, agg_ids = aggregate_by_patient(
            tumour_manifest, tumour_embeddings, group_col="slide_id", method="mean"
        )
        id_to_site           = tumour_manifest.drop_duplicates("slide_id").set_index("slide_id")["site"]
        heatmap_group_labels = id_to_site.loc[agg_ids].values
        heatmap_embeddings   = agg_embeddings
        heatmap_title        = f"Patient Cancer Similarity ({encoder})  —  expected: uniformly high"
    else:
        heatmap_embeddings   = tumour_embeddings
        heatmap_group_labels = tumour_manifest["slide_id"].values
        heatmap_title        = f"Tumour Patch Similarity ({encoder})  —  expected: uniformly high"

    corr = correlation_matrix(heatmap_embeddings)
    plot_heatmap(
        corr,
        group_labels=heatmap_group_labels,
        out_path=out_dir / "heatmap_tumour_similarity.png",
        title=heatmap_title,
        order_by_group=True,
    )
    np.save(out_dir / "correlation_matrix.npy", corr)

    # --------------------------------------------------------------- summary
    summary = {
        "experiment":                   cfg["experiment"]["name"],
        "encoder":                      encoder,
        "patch_size":                   patch_size,
        "n_patches_total":              total_patches,
        "n_tumour_patches":             n_tumour_total,
        "tumour_pct_overall":           round(n_tumour_total / total_patches * 100, 2),
        "aggregation_mode":             agg_mode,
        "median_tumour_patches_per_pt": float(counts.median()) if len(counts) > 0 else 0,
        "outputs": [
            "umap_tumour_patches_by_site.png",
            "heatmap_tumour_similarity.png",
            "correlation_matrix.npy",
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
    args = parser.parse_args()
    run(args.config, encoder_override=args.encoder,
        embeddings_root_override=args.embeddings_root,
        geojson_dir_override=args.geojson_dir)
