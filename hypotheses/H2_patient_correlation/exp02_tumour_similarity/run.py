"""
H2 Experiment 02 — Tumour Patch Similarity (Q2)

Pipeline:
    1. Load patch embeddings (all patches, provided by user)
    2. Derive tumour labels from QuPath .geojson annotations
    3. Also load anatomical site labels (for UMAP colouring only)
    4. Filter to tumour patches only
    5. UMAP on individual tumour patch embeddings, coloured by anatomical site
       Expected: patches from all sites are MIXED (no site-based separation)
    6. Adaptive aggregation decision:
         - Count tumour patches per patient
         - If median >= threshold → aggregate (patient-level heatmap)
         - If median <  threshold → keep individual patches (patch-level heatmap)
    7. Correlation matrix heatmap
       Expected: uniformly HIGH similarity — cancer histology shared across sites
    8. Save outputs/

To run:
    python hypotheses/H2_patient_correlation/exp02_tumour_similarity/run.py
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import yaml

from histoRAG.labels import tumour_labels_from_geojson, site_labels_from_clinical
from histoRAG.correlate import (
    aggregate_by_patient,
    correlation_matrix,
    compute_umap,
    plot_umap,
    plot_heatmap,
    tumour_patch_counts,
    decide_aggregation,
)


def run(config_path: str | Path) -> None:
    # ------------------------------------------------------------------ setup
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["outputs"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------- load embeddings
    emb_path = Path(cfg["inputs"]["embeddings_path"])
    print(f"[H2 exp02] Loading embeddings: {emb_path}")
    embeddings = np.load(emb_path)                       # (N_patches, dim)
    manifest   = pd.read_parquet(cfg["inputs"]["manifest_path"])

    assert len(embeddings) == len(manifest), (
        f"Row count mismatch: embeddings={len(embeddings)}, manifest={len(manifest)}. "
        "Embeddings must be row-aligned to the manifest."
    )
    print(f"[H2 exp02] {len(manifest)} patches | {manifest['slide_id'].nunique()} patients")

    manifest = manifest.copy()

    # ---------------------------------------------------- derive labels
    print("[H2 exp02] Deriving tumour labels from .geojson annotations...")
    tumour_labels = tumour_labels_from_geojson(manifest, cfg["inputs"]["geojson_dir"])
    manifest["tumour_label"] = tumour_labels.values

    # site labels are used only for UMAP colouring (not for filtering/analysis)
    print("[H2 exp02] Loading site labels for UMAP colouring...")
    site_labels = site_labels_from_clinical(
        manifest,
        cfg["inputs"]["clinical_metadata"],
        slide_id_col=cfg["inputs"].get("slide_id_col", "patient_id"),
        site_col=cfg["inputs"].get("site_col", "primary_site"),
    )
    manifest["site"] = site_labels.values

    # ----------------------------------------------- filter to tumour patches
    tumour_mask = manifest["tumour_label"] == "tumour"
    tumour_embeddings = embeddings[tumour_mask.values]
    tumour_manifest   = manifest[tumour_mask].reset_index(drop=True)
    print(f"[H2 exp02] Tumour patches: {len(tumour_manifest)} / {len(manifest)}")

    if len(tumour_manifest) == 0:
        raise ValueError(
            "No tumour patches found. Check that geojson_dir contains annotation files "
            "matching the slide IDs in the manifest."
        )

    # ----------------------------------------------------------------- UMAP
    # UMAP uses INDIVIDUAL tumour patches (no aggregation) coloured by site.
    # Expected: all three sites are MIXED — cancer patches look similar regardless of location.
    print("[H2 exp02] Computing UMAP on individual tumour patches...")
    umap_params = cfg["params"].get("umap", {})
    umap_coords = compute_umap(
        tumour_embeddings,
        n_neighbors=umap_params.get("n_neighbors", 15),
        min_dist=umap_params.get("min_dist", 0.1),
        random_state=umap_params.get("random_state", 42),
    )
    plot_umap(
        umap_coords,
        labels=tumour_manifest["site"].values,
        out_path=out_dir / "umap_tumour_patches_by_site.png",
        title=f"Tumour Patch Embeddings ({cfg['encoder']})  —  expected: sites mixed",
    )

    # ----------------------------------- adaptive aggregation + correlation map
    counts    = tumour_patch_counts(tumour_manifest, tumour_manifest["tumour_label"])
    threshold = cfg["params"].get("aggregation_threshold", 20)
    agg_mode  = decide_aggregation(counts, threshold=threshold)

    print(f"[H2 exp02] Building correlation matrix (mode: '{agg_mode}')...")

    if agg_mode == "patient":
        # one vector per patient — mean-pool all tumour patches for that patient
        agg_embeddings, agg_ids = aggregate_by_patient(
            tumour_manifest, tumour_embeddings, group_col="slide_id", method="mean"
        )
        id_to_site = tumour_manifest.drop_duplicates("slide_id").set_index("slide_id")["site"]
        heatmap_group_labels = id_to_site.loc[agg_ids].values
        heatmap_embeddings   = agg_embeddings
        heatmap_title = (
            f"Patient Cancer Similarity ({cfg['encoder']})  —  expected: uniformly high"
        )
    else:
        # individual patch-level correlation; group axis by patient for readability
        heatmap_embeddings   = tumour_embeddings
        heatmap_group_labels = tumour_manifest["slide_id"].values
        heatmap_title = (
            f"Tumour Patch Similarity ({cfg['encoder']})  —  expected: uniformly high"
        )

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
        "experiment":                    cfg["experiment"]["name"],
        "encoder":                       cfg["encoder"],
        "n_patches_total":               int(len(manifest)),
        "n_tumour_patches":              int(len(tumour_manifest)),
        "aggregation_mode":              agg_mode,
        "median_tumour_patches_per_pt":  float(counts.median()) if len(counts) > 0 else 0,
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
    _config = Path(__file__).parent / "config.yaml"
    run(_config)
