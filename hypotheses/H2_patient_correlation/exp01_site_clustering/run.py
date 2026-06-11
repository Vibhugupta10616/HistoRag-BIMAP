"""
H2 Experiment 01 — Site Clustering (Q1)

Pipeline:
    1. Load all patch embeddings for the configured encoder via histoRAG.loader
       (reads directly from h5 files under embeddings_root)
    2. Site labels come from the loader manifest — no clinical CSV needed
    3. Aggregate patches per patient: mean pool -> 1 vector/patient
    4. UMAP — 2-D scatter of patient vectors coloured by anatomical site
       Expected: distinct clusters per site
    5. Correlation matrix — N x N patient cosine similarity heatmap
       Expected: block-diagonal pattern (high within-site, low across-site)
    6. Save outputs/

To run:
    python hypotheses/H2_patient_correlation/exp01_site_clustering/run.py
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import yaml

from histoRAG.loader import load_encoder
from histoRAG.correlate import (
    aggregate_by_patient,
    correlation_matrix,
    compute_umap,
    plot_umap,
    plot_heatmap,
)


def run(config_path: str | Path) -> None:
    # ------------------------------------------------------------------ setup
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["outputs"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = cfg["encoder"]
    embeddings_root = cfg["inputs"]["embeddings_root"]

    # -------------------------------------------------------- load embeddings
    print(f"[H2 exp01] Loading embeddings for encoder '{encoder}' from {embeddings_root}")
    embeddings, manifest = load_encoder(encoder, embeddings_root)

    print(f"[H2 exp01] {len(manifest)} patches | {manifest['slide_id'].nunique()} patients")
    site_counts = dict(manifest.drop_duplicates("slide_id")["site"].value_counts())
    print(f"[H2 exp01] Patients per site: {site_counts}")

    # ------------------------------------------------- aggregate per patient
    print("[H2 exp01] Aggregating patches per patient (mean pool)...")
    umap_params = cfg["params"].get("umap", {})
    patient_embeddings, patient_ids = aggregate_by_patient(
        manifest, embeddings,
        group_col="slide_id",
        method=cfg["params"].get("aggregation", "mean"),
    )

    # one site label per patient
    id_to_site = manifest.drop_duplicates("slide_id").set_index("slide_id")["site"]
    patient_sites = id_to_site.loc[patient_ids].values
    print(f"[H2 exp01] {len(patient_ids)} patient vectors | dim={patient_embeddings.shape[1]}")

    # ----------------------------------------------------------------- UMAP
    print("[H2 exp01] Computing UMAP on patient-level embeddings...")
    umap_coords = compute_umap(
        patient_embeddings,
        n_neighbors=umap_params.get("n_neighbors", 15),
        min_dist=umap_params.get("min_dist", 0.1),
        random_state=umap_params.get("random_state", 42),
    )
    plot_umap(
        umap_coords,
        labels=patient_sites,
        out_path=out_dir / "umap_patients_by_site.png",
        title=f"WSI-level Embeddings ({encoder})  —  coloured by anatomical site",
    )

    # ------------------------------------------- correlation matrix + heatmap
    print("[H2 exp01] Computing patient-patient correlation matrix...")
    corr = correlation_matrix(patient_embeddings)

    plot_heatmap(
        corr,
        group_labels=patient_sites,
        out_path=out_dir / "heatmap_patient_correlation.png",
        title=f"Patient Correlation by Site ({encoder})  —  expected: block-diagonal",
        order_by_group=True,
    )
    np.save(out_dir / "correlation_matrix.npy", corr)

    # --------------------------------------------------------------- summary
    summary = {
        "experiment":   cfg["experiment"]["name"],
        "encoder":      encoder,
        "n_patients":   int(len(patient_ids)),
        "n_patches":    int(len(manifest)),
        "site_counts":  {str(k): int(v) for k, v in site_counts.items()},
        "embedding_dim": int(patient_embeddings.shape[1]),
        "outputs": [
            "umap_patients_by_site.png",
            "heatmap_patient_correlation.png",
            "correlation_matrix.npy",
            "summary.json",
        ],
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[H2 exp01] Done — outputs in {out_dir}")


if __name__ == "__main__":
    _config = Path(__file__).parent / "config.yaml"
    run(_config)
