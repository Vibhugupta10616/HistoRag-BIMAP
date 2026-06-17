"""
H2 Experiment 01 — Site Clustering (Q1)

Pipeline:
    1. Stream patch embeddings slide by slide via histoRAG.loader.iter_encoder
    2. Mean-pool each slide's patches on the fly -> 1 vector per patient
       Peak RAM = one slide at a time + accumulated patient vectors (< 2 MB total)
    3. UMAP — 2-D scatter of patient vectors coloured by anatomical site
       Expected: distinct clusters per site
    4. Correlation matrix — N x N patient cosine similarity heatmap
       Expected: block-diagonal pattern (high within-site, low across-site)
    5. Save outputs/

To run:
    python hypotheses/H2_patient_correlation/exp01_site_clustering/run.py
    python hypotheses/H2_patient_correlation/exp01_site_clustering/run.py --encoder conch
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import yaml

from histoRAG.loader import iter_encoder
from histoRAG.correlate import (
    correlation_matrix,
    compute_umap,
    plot_umap,
    plot_heatmap,
)


def run(
    config_path: str | Path,
    encoder_override: str | None = None,
    embeddings_root_override: str | None = None,
) -> None:
    # ------------------------------------------------------------------ setup
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    encoder         = encoder_override or cfg["encoder"]
    embeddings_root = embeddings_root_override or cfg["inputs"]["embeddings_root"]
    out_dir         = Path(cfg["outputs"]["dir"]) / encoder
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------- stream + mean-pool per patient
    # Process one slide at a time; only the mean vector is kept after each slide.
    print(f"\n[H2 exp01] encoder={encoder}")
    print(f"[H2 exp01] Streaming slides and mean-pooling per patient...")

    patient_vecs  = []
    patient_ids   = []
    patient_sites = []

    for emb, slide_df in iter_encoder(encoder, embeddings_root):
        slide_id  = slide_df["slide_id"].iloc[0]
        site      = slide_df["site"].iloc[0]
        mean_vec  = emb.mean(axis=0).astype(np.float32)
        patient_vecs.append(mean_vec)
        patient_ids.append(slide_id)
        patient_sites.append(site)
        print(f"  {slide_id}: {len(emb):>6,} patches | site={site}")

    patient_embeddings = np.stack(patient_vecs, axis=0)
    patient_ids        = np.array(patient_ids, dtype=str)
    patient_sites      = np.array(patient_sites, dtype=str)

    # re-normalize to unit length (mean-pooling breaks L2-normalization)
    norms = np.linalg.norm(patient_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    patient_embeddings = patient_embeddings / norms

    n_patients = len(patient_ids)
    site_counts = {s: int((patient_sites == s).sum()) for s in np.unique(patient_sites)}
    print(f"\n[H2 exp01] {n_patients} patient vectors | dim={patient_embeddings.shape[1]}")
    print(f"[H2 exp01] Patients per site: {site_counts}")

    # ----------------------------------------------------------------- UMAP
    umap_params = cfg["params"].get("umap", {})
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
        "experiment":    cfg["experiment"]["name"],
        "encoder":       encoder,
        "n_patients":    int(n_patients),
        "site_counts":   site_counts,
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--encoder", default=None, help="Override encoder (conch | uni2h)")
    parser.add_argument("--embeddings-root", default=None,
                        help="Override embeddings_root from config (useful on HPC).")
    args = parser.parse_args()
    run(args.config, encoder_override=args.encoder,
        embeddings_root_override=args.embeddings_root)
