"""
Run H1 for all encoders and both clustering settings.

This script keeps the H1 logic intentionally simple:
  1. Load frozen patch embeddings.
  2. Cluster with K-means, without labels.
  3. Create GeoJSON tumour/other labels after clustering.
  4. Name each cluster by majority vote.
  5. Report global, overall, per-site, and per-WSI results.

Run from repo root:
    python hypotheses/H1_tumour_classification/run_all.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import yaml

from histoRAG.classify import (
    classification_metrics,
    cluster_embeddings,
    cluster_summary,
    grouped_metrics,
    match_clusters_to_labels,
)
from histoRAG.labels import tumour_labels_from_geojson
from histoRAG.loader import load_encoder


DEFAULT_ENCODERS = ["clip-vitb16", "conch", "uni2h"]
DEFAULT_K_VALUES = [2, 8]


def _print_table(title: str, df: pd.DataFrame, max_rows: int | None = None) -> None:
    print(f"\n{title}")
    shown = df if max_rows is None else df.head(max_rows)
    print(shown.to_string(index=False))
    if max_rows is not None and len(df) > max_rows:
        print(f"... {len(df) - max_rows} more rows saved to CSV")


def _round_metrics(metrics: dict) -> dict:
    return {
        "accuracy": round(metrics["accuracy"], 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1": round(metrics["f1"], 4),
        "tumour_prevalence": round(metrics["tumour_prevalence"], 4),
    }


def run_one(
    encoder: str,
    k: int,
    embeddings_root: str,
    geojson_dir: str,
    out_root: Path,
    random_state: int,
    patch_size: int,
    print_wsi_rows: int,
) -> None:
    print("\n" + "=" * 80)
    print(f"H1 | encoder={encoder} | k={k}")
    print("=" * 80)

    embeddings, manifest = load_encoder(encoder, embeddings_root)
    manifest = manifest.reset_index(drop=True)

    print("\nDeriving GeoJSON tumour/other labels...")
    tumour_labels = tumour_labels_from_geojson(
        manifest,
        geojson_dir,
        patch_size=patch_size,
    )
    true_labels = (tumour_labels == "tumour").astype(int).to_numpy()

    print(f"\nClustering {len(manifest):,} patches with k={k}...")
    cluster_ids = cluster_embeddings(
        embeddings,
        n_clusters=k,
        random_state=random_state,
    )
    predicted = match_clusters_to_labels(cluster_ids, true_labels)

    clusters_df = cluster_summary(cluster_ids, true_labels)
    overall = _round_metrics(classification_metrics(true_labels, predicted))
    site_df = grouped_metrics(manifest, true_labels, predicted, group_col="site")
    wsi_df = grouped_metrics(manifest, true_labels, predicted, group_col="slide_id")

    overall_df = pd.DataFrame([{
        "accuracy": overall["accuracy"],
        "precision": overall["precision"],
        "recall": overall["recall"],
        "f1": overall["f1"],
        "tumour_prevalence_pct": round(overall["tumour_prevalence"] * 100, 2),
    }])

    _print_table("1. Global cluster summary", clusters_df)
    _print_table("2. Overall post-hoc evaluation", overall_df)
    _print_table("3. Per tissue/site evaluation", site_df)
    _print_table("4. Per WSI evaluation", wsi_df, max_rows=print_wsi_rows)

    run_dir = out_root / encoder / f"k{k}"
    run_dir.mkdir(parents=True, exist_ok=True)
    clusters_df.to_csv(run_dir / "global_cluster_summary.csv", index=False)
    overall_df.to_csv(run_dir / "overall_metrics.csv", index=False)
    site_df.to_csv(run_dir / "per_site_metrics.csv", index=False)
    wsi_df.to_csv(run_dir / "per_wsi_metrics.csv", index=False)

    summary = {
        "encoder": encoder,
        "k": k,
        "n_patches": int(len(manifest)),
        "n_slides": int(manifest["slide_id"].nunique()),
        "overall": overall,
        "outputs": [
            "global_cluster_summary.csv",
            "overall_metrics.csv",
            "per_site_metrics.csv",
            "per_wsi_metrics.csv",
        ],
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved H1 outputs -> {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H1 for CLIP, CONCH, and UNI.")
    parser.add_argument(
        "--config",
        default="hypotheses/H1_tumour_classification/exp01_kmeans_k2/config.yaml",
        help="Config used for embeddings_root, geojson_dir, random_state, and output dir.",
    )
    parser.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS)
    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_K_VALUES)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument(
        "--print-wsi-rows",
        type=int,
        default=20,
        help="How many WSI rows to print per run. Full table is always saved.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    embeddings_root = cfg["inputs"]["embeddings_root"]
    geojson_dir = cfg["inputs"]["geojson_dir"]
    random_state = cfg.get("params", {}).get("random_state", 42)
    out_root = Path("hypotheses/H1_tumour_classification/outputs")

    for encoder in args.encoders:
        for k in args.k_values:
            run_one(
                encoder=encoder,
                k=k,
                embeddings_root=embeddings_root,
                geojson_dir=geojson_dir,
                out_root=out_root,
                random_state=random_state,
                patch_size=args.patch_size,
                print_wsi_rows=args.print_wsi_rows,
            )


if __name__ == "__main__":
    main()
