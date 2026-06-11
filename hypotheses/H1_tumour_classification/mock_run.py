"""
H1 Mock Run — uses synthetic data to test the full pipeline without real files.

Creates fake embeddings and .geojson annotations in a temp directory,
then runs both exp01 (k=2) and exp02 (k=8) end-to-end.

Run from repo root:
    python hypotheses/H1_tumour_classification/mock_run.py
"""

import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import h5py
import numpy as np
import pandas as pd

from histoRAG.labels import tumour_labels_from_geojson
from histoRAG.classify import (
    cluster_embeddings,
    match_clusters_to_labels,
    classification_metrics,
)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def make_fake_embeddings(tmp: Path, n_slides: int = 4, n_patches: int = 200, dim: int = 512):
    """
    Write fake h5 files mimicking the CLIP/CONCH schema.
    Tumour patches are placed in a clearly separable region of embedding space
    so k=2 clustering should work.
    """
    rng = np.random.default_rng(42)
    h5_dir = tmp / "fake_encoder" / "h5_files"
    h5_dir.mkdir(parents=True)

    patch_size = 256  # pixels

    all_slides = []
    for s in range(n_slides):
        slide_id = f"PrimaryTumor_HE_{s+1:03d}"

        # half patches = tumour (centred around +1), half = other (around -1)
        n_tumour = n_patches // 2
        n_other  = n_patches - n_tumour

        tumour_emb = rng.standard_normal((n_tumour, dim)).astype(np.float32) + 1.0
        other_emb  = rng.standard_normal((n_other,  dim)).astype(np.float32) - 1.0
        embeddings = np.concatenate([tumour_emb, other_emb], axis=0)
        # L2 normalise
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        # coordinates: lay patches on a grid
        total = n_patches
        cols  = int(np.ceil(np.sqrt(total)))
        xs = np.array([(i % cols) * patch_size for i in range(total)], dtype=np.int32)
        ys = np.array([(i // cols) * patch_size for i in range(total)], dtype=np.int32)

        with h5py.File(h5_dir / f"{slide_id}.h5", "w") as hf:
            hf.create_dataset("embeddings", data=embeddings)
            hf.create_dataset("x", data=xs)
            hf.create_dataset("y", data=ys)
            hf.create_dataset("patch_ids", data=np.array(
                [f"{slide_id}_{i}".encode() for i in range(total)]
            ))

        all_slides.append({
            "slide_id":  slide_id,
            "n_tumour":  n_tumour,
            "xs":        xs[:n_tumour],
            "ys":        ys[:n_tumour],
            "patch_size": patch_size,
        })

    return h5_dir.parent, all_slides


def make_fake_geojson(tmp: Path, slides: list[dict]):
    """
    Write one .geojson per slide containing a polygon that covers the tumour patches.
    The polygon is a bounding box around the first n_tumour patch positions.
    """
    geojson_dir = tmp / "annotations"
    geojson_dir.mkdir(parents=True)

    for slide in slides:
        xs = slide["xs"]
        ys = slide["ys"]
        ps = slide["patch_size"]

        x_min = int(xs.min())
        x_max = int(xs.max()) + ps
        y_min = int(ys.min())
        y_max = int(ys.max()) + ps

        # simple bounding-box polygon (closed ring)
        polygon = [
            [x_min, y_min],
            [x_max, y_min],
            [x_max, y_max],
            [x_min, y_max],
            [x_min, y_min],   # close the ring
        ]

        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [polygon],
                },
                "properties": {"classification": {"name": "Tumor"}},
            }],
        }

        out = geojson_dir / f"{slide['slide_id']}.geojson"
        with open(out, "w") as f:
            json.dump(geojson, f)

    return geojson_dir


# ---------------------------------------------------------------------------
# Loader shim (reads from our fake h5 dir directly)
# ---------------------------------------------------------------------------

def load_fake(h5_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Load all h5 files in h5_dir into embeddings + manifest."""
    emb_parts, manifest_parts = [], []
    for h5_path in sorted(h5_dir.glob("*.h5")):
        with h5py.File(h5_path, "r") as hf:
            emb = np.array(hf["embeddings"], dtype=np.float32)
            xs  = np.array(hf["x"], dtype=np.int32)
            ys  = np.array(hf["y"], dtype=np.int32)
        emb_parts.append(emb)
        manifest_parts.append(pd.DataFrame({
            "slide_id": h5_path.stem,
            "site": "mock",
            "x": xs,
            "y": ys,
        }))

    embeddings = np.concatenate(emb_parts, axis=0).astype(np.float32)
    manifest   = pd.concat(manifest_parts, ignore_index=True)
    return embeddings, manifest


# ---------------------------------------------------------------------------
# Run both H1 experiments on the synthetic data
# ---------------------------------------------------------------------------

def run_experiment(name: str, n_clusters: int, embeddings, manifest, geojson_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  {name}  (k={n_clusters})")
    print(f"{'='*50}")
    print(f"  Patches: {len(manifest)} | Slides: {manifest['slide_id'].nunique()}")

    # ground-truth labels from geojson
    tumour_labels = tumour_labels_from_geojson(manifest, geojson_dir)
    true_labels   = (tumour_labels == "tumour").astype(int).values
    print(f"  GT -> Tumour: {true_labels.sum()} | Other: {(true_labels==0).sum()}")

    # unsupervised clustering
    cluster_ids = cluster_embeddings(embeddings, n_clusters=n_clusters, random_state=42)

    # map clusters -> tumour/other by majority vote
    predicted = match_clusters_to_labels(cluster_ids, true_labels)

    for cid in range(n_clusters):
        mask     = cluster_ids == cid
        label    = "tumour" if predicted[mask][0] == 1 else "other"
        gt_pct   = true_labels[mask].mean() * 100
        print(f"  Cluster {cid}: {mask.sum():>5} patches | {gt_pct:.0f}% GT tumour -> '{label}'")

    metrics = classification_metrics(true_labels, predicted)
    print(f"  Accuracy={metrics['accuracy']:.3f}  Precision={metrics['precision']:.3f}  Recall={metrics['recall']:.3f}")

    summary = {
        "experiment": name, "n_clusters": n_clusters,
        "n_patches": int(len(manifest)),
        "n_tumour_gt": int(true_labels.sum()),
        "metrics": metrics,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved -> {out_dir / 'summary.json'}")
    return metrics


def main():
    print("H1 Mock Run — synthetic data")
    print("Building fake embeddings and annotations...")

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)

        fake_enc_dir, slides = make_fake_embeddings(tmp)
        geojson_dir          = make_fake_geojson(tmp, slides)
        embeddings, manifest = load_fake(fake_enc_dir / "h5_files")

        out_root = Path("hypotheses/H1_tumour_classification")

        m1 = run_experiment(
            "H1_exp01_kmeans_k2", n_clusters=2,
            embeddings=embeddings, manifest=manifest,
            geojson_dir=geojson_dir,
            out_dir=out_root / "exp01_kmeans_k2" / "outputs",
        )

        m2 = run_experiment(
            "H1_exp02_overcluster_assign", n_clusters=8,
            embeddings=embeddings, manifest=manifest,
            geojson_dir=geojson_dir,
            out_dir=out_root / "exp02_overcluster_assign" / "outputs",
        )

    print("\n" + "="*50)
    print("  MOCK RUN COMPLETE")
    print("="*50)
    print(f"  exp01 (k=2): Accuracy={m1['accuracy']:.3f}  Precision={m1['precision']:.3f}  Recall={m1['recall']:.3f}")
    print(f"  exp02 (k=8): Accuracy={m2['accuracy']:.3f}  Precision={m2['precision']:.3f}  Recall={m2['recall']:.3f}")
    print("\n  Both experiments ran end-to-end. Pipeline is working.")
    print("  To run on real data: download .geojson files from HANCOCK, update config.yaml, run run.py")


if __name__ == "__main__":
    main()
