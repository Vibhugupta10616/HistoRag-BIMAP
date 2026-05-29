"""
HPC pipeline: auto-discover WSIs -> tile -> embed -> save per-patient embeddings.

Completely self-contained — only imports from HPC/ folder.
No dependency on histoRAG/, MVP/, or any other part of the project.

Supports CLIP and CONCH encoders via --encoder flag.
Run once per encoder — tiling is shared and computed only once.

Usage (from HPC/ directory, after activating HPC/hpcenv):
    # CLIP embeddings
    python hpc_pipeline.py \
        --wsi_dir     $WORK/hancock/wsi \
        --out_dir     $WORK/hancock/embeddings \
        --patches_dir $WORK/hancock/patches \
        --encoder     clip

    # CONCH embeddings (run after: pip install git+https://github.com/mahmoodlab/CONCH)
    python hpc_pipeline.py \
        --wsi_dir     $WORK/hancock/wsi \
        --out_dir     $WORK/hancock/embeddings \
        --patches_dir $WORK/hancock/patches \
        --encoder     conch

Output structure:
    $out_dir/
      manifest.parquet                          shared across encoders
      per_slide/
        {slide_id}/
          clip/
            embeddings.h5                       CLIP patch embeddings
          conch/
            embeddings.h5                       CONCH patch embeddings

Each embeddings.h5 contains:
    embeddings  (N_patches, 512) float32   L2-normalised vectors
    patch_ids   (N_patches,)     bytes     unique patch identifier
    x           (N_patches,)     int32     top-left x in level-0 pixels
    y           (N_patches,)     int32     top-left y in level-0 pixels

Reading locally:
    import h5py, numpy as np
    with h5py.File("per_slide/patient_001/clip/embeddings.h5", "r") as f:
        emb = f["embeddings"][:]   # (N, 512)
        x, y = f["x"][:], f["y"][:]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure HPC/ directory is on path so local modules resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent))

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from encoders import get_encoder
from wsi_tiler import Tiler, WSI

_WSI_EXTENSIONS = [".svs", ".tiff", ".tif", ".ndpi", ".mrxs", ".scn"]


def discover_slides(wsi_dir: Path) -> list[Path]:
    """Return all WSI files found recursively under wsi_dir, sorted by name."""
    found = []
    for ext in _WSI_EXTENSIONS:
        found.extend(wsi_dir.rglob(f"*{ext}"))
    found.sort()
    return found


def tile_all(
    slide_files: list[Path],
    patches_dir: Path,
    max_patches: int,
    label: str,
) -> pd.DataFrame:
    """Tile every slide and return the combined patch manifest.

    Saves manifest after every slide — safe to resume after a crash.
    """
    manifest_path = patches_dir / "manifest.parquet"

    already_done: set[str] = set()
    if manifest_path.exists():
        existing = pd.read_parquet(manifest_path)
        already_done = set(existing["slide_id"].unique())
        print(f"Resuming tiling: {len(already_done)} slides already done, skipping.")
        all_rows = existing.to_dict("records")
    else:
        all_rows = []

    tiler = Tiler(
        patch_size=256,
        stride=256,
        target_magnification=20.0,
        thumb_downsample=32,
        min_tissue_frac=0.01,  # discard only pure-white patches
        max_patches_per_slide=max_patches,
        seed=42,
    )

    for slide_path in slide_files:
        slide_id = slide_path.stem
        if slide_id in already_done:
            continue

        print(f"\nTiling {slide_id} ...")
        try:
            with WSI(slide_path) as wsi:
                rows = tiler.extract(wsi, slide_id=slide_id, out_dir=patches_dir, label=label)
            print(f"  -> {len(rows)} patches")
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_parquet(manifest_path, index=False)
        except Exception as exc:
            print(f"  WARNING: {slide_id} failed ({exc}), skipping.")

    manifest = pd.DataFrame(all_rows)
    print(f"\nTiling complete: {len(manifest)} patches across {manifest['slide_id'].nunique()} slides")
    return manifest


def embed_and_save_per_slide(
    manifest: pd.DataFrame,
    out_dir: Path,
    encoder_name: str,
    batch_size: int,
) -> None:
    """Encode all patches and save one HDF5 per patient under the encoder subfolder.

    Output: out_dir/per_slide/{slide_id}/{encoder_name}/embeddings.h5
    Skips slides whose HDF5 already exists — safe to resume after a crash.
    """
    slide_ids = list(manifest["slide_id"].unique())

    already_done = {
        s for s in slide_ids
        if (out_dir / "per_slide" / s / encoder_name / "embeddings.h5").exists()
    }
    remaining = [s for s in slide_ids if s not in already_done]

    if already_done:
        print(f"Resuming {encoder_name}: {len(already_done)} done, {len(remaining)} remaining.")
    if not remaining:
        print(f"All slides already embedded with {encoder_name}.")
        return

    print(f"\nLoading {encoder_name} encoder ...")
    encoder = get_encoder(encoder_name)

    t_total = time.time()

    for slide_id in remaining:
        slide_rows = manifest[manifest["slide_id"] == slide_id].reset_index(drop=True)
        print(f"\nEmbedding {slide_id}  ({len(slide_rows)} patches)  [{encoder_name}] ...")

        images = [
            Image.open(p).convert("RGB")
            for p in tqdm(slide_rows["path"], desc="  Loading", leave=False)
        ]

        t0 = time.time()
        embeddings = encoder.encode_batched(images, batch_size=batch_size)
        elapsed = time.time() - t0

        slide_dir = out_dir / "per_slide" / slide_id / encoder_name
        slide_dir.mkdir(parents=True, exist_ok=True)
        h5_path = slide_dir / "embeddings.h5"

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("embeddings", data=embeddings, compression="gzip", compression_opts=4)
            f.create_dataset("patch_ids",  data=np.array(slide_rows["patch_id"].tolist(), dtype="S"))
            f.create_dataset("x",          data=slide_rows["x"].to_numpy(dtype=np.int32))
            f.create_dataset("y",          data=slide_rows["y"].to_numpy(dtype=np.int32))
            f.attrs["slide_id"]    = slide_id
            f.attrs["encoder"]     = encoder_name
            f.attrs["n_patches"]   = len(slide_rows)
            f.attrs["dim"]         = embeddings.shape[1]

        print(f"  -> {h5_path}  shape={embeddings.shape}  ({elapsed:.0f}s, {len(slide_rows)/elapsed:.0f} p/s)")

    print(f"\nAll slides embedded in {time.time() - t_total:.0f}s")
    print(f"Output: {out_dir / 'per_slide'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HistoRAG HPC pipeline — CLIP patch embedding")
    parser.add_argument("--wsi_dir",     required=True, help="Directory containing extracted WSI files")
    parser.add_argument("--out_dir",     required=True, help="Output directory for embeddings")
    parser.add_argument("--patches_dir", required=True, help="Directory to save patch PNG files")
    parser.add_argument("--encoder",     default="clip", choices=["clip", "conch"], help="Encoder to use")
    parser.add_argument("--label",       default="larynx", help="Label applied to all slides (e.g. larynx)")
    parser.add_argument("--max_patches", type=int, default=5000, help="Max patches sampled per slide")
    parser.add_argument("--batch_size",  type=int, default=64,   help="Encoding batch size")
    args = parser.parse_args()

    wsi_dir     = Path(args.wsi_dir)
    out_dir     = Path(args.out_dir)
    patches_dir = Path(args.patches_dir)

    print(f"\n{'='*60}")
    print(f"HistoRAG HPC Pipeline  |  encoder={args.encoder}")
    print(f"  WSI dir    : {wsi_dir}")
    print(f"  Patches    : {patches_dir}")
    print(f"  Output     : {out_dir}")
    print(f"{'='*60}\n")

    slides = discover_slides(wsi_dir)
    if not slides:
        raise RuntimeError(f"No WSI files found in {wsi_dir}. Check that extraction completed.")
    print(f"Discovered {len(slides)} slides\n")

    manifest = tile_all(slides, patches_dir, args.max_patches, args.label)
    embed_and_save_per_slide(manifest, out_dir, args.encoder, args.batch_size)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
