"""
HPC pipeline: auto-discover WSIs -> tile -> embed -> save per-patient embeddings.

Completely self-contained — only imports from HPC/ folder.
No dependency on histoRAG/, MVP/, or any other part of the project.

Supports CLIP and CONCH encoders via --encoder flag.
Run once per encoder — tiling is shared and computed only once.

Usage (from HPC/ directory, after activating HPC/hpcenv):
    # CLIP embeddings for Larynx
    python hpc_pipeline.py \
        --wsi_dir     $WORK/hancock/larynx/wsi \
        --out_dir     $WORK/hancock/embeddings \
        --patches_dir $WORK/hancock/larynx/patches \
        --encoder     clip \
        --tissue      Larynx

    # CONCH embeddings for Hypopharynx
    python hpc_pipeline.py \
        --wsi_dir     $WORK/hancock/hypopharynx/wsi \
        --out_dir     $WORK/hancock/embeddings \
        --patches_dir $WORK/hancock/hypopharynx/patches \
        --encoder     conch \
        --tissue      Hypopharynx

Output structure:
    $out_dir/
      CLIP/
        Primary_Tumour/
          Larynx/
            h5_files/
              {slide_id}.h5
              Larynx_CLIP_embeddings.zip   (created by embed_job.sh for download)
          Hypopharynx/
            h5_files/
              {slide_id}.h5
      CONCH/
        Primary_Tumour/
          Larynx/
            h5_files/
              {slide_id}.h5

Each .h5 file contains:
    embeddings  (N_patches, 512) float32   L2-normalised vectors
    patch_ids   (N_patches,)     bytes     unique patch identifier
    x           (N_patches,)     int32     top-left x in level-0 pixels
    y           (N_patches,)     int32     top-left y in level-0 pixels

Reading locally:
    import h5py, numpy as np
    with h5py.File("CLIP/Primary_Tumour/Larynx/h5_files/PrimaryTumor_HE_152.h5", "r") as f:
        emb = f["embeddings"][:]   # (N, 512)
        x, y = f["x"][:], f["y"][:]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
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

# Map CLI encoder name -> folder name used in output path
_ENCODER_DIR = {"clip": "CLIP", "conch": "CONCH"}


def discover_slides(wsi_dir: Path) -> list[Path]:
    """Return all WSI files found recursively under wsi_dir, sorted by name."""
    found = []
    for ext in _WSI_EXTENSIONS:
        found.extend(wsi_dir.rglob(f"*{ext}"))
    found.sort()
    return found


def _tile_worker(args: tuple) -> list[dict]:
    """Tile a single slide in a subprocess. Must be top-level for pickling."""
    slide_path, slide_id, patches_dir, max_patches, tissue, hpc_dir = args

    # Re-insert HPC/ path since this runs in a fresh subprocess
    sys.path.insert(0, hpc_dir)
    from wsi_tiler import Tiler, WSI  # noqa: PLC0415

    tiler = Tiler(
        patch_size=256,
        stride=256,
        target_magnification=20.0,
        thumb_downsample=32,
        min_tissue_frac=0.01,  # discard only pure-white patches
        max_patches_per_slide=max_patches,
        seed=42,
    )
    try:
        with WSI(Path(slide_path)) as wsi:
            rows = tiler.extract(wsi, slide_id=slide_id, out_dir=Path(patches_dir), label=tissue)
        print(f"  [done] {slide_id}: {len(rows)} patches", flush=True)
        return rows
    except Exception as exc:
        print(f"  [skip] {slide_id} failed: {exc}", flush=True)
        return []


def tile_all(
    slide_files: list[Path],
    patches_dir: Path,
    max_patches: int,
    tissue: str,
    num_workers: int,
) -> pd.DataFrame:
    """Tile every slide in parallel and return the combined patch manifest.

    Saves manifest after each completed slide — safe to resume after a crash.
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

    remaining = [p for p in slide_files if p.stem not in already_done]
    if not remaining:
        print("All slides already tiled.")
        return pd.DataFrame(all_rows)

    hpc_dir = str(Path(__file__).resolve().parent)
    worker_args = [
        (str(p), p.stem, str(patches_dir), max_patches, tissue, hpc_dir)
        for p in remaining
    ]

    print(f"Tiling {len(remaining)} slides with {num_workers} workers ...\n")

    # imap_unordered returns results as each worker finishes — saves incrementally
    with mp.Pool(processes=num_workers) as pool:
        for rows in pool.imap_unordered(_tile_worker, worker_args):
            if rows:
                all_rows.extend(rows)
                pd.DataFrame(all_rows).to_parquet(manifest_path, index=False)

    manifest = pd.DataFrame(all_rows)
    print(f"\nTiling complete: {len(manifest)} patches across {manifest['slide_id'].nunique()} slides")
    return manifest


def embed_and_save_per_slide(
    manifest: pd.DataFrame,
    out_dir: Path,
    encoder_name: str,
    tissue: str,
    batch_size: int,
) -> None:
    """Encode all patches and save one HDF5 per slide.

    Output: out_dir/{ENCODER}/Primary_Tumour/{tissue}/h5_files/PrimaryTumor_HE_{slide_id}.h5
    Skips slides whose HDF5 already exists — safe to resume after a crash.
    """
    encoder_dir = _ENCODER_DIR[encoder_name]
    h5_dir = out_dir / encoder_dir / "Primary_Tumour" / tissue / "h5_files"
    h5_dir.mkdir(parents=True, exist_ok=True)

    slide_ids = list(manifest["slide_id"].unique())

    already_done = {
        s for s in slide_ids
        if (h5_dir / f"{s}.h5").exists()
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
        print(f"\nEmbedding {slide_id}  ({len(slide_rows)} patches)  [{encoder_dir}] ...")

        images = [
            Image.open(p).convert("RGB")
            for p in tqdm(slide_rows["path"], desc="  Loading", leave=False)
        ]

        t0 = time.time()
        embeddings = encoder.encode_batched(images, batch_size=batch_size)
        elapsed = time.time() - t0

        h5_path = h5_dir / f"{slide_id}.h5"

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("embeddings", data=embeddings, compression="gzip", compression_opts=4)
            f.create_dataset("patch_ids",  data=np.array(slide_rows["patch_id"].tolist(), dtype="S"))
            f.create_dataset("x",          data=slide_rows["x"].to_numpy(dtype=np.int32))
            f.create_dataset("y",          data=slide_rows["y"].to_numpy(dtype=np.int32))
            f.attrs["slide_id"]    = slide_id
            f.attrs["encoder"]     = encoder_dir
            f.attrs["tissue"]      = tissue
            f.attrs["n_patches"]   = len(slide_rows)
            f.attrs["dim"]         = embeddings.shape[1]

        print(f"  -> {h5_path}  shape={embeddings.shape}  ({elapsed:.0f}s, {len(slide_rows)/elapsed:.0f} p/s)")

    print(f"\nAll slides embedded in {time.time() - t_total:.0f}s")
    print(f"Output: {h5_dir}")


def cleanup_wsi_if_patching_complete(manifest: pd.DataFrame, patches_dir: Path, wsi_dir: Path) -> None:
    """Delete WSI files after verifying all slides have patch directories with PNGs."""
    slide_ids = list(manifest["slide_id"].unique())

    # Verify every slide has a non-empty patch directory
    missing = [s for s in slide_ids if not any((patches_dir / s).glob("*.png"))]
    if missing:
        print(f"  WARNING: {len(missing)} slides have no patches — skipping WSI deletion.")
        print(f"  Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        return

    print(f"  All {len(slide_ids)} slides verified — deleting WSI directory to free disk space ...")
    shutil.rmtree(wsi_dir, ignore_errors=True)
    print(f"  Deleted: {wsi_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HistoRAG HPC pipeline — patch embedding")
    parser.add_argument("--wsi_dir",     required=True, help="Directory containing extracted WSI files")
    parser.add_argument("--out_dir",     required=True, help="Root embeddings output directory ($WORK/hancock/embeddings)")
    parser.add_argument("--patches_dir", required=True, help="Directory to save patch PNG files")
    parser.add_argument("--encoder",     default="clip", choices=["clip", "conch"], help="Encoder to use")
    parser.add_argument("--tissue",      default="Larynx", help="Tissue name for output path (e.g. Larynx, Hypopharynx)")
    parser.add_argument("--max_patches", type=int, default=None, help="Max patches sampled per slide (default: no limit)")
    parser.add_argument("--batch_size",  type=int, default=256,  help="Encoding batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="Parallel tiling workers (default: CPU count - 2)")
    args = parser.parse_args()

    wsi_dir     = Path(args.wsi_dir)
    out_dir     = Path(args.out_dir)
    patches_dir = Path(args.patches_dir)
    num_workers = args.num_workers or max(1, (os.cpu_count() or 4) - 2)

    print(f"\n{'='*60}")
    print(f"HistoRAG HPC Pipeline  |  encoder={args.encoder.upper()}  tissue={args.tissue}")
    print(f"  WSI dir    : {wsi_dir}")
    print(f"  Patches    : {patches_dir}")
    print(f"  Output     : {out_dir / _ENCODER_DIR[args.encoder] / 'Primary_Tumour' / args.tissue / 'h5_files'}")
    print(f"  Workers    : {num_workers}")
    print(f"{'='*60}\n")

    manifest_path = patches_dir / "manifest.parquet"
    if manifest_path.exists():
        # Tiling already done — load manifest directly, no WSI files needed
        print(f"Manifest found — skipping tiling, loading from {manifest_path}")
        manifest = pd.read_parquet(manifest_path)
        print(f"Loaded {len(manifest)} patches across {manifest['slide_id'].nunique()} slides\n")
    else:
        slides = discover_slides(wsi_dir)
        if not slides:
            raise RuntimeError(f"No WSI files found in {wsi_dir}. Check that extraction completed.")
        print(f"Discovered {len(slides)} slides\n")
        manifest = tile_all(slides, patches_dir, args.max_patches, args.tissue, num_workers)
        print("\nVerifying patches and cleaning up WSI files ...")
        cleanup_wsi_if_patching_complete(manifest, patches_dir, wsi_dir)

    embed_and_save_per_slide(manifest, out_dir, args.encoder, args.tissue, args.batch_size)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
