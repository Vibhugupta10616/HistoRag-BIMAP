"""
Load pre-computed patch embeddings from h5 files on disk into a flat
numpy array + manifest DataFrame ready for H1 and H2 experiments.

Two h5 schemas are supported (auto-detected per file):
  CLIP / CONCH  — keys: embeddings (N,dim), x (N,), y (N,), patch_ids (N,)
  UNI           — keys: features (N,dim), coords (N,2)

Directory layout expected under embeddings_root:
  CLIP/Primary_Tumour/<Site>/h5_files/*.h5
  CONCH/Primary_Tumour/<Site>/h5_files/*.h5
  UNI/WSI_PrimaryTumor/WSI_PrimaryTumor_<Site>/h5_files/*.h5

Site folder names (CLIP/CONCH)  ->  canonical site label used in manifest:
  Hypopharynx   -> hypopharynx
  Larynx        -> larynx
  Oral_Cavity   -> oral_cavity
  Oropharynx1   -> oropharynx
  Oropharynx2   -> oropharynx   (merged — same anatomical site, two download batches)

UNI folder suffix  ->  canonical site label:
  Hypopharynx        -> hypopharynx
  Oropharynx_Part1   -> oropharynx
  (remaining sites added as they are unzipped)
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Site-name normalisation maps
# ---------------------------------------------------------------------------

# CLIP / CONCH subfolder name -> canonical site
_CLIP_CONCH_SITE_MAP: dict[str, str] = {
    "Hypopharynx":  "hypopharynx",
    "Larynx":       "larynx",
    "Oral_Cavity":  "oral_cavity",
    "Oropharynx1":  "oropharynx",
    "Oropharynx2":  "oropharynx",
}

# UNI subfolder suffix (after "WSI_PrimaryTumor_") -> canonical site
_UNI_SITE_MAP: dict[str, str] = {
    "Hypopharynx":      "hypopharynx",
    "Larynx":           "larynx",
    "OralCavity":       "oral_cavity",   # UNI uses OralCavity (no underscore)
    "Oropharynx_Part1": "oropharynx",
    "Oropharynx_Part2": "oropharynx",
}


# ---------------------------------------------------------------------------
# Single-file reader (handles both schemas)
# ---------------------------------------------------------------------------

def _read_h5(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read one h5 file and return (embeddings, x_coords, y_coords).

    embeddings: (N, dim) float32
    x_coords:   (N,)    int32
    y_coords:   (N,)    int32
    """
    with h5py.File(path, "r") as hf:
        keys = set(hf.keys())

        if "features" in keys:
            # UNI schema
            embeddings = np.array(hf["features"], dtype=np.float32)
            coords = np.array(hf["coords"], dtype=np.int32)  # (N, 2)
            x = coords[:, 0]
            y = coords[:, 1]
        elif "embeddings" in keys:
            # CLIP / CONCH schema
            embeddings = np.array(hf["embeddings"], dtype=np.float32)
            x = np.array(hf["x"], dtype=np.int32) if "x" in keys else np.zeros(len(embeddings), dtype=np.int32)
            y = np.array(hf["y"], dtype=np.int32) if "y" in keys else np.zeros(len(embeddings), dtype=np.int32)
        else:
            raise KeyError(
                f"Unrecognised h5 schema in {path}. "
                f"Expected 'embeddings' or 'features'. Found: {sorted(keys)}"
            )

    return embeddings, x, y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_encoder(
    encoder: str,
    embeddings_root: str | Path,
    sites: list[str] | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load all patch embeddings for one encoder across all available sites.

    Args:
        encoder:         One of 'clip-vitb16', 'conch', 'uni2h'.
        embeddings_root: Root directory (e.g. E:/embeddings).
        sites:           Optional list of canonical site names to restrict to
                         (e.g. ['larynx', 'hypopharynx']). None = all available.

    Returns:
        embeddings: (N_total_patches, dim) float32, row-aligned to manifest.
        manifest:   DataFrame with columns:
                      slide_id  — patient identifier (stem of h5 filename)
                      site      — canonical anatomical site label
                      x, y      — patch top-left coordinates in WSI pixel space
    """
    embeddings_root = Path(embeddings_root)
    h5_paths = _discover_h5_files(encoder, embeddings_root)

    if not h5_paths:
        raise FileNotFoundError(
            f"No h5 files found for encoder '{encoder}' under {embeddings_root}. "
            "Check that the embeddings are unzipped and the directory layout matches "
            "the expected structure (see module docstring)."
        )

    emb_parts: list[np.ndarray] = []
    manifest_parts: list[pd.DataFrame] = []

    for h5_path, site in sorted(h5_paths, key=lambda t: (t[1], t[0].name)):
        if sites is not None and site not in sites:
            continue

        slide_id = h5_path.stem
        emb, x_coords, y_coords = _read_h5(h5_path)

        emb_parts.append(emb)
        manifest_parts.append(pd.DataFrame({
            "slide_id": slide_id,
            "site":     site,
            "x":        x_coords,
            "y":        y_coords,
        }))
        print(f"[loader] {slide_id}  site={site}  {len(emb)} patches  dim={emb.shape[1]}")

    if not emb_parts:
        available = {site for _, site in h5_paths}
        raise ValueError(
            f"No data loaded. Requested sites: {sites}. "
            f"Available: {sorted(available)}"
        )

    all_embeddings = np.concatenate(emb_parts, axis=0).astype(np.float32)
    manifest = pd.concat(manifest_parts, ignore_index=True)

    print(
        f"[loader] {encoder}: {len(manifest)} patches | "
        f"{manifest['slide_id'].nunique()} slides | "
        f"sites={sorted(manifest['site'].unique())} | dim={all_embeddings.shape[1]}"
    )
    return all_embeddings, manifest


def available_sites(encoder: str, embeddings_root: str | Path) -> list[str]:
    """Return sorted list of canonical site names available for an encoder."""
    h5_paths = _discover_h5_files(encoder, Path(embeddings_root))
    return sorted({site for _, site in h5_paths})


def detect_patch_size(encoder: str, embeddings_root: str | Path) -> int:
    """
    Infer the patch size (in WSI level-0 pixels) from coordinate step patterns.

    Reads a single h5 file and computes the most common distance between
    consecutive unique x-coordinates. For non-overlapping tiles this equals
    the patch width at the base WSI resolution.

    Returns the detected patch size as an int.
    """
    embeddings_root = Path(embeddings_root)
    h5_paths = _discover_h5_files(encoder, embeddings_root)
    if not h5_paths:
        raise FileNotFoundError(f"No h5 files found for encoder '{encoder}'")

    h5_path = sorted(h5_paths, key=lambda t: t[0].name)[0][0]
    with h5py.File(h5_path, "r") as hf:
        if "coords" in hf:
            x = np.sort(np.unique(np.array(hf["coords"])[:, 0]))
        else:
            x = np.sort(np.unique(np.array(hf["x"])))

    steps = np.diff(x).astype(int)
    if len(steps) == 0:
        return 256
    patch_size = int(np.bincount(steps).argmax())
    return patch_size


def count_encoder_patches(
    encoder: str,
    embeddings_root: str | Path,
    sites: list[str] | None = None,
) -> dict[str, int]:
    """
    Return {slide_id: n_patches} by reading only h5 dataset shapes.

    No embedding data is loaded — this is fast and uses negligible RAM.
    Use this to compute proportional subsample quotas before loading.
    """
    embeddings_root = Path(embeddings_root)
    h5_paths = _discover_h5_files(encoder, embeddings_root)

    counts: dict[str, int] = {}
    for h5_path, site in sorted(h5_paths, key=lambda t: (t[1], t[0].name)):
        if sites is not None and site not in sites:
            continue
        with h5py.File(h5_path, "r") as hf:
            key = "features" if "features" in hf else "embeddings"
            counts[h5_path.stem] = hf[key].shape[0]
    return counts


def iter_encoder(
    encoder: str,
    embeddings_root: str | Path,
    sites: list[str] | None = None,
):
    """
    Yield (embeddings, manifest_df) one slide at a time.

    Memory-efficient alternative to load_encoder — never holds more than one
    slide's embeddings in RAM simultaneously.  Use when the full dataset does
    not fit in memory.

    Yields:
        embeddings:   (N_patches, dim) float32 for the slide
        manifest_df:  DataFrame with columns slide_id, site, x, y for the slide
    """
    embeddings_root = Path(embeddings_root)
    h5_paths = _discover_h5_files(encoder, embeddings_root)

    if not h5_paths:
        raise FileNotFoundError(
            f"No h5 files found for encoder '{encoder}' under {embeddings_root}."
        )

    for h5_path, site in sorted(h5_paths, key=lambda t: (t[1], t[0].name)):
        if sites is not None and site not in sites:
            continue
        slide_id = h5_path.stem
        emb, x_coords, y_coords = _read_h5(h5_path)
        df = pd.DataFrame({
            "slide_id": slide_id,
            "site":     site,
            "x":        x_coords,
            "y":        y_coords,
        })
        yield emb, df





def _discover_h5_files(
    encoder: str,
    embeddings_root: Path,
) -> list[tuple[Path, str]]:
    """
    Walk the directory tree and return [(h5_path, canonical_site), ...].
    """
    results: list[tuple[Path, str]] = []

    enc_lower = encoder.lower().replace("-", "").replace("_", "")

    if "clip" in enc_lower:
        base = embeddings_root / "CLIP" / "Primary_Tumour"
        for folder_name, site in _CLIP_CONCH_SITE_MAP.items():
            h5_dir = base / folder_name / "h5_files"
            if h5_dir.is_dir():
                for h5_file in h5_dir.glob("*.h5"):
                    results.append((h5_file, site))

    elif "conch" in enc_lower:
        base = embeddings_root / "CONCH" / "Primary_Tumour"
        for folder_name, site in _CLIP_CONCH_SITE_MAP.items():
            h5_dir = base / folder_name / "h5_files"
            if h5_dir.is_dir():
                for h5_file in h5_dir.glob("*.h5"):
                    results.append((h5_file, site))

    elif "uni" in enc_lower:
        base = embeddings_root / "UNI" / "WSI_PrimaryTumor"
        for folder_name, site in _UNI_SITE_MAP.items():
            h5_dir = base / f"WSI_PrimaryTumor_{folder_name}" / "h5_files"
            if h5_dir.is_dir():
                for h5_file in h5_dir.glob("*.h5"):
                    results.append((h5_file, site))

    else:
        raise ValueError(
            f"Unknown encoder '{encoder}'. Expected one of: clip-vitb16, conch, uni2h."
        )

    return results
