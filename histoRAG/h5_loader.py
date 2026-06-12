"""
H1 embedding loader optimized for efficiency.

Provides fast loading from h5 files with optional caching.
"""

from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pandas as pd


def load_h5_embeddings_batch(h5_files: list[Path], skip_cache: bool = False) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load embeddings from multiple h5 files efficiently.
    
    Args:
        h5_files: List of .h5 file paths
        skip_cache: If False, checks for embeddings.npy + manifest.parquet
        
    Returns:
        embeddings: (N, dim) float32 array
        manifest: DataFrame with slide_id, x, y columns
    """
    cache_dir = Path("data/h5_cache")
    cache_emb = cache_dir / "embeddings.npy"
    cache_manifest = cache_dir / "manifest.parquet"
    
    # Try to load from cache
    if not skip_cache and cache_emb.exists() and cache_manifest.exists():
        print(f"Loading cached embeddings from {cache_dir}")
        return np.load(cache_emb), pd.read_parquet(cache_manifest)
    
    # Load from h5 files
    emb_parts, manifest_parts = [], []
    
    for h5_path in sorted(h5_files):
        try:
            with h5py.File(h5_path, "r") as hf:
                emb = np.array(hf["embeddings"], dtype=np.float32)
                x = np.array(hf["x"], dtype=np.int32) if "x" in hf else np.zeros(len(emb), dtype=np.int32)
                y = np.array(hf["y"], dtype=np.int32) if "y" in hf else np.zeros(len(emb), dtype=np.int32)
                
            emb_parts.append(emb)
            
            manifest_parts.append(pd.DataFrame({
                "slide_id": h5_path.stem,
                "x": x,
                "y": y,
            }))
        except Exception as e:
            print(f"Warning: Failed to load {h5_path}: {e}")
            continue
    
    if not emb_parts:
        raise RuntimeError("No h5 files loaded successfully")
    
    embeddings = np.concatenate(emb_parts, axis=0).astype(np.float32)
    manifest = pd.concat(manifest_parts, ignore_index=True)
    
    # Save to cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_emb, embeddings)
    manifest.to_parquet(cache_manifest, index=False)
    
    return embeddings, manifest


def load_h5_by_slide(h5_dir: Path) -> dict:
    """
    Load h5 files organized by slide_id.
    
    Useful for per-slide experiments.
    
    Args:
        h5_dir: Directory containing .h5 files
        
    Returns:
        dict[slide_id] -> (embeddings, manifest_row)
    """
    slides = {}
    
    for h5_path in sorted(h5_dir.glob("*.h5")):
        with h5py.File(h5_path, "r") as hf:
            emb = np.array(hf["embeddings"], dtype=np.float32)
            x = np.array(hf["x"], dtype=np.int32) if "x" in hf else np.zeros(len(emb), dtype=np.int32)
            y = np.array(hf["y"], dtype=np.int32) if "y" in hf else np.zeros(len(emb), dtype=np.int32)
        
        slide_id = h5_path.stem
        slides[slide_id] = {
            "embeddings": emb,
            "x": x,
            "y": y,
        }
    
    return slides
