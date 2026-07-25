"""
Split the combined 4-curve similarity_distribution.png into 4 separate
2-curve graphs (tumour-only, non-tumour-only) per encoder, from cache.

Usage:
    python hypotheses/H2_patient_correlation/exp02_tumour_similarity/plot_split_similarity.py
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from histoRAG.correlate import plot_similarity_distribution

_HERE = Path(__file__).parent

for encoder in ["conch", "uni2h"]:
    cache_dir = _HERE / "outputs" / encoder / "cache"
    out_dir   = _HERE / "outputs" / encoder
    within_t  = np.load(cache_dir / "tumour_within_sims.npy")
    cross_t   = np.load(cache_dir / "tumour_cross_sims.npy")
    within_nt = np.load(cache_dir / "nontumour_within_sims.npy")
    cross_nt  = np.load(cache_dir / "nontumour_cross_sims.npy")

    plot_similarity_distribution(
        within_t=within_t, cross_t=cross_t,
        out_path=out_dir / "similarity_distribution_tumour.png",
        title=f"Tumour Patch Similarity — within-site vs cross-site ({encoder})",
        class_label="Tumour",
    )
    plot_similarity_distribution(
        within_t=within_nt, cross_t=cross_nt,
        out_path=out_dir / "similarity_distribution_nontumour.png",
        title=f"Non-tumour Patch Similarity — within-site vs cross-site ({encoder})",
        class_label="Non-tumour",
        color="#1e88e5",
    )
