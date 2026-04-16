"""Query/gallery split strategies for retrieval evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def stratified_within_slide(
    manifest: pd.DataFrame,
    query_frac: float = 0.2,
    seed: int = 42,
) -> tuple[pd.Index, pd.Index]:
    """Split patches into query and gallery, stratified by (slide_id, label).

    Each (slide_id, label) group is split independently so that each class
    is proportionally represented in the query set. Patches from the same
    slide and class appear in either query or gallery, never both.

    Parameters
    ----------
    manifest:
        DataFrame with columns ['patch_id', 'slide_id', 'label'] at minimum.
    query_frac:
        Fraction of patches allocated to queries (per group).
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    (query_ids, gallery_ids) as pandas Index over manifest.index.
    """
    rng = np.random.default_rng(seed)
    query_idx, gallery_idx = [], []

    for _, group in manifest.groupby(["slide_id", "label"]):
        n = len(group)
        n_query = max(1, int(np.round(n * query_frac)))
        perm = rng.permutation(n)
        query_idx.extend(group.index[perm[:n_query]].tolist())
        gallery_idx.extend(group.index[perm[n_query:]].tolist())

    return pd.Index(query_idx), pd.Index(gallery_idx)


def slide_leave_out(
    manifest: pd.DataFrame,
    held_out_slides: list[str],
    seed: int = 42,
) -> tuple[pd.Index, pd.Index]:
    """Use *held_out_slides* as query set; all other slides as gallery.

    Stub for Phase-2 Pro-3 cross-slide retrieval evaluation.
    Tested in unit tests but not invoked in the Phase 0 pipeline.
    """
    query_mask = manifest["slide_id"].isin(held_out_slides)
    return manifest.index[query_mask], manifest.index[~query_mask]
