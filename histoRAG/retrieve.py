"""Query/gallery split and retrieval evaluation metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def stratified_within_slide(
    manifest: pd.DataFrame, query_frac: float = 0.2, seed: int = 42
) -> tuple[pd.Index, pd.Index]:
    """Split patches into query/gallery, stratified by (slide_id, label).

    Returns (query_ids, gallery_ids) as pandas Index over manifest.index.
    """
    rng = np.random.default_rng(seed)
    query_idx, gallery_idx = [], []
    for _, group in manifest.groupby(["slide_id", "label"]):
        n = len(group)
        perm = rng.permutation(n)
        n_query = max(1, int(np.round(n * query_frac)))
        query_idx.extend(group.index[perm[:n_query]].tolist())
        gallery_idx.extend(group.index[perm[n_query:]].tolist())
    return pd.Index(query_idx), pd.Index(gallery_idx)


def slide_leave_out(
    manifest: pd.DataFrame, held_out_slides: list[str], seed: int = 42
) -> tuple[pd.Index, pd.Index]:
    """Use held_out_slides as query set; all other slides as gallery (Phase-2 hook)."""
    mask = manifest["slide_id"].isin(held_out_slides)
    return manifest.index[mask], manifest.index[~mask]


def top_k_accuracy(retrieved_labels: np.ndarray, query_labels: np.ndarray, k: int) -> float:
    """Fraction of queries where the correct label appears in the top-k results."""
    hits = (retrieved_labels[:, :k] == query_labels[:, None]).any(axis=1)
    return float(hits.mean())


def mean_average_precision(retrieved_labels: np.ndarray, query_labels: np.ndarray, k: int) -> float:
    """Mean Average Precision at k (mAP@k)."""
    top_k = retrieved_labels[:, :k]
    aps = []
    for q in range(len(query_labels)):
        rels = (top_k[q] == query_labels[q]).astype(float)
        n_rel = rels.sum()
        if n_rel == 0:
            aps.append(0.0)
            continue
        cumulative = np.cumsum(rels)
        ranks = np.arange(1, k + 1, dtype=float)
        aps.append(float((cumulative / ranks * rels).sum() / n_rel))
    return float(np.mean(aps))


def random_baseline(n_classes: int, k: int) -> float:
    """Top-k accuracy expected from a uniformly random retriever."""
    if n_classes <= 0:
        return 0.0
    return 1.0 - ((n_classes - 1) / n_classes) ** k
