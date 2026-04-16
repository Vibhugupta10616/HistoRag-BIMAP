"""Retrieval evaluation metrics: top-k accuracy, mAP@k, random baseline."""
from __future__ import annotations

import numpy as np


def top_k_accuracy(
    retrieved_labels: np.ndarray,
    query_labels: np.ndarray,
    k: int,
) -> float:
    """Fraction of queries where the correct label appears in the top-k results.

    Parameters
    ----------
    retrieved_labels:
        (Q, K_max) array of class labels for retrieved patches (best rank first).
    query_labels:
        (Q,) ground-truth labels for each query patch.
    k:
        Cutoff rank; only the first *k* columns of *retrieved_labels* are used.

    Returns
    -------
    float in [0, 1]
    """
    if k > retrieved_labels.shape[1]:
        raise ValueError(f"k={k} exceeds available retrieved columns {retrieved_labels.shape[1]}")
    hits = (retrieved_labels[:, :k] == query_labels[:, None]).any(axis=1)
    return float(hits.mean())


def mean_average_precision(
    retrieved_labels: np.ndarray,
    query_labels: np.ndarray,
    k: int,
) -> float:
    """Mean Average Precision at k (mAP@k).

    For each query:
        AP@k = (1 / R) * sum_{i=1}^{k} P@i * rel(i)
    where R = number of relevant items in top-k, P@i = precision at rank i,
    rel(i) = 1 if rank-i result matches query label, else 0.

    Parameters
    ----------
    retrieved_labels:
        (Q, K_max) array of retrieved labels (best rank first).
    query_labels:
        (Q,) ground-truth labels.
    k:
        Cutoff rank.

    Returns
    -------
    float in [0, 1]
    """
    top_k = retrieved_labels[:, :k]
    aps = []
    for q in range(len(query_labels)):
        ql = query_labels[q]
        rels = (top_k[q] == ql).astype(float)
        n_rel = rels.sum()
        if n_rel == 0:
            aps.append(0.0)
            continue
        cumulative = np.cumsum(rels)
        ranks = np.arange(1, k + 1, dtype=float)
        ap = float((cumulative / ranks * rels).sum() / n_rel)
        aps.append(ap)
    return float(np.mean(aps))


def random_baseline(n_classes: int, k: int) -> float:
    """Top-k accuracy expected from a uniformly random retriever.

    P(at least one match in k draws with replacement) = 1 - ((n-1)/n)^k.
    """
    if n_classes <= 0:
        return 0.0
    return 1.0 - ((n_classes - 1) / n_classes) ** k
