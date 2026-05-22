"""
H1 clustering utilities for unsupervised tumour grouping.

Approach:
  - Cluster frozen patch embeddings with K-means (no labels used during clustering).
  - Map each cluster to tumour/other by majority vote against ground-truth .geojson labels.
  - Measure how well the encoder's embedding space naturally separates tumour tissue.

Provides:
  - cluster_embeddings:        K-means on patch embeddings -> per-patch cluster ids
  - match_clusters_to_labels:  majority-vote map from cluster ids to tumour/other (0/1)
  - classification_metrics:    Accuracy, Precision, Recall (vs ground-truth labels)
  - explain_dimensions:        STUB — XAI strategy deferred until clustering results available
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """
    Cluster patch embeddings with K-means (unsupervised — no labels used).

    Args:
        embeddings:   (N, dim) float32 patch embeddings.
        n_clusters:   number of clusters. Use 2 for exp01 (dominant-axis test),
                      8 for exp02 (over-cluster to recover buried tumour signal).
        random_state: seed for reproducibility.

    Returns:
        (N,) int array of cluster ids in range [0, n_clusters).
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    return km.fit_predict(embeddings).astype(np.int32)


def match_clusters_to_labels(
    cluster_ids: np.ndarray,
    true_labels: np.ndarray,
) -> np.ndarray:
    """
    Map cluster ids to binary tumour/other predictions via per-cluster majority vote.

    For each cluster, count how many ground-truth labels are tumour (1) vs other (0).
    The majority label for that cluster becomes its predicted class.  Works for any
    number of clusters — with k=2 each cluster flips to one class; with k=8 several
    clusters may map to the same class (all evaluated together via metrics).

    Args:
        cluster_ids:  (N,) int cluster assignments from cluster_embeddings().
        true_labels:  (N,) int ground-truth labels — 1 = tumour, 0 = other.

    Returns:
        (N,) int predicted binary labels — 1 = tumour, 0 = other.
    """
    cluster_ids  = np.asarray(cluster_ids,  dtype=np.int32)
    true_labels  = np.asarray(true_labels,  dtype=np.int32)
    predicted    = np.zeros_like(cluster_ids, dtype=np.int32)

    for cid in np.unique(cluster_ids):
        mask          = cluster_ids == cid
        majority_vote = int(np.round(true_labels[mask].mean()))  # 1 if >50% tumour
        predicted[mask] = majority_vote

    return predicted


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute Accuracy, Precision, and Recall for binary classification.

    Positive class is 1 (tumour); negative class is 0 (other).

    Args:
        y_true: (N,) int ground-truth labels — 1 = tumour, 0 = other
        y_pred: (N,) int predicted labels

    Returns:
        dict with keys 'accuracy', 'precision', 'recall' (all float, 0-1)
    """
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# XAI — STUB (deferred until clustering results are available)
# ---------------------------------------------------------------------------

def explain_dimensions(
    cluster_ids: np.ndarray,
    embeddings: np.ndarray,
    true_labels: np.ndarray,
) -> dict:
    """
    Identify which embedding dimensions most separate tumour from other patches.

    [STUB] XAI strategy deferred until clustering results are available.

    Planned approach (centroid difference):
      - Compute mean embedding of all tumour-assigned patches (centroid_tumour).
      - Compute mean embedding of all other-assigned patches (centroid_other).
      - Rank dimensions by |centroid_tumour - centroid_other|.
      - Top dimensions are what separates tumour in this encoder's space.

    Args:
        cluster_ids:  (N,) cluster assignments from cluster_embeddings().
        embeddings:   (N, dim) float32 patch embeddings.
        true_labels:  (N,) int ground-truth labels for reference.

    Returns:
        dict mapping dimension index (int) -> importance score (float)

    Raises:
        NotImplementedError: always, until implemented after results review.
    """
    raise NotImplementedError(
        "XAI not yet implemented — strategy to be decided after clustering results. "
        "Planned: centroid-difference ranking (|tumour_centroid - other_centroid| per dim)."
    )
