"""
H1 clustering utilities for unsupervised tumour grouping.

Approach:
  - Cluster frozen patch embeddings with K-means (no labels used during clustering).
  - Map each cluster to tumour/other by majority vote against ground-truth .geojson labels.
  - Measure how well the encoder's embedding space naturally separates tumour tissue.

Provides:
  - cluster_embeddings:        K-means on patch embeddings -> per-patch cluster ids
  - match_clusters_to_labels:  majority-vote map from cluster ids to tumour/other (0/1)
  - classification_metrics:    Accuracy, Precision, Recall, F1 (vs ground truth)
  - cluster_summary:           per-cluster GeoJSON tumour percentage + assigned label
  - grouped_metrics:           per-site or per-WSI post-hoc evaluation
  - explain_dimensions:        STUB — XAI strategy deferred until clustering results available
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int = 2,
    random_state: int = 42,
    n_init: int = 10,
) -> np.ndarray:
    """
    Cluster patch embeddings with K-means (unsupervised — no labels used).

    Args:
        embeddings:   (N, dim) float32 patch embeddings.
        n_clusters:   number of clusters. Use 2 for exp01 (dominant-axis test),
                      8 for exp02 (over-cluster to recover buried tumour signal).
        random_state: seed for reproducibility.
        n_init:       number of times K-means algorithm runs with different centroid seeds.
                      Higher = more stable but slower. Default 10; use 3-5 for large datasets.

    Returns:
        (N,) int array of cluster ids in range [0, n_clusters).
    """
    km = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    return km.fit_predict(embeddings).astype(np.int32)


def fit_kmeans(
    sample_embeddings: np.ndarray,
    n_clusters: int = 2,
    random_state: int = 42,
    n_init: int = 10,
):
    """
    Fit KMeans on a subsample and return the fitted model.

    Use km.predict(chunk) to assign cluster IDs to arbitrary batches without
    holding all embeddings in memory at once.
    """
    from sklearn.cluster import KMeans as _KMeans
    km = _KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    km.fit(sample_embeddings)
    return km


def match_clusters_to_labels(
    cluster_ids: np.ndarray,
    true_labels: np.ndarray,
) -> np.ndarray:
    """
    Map cluster ids to binary tumour/other predictions via per-cluster majority vote.

    Vectorized for efficiency: pre-compute majority label for each cluster,
    then apply in single operation.

    Args:
        cluster_ids:  (N,) int cluster assignments from cluster_embeddings().
        true_labels:  (N,) int ground-truth labels — 1 = tumour, 0 = other.

    Returns:
        (N,) int predicted binary labels — 1 = tumour, 0 = other.
    """
    cluster_ids  = np.asarray(cluster_ids,  dtype=np.int32)
    true_labels  = np.asarray(true_labels,  dtype=np.int32)
    predicted    = np.zeros_like(cluster_ids, dtype=np.int32)

    # Pre-compute majority label per cluster for efficiency
    unique_clusters = np.unique(cluster_ids)
    cluster_to_label = {}
    for cid in unique_clusters:
        mask = cluster_ids == cid
        majority_vote = int(true_labels[mask].mean() >= 0.5)
        cluster_to_label[cid] = majority_vote
    
    # Vectorized assignment
    for cid, label in cluster_to_label.items():
        predicted[cluster_ids == cid] = label

    return predicted


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute post-hoc tumour/other metrics after cluster-to-label assignment.

    Fast vectorized implementation. Positive class is 1 (tumour); negative is 0 (other).

    Args:
        y_true: (N,) int ground-truth labels — 1 = tumour, 0 = other
        y_pred: (N,) int predicted labels

    Returns:
        dict with accuracy, precision, recall, f1, and tumour_prevalence.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tumour_prevalence": 0.0,
        }
    
    correct = y_true == y_pred
    accuracy = float(correct.mean())
    
    # True positives, false positives, false negatives
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    tumour_prevalence = float((y_true == 1).mean())
    
    return {
        "accuracy":          accuracy,
        "precision":         precision,
        "recall":            recall,
        "f1":                f1,
        "tumour_prevalence": tumour_prevalence,
    }


def cluster_summary(cluster_ids: np.ndarray, true_labels: np.ndarray) -> pd.DataFrame:
    """
    Summarise each unsupervised cluster using GeoJSON labels after clustering.

    GeoJSON tumour % = percentage of patches in the cluster whose center falls
    inside a tumour polygon. This is also the majority-vote signal used to name
    the cluster as tumour or other.
    """
    cluster_ids = np.asarray(cluster_ids, dtype=np.int32)
    true_labels = np.asarray(true_labels, dtype=np.int32)

    rows = []
    for cid in sorted(np.unique(cluster_ids).tolist()):
        mask = cluster_ids == cid
        tumour_pct = float(true_labels[mask].mean() * 100)
        rows.append({
            "cluster": cid,
            "patches": int(mask.sum()),
            "geojson_tumour_pct": round(tumour_pct, 2),
            "assigned_label": "tumour" if tumour_pct >= 50 else "other",
        })

    return pd.DataFrame(rows)


def grouped_metrics(
    manifest: pd.DataFrame,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    group_col: str,
) -> pd.DataFrame:
    """
    Compute post-hoc evaluation metrics per site or per WSI.

    Args:
        manifest: DataFrame aligned to true_labels and predicted_labels.
        true_labels: 1=tumour, 0=other from GeoJSON.
        predicted_labels: 1=tumour, 0=other from cluster majority vote.
        group_col: "site" for per-tissue results, "slide_id" for per-WSI.
    """
    true_labels = np.asarray(true_labels, dtype=np.int32)
    predicted_labels = np.asarray(predicted_labels, dtype=np.int32)

    rows = []
    for group_value, group in manifest.groupby(group_col, sort=True):
        idx = manifest.index.get_indexer(group.index)
        metrics = classification_metrics(true_labels[idx], predicted_labels[idx])

        if group_col == "slide_id":
            row = {
                "slide_id": group_value,
                "site": str(group["site"].iloc[0]) if "site" in group else "",
                "patches": int(len(group)),
                "tumour_patches": int(true_labels[idx].sum()),
                "predicted_tumour_patches": int(predicted_labels[idx].sum()),
                "accuracy": round(metrics["accuracy"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1"], 4),
            }
        else:
            row = {
                group_col: group_value,
                "slides": int(group["slide_id"].nunique()),
                "patches": int(len(group)),
                "tumour_pct": round(metrics["tumour_prevalence"] * 100, 2),
                "accuracy": round(metrics["accuracy"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1"], 4),
            }

        rows.append(row)

    return pd.DataFrame(rows)


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
