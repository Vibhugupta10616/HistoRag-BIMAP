"""
H1 clustering utilities for unsupervised tumour grouping.

Approach:
  - Cluster frozen patch embeddings with K-means (no labels used during clustering).
  - Map each cluster to tumour/other by majority vote against ground-truth .geojson labels.
  - Measure how well the encoder's embedding space naturally separates tumour tissue.

Provides:
  - match_clusters_to_labels:  majority-vote map from cluster ids to tumour/other (0/1)
  - classification_metrics:    Accuracy, Precision, Recall, F1 (vs ground truth)
  - cluster_summary:           per-cluster GeoJSON tumour percentage + assigned label
  - grouped_metrics:           per-site or per-WSI post-hoc evaluation
  - explain_dimensions:        STUB — XAI strategy deferred until clustering results available
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

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
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    km.fit(sample_embeddings)
    return km


def fit_minibatch_kmeans(
    embeddings: np.ndarray,
    n_clusters: int = 2,
    random_state: int = 42,
    batch_size: int = 10_000,
    n_init: int = 3,
    max_iter: int = 300,
):
    """
    Fit MiniBatchKMeans on the full embedding array and return the fitted model.

    Processes the full dataset in mini-batches — near-identical cluster quality
    to standard KMeans but substantially faster and with lower peak RAM.
    Suitable for fitting on all patches (8M+) on an HPC node.

    Args:
        embeddings:   (N, dim) float32 — the FULL patch embedding array.
        n_clusters:   number of clusters (2 for exp01, 8 for exp02).
        random_state: seed for reproducibility.
        batch_size:   patches per mini-batch update. Default 10,000.
        n_init:       number of initializations. Default 3.
        max_iter:     max passes over the data. Default 300.

    Returns:
        Fitted MiniBatchKMeans model. Call .predict(embeddings) to assign IDs.
    """
    mbk = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
        n_init=n_init,
        max_iter=max_iter,
    )
    mbk.fit(embeddings)
    return mbk


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

    # Use global tumour prevalence as threshold — a cluster is "tumour" if its
    # tumour fraction exceeds the overall rate. This works even when tumour is a
    # minority class (< 50% globally), where a fixed 0.5 threshold would label
    # every cluster "other" and produce precision=recall=0.
    global_tumour_rate = true_labels.mean()
    for cid in np.unique(cluster_ids):
        mask = cluster_ids == cid
        predicted[mask] = int(true_labels[mask].mean() > global_tumour_rate)

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
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "tumour_prevalence": 0.0}

    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average="binary", zero_division=0
    )
    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "precision":         float(p),
        "recall":            float(r),
        "f1":                float(f),
        "tumour_prevalence": float((y_true == 1).mean()),
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

    global_tumour_rate = true_labels.mean() * 100
    rows = []
    for cid in np.unique(cluster_ids):
        mask = cluster_ids == cid
        tumour_pct = float(true_labels[mask].mean() * 100)
        rows.append({
            "cluster_id":    cid,
            "n_patches":     int(mask.sum()),
            "gt_tumour_pct": round(tumour_pct, 2),
            "assigned_label": "tumour" if tumour_pct > global_tumour_rate else "other",
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

    by_wsi = group_col == "slide_id"
    rows = []
    for group_value, group in manifest.groupby(group_col, sort=True):
        idx = manifest.index.get_indexer(group.index)
        m = classification_metrics(true_labels[idx], predicted_labels[idx])
        metric_cols = {k: round(m[k], 4) for k in ("accuracy", "precision", "recall", "f1")}
        if by_wsi:
            row = {
                "slide_id": group_value,
                "site": str(group["site"].iloc[0]) if "site" in group else "",
                "patches": int(len(group)),
                "tumour_patches": int(true_labels[idx].sum()),
                "predicted_tumour_patches": int(predicted_labels[idx].sum()),
                **metric_cols,
            }
        else:
            row = {
                group_col: group_value,
                "slides": int(group["slide_id"].nunique()),
                "patches": int(len(group)),
                "tumour_pct": round(m["tumour_prevalence"] * 100, 2),
                **metric_cols,
            }
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# XAI — STUB (deferred until clustering results are available)
# ---------------------------------------------------------------------------

def explain_dimensions(
    embeddings: np.ndarray,
    true_labels: np.ndarray,
    top_k: int = 20,
) -> dict:
    """
    Identify which embedding dimensions most separate tumour from other patches.

    Uses centroid-difference ranking: for each dimension, compute the absolute
    difference between the tumour-patch mean and the other-patch mean, normalised
    by the pooled standard deviation (Cohen's d effect size). Dimensions with the
    largest effect size are the most tumour-discriminative axes in this encoder.

    Args:
        embeddings:  (N, dim) float32 — patch embeddings (full dataset).
        true_labels: (N,) int — ground-truth labels: 1 = tumour, 0 = other.
        top_k:       number of top dimensions to return.

    Returns:
        dict with keys:
          top_dims         — list[int] of top_k dimension indices (ranked)
          effect_sizes     — list[float] Cohen's d per top dimension
          tumour_centroid  — list[float] mean embedding of tumour patches (all dims)
          other_centroid   — list[float] mean embedding of other patches (all dims)
          n_tumour         — int number of tumour patches
          n_other          — int number of other patches
    """
    embeddings  = np.asarray(embeddings,  dtype=np.float32)
    true_labels = np.asarray(true_labels, dtype=np.int32)

    tumour_mask = true_labels == 1

    emb_tumour = embeddings[tumour_mask]
    n_t   = len(emb_tumour)
    mu_t  = emb_tumour.mean(axis=0)
    var_t = emb_tumour.var(axis=0)
    del emb_tumour   # free before allocating other subset

    emb_other = embeddings[~tumour_mask]
    n_o   = len(emb_other)
    mu_o  = emb_other.mean(axis=0)
    var_o = emb_other.var(axis=0)
    del emb_other

    # Pooled std — adds small epsilon to avoid division by zero
    pooled_std = np.sqrt((var_t * n_t + var_o * n_o) / (n_t + n_o)) + 1e-8

    effect_size = np.abs(mu_t - mu_o) / pooled_std   # Cohen's d per dimension

    top_idx = np.argsort(effect_size)[::-1][:top_k]

    return {
        "top_dims":        top_idx.tolist(),
        "effect_sizes":    effect_size[top_idx].tolist(),
        "tumour_centroid": mu_t.tolist(),
        "other_centroid":  mu_o.tolist(),
        "n_tumour":        int(n_t),
        "n_other":         int(n_o),
    }
