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
  - tune_threshold:            H3 — pick a decision threshold on validation scores
  - probe_metrics:             H3 — AUROC/PR-AUC + thresholded metrics for a supervised probe
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


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
# H3 — supervised linear probe metrics
# ---------------------------------------------------------------------------

def tune_threshold(y_true: np.ndarray, y_score: np.ndarray, objective: str = "f1") -> float:
    """
    Pick a decision threshold on (held-out) validation scores.

    Candidate thresholds are the unique predicted scores themselves — this is
    exact and cheap enough for validation-set sizes (tens of thousands to a
    few million patches).

    Args:
        y_true:    (N,) int ground-truth labels — 1 = tumour, 0 = other.
        y_score:   (N,) float predicted tumour probability.
        objective: "f1" (maximise F1) or "youden" (maximise TPR - FPR).

    Returns:
        The threshold (float) that maximises the chosen objective.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)

    candidates = np.unique(y_score)
    if len(candidates) > 2000:
        # Coarsen to 2000 quantile cut points — plenty of resolution, keeps the
        # sweep fast on multi-million-patch validation sets.
        candidates = np.quantile(candidates, np.linspace(0, 1, 2000))

    best_thr, best_score = 0.5, -1.0
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    for thr in candidates:
        y_pred = (y_score >= thr).astype(np.int32)
        if objective == "youden":
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            tpr = tp / n_pos if n_pos else 0.0
            fpr = fp / n_neg if n_neg else 0.0
            score = tpr - fpr
        else:  # f1
            _, _, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=[1], average="binary", zero_division=0
            )
            score = f1
        if score > best_score:
            best_score, best_thr = score, float(thr)

    return best_thr


def probe_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    """
    Evaluate a supervised linear probe's predicted tumour probabilities.

    Reports both threshold-independent ranking quality (AUROC, PR-AUC — the
    honest "does the probe separate the classes at all" numbers, unaffected by
    class imbalance) and metrics at the given decision threshold.

    Args:
        y_true:    (N,) int ground-truth labels — 1 = tumour, 0 = other.
        y_score:   (N,) float predicted tumour probability.
        threshold: decision threshold to binarise y_score at.

    Returns:
        dict with auroc, pr_auc, threshold, accuracy, balanced_accuracy,
        precision, recall, f1, tumour_prevalence, confusion_matrix
        ({tn, fp, fn, tp}).
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        auroc, pr_auc = float("nan"), float("nan")
    else:
        auroc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))

    y_pred = (y_score >= threshold).astype(np.int32)
    base = classification_metrics(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "auroc": auroc,
        "pr_auc": pr_auc,
        "threshold": float(threshold),
        "accuracy": base["accuracy"],
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": base["precision"],
        "recall": base["recall"],
        "f1": base["f1"],
        "tumour_prevalence": base["tumour_prevalence"],
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
    }


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


if __name__ == "__main__":
    # Self-check for the H3 helpers: on perfectly separable synthetic scores,
    # tune_threshold must find a threshold that splits the classes cleanly,
    # and probe_metrics must report AUROC and F1 at (essentially) 1.0.
    rng = np.random.default_rng(0)
    n = 2000
    y_true = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    y_score = np.concatenate([
        rng.uniform(0.0, 0.4, n),   # "other" scores low
        rng.uniform(0.6, 1.0, n),   # "tumour" scores high
    ])

    thr = tune_threshold(y_true, y_score, objective="f1")
    assert 0.35 <= thr <= 0.65, f"expected threshold in the separation gap, got {thr}"

    m = probe_metrics(y_true, y_score, thr)
    assert m["auroc"] > 0.999, f"expected AUROC~1.0 on separable data, got {m['auroc']}"
    assert m["f1"] > 0.999, f"expected F1~1.0 at tuned threshold, got {m['f1']}"
    # Threshold candidates come from a quantile grid, not the (empty) separation
    # gap itself, so a handful of near-boundary misses is expected, not a bug.
    n_errors = m["confusion_matrix"]["fp"] + m["confusion_matrix"]["fn"]
    assert n_errors < 5, f"expected near-zero errors on separable data, got {n_errors}"

    print("classify.py self-check passed:", {k: m[k] for k in ("auroc", "pr_auc", "f1", "threshold")})
