"""
Patient correlation analysis utilities for H2 hypotheses.

Provides:
  - aggregate_by_patient:  mean-pool patch embeddings per patient -> 1 vector/patient
  - correlation_matrix:    N×N cosine similarity matrix
  - compute_umap:          2-D UMAP projection
  - plot_umap:             scatter plot coloured by label
  - plot_heatmap:          correlation heatmap ordered by group (exposes block-diagonal)
  - tumour_patch_counts:   count tumour patches per patient
  - decide_aggregation:    adaptive rule — aggregate or use individual patches

All plotting functions save PNG files and print the output path; they do not
show interactive windows (non-interactive Agg backend).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive; works without a display / on servers
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from umap import UMAP


# ---------------------------------------------------------------------------
# Embedding aggregation
# ---------------------------------------------------------------------------

def aggregate_by_patient(
    manifest: pd.DataFrame,
    embeddings: np.ndarray,
    group_col: str = "slide_id",
    method: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean-pool patch embeddings per patient (or per any group column) and
    re-normalize the result to unit length so cosine similarity == dot product.

    This is a generalized version of histoRAG.embed.aggregate_slide_embeddings
    that works with any grouping column (slide_id, patient_id, etc.).

    Args:
        manifest:    patch manifest DataFrame, rows aligned to embeddings.
        embeddings:  (N_patches, dim) float32; ideally already L2-normalized.
        group_col:   manifest column to group by. Default 'slide_id' (1 WSI = 1 patient).
        method:      aggregation method; only 'mean' is supported.

    Returns:
        (patient_embeddings, patient_ids)
          patient_embeddings: (N_patients, dim) float32, L2-normalized
          patient_ids:        (N_patients,) array of group values, same order as rows
    """
    if method != "mean":
        raise NotImplementedError(
            f"Aggregation method '{method}' is not implemented. Use 'mean'."
        )

    group_ids_all = manifest[group_col].values
    unique_ids = list(dict.fromkeys(group_ids_all))  # preserves first-occurrence order

    agg_rows = []
    for gid in unique_ids:
        mask = group_ids_all == gid
        mean_vec = embeddings[mask].mean(axis=0)
        agg_rows.append(mean_vec)

    agg = np.stack(agg_rows, axis=0).astype(np.float32)

    # re-normalize to unit length (mean-pooling breaks L2-normalization)
    norms = np.linalg.norm(agg, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    agg = agg / norms

    return agg, np.array(unique_ids)


# ---------------------------------------------------------------------------
# Similarity matrix
# ---------------------------------------------------------------------------

def correlation_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute the N×N cosine similarity matrix for a set of L2-normalized embeddings.

    Since embeddings are L2-normalized, cosine similarity equals the dot product.
    Values range from -1 (opposite directions) to 1 (identical directions).

    Args:
        embeddings: (N, dim) float32, L2-normalized

    Returns:
        (N, N) float32 cosine similarity matrix
    """
    return (embeddings @ embeddings.T).astype(np.float32)


# ---------------------------------------------------------------------------
# UMAP dimensionality reduction
# ---------------------------------------------------------------------------

def compute_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """
    Reduce high-dimensional embeddings to 2-D using UMAP.

    Args:
        embeddings:   (N, dim) float32 embeddings to project.
        n_neighbors:  size of the local neighbourhood used by UMAP.
                      Smaller = more local structure; larger = more global.
        min_dist:     minimum distance between points in 2-D.
                      Smaller = tighter clusters; larger = more spread.
        random_state: seed for reproducibility.

    Returns:
        (N, 2) float32 2-D coordinates
    """
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        verbose=False,
    )
    return reducer.fit_transform(embeddings).astype(np.float32)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_umap(
    coords: np.ndarray,
    labels: list | np.ndarray | pd.Series,
    out_path: str | Path,
    title: str = "UMAP",
) -> None:
    """
    Scatter plot of 2-D UMAP coordinates coloured by label.

    Args:
        coords:   (N, 2) coordinates from compute_umap.
        labels:   length-N sequence of string group labels.
        out_path: file path for the saved PNG.
        title:    plot title.
    """
    labels = np.asarray(labels, dtype=str)
    unique_labels = list(dict.fromkeys(labels))  # ordered by first occurrence
    palette = _label_palette(unique_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl in unique_labels:
        mask = labels == lbl
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=palette[lbl], label=lbl, s=20, alpha=0.8, linewidths=0,
        )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=2, fontsize=9, loc="best", framealpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[correlate] UMAP saved -> {out_path}")


def plot_heatmap(
    matrix: np.ndarray,
    group_labels: list | np.ndarray | pd.Series,
    out_path: str | Path,
    title: str = "Correlation Map",
    order_by_group: bool = True,
) -> None:
    """
    Plot an N×N cosine similarity heatmap.

    When order_by_group=True, rows and columns are sorted by group label so
    that members of the same group are adjacent.  Group names are shown at
    block midpoints on both axes.

    Colormap is data-driven (YlOrRd): light yellow = low similarity within
    the observed range, dark red = high similarity.  vmin is set just below
    the off-diagonal minimum so the full colour range is used.

    Args:
        matrix:         (N, N) cosine similarity matrix from correlation_matrix().
        group_labels:   length-N sequence of group names aligned to matrix rows/cols.
        out_path:       file path for the saved PNG.
        title:          plot title.
        order_by_group: sort rows/cols by group to expose block-diagonal. Default True.
    """
    group_labels = np.asarray(group_labels, dtype=str)

    if order_by_group:
        sort_idx = np.argsort(group_labels, kind="stable")
        matrix = matrix[np.ix_(sort_idx, sort_idx)]
        group_labels = group_labels[sort_idx]

    # discrete 5-bin colormap — each bin is a visually distinct hue
    _bounds = [0.00, 0.50, 0.70, 0.85, 0.95, 1.01]  # 1.01 captures exact 1.0
    _colors = ["#fff176", "#64b5f6", "#43a047", "#ef6c00", "#b71c1c"]
    #           light yellow  light blue   green      deep orange   dark red
    _labels = ["0.00 – 0.50", "0.51 – 0.70", "0.70 – 0.85", "0.85 – 0.95", "0.95+"]
    _cmap = ListedColormap(_colors)
    _norm = BoundaryNorm(_bounds, ncolors=len(_colors))

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(matrix, cmap=_cmap, norm=_norm, aspect="auto", interpolation="nearest")

    legend_handles = [
        Patch(facecolor=c, edgecolor="#888888", linewidth=0.6, label=l)
        for c, l in zip(_colors, _labels)
    ]
    ax.legend(
        handles=legend_handles,
        title="Cosine Similarity",
        title_fontsize=9,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        framealpha=0.9,
    )

    # white boundary lines between groups
    block_starts, block_ends, unique_groups = [], [], []
    start = 0
    for i in range(1, len(group_labels)):
        if group_labels[i] != group_labels[i - 1]:
            block_starts.append(start)
            block_ends.append(i - 1)
            unique_groups.append(group_labels[i - 1])
            ax.axhline(y=i - 0.5, color="white", linewidth=1.5)
            ax.axvline(x=i - 0.5, color="white", linewidth=1.5)
            start = i
    block_starts.append(start)
    block_ends.append(len(group_labels) - 1)
    unique_groups.append(group_labels[-1])

    # group name ticks at block midpoints on both axes
    midpoints = [(s + e) / 2 for s, e in zip(block_starts, block_ends)]
    ax.set_xticks(midpoints)
    ax.set_xticklabels(unique_groups, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(midpoints)
    ax.set_yticklabels(unique_groups, fontsize=10)

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Patients (grouped by site)", labelpad=8)
    ax.set_ylabel("Patients (grouped by site)", labelpad=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[correlate] Heatmap saved -> {out_path}")


# ---------------------------------------------------------------------------
# Adaptive aggregation helpers (H2 Q2)
# ---------------------------------------------------------------------------

def tumour_patch_counts(
    manifest: pd.DataFrame,
    tumour_labels: pd.Series,
) -> pd.Series:
    """
    Count the number of tumour patches per patient (slide_id).

    Args:
        manifest:      patch manifest with column 'slide_id'.
        tumour_labels: per-patch string labels aligned to manifest; 'tumour' or 'other'.

    Returns:
        pd.Series mapping slide_id -> count of tumour patches (sorted descending).
    """
    tumour_mask = tumour_labels == "tumour"
    return manifest.loc[tumour_mask.values, "slide_id"].value_counts()


def decide_aggregation(counts: pd.Series, threshold: int = 20) -> str:
    """
    Choose whether to aggregate tumour patches per patient or keep them individual.

    Rule:
      - If the MEDIAN tumour patch count across patients >= threshold -> aggregate
        (mean-pool per patient -> patient × patient correlation matrix)
      - Otherwise -> keep individual patches
        (patch × patch correlation matrix, rows/cols grouped by patient)

    Rationale: mean-pooling with very few patches (< ~10) dilutes the embedding
    signal and can make different patients look artificially similar.

    Args:
        counts:    per-patient tumour patch counts from tumour_patch_counts().
        threshold: median count threshold; default 20.

    Returns:
        'patient' (aggregate) or 'patch' (keep individual)
    """
    median_count = float(counts.median())
    decision = "patient" if median_count >= threshold else "patch"
    print(
        f"[correlate] Tumour patches per patient: "
        f"min={counts.min()}, median={median_count:.1f}, max={counts.max()} "
        f"-> aggregation mode: '{decision}'"
    )
    return decision


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _label_palette(labels: list[str]) -> dict[str, str]:
    """Map a list of unique labels to distinct Tableau colours."""
    colours = list(mcolors.TABLEAU_COLORS.values())
    return {lbl: colours[i % len(colours)] for i, lbl in enumerate(labels)}
