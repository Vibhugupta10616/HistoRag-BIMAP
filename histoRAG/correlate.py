"""
Patient correlation analysis utilities for H2 hypotheses.

Provides:
  - aggregate_by_patient:       mean-pool patch embeddings per patient -> 1 vector/patient
  - select_centroid_patches:    select N patches closest to patient tumour centroid
  - correlation_matrix:         N×N cosine similarity matrix
  - compute_umap:               2-D UMAP projection
  - plot_umap:                  scatter plot coloured by label
  - plot_heatmap:               correlation heatmap ordered by group (exposes block-diagonal)
  - plot_similarity_distribution: within-site vs cross-site KDE plot
  - tumour_patch_counts:        count tumour patches per patient
  - decide_aggregation:         adaptive rule — aggregate or use individual patches

All plotting functions save PNG files and print the output path; they do not
show interactive windows (non-interactive Agg backend).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive; works without a display / on servers
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
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


def select_centroid_patches(embeddings: np.ndarray, n: int) -> np.ndarray:
    """
    Select the N patches closest (by L2 distance) to the mean of the embedding cloud.

    The centroid is used only for ranking — it is never stored or returned.
    If the patient has <= n patches, all indices are returned unchanged.

    Args:
        embeddings: (M, dim) float32 tumour patches for one patient.
        n:          number of patches to keep.

    Returns:
        (min(n, M),) integer indices of the selected patches.
    """
    if len(embeddings) <= n:
        return np.arange(len(embeddings))
    centroid = embeddings.mean(axis=0)
    dists = np.linalg.norm(embeddings - centroid, axis=1)
    return np.argsort(dists)[:n]


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
    n_components: int = 2,
) -> np.ndarray:
    """
    Reduce high-dimensional embeddings to n_components-D using UMAP.

    Args:
        embeddings:   (N, dim) float32 embeddings to project.
        n_neighbors:  size of the local neighbourhood used by UMAP.
        min_dist:     minimum distance between points in reduced space.
        random_state: seed for reproducibility.
        n_components: output dimensionality (2 or 3).

    Returns:
        (N, n_components) float32 coordinates
    """
    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        n_components=n_components,
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


def plot_umap_3d(
    coords: np.ndarray,
    labels: list | np.ndarray | pd.Series,
    out_path: str | Path,
    title: str = "UMAP (3D)",
) -> None:
    """
    3-D scatter plot of UMAP coordinates coloured by label.

    Args:
        coords:   (N, 3) coordinates from compute_umap(n_components=3).
        labels:   length-N sequence of string group labels.
        out_path: file path for the saved PNG.
        title:    plot title.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

    labels = np.asarray(labels, dtype=str)
    unique_labels = list(dict.fromkeys(labels))
    palette = _label_palette(unique_labels)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for lbl in unique_labels:
        mask = labels == lbl
        ax.scatter(
            coords[mask, 0], coords[mask, 1], coords[mask, 2],
            c=palette[lbl], label=lbl, s=12, alpha=0.7, linewidths=0,
        )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UMAP-1", fontsize=9)
    ax.set_ylabel("UMAP-2", fontsize=9)
    ax.set_zlabel("UMAP-3", fontsize=9)
    ax.legend(markerscale=2, fontsize=9, loc="upper left", framealpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[correlate] UMAP 3D saved -> {out_path}")


def plot_heatmap(
    matrix: np.ndarray,
    group_labels: list | np.ndarray | pd.Series,
    out_path: str | Path,
    title: str = "Correlation Map",
    order_by_group: bool = True,
    xlabel: str = "Patients (grouped by site)",
    ylabel: str = "Patients (grouped by site)",
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

    n_full = len(matrix)

    # compute block boundaries on the FULL matrix (used for lines + tick positions)
    block_starts, block_ends, unique_groups = [], [], []
    start = 0
    for i in range(1, n_full):
        if group_labels[i] != group_labels[i - 1]:
            block_starts.append(start)
            block_ends.append(i - 1)
            unique_groups.append(group_labels[i - 1])
            start = i
    block_starts.append(start)
    block_ends.append(n_full - 1)
    unique_groups.append(group_labels[-1])

    # downsample matrix for display if too large (cap at 2000×2000 pixels)
    max_display = 2000
    if n_full > max_display:
        step = n_full // max_display
        display_matrix = matrix[::step, ::step]
        scale = 1.0 / step
        print(f"[correlate] Heatmap downsampled {n_full}->{len(display_matrix)} for rendering (step={step})")
    else:
        display_matrix = matrix
        scale = 1.0

    # data-driven colour range — exclude self-similarity diagonal (always 1.0)
    off_diag = matrix[~np.eye(n_full, dtype=bool)]
    vmin = max(0.0, float(off_diag.min()) - 0.02)
    vmax = 1.0

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(display_matrix, cmap="YlOrRd", vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Cosine Similarity", shrink=0.8)

    # boundary lines and tick positions scaled to display coordinates
    for s, e in zip(block_starts[1:], block_ends[:-1]):
        boundary = (e + 0.5) * scale
        ax.axhline(y=boundary, color="white", linewidth=1.5)
        ax.axvline(x=boundary, color="white", linewidth=1.5)

    midpoints = [((s + e) / 2) * scale for s, e in zip(block_starts, block_ends)]
    ax.set_xticks(midpoints)
    ax.set_xticklabels(unique_groups, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(midpoints)
    ax.set_yticklabels(unique_groups, fontsize=10)

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"[correlate] Heatmap saved -> {out_path}")


def plot_umap_3d_interactive(
    coords: np.ndarray,
    labels: list | np.ndarray | pd.Series,
    out_path: str | Path,
    title: str = "UMAP (3D interactive)",
) -> None:
    """
    Interactive 3-D UMAP scatter saved as a self-contained HTML file.

    Open the HTML in any browser — you can rotate, zoom, and hover over
    points to see their site label.

    Args:
        coords:   (N, 3) coordinates from compute_umap(n_components=3).
        labels:   length-N sequence of string group labels.
        out_path: file path for the saved HTML (e.g. 'umap_3d.html').
        title:    plot title shown inside the interactive figure.
    """
    import plotly.graph_objects as go

    labels = np.asarray(labels, dtype=str)
    unique_labels = list(dict.fromkeys(labels))
    palette = _label_palette(unique_labels)

    traces = []
    for lbl in unique_labels:
        mask = labels == lbl
        traces.append(go.Scatter3d(
            x=coords[mask, 0],
            y=coords[mask, 1],
            z=coords[mask, 2],
            mode="markers",
            name=lbl,
            marker=dict(size=3, color=palette[lbl], opacity=0.7),
            hovertemplate=f"<b>{lbl}</b><extra></extra>",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="UMAP-3",
        ),
        legend=dict(title="Site", itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    out_path = Path(out_path)
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"[correlate] Interactive 3D UMAP saved -> {out_path}")


def compute_similarity_pairs(
    embeddings: np.ndarray,
    site_labels: np.ndarray,
    n_sample_per_site: int = 1000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample patches per site and return (within_sims, cross_sims) arrays.

    Returned arrays can be cached and passed directly to plot_similarity_distribution
    to regenerate the plot without re-running the heavy embedding scan.
    """
    rng = np.random.default_rng(random_state)
    site_labels = np.asarray(site_labels, dtype=str)
    unique_sites = np.unique(site_labels)

    sample_idx_parts, sample_site_list = [], []
    for site in unique_sites:
        idx = np.where(site_labels == site)[0]
        if len(idx) > n_sample_per_site:
            idx = rng.choice(idx, n_sample_per_site, replace=False)
        sample_idx_parts.append(idx)
        sample_site_list.extend([site] * len(idx))

    sample_idx   = np.concatenate(sample_idx_parts)
    sample_emb   = embeddings[sample_idx].astype(np.float32)
    sample_sites = np.array(sample_site_list, dtype=str)

    sim_mat = (sample_emb @ sample_emb.T).astype(np.float32)
    n = len(sample_emb)
    rows, cols = np.triu_indices(n, k=1)
    within_mask = sample_sites[rows] == sample_sites[cols]
    within_sims = sim_mat[rows[within_mask],  cols[within_mask]]
    cross_sims  = sim_mat[rows[~within_mask], cols[~within_mask]]
    return within_sims, cross_sims


def plot_similarity_distribution(
    within_t: np.ndarray,
    cross_t: np.ndarray,
    out_path: str | Path,
    title: str = "Patch Similarity Distribution",
    within_nt: np.ndarray | None = None,
    cross_nt: np.ndarray | None = None,
) -> None:
    """
    KDE distribution of within-site vs cross-site cosine similarities.

    When within_nt / cross_nt are provided, draws 4 curves (tumour + non-tumour)
    on the same axis for direct comparison; otherwise draws 2 curves (tumour only).
    Takes pre-computed arrays from compute_similarity_pairs so the plot can be
    regenerated without re-running the expensive embedding scan.
    """
    four_curve = within_nt is not None and cross_nt is not None

    all_arrays = [within_t, cross_t]
    if four_curve:
        all_arrays += [within_nt, cross_nt]
    all_sims = np.concatenate(all_arrays)
    x_lo = max(0.0, float(all_sims.min()) - 0.02)
    x_hi = min(1.0, float(all_sims.max()) + 0.02)
    bins = np.linspace(x_lo, x_hi, 60)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    series = [
        (within_t, "#e53935", "-",  "Tumour – within-site"),
        (cross_t,  "#e53935", "--", "Tumour – cross-site"),
    ]
    if four_curve:
        series += [
            (within_nt, "#1e88e5", "-",  "Non-tumour – within-site"),
            (cross_nt,  "#1e88e5", "--", "Non-tumour – cross-site"),
        ]

    fig, ax = plt.subplots(figsize=(10, 5))

    computed_modes = []
    for sims, color, ls, label in series:
        counts, _ = np.histogram(sims, bins=bins)
        mode = float(bin_centers[np.argmax(counts)])
        computed_modes.append(mode)
        ax.plot(bin_centers, counts, label=f"{label}  (mode={mode:.3f})",
                color=color, linestyle=ls, linewidth=2)
        ax.fill_between(bin_centers, counts, alpha=0.10, color=color)
        ax.axvline(mode, color=color, linestyle=ls, linewidth=1.0, alpha=0.6)

    gap_t = computed_modes[0] - computed_modes[1]
    annotation = f"Tumour gap = {gap_t:+.4f}"
    if four_curve:
        annotation += f"\nNon-tumour gap = {computed_modes[2] - computed_modes[3]:+.4f}"
    ax.annotate(
        annotation,
        xy=(0.03, 0.95), xycoords="axes fraction",
        ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#888", alpha=0.8),
    )

    ax.set_xlabel(
        "Cosine similarity between two patches\n"
        "(0 = completely different directions,  1 = identical)",
        fontsize=10,
    )
    ax.set_ylabel("Number of patch pairs", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[correlate] Similarity distribution saved -> {out_path}")


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
