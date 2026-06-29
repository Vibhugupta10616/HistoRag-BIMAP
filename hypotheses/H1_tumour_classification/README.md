# H1 — Unsupervised Tumour Grouping + XAI

## Hypothesis

Do frozen encoder embeddings **naturally group tumour patches together** — without ever
seeing labels?

K-means clustering is applied to frozen patch embeddings with **no label information**.
The QuPath `.geojson` annotations are used **only at evaluation time** to compare cluster
assignments against the real tumour regions.

---

## Method

1. Load frozen patch embeddings `(N_patches, dim)`.
2. Cluster with K-means (no labels used).
3. Map each cluster to tumour/other by **majority vote** against `.geojson` ground truth.
4. Compute **Accuracy, Precision, Recall** vs ground truth.

---

## Patch labelling (ground truth, evaluation only)

| Label | Definition |
|---|---|
| `tumour` | Patch center inside a QuPath `.geojson` polygon for that slide |
| `other` | Patch center outside all polygons (unannotated = non-tumour) |

See `histoRAG/labels.py → tumour_labels_from_geojson`.

---

## Three experiments

| Folder | Method | Question answered |
|---|---|---|
| `exp01_kmeans_k2` | K-means k=2 | Is tumour the **dominant** axis of variation? |
| `exp02_overcluster_assign` | K-means k=8 | Is tumour signal **present at all**, even if not dominant? |
| `exp03_hdbscan_clustering` | HDBSCAN + IncrementalPCA | What is the natural cluster structure? How does it align with tumour labels? |

**Interpretation matrix:**

| exp01 (k=2) | exp02 (k=8) | Conclusion |
|---|---|---|
| Pass | Pass | Tumour is the dominant signal — strong encoder |
| Fail | Pass | Tumour signal exists but is buried — weak encoder |
| Fail | Fail | Encoder does not encode tumour-discriminative features |

Why k=8 helps: if stain colour or tissue type dominates the top split, k=2 fails. With
k=8, tumour patches can still form their own sub-cluster *inside* one of those larger
groups, giving tumour a chance to be recovered even in a weaker encoder.

---

## Results (full dataset, HPC)

**Dataset**: 8.21M patches (CLIP/CONCH) · 2.07M patches (UNI) across 4 tissue sites.
**Tumour prevalence**: ~5% — baseline precision if a model labels everything as tumour.

### Overall metrics

| Encoder | Exp | Precision | vs baseline | Recall | F1 | Verdict |
|---|---|---|---|---|---|---|
| CLIP | k=2 | 0.054 | 1.06× | 0.998 | 0.103 | ❌ fail |
| CONCH | k=2 | 0.115 | 2.3× | 0.964 | 0.206 | ⚠️ weak |
| UNI | k=2 | 0.108 | 2.2× | 0.979 | 0.194 | ⚠️ weak |
| CLIP | k=8 | 0.090 | 1.75× | 0.804 | 0.162 | ⚠️ weak |
| CONCH | k=8 | 0.127 | 2.5× | 0.801 | 0.220 | ✅ present |
| **UNI** | **k=8** | **0.154** | **3.1×** | **0.818** | **0.259** | ✅ **best** |

### Per-tissue breakdown (k=8 precision)

| Tissue | CLIP | CONCH | UNI |
|---|---|---|---|
| Hypopharynx | 0.089 | 0.117 | **0.141** |
| Larynx | 0.075 | 0.103 | **0.129** |
| Oral Cavity | 0.095 | 0.130 | **0.165** |
| Oropharynx | 0.093 | 0.136 | **0.160** |

UNI is the strongest encoder across every tissue site.

### Conclusion

**None of the encoders have tumour as their dominant axis of variation** (exp01 fails for
all). The dominant split in embedding space reflects stain colour and tissue-type variation,
not tumour vs. normal tissue.

**Tumour signal is buried but present** in all encoders (exp02 shows improvement for all):
- **CLIP**: weakest signal — precision barely above baseline even at k=8
- **CONCH**: moderate buried signal — 2.5× baseline precision at k=8
- **UNI**: strongest buried signal — 3.1× baseline precision at k=8, best F1 (0.259)

**Encoder ranking for tumour discrimination**: UNI > CONCH >> CLIP

This matches the `exp01 fail, exp02 pass` pattern = *tumour signal present but buried*.

---

## Exp03 — HDBSCAN Results

**Setup:** 20-component IncrementalPCA → HDBSCAN (min\_cluster\_size=500) on 480K patches
(120K per site cap). ARI/NMI measure alignment with tumour ground truth.

| Encoder | PCA variance | Clusters found | Noise % | Best cluster purity | Precision | Recall | F1 | ARI | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| CONCH | 85.3% (20 dims) | 3 | 64.5% | 3.57% (< 5% baseline) | 0.036 | 0.252 | 0.063 | -0.013 | ❌ fail |
| UNI | 33.5% (20 dims) | 10 | 84.4% | 20.9% (4.2× baseline) | 0.209 | 0.007 | 0.014 | -0.035 | ⚠️ tiny |

### Per-cluster breakdown — UNI (notable clusters only)

| Cluster | Patches | Tumour % | vs baseline |
|---|---|---|---|
| cluster_02 | 756 | 20.9% | 4.2× |
| cluster_04 | 1,544 | 20.2% | 4.0× |
| cluster_05 | 5,734 | 14.3% | 2.9× |
| cluster_06 | 1,709 | 12.5% | 2.5× |
| noise | 404,953 | 5.2% | ≈ baseline |

### Conclusion (exp03)

HDBSCAN with 20 PCA components performs **worse than K-means** for both encoders:

- **CONCH**: 85.3% variance captured by 20 PCA dims, yet HDBSCAN finds only 3 blobs
  and the best cluster purity (3.57%) is **below the 5% baseline** — no tumour structure found.
- **UNI**: only 33.5% variance in 20 PCA dims — HDBSCAN is working in a severely
  compressed space that discards most of UNI's information. Of 480K input patches,
  84.4% are labelled noise. Four tiny clusters achieve 12–21% tumour purity but together
  capture <1% of tumour patches (recall = 0.007). Near-zero ARI (-0.035) confirms no
  global alignment with tumour labels.

Two reasons HDBSCAN struggles here:
1. Tumour patches are **not geometrically dense** in PCA space — they are a diffuse,
   buried minority signal, not a compact manifold that HDBSCAN can isolate.
2. The density threshold causes almost all tumour patches to be absorbed into the noise
   class or into large non-tumour blobs.

**Exp03 confirms**: K-means (exp02, k=8) extracts more usable tumour signal than HDBSCAN
for this dataset. The natural cluster structure found by HDBSCAN reflects morphological
outliers and stain-type groups, not tumour biology.

---

## XAI — Dimension Importance

Implemented in `histoRAG/classify.py → explain_dimensions`.

**Method (centroid difference with Cohen's d):**
1. Compute the mean embedding of all **true tumour** patches → `centroid_tumour`
2. Compute the mean embedding of all **true other** patches → `centroid_other`
3. Per dimension: `effect_size = |centroid_tumour - centroid_other| / pooled_std`
4. Rank dimensions by effect size (Cohen's d) — higher = more tumour-discriminative

XAI requires all embeddings in memory simultaneously, so it is not computed in the two-pass streaming pipeline used by exp01/exp02. The results below were produced from a one-shot full-load run on HPC. Output is saved under `summary.json → xai` with keys `top_dims`, `effect_sizes`, `tumour_centroid`, `other_centroid`.

### XAI Results (full dataset, HPC)

XAI top dims are identical between exp01 and exp02 for each encoder — correct, because
XAI depends only on ground-truth labels vs embeddings, not on the number of clusters.

| Encoder | Embedding dim | Top-5 tumour dims | Max Cohen's d | Mean top-20 d |
|---|---|---|---|---|
| CLIP | 512 | 301, 355, 2, 249, 349 | 0.895 | 0.777 |
| CONCH | 512 | 122, 177, 208, 169, 432 | **1.261** | **1.100** |
| UNI | 1024 | 593, 914, 365, 943, 905 | 1.160 | 0.912 |

**Cohen's d interpretation:** d < 0.5 = small, d ~0.8 = medium, d > 1.0 = large effect.

**CLIP** — max d = 0.895 (medium). No single dimension strongly separates tumour from
other. Tumour signal is weak and diffuse across the 512 dimensions. Matches CLIP's
poor clustering precision.

**CONCH** — max d = 1.261 (large), mean top-20 d = 1.100. All top-20 dimensions exceed
d = 1.0. CONCH has a concentrated, consistent tumour axis — specific dimensions carry
strong and reliable tumour signal.

**UNI** — max d = 1.160 (large), but mean top-20 d drops to 0.912 — the signal spreads
more across the larger 1024-dim space. UNI achieves the best clustering F1 (0.259) not
because one axis is sharpest, but because multiple sub-clusters collectively capture the
buried tumour signal at k=8.

### Practical implication for HistoRAG

| Use case | Best choice |
|---|---|
| Efficient tumour-aware indexing on a few dims | **CONCH dims 122, 177, 208** (d > 1.2) |
| Full-vector retrieval for best patch separation | **UNI** (best F1 = 0.259) |
| Avoid | CLIP — tumour signal too diffuse for reliable retrieval |

---

## Metrics

- **Precision** — of all patches predicted as tumour, how many are truly tumour
- **Recall** — of all true tumour patches, how many were grouped correctly
- **F1** — harmonic mean of precision and recall
- **Note**: metrics use post-hoc cluster alignment — same labels for assignment and
  evaluation. This is standard for unsupervised clustering analysis but is not
  independent validation.

---

## Shared modules

`histoRAG/classify.py`:
- `fit_kmeans` — clustering
- `match_clusters_to_labels` — majority-vote cluster → tumour/other mapping
- `classification_metrics` — Precision, Recall, F1
- `grouped_metrics` — per-site and per-WSI breakdown
- `explain_dimensions` — XAI via centroid difference (Cohen's d per dimension)

Each experiment has its own self-contained `run.py`:
- `exp01_kmeans_k2/run.py` — two-pass KMeans k=2 pipeline
- `exp02_overcluster_assign/run.py` — two-pass KMeans k=8 pipeline
- `exp03_hdbscan_clustering/run.py` — PCA + HDBSCAN pipeline

**Output structure** (per encoder):
```
outputs/{encoder}/
  summary_{encoder}.json
  cache/   kmeans_cluster_ids.npy  kmeans_true_labels.npy  kmeans_predicted.npy
           manifest.parquet  umap_emb.npy  umap_coords_2d.npy  umap_coords_3d.npy
           viz_cluster_labels.npy  viz_gt_labels.npy
  vis/     umap_by_cluster.png  umap_by_cluster_3d.png  umap_by_cluster_3d.html
           umap_by_groundtruth.png  umap_by_groundtruth_3d.png  umap_by_groundtruth_3d.html
```

---

## Running

```bash
# exp01 / exp02 — three run modes each
python hypotheses/H1_tumour_classification/exp01_kmeans_k2/run.py --encoder conch
python hypotheses/H1_tumour_classification/exp01_kmeans_k2/run.py --encoder conch --skip-kmeans  # rerun UMAP + plots
python hypotheses/H1_tumour_classification/exp01_kmeans_k2/run.py --encoder conch --plots-only   # replot only

python hypotheses/H1_tumour_classification/exp02_overcluster_assign/run.py --encoder conch
python hypotheses/H1_tumour_classification/exp02_overcluster_assign/run.py --encoder conch --skip-kmeans
python hypotheses/H1_tumour_classification/exp02_overcluster_assign/run.py --encoder conch --plots-only

# HPC — all encoders, exp01 + exp02 in one job
sbatch hypotheses/H1_tumour_classification/run_h1_hpc.sh

# exp03 — HDBSCAN (same three modes)
python hypotheses/H1_tumour_classification/exp03_hdbscan_clustering/run.py --encoder conch
python hypotheses/H1_tumour_classification/exp03_hdbscan_clustering/run.py --encoder conch --skip-hdbscan
python hypotheses/H1_tumour_classification/exp03_hdbscan_clustering/run.py --encoder conch --plots-only
sbatch hypotheses/H1_tumour_classification/exp03_hdbscan_clustering/run_h1_exp03_hpc.sh
```

---

## Data requirements

| Input | Source |
|---|---|
| Patch embeddings (h5) | HANCOCK download — CLIP, CONCH, UNI |
| `.geojson` annotation files | HANCOCK `WSI_PrimaryTumor_Annotations` |
