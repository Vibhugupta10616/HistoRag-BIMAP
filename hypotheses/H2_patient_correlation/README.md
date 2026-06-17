# H2 — Patient Correlation on a Patch Level

## Hypothesis

How do patients relate to each other based on their patch-level embeddings?

Two questions answered at patient level using pre-computed frozen embeddings:

---

## Q1 — Do aggregated patient embeddings cluster by anatomical site?

**Method:** Stream all patches slide by slide → mean pool per slide → 1 vector/patient
→ L2-normalize → UMAP scatter + patient×patient cosine similarity heatmap.

**Expected UMAP:** Distinct clusters per anatomical site.

**Expected heatmap:** Block-diagonal — high similarity within site, low across sites.

---

## Q2 — How similar is cancer tissue across patients regardless of site?

**Method:** Filter to tumour patches only (from `.geojson` annotations) → UMAP on
individual patches + patient-patient correlation matrix (adaptive aggregation).

**Expected UMAP:** Patches from all sites are **mixed** — no site-based separation.

**Expected heatmap:** Uniformly high similarity across all patients — cancer tissue
shares common histopathological patterns regardless of anatomical site.

### Adaptive aggregation (Q2 correlation matrix)

| Median tumour patches/patient | Method |
|---|---|
| ≥ 20 | Mean pool per patient → patient × patient matrix |
| < 20 | No aggregation → patch × patch matrix, grouped by patient |

See `histoRAG/correlate.py → decide_aggregation`.

---

## Exp01 — Site Clustering Results (Q1)

**Dataset:** 708 patients across 4 anatomical sites.

| Site | Patients |
|---|---|
| hypopharynx | 80 |
| larynx | 182 |
| oral_cavity | 129 |
| oropharynx | 317 |

---

### UMAP

#### CONCH (dim=512)

Larynx dominates a tight cluster in the bottom-left, with only 3–5 hypopharynx
points mixed in. Larynx also appears spread through the central cloud, suggesting
it is not fully consolidated into one region. Oral cavity tends to cluster on the
right side of the main cloud. Hypopharynx and oropharynx are largely scattered
throughout with no clear spatial grouping — oropharynx in particular bleeds into
every region, which is expected given it is the largest group (317 patients).
Overall structure is present but partial.

#### UNI (dim=1024)

Larynx forms a single isolated, tight cluster on the far right — pure orange with
almost no mixing from other sites. This is a stronger and more consolidated separation
than CONCH, where larynx was split across two locations. A second small cluster
appears in the top-left, predominantly oropharynx with a few hypopharynx and larynx
points. The large central blob contains all four sites mixed together. Hypopharynx
and oropharynx remain scattered with no clean grouping, consistent with their
anatomical proximity and histological overlap. UNI produces more organised spatial
structure than CONCH, particularly for larynx.

---

### Correlation Heatmap

| Metric | CONCH | UNI |
|---|---|---|
| Within-site mean similarity | 0.835 | 0.663 |
| Cross-site mean similarity | 0.819 | 0.621 |
| **Gap (within − cross)** | **0.016** | **0.043** |

**Per-site within-similarity:**

| Site | CONCH | UNI |
|---|---|---|
| hypopharynx | 0.835 | 0.623 |
| larynx | 0.835 | 0.646 |
| oral_cavity | **0.903** | **0.760** |
| oropharynx | 0.824 | 0.655 |

**CONCH heatmap** appears almost uniformly red across the entire 708×708 matrix.
The within-site and cross-site values are very close (gap = 0.016), meaning CONCH
compresses all patients into a tight high-similarity ball with very little room to
discriminate between sites. The block-diagonal structure is barely visible.

**UNI heatmap** shows more contrast. The gap (0.043) is 2.7× larger than CONCH,
and the block-diagonal blocks are visibly brighter than the off-diagonal regions.
The oral_cavity block stands out clearly (0.760 within vs 0.621 overall cross-site
mean) — oral cavity is the most cohesive site in both encoders, likely due to its
distinct mucosal and salivary gland tissue composition.

---

### Conclusion (Q1)

Both encoders capture weak site structure. UNI is meaningfully better:

- **Gap 2.7× larger** (0.043 vs 0.016) → same-site patients rank higher on average
- **UMAP shows cleaner larynx isolation** in UNI vs split/partial in CONCH
- **Oral cavity is the most cohesive site** in both encoders
- **Hypopharynx and oropharynx do not cluster cleanly** in either encoder —
  their anatomical and histological proximity makes them hard to separate
- **CONCH compresses embeddings too tightly** — high absolute similarity everywhere
  reduces discriminability between sites

Neither encoder achieves clean 4-cluster separation. The site signal is present but
weak. Q2 (tumour similarity) is the more clinically relevant test — whether cancer
patches cluster by shared biology rather than anatomical site.

---

## Exp02 — Tumour Similarity Results (Q2)

**Dataset:** 708 patients, tumour patches only (streamed per slide, variable rate per slide).

| Encoder | Total patches scanned | Tumour patches | Overall tumour % | Aggregation |
|---|---|---|---|---|
| CONCH | 8,212,546 | 415,398 | 5.1% | patient (median 310 patches/pt) |
| UNI | 2,076,207 | 103,812 | 5.0% | patient (median 78 patches/pt) |

Aggregation mode was `patient` for both — median tumour patches per patient well above the
threshold of 20, so each patient is represented by a mean-pooled tumour embedding.

---

### UMAP (50k subsampled tumour patches)

#### CONCH

The large central mass is heavily mixed across all four sites with no clean spatial
separation. Oral cavity (green) has a noticeably higher density in the upper-left region,
suggesting some oral cavity tumour patches retain site-specific features even within tumour
space. A small isolated cluster appears far right — a handful of hypopharynx outlier points
that look unlike the rest of the dataset. Overall the mixing is strong and close to the
expected result.

#### UNI

More "exploded" structure than CONCH. The central core is densely mixed with all four sites
overlapping, but many scattered satellite clusters radiate outward, most of them dominated
by oropharynx (red). This reflects UNI's wider spread in 1024-dim space: typical tumour
patches converge in the core regardless of site, while atypical or site-specific patches
form isolated groups at the periphery. UMAP issued a "graph not fully connected" warning,
which is expected when a high-dimensional space separates some patches too far to bridge
into one connected neighbourhood structure.

---

### Correlation Heatmap

| Metric | CONCH | UNI |
|---|---|---|
| Within-site mean | 0.717 | 0.499 |
| Cross-site mean | 0.696 | 0.468 |
| **Gap (within − cross)** | **0.021** | **0.030** |

**Per-site within-similarity (exp02):**

| Site | CONCH | UNI |
|---|---|---|
| hypopharynx | 0.714 | 0.473 |
| larynx | 0.725 | 0.499 |
| oral_cavity | **0.789** | **0.561** |
| oropharynx | 0.702 | 0.490 |

CONCH heatmap remains uniformly red with faint block structure. UNI heatmap shows
lower absolute similarity (~0.47–0.56 vs CONCH's 0.70–0.79) and slightly more visible
block boundaries, particularly for oral cavity. The diagonal self-similarity line is
clearly visible in UNI due to its wider embedding spread.

---

### Key Finding — exp01 vs exp02 Gap Shift

| | CONCH exp01 | CONCH exp02 | UNI exp01 | UNI exp02 |
|---|---|---|---|---|
| Input | all patches | tumour only | all patches | tumour only |
| Within-site | 0.835 | 0.717 | 0.663 | 0.499 |
| Cross-site | 0.819 | 0.696 | 0.621 | 0.468 |
| **Gap** | **0.016** | **0.021** | **0.043** | **0.030** |

**CONCH gap increases** (0.016 → 0.021) when filtering to tumour patches — tumour
embeddings still carry site-specific information. Filtering to cancer tissue does not
reduce site bias in CONCH.

**UNI gap decreases** (0.043 → 0.030) when filtering to tumour patches — tumour
embeddings become more site-agnostic. UNI encodes tumour biology more independently
of anatomical location. This is consistent with UNI achieving the best tumour
discrimination in H1 (F1=0.259).

Oral cavity is the most self-similar site in both encoders across both experiments,
reflecting its distinct mucosal and salivary gland tissue composition.

---

### Conclusion (Q2)

Both encoders show strong tumour patch mixing in UMAP (hypothesis broadly supported),
but neither achieves fully site-agnostic cancer similarity in the correlation matrix.
UNI comes closer — its tumour space gap shrinks when filtering to cancer patches, while
CONCH's site signal actually strengthens. For HistoRAG tumour-specific retrieval, UNI
is the better choice: it is more likely to surface similar cancer patients regardless
of anatomical site. CONCH retrieval in tumour space will still favour same-site patients.

---

## Running

```bash
# Local (streaming, low RAM)
python hypotheses/H2_patient_correlation/exp01_site_clustering/run.py
python hypotheses/H2_patient_correlation/exp01_site_clustering/run.py --encoder uni2h

python hypotheses/H2_patient_correlation/exp02_tumour_similarity/run.py
python hypotheses/H2_patient_correlation/exp02_tumour_similarity/run.py --encoder uni2h

# HPC (both encoders)
sbatch hypotheses/H2_patient_correlation/run_h2_hpc.sh
```

---

## Shared modules

- `histoRAG/loader.py` — `iter_encoder` (streaming), `detect_patch_size`
- `histoRAG/labels.py` — tumour labels from `.geojson` annotations
- `histoRAG/correlate.py` — aggregation, UMAP, correlation matrix, heatmap
