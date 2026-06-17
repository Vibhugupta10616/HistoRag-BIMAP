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

*To be run — see `exp02_tumour_similarity/run.py`.*

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
