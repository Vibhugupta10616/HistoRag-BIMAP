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
individual patches (no aggregation) + pairwise cosine similarity distribution
comparing within-site vs cross-site patch pairs.

**Expected UMAP:** Patches from all sites are **mixed** — no site-based separation.

**Expected distribution:** Within-site and cross-site similarity curves heavily
overlap — cancer tissue shares common histopathological patterns regardless of
anatomical site.

---

## Config

Each experiment reads `config.yaml` in its own folder — set `inputs.embeddings_root`
(and `inputs.geojson_dir` for exp02) to point at your local paths, or override per-run
with `--embeddings-root` / `--geojson-dir`.


## Running

```bash
# exp01 — one mode only, no cache/subsample flags: streams all slides, mean-pools per
# patient (708 vectors total — cheap regardless), always a "full" run.
python hypotheses/H2_patient_correlation/exp01_site_clustering/run.py --encoder conch
python hypotheses/H2_patient_correlation/exp01_site_clustering/run.py --encoder uni2h

# exp02 — full run, tumour patches (all of them) + non-tumour patches CAPPED at
# 250k/site for local RAM safety (~1M total, not the full ~95% of the dataset).
python hypotheses/H2_patient_correlation/exp02_tumour_similarity/run.py --encoder conch
python hypotheses/H2_patient_correlation/exp02_tumour_similarity/run.py --encoder uni2h

# exp02 --full — collect ALL non-tumour patches, no cap (HPC only; ~16GB CONCH / ~32GB UNI RAM)
python hypotheses/H2_patient_correlation/exp02_tumour_similarity/run.py --encoder conch --full

# exp02 --plots-only — replot from cached arrays, skips the ~20 min embedding scan entirely
python hypotheses/H2_patient_correlation/exp02_tumour_similarity/run.py --encoder conch --plots-only
python hypotheses/H2_patient_correlation/exp02_tumour_similarity/run.py --encoder uni2h --plots-only

# HPC (exp02, both encoders, runs with --full)
sbatch hypotheses/H2_patient_correlation/exp02_tumour_similarity/run_h2_exp02_hpc.sh
```
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

**Dataset:** 708 patients, tumour patches only (streamed per slide, no aggregation).

| Encoder | Total patches scanned | Tumour patches | Overall tumour % |
|---|---|---|---|
| CONCH | 8,212,546 | 415,398 | 5.1% |
| UNI | 2,076,207 | 103,812 | 5.0% |

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
form isolated groups at the periphery.

---

### Similarity Distribution (KDE)

6,000 patches sampled per site → ~72M within-site pairs and ~216M cross-site pairs.

| Metric | CONCH | UNI |
|---|---|---|
| Within-site mean similarity | 0.535 | 0.296 |
| Cross-site mean similarity | 0.518 | 0.280 |
| **Gap (within − cross)** | **+0.017** | **+0.017** |

Both distributions show **heavy overlap** between within-site and cross-site curves —
the majority of patch pairs share similar cosine similarity regardless of whether the
two patches come from the same or a different anatomical site. This is the core finding:
cancer tissue is broadly similar across sites.

The small positive gap (+0.017 for both encoders) means patches from the same site are
very slightly more similar to each other than to patches from other sites, but this
difference is small relative to the overall spread of the distribution.

**CONCH** operates at higher absolute similarity (~0.52–0.54) consistent with its
tighter embedding space seen in exp01. **UNI** operates at lower absolute values
(~0.28–0.30) due to its wider spread in 1024-dim space, but the gap is identical,
suggesting both encoders capture the same degree of site-agnostic tumour biology.

---

### Conclusion (Q2)

Both encoders strongly support the hypothesis: tumour patches from all four anatomical
sites (hypopharynx, larynx, oral cavity, oropharynx) are broadly similar in embedding
space. The UMAP shows sites mixed together with no clean boundaries, and the similarity
distribution curves for within-site and cross-site pairs nearly completely overlap.

The identical gap (+0.017) across CONCH and UNI suggests a consistent small residual
site signal — cancer tissue is not completely site-agnostic, but site accounts for only
a tiny fraction of the total similarity variance. For HistoRAG tumour-specific retrieval,
both encoders will surface similar cancer patients across sites, making them viable for
cross-site patient matching.

---

## Shared modules

- `histoRAG/loader.py` — `iter_encoder` (streaming), `detect_patch_size`
- `histoRAG/labels.py` — tumour labels from `.geojson` annotations
- `histoRAG/correlate.py` — UMAP, similarity distribution, interactive 3D UMAP
