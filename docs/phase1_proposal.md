# HistoRAG — Phase 1 Proposal
**Project:** Retrieval-Augmented Histopathology Atlas (FAU BIMAP SS2026)
**Phase 1 status:** Awaiting professor approval

---

## 1. Project Aim — Plain Language

Pathologists diagnose cancer by looking at tissue samples under a microscope. When they see an unusual pattern, a natural question is: *"Have I seen something like this before, and what was the diagnosis?"* Today, there is no easy way to search a database of tissue images the way we search Google Images.

**HistoRAG builds that search engine for tissue images.** Given a small patch of tissue (a 256×256 pixel crop from a scanned slide), our system finds the most visually similar patches from a reference database and returns their known diagnoses. This is called *image retrieval* — and it is the foundation of a computer-aided diagnostic atlas.

In Phase 0 we built and validated a working retrieval pipeline end-to-end: scan a tissue microarray (TMA) → extract patches → encode them with a deep network → store in a searchable index → evaluate how often the top retrieved patch shares the correct tissue class. We achieved **top-1 accuracy of 0.892 and mAP@10 of 0.894** using CLIP ViT-B/16 (a general-purpose vision model) as the encoder.

**Phase 1 asks the next scientific question:** Does an encoder *specifically trained on histopathology images* do better than a general-purpose one? And at what computational cost?

---

## 2. System Overview

### Figure 1 — Retrieval Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                     HistoRAG Retrieval Pipeline                      │
│                                                                      │
│  Whole Slide Image (WSI) /TMAs                                       │
│        │                                                             │
│        ▼                                                             │
│  ┌─────────────┐    ┌─────────────────────┐    ┌──────────────────┐  │
│  │   Tiling    │    │   Tissue Filter     │    │  Patch Library   │  │
│  │ 256×256 px  │───▶│  (Otsu-HSV, ≥15%   │───▶│  ~5,000–50,000   │  │
│  │    20×      │    │   tissue coverage)  │    │  patches         │  │
│  └─────────────┘    └─────────────────────┘    └────────┬─────────┘  │
│                                                         │            │
│                                                         ▼            │
│                                               ┌──────────────────┐   │
│                                               │  Encoder (CLIP)  │   │
│                                               │  → d-dim vector  │   │
│                                               └────────┬─────────┘   │
│                                                        │             │
│                                                        ▼             │
│                                               ┌──────────────────┐   │
│                                               │   FAISS Index    │   │
│                                               │  (cosine search) │   │
│                                               └────────┬─────────┘   │
│                                                        │             │
│  Query patch ──────────────────────────────────────────┘             │
│       │                                                              │
│       ▼                                                              │
│  Top-K retrieved patches  →  label match?  →  Top5 / mAP@10          │
└──────────────────────────────────────────────────────────────────────┘
```

### Figure 2 — Encoder Comparison (Phase 1 Central Experiment)

```
┌─────────────────────────────────┐    ┌──────────────────────────────────┐
│       CLIP ViT-B/16             │    │           UNI2-h                 │
│   (Baseline — Phase 0)          │    │   (Phase 1 Challenger)           │
├─────────────────────────────────┤    ├──────────────────────────────────┤
│ Pretrained on: 400M natural     │    │ Pretrained on: 100,000+          │
│ image–text pairs (OpenAI)       │    │ histopathology WSI patches       │
│                                 │    │ (MahmoodLab, SSL)                │
│ Architecture:  ViT-B/16         │    │ Architecture: ViT-H/14           │
│ Parameters:    86 M             │    │ Parameters:   630 M              │
│ Output dim:    512              │    │ Output dim:   1,536              │
│ VRAM (batch32): ~1.2 GB         │    │ VRAM (batch32): ~10–14 GB        │
│                                 │    │                                  │
│  "Knows" natural images.        │    │ "Knows" tissue staining,         │
│  Never saw H&E staining.        │    │  cell morphology, IHC patterns.  │
└─────────────────────────────────┘    └──────────────────────────────────┘
         │                                          │
         └─────────────────┬────────────────────────┘
                           ▼
              Which encoder finds more
              tissue-relevant neighbours?
```

---

## 3. Phase 0 Baseline Results (Reference Point for Phase 1)

| Metric | Mean (3 seeds) | Std |
|---|---|---|
| top-1 accuracy | 0.892 | ± 0.011 |
| top-5 accuracy | 0.994 | ± 0.002 |
| top-10 accuracy | 0.999 | ± 0.001 |
| mAP@10 | 0.894 | ± 0.004 |
| Random baseline (top-5) | 0.969 | — |

**Dataset:** 2 HANCOCK TMA blocks · 3,044 patches · 2 classes (`invasion_front`, `tumor_center`)
**Encoder:** CLIP ViT-B/16 · **Index:** FAISS flat cosine · **Split:** stratified within-slide 80/20

> **Important note on Phase 0 metrics:** top-5 and top-10 are near-ceiling because there are only 2 classes — a random retrieval has a 97% chance of hitting the right class in the top 5. These metrics are therefore **not informative** for Phase 1. All Phase 1 hypotheses are evaluated on **top-1 and mAP@10**, which remain discriminative regardless of class count.

> **Phase 0 vs Phase 1 data:** Phase 0 ran on TMA blocks (small pre-cropped tissue cores). Phase 1 will run on full Whole Slide Images (WSIs) — gigapixel scans where the tiling step actually matters. The CLIP baseline will be re-run on WSI data to ensure apples-to-apples comparison.

---

## 4. Hypotheses

### H1 — Primary: Domain-specific pretraining improves histopathology retrieval accuracy

**Plain language:** CLIP was trained on photographs and text captions from the internet — it has never seen an H&E-stained tissue section. UNI2-h was trained on over 100,000 histopathology whole-slide image patches using self-supervised learning. We hypothesize that this domain-specific training makes UNI2-h better at encoding tissue morphology, and therefore better at finding patches that actually belong to the same tissue class.

**Formal statement:**

> Replacing CLIP ViT-B/16 with UNI2-h as the patch encoder significantly increases retrieval accuracy on HANCOCK WSI tissue patches at 20× magnification.

| Element | Value |
|---|---|
| Independent variable | Encoder type: CLIP ViT-B/16 vs UNI2-h |
| Dependent variable | top-1 accuracy, mAP@10 |
| Controlled variables | Patch size (256 px), magnification (20×), tiling parameters, FAISS flat-IP index, eval split method, random seeds (42, 123, 2024) |
| Expected direction | UNI2-h > CLIP on both metrics |
| Null hypothesis | UNI2-h ≤ CLIP, or improvement < 2 percentage points |

---

### H2 — Paired: The accuracy improvement comes at a significant computational cost

**Plain language:** UNI2-h has 630 million parameters compared to CLIP's 86 million — more than 7× larger. We expect it to use more GPU memory and be slower to run. This hypothesis quantifies that tradeoff. A histopathology atlas needs to be scalable — if UNI2-h is both more accurate *and* computationally feasible, it is the right choice for a production system. If the gains are marginal but the cost is huge, CLIP may still be preferred.

**Formal statement:**

> The accuracy improvement of UNI2-h over CLIP ViT-B/16 is accompanied by substantially higher inference cost, quantified as trainable parameter count, peak GPU memory, and inference throughput.

| Element | Value |
|---|---|
| Independent variable | Encoder type: CLIP ViT-B/16 vs UNI2-h |
| Dependent variable | Trainable parameters, peak VRAM (MB), throughput (patches/sec) |
| Controlled variables | Hardware (GTX 1650 or Colab T4 if OOM), batch size (32), same patch set |
| Expected direction | UNI2-h is larger, uses more VRAM, and is slower |
| Interpretation | Whether the H1 accuracy gain justifies the H2 cost |

---

### H3 — Secondary (conditional): Within-slide evaluation overstates real-world retrieval performance

**Plain language:** In Phase 0, we tested retrieval by splitting query and gallery patches from *the same slide*. In a real diagnostic atlas, a pathologist's query would come from a *new patient's slide* that was never in the database. We hypothesize that our current numbers are optimistic — retrieval becomes harder when the query and gallery come from different slides (different patients, different staining batches).

**Condition:** This hypothesis is only tested if the Phase 1 WSI dataset contains **≥4 slides** (leave-one-slide-out requires at least 2 slides in the gallery).

**Formal statement:**

> Stratified within-slide retrieval accuracy overestimates cross-slide (leave-slide-out) retrieval accuracy for both CLIP and UNI2-h encoders.

| Element | Value |
|---|---|
| Independent variable | Evaluation split: stratified within-slide vs leave-slide-out |
| Dependent variable | top-1 accuracy and mAP@10 gap between split types, per encoder |
| Expected direction | Leave-slide-out accuracy < within-slide accuracy (generalization gap) |
| Scientific value | Establishes a more realistic and clinically relevant benchmark |

---

## 5. Evaluation Protocol

### 5.1 Dataset

- **Phase 1 dataset:** HANCOCK WSIs (exact slides TBD — provided by course instructors)
- **Phase 0 dataset (reference):** 2 TMA blocks, 3,044 patches, 2 classes
- **Preprocessing:** identical to Phase 0 — 256×256 px patches at 20× magnification, Otsu-HSV tissue filter (≥30% tissue coverage), max 5,000 patches per slide (tunable)
- **Labels:** slide-level labels from `configs/label_map.json` (extended when new slides arrive)

### 5.2 Experimental Design

**Design:** Controlled ablation study — single independent variable (encoder), all else held constant.

| Factor | Value |
|---|---|
| Encoder variants | CLIP ViT-B/16 (baseline), UNI2-h (treatment) |
| Seeds | 42, 123, 2024 (3 replications per encoder) |
| Split (primary) | Stratified within-slide, 80% gallery / 20% queries |
| Split (secondary, H3) | Leave-slide-out (one slide as query source, rest as gallery) |
| Index | FAISS IndexFlatIP (exact cosine search, unchanged from Phase 0) |
| Total experiment rows | 6 minimum (2 encoders × 3 seeds); 12 if H3 is added |

### 5.3 Metrics

| Metric | Definition | Why chosen |
|---|---|---|
| **top-1 accuracy** | Fraction of queries where the single nearest neighbour shares the query's class label | Most stringent; directly meaningful for a "look up a case" workflow |
| **mAP@10** | Mean average precision at 10 — rewards returning many relevant results early in the ranked list | Captures ranking quality, not just yes/no at position 1 |
| **Trainable parameters** | `sum(p.numel() for p in model.parameters())` | Encoder size proxy |
| **Peak VRAM** | `torch.cuda.max_memory_allocated()` during batch encoding | GPU feasibility on local hardware |
| **Throughput** | patches / second at batch size 32 | Practical scalability |

**Reporting format:** mean ± std over 3 seeds for top-1 and mAP@10. Single values for params/VRAM/throughput (deterministic given hardware).

### 5.4 Readout Examples (Course Format)

**H1 readout:**
> "Switching the encoder from CLIP ViT-B/16 to UNI2-h increased top-1 accuracy from 0.892 ± 0.011 to X.XXX ± Y.YYY and mAP@10 from 0.894 ± 0.004 to X.XXX ± Y.YYY on HANCOCK WSI patches at 20×. [Confirms / Refutes] hypothesis: histopathology-specific pretraining [does / does not] meaningfully improve tissue patch retrieval."

**H2 readout:**
> "UNI2-h has 630M trainable parameters vs CLIP's 86M (7.3× larger), uses X GB peak VRAM vs Y GB, and processes Z patches/sec vs W patches/sec (N× slower). The accuracy gain of [H1 delta] percentage points [is / is not] justified given the computational overhead for a WSI-scale atlas."

---

## 6. Required Changes to the MVP

The Phase 0 pipeline is functional but has four hardcoded components that must be made configurable before Phase 1 experiments can run.

### Change 1 — Encoder registry (REQUIRED, blocker for all experiments)

**Problem:** `pipeline.py` line 90 always creates a `ClipEncoder` object regardless of the config file. The `encoder.name` key exists in the YAML but nothing reads it.

**Fix:** Add a `UNIEncoder` class to `histoRAG/embed.py` and an `ENCODERS` dictionary that maps config names to classes. Change `pipeline.py` to use `get_encoder(config["encoder"]["name"])` instead of the hardcoded constructor. Swapping encoders then becomes a single line change in the YAML config.

**Files modified:** `histoRAG/embed.py`, `pipeline.py`

### Change 2 — New Phase 1 config file

**Fix:** Create `configs/phase1_uni2h.yaml` — a copy of the Phase 0 config with `encoder.name: uni2h`. All other parameters stay identical to ensure a controlled comparison.

**Files modified:** `configs/phase1_uni2h.yaml` (new file)

### Change 3 — Efficiency metrics in the experiment log (REQUIRED for H2)

**Problem:** The current `experiments.csv` schema has no fields for parameter count, VRAM, or throughput — these are not logged anywhere.

**Fix:** Add three new columns to the CSV schema: `param_count`, `peak_vram_mb`, `throughput_patches_per_sec`. Capture these during the embed step and pass them to `append_experiment_row()` in `histoRAG/log.py`. These fields will be `null` for CPU-only runs.

**Files modified:** `histoRAG/log.py`, existing rows in `experiments.csv` will have null for new columns (backward compatible)

### Change 4 — WSI tiling validation (REQUIRED before running, not a code change)

**Why:** Phase 0 tiled TMA blocks — small, uniform tissue cores. WSIs are gigapixel scans. The tiling code is correct but `max_patches_per_slide = 5000` and tissue filter thresholds need empirical validation on the new data format before running full experiments.

**Action:** Run one WSI through the tiler, inspect patch quality visually, adjust `max_patches_per_slide` and `min_tissue_frac` in the config as needed.

### Change 5 — Eval split dispatch from config (REQUIRED for H3 only)

**Problem:** `pipeline.py` line 114 hardcodes `stratified_within_slide`. The `slide_leave_out()` function already exists in `histoRAG/retrieve.py` but is unreachable from the command line.

**Fix:** Add an `if/elif` dispatch on `config["eval"]["query_gallery_split"]["method"]` to call either split function. Add a `held_out_slides` list to the config's eval section for the leave-out experiment.

**Files modified:** `pipeline.py`, `configs/phase1_crossslide.yaml` (new file for H3)

---

## 7. What Is Needed for the Experiments

| Requirement | Status | Notes |
|---|---|---|
| HANCOCK WSI dataset | **Pending** — provided by course | Phase 1 experiments cannot start without these |
| UNI2-h model access | **Granted** (HuggingFace MahmoodLab) | Requires `huggingface-cli login` at run time |
| Encoder registry code | **To be implemented** | ~50 lines of Python in `histoRAG/embed.py` + 5 lines in `pipeline.py` |
| Phase 1 YAML config | **To be created** | Copy of phase0 config with `encoder.name: uni2h` |
| Efficiency metric logging | **To be implemented** | 3 new CSV columns + 5 lines of capture code |
| GPU — GTX 1650 (4 GB VRAM) | **Available** | Sufficient for CLIP; UNI2-h (ViT-H) may OOM — see risk below |
| Colab T4 (fallback) | **Available** | Needed if UNI2-h OOMs on local GPU |
| Eval split dispatch | **To be implemented** | Required only for H3; ~10 lines in `pipeline.py` |

### Hardware Risk: UNI2-h VRAM

UNI2-h is a ViT-H model (630M parameters). At batch size 32, it requires approximately 10–14 GB of VRAM during inference — exceeding the local GTX 1650 (4 GB). **Mitigation options:**

1. **CPU fallback** (`device: cpu` in config) — slow (~10–20× lower throughput) but produces identical embeddings. Throughput numbers are then reported as "CPU throughput." The embed cache means this only needs to run once per encoder.
2. **Colab T4** — 16 GB VRAM, sufficient for UNI2-h. Export embeddings to `.npy` file and continue local evaluation.
3. **Reduced batch size** — reduce batch to 4–8 and use `torch.cuda.empty_cache()` between batches. May stay within 4 GB, at reduced throughput.

The fallback strategy will be determined empirically after the WSIs arrive.

---

## 8. Timeline

| Milestone | Action | Depends on |
|---|---|---|
| WSIs received | Validate tiling on one slide, inspect patches | Course delivery |
| Week 1 | Implement encoder registry + UNIEncoder class | — |
| Week 1 | Add efficiency metrics to log | — |
| Week 1–2 | Re-run CLIP on WSI data (3 seeds) — new baseline | WSIs + registry |
| Week 2 | Run UNI2-h on WSI data (3 seeds) — H1 + H2 | CLIP baseline done |
| Week 2–3 | H3 (leave-slide-out) if ≥4 WSIs available | H1 + H2 done |
| Week 3 | Interpret results, update EXPERIMENT_LOG.md | All runs done |
| Week 3–4 | Present Phase 1 results | — |

---

## Summary Table

| Question | Answer |
|---|---|
| **What is the project?** | A retrieval search engine for histopathology tissue patches — given a query patch, find the most similar patches and their diagnoses |
| **Phase 1 main hypothesis** | UNI2-h (histopath SSL, 630M params) retrieves more accurate tissue-class matches than CLIP ViT-B/16 (natural-image, 86M params) |
| **Primary metrics** | top-1 accuracy, mAP@10 (mean ± std over 3 seeds); NOT top-5/top-10 (saturated at few classes) |
| **Secondary metrics** | Trainable params, peak VRAM (MB), throughput (patches/sec) |
| **Evaluation design** | Controlled ablation: one IV (encoder), everything else fixed; 3 seed replications |
| **What changes in the MVP** | Wire encoder registry (blocker), add efficiency metrics to log, validate WSI tiling, expose eval split from config |
| **What is needed** | HANCOCK WSIs (pending), UNI2-h access (granted), ~1 week of coding, GPU fallback plan for UNI2-h |
