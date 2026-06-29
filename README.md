# HistoRAG — Retrieval-Augmented Histopathology Atlas

**FAU BIMAP SS2026 · Individual project · Vibhu Gupta**

This project investigates whether patch-level and slide-level image embeddings from
whole slide images (WSIs) can support clinically meaningful analyses: finding similar
patients, classifying tumour tissue, and understanding how patients relate to each other
at the cancer-patch level.

The dataset is [HANCOCK](https://hancock.research.fau.eu/) — a multimodal head and neck
cancer dataset with 763 patients, 701 primary tumour WSIs, and sparse QuPath tumour
polygon annotations (Dörrich et al., *Nature Communications* 2025).

---

## Quickstart

```bash
# 1. Activate virtualenv (Python 3.12)
bimap\Scripts\activate          # Windows
# source bimap/bin/activate     # Unix

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify
python -c "import histoRAG; print(histoRAG.__version__)"

# 4. Run Phase 0 MVP (patch-level retrieval baseline)
python MVP/pipeline.py --config MVP/configs/phase0_mvp.yaml --seed 42
```

---

## Phase roadmap

| Phase | Goal | Status |
|---|---|---|
| **0 — MVP** | End-to-end patch retrieval pipeline (CLIP + FAISS) | **Complete** |
| **1 — H1** | Tumour vs Other classification on patch embeddings + XAI | **Complete** |
| **1 — H2** | Patient correlation maps: site clustering + cancer similarity | **Complete** |
| 2 — Extensions | Spatial arrangement (ABMIL/graph), index ablations | Planned |
| 3 — Consolidation | Reproducibility, figures, report | Planned |

---

## Phase 0 results (MVP baseline)

**Dataset:** 2 HANCOCK TMA blocks · 3,044 patches · 2 tissue classes  
**Encoder:** CLIP ViT-B/16 (frozen) · **Index:** FAISS flat cosine

| Metric | Mean ± SD (3 seeds) |
|---|---|
| top-1 accuracy | 0.892 ± 0.011 |
| top-5 accuracy | 0.994 ± 0.002 |
| mAP@10 | 0.894 ± 0.004 |

Full results and per-run interpretation in `EXPERIMENT_LOG.md`.

---

## Phase 1 hypotheses

See each hypothesis README for full definition, experiments, results, and run instructions.

| Hypothesis | Folder | README |
|---|---|---|
| H1 — Unsupervised tumour grouping | `hypotheses/H1_tumour_classification/` | [H1 README](hypotheses/H1_tumour_classification/README.md) |
| H2 — Patient correlation maps | `hypotheses/H2_patient_correlation/` | [H2 README](hypotheses/H2_patient_correlation/README.md) |

---

## Repo structure

```
HistoRag-BIMAP/
├── pipeline.py                  # Phase 0/1 retrieval pipeline (tile→embed→index→eval→log)
├── requirements.txt
├── histoRAG/
│   ├── tile.py                  # WSI loading + Otsu patch extraction
│   ├── embed.py                 # encoders (CLIP/CONCH/UNI2-h) + FAISS index
│   ├── retrieve.py              # retrieval splits + top-k accuracy + mAP
│   ├── log.py                   # config loading, seeding, experiment CSV logging
│   ├── labels.py                # tumour labels (geojson) + site labels (clinical CSV)
│   ├── correlate.py             # UMAP, similarity distribution, interactive 3D UMAP
│   ├── classify.py              # H1 classifier stub + real classification metrics
│   └── viz/                     # Streamlit retrieval demo
├── hypotheses/
│   ├── H1_tumour_classification/
│   │   ├── exp01_kmeans_k2/             # k=2 dominant-axis test
│   │   └── exp02_overcluster_assign/    # k=8 buried-signal test
│   └── H2_patient_correlation/
│       ├── exp01_site_clustering/       # H2 Q1: config.yaml + run.py + outputs/
│       └── exp02_tumour_similarity/     # H2 Q2: config.yaml + run.py + outputs/
├── configs/
│   ├── phase0_mvp.yaml          # Phase 0 retrieval config
│   ├── phase1_*.yaml            # Phase 1 slide-level retrieval configs (3 encoders)
│   └── runs/                    # auto-generated immutable per-run config snapshots
├── experiments/
│   └── experiments.csv          # all retrieval pipeline runs
└── data/                        # gitignored — WSIs, patches, indexes, embeddings
```

---

## Encoders

| Encoder | Domain | Output dim | Notes |
|---|---|---|---|
| CLIP ViT-B/16 | General (natural images) | 512 | Open weights via `open_clip` |
| CONCH | Histopathology vision-language | 512 | MahmoodLab |
| UNI2-h | Histopathology vision-only SSL | 1024 | MahmoodLab; requires HuggingFace access |

All encoders are used **frozen** — no model training required.

---

## Providing embeddings (H1 / H2)

Place pre-computed embedding `.npy` files at:
```
data/embeddings/<encoder-name>/patch_embeddings.npy
```
Shape: `(N_patches, dim)`, `float32`, **row-aligned to** `data/patches/manifest.parquet`.
Then update `inputs.embeddings_path` in the relevant experiment `config.yaml`.

---

## Required reading

- HANCOCK paper: Dörrich et al., *Nat Commun* 16, 7163 (2025) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12322140/)
- CLAM (Mahmood Lab) — attention MIL for WSI classification; used in HANCOCK paper for tumour localization
