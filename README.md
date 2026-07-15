# HistoRAG — Retrieval-Augmented Histopathology Atlas

**FAU BIMAP SS2026 · Individual project · Vibhu Gupta**

This project investigates whether patch-level image embeddings from whole slide images
(WSIs), produced by **frozen** encoders, carry clinically meaningful structure: whether
tumour tissue groups on its own without labels, and how patients relate to each other at
the cancer-patch level. The goal is to justify — with evidence — the design of a
patient-level retrieval system built on patch-level tumour biology.

The dataset is [HANCOCK](https://hancock.research.fau.eu/) — a multimodal head and neck
cancer dataset with 763 patients, 701 primary tumour WSIs, and sparse QuPath tumour
polygon annotations (Dörrich et al., *Nature Communications* 2025).

No encoder training is involved anywhere. All three encoders (CLIP, CONCH, UNI) are used
frozen; H1/H2 are unsupervised clustering / similarity analysis with `.geojson` tumour
annotations used only at evaluation time, while H3 fits a linear probe (logistic
regression) on top of the same frozen embeddings.

---

## What's here

| Component | What it does |
|---|---|
| **H1** - `hypotheses/H1_tumour_classification/` | Does tumour tissue cluster on its own? K-means (k=2, k=8), HDBSCAN, + XAI dimension importance |
| **H2** - `hypotheses/H2_patient_correlation/` | How do patients relate? Site clustering (Q1) + cross-site tumour similarity (Q2) |
| **H3** - `hypotheses/H3_supervised_probe/` | Is tumour linearly separable with supervision? Logistic regression probe on frozen embeddings |
| **Hackday2** - `Hackday2/` | Self-contained single-WSI demo: patch → CONCH → HDBSCAN → 2 UMAP plots |
| **`histoRAG/`** | Shared library used by all experiments |

The headline conclusion: tumour is **not** the dominant axis of variation in any
encoder, but tumour signal is present and buried, and **tumour patches are broadly
site-agnostic** — which is the signal a patient-level RAG should be built on.

---

## Quickstart

The fastest way to see the pipeline end to end is the Hackday2 single-WSI demo — it has
its own step-by-step guide including the Google Drive data link:

➡️ **[Hackday2/README.md](Hackday2/README.md)**

```bash
# Python 3.12 venv
python -m venv .bimap
.bimap\Scripts\activate          # Windows
# source .bimap/bin/activate     # Unix
pip install -r requirements.txt
```

To reproduce the full H1 / H2 / H3 results you need the pre-computed HANCOCK patch
embeddings (per-site `.h5` files) and the `.geojson` annotations — see **Data** below and
each hypothesis README.

---

## Results at a glance

Full tables, per-tissue breakdowns, and interpretation live in each hypothesis README.

**H1 — buried tumour signal** (best precision at k=8, ~5% tumour baseline):

| Encoder | Precision | vs baseline | Verdict |
|---|---|---|---|
| CLIP | 0.090 | 1.75× | weak |
| CONCH | 0.127 | 2.5× | present |
| **UNI** | **0.154** | **3.1×** | **best** |

Tumour is never the dominant split (k=2 fails for all encoders). Ranking: **UNI > CONCH >> CLIP**.

**H2 — cancer tissue is site-agnostic:** within-site vs cross-site tumour-patch
similarity gap is only **+0.017** (both CONCH and UNI) — tumour patches from all four
anatomical sites overlap heavily in embedding space.

**H3 — tumour is linearly separable with supervision:** a logistic regression probe on
frozen UNI2-h embeddings gets **test AUROC 0.88** — a night-and-day jump over H1's best
unsupervised F1 of 0.26, confirming the tumour signal is real but buried, not absent.

See [H1 README](hypotheses/H1_tumour_classification/README.md) ·
[H2 README](hypotheses/H2_patient_correlation/README.md) ·
[H3 README](hypotheses/H3_supervised_probe/README.md).

---

## Repo structure

```
HistoRag-BIMAP/
├── histoRAG/                    # shared library
│   ├── loader.py               # streaming .h5 embedding loader (iter_encoder), patch-size detection
│   ├── labels.py               # tumour labels (.geojson) + site labels (clinical CSV)
│   ├── classify.py             # k-means / minibatch k-means, cluster→label matching, metrics, XAI
│   └── correlate.py            # UMAP (2D/3D), similarity distributions, heatmaps
├── hypotheses/
│   ├── H1_tumour_classification/
│   │   ├── exp01_kmeans_k2/            # is tumour the dominant axis?
│   │   ├── exp02_overcluster_assign/   # k=8 — is tumour signal present at all?
│   │   ├── exp03_hdbscan_clustering/    # natural density structure
│   │   ├── run_h1_exp12_hpc.sh          # HPC: exp01+exp02, all encoders
│   │   └── visualize_kmeans_umap.py
│   ├── H2_patient_correlation/
│   │   ├── exp01_site_clustering/       # Q1: patient vectors cluster by site?
│   │   └── exp02_tumour_similarity/     # Q2: cross-site tumour similarity (+ HPC script)
│   └── H3_supervised_probe/
│       └── exp01_linear_probe/          # logistic regression on frozen embeddings
├── Hackday2/                    # single-WSI HDBSCAN demo (self-contained)
├── requirements.txt            # local
└── requirements_hpc.txt        # TinyGPU / HPC
```

Each experiment folder is self-contained: `config.yaml` + `run.py` + `outputs/`.
WSIs, patches, embeddings, and index artifacts are gitignored.

---

## Encoders

| Encoder | Domain | Output dim | Notes |
|---|---|---|---|
| CLIP ViT-B/16 | General (natural images) | 512 | Open weights via `open_clip` |
| CONCH | Histopathology vision-language | 512 | MahmoodLab |
| UNI (UNI2-h) | Histopathology vision-only SSL | 1024 | MahmoodLab; requires HuggingFace access |

All encoders are used **frozen** — no training.

---

## Data

Experiments consume pre-computed per-site patch embeddings and QuPath annotations:

| Input | Layout | Source |
|---|---|---|
| Patch embeddings | `.h5` per anatomical site, per encoder | HANCOCK download (CLIP / CONCH / UNI) |
| Tumour annotations | `.geojson` per slide | HANCOCK `WSI_PrimaryTumor_Annotations` |

Point each experiment's `config.yaml` at your embeddings root. The Hackday2 demo bundles
its own Drive folder with a single WSI, CONCH weights, and one `.h5` — see its README.

---

## Required reading

- HANCOCK paper: Dörrich et al., *Nat Commun* 16, 7163 (2025) — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12322140/)
- CLAM (Mahmood Lab) — attention MIL for WSI classification; used in HANCOCK for tumour localization
