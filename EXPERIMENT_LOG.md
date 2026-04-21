# Experiment Log — HistoRAG

Human-readable narrative for each experiment run.
Machine-readable data lives in `experiments/experiments.csv`.
Config snapshots live in `configs/runs/<uid>.yaml`.

---

## Phase 0 — MVP Baseline (CLIP ViT-B/16 + FAISS flat IP)

**Hypothesis**: A pretrained CLIP ViT-B/16 vision encoder combined with FAISS exact cosine
retrieval achieves top-5 label-match accuracy significantly above a random baseline on HANCOCK
histopathology patches extracted at 20× magnification, 224×224 px, tissue-filtered.

**Single controlled variable**: random seed (42, 123, 2024). All other config held constant.

### Run 001 — seed 42 · 2026-04-21 *(pre-fix, superseded by Run 002)*

Pre-bugfix run using a seed-coupled embedding cache key (pipeline bug — query labels returned as Arrow-backed array, crashing numpy indexing). Results are numerically identical to Run 002 but the pipeline code was incorrect. Disregard for analysis.

**UID**: `20260421_001_clip-vitb16_faiss-flatip_hancock5_seed42`

---

### Run 002 — seeds {42, 123, 2024} · 2026-04-21

3,044 patches · 2 slides · 20× · 256 px · Otsu tissue filter · stratified within-slide split (80/20)  
Embeddings computed once (seed-agnostic cache), evaluation repeated per seed.  
Encoding on CPU (~114 s); CUDA build installed in parallel — expect ~10× speedup on next run.

| Metric | seed 42 | seed 123 | seed 2024 | Mean ± SD |
|---|---|---|---|---|
| top-1 accuracy | 0.8998 | 0.8998 | 0.8768 | 0.892 ± 0.011 |
| top-5 accuracy | 0.9967 | 0.9934 | 0.9918 | 0.994 ± 0.002 |
| top-10 accuracy | 1.0000 | 0.9984 | 0.9984 | 0.999 ± 0.001 |
| mAP@10 | 0.8990 | 0.8955 | 0.8887 | 0.894 ± 0.004 |
| Random baseline (top-5) | 0.9688 | 0.9688 | 0.9688 | — |
| Query time (s) | 0.75 | 0.74 | 0.74 | — |

**Observation**: Results are stable across seeds (SD < 0.011 on all metrics). The headline top-5 (99.4%) is near-ceiling but misleading — with only 2 label classes the random top-5 baseline is already 96.9%, leaving ~2.5 pp of real headroom. The informative metrics are **top-1 = 89.2%** and **mAP@10 = 89.4%**, which measure single-retrieval precision and ranking quality. Seed 2024 shows a small top-1 dip (0.877 vs 0.900), likely due to query set sampling variance; this will stabilize once Phase 1 adds more slides and label diversity.

**Config**: `configs/runs/20260421_00{2,3,4}_clip-vitb16_faiss-flatip_hancock5_seed{42,123,2024}.yaml`  
**UIDs**: `20260421_00{2,3,4}_clip-vitb16_faiss-flatip_hancock5_seed{42,123,2024}`
