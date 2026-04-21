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

### Run 001 — seed 42 · 2026-04-21

| Metric | Value |
|---|---|
| Patches | 3,044 (2 slides, 20×, 256 px, Otsu tissue filter) |
| Query / Gallery split | 609 / 2,435 (stratified within-slide, 20%) |
| top-1 accuracy | **0.8998** |
| top-5 accuracy | **0.9967** |
| top-10 accuracy | **1.0000** |
| mAP@10 | **0.8990** |
| Random baseline (top-5) | 0.9688 |
| Embed time | 145 s (CPU — CUDA build not yet installed) |
| Index build time | 0.16 s |
| Query time | 1.53 s |

**Observation**: top-5 accuracy of 99.7% is exceptionally high and actually exceeds the random baseline by a very thin margin — suspicious given only 2 label classes (InvasionFront vs TumorCenter). With 2 classes the random top-5 baseline is already 96.9%, so the headroom is small. Results should be re-evaluated once more slides and label diversity are added in Phase 1. Encoding was done on CPU; expect ~10× speedup once CUDA torch is active.

**Config**: `configs/runs/20260421_001_clip-vitb16_faiss-flatip_hancock5_seed42.yaml`  
**UID**: `20260421_001_clip-vitb16_faiss-flatip_hancock5_seed42`
