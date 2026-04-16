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

<!-- Entries will be added here after runs complete (Step 8) -->
