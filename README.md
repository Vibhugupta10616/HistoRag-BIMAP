# HistoRAG — Retrieval-Augmented Histopathology Atlas

**FAU BIMAP SS2026 · Individual project · Vibhu Gupta**

This project investigates whether retrieval-augmented methods can be used to build an interactive histopathology atlas from whole slide images (WSIs). Image patches extracted from slides are embedded using pretrained vision or vision-language models and stored in a vector database. The system allows image queries to retrieve visually similar tissue regions and evaluates retrieval performance on labeled histopathology data.

## Quickstart

```bash
# 1. Activate venv (Python 3.12)
bimap\Scripts\activate   # Windows

# 2. Install package + dev deps
pip install -e ".[dev]"

# 3. Verify install
python -c "import histoRAG; print(histoRAG.__version__)"

# 4. Run MVP pipeline (after placing WSIs in data/raw/ and updating configs/phase0_mvp.yaml)
python pipeline.py --config configs/phase0_mvp.yaml --seed 42
```

## Dataset

HANCOCK dataset (hancock.research.fau.eu). A subset of 5–10 slides is used for Phase 0.
Slides are not redistributed; `data/raw/` is git-ignored.

## Experiment log

All runs are logged in `experiments/experiments.csv` with per-run config snapshots in `configs/runs/`.
See `EXPERIMENT_LOG.md` for human-readable interpretations.

## Phase roadmap

| Phase | Goal | Encoder |
|---|---|---|
| 0 — MVP | End-to-end retrieval pipeline | CLIP ViT-B/16 |
| 1 — Formalize | Controlled baselines comparison | CLIP vs UNI2-h |
| 2 — Experiments | HPO, index ablations, Pro-tasks | Multiple |
| 3 — Consolidation | Reproducibility + figures | — |
| 4 — Finish line | Report + final presentation | — |
