# HistoRAG — Phase 0 MVP Execution Plan (Individual)

> **Status**: Ultraplan (remote) variant rejected on 2026-04-16 — it dropped the Phase-1 UNI2-h experiment hook and silently switched the demo from Streamlit to Gradio, both contrary to locked decisions. This local plan is the authoritative version.
>
> **Note on plan location**: Plan mode restricts edits to this file (`C:\Users\vibhu\.claude\plans\polished-doodling-zephyr.md`). First post-approval action will be copying this document into the project repo at `D:\College\Sem_5\HistoRag-BIMAP\PLAN.md` (filename adjustable) so it lives alongside the code and is git-tracked.

---

## Context

**Project**: HISTORAG — Retrieval-Augmented Histopathology Atlas (FAU BIMAP SS2026).
**Role**: Individual project. Course lists three students on HISTORAG (Vibhu Gupta, Satyaki Bhattacharjee, Taimoor Ajmal), but each delivers the full pipeline independently with their own experimental setup (e.g., different encoders, different index variants). **This plan is scoped to Vibhu only**.
**Main contact**: Prof. Dr. Andreas Kist (andreas.kist@fau.de).
**Deadline**: MVP presentation **Fri 2026-04-24** (~8 days from today, 2026-04-16).
**Gate semantics**: Phase 0 MVP is pass/fail for continuing in the course (kickoff deck slide: "Gate to stay in project").

**Why this plan exists**: Course docs (`docs/BIMAP SS26 - Kickoff.pdf`, `docs/HISTORAG – Retrieval.pdf`) specify a semester-long research project styled like a mini NeurIPS paper — not a Kaggle notebook. Scientific discipline is enforced: every experiment must have hypothesis / single controlled change / documented configuration / run ID / metric / interpretation. Tier-A logging is mandatory: `experiments.csv`, `config.yaml` per run, ≥3 seeds per experiment. "Vibe coding" is explicitly discouraged — every line must be explainable at the *what does it do* level (e.g., "computes the local attention mask"), not merely *how does it work mechanically*.

**Phase 0 goal**: Deliver an end-to-end `WSI → patch → embedding → FAISS → top-k retrieval` pipeline with evaluation and a live demo, architected so Phase 1 baselines comparison and Pro-tasks (text query, LLM descriptions, cross-slide) are config-swap extensions rather than rewrites.

**Intended outcome (concrete)**:
- Reproducible pipeline runs on 5–10 HANCOCK slides using CLIP ViT-B/16 + FAISS flat L2.
- ≥3 rows in `experiments/experiments.csv` (seeds 42, 123, 2024) with a top-5 accuracy / mAP@5 number and an interpretation paragraph in `EXPERIMENT_LOG.md`.
- Streamlit demo: upload a patch → top-k visually similar patches with slide ID, coords, label, distance.
- Repo scaffolded so `configs/` toggles encoder/index for Phase 1 without touching source code.

---

## Locked decisions (from clarification Q&A)

| Decision | Value | Rationale |
|---|---|---|
| Dataset | 5–10 HANCOCK WSIs (subset) | Full corpus too large to download now; subset sufficient for MVP; full-set later = config change |
| Compute | Local, NVIDIA GTX 1650 4 GB VRAM | CLIP ViT-B/16 fits easily; Colab deferred to Phase 1 if UNI benchmarking exceeds 4 GB |
| MVP encoder | **CLIP ViT-B/16 (OpenAI, via `open_clip_torch`)** | Open weights, 4-GB-safe, vision tower covers MVP requirements, text tower = free future unlock for Pro-1 |
| Phase 1 encoder | UNI2-h (MahmoodLab) — **access already granted** | First controlled experiment in Phase 1: CLIP vs UNI2-h on HANCOCK |
| Vector index | FAISS `IndexFlatIP` with L2-normalized embeddings (cosine similarity) | Exact search at MVP scale (≤100k vectors); IVF/HNSW reserved as Phase 2 ablations |
| Demo UI | Streamlit | Highest live-demo impact, runs offline on laptop, ~half-day build |
| Packaging | `pyproject.toml` + `src/` layout, pip-installable | Matches grading rubric optional "pip-installable"; avoids import-path bugs; `openslide-bin` pip wheel ships the Windows DLL |

---

## Scientific framing for Phase 0

**Primary hypothesis (Run set #001–#003)**:
> A pretrained CLIP ViT-B/16 vision encoder combined with FAISS exact cosine retrieval achieves top-5 label-match accuracy significantly above a random baseline on HANCOCK histopathology patches extracted at 20× magnification, 224×224 px, tissue-filtered.

**Single controlled variable in Phase 0**: random seed (42, 123, 2024). All other config held constant. Seed governs: query/gallery split, any stochastic preprocessing order, NumPy/PyTorch RNG. Encoder and index are deterministic given fixed inputs.

**Metrics** (reported mean ± std over the 3 seeds):
- `top-1`, `top-5`, `top-10` accuracy = fraction of queries whose top-K retrieved patches contain ≥1 with the same class label.
- `mAP@10` = mean average precision at 10 over queries.
- Random baseline = 1/|classes|; stratified random baseline also computed for context.

**Run ID convention**: `YYYYMMDD_NNN_<encoder>_<index>_<dataset>_seed<N>` — example `20260422_001_clip-vitb16_faiss-flatip_hancock5_seed42`.

**Interpretation (written per run, by hand)**: 1–3 sentences stating what the number means in context of the random baseline and what the next controlled change will be. Stored in `EXPERIMENT_LOG.md`.

---

## Repo structure (target end of Phase 0)

> **Updated:** Simplified from pip-installable `src/` layout to a flat local project.
> `requirements.txt` replaces `pyproject.toml`. Single `pipeline.py` entrypoint replaces `scripts/`.
> Tests are gitignored (local only).

```
HistoRag-BIMAP/
├── requirements.txt               # deps (pip install -r requirements.txt)
├── pipeline.py                    # single CLI entrypoint: tile → embed → index → eval → log
├── README.md                      # quickstart + reproduction instructions
├── PLAN.md                        # this document
├── EXPERIMENT_LOG.md              # human-readable narrative log
├── .gitignore                     # data/, bimap/, tests/, __pycache__, *.faiss, *.npy …
├── configs/
│   ├── phase0_mvp.yaml            # canonical MVP config (single source of truth)
│   └── runs/                      # immutable per-run config snapshots (git-tracked)
├── experiments/
│   └── experiments.csv            # Tier-A log (schema below)
├── data/                          # git-ignored
│   ├── raw/                       # downloaded HANCOCK WSIs
│   ├── patches/                   # tiled patches + manifest.parquet
│   └── indexes/                   # cached embeddings (.npy) + FAISS indexes
├── histoRAG/
│   ├── __init__.py                # exports __version__
│   ├── tile.py                    # WSI class + Otsu tissue filter + Tiler
│   ├── embed.py                   # ClipEncoder + FaissFlatIP
│   ├── retrieve.py                # top_k_accuracy, mAP@k, query/gallery split
│   └── log.py                     # load_config, hash_config, set_all_seeds, append_experiment_row
└── tests/                         # gitignored — local regression tests only
    ├── __init__.py
    ├── conftest.py                # shared fixtures (dummy manifest, random embeddings)
    ├── test_tile.py
    ├── test_embed.py
    └── test_retrieve.py
```

---

## Canonical config schema (`configs/phase0_mvp.yaml`)

```yaml
# HistoRAG Phase-0 MVP canonical config
run:
  name: phase0_mvp
  seed: 42                      # overridden by --seed CLI flag
  out_dir: experiments/         # where logs/snapshots land
  device: auto                  # "auto" | "cuda" | "cpu"; auto picks cuda if available

data:
  dataset: hancock5             # identifier appearing in run UID
  raw_dir: data/raw
  slide_ids:                    # exact HANCOCK slide IDs, to be filled in Step 2
    - TBD_slide_01
    - TBD_slide_02
    # ... up to 10
  label_source: slide_level     # "slide_level" | "patch_level" — confirmed during Step 2
  label_map: null               # path to JSON mapping slide_id → label; filled in Step 2

tiling:
  magnification: 20             # target magnification (MPP ≈ 0.5)
  patch_size: 256               # extracted pixel size at selected level
  stride: 256                   # non-overlapping grid
  tissue_filter:
    method: otsu_hsv            # Otsu threshold on HSV saturation channel
    thumb_downsample: 32        # downsample factor for thumbnail
    min_tissue_frac: 0.3        # keep patches whose tissue fraction ≥ this
  max_patches_per_slide: 5000   # cap for MVP; null = no cap
  patches_dir: data/patches

encoder:
  name: clip-vitb16             # looked up in encoders/registry.py
  model_id: ViT-B-16
  pretrained: openai            # open_clip weight tag
  batch_size: 32                # halve on OOM
  preprocess:
    resize: 224
    normalize: clip_openai      # keyed lookup for mean/std

index:
  name: faiss-flatip            # FaissFlatIP (cosine via L2-normalized IP)
  normalize: true
  save_path: data/indexes/phase0_mvp.faiss

eval:
  query_gallery_split:
    method: stratified_within_slide  # 80/20 per slide, stratified by label
    query_frac: 0.2
  k_values: [1, 5, 10]
  compute_map_at: 10
  random_baseline: true         # compute random + stratified-random baselines
```

**Config hash**: SHA-256 of canonicalized YAML → logged in every experiments.csv row so two runs with identical config are identifiable.

---

## `experiments.csv` schema (Tier-A mandatory log)

| column | type | meaning |
|---|---|---|
| `uid` | str | `YYYYMMDD_NNN_<encoder>_<index>_<dataset>_seed<N>` |
| `date_utc` | ISO8601 str | timestamp of row write |
| `git_commit` | str | `git rev-parse HEAD` at run time |
| `config_hash` | str | SHA-256 of canonicalized config YAML |
| `config_path` | str | path to immutable snapshot under `configs/runs/` |
| `encoder` | str | encoder name, e.g. `clip-vitb16` |
| `index` | str | index name, e.g. `faiss-flatip` |
| `dataset` | str | dataset identifier, e.g. `hancock5` |
| `num_patches` | int | total patches embedded |
| `num_query` | int | query set size |
| `num_gallery` | int | gallery set size |
| `seed` | int | RNG seed |
| `top1` | float | top-1 accuracy |
| `top5` | float | top-5 accuracy |
| `top10` | float | top-10 accuracy |
| `map_at_10` | float | mAP@10 |
| `random_baseline_top5` | float | reference for interpretation |
| `embed_time_s` | float | wall clock for encoding |
| `index_time_s` | float | wall clock for FAISS build |
| `query_time_s` | float | wall clock for eval queries |
| `notes` | str | free-text, 1-line summary — e.g. "Phase 0 baseline, seed 42" |

---

## Execution steps — 8-day sprint, ultra-detailed

> **Day numbering starts 2026-04-17** (day after today). MVP presentation = 2026-04-24 = end of Day 8.

### Step 0 — Pre-flight (before Day 1, ~30 min)
- Confirm Python venv per CLAUDE.md rule: scan repo root for any folder containing `Scripts\activate.bat`; if one found, ask to use; if none, create `.venv` via `python -m venv .venv` after confirmation.
- Activate: `.venv\Scripts\activate` (Windows).
- Confirm GPU visible: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` → expect `True GTX 1650`.
- **Acceptance**: GPU visible; pip upgraded to latest.

### Step 1 — Repo bootstrap (Day 1, ~3 h)
- `git init` in `D:\College\Sem_5\HistoRag-BIMAP`; create GitHub repo (`HistoRag-BIMAP`, public or private — private OK); push `main`.
- Add `.gitignore` excluding `data/`, `.venv/`, `__pycache__/`, `*.pt`, `*.bin`, `data/indexes/*.faiss`.
- Write `pyproject.toml` with:
  - Build system: `setuptools>=61`.
  - Package name `histoRAG`, version `0.0.1`, Python `>=3.10`.
  - Runtime deps: `torch>=2.1`, `open_clip_torch>=2.24`, `timm>=0.9.8`, `openslide-bin>=4.0`, `openslide-python>=1.3`, `faiss-cpu>=1.7`, `numpy`, `pandas`, `pyarrow`, `Pillow`, `streamlit>=1.30`, `pyyaml`, `tqdm`, `scikit-learn`, `huggingface_hub`.
  - Optional `[dev]`: `pytest`, `pytest-cov`, `ruff`, `black`.
- Create empty module skeleton (all `__init__.py` files + placeholder modules listed in repo structure).
- Create `README.md` stub with project title, one-paragraph description (copied/paraphrased from `docs/HISTORAG – Retrieval.pdf`), and a "Quickstart" section to be filled.
- `pip install -e ".[dev]"` → verify import: `python -c "import histoRAG"`.
- Smoke test OpenSlide: `python -c "import openslide; print(openslide.__version__)"` — fail here triggers Windows DLL debug before proceeding.
- Smoke test FAISS: `python -c "import faiss; idx=faiss.IndexFlatIP(8); print(idx.ntotal)"` → `0`.
- Commit: `init: scaffold histoRAG package, deps, and CI-free skeleton`.
- **Acceptance**: `pip install -e .` succeeds; `pytest` runs (even with zero tests passing); GitHub repo visible with at least one commit.

### Step 2 — HANCOCK data pull (Day 1–2, ~4 h, parallel with Step 3)
- Visit `www.hancock.research.fau.eu`; register/log in; locate slide access portal.
- Pick 5–10 slides aiming for class label balance (exact IDs written into `configs/phase0_mvp.yaml` → `data.slide_ids`).
- `scripts/download_hancock.py`: reads config, downloads slides to `data/raw/<slide_id>.svs` (or `.tiff`/`.mrxs` depending on HANCOCK format) with integrity check (SHA256 if provided; otherwise file-size sanity).
- Record label source: is HANCOCK label slide-level (one label per WSI), ROI-level (polygon annotations), or patch-level? Fill in `label_source` + `label_map` in config.
- Update `README.md` dataset section with chosen slide IDs + license note (no redistribution).
- Commit: `data: HANCOCK 5-slide subset selection + download script`.
- **Acceptance**: `data/raw/` contains all slide files; `openslide.OpenSlide(path).dimensions` succeeds for each; label source documented.

### Step 3 — Tiling (Day 2–3, ~6 h)
- **`wsi_loader.py`**: `WSI(path)` class exposing `.best_level_for_mag(target_mag) → int` using `slide.properties["openslide.objective-power"]` (or fallback to `openslide.mpp-x` → derive magnification from MPP). Handles missing metadata with warning + fallback to level 0.
- **`tiler.py`**:
  - Compute thumbnail at `thumb_downsample` factor.
  - Convert RGB → HSV, take saturation channel, apply Otsu threshold → binary tissue mask.
  - Generate candidate grid at `(patch_size, stride)` on the chosen level; for each candidate, project bounding box into thumbnail space, compute fraction of tissue pixels; keep if ≥ `min_tissue_frac`.
  - Extract kept patches via `slide.read_region(location, level, size)`; convert RGBA → RGB; save as PNG under `data/patches/<slide_id>/<x>_<y>.png`.
  - Cap at `max_patches_per_slide`; subsample with seeded RNG if over.
- **Manifest**: `data/patches/manifest.parquet` with columns `patch_id, slide_id, x, y, level, magnification, label`. Patch ID = `f"{slide_id}__{x:06d}_{y:06d}"`.
- **Unit test** (`tests/test_tiler.py`):
  - Create synthetic 2048×2048 RGB "slide" written via `pyvips` or `tifffile` (or pre-committed tiny fixture). Assert tissue filter discards empty quadrant; assert patch count deterministic across reruns.
- Run tiling on the 5–10 slides: `python scripts/tile_wsis.py --config configs/phase0_mvp.yaml`.
- Commit: `feat(data): WSI tiler with Otsu-HSV tissue filter + parquet manifest`.
- **Acceptance**: `data/patches/manifest.parquet` has N rows (5–50k range); each slide has ≥100 patches; spot-check tile images in a notebook cell are actual tissue (not whitespace).

### Step 4 — Encoder abstraction + CLIP (Day 3, ~4 h)
- **`encoders/base.py`**:
  ```python
  class Encoder(ABC):
      name: str
      embed_dim: int
      @abstractmethod
      def encode(self, images: list[PIL.Image.Image]) -> np.ndarray: ...
      def encode_batched(self, images, batch_size) -> np.ndarray: ...  # default impl
  ```
- **`encoders/clip.py`**:
  - Load via `open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')`.
  - Store both vision + text modules; MVP only calls `encode_image`. Expose `encode_text(list[str])` method that raises `NotImplementedError("Pro-1 hook; not used in Phase 0")` — toggled on by Phase 1.
  - Return L2-normalized `(B, 512)` float32 numpy array.
- **`encoders/registry.py`**: `ENCODERS = {"clip-vitb16": ClipEncoder}`; `get_encoder(name, **cfg)` factory.
- **Unit test** (`tests/test_encoder.py`): fixed seed + fixed input tensor → `allclose` between two invocations; output shape `(B, 512)`; output L2-normalized (row norms ≈ 1.0).
- Commit: `feat(encoders): Encoder ABC + CLIP ViT-B/16 impl`.
- **Acceptance**: passes unit test; benchmark prints ≥300 patches/sec on GTX 1650 at batch 32.

### Step 5 — UNI2-h stub (Day 3, ~1 h) — Phase 1 hook only
- **`encoders/uni.py`**: `UNIEncoder` using the official `timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **kwargs)` pattern per HF docs. Requires `huggingface-cli login` first (token already set up since access granted).
- Register in `registry.py`: `ENCODERS["uni2h"] = UNIEncoder`.
- **Import-only test**: `tests/test_encoder.py::test_uni_importable` — checks `get_encoder("uni2h")` doesn't error on *class* construction but skips actual weight load unless `RUN_SLOW=1` env var set.
- Commit: `feat(encoders): UNI2-h stub registered for Phase 1`.
- **Acceptance**: `from histoRAG.encoders.uni import UNIEncoder` succeeds; registry lookup works.

### Step 6 — FAISS index (Day 4, ~3 h)
- **`index/base.py`**:
  ```python
  class VectorIndex(ABC):
      @abstractmethod
      def add(self, embeddings: np.ndarray, ids: np.ndarray) -> None: ...
      @abstractmethod
      def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...
      @abstractmethod
      def save(self, path: str) -> None: ...
      @classmethod
      @abstractmethod
      def load(cls, path: str) -> "VectorIndex": ...
  ```
- **`index/faiss_index.py`**: `FaissFlatIP` wraps `faiss.IndexIDMap2(faiss.IndexFlatIP(dim))`. `add` expects already-normalized vectors; `search` returns `(similarities, patch_ids)` tuple. `save`/`load` use `faiss.write_index` / `read_index`.
- **Unit test** (`tests/test_index.py`): 100 random 512-d vectors → add → query each → recall@1 == 1.0; round-trip save/load preserves behavior.
- Commit: `feat(index): FAISS flat IP index with ID mapping + persistence`.
- **Acceptance**: unit test green; save/load round-trip works.

### Step 7 — Evaluation protocol (Day 4, ~3 h)
- **`eval/protocol.py`**:
  - `stratified_within_slide(manifest, query_frac, seed)` → returns `(query_ids, gallery_ids)`, balanced by label per slide.
  - Alternate `slide_leave_out(manifest, held_out_slides, seed)` → stub for Phase 2 Pro-3; tested but unused in MVP.
- **`eval/metrics.py`**:
  - `top_k_accuracy(retrieved_labels: np.ndarray, query_labels: np.ndarray, k: int) → float`: fraction of queries where `query_label ∈ retrieved_labels[:k]`.
  - `mean_average_precision(retrieved_labels, query_labels, k: int) → float`.
  - Both tested against hand-computed toy inputs.
- **Unit test** (`tests/test_eval.py`): known 5-query / 20-gallery toy dataset with hand-computed expected metrics → assert match.
- Commit: `feat(eval): retrieval split + top-k + mAP@k metrics`.
- **Acceptance**: metrics match toy-input hand computations; split is reproducible given seed.

### Step 8 — End-to-end pipeline + logging (Day 5, ~5 h)
- **`utils/config.py`**: `load_config(path) → dict`; `canonicalize(cfg) → str` (sorted-keys JSON); `hash_config(cfg) → str` (sha256).
- **`utils/logging.py`**:
  - `append_experiment_row(cfg, metrics, timings) → uid`: generates UID, copies config to `configs/runs/<uid>.yaml` (immutable), appends row to `experiments/experiments.csv`.
  - Idempotent: refuses to overwrite an existing UID.
- **`utils/seeds.py`**: `set_all_seeds(seed)` sets Python `random`, NumPy, PyTorch CPU + CUDA, and `torch.backends.cudnn.deterministic = True`.
- **`scripts/run_mvp.py --config CFG --seed N`**:
  1. Load + validate config; override seed from CLI.
  2. If `data/patches/manifest.parquet` missing → call `tile_wsis.main()` internally.
  3. Load encoder from registry.
  4. Encode all patches in `manifest` → `embeddings.npy` cached under `data/indexes/<config_hash>/`.
  5. Build FAISS index; save to disk.
  6. Split queries/gallery per eval protocol (seeded).
  7. For each query, search top-k → compute metrics.
  8. Append row to `experiments.csv`; write interpretation template to `EXPERIMENT_LOG.md`.
- **Run for seeds 42, 123, 2024** → 3 rows in CSV.
- **Interpretation** written manually in `EXPERIMENT_LOG.md`: format
  ```markdown
  ### 20260422_001_clip-vitb16_faiss-flatip_hancock5_seed42
  - top-5 accuracy: 0.XX (random baseline 0.YY)
  - Interpretation: CLIP (natural-image pretrain) achieves ZZZ above random on HANCOCK,
    suggesting transferable low-level features; Phase 1 will test whether UNI2-h
    (histopath-SSL) raises this further.
  ```
- Commit: `run: phase-0 MVP baseline (CLIP ViT-B/16, seeds 42/123/2024)`.
- **Acceptance**: 3 CSV rows; 3 config snapshots; `EXPERIMENT_LOG.md` has 3 interpretation entries; re-running any seed produces identical metrics (reproducibility).

### Step 9 — Streamlit demo (Day 6, ~4 h)
- **`src/histoRAG/viz/streamlit_app.py`**:
  - Sidebar: config file path selector; top-k slider (1–20); "Use random gallery patch" button.
  - Main: file uploader accepts PNG/JPEG; displays query image + metadata if a gallery patch was picked.
  - Inference: load cached embeddings + FAISS index via `@st.cache_resource`; encode query; search top-k; display as grid with caption `f"{slide_id} ({x},{y}) · label={label} · sim={score:.3f}"`.
  - Error handling: if no index exists, show button "Build index now" that invokes `run_mvp.py`.
- Add a demo script: `scripts/run_demo.cmd` (Windows) → `streamlit run src/histoRAG/viz/streamlit_app.py`.
- Pre-cache: run the demo once before rehearsal so model weights are on disk.
- Commit: `feat(viz): Streamlit retrieval demo UI`.
- **Acceptance**: demo opens, upload patch → grid displays; offline run succeeds on laptop wifi off.

### Step 10 — Presentation prep (Day 7, ~4 h)
- 7-slide deck (PowerPoint/Keynote):
  1. Problem + motivation (pathology atlas, CBIR background)
  2. Data (HANCOCK subset, label granularity)
  3. Method (pipeline diagram: WSI → Otsu → CLIP → FAISS → top-k)
  4. Results table (mean ± std over 3 seeds, top-1/5/10, mAP@10, random baseline)
  5. Retrieval examples (2–3 query → top-5 grid screenshots)
  6. Demo (screenshot backup of Streamlit in case live fails)
  7. Phase 1 plan (CLIP vs UNI2-h, first controlled experiment already scoped)
- Rehearse demo with WiFi off; time at 12 min to leave 3 min Q&A buffer (total slot 15 min).
- Git tag: `git tag phase0-mvp && git push --tags`.
- **Acceptance**: rehearsal hits ≤12 min; demo works offline; results table present.

### Step 11 — Buffer / contingency (Day 8, ~4 h)
- Reserved for: slow HANCOCK download, OpenSlide install debugging, unexpected low metrics requiring param tweak (one controlled change, still logged), rehearsal iteration.
- **If ahead of schedule**: start Phase 1 groundwork — request Virchow2/Phikon-v2 HF access, sketch `configs/phase1_uni.yaml`.

---

## Critical files to create (reuse check: nothing relevant exists in repo yet)

Repo currently contains only `docs/` and `CLAUDE.md` — no Python source, no reusable utilities. All files listed in the repo-structure section above are new.

**Files referenced by this plan**:

| Path | Purpose |
|---|---|
| `pyproject.toml` | Deps + package metadata |
| `configs/phase0_mvp.yaml` | Canonical MVP config (single source of truth) |
| `src/histoRAG/data/{wsi_loader,tiler,dataset}.py` | WSI I/O + tiling + torch Dataset |
| `src/histoRAG/encoders/{base,clip,uni,registry}.py` | Encoder abstraction + CLIP + UNI stub |
| `src/histoRAG/index/{base,faiss_index}.py` | Vector-index abstraction + FAISS impl |
| `src/histoRAG/eval/{metrics,protocol}.py` | Metrics + query/gallery split |
| `src/histoRAG/viz/streamlit_app.py` | Demo UI |
| `src/histoRAG/utils/{config,logging,seeds}.py` | Config loader, experiment logger, seed helper |
| `scripts/{download_hancock,tile_wsis,embed_patches,evaluate,run_mvp}.py` | CLI entry points |
| `tests/test_{tiler,encoder,index,eval}.py` | Unit tests |
| `EXPERIMENT_LOG.md` | Narrative log alongside CSV |
| `PLAN.md` | Copy of this document into project root post-approval |

---

## Verification (end-to-end MVP acceptance)

1. **Install deps**: `pip install -r requirements.txt` → `python -c "import histoRAG; print(histoRAG.__version__)"` → `0.1.0`.
2. **Tests**: `pytest tests/ -q` → all green (tests are gitignored, run locally only).
3. **Pipeline**: `python pipeline.py --config configs/phase0_mvp.yaml --seed 42` → tiles (if not cached), embeds, builds FAISS index, appends 1 row to `experiments/experiments.csv`.
4. **Reproducibility**: re-run with same seed → identical metrics bit-exact (within FAISS tie-breaking tolerance).
5. **Seeds**: run for 42, 123, 2024 → 3 rows; mean ± std reported in `EXPERIMENT_LOG.md`.
7. **Grading-rubric dry-check** (Code pillar, 20 P possible):
   - Reproducibility (5 P): fixed seeds + config snapshots + `config_hash` + `git_commit` in CSV ✓
   - Structure & docs (5 P): flat `histoRAG/` package, README quickstart ✓
   - Evaluation correctness (5 P): unit-tested metrics, documented protocol, random baseline ✓
   - Experimental log (5 P): `experiments.csv` + `configs/runs/*.yaml` + `EXPERIMENT_LOG.md` narrative ✓

---

## Phase-aware future-proofing (designed in MVP, executed later)

| Phase | Extension | MVP hook already in place |
|---|---|---|
| Phase 1 — Formalize | CLIP vs UNI2-h (+ ResNet50, OpenCLIP) ablation | Add new encoder class to `histoRAG/embed.py`; swap `encoder.name` in config |
| Phase 1 — Index ablation | FAISS Flat vs IVF vs HNSW | Add new index class to `histoRAG/embed.py`; swap `index.name` in config |
| Phase 2 — Pro-1 text query | CLIP text tower | Add `encode_text()` to `ClipEncoder` in `histoRAG/embed.py` |
| Phase 2 — Pro-3 cross-slide | Leave-slide-out eval split | `slide_leave_out()` already in `histoRAG/retrieve.py`, unused in Phase 0 |
| Phase 3 — Pro-2 LLM descriptions | Feed top-k context → lightweight LLM | New file `histoRAG/describe.py`; doesn't touch existing code |
| Future — pip-installable | Package for distribution | Add `pyproject.toml` back; zero code changes needed |

---

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| OpenSlide fails to install on Windows | Use `openslide-bin` pip wheel (bundles DLL). Smoke-test on Day 1 before writing more code. |
| HANCOCK slide download slow/blocked | Start download as background task Day 1, parallel with repo scaffold. If portal rate-limits, pick 5 slides instead of 10. |
| 4 GB VRAM OOM on CLIP inference | Start batch=32; halve on OOM. Empirically CLIP-B/16 works at batch 32 on 4 GB. |
| FAISS GPU unreliable on Windows | Use `faiss-cpu` — fine for ≤100 k vectors with exact IP search. |
| HANCOCK labels unclear / only slide-level | Fallback eval = within-slide nearest-neighbor consistency (spatial-proximity pseudo-labels). Decide at end of Step 2 before coding Step 7. |
| Streamlit crashes during live demo | Pre-record a 30 s screencast as backup; include static retrieval-example grids in slides. |
| UNI2-h pushes 4 GB VRAM past limit in Phase 1 | Out of MVP scope; Phase 1 fallback = Colab T4 with `device: auto` config flag. |
| Scope creep toward Pro-tasks pre-MVP | Hard rule: do not implement Pro-1/2/3 before Phase 0 acceptance criteria met. |

---

## Open questions to resolve during Step 1–2 (not blocking plan approval)

1. **HANCOCK label granularity** — slide-level, ROI polygons, or patch-level? Affects eval protocol choice in Step 7.
2. **HANCOCK total download size** for chosen 10 slides — influences whether Step 2 needs to run overnight.
3. **Which openslide magnification level** corresponds to 20× on the chosen slides? Confirmed per-slide at end of Step 3.

---

## Definition of done (Phase 0 MVP)

- [ ] Repo on GitHub, pip-installable, `pytest` green
- [ ] ≥3 rows in `experiments/experiments.csv` (3 seeds, CLIP ViT-B/16 baseline)
- [ ] `configs/runs/` has matching immutable config snapshots per row
- [ ] `EXPERIMENT_LOG.md` has interpretation entries referencing UIDs
- [ ] Streamlit demo runs locally from a single command, offline
- [ ] 7-slide presentation deck drafted + rehearsed ≤12 min
- [ ] `README.md` has quickstart, dataset note, reproduction instructions, Phase 1 outline
- [ ] `PLAN.md` (this document) copied into project root, git-tracked
- [ ] `git tag phase0-mvp` pushed
