# HackDay2 — Single-WSI HDBSCAN Pipeline

Given one WSI, this pipeline patches it, embeds with CONCH, runs HDBSCAN, and produces two 2D UMAP plots — one coloured by cluster, one by ground-truth tumour annotation.

---

## Step 1 — Clone the repo

```bash
git clone https://github.com/Vibhugupta10616/HistoRag-BIMAP.git
cd HistoRag-BIMAP
```

---

## Step 2 — Install dependencies

```bash
python -m venv .bimap
.bimap\Scripts\activate
```

Install PyTorch — pick **one** depending on your machine:
```bash
# GPU (CUDA 12.4) — recommended
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# CPU only
pip install torch==2.6.0 torchvision==0.21.0
```

Then install the rest:
```bash
pip install -r requirements.txt
```

---

## Step 3 — Download data from Google Drive

**Drive link:** https://drive.google.com/drive/folders/1UR_ykbdqf89zng0bpp5aUi1dPvYuyFAo?usp=sharing

```
Google Drive/
├── conch/
│   └── pytorch_model.bin              (766 MB)  — Option B only
├── wsi/
│   ├── PrimaryTumor_HE_010.svs                  — Option B only
│   └── PrimaryTumor_HE_XXX.svs
├── annotations/
│   ├── PrimaryTumor_HE_010.geojson              — both options (ground-truth plot)
│   └── PrimaryTumor_HE_XXX.geojson
└── embeddings/
    ├── PrimaryTumor_HE_010.h5                   — Option A only
    └── PrimaryTumor_HE_XXX.h5
```

Download the entire folder into the **repo root** and keep the name `Weights and WSIs`:
```
HistoRag-BIMAP/
├── Weights and WSIs/     ← downloaded Drive folder goes here
│   ├── conch/
│   ├── wsi/
│   ├── annotations/
│   └── embeddings/
├── Hackday2/
└── ...
```
You do not need to download everything — see Step 5 for what each option requires.

---

## Step 4 — Configure

Open `Hackday2/config.yaml` and set **only these two lines**:

```yaml
data_root: "Weights and WSIs"        # Drive folder downloaded into repo root — do not rename
slide_id:  "PrimaryTumor_HE_010"     # which slide to run — change to switch WSIs
```

All other paths (WSI, annotations, CONCH weights, embeddings) are resolved automatically from these two values.

---

## Step 5 — Run

### Option A — Fast: start from pre-computed embeddings (recommended)

> Download from Drive: `embeddings/<slide_id>/` + `annotations/`
> Time: ~5-10 min | GPU: not required

```bash
python Hackday2/pipeline.py --skip-embed
```

Embeddings are picked up from `data_root/embeddings/` automatically — no extra config needed.

---

### Option B — Full pipeline: WSI -> patches -> embed -> HDBSCAN -> UMAP

> Download from Drive: `wsi/<slide_id>.svs` + `conch/` + `annotations/`
> Time: ~15-25 min (GPU) / ~60-90 min (CPU)

Install the CONCH package first (public repo, no login needed):
```bash
pip install git+https://github.com/mahmoodlab/CONCH.git
```

Then run:
```bash
python Hackday2/pipeline.py
```

---

### Replot from cache (fastest, after any completed run)

```bash
python Hackday2/pipeline.py --plots-only
```

---

### Quick test with fewer patches

To get results in ~3 min, set `max_patches` in `config.yaml` before running:
```yaml
patching:
  max_patches: 1000
```
Set back to `null` for the full slide.

---

## Outputs

Saved to `Hackday2/outputs/<slide_id>/`:

```
umap_by_cluster.png        <- UMAP coloured by HDBSCAN cluster label
umap_by_groundtruth.png    <- UMAP coloured by tumour / other (requires annotations)
summary.json               <- patch count, cluster count, params used
cache/
    embeddings.h5
    umap_2d.npy
    cluster_labels.npy
    gt_labels.npy
```

---

## Time reference

| Step | GPU | CPU |
|---|---|---|
| Patching | ~5-10 min | ~5-10 min |
| CONCH embedding | ~3-8 min | ~40-90 min |
| HDBSCAN + UMAP | ~5 min | ~10 min |
| **Option A total** | **~5-10 min** | **~10 min** |
| **Option B total** | **~15-25 min** | **~60-110 min** |
