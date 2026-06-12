# HistoRAG — HPC Embedding Pipeline

This branch contains everything needed to download HANCOCK WSI zip files on FAU TinyGPU,
extract patches, embed them with CLIP or CONCH, and zip the results for local download.

**This branch is independent of `main`.** It has no dependency on `histoRAG/`, `hypotheses/`, or any other project folder.

---

## Workflow overview

```
parallel_download.py        download WSI zip from FAU data server
        ↓
watch_and_submit.sh         (optional) poll until download finishes, then auto-submit
        ↓
embed_job.sh / embed_job_work.sh    SLURM job: unzip → tile → embed → zip h5 files
        ↓
scp tinygpu:<zip>  D:/College/Sem_5/HistoRag-BIMAP/data/    download to local machine
```

---

## One-time setup

```bash
module load python/pytorch2.6py3.12
python -m venv ~/HistoRag-BIMAP/HPC/hpcenv
source ~/HistoRag-BIMAP/HPC/hpcenv/bin/activate

# Install PyTorch with CUDA (check cluster CUDA version first)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r ~/HistoRag-BIMAP/HPC/requirements.txt
```

**CONCH encoder only** (optional — needed for `ENCODER="conch"`):
```bash
pip install git+https://github.com/mahmoodlab/CONCH
huggingface-cli login    # request access at hf.co/MahmoodLab/conch first
```

---

## Step 1 — Download a tissue zip

Edit the three `# CHANGE:` lines in [parallel_download.py](parallel_download.py):

| Variable | What to set |
|---|---|
| `URL` | FAU download URL for the target tissue zip |
| `TISSUE_LOWER` | lowercase tissue name (see table below) |
| `EXPECTED_SIZE` | exact byte size of the zip (see table below) |

Then run in a screen session so it survives logout:
```bash
screen -S download
python ~/HistoRag-BIMAP/HPC/parallel_download.py
# Ctrl+A then D to detach
```

Supports resume — re-run the same command if the job is interrupted.

---

## Step 2 — Submit the embedding job

Edit the two `# CHANGE:` lines at the top of the script:

| Variable | What to set |
|---|---|
| `TISSUE` | tissue name (see table below) |
| `ENCODER` | `clip` or `conch` |

**A100 partition** (recommended — faster, more VRAM):
```bash
sbatch ~/HistoRag-BIMAP/HPC/embed_job.sh
```

**work partition** (RTX 2080 Ti / 3080, 10–11 GB VRAM — use if A100 queue is long):
```bash
sbatch ~/HistoRag-BIMAP/HPC/embed_job_work.sh
```

Monitor the job:
```bash
squeue -u $USER
tail -f ~/embed_<job_id>.log
```

The job automatically zips the output h5 files at the end and prints the `scp` command to download them.

---

## Optional — Auto-submit after download

[watch_and_submit.sh](watch_and_submit.sh) polls every 5 minutes until the zip download is complete, then submits `embed_job.sh` automatically. Edit the two `# CHANGE:` lines (`TISSUE` and `ENCODER`), then:

```bash
screen -S watcher
bash ~/HistoRag-BIMAP/HPC/watch_and_submit.sh
# Ctrl+A then D to detach
```

---

## Tissue reference

| Tissue | Folder name | Zip size (bytes) |
|---|---|---|
| Larynx | `larynx` | 314,557,232,221 |
| Hypopharynx | `hypopharynx` | 213,184,293,362 |
| Oral_Cavity | `oral_cavity` | 429,916,307,251 |
| Oropharynx (Part 1) | `oropharynx1` | 516,446,064,041 |
| Oropharynx (Part 2) | `oropharynx2` | 528,375,773,776 |

---

## Output structure

```
$WORK/hancock/embeddings/
  CLIP/
    Primary_Tumour/
      <Tissue>/
        h5_files/
          PrimaryTumor_HE_<id>.h5
          <Tissue>_CLIP_embeddings.zip   ← download this
  CONCH/
    Primary_Tumour/
      <Tissue>/
        h5_files/
          PrimaryTumor_HE_<id>.h5
          <Tissue>_CONCH_embeddings.zip  ← download this
```

Each `.h5` file contains:

| Dataset | Shape | Description |
|---|---|---|
| `embeddings` | `(N, 512)` float32 | L2-normalised patch vectors |
| `patch_ids` | `(N,)` bytes | unique patch identifier |
| `x` | `(N,)` int32 | top-left x coordinate (level-0 pixels) |
| `y` | `(N,)` int32 | top-left y coordinate (level-0 pixels) |

---

## Files

| File | Purpose |
|---|---|
| `parallel_download.py` | Multi-threaded HTTP range downloader with resume |
| `watch_and_submit.sh` | Watcher that auto-submits the SLURM job after download |
| `embed_job.sh` | SLURM job for A100 partition (CLIP or CONCH) |
| `embed_job_work.sh` | SLURM job for work partition (smaller VRAM) |
| `hpc_pipeline.py` | Main pipeline: tile WSIs → embed patches → save h5 files |
| `encoders.py` | CLIP and CONCH encoder wrappers |
| `wsi_tiler.py` | WSI patch extractor with Otsu-HSV tissue masking |
| `requirements.txt` | Python dependencies (excludes torch — installed separately) |
