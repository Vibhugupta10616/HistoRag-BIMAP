#!/bin/bash -l
# ============================================================================
# SLURM job — H1 hypothesis (KMeans k=2 and k=8) on COMPLETE dataset
#
# Fits MiniBatchKMeans on ALL patches for each encoder (no subsampling).
# Writes results to:
#   hypotheses/H1_tumour_classification/exp01_kmeans_k2/outputs/<encoder>/
#   hypotheses/H1_tumour_classification/exp02_overcluster_assign/outputs/<encoder>/
#
# Usage:
#   sbatch hypotheses/H1_tumour_classification/run_h1_hpc.sh
#
# After completion, commit the output JSON files and push to GitHub.
# ============================================================================

# ── SLURM directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=h1_full
#SBATCH --partition=singlenode          # CHANGE: CPU/high-mem partition name
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G                       # CLIP/CONCH: ~17 GB array; k=8 needs headroom
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.log             # e.g. h1_full_1234567.log

# ── CHANGE: paths specific to your HPC account ───────────────────────────────
REPO="$HOME/HistoRag-BIMAP"                           # cloned repo location
VENV="$REPO/HPC/hpcenv"                               # virtualenv created during HPC setup
EMB="$WORK/hancock/embeddings"                         # uploaded embeddings root (CLIP/, CONCH/, UNI/)
GEO="$WORK/hancock/WSI_PrimaryTumor_Annotations"      # uploaded geojson annotations dir
# ─────────────────────────────────────────────────────────────────────────────

echo "=========================================="
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME  CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo "=========================================="

# ── Environment ──────────────────────────────────────────────────────────────
module load python/pytorch2.6py3.12          # CHANGE: match your cluster's Python module

source "$VENV/bin/activate"

# Install H1-specific deps missing from HPC/requirements.txt (fast if cached)
pip install --quiet -r "$REPO/hypotheses/H1_tumour_classification/requirements_hpc_h1.txt"

# Let sklearn use all allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$REPO"

# ── Run H1 for all encoders ───────────────────────────────────────────────────
# Each encoder-experiment pair is run sequentially to avoid competing for RAM.
# Peak RAM per run: ~17 GB (CLIP/CONCH 8.2M×512) or ~6 GB (UNI 2M×1536).

for ENC in clip-vitb16 conch uni2h; do
    echo ""
    echo "── exp01 k=2  encoder=$ENC  $(date) ──"
    python hypotheses/H1_tumour_classification/exp01_kmeans_k2/run.py \
        --full \
        --encoder "$ENC" \
        --embeddings-root "$EMB" \
        --geojson-dir "$GEO"

    echo ""
    echo "── exp02 k=8  encoder=$ENC  $(date) ──"
    python hypotheses/H1_tumour_classification/exp02_overcluster_assign/run.py \
        --full \
        --encoder "$ENC" \
        --embeddings-root "$EMB" \
        --geojson-dir "$GEO"
done

echo ""
echo "=========================================="
echo "H1 complete. $(date)"
echo "Results in:"
find "$REPO/hypotheses/H1_tumour_classification" -name "summary_*.json" | sort
echo "=========================================="
