#!/bin/bash -l
# ============================================================================
# SLURM job — H1 hypothesis, UNI encoder only (k=2 and k=8)
#
# Run this after downloading and unzipping WSI_UNI_encodings.zip into
#   $WORK/hancock/embeddings/UNI/
#
# Usage:
#   sbatch hypotheses/H1_tumour_classification/run_h1_uni_hpc.sh
# ============================================================================

#SBATCH --job-name=h1_uni
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.log
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1

# ── Paths ────────────────────────────────────────────────────────────────────
REPO="/home/hpc/vlbi/vlbi113v/HistoRag-BIMAP"
VENV="$REPO/hpcenv"
EMB="$WORK/hancock/embeddings"
GEO="$WORK/hancock/WSI_PrimaryTumor_Annotations"
# ─────────────────────────────────────────────────────────────────────────────

echo "=========================================="
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME  CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo "=========================================="

module load python/pytorch2.6py3.12

source "$VENV/bin/activate"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$REPO"

echo ""
echo "── Verifying UNI embeddings directory ──"
ls "$EMB/UNI/WSI_PrimaryTumor/" 2>/dev/null || echo "ERROR: UNI embeddings not found at $EMB/UNI/"

echo ""
echo "── exp01 k=2  encoder=uni2h  $(date) ──"
python hypotheses/H1_tumour_classification/exp01_kmeans_k2/run.py \
    --full \
    --encoder uni2h \
    --embeddings-root "$EMB" \
    --geojson-dir "$GEO"

echo ""
echo "── exp02 k=8  encoder=uni2h  $(date) ──"
python hypotheses/H1_tumour_classification/exp02_overcluster_assign/run.py \
    --full \
    --encoder uni2h \
    --embeddings-root "$EMB" \
    --geojson-dir "$GEO"

echo ""
echo "=========================================="
echo "H1 UNI complete. $(date)"
echo "Results in:"
find "$REPO/hypotheses/H1_tumour_classification" -name "summary_uni2h.json" | sort
echo "=========================================="
