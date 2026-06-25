#!/bin/bash -l
# ============================================================================
# SLURM job — H1 exp03 HDBSCAN Patch Clustering (Q3)
#
# Two passes over all slides per encoder:
#   Pass 1: IncrementalPCA fit (streaming, low peak RAM)
#   Pass 2: PCA transform + HDBSCAN on all ~8M patches + UMAP visualisation
#
# Outputs:
#   hypotheses/H1_tumour_classification/exp03_hdbscan_clustering/outputs/{encoder}/
#
# Usage:
#   sbatch hypotheses/H1_tumour_classification/exp03_hdbscan_clustering/run_h1_exp03_hpc.sh
# ============================================================================

#SBATCH --job-name=h1_exp03_hdbscan
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=%x_%j.log
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1

# ── CHANGE: paths specific to your HPC account ───────────────────────────────
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

# HDBSCAN uses joblib internally — give it all allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$REPO"

for ENC in conch uni2h; do
    echo ""
    echo "── H1 exp03  encoder=$ENC  $(date) ──"
    python hypotheses/H1_tumour_classification/exp03_hdbscan_clustering/run.py \
        --full \
        --encoder "$ENC" \
        --embeddings-root "$EMB" \
        --geojson-dir "$GEO"
done

echo ""
echo "=========================================="
echo "H1 exp03 complete. $(date)"
find "$REPO/hypotheses/H1_tumour_classification/exp03_hdbscan_clustering/outputs" \
     -name "summary.json" | sort
echo "=========================================="
