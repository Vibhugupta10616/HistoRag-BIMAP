#!/bin/bash -l
# ============================================================================
# SLURM job — H2 exp02 Tumour Patch Similarity (Q2) — full dataset on HPC
#
# Collects ALL tumour + non-tumour patches per encoder, runs UMAP and
# cosine similarity distribution for both classes, and writes outputs to:
#   hypotheses/H2_patient_correlation/exp02_tumour_similarity/outputs/{encoder}/tumour/
#   hypotheses/H2_patient_correlation/exp02_tumour_similarity/outputs/{encoder}/nontumour/
#
# Memory estimates (full non-tumour set):
#   CONCH: ~17 GB  (7.8M × 512d)
#   UNI:   ~34 GB  (7.8M × 1024d)
#   → 64 GB requested to cover both encoders' peak + UMAP workspace
#
# Usage:
#   sbatch hypotheses/H2_patient_correlation/exp02_tumour_similarity/run_h2_exp02_hpc.sh
#
# After completion, commit output PNGs and summary.json, then push to GitHub.
# ============================================================================

# ── SLURM directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=h2_exp02_sim
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$REPO"

# Run encoders sequentially to avoid competing for RAM
for ENC in conch uni2h; do
    echo ""
    echo "── H2 exp02  encoder=$ENC  $(date) ──"
    python hypotheses/H2_patient_correlation/exp02_tumour_similarity/run.py \
        --full \
        --encoder "$ENC" \
        --embeddings-root "$EMB" \
        --geojson-dir "$GEO"
done

echo ""
echo "=========================================="
echo "H2 exp02 complete. $(date)"
echo "Outputs:"
find "$REPO/hypotheses/H2_patient_correlation/exp02_tumour_similarity/outputs" \
     -name "summary.json" | sort
echo "=========================================="
