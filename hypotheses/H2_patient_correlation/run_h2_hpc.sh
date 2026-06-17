#!/bin/bash -l
# ============================================================================
# SLURM job — H2 hypothesis (site clustering + tumour similarity)
#
# exp01: mean-pool all patches per patient → UMAP + correlation heatmap
#        Expected: block-diagonal by anatomical site
# exp02: filter to tumour patches only → UMAP (sites mixed) + correlation map
#        Expected: uniformly high similarity across sites
#
# Writes results to:
#   hypotheses/H2_patient_correlation/exp01_site_clustering/outputs/<encoder>/
#   hypotheses/H2_patient_correlation/exp02_tumour_similarity/outputs/<encoder>/
#
# Usage:
#   sbatch hypotheses/H2_patient_correlation/run_h2_hpc.sh
#
# After completion, commit the output files and push to GitHub.
# ============================================================================

# ── SLURM directives ─────────────────────────────────────────────────────────
#SBATCH --job-name=h2_patient
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.log             # e.g. h2_patient_1234567.log
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1

# ── CHANGE: paths specific to your HPC account ───────────────────────────────
REPO="/home/hpc/vlbi/vlbi113v/HistoRag-BIMAP"          # cloned repo location
VENV="$REPO/hpcenv"                                    # virtualenv created during HPC setup
EMB="$WORK/hancock/embeddings"                          # uploaded embeddings root (CLIP/, CONCH/, UNI/)
GEO="$WORK/hancock/WSI_PrimaryTumor_Annotations"       # uploaded geojson annotations dir
# ─────────────────────────────────────────────────────────────────────────────

echo "=========================================="
echo "Job: $SLURM_JOB_NAME  ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME  CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"
echo "=========================================="

# ── Environment ──────────────────────────────────────────────────────────────
module load python/pytorch2.6py3.12          # CHANGE: match your cluster's Python module

source "$VENV/bin/activate"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$REPO"

# ── Run H2 for selected encoders ─────────────────────────────────────────────
# Run sequentially to avoid competing for RAM.
# Peak RAM: ~17 GB (CONCH 8.2M×512 patches loaded for exp02 tumour labelling).

for ENC in conch uni2h; do
    echo ""
    echo "── exp01 site_clustering  encoder=$ENC  $(date) ──"
    python hypotheses/H2_patient_correlation/exp01_site_clustering/run.py \
        --encoder "$ENC" \
        --embeddings-root "$EMB"

    echo ""
    echo "── exp02 tumour_similarity  encoder=$ENC  $(date) ──"
    python hypotheses/H2_patient_correlation/exp02_tumour_similarity/run.py \
        --encoder "$ENC" \
        --embeddings-root "$EMB" \
        --geojson-dir "$GEO"
done

echo ""
echo "=========================================="
echo "H2 complete. $(date)"
echo "Outputs in:"
find "$REPO/hypotheses/H2_patient_correlation" -name "summary.json" | sort
echo "=========================================="
