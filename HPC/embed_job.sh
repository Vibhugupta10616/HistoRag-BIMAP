#!/bin/bash
# SLURM job: unzip WSI archive -> tile -> embed (CLIP) -> zip per-patient embeddings.
# Automatically submitted by watch_and_submit.sh once the download completes.
# Submit manually with: sbatch ~/HistoRag-BIMAP/HPC/embed_job.sh

#SBATCH --job-name=larynx_embed
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/hpc/vlbi/vlbi113v/scripts/embed_%j.log

set -e

log() { echo "[$(date +%H:%M:%S)] $1"; }

ZIP="$WORK/hancock/zips/WSI_PrimaryTumor_Larynx.zip"
WSI_DIR="$WORK/hancock/wsi"
PATCHES_DIR="$WORK/hancock/patches"
OUT_DIR="$WORK/hancock/embeddings"
HPC_DIR="$HOME/HistoRag-BIMAP/HPC"
VENV="$HPC_DIR/hpcenv"

# --- Step 1: Extract zip ---
log "Extracting $ZIP ..."
mkdir -p "$WSI_DIR"
unzip -o "$ZIP" -d "$WSI_DIR"
log "Extraction complete:"
ls "$WSI_DIR"

# --- Step 2: Activate HPC venv ---
log "Activating virtualenv $VENV ..."
source "$VENV/bin/activate"

# Confirm GPU is visible
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# --- Step 3: Tile + embed ---
log "Starting pipeline ..."
python "$HPC_DIR/hpc_pipeline.py" \
    --wsi_dir     "$WSI_DIR" \
    --out_dir     "$OUT_DIR" \
    --patches_dir "$PATCHES_DIR" \
    --label       larynx \
    --max_patches 5000 \
    --batch_size  64

# --- Step 4: Zip embeddings for local download ---
log "Zipping embeddings ..."
cd "$WORK/hancock"
zip -r embeddings.zip embeddings/
log "Done: $WORK/hancock/embeddings.zip"
log "Download with: scp tinygpu:$WORK/hancock/embeddings.zip D:/College/Sem_5/HistoRag-BIMAP/data/"
