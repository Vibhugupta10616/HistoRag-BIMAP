#!/bin/bash
# SLURM job: unzip WSI archive -> tile -> embed (CLIP) -> zip embeddings.
# Usage: Edit TISSUE below, then submit with: sbatch ~/HistoRag-BIMAP/HPC/embed_job.sh

#SBATCH --job-name=embed
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=/home/hpc/vlbi/vlbi113v/embed_%j.log

set -e

log() { echo "[$(date +%H:%M:%S)] $1"; }

# ── CONFIGURE HERE ────────────────────────────────────────────────────────────
TISSUE="Larynx"       # Larynx | Hypopharynx | Oral_Cavity | Oropharynx1 | Oropharynx2
ENCODER="clip"        # clip | conch
# ─────────────────────────────────────────────────────────────────────────────

TISSUE_LOWER=$(echo "$TISSUE" | tr '[:upper:]' '[:lower:]')
HPC_DIR="$HOME/HistoRag-BIMAP/HPC"
VENV="$HPC_DIR/hpcenv"
WSI_DIR="$WORK/hancock/$TISSUE_LOWER/wsi"
PATCHES_DIR="$WORK/hancock/$TISSUE_LOWER/patches"
OUT_DIR="$WORK/hancock/embeddings"

# Larynx zip is under hancock/zips/ (downloaded before tissue-specific structure).
# All other tissues have their zip under hancock/{tissue}/zips/.
if [ "$TISSUE" = "Larynx" ]; then
    ZIP="$WORK/hancock/zips/WSI_PrimaryTumor_Larynx.zip"
else
    ZIP=$(find "$WORK/hancock/$TISSUE_LOWER/zips/" -name "*.zip" | head -1)
fi

log "Tissue   : $TISSUE"
log "Encoder  : $ENCODER"
log "ZIP      : $ZIP"
log "WSI dir  : $WSI_DIR"
log "Patches  : $PATCHES_DIR"
log "Output   : $OUT_DIR"

# --- Step 1: Load Python module ---
log "Loading Python module ..."
module load python/pytorch2.6py3.12

# --- Step 2: Extract zip ---
log "Extracting $ZIP ..."
mkdir -p "$WSI_DIR"
unzip -o "$ZIP" -d "$WSI_DIR"
log "Extraction complete — $(ls "$WSI_DIR" | wc -l) files"

# --- Step 3: Activate venv ---
log "Activating virtualenv ..."
source "$VENV/bin/activate"
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# --- Step 4: Tile + embed ---
log "Starting pipeline ..."
python "$HPC_DIR/hpc_pipeline.py" \
    --wsi_dir     "$WSI_DIR" \
    --out_dir     "$OUT_DIR" \
    --patches_dir "$PATCHES_DIR" \
    --tissue      "$TISSUE" \
    --encoder     "$ENCODER" \
    --max_patches 5000 \
    --batch_size  256

# --- Step 5: Zip embeddings for local download ---
log "Zipping embeddings ..."
cd "$WORK/hancock"
zip -r embeddings.zip embeddings/
log "Done: $WORK/hancock/embeddings.zip"
log "Download with: scp tinygpu:$WORK/hancock/embeddings.zip D:/College/Sem_5/HistoRag-BIMAP/data/"
