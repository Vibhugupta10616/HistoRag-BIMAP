#!/bin/bash
# Watcher: polls every 5 minutes until the tissue zip download is complete,
# then automatically submits the embed_job.sh SLURM job.
#
# Usage: Edit TISSUE below, then run in a screen session:
#   screen -S watcher
#   bash ~/HistoRag-BIMAP/HPC/watch_and_submit.sh
#   Ctrl+A then D  (detach)

# ── CONFIGURE HERE ────────────────────────────────────────────────────────────
TISSUE="Hypopharynx"   # Larynx | Hypopharynx | Oral_Cavity | Oropharynx1 | Oropharynx2
# ─────────────────────────────────────────────────────────────────────────────

TISSUE_LOWER=$(echo "$TISSUE" | tr '[:upper:]' '[:lower:]')
ZIP_DIR="$WORK/hancock/$TISSUE_LOWER/zips"
LOG="$HOME/scripts/watch.log"
REPO="$HOME/HistoRag-BIMAP"

log() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG"; }

log "========================================"
log "Watcher started for: $TISSUE"
log "Watching : $ZIP_DIR"
log "========================================"

while true; do
    ZIP=$(find "$ZIP_DIR" -name "*.zip" -not -name "*.part*" 2>/dev/null | head -1)

    if [ -n "$ZIP" ]; then
        ACTUAL=$(stat -c%s "$ZIP" 2>/dev/null || echo 0)
        log "Found zip: $ZIP ($ACTUAL bytes)"

        # Update embed_job.sh tissue and submit
        sed -i "s/^TISSUE=.*/TISSUE=\"$TISSUE\"/" "$REPO/HPC/embed_job.sh"
        JOB_ID=$(sbatch "$REPO/HPC/embed_job.sh" | awk '{print $NF}')
        log "Submitted job ID: $JOB_ID"
        log "Monitor with:  squeue -u $USER"
        log "Live logs:     tail -f $HOME/embed_${JOB_ID}.log"
        log "Watcher exiting."
        break
    else
        log "No zip found yet in $ZIP_DIR — waiting ..."
    fi

    sleep 300
done
