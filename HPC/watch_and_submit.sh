#!/bin/bash
# Watcher: polls every 5 minutes until the Larynx zip download is complete,
# then automatically submits the embed_job.sh SLURM job.
#
# Run in a screen session BEFORE going to sleep:
#   screen -S watcher
#   bash ~/HistoRag-BIMAP/HPC/watch_and_submit.sh
#   Ctrl+A then D  (detach)

ZIP="$WORK/hancock/zips/WSI_PrimaryTumor_Larynx.zip"
EXPECTED_SIZE=314557232221
LOG="$HOME/scripts/watch.log"
REPO="$HOME/HistoRag-BIMAP"

log() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG"; }

log "========================================"
log "Watcher started"
log "Watching : $ZIP"
log "Expected : $EXPECTED_SIZE bytes"
log "========================================"

while true; do
    if [ -f "$ZIP" ]; then
        ACTUAL=$(stat -c%s "$ZIP" 2>/dev/null || echo 0)
        PCT=$(awk "BEGIN { printf \"%.1f\", $ACTUAL * 100 / $EXPECTED_SIZE }")
        log "Download progress: $ACTUAL / $EXPECTED_SIZE bytes ($PCT%)"

        if [ "$ACTUAL" -eq "$EXPECTED_SIZE" ]; then
            log "Download complete. Submitting SLURM embedding job ..."
            JOB_ID=$(sbatch "$REPO/HPC/embed_job.sh" | awk '{print $NF}')
            log "Submitted job ID: $JOB_ID"
            log "Monitor with:  squeue -u $USER"
            log "Live logs:     tail -f $HOME/scripts/embed_${JOB_ID}.log"
            log "Watcher exiting."
            break
        fi
    else
        log "Zip not found yet — download may still be starting."
    fi

    # Check again in 5 minutes
    sleep 300
done
