"""
Parallel multi-threaded downloader for HANCOCK WSI zip files.
Uses HTTP Range requests to download in chunks simultaneously.
Supports resume — re-run to continue interrupted downloads.
Reassembly is built-in — runs automatically after all chunks complete.

Usage:
    Edit URL, OUTPUT, and EXPECTED_SIZE below, then:
    python3 parallel_download.py

Tissue configs:
    Larynx       : 314,557,232,221 bytes  (exception: saved under larynx/zips/)
    Hypopharynx  : 213,184,293,362 bytes
    Oral_Cavity  : TBD
    Oropharynx1  : TBD
    Oropharynx2  : TBD
"""
import urllib.request
import threading
import os
import sys
import time
import shutil

# ── CONFIGURE HERE ────────────────────────────────────────────────────────────
URL           = "https://data.fau.de/public/24/87/322108724/WSI_PrimaryTumor_Hypopharynx.zip"
TISSUE_LOWER  = "hypopharynx"   # lowercase, matches directory name on HPC
EXPECTED_SIZE = 213184293362
# ─────────────────────────────────────────────────────────────────────────────

output    = os.environ["WORK"] + f"/hancock/{TISSUE_LOWER}/zips/" + os.path.basename(URL)
log_file  = os.path.expanduser("~/scripts/download.log")
num_threads      = min(16, os.cpu_count() or 8)
chunk            = EXPECTED_SIZE // num_threads
max_retries      = 3
stall_limit      = 30 * 60
copy_buffer_size = (
    128 * 1024 * 1024
    if (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) >= 128 * 1024 * 1024 * 1024
    else 64 * 1024 * 1024
)

downloaded  = [0] * num_threads
lock        = threading.Lock()
abort       = threading.Event()
reassembling = threading.Event()


def log(msg):
    ts   = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_file, "a") as f:
        f.write(line + "\n")


def download_chunk(start, end, part):
    part_file  = f"{output}.part{part}"
    resume     = os.path.getsize(part_file) if os.path.exists(part_file) else 0
    chunk_size = end - start + 1

    if resume >= chunk_size:
        log(f"Chunk {part} already complete, skipping")
        with lock:
            downloaded[part] += chunk_size
        return

    if resume > 0:
        log(f"Chunk {part}: resuming from {resume/1e6:.0f} MB")

    for attempt in range(1, max_retries + 1):
        if abort.is_set():
            return
        try:
            req = urllib.request.Request(URL)
            req.add_header("Range", f"bytes={start + resume}-{end}")
            with urllib.request.urlopen(req, timeout=60) as r:
                with open(part_file, "ab") as f:
                    while not abort.is_set():
                        buf = r.read(1024 * 1024)
                        if not buf:
                            break
                        f.write(buf)
                        resume += len(buf)
                        with lock:
                            downloaded[part] += len(buf)
            log(f"Chunk {part} complete")
            return
        except Exception as e:
            log(f"Chunk {part} attempt {attempt}/{max_retries} failed: {e}")
            time.sleep(5)

    log(f"Chunk {part} failed after {max_retries} attempts — aborting")
    abort.set()


def watchdog():
    last_total         = 0
    last_progress_time = time.time()
    while not abort.is_set():
        time.sleep(60)
        if reassembling.is_set():
            continue
        total = sum(downloaded)
        if total > last_total:
            last_total         = total
            last_progress_time = time.time()
        else:
            stalled = time.time() - last_progress_time
            log(f"Watchdog: no progress for {stalled/60:.1f} min")
            if stalled >= stall_limit:
                log("No progress for 30 minutes — aborting download")
                abort.set()


os.makedirs(os.path.dirname(output), exist_ok=True)
log(f"Starting download: {EXPECTED_SIZE/1e9:.1f} GB in {num_threads} chunks")
log(f"Output: {output}")

wd = threading.Thread(target=watchdog, daemon=True)
wd.start()

threads = []
for i in range(num_threads):
    start = i * chunk
    end   = (i + 1) * chunk - 1 if i < num_threads - 1 else EXPECTED_SIZE - 1
    t     = threading.Thread(target=download_chunk, args=(start, end, i))
    threads.append(t)
    t.start()

while any(t.is_alive() for t in threads):
    total = sum(downloaded)
    pct   = 100 * total / EXPECTED_SIZE
    print(f"\rProgress: {total/1e9:.1f}/{EXPECTED_SIZE/1e9:.1f} GB ({pct:.1f}%)", end="", flush=True)
    if abort.is_set():
        break
    time.sleep(5)

for t in threads:
    t.join()

if abort.is_set():
    log("Download aborted — part files kept for resume")
    sys.exit(1)

reassembling.set()
log("Reassembling chunks...")
try:
    with open(output, "wb") as f:
        for i in range(num_threads):
            part = f"{output}.part{i}"
            if not os.path.exists(part):
                raise FileNotFoundError(f"Missing part file: {part}")
            with open(part, "rb") as pf:
                shutil.copyfileobj(pf, f, length=copy_buffer_size)
            try:
                os.remove(part)
            except FileNotFoundError:
                pass  # GPFS may report file as gone after successful read
except Exception as e:
    import traceback
    log(f"Reassembly failed: {repr(e)}")
    log(traceback.format_exc())
    sys.exit(1)

actual = os.path.getsize(output)
if actual != EXPECTED_SIZE:
    log(f"Size mismatch! Expected {EXPECTED_SIZE}, got {actual} — file may be corrupt")
    sys.exit(1)

log(f"Download complete and verified: {output}")
