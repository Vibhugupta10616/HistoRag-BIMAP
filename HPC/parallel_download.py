"""
Parallel multi-threaded downloader for HANCOCK WSI zip files.
Uses HTTP Range requests to download chunks simultaneously,
writing each chunk DIRECTLY to its correct offset in the final file.
No part files, no reassembly step — download is the final file.

Supports resume via a .state file tracking completed chunks.

Usage:
    Edit URL, TISSUE_LOWER, and EXPECTED_SIZE below, then:
    python3 parallel_download.py

Tissue configs (all save to hancock/{tissue}/zips/):
    Larynx       : 314,557,232,221 bytes
    Hypopharynx  : 213,184,293,362 bytes
    Oral_Cavity  : 429,916,307,251 bytes
    Oropharynx1  : 516,446,064,041 bytes
    Oropharynx2  : 528,375,773,776 bytes
"""
import urllib.request
import threading
import os
import sys
import time
import json

# ── CONFIGURE HERE ────────────────────────────────────────────────────────────
URL           = "https://data.fau.de/public/24/87/322108724/WSI_PrimaryTumor_Oropharynx_Part2.zip"  # CHANGE: FAU download URL for the target tissue zip
TISSUE_LOWER  = "oropharynx2"   # CHANGE: lowercase tissue name (larynx | hypopharynx | oral_cavity | oropharynx1 | oropharynx2)
EXPECTED_SIZE = 528375773776    # CHANGE: exact byte size of the zip (see docstring above for all tissue sizes)
# ─────────────────────────────────────────────────────────────────────────────

output     = os.environ["WORK"] + f"/hancock/{TISSUE_LOWER}/zips/" + os.path.basename(URL)
state_file = output + ".state"
log_file   = os.path.expanduser("~/scripts/download.log")
num_threads = min(16, os.cpu_count() or 8)
chunk       = EXPECTED_SIZE // num_threads
max_retries = 3
stall_limit = 30 * 60
read_size   = 8 * 1024 * 1024  # 8MB chunks for fewer syscalls

downloaded = [0] * num_threads
lock       = threading.Lock()
abort      = threading.Event()


def log(msg):
    ts   = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_file, "a") as f:
        f.write(line + "\n")


def load_state() -> set:
    """Load set of already-completed chunk indices."""
    if os.path.exists(state_file):
        with open(state_file) as f:
            return set(json.load(f))
    return set()


def save_state(completed: set) -> None:
    with open(state_file, "w") as f:
        json.dump(list(completed), f)


def download_chunk(start: int, end: int, part: int, completed: set) -> None:
    if part in completed:
        log(f"Chunk {part} already complete, skipping")
        with lock:
            downloaded[part] += end - start + 1
        return

    chunk_size = end - start + 1
    written    = 0

    for attempt in range(1, max_retries + 1):
        if abort.is_set():
            return
        try:
            req = urllib.request.Request(url=URL)
            req.add_header("Range", f"bytes={start + written}-{end}")
            with urllib.request.urlopen(req, timeout=60) as r:
                with open(output, "r+b") as f:
                    f.seek(start + written)
                    while not abort.is_set():
                        buf = r.read(read_size)
                        if not buf:
                            break
                        f.write(buf)
                        written += len(buf)
                        with lock:
                            downloaded[part] += len(buf)

            if written >= chunk_size:
                log(f"Chunk {part} complete")
                with lock:
                    completed.add(part)
                    save_state(completed)
                return
        except Exception as e:
            log(f"Chunk {part} attempt {attempt}/{max_retries} failed: {e}")
            time.sleep(5)

    log(f"Chunk {part} failed after {max_retries} attempts — aborting")
    abort.set()


def watchdog() -> None:
    last_total         = 0
    last_progress_time = time.time()
    while not abort.is_set():
        time.sleep(60)
        total = sum(downloaded)
        if total > last_total:
            last_total         = total
            last_progress_time = time.time()
        else:
            stalled = time.time() - last_progress_time
            log(f"Watchdog: no progress for {stalled/60:.1f} min")
            if stalled >= stall_limit:
                log("No progress for 30 minutes — aborting")
                abort.set()


os.makedirs(os.path.dirname(output), exist_ok=True)

# Already complete — nothing to do
if os.path.exists(output) and os.path.getsize(output) == EXPECTED_SIZE and not os.path.exists(state_file):
    log(f"File already complete: {output}")
    sys.exit(0)

completed = load_state()
is_resume = len(completed) > 0 and os.path.exists(output)

if not is_resume:
    # Fresh start — pre-allocate file so all threads can write in parallel
    completed = set()
    log(f"Pre-allocating {EXPECTED_SIZE/1e9:.1f} GB file ...")
    with open(output, "wb") as f:
        os.truncate(f.fileno(), EXPECTED_SIZE)
    save_state(completed)  # create state file immediately so re-run detects incomplete
else:
    log(f"Resuming: {len(completed)}/{num_threads} chunks already done")

log(f"Downloading: {EXPECTED_SIZE/1e9:.1f} GB in {num_threads} chunks")
log(f"Output: {output}")

wd = threading.Thread(target=watchdog, daemon=True)
wd.start()

threads = []
for i in range(num_threads):
    start = i * chunk
    end   = (i + 1) * chunk - 1 if i < num_threads - 1 else EXPECTED_SIZE - 1
    t     = threading.Thread(target=download_chunk, args=(start, end, i, completed))
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
    log("Download aborted — re-run to resume from completed chunks")
    sys.exit(1)

# Verify final size
actual = os.path.getsize(output)
if actual != EXPECTED_SIZE:
    log(f"Size mismatch! Expected {EXPECTED_SIZE}, got {actual}")
    sys.exit(1)

# Clean up state file
os.remove(state_file)
log(f"Download complete and verified: {output}")
