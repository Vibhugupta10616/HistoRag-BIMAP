"""Config loading, experiment CSV logging, and seed setting."""
from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# Phase 0 fields — kept for backward compatibility with existing experiments.csv rows.
_CSV_FIELDS_V0 = [
    "uid", "date_utc", "git_commit", "config_hash", "config_path",
    "encoder", "index", "dataset", "num_patches", "num_query", "num_gallery",
    "seed", "top1", "top5", "top10", "map_at_10", "random_baseline_top5",
    "embed_time_s", "index_time_s", "query_time_s", "notes",
]

# Phase 1 additions — slide-level metrics and encoder cost metrics.
_CSV_FIELDS_V1_EXTRA = [
    "eval_level",           # "patch" | "slide"
    "num_slides",           # total slides in dataset
    "num_query_slides",     # slides used as queries
    "num_gallery_slides",   # slides used as gallery
    "slide_top1",           # slide-level top-1 accuracy
    "slide_map_at_k",       # slide-level mAP@K (K from config)
    "param_count",          # total encoder parameters
    "peak_vram_mb",         # peak GPU memory during encoding (null on CPU)
    "throughput_patches_per_sec",  # encoding speed
]

_CSV_FIELDS = _CSV_FIELDS_V0 + _CSV_FIELDS_V1_EXTRA


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def hash_config(cfg: dict[str, Any]) -> str:
    """12-char SHA-256 prefix of the canonicalized config."""
    raw = json.dumps(cfg, sort_keys=True, ensure_ascii=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def embed_cache_key(cfg: dict[str, Any]) -> str:
    """Config hash for the embedding cache — excludes run.seed.

    Encoding is deterministic given fixed encoder + data; only the
    query/gallery split varies by seed, so all seeds share one cache dir.
    """
    cfg_copy = {**cfg, "run": {k: v for k, v in cfg.get("run", {}).items() if k != "seed"}}
    raw = json.dumps(cfg_copy, sort_keys=True, ensure_ascii=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def set_all_seeds(seed: int) -> None:
    """Set seeds for Python random, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _read_csv_fields(csv_path: Path) -> list[str]:
    """Return the field names from the first line of an existing CSV, or _CSV_FIELDS if empty."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return _CSV_FIELDS
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return _CSV_FIELDS


def append_experiment_row(
    cfg: dict[str, Any],
    metrics: dict[str, Any],
    timings: dict[str, float],
    notes: str = "",
    experiments_dir: str | Path = "experiments",
    runs_dir: str | Path = "configs/runs",
) -> str:
    """Append one row to experiments.csv and save an immutable config snapshot.

    Returns the run UID. Raises FileExistsError if the UID already exists.

    Field schema is backward-compatible: Phase 0 rows have empty strings for the
    Phase 1 extra columns; Phase 1 rows populate all columns.
    """
    experiments_dir, runs_dir = Path(experiments_dir), Path(runs_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    encoder = cfg.get("encoder", {}).get("name", "unknown")
    index_name = cfg.get("index", {}).get("name", "unknown")
    dataset = cfg.get("data", {}).get("dataset", "unknown")
    seed = cfg.get("run", {}).get("seed", 0)
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")

    csv_path = experiments_dir / "experiments.csv"
    existing: set[str] = set()
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            existing = {row["uid"] for row in csv.DictReader(f)}

    uid = f"{date_str}_{len(existing)+1:03d}_{encoder}_{index_name}_{dataset}_seed{seed}"
    if uid in existing:
        raise FileExistsError(f"UID '{uid}' already in experiments.csv — change seed or config.")

    snapshot = runs_dir / f"{uid}.yaml"
    with open(snapshot, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=True)

    # Use the field list already written as the CSV header for backward compat.
    # New files get the full Phase 1 schema; existing Phase 0 CSVs keep their schema.
    fieldnames = _read_csv_fields(csv_path)

    row: dict[str, Any] = {f: "" for f in fieldnames}
    row.update({
        "uid": uid,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config_hash": hash_config(cfg),
        "config_path": str(snapshot),
        "encoder": encoder,
        "index": index_name,
        "dataset": dataset,
        "num_patches": metrics.get("num_patches", ""),
        "num_query": metrics.get("num_query", ""),
        "num_gallery": metrics.get("num_gallery", ""),
        "seed": seed,
        "top1": metrics.get("top1", ""),
        "top5": metrics.get("top5", ""),
        "top10": metrics.get("top10", ""),
        "map_at_10": metrics.get("map_at_10", ""),
        "random_baseline_top5": metrics.get("random_baseline_top5", ""),
        "embed_time_s": timings.get("embed_time_s", ""),
        "index_time_s": timings.get("index_time_s", ""),
        "query_time_s": timings.get("query_time_s", ""),
        "notes": notes,
        # Phase 1 fields — empty string if not provided (Phase 0 rows)
        "eval_level": metrics.get("eval_level", ""),
        "num_slides": metrics.get("num_slides", ""),
        "num_query_slides": metrics.get("num_query_slides", ""),
        "num_gallery_slides": metrics.get("num_gallery_slides", ""),
        "slide_top1": metrics.get("slide_top1", ""),
        "slide_map_at_k": metrics.get("slide_map_at_k", ""),
        "param_count": metrics.get("param_count", ""),
        "peak_vram_mb": metrics.get("peak_vram_mb", ""),
        "throughput_patches_per_sec": metrics.get("throughput_patches_per_sec", ""),
    })

    # Only write fields that exist in the current CSV schema
    row = {k: row[k] for k in fieldnames if k in row}

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return uid
