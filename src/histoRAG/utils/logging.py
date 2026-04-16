"""Experiment logging: append rows to experiments.csv and snapshot configs."""
from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from histoRAG.utils.config import hash_config

_CSV_FIELDNAMES = [
    "uid", "date_utc", "git_commit", "config_hash", "config_path",
    "encoder", "index", "dataset", "num_patches", "num_query", "num_gallery",
    "seed", "top1", "top5", "top10", "map_at_10", "random_baseline_top5",
    "embed_time_s", "index_time_s", "query_time_s", "notes",
]


def _git_commit() -> str:
    """Return the current HEAD commit hash, or 'unknown' if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def append_experiment_row(
    cfg: dict[str, Any],
    metrics: dict[str, Any],
    timings: dict[str, float],
    notes: str = "",
    experiments_dir: str | Path = "experiments",
    runs_dir: str | Path = "configs/runs",
) -> str:
    """Append one row to experiments.csv and save an immutable config snapshot.

    Parameters
    ----------
    cfg:
        Full config dict for this run.
    metrics:
        Dict containing: num_patches, num_query, num_gallery, top1, top5, top10,
        map_at_10, random_baseline_top5.
    timings:
        Dict containing: embed_time_s, index_time_s, query_time_s.
    notes:
        One-line free-text description (e.g. 'Phase 0 baseline, seed 42').
    experiments_dir:
        Directory containing experiments.csv.
    runs_dir:
        Directory where immutable per-run config snapshots are saved.

    Returns
    -------
    uid : str
        The generated run UID (used to reference this row later).

    Raises
    ------
    FileExistsError
        If a row with the same UID already exists — refuses to overwrite.
    """
    experiments_dir = Path(experiments_dir)
    runs_dir = Path(runs_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    encoder = cfg.get("encoder", {}).get("name", "unknown")
    index = cfg.get("index", {}).get("name", "unknown")
    dataset = cfg.get("data", {}).get("dataset", "unknown")
    seed = cfg.get("run", {}).get("seed", 0)
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")

    # Determine sequential run number from existing CSV rows
    csv_path = experiments_dir / "experiments.csv"
    existing_uids: set[str] = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                existing_uids.add(row["uid"])
    run_number = len(existing_uids) + 1

    uid = f"{date_str}_{run_number:03d}_{encoder}_{index}_{dataset}_seed{seed}"

    if uid in existing_uids:
        raise FileExistsError(
            f"Experiment UID '{uid}' already exists in experiments.csv. "
            "Refusing to overwrite — increment seed or change config."
        )

    # Snapshot config to configs/runs/<uid>.yaml (immutable reference)
    snapshot_path = runs_dir / f"{uid}.yaml"
    with open(snapshot_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=True)

    row = {
        "uid": uid,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config_hash": hash_config(cfg),
        "config_path": str(snapshot_path),
        "encoder": encoder,
        "index": index,
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
    }

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return uid
