"""YAML configuration loading, canonicalization, and hashing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return it as a nested dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def canonicalize(cfg: dict[str, Any]) -> str:
    """Return a deterministic JSON string of *cfg* with recursively sorted keys.

    Two configs that are semantically identical but differ in key order
    will produce the same canonical string.
    """
    return json.dumps(cfg, sort_keys=True, ensure_ascii=True, default=str)


def hash_config(cfg: dict[str, Any]) -> str:
    """Return a 12-character SHA-256 hex prefix of the canonical config string.

    Used to detect duplicate runs and link experiment rows to config snapshots.
    """
    raw = canonicalize(cfg).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]
