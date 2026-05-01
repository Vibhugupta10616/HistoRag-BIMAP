"""Phase registry for the HistoRAG Streamlit demo.

Each entry in PHASES maps a display name to the paths and metadata needed to
load that phase's encoder, FAISS index, and embeddings at demo time.

All paths are relative to the repo root.  Add a new phase entry here once the
Phase 1 experiments have been run and the UNI2-h encoder registry is wired.
"""
from __future__ import annotations

from pathlib import Path

from histoRAG.log import embed_cache_key, load_config

# Repo root — two levels above histoRAG/viz/
_ROOT = Path(__file__).resolve().parents[2]


def _cache_key(config_path: str) -> str:
    """Derive the embedding cache directory name from a config file."""
    cfg = load_config(_ROOT / config_path)
    return embed_cache_key(cfg)


# ── Phase registry ────────────────────────────────────────────────────────────
# Keys are display names shown in the sidebar dropdown.
# Set  "placeholder": True  for phases that don't have data yet; the app will
# render an informational screen instead of the retrieval interface.

_P0_KEY = _cache_key("configs/phase0_mvp.yaml")

PHASES: dict[str, dict] = {
    # ── Phase 0 ── CLIP ViT-B/16 on TMA blocks (baseline) ───────────────────
    "Phase 0 — CLIP / TMA  (Baseline)": {
        "encoder_kwargs": {
            "model_id": "ViT-B-16",
            "pretrained": "openai",
            "device": "auto",
        },
        "embed_dim": 512,
        # Pre-computed embeddings (one .npy per full manifest, seed-agnostic)
        "embeddings_path": f"data/indexes/{_P0_KEY}/embeddings.npy",
        "ids_path":        f"data/indexes/{_P0_KEY}/patch_int_ids.npy",
        # FAISS index covering all patches (built in pipeline step)
        "index_path":      "data/indexes/phase0_mvp.faiss",
        "manifest_path":   "data/patches/manifest.parquet",
        # Filter experiments.csv to only Phase 0 rows for the metrics panel
        "metrics_filter":  {"encoder": "clip-vitb16"},
        "description":     "CLIP ViT-B/16 (OpenAI) · 2 HANCOCK TMA blocks · 3,044 patches · 2 classes.",
        "badge_color":     "#3B82F6",
        "badge_text":      "Phase 0",
    },

    # ── Phase 1 ── UNI2-h on WSIs (added after experiments run) ─────────────
    "Phase 1 — UNI2-h / WSI  (Coming Soon)": {
        "placeholder":  True,
        "description":  (
            "UNI2-h (MahmoodLab, 630M params) on HANCOCK WSIs. "
            "Run Phase 1 experiments first, then add the index and embeddings paths here."
        ),
        "badge_color":  "#F59E0B",
        "badge_text":   "Phase 1",
    },

    # ── Phase 2 ── placeholder for future work ───────────────────────────────
    "Phase 2 — Text Query / Cross-Slide  (Planned)": {
        "placeholder":  True,
        "description":  "CLIP text-tower queries and leave-slide-out evaluation. Scope defined in Phase 2 planning.",
        "badge_color":  "#8B5CF6",
        "badge_text":   "Phase 2",
    },
}
