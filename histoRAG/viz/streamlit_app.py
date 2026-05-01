"""HistoRAG Streamlit demo — versioned histopathology retrieval interface.

Run:
    streamlit run histoRAG/viz/streamlit_app.py

Phases are defined in histoRAG/viz/versions.py.  Switch phases via the sidebar
dropdown; each phase loads its own encoder, FAISS index, and embeddings.
"""
from __future__ import annotations

import random as _random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from histoRAG.embed import ClipEncoder, FaissFlatIP
from histoRAG.viz.versions import PHASES

# Repo root — two levels above histoRAG/viz/
ROOT = Path(__file__).resolve().parents[2]

# ── Page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="HistoRAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Label colour palette ──────────────────────────────────────────────────────

_LABEL_COLORS: dict[str, str] = {
    "invasion_front": "#F59E0B",
    "tumor_center":   "#3B82F6",
    "unknown":        "#64748B",
}


def _label_color(label: str) -> str:
    return _LABEL_COLORS.get(label, _LABEL_COLORS["unknown"])


# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');

/* Global typography */
html, body, [class*="css"]   { font-family: 'Inter', -apple-system, sans-serif; }

/* Hide Streamlit chrome */
#MainMenu, footer, header    { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"]    { border-right: 1px solid #1E2030 !important; }

/* Label chip */
.lchip {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.6px;
    padding: 2px 8px;
    border-radius: 10px;
    text-transform: uppercase;
}

/* Similarity bar */
.sim-track {
    background: #1E2030;
    border-radius: 3px;
    height: 3px;
    margin: 4px 0 6px;
}
.sim-fill {
    height: 3px;
    border-radius: 3px;
    background: linear-gradient(90deg, #1E40AF 0%, #3B82F6 100%);
}

/* Phase badge */
.pbadge {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 3px 9px;
    border-radius: 6px;
}

/* Section divider row */
.sec-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 22px 0 12px;
}
.sec-line { flex: 1; height: 1px; background: #1E2030; }
.sec-title {
    color: #475569;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    white-space: nowrap;
}

/* KPI card */
.kpi {
    background: #1A1D27;
    border: 1px solid #1E2030;
    border-radius: 12px;
    padding: 18px 20px 14px;
    text-align: center;
}
.kpi-value {
    font-family: 'Fira Code', monospace;
    font-size: 26px;
    font-weight: 600;
    color: #3B82F6;
    line-height: 1.1;
}
.kpi-std  {
    font-family: 'Fira Code', monospace;
    font-size: 11px;
    color: #334155;
    margin-top: 2px;
}
.kpi-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    color: #475569;
    text-transform: uppercase;
    margin-top: 6px;
}

/* Result rank badge */
.rank {
    font-family: 'Fira Code', monospace;
    font-size: 10px;
    font-weight: 600;
    color: #64748B;
    background: #1E2030;
    padding: 1px 6px;
    border-radius: 4px;
}
.rank-1 { color: #93C5FD; background: #1E3A8A44; }

/* Coord text */
.coord {
    font-family: 'Fira Code', monospace;
    font-size: 10px;
    color: #334155;
}
</style>
""", unsafe_allow_html=True)

# ── Cached resource loaders ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading encoder…")
def _load_encoder(phase_key: str) -> ClipEncoder:
    cfg = PHASES[phase_key]
    return ClipEncoder(**cfg["encoder_kwargs"])


@st.cache_resource(show_spinner="Loading FAISS index…")
def _load_index(phase_key: str) -> FaissFlatIP:
    return FaissFlatIP.load(str(ROOT / PHASES[phase_key]["index_path"]))


@st.cache_data(show_spinner="Loading manifest…")
def _load_manifest(phase_key: str) -> pd.DataFrame:
    return pd.read_parquet(str(ROOT / PHASES[phase_key]["manifest_path"]))


@st.cache_data(show_spinner="Loading embeddings…")
def _load_embeddings(phase_key: str) -> tuple[np.ndarray, np.ndarray]:
    cfg = PHASES[phase_key]
    embs = np.load(str(ROOT / cfg["embeddings_path"]))
    ids  = np.load(str(ROOT / cfg["ids_path"]))
    return embs, ids


@st.cache_data
def _load_metrics(phase_key: str) -> pd.DataFrame:
    csv_path = ROOT / "experiments" / "experiments.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df   = pd.read_csv(csv_path)
    filt = PHASES[phase_key].get("metrics_filter", {})
    for col, val in filt.items():
        df = df[df[col] == val]
    return df


# ── Helper: search ────────────────────────────────────────────────────────────

def _search(
    query_emb: np.ndarray,
    index: FaissFlatIP,
    manifest: pd.DataFrame,
    k: int,
    exclude_id: int | None = None,
) -> pd.DataFrame:
    """Return top-k results as a DataFrame, optionally skipping one patch ID."""
    fetch = k + (1 if exclude_id is not None else 0)
    sims, ids = index.search(query_emb.reshape(1, -1), fetch)
    rows = []
    for sim, iid in zip(sims[0], ids[0]):
        if iid < 0 or iid == exclude_id:
            continue
        r = manifest.iloc[int(iid)].to_dict()
        r["_sim"]  = float(sim)
        r["_iid"]  = int(iid)
        rows.append(r)
        if len(rows) == k:
            break
    return pd.DataFrame(rows)


# ── Helper: HTML fragments ────────────────────────────────────────────────────

def _badge_html(text: str, color: str) -> str:
    return (
        f'<span class="pbadge" style="background:{color}22;'
        f'color:{color};border:1px solid {color}55;">{text}</span>'
    )


def _label_chip(label: str) -> str:
    c = _label_color(label)
    return (
        f'<span class="lchip" style="background:{c}22;color:{c};border:1px solid {c}44;">'
        f'{label.replace("_", " ")}</span>'
    )


def _sim_bar(sim: float) -> str:
    pct = f"{max(0.0, min(1.0, sim)) * 100:.1f}"
    return f'<div class="sim-track"><div class="sim-fill" style="width:{pct}%"></div></div>'


def _section(title: str, suffix: str = "") -> None:
    suf = f'<span style="color:#1E2A3A;font-size:11px;">{suffix}</span>' if suffix else ""
    st.markdown(
        f'<div class="sec-row"><span class="sec-title">{title}</span>'
        f'<div class="sec-line"></div>{suf}</div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## HistoRAG")
    st.caption("Histopathology Retrieval Atlas · FAU BIMAP SS2026")
    st.divider()

    phase_key = st.selectbox(
        "Phase version",
        options=list(PHASES.keys()),
        index=0,
        help="Each phase uses a different encoder, dataset, or evaluation protocol.",
    )
    pcfg = PHASES[phase_key]

    st.markdown(
        _badge_html(pcfg["badge_text"], pcfg["badge_color"]),
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:#475569;font-size:12px;margin-top:6px;line-height:1.5;">'
        f'{pcfg["description"]}</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    if pcfg.get("placeholder"):
        top_k      = 8
        query_mode = None
    else:
        top_k = st.slider("Top-K results", min_value=1, max_value=20, value=8)
        query_mode = st.radio(
            "Query source",
            ["Random gallery patch", "Upload image"],
        )

    st.divider()
    st.markdown(
        '<p style="color:#1E2A3A;font-size:11px;">HANCOCK · FAISS flat-IP · cosine sim</p>',
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────

hcol, bcol = st.columns([6, 1])
with hcol:
    st.markdown(
        '<h1 style="font-size:26px;font-weight:700;color:#E2E8F0;margin-bottom:2px;">'
        'HistoRAG <span style="color:#3B82F6">—</span> Histopathology Retrieval Atlas'
        '</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#475569;font-size:13px;margin:0;">Content-based image retrieval for tissue patches · '
        'ranked by cosine similarity</p>',
        unsafe_allow_html=True,
    )
with bcol:
    st.markdown(
        f'<div style="text-align:right;padding-top:8px;">'
        f'{_badge_html(pcfg["badge_text"], pcfg["badge_color"])}</div>',
        unsafe_allow_html=True,
    )


# ── Placeholder screen ────────────────────────────────────────────────────────

if pcfg.get("placeholder"):
    st.divider()
    st.markdown(
        f'<div style="text-align:center;padding:80px 20px;">'
        f'<div style="font-size:52px;margin-bottom:20px;">🔬</div>'
        f'<h2 style="color:#E2E8F0;font-weight:600;">{pcfg["badge_text"]}</h2>'
        f'<p style="color:#475569;max-width:520px;margin:10px auto;line-height:1.7;">'
        f'{pcfg["description"]}</p>'
        f'<div style="margin-top:24px;">'
        f'{_badge_html("Coming Soon", pcfg["badge_color"])}'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()


# ── Load phase resources ──────────────────────────────────────────────────────

encoder    = _load_encoder(phase_key)
index      = _load_index(phase_key)
manifest   = _load_manifest(phase_key)
embs, _    = _load_embeddings(phase_key)
metrics_df = _load_metrics(phase_key)


# ── Query section ─────────────────────────────────────────────────────────────

_section("QUERY PATCH")

query_emb:   np.ndarray | None = None
query_row:   dict        | None = None
query_iid:   int         | None = None

img_col, meta_col = st.columns([1, 3], gap="large")

if query_mode == "Random gallery patch":
    if st.button("New random patch", type="primary"):
        st.session_state["rand_idx"] = int(_random.randrange(len(manifest)))
    if "rand_idx" not in st.session_state:
        st.session_state["rand_idx"] = int(_random.randrange(len(manifest)))

    idx        = st.session_state["rand_idx"]
    query_row  = manifest.iloc[idx].to_dict()
    query_iid  = idx
    query_emb  = embs[idx]

    with img_col:
        with st.container(border=True):
            st.image(str(ROOT / query_row["path"]), use_container_width=True)

    with meta_col:
        st.markdown(
            f'<div style="padding:4px 0;">'
            f'<div style="color:#475569;font-size:10px;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px;">Source patch</div>'
            f'<div style="color:#E2E8F0;font-size:15px;font-weight:600;font-family:\'Fira Code\',monospace;">'
            f'{query_row["slide_id"]}</div>'
            f'<div class="coord" style="margin-top:3px;">x = {query_row["x"]}  ·  y = {query_row["y"]}</div>'
            f'<div style="margin-top:10px;">{_label_chip(query_row["label"])}</div>'
            f'<div class="coord" style="margin-top:8px;color:#334155;">'
            f'{query_row["magnification"]}×  ·  patch {idx + 1:,} / {len(manifest):,}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

else:  # Upload image
    uploaded = st.file_uploader(
        "Upload a tissue patch (PNG or JPEG)",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        with st.spinner("Encoding with CLIP…"):
            query_emb = encoder.encode([img])[0]
        with img_col:
            with st.container(border=True):
                st.image(img, use_container_width=True)
        with meta_col:
            st.markdown(
                '<div style="color:#94A3B8;font-size:13px;padding:4px 0;">'
                'Uploaded patch — encoded with CLIP ViT-B/16.'
                '</div>',
                unsafe_allow_html=True,
            )
    else:
        with img_col:
            st.markdown(
                '<div style="height:160px;background:#1A1D27;border:2px dashed #1E2030;'
                'border-radius:10px;display:flex;align-items:center;justify-content:center;'
                'color:#334155;font-size:13px;">Drop a patch here</div>',
                unsafe_allow_html=True,
            )


# ── Results grid ──────────────────────────────────────────────────────────────

if query_emb is not None:
    results = _search(query_emb, index, manifest, k=top_k, exclude_id=query_iid)

    # Accuracy stats for this query
    n_correct = int((results["label"] == query_row["label"]).sum()) if query_row else 0
    _section(
        f"TOP-{top_k} RESULTS",
        suffix=f"{n_correct}/{top_k} correct label" if query_row else "",
    )

    n_cols = min(4, top_k)
    for row_start in range(0, len(results), n_cols):
        chunk = results.iloc[row_start : row_start + n_cols]
        cols  = st.columns(n_cols, gap="small")

        for col, (_, r) in zip(cols, chunk.iterrows()):
            rank     = row_start + list(chunk.index).index(r.name) + 1
            sim      = r["_sim"]
            label    = r["label"]
            correct  = query_row is not None and label == query_row["label"]
            rank_cls = "rank-1" if rank == 1 else "rank"

            with col:
                with st.container(border=True):
                    st.image(str(ROOT / r["path"]), use_container_width=True)
                    st.markdown(
                        f'<div style="padding:0 2px 2px;">'
                        # Rank + correctness indicator on one line
                        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                        f'<span class="{rank_cls}">#{rank}</span>'
                        f'<span style="font-size:13px;color:{"#22C55E" if correct else "#EF4444"};">'
                        f'{"✓" if correct else "✗"}</span>'
                        f'</div>'
                        # Similarity score + bar
                        f'<div class="coord" style="margin-top:3px;">{sim:.3f} sim</div>'
                        f'{_sim_bar(sim)}'
                        # Label chip
                        f'{_label_chip(label)}'
                        # Slide + coords
                        f'<div class="coord" style="margin-top:5px;color:#1E2A3A;">'
                        f'{r["slide_id"][:18]}<br/>({r["x"]}, {r["y"]})</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ── Experiment metrics ────────────────────────────────────────────────────────

if not metrics_df.empty:
    # Use only the three canonical seeds (exclude any bugfix/re-run rows)
    canonical = metrics_df[metrics_df["seed"].isin([42, 123, 2024])].copy()

    _section(
        "EXPERIMENT METRICS",
        suffix=f"{len(canonical)} runs · seeds 42 / 123 / 2024",
    )

    kc1, kc2, kc3, kc4 = st.columns(4, gap="medium")

    def _kpi(col: st.delta_generator.DeltaGenerator, mean: float, std: float, label: str) -> None:
        with col:
            st.markdown(
                f'<div class="kpi">'
                f'<div class="kpi-value">{mean:.3f}</div>'
                f'<div class="kpi-std">± {std:.3f}</div>'
                f'<div class="kpi-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    _kpi(kc1, canonical["top1"].mean(),      canonical["top1"].std(),      "top-1 accuracy")
    _kpi(kc2, canonical["top5"].mean(),      canonical["top5"].std(),      "top-5 accuracy")
    _kpi(kc3, canonical["map_at_10"].mean(), canonical["map_at_10"].std(), "mAP @ 10")
    _kpi(kc4, canonical["random_baseline_top5"].mean(), 0.0,               "random baseline")

    with st.expander("Raw experiment log"):
        display_cols = ["uid", "seed", "encoder", "top1", "top5", "map_at_10", "embed_time_s", "notes"]
        st.dataframe(
            canonical[display_cols].reset_index(drop=True),
            use_container_width=True,
        )
