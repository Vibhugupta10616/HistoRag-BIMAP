"""Streamlit demo UI for HistoRAG patch retrieval.

Run:
    streamlit run src/histoRAG/viz/streamlit_app.py

Features:
- Upload a query patch (PNG/JPEG) or pick a random gallery patch
- Retrieve top-k most similar patches from the FAISS index
- Display results as an image grid with slide ID, coordinates, label, similarity score
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HistoRAG — Patch Retrieval Demo",
    page_icon="🔬",
    layout="wide",
)

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.title("HistoRAG")
st.sidebar.markdown("**Retrieval-Augmented Histopathology Atlas**")
st.sidebar.markdown("---")

index_path = st.sidebar.text_input(
    "FAISS index path",
    value="data/indexes/phase0_mvp.faiss",
)
manifest_path = st.sidebar.text_input(
    "Manifest path",
    value="data/patches/manifest.parquet",
)
top_k = st.sidebar.slider("Top-k results", min_value=1, max_value=20, value=9)
encoder_name = st.sidebar.selectbox("Encoder", ["clip-vitb16"])
device = st.sidebar.selectbox("Device", ["auto", "cuda", "cpu"])

st.sidebar.markdown("---")
st.sidebar.caption("Phase 0 MVP — CLIP ViT-B/16 + FAISS flat IP")


# ── Cached resource loading ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading encoder…")
def load_encoder(name: str, dev: str):
    from histoRAG.encoders.registry import get_encoder
    return get_encoder(name, device=dev)


@st.cache_resource(show_spinner="Loading FAISS index…")
def load_index(path: str):
    from histoRAG.index.faiss_index import FaissFlatIP
    return FaissFlatIP.load(path)


@st.cache_data(show_spinner="Loading manifest…")
def load_manifest(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


# ── Main layout ───────────────────────────────────────────────────────────────
st.title("🔬 HistoRAG — Histopathology Patch Retrieval")

# Check if index and manifest exist
index_exists = Path(index_path).exists()
manifest_exists = Path(manifest_path).exists()

if not index_exists or not manifest_exists:
    st.warning(
        "Index or manifest not found. Run the MVP pipeline first:\n\n"
        "```bash\npython scripts/run_mvp.py --config configs/phase0_mvp.yaml --seed 42\n```"
    )
    st.stop()

# Load resources
encoder = load_encoder(encoder_name, device)
index = load_index(index_path)
manifest = load_manifest(manifest_path)

# Build patch_id → manifest row lookup
id_to_row = manifest.set_index("patch_id")

# Build integer ID ↔ patch_id mapping (FAISS stores int64 IDs)
patch_ids = manifest["patch_id"].tolist()
patch_id_to_int = {pid: i for i, pid in enumerate(patch_ids)}
int_to_patch_id = {i: pid for i, pid in enumerate(patch_ids)}

st.markdown(f"**Index:** {index.ntotal} patches · **Manifest:** {len(manifest)} rows")

# ── Query input ───────────────────────────────────────────────────────────────
col_upload, col_random = st.columns([2, 1])

query_img: Image.Image | None = None
query_patch_id: str | None = None

with col_upload:
    uploaded = st.file_uploader("Upload a query patch", type=["png", "jpg", "jpeg"])
    if uploaded:
        query_img = Image.open(uploaded).convert("RGB")
        query_patch_id = uploaded.name

with col_random:
    st.markdown("#### or pick random")
    if st.button("🎲 Random gallery patch"):
        row = manifest.sample(1, random_state=random.randint(0, 99999)).iloc[0]
        query_img = Image.open(row["path"]).convert("RGB")
        query_patch_id = row["patch_id"]
        st.session_state["random_row"] = row.to_dict()

# Show stored random patch between button clicks
if query_img is None and "random_row" in st.session_state:
    row = st.session_state["random_row"]
    query_img = Image.open(row["path"]).convert("RGB")
    query_patch_id = row["patch_id"]

# ── Retrieval ─────────────────────────────────────────────────────────────────
if query_img is not None:
    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 3])

    with res_col1:
        st.subheader("Query")
        st.image(query_img, use_container_width=True)
        if query_patch_id and query_patch_id in id_to_row.index:
            qrow = id_to_row.loc[query_patch_id]
            st.caption(
                f"**{qrow['slide_id']}** · ({qrow['x']},{qrow['y']}) · label: `{qrow['label']}`"
            )

    with res_col2:
        st.subheader(f"Top-{top_k} most similar patches")
        with st.spinner("Searching…"):
            query_emb = encoder.encode([query_img])  # (1, D)
            sims, ids = index.search(query_emb, k=top_k + 1)  # +1 to skip self-match

        # Filter out the query patch itself (by integer ID if present)
        query_int_id = patch_id_to_int.get(query_patch_id, -1)
        results = [
            (int_to_patch_id[int(i)], float(s))
            for i, s in zip(ids[0], sims[0])
            if int(i) != query_int_id and int(i) in int_to_patch_id
        ][:top_k]

        # Display as grid (3 columns)
        n_cols = 3
        n_rows = math.ceil(len(results) / n_cols)
        for r in range(n_rows):
            cols = st.columns(n_cols)
            for c in range(n_cols):
                idx = r * n_cols + c
                if idx >= len(results):
                    break
                pid, score = results[idx]
                if pid not in id_to_row.index:
                    continue
                row = id_to_row.loc[pid]
                with cols[c]:
                    try:
                        img = Image.open(row["path"]).convert("RGB")
                        st.image(img, use_container_width=True)
                    except Exception:
                        st.error("Image not found")
                    st.caption(
                        f"sim={score:.3f} · `{row['label']}`\n"
                        f"{row['slide_id']} ({row['x']},{row['y']})"
                    )
