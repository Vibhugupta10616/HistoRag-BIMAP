"""Unit tests for retrieval metrics and query/gallery split protocol."""
from __future__ import annotations

import numpy as np
import pytest

from histoRAG.eval.metrics import mean_average_precision, random_baseline, top_k_accuracy
from histoRAG.eval.protocol import slide_leave_out, stratified_within_slide


# ── top_k_accuracy ────────────────────────────────────────────────────────────

def test_top_k_perfect():
    retrieved = np.array([["A", "B"], ["B", "A"], ["C", "D"]])
    queries = np.array(["A", "B", "C"])
    assert top_k_accuracy(retrieved, queries, k=1) == 1.0


def test_top_k_zero():
    retrieved = np.array([["B", "C"], ["A", "C"], ["A", "B"]])
    queries = np.array(["A", "B", "C"])
    assert top_k_accuracy(retrieved, queries, k=1) == 0.0


def test_top_k_partial():
    # Query 0: hit at rank 0; Query 1: hit at rank 1; Query 2: hit at rank 0
    retrieved = np.array([["A", "B"], ["C", "B"], ["C", "D"]])
    queries = np.array(["A", "B", "C"])
    assert top_k_accuracy(retrieved, queries, k=1) == pytest.approx(2 / 3)
    assert top_k_accuracy(retrieved, queries, k=2) == pytest.approx(1.0)


# ── mean_average_precision ────────────────────────────────────────────────────

def test_map_perfect():
    retrieved = np.array([["A", "B"], ["B", "A"]])
    queries = np.array(["A", "B"])
    assert mean_average_precision(retrieved, queries, k=2) == pytest.approx(1.0)


def test_map_no_hits():
    retrieved = np.array([["B", "C"], ["A", "C"]])
    queries = np.array(["A", "B"])
    assert mean_average_precision(retrieved, queries, k=1) == pytest.approx(0.0)


# ── random_baseline ───────────────────────────────────────────────────────────

def test_random_baseline_two_classes():
    # P(hit in 1 draw from 2 classes) = 0.5
    assert random_baseline(n_classes=2, k=1) == pytest.approx(0.5)


def test_random_baseline_certain():
    # Only 1 class → always a hit; 1 - (0/1)^1 = 1.0
    assert random_baseline(n_classes=1, k=1) == pytest.approx(1.0)


# ── stratified_within_slide ───────────────────────────────────────────────────

def test_split_covers_all(dummy_manifest):
    q, g = stratified_within_slide(dummy_manifest, query_frac=0.2, seed=42)
    assert len(q) + len(g) == len(dummy_manifest)


def test_split_no_overlap(dummy_manifest):
    q, g = stratified_within_slide(dummy_manifest, query_frac=0.2, seed=42)
    assert len(set(q).intersection(set(g))) == 0


def test_split_reproducible(dummy_manifest):
    q1, _ = stratified_within_slide(dummy_manifest, seed=42)
    q2, _ = stratified_within_slide(dummy_manifest, seed=42)
    assert list(q1) == list(q2)


def test_split_different_seeds(dummy_manifest):
    q1, _ = stratified_within_slide(dummy_manifest, seed=42)
    q2, _ = stratified_within_slide(dummy_manifest, seed=99)
    assert list(q1) != list(q2)


# ── slide_leave_out ───────────────────────────────────────────────────────────

def test_leave_out_correct_slides(dummy_manifest):
    q, g = slide_leave_out(dummy_manifest, held_out_slides=["slide_001"])
    q_slides = dummy_manifest.loc[q, "slide_id"].unique().tolist()
    assert q_slides == ["slide_001"]
    g_slides = dummy_manifest.loc[g, "slide_id"].unique().tolist()
    assert "slide_001" not in g_slides
