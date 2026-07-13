"""
H3 Experiment 01 — Supervised Linear Probe (LogisticRegression)

H1's unsupervised clustering (K-means, HDBSCAN) could not separate tumour from
non-tumour patches on any encoder (best F1 0.26 — see H1 README). Diagnosis:
the dominant axes of a foundation-model embedding are tissue/stain/scanner
variation, not tumour status, so clustering finds the wrong structure. This
experiment tests whether a small amount of supervision — a logistic
regression on the same frozen embeddings — separates the classes cleanly.

Pipeline:
    1. Patient-level split: group slides into patients (case = slide_id with
       any trailing section suffix like "_a" stripped), split cases 70/15/15
       into train/val/test. Prevents leakage from patches of the same slide
       appearing in both train and test.
    2. Pass 1: stream train+val slides. Keep all tumour (positive) patches;
       subsample non-tumour (negative) patches to bound training set size.
       Val patches are also capped, for cheap threshold tuning.
    3. Fit StandardScaler + LogisticRegression(class_weight="balanced") on
       the assembled training set.
    4. Tune the decision threshold on val (maximise F1, by default).
    5. Pass 2: stream test slides, score with the fitted probe, evaluate
       once — AUROC, PR-AUC, and thresholded metrics (never touched until
       here).
    6. XAI: top embedding dimensions by |coefficient|, cross-checked against
       classify.explain_dimensions (Cohen's d) on the training set.

Usage:
    python run.py --encoder uni2h
    python run.py --encoder uni2h --sites larynx   # fast smoke test
"""

import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive; works without a display / on servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from histoRAG.loader import count_encoder_patches, detect_patch_size, iter_encoder
from histoRAG.labels import tumour_labels_from_geojson
from histoRAG.classify import explain_dimensions, grouped_metrics, probe_metrics, tune_threshold


def patient_id_from_slide(slide_id: str) -> str:
    """Strip a trailing single-letter section suffix (e.g. "_a") to get the
    patient/case id. HE_036 and HE_036_a are the same patient — a second
    tissue section of the same tumour, verified against site + geojson class."""
    return re.sub(r"_[a-z]$", "", slide_id)


def build_patient_split(
    slide_ids: list[str],
    val_frac: float,
    test_frac: float,
    random_state: int,
) -> dict[str, str]:
    """Assign each slide to train/val/test by patient (case), not by patch.

    Returns {slide_id: "train"|"val"|"test"}.
    """
    case_ids = sorted({patient_id_from_slide(s) for s in slide_ids})

    trainval_cases, test_cases = train_test_split(
        case_ids, test_size=test_frac, random_state=random_state
    )
    val_rel = val_frac / (1 - test_frac)
    train_cases, val_cases = train_test_split(
        trainval_cases, test_size=val_rel, random_state=random_state
    )

    train_set, val_set, test_set = set(train_cases), set(val_cases), set(test_cases)
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    case_to_split = {c: "train" for c in train_set}
    case_to_split.update({c: "val" for c in val_set})
    case_to_split.update({c: "test" for c in test_set})

    return {s: case_to_split[patient_id_from_slide(s)] for s in slide_ids}


def _plot_curves(out_dir: Path, y_true: np.ndarray, y_score: np.ndarray, encoder: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC — {encoder} (H3 linear probe)")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall — {encoder} (H3 linear probe)")
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve.png", dpi=150)
    plt.close(fig)


def run(
    config_path: str | Path,
    encoder_override: str | None = None,
    embeddings_root_override: str | None = None,
    geojson_dir_override: str | None = None,
    sites_override: list[str] | None = None,
) -> None:
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    tag = cfg["experiment"]["name"]
    encoder = encoder_override or cfg["encoder"]
    geojson_dir = Path(geojson_dir_override or cfg["inputs"]["geojson_dir"])
    emb_root = embeddings_root_override or cfg["inputs"]["embeddings_root"]
    sites = sites_override or cfg["inputs"].get("sites")
    out_dir = Path(cfg["outputs"]["dir"]) / encoder
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    val_frac = cfg["split"].get("val_frac", 0.15)
    test_frac = cfg["split"].get("test_frac", 0.15)
    random_state = cfg["split"].get("random_state", 42)

    p = cfg["params"]
    C = p.get("C", 1.0)
    max_iter = p.get("max_iter", 1000)
    class_weight = p.get("class_weight", "balanced")
    max_train_neg = p.get("max_train_neg", 1_000_000)
    max_val_patches = p.get("max_val_patches", 300_000)
    assumed_prevalence = p.get("assumed_tumour_prevalence", 0.05)
    threshold_objective = p.get("threshold_objective", "f1")
    patch_size_cfg = p.get("patch_size")

    if not geojson_dir.exists():
        raise FileNotFoundError(f"geojson_dir not found: {geojson_dir}")

    patch_size = patch_size_cfg or detect_patch_size(encoder, emb_root)
    print(f"\n[{tag}] encoder={encoder}  patch_size={patch_size}  sites={sites or 'all'}")

    # ── Step 1: patient-level split ─────────────────────────────────────────
    patch_counts = count_encoder_patches(encoder, emb_root, sites=sites)
    slide_ids = sorted(patch_counts)
    slide_to_split = build_patient_split(slide_ids, val_frac, test_frac, random_state)

    n_cases = len({patient_id_from_slide(s) for s in slide_ids})
    for split_name in ("train", "val", "test"):
        n_slides = sum(1 for s in slide_ids if slide_to_split[s] == split_name)
        n_patches = sum(patch_counts[s] for s in slide_ids if slide_to_split[s] == split_name)
        print(f"[{tag}] split={split_name:5s}  slides={n_slides:4d}  patches={n_patches:,}")
    print(f"[{tag}] total patients={n_cases}  total slides={len(slide_ids)}")

    total_train_patches = sum(
        patch_counts[s] for s in slide_ids if slide_to_split[s] == "train"
    )
    # ponytail: neg_keep_prob assumes global tumour prevalence (from H1 findings)
    # rather than an exact per-slide count — a cheap approximation. The hard
    # per-slide budget check below still guarantees max_train_neg is never
    # exceeded even if the assumption is off. Upgrade to an exact count-first
    # pass only if class balance in practice drifts noticeably from 5%.
    expected_train_neg = max(1, int(total_train_patches * (1 - assumed_prevalence)))
    neg_keep_prob = min(1.0, max_train_neg / expected_train_neg)
    print(f"[{tag}] neg_keep_prob={neg_keep_prob:.4f} (train negative subsample rate)")

    # ── Step 2: Pass 1 — stream train + val slides ──────────────────────────
    rng = np.random.default_rng(random_state)
    train_X_parts, train_y_parts = [], []
    val_X_parts, val_y_parts, val_manifest_parts = [], [], []
    n_train_neg_kept = 0
    n_val_kept = 0

    for emb, slide_df in iter_encoder(encoder, emb_root, sites=sites):
        slide_id = slide_df["slide_id"].iloc[0]
        split = slide_to_split[slide_id]
        if split == "test":
            continue

        tumour_lbl = tumour_labels_from_geojson(slide_df, geojson_dir, patch_size=patch_size)
        y = (tumour_lbl == "tumour").astype(int).values

        if split == "train":
            pos_idx = np.where(y == 1)[0]
            if len(pos_idx):
                train_X_parts.append(emb[pos_idx])
                train_y_parts.append(y[pos_idx])

            neg_idx = np.where(y == 0)[0]
            if len(neg_idx) and n_train_neg_kept < max_train_neg:
                keep_idx = neg_idx[rng.random(len(neg_idx)) < neg_keep_prob]
                budget = max_train_neg - n_train_neg_kept
                if len(keep_idx) > budget:
                    keep_idx = rng.choice(keep_idx, budget, replace=False)
                if len(keep_idx):
                    train_X_parts.append(emb[keep_idx])
                    train_y_parts.append(y[keep_idx])
                    n_train_neg_kept += len(keep_idx)

        else:  # val — capped random subsample, all classes
            if n_val_kept < max_val_patches:
                budget = max_val_patches - n_val_kept
                keep_idx = (
                    np.arange(len(y)) if len(y) <= budget
                    else rng.choice(len(y), budget, replace=False)
                )
                val_X_parts.append(emb[keep_idx])
                val_y_parts.append(y[keep_idx])
                val_manifest_parts.append(slide_df.iloc[keep_idx])
                n_val_kept += len(keep_idx)

        print(f"  [{split}] {slide_id}: {len(emb):>6,} patches | tumour={y.sum():>5,}")

    train_X = np.concatenate(train_X_parts).astype(np.float32)
    train_y = np.concatenate(train_y_parts)
    val_X = np.concatenate(val_X_parts).astype(np.float32)
    val_y = np.concatenate(val_y_parts)
    val_manifest = pd.concat(val_manifest_parts, ignore_index=True)
    del train_X_parts, train_y_parts, val_X_parts, val_y_parts, val_manifest_parts

    print(f"[{tag}] train: {len(train_X):,} patches ({train_y.sum():,} tumour) dim={train_X.shape[1]}")
    print(f"[{tag}] val:   {len(val_X):,} patches ({val_y.sum():,} tumour)")

    # ── Step 3: fit the linear probe ────────────────────────────────────────
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight=class_weight, C=C, max_iter=max_iter, random_state=random_state
        ),
    )
    print(f"[{tag}] Fitting StandardScaler + LogisticRegression...")
    pipe.fit(train_X, train_y)

    # ── Step 4: tune threshold on val ───────────────────────────────────────
    val_score = pipe.predict_proba(val_X)[:, 1]
    threshold = tune_threshold(val_y, val_score, objective=threshold_objective)
    val_metrics = probe_metrics(val_y, val_score, threshold)
    print(f"[{tag}] Tuned threshold={threshold:.4f} (val AUROC={val_metrics['auroc']:.3f}, "
          f"F1={val_metrics['f1']:.3f})")

    # ── Step 5: Pass 2 — stream test slides, evaluate once ──────────────────
    print(f"[{tag}] Pass 2 — scoring test slides...")
    test_scores, test_labels, test_manifest_parts = [], [], []
    for emb, slide_df in iter_encoder(encoder, emb_root, sites=sites):
        slide_id = slide_df["slide_id"].iloc[0]
        if slide_to_split[slide_id] != "test":
            continue
        tumour_lbl = tumour_labels_from_geojson(slide_df, geojson_dir, patch_size=patch_size)
        y = (tumour_lbl == "tumour").astype(int).values
        score = pipe.predict_proba(emb)[:, 1]
        test_scores.append(score)
        test_labels.append(y)
        test_manifest_parts.append(slide_df)
        print(f"  [test] {slide_id}: {len(emb):>6,} patches | tumour={y.sum():>5,}")

    test_score = np.concatenate(test_scores)
    test_label = np.concatenate(test_labels)
    test_manifest = pd.concat(test_manifest_parts, ignore_index=True)
    del test_scores, test_labels, test_manifest_parts

    metrics_tuned = probe_metrics(test_label, test_score, threshold)
    metrics_default = probe_metrics(test_label, test_score, 0.5)
    print(f"\n[{tag}] TEST  AUROC={metrics_tuned['auroc']:.4f}  PR-AUC={metrics_tuned['pr_auc']:.4f}  "
          f"F1={metrics_tuned['f1']:.4f}  precision={metrics_tuned['precision']:.4f}  "
          f"recall={metrics_tuned['recall']:.4f}  balanced_acc={metrics_tuned['balanced_accuracy']:.4f}")

    test_pred_tuned = (test_score >= threshold).astype(int)
    site_df = grouped_metrics(test_manifest, test_label, test_pred_tuned, group_col="site")
    wsi_df = grouped_metrics(test_manifest, test_label, test_pred_tuned, group_col="slide_id")
    print(f"\n[{tag}] Per-tissue results (tuned threshold):")
    print(site_df.to_string(index=False))

    # ── Step 6: XAI ──────────────────────────────────────────────────────────
    lr = pipe.named_steps["logisticregression"]
    coef = lr.coef_.ravel()
    coef_top_idx = np.argsort(np.abs(coef))[::-1][:20]
    xai_coef = {"top_dims": coef_top_idx.tolist(), "coefficients": coef[coef_top_idx].tolist()}
    xai_cohend = explain_dimensions(train_X, train_y, top_k=20)

    # ── Step 7: save ─────────────────────────────────────────────────────────
    joblib.dump(pipe, cache_dir / "model.joblib")
    np.save(cache_dir / "test_y_true.npy", test_label)
    np.save(cache_dir / "test_y_score.npy", test_score)
    test_manifest.to_parquet(cache_dir / "test_manifest.parquet", index=False)
    _plot_curves(out_dir, test_label, test_score, encoder)

    summary = {
        "experiment": tag,
        "encoder": encoder,
        "patch_size": patch_size,
        "embedding_dim": train_X.shape[1],
        "split": {
            "val_frac": val_frac, "test_frac": test_frac, "random_state": random_state,
            "n_patients": n_cases,
            "train_patches": int(len(train_X)), "train_tumour": int(train_y.sum()),
            "val_patches": int(len(val_X)), "val_tumour": int(val_y.sum()),
            "test_patches": int(len(test_label)), "test_tumour": int(test_label.sum()),
        },
        "params": {
            "C": C, "max_iter": max_iter, "class_weight": class_weight,
            "max_train_neg": max_train_neg, "neg_keep_prob": neg_keep_prob,
            "threshold_objective": threshold_objective,
        },
        "threshold": threshold,
        "metrics_val": val_metrics,
        "metrics_test_tuned": metrics_tuned,
        "metrics_test_default_0.5": metrics_default,
        "tissue_results": site_df.to_dict(orient="records"),
        "wsi_results": wsi_df.to_dict(orient="records"),
        "xai_coefficients": xai_coef,
        "xai_cohens_d": {k: v for k, v in xai_cohend.items()
                          if k in ("top_dims", "effect_sizes", "n_tumour", "n_other")},
        "outputs": ["roc_curve.png", "pr_curve.png"],
        "cache": ["cache/model.joblib", "cache/test_y_true.npy", "cache/test_y_score.npy",
                  "cache/test_manifest.parquet"],
    }
    summary_path = out_dir / f"summary_{encoder}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[{tag}] Done — summary saved -> {summary_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="H3 exp01 supervised linear probe.")
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--encoder", default=None, help="clip-vitb16 | conch | uni2h")
    parser.add_argument("--embeddings-root", default=None)
    parser.add_argument("--geojson-dir", default=None)
    parser.add_argument("--sites", default=None,
                        help="Comma-separated canonical site names for a fast smoke test, "
                             "e.g. 'larynx' or 'larynx,oral_cavity'. Default: all sites.")
    args = parser.parse_args()
    run(
        args.config,
        encoder_override=args.encoder,
        embeddings_root_override=args.embeddings_root,
        geojson_dir_override=args.geojson_dir,
        sites_override=args.sites.split(",") if args.sites else None,
    )
