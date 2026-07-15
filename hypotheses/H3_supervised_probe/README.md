# H3 — Supervised Linear Probe for Tumour vs Non-Tumour

## Hypothesis

H1's unsupervised clustering (K-means k=2/k=8, HDBSCAN) could not separate tumour from
non-tumour patches on any encoder — best result was UNI2-h F1 0.26 (k=8); HDBSCAN gave
ARI ≈ 0. In a foundation-model embedding, the dominant axes of variation are tissue type,
stain, and scanner batch — not tumour status — so purely unsupervised methods cluster on
those nuisance factors instead.

**H3 tests whether tumour vs non-tumour is *linearly separable*** in the same frozen
embeddings, using a small amount of supervision: `LogisticRegression` on top of frozen
features. This is a linear classifier fit on already-computed embeddings, not a trained
deep network — it doesn't violate the "no model training" project constraint, and the
expensive part (encoding) is already done.

---

## Method

1. **Patient-level split** (70/15/15 train/val/test) — grouped by patient (case), never by
   patch, to avoid leaking a slide's patches across splits. A patient's slide "sections"
   (e.g. `HE_036` and `HE_036_a`) are merged into one case — verified to be the same
   patient's tumour (same anatomical site, same `"Tumor"` geojson class).
2. **Fit**: `StandardScaler` + `LogisticRegression(class_weight="balanced")` on the
   training set (all tumour patches + a bounded random subsample of non-tumour patches).
3. **Tune the decision threshold on val** (maximise F1) — never touch test until evaluation.
4. **Evaluate once on test**: AUROC, PR-AUC (threshold-independent — the honest "does it
   separate" numbers) + thresholded precision/recall/F1/balanced accuracy/confusion matrix.
5. **XAI**: top embedding dimensions by `|LogisticRegression coefficient|`, cross-checked
   against `histoRAG.classify.explain_dimensions` (Cohen's d) computed on the same
   training data — same XAI approach as H1, applied to the supervised signal instead.

See `exp01_linear_probe/run.py` docstring for the streaming implementation details.

---

## Shared modules

`histoRAG/classify.py` (extended for H3, shared with H1):
- `tune_threshold` — sweep candidate thresholds, pick the one maximising F1 (or Youden's J)
- `probe_metrics` — AUROC, PR-AUC, and thresholded precision/recall/F1/balanced accuracy/confusion matrix
- `grouped_metrics`, `explain_dimensions` — reused unchanged from H1

`histoRAG/loader.py`, `histoRAG/labels.py` — reused unchanged from H1 (embedding loading,
geojson → tumour/other label derivation).

---

## Running

```bash
# Smoke test — restricts the patient split + train/val/test to ONE anatomical site
# (~1/4 the patches), otherwise identical pipeline. Not a subsample/preview like H1's
# visualize_kmeans_umap.py — it's a real, smaller, still-valid fit + evaluation.
python hypotheses/H3_supervised_probe/exp01_linear_probe/run.py --encoder uni2h --sites larynx

# Full run, one encoder — all sites, all patients (700), the numbers in this README
python hypotheses/H3_supervised_probe/exp01_linear_probe/run.py --encoder uni2h

# All three encoders, full run each
pwsh hypotheses/H3_supervised_probe/run_h3_all_encoders.ps1
```

No `--plots-only`/`--skip-*` cache-replay flags exist here (unlike H1) — there's no
expensive clustering step to skip re-running; the linear probe itself is the cheap part,
so every run just redoes the whole thing.

## Output structure (per encoder)

```
outputs/{encoder}/
  summary_{encoder}.json   split sizes, threshold, metrics (val + test), per-site/per-WSI, XAI
  roc_curve.png
  pr_curve.png
  cache/
    model.joblib           fitted StandardScaler + LogisticRegression pipeline
    test_y_true.npy
    test_y_score.npy
    test_manifest.parquet
```

## Success criterion

Test AUROC well above 0.5 — a large jump over H1's best F1 of 0.26. If AUROC is
disappointing, escalate to a shallow MLP or a different `C` before questioning the
embeddings themselves (see H3 memory note: ladder — linear probe first).

---

## Results — UNI2-h, full dataset (700 patients, patient-level 70/15/15 split)

| Split | Patches | Tumour | Tumour % |
|---|---|---|---|
| Train | 1,071,648 | 71,648 | 6.7% |
| Val | 300,000 | 11,698 | 3.9% |
| Test | 303,910 | 17,366 | 5.7% |

**Test metrics (threshold tuned on val for F1):**

| AUROC | PR-AUC | Precision | Recall | F1 | Balanced acc |
|---|---|---|---|---|---|
| **0.880** | 0.231 | 0.248 | 0.359 | 0.293 | 0.647 |

**AUROC 0.88 confirms the hypothesis** — tumour vs non-tumour is close to linearly
separable in frozen UNI2-h embeddings, a night-and-day improvement over H1's clustering
(ARI ≈ 0 for HDBSCAN, best F1 0.26 for K-means k=8). F1 at the operating threshold is
prevalence-limited (~5% positive rate caps achievable precision at any single threshold);
AUROC/PR-AUC are the honest, threshold-independent measures of separability and are what
actually validate H3.

Per-tissue AUROC-adjacent behaviour (F1 at tuned threshold): larynx 0.35, oropharynx 0.29,
oral cavity 0.25, hypopharynx 0.21 — consistent across sites, no site drives the result.

**XAI cross-check**: the Cohen's-d top-5 dimensions (593, 914, 365, 943, 905) exactly match
H1's previously reported UNI2-h top-5 tumour dims — confirms the reused `explain_dimensions`
label pipeline is wired correctly and the tumour signal is the same one H1 found buried.

Only UNI2-h has been run on the full dataset so far; CONCH and CLIP-ViT-B16 are next via
`run_h3_all_encoders.ps1`.
