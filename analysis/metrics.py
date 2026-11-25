"""Scoring: counts, micro-F1, bootstrap CIs, paired tests and FDR.

Every metric is built from per-article (tp, fp, fn, gold_empty) counts, so F1,
precision, recall and Jaccard come from the same numbers. Tract names are
compared as vocabulary ids (see vocabulary.py); other list fields as normalised
strings.
"""

from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

N_BOOT = 10_000      # smallest attainable two-sided p = 2/(B+1) = 0.0002

# ── COUNTING ──────────────────────────────────────────────────────────────────


def row_counts(pred_ids: Sequence[Set[str]],
               gold_ids: Sequence[Set[str]]) -> np.ndarray:
    """Per-row (tp, fp, fn, gold_empty, pred_empty) as a float array."""
    rows = []
    for p, g in zip(pred_ids, gold_ids):
        rows.append((len(p & g), len(p - g), len(g - p),
                     float(not g), float(not p)))
    return np.asarray(rows, dtype=float)


def _f1(tp: float, fp: float, fn: float) -> float:
    den = 2 * tp + fp + fn
    return float(2 * tp / den) if den else 0.0


def micro_f1(counts: np.ndarray, skip_empty_gold: bool = False) -> float:
    """Micro-F1; `skip_empty_gold` scores only articles with reference labels."""
    sel = counts[counts[:, 3] == 0] if skip_empty_gold else counts
    if not len(sel):
        return 0.0
    tp, fp, fn = sel[:, 0].sum(), sel[:, 1].sum(), sel[:, 2].sum()
    return _f1(tp, fp, fn)


def precision_recall(counts: np.ndarray,
                     skip_empty_gold: bool = False) -> Tuple[float, float]:
    sel = counts[counts[:, 3] == 0] if skip_empty_gold else counts
    if not len(sel):
        return 0.0, 0.0
    tp, fp, fn = sel[:, 0].sum(), sel[:, 1].sum(), sel[:, 2].sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return float(prec), float(rec)


def prefilter_metrics(counts: np.ndarray) -> Dict[str, float]:
    """Does the model return no tracts when the reference has none (all rows)?"""
    gold_empty, pred_empty = counts[:, 3] == 1, counts[:, 4] == 1
    tp = float((gold_empty & pred_empty).sum())        # correctly silent
    fn = float((gold_empty & ~pred_empty).sum())       # hallucinated
    fp = float((~gold_empty & pred_empty).sum())       # wrongly silent
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spurious = float(counts[gold_empty][:, 1].sum())   # tracts invented on empty gold
    return {"precision": prec, "recall": rec,
            "f1": _f1(tp, fp, fn) if (tp or fp or fn) else 0.0,
            "n_gold_empty": tp + fn, "hallucinated_rows": fn,
            "spurious_tracts": spurious}


# ── BOOTSTRAP ─────────────────────────────────────────────────────────────────


def _boot_f1(counts: np.ndarray, idx: np.ndarray,
             skip_empty_gold: bool) -> np.ndarray:
    sel = counts[counts[:, 3] == 0] if skip_empty_gold else counts
    agg = sel[idx].sum(axis=1)
    tp, fp, fn = agg[:, 0], agg[:, 1], agg[:, 2]
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)


def bootstrap_ci(counts: np.ndarray, n_boot: int = N_BOOT, seed: int = 0,
                 skip_empty_gold: bool = False) -> Tuple[float, float]:
    sel = counts[counts[:, 3] == 0] if skip_empty_gold else counts
    if not len(sel):
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sel), (n_boot, len(sel)))
    dist = _boot_f1(counts, idx, skip_empty_gold)
    lo, hi = np.percentile(dist, [2.5, 97.5])
    return float(lo), float(hi)


def paired_test(counts_a: np.ndarray, counts_b: np.ndarray,
                n_boot: int = N_BOOT, seed: int = 0,
                skip_empty_gold: bool = False) -> Dict[str, float]:
    """Paired bootstrap of F1(B) - F1(A) on the same articles.

    Two-sided p with the add-one correction (Davison & Hinkley 1997; North et
    al. 2002): p = min(2 (1 + r) / (1 + B), 1), r = the smaller tail count.
    When r = 0 the p-value sits at its floor 2/(B+1); report it as p < 0.001.
    """
    sel_a = counts_a[counts_a[:, 3] == 0] if skip_empty_gold else counts_a
    sel_b = counts_b[counts_b[:, 3] == 0] if skip_empty_gold else counts_b
    if len(sel_a) != len(sel_b):
        raise ValueError(f"unpaired rows: {len(sel_a)} vs {len(sel_b)}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sel_a), (n_boot, len(sel_a)))
    diff = (_boot_f1(counts_b, idx, skip_empty_gold)
            - _boot_f1(counts_a, idx, skip_empty_gold))

    n_extreme = min(int((diff <= 0).sum()), int((diff >= 0).sum()))
    p = min(2.0 * (1 + n_extreme) / (1 + n_boot), 1.0)

    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {"delta": micro_f1(counts_b, skip_empty_gold) - micro_f1(counts_a, skip_empty_gold),
            "ci_lower": float(lo), "ci_upper": float(hi),
            "p_value": float(p), "n_boot": n_boot,
            "n_extreme": n_extreme,
            "p_floor": 2.0 / (1 + n_boot)}


def benjamini_hochberg(pvals: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg q-values, monotone in p and clipped at 1."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):                 # largest p first
        i = order[rank - 1]
        running = min(running, pvals[i] * m / rank)
        q[i] = min(running, 1.0)
    return q


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ── TRACTS ────────────────────────────────────────────────────────────────────


def evaluate(pred_cells: Iterable, gold_cells: Iterable, lut) -> Dict[str, object]:
    """Resolve both sides through the LUT and return the full metric set."""
    gold_ids = [lut.resolve_cell(c) for c in gold_cells]
    pred_ids = [lut.resolve_cell(c) for c in pred_cells]
    counts = row_counts(pred_ids, gold_ids)

    prec_all, rec_all = precision_recall(counts, skip_empty_gold=False)
    prec_ne, rec_ne = precision_recall(counts, skip_empty_gold=True)
    lo, hi = bootstrap_ci(counts, skip_empty_gold=False)

    return {
        "primary_f1": micro_f1(counts, skip_empty_gold=False),
        "primary_ci": (lo, hi),
        "primary_precision": prec_all,
        "primary_recall": rec_all,
        "tract_f1": micro_f1(counts, skip_empty_gold=True),
        "tract_precision": prec_ne,
        "tract_recall": rec_ne,
        "prefilter": prefilter_metrics(counts),
        "n_rows": len(counts),
        "n_gold_empty": int((counts[:, 3] == 1).sum()),
        "counts": counts,
    }


# ── OTHER FIELDS ──────────────────────────────────────────────────────────────

SIM_THRESH = 0.95    # string similarity that counts as the same label (non-tract list fields)

EMPTY_TOKENS = {"", "none", "n.a.", "na", "n a", "n/a", "null", "_", "-", "nan",
                "not reported", "unknown"}


def norm(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return " ".join(str(x).lower().strip().split())


def is_empty(s) -> bool:
    return norm(s) in EMPTY_TOKENS


def clean_split(x):
    s = norm(x)
    if is_empty(s):
        return []
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    return [norm(p) for p in parts if p and not is_empty(p)]


CANON_DISEASE = {"alzheimers disease": "alzheimers disease", "ad": "alzheimers disease",
                 "parkinson disease": "parkinson disease"}
CANON_TYPE = {}


def canon(v, cmap):
    return cmap.get(norm(v), norm(v))


def best_match(pred, gold_list, cmap):
    """Exact canonical hit, else the closest reference label if similarity >= 0.95."""
    p_can = canon(pred, cmap)
    if p_can in gold_list:
        return p_can
    best, score = None, 0.0
    for g in gold_list:
        r = SequenceMatcher(None, norm(pred), g).ratio()
        if r > score:
            best, score = g, r
    return best if score >= SIM_THRESH else norm(pred)


def binary_counts(gold, pred, positive="yes"):
    rows = []
    for g, p in zip(gold, pred):
        gy, py = norm(g) == positive, norm(p) == positive
        rows.append((float(gy and py), float(py and not gy), float(gy and not py),
                     float(not gy)))
    return np.asarray(rows, dtype=float)


def multilabel_counts(gold, pred, cmap):
    rows = []
    for g, p in zip(gold, pred):
        g_set = set(canon(v, cmap) for v in clean_split(g))
        p_raw = clean_split(p)
        p_set = set(best_match(v, sorted(g_set), cmap) for v in p_raw) if g_set else set(p_raw)
        rows.append((len(p_set & g_set), len(p_set - g_set), len(g_set - p_set),
                     float(not g_set)))
    return np.asarray(rows, dtype=float)


def lut_counts(gold, pred, lut):
    g_ids = [lut.resolve_cell(c) for c in gold]
    p_ids = [lut.resolve_cell(c) for c in pred]
    return row_counts(p_ids, g_ids)[:, :4]


def _agg(sel):
    return sel[:, 0].sum(), sel[:, 1].sum(), sel[:, 2].sum()


def point(counts, metric):
    tp, fp, fn = _agg(counts)
    if metric == "f1":
        den = 2 * tp + fp + fn
        return float(2 * tp / den) if den else 0.0
    den = tp + fp + fn
    return float(tp / den) if den else 0.0


def boot_ci(counts, metric, seed=0, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(counts), (n_boot, len(counts)))
    agg = counts[idx].sum(axis=1)
    tp, fp, fn = agg[:, 0], agg[:, 1], agg[:, 2]
    den = (2 * tp + fp + fn) if metric == "f1" else (tp + fp + fn)
    num = (2 * tp) if metric == "f1" else tp
    dist = np.where(den > 0, num / np.maximum(den, 1), 0.0)
    lo, hi = np.percentile(dist, [2.5, 97.5])
    return float(lo), float(hi)
