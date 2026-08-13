"""White-matter tract F1 for GPT-4o-mini and GPT-5-mini, with and without the LUT.

The paper's headline result: the ontology-derived lookup table lifts GPT-5-mini
from 0.467 to 0.682, while GPT-4o-mini shows no significant change.

Output: results/tables/lut_ablation.csv
"""

import os
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

SIM_THRESH = 0.95
USE_SEMANTIC_MATCHING = True
DEDUP_PER_SAMPLE = False
SKIP_EMPTY_GOLD = True

N_BOOT = 1000
RANDOM_SEED = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "whitematter_dataset.csv")
GPT4_no_LUT = os.path.join(BASE_DIR, "predictions", "lut_ablation", "whitematter_no_lut_GPT_4.csv")
GPT5_no_LUT = os.path.join(BASE_DIR, "predictions", "lut_ablation", "whitematter_no_lut_GPT_5.csv")
GPT4_LUT = os.path.join(BASE_DIR, "predictions", "lut_ablation", "whitematter_fulltext_GPT4.csv")
GPT5_LUT = os.path.join(BASE_DIR, "predictions", "lut_ablation", "whitematter_fulltext_GPT_5.csv")
OUT_TABLE = os.path.join(BASE_DIR, "results", "tables", "lut_ablation.csv")

# TEXT NORMALIZATION
EMPTY_TOKENS = {
    "", "none", "n.a.", "na", "n a", "n/a", "null", "_", "-", "nan",
    "not reported", "unknown",
}

def normalize_text(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return " ".join(str(x).lower().strip().split())

def normalize_cmap(cmap: Dict[str, str]) -> Dict[str, str]:
    return {normalize_text(k): normalize_text(v) for k, v in cmap.items()}

def is_empty(s: str) -> bool:
    return normalize_text(s) in EMPTY_TOKENS

def clean_split(x) -> List[str]:
    s = normalize_text(x)
    if is_empty(s):
        return []
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    return [normalize_text(p) for p in parts if p and not is_empty(p)]

def canonicalize(value: str, cmap: Dict[str, str]) -> str:
    return cmap.get(normalize_text(value), normalize_text(value))

def canonicalize_list(values: List[str], cmap: Dict[str, str]) -> List[str]:
    return [canonicalize(v, cmap) for v in values if not is_empty(v)]

def seq_sim(a: str, b: str) -> float:
    a, b = normalize_text(a), normalize_text(b)
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return SequenceMatcher(None, a, b).ratio()

def best_semantic_match(pred: str, refs: List[str], cmap: Dict[str, str],
                        thresh: float = SIM_THRESH) -> Tuple[str, float]:
    if not refs:
        return None, 0.0
    p_can = normalize_text(pred)
    best_ref, best_score = None, 0.0
    for r in refs:
        r_can = canonicalize(r, cmap)
        if p_can == r_can:
            return r_can, 1.0
        score = seq_sim(p_can, r_can)
        if score > best_score:
            best_score, best_ref = score, r_can
    return (best_ref, best_score) if best_score >= thresh else (None, best_score)

# WMT CANONICAL MAP
canon_wmt = normalize_cmap({
    "corpus callosum": "corpus callosum",
    "corpus callosum - splenium": "corpus callosum - splenium",
    "cingulum": "cingulum",
    "uncinate fasciculus": "uncinate fasciculus",
    "fornix": "fornix",
    "genu": "genu",
    "inferior fronto occipital fasciculus": "inferior fronto occipital fasciculus",
    "superior longitudinal fasciculus": "superior longitudinal fasciculus",
    "corticospinal tract": "corticospinal tract",
    "forceps minor": "forceps minor",
    "ilf": "inferior longitudinal fasciculus",
    "ifo": "inferior fronto occipital fasciculus",
    "uncinate fasc.": "uncinate fasciculus",
    "slf": "superior longitudinal fasciculus",
    "cc": "corpus callosum",
    "cc- corpus callosum": "corpus callosum",   
})

# METRIC HELPERS
def _jaccard_samples(preds, refs) -> float:
    vals = []
    for p, r in zip(preds, refs):
        ps, rs = set(p), set(r)
        union = ps | rs
        vals.append(len(ps & rs) / len(union) if union else 1.0)
    return float(np.mean(vals)) if vals else 0.0

def compute_wmt_f1(preds: List[List[str]], refs: List[List[str]]) -> float:
    """Return micro-F1 for the WMT multilabel field."""
    if SKIP_EMPTY_GOLD:
        preds, refs = zip(*[(p, r) for p, r in zip(preds, refs) if r]) if any(refs) else ([], [])
        preds, refs = list(preds), list(refs)

    if not refs:
        return 0.0

    all_labels = sorted(set(x for sub in list(preds) + list(refs) for x in sub))
    if not all_labels:
        return 0.0

    mlb    = MultiLabelBinarizer(classes=all_labels)
    Y_true = mlb.fit_transform(refs)
    Y_pred = mlb.transform(preds)

    return float(f1_score(Y_true, Y_pred, average="micro", zero_division=0))

# EVALUATE ONE MODEL (WMT only) 
def evaluate_model(pred_path: str, model_label: str) -> Tuple[float, List[List[str]], List[List[str]]]:
    """
    Load predictions, match against ground truth on the WM Tracts field,
    and return the micro-F1 score, raw predictions, and raw references.
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {model_label}")
    print(f"  File: {pred_path}")
    print("=" * 60)

    golden_data    = pd.read_csv(DATA_RAW)
    predicted_data = pd.read_csv(pred_path)

    # Drop metadata columns from ground truth
    drop_cols = [
        "PMCID", "Open Source?", "Authors", "Citation", "First Author",
        "Which imaging modality was used? e.g electroencephalogram (EEG), Positron emission tomography (PET), Anatomical MRI, fMRI, diffusion MRI (dMRI) etc",
        "Journal/Book", "Publication Year", "Create Date", "PMCID.1",
        "NIHMS ID", "DOI", "Group Difference Explored?",
        "Do they present results using x,y,z coordinates?", "Other notes",
        "Unnamed: 27", "Unnamed: 28",
    ]
    for c in drop_cols:
        if c in golden_data.columns:
            golden_data = golden_data.drop(c, axis=1)

    assert len(golden_data) == len(predicted_data), (
        f"Row count mismatch: golden={len(golden_data)}, predicted={len(predicted_data)}"
    )

    df = pd.concat(
        [golden_data.reset_index(drop=True),
         predicted_data.reset_index(drop=True)],
        axis=1, join="inner"
    ).copy()

    # Rename ground-truth WMT column
    df = df.rename(columns={
        "What tracts were studied?": "Whitematter_tracts_gt",
        "whitematter_tracts":        "Whitematter_tracts_pred",
    })

    if "Whitematter_tracts_gt" not in df.columns or "Whitematter_tracts_pred" not in df.columns:
        print("  WARNING: WMT columns not found — returning F1=0")
        return 0.0, [], []

    # Build prediction/reference lists per row
    references, predictions = [], []
    for _, row in df.iterrows():
        g_list = canonicalize_list(clean_split(row.get("Whitematter_tracts_gt")),  canon_wmt)
        p_raw  = [normalize_text(x) for x in clean_split(row.get("Whitematter_tracts_pred"))]

        if USE_SEMANTIC_MATCHING and p_raw and g_list:
            p_mapped = []
            for p in p_raw:
                m, _ = best_semantic_match(p, g_list, canon_wmt, SIM_THRESH)
                p_mapped.append(m if m is not None else p)
        else:
            p_mapped = p_raw

        if DEDUP_PER_SAMPLE:
            p_mapped = sorted(set(p_mapped))

        references.append(g_list)
        predictions.append(p_mapped)

    micro_f1 = compute_wmt_f1(predictions, references)
    print(f"  WMT micro-F1 = {micro_f1:.3f}")
    return micro_f1, predictions, references

# RUN ALL FOUR CONDITIONS 
f1_gpt4_no_lut, preds_gpt4_no, refs_gpt4_no = evaluate_model(GPT4_no_LUT, "GPT-4  No LUT")
f1_gpt4_lut, preds_gpt4_lut, refs_gpt4_lut  = evaluate_model(GPT4_LUT,    "GPT-4  With LUT")
f1_gpt5_no_lut, preds_gpt5_no, refs_gpt5_no = evaluate_model(GPT5_no_LUT, "GPT-5  No LUT")
f1_gpt5_lut, preds_gpt5_lut, refs_gpt5_lut  = evaluate_model(GPT5_LUT,    "GPT-5  With LUT")

# BOOTSTRAP PAIRED SIGNIFICANCE TESTING 

print("\n" + "=" * 60)
print(f"RUNNING PAIRED BOOTSTRAP ({N_BOOT} iterations)...")
print("=" * 60)

num_samples = len(refs_gpt4_no)
np.random.seed(RANDOM_SEED)
shared_boot_idx = [np.random.choice(num_samples, num_samples, replace=True) for _ in range(N_BOOT)]

def get_f1_dist(preds_list, refs_list, boot_idxs):
    preds = np.empty(len(preds_list), dtype=object); preds[:] = preds_list
    refs = np.empty(len(refs_list), dtype=object); refs[:] = refs_list
    dist = []
    for idxs in boot_idxs:
        pi = preds[idxs].tolist()
        ri = refs[idxs].tolist()
        dist.append(compute_wmt_f1(pi, ri))
    return np.array(dist)

f1_dist_gpt4_no = get_f1_dist(preds_gpt4_no, refs_gpt4_no, shared_boot_idx)
f1_dist_gpt4_lut = get_f1_dist(preds_gpt4_lut, refs_gpt4_lut, shared_boot_idx)
f1_dist_gpt5_no = get_f1_dist(preds_gpt5_no, refs_gpt5_no, shared_boot_idx)
f1_dist_gpt5_lut = get_f1_dist(preds_gpt5_lut, refs_gpt5_lut, shared_boot_idx)

def compute_paired_stats(dist_base, dist_new):
    diff = dist_new - dist_base
    ci_lo = float(np.percentile(diff, 2.5))
    ci_hi = float(np.percentile(diff, 97.5))
    p_val = 2.0 * min(np.mean(diff <= 0), np.mean(diff >= 0))
    p_val = min(1.0, float(p_val))
    return ci_lo, ci_hi, p_val

ci_lo_diff_4, ci_hi_diff_4, p_raw_4 = compute_paired_stats(f1_dist_gpt4_no, f1_dist_gpt4_lut)
ci_lo_diff_5, ci_hi_diff_5, p_raw_5 = compute_paired_stats(f1_dist_gpt5_no, f1_dist_gpt5_lut)

def sig_symbol(p: float) -> str:
    if pd.isna(p): return ""
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return ""

def fmt_p(p: float) -> str:
    """Two-sided bootstrap resolves to 2/N_BOOT, so 0 reports as an upper bound."""
    if pd.isna(p): return ""
    return "<0.002" if p < 0.002 else f"{p:.3f}"

sig_4 = sig_symbol(p_raw_4)
sig_5 = sig_symbol(p_raw_5)

print(f"  GPT-4 (LUT vs No LUT): diff CI=[{ci_lo_diff_4:+.3f}, {ci_hi_diff_4:+.3f}], p={fmt_p(p_raw_4)} {sig_4}")
print(f"  GPT-5 (LUT vs No LUT): diff CI=[{ci_lo_diff_5:+.3f}, {ci_hi_diff_5:+.3f}], p={fmt_p(p_raw_5)} {sig_5}")

print("\n" + "=" * 60)
print("WMT F1 SUMMARY")
print("=" * 60)
print(f"  GPT-4  No LUT  : {f1_gpt4_no_lut:.3f}")
print(f"  GPT-4  With LUT: {f1_gpt4_lut:.3f}")
print(f"  GPT-5  No LUT  : {f1_gpt5_no_lut:.3f}")
print(f"  GPT-5  With LUT: {f1_gpt5_lut:.3f}")

# Save CSV
summary_df = pd.DataFrame([
    {"model": "GPT-4", "condition": "No LUT",   "wmt_f1": f1_gpt4_no_lut, "p_value": "", "sig_symbol": ""},
    {"model": "GPT-4", "condition": "With LUT",  "wmt_f1": f1_gpt4_lut, "p_value": fmt_p(p_raw_4), "sig_symbol": sig_4},
    {"model": "GPT-5", "condition": "No LUT",   "wmt_f1": f1_gpt5_no_lut, "p_value": "", "sig_symbol": ""},
    {"model": "GPT-5", "condition": "With LUT",  "wmt_f1": f1_gpt5_lut, "p_value": fmt_p(p_raw_5), "sig_symbol": sig_5},
])
summary_df.to_csv(OUT_TABLE, index=False)

# Bootstrap CIs, consumed by figure_lut_complexity.py.
def get_ci(dist):
    return float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))

ci_lo_4_no, ci_hi_4_no = get_ci(f1_dist_gpt4_no)
ci_lo_4_lut, ci_hi_4_lut = get_ci(f1_dist_gpt4_lut)
ci_lo_5_no, ci_hi_5_no = get_ci(f1_dist_gpt5_no)
ci_lo_5_lut, ci_hi_5_lut = get_ci(f1_dist_gpt5_lut)
