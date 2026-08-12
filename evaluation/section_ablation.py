"""F1 per extraction variable across the 14 section combinations, GPT-5-mini.

Each combination is compared against full text with a paired bootstrap and
FDR correction. Metrics only; the figure is figure_section_ablation.py.

Output: results/tables/section_ablation.csv, results/tables/bootstrap_dist_*.csv
"""

import os
import re
import warnings
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from statsmodels.stats.multitest import fdrcorrection

warnings.filterwarnings("ignore")

SIM_THRESH = 0.95
USE_SEMANTIC_MATCHING = True
DEDUP_PER_SAMPLE = False
SKIP_EMPTY_GOLD = True

N_BOOT = 1000
RANDOM_SEED = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "whitematter_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "predictions", "no_lut")
TABLES_DIR = os.path.join(BASE_DIR, "results", "tables")

# ── SECTION DEFINITIONS ───────────────────────────────────────────────────────
SECTION_CSVS = {
    # Single sections
    "Abstract":                   "GPT_section_ABSTRACT_GPT_5.csv",
    "Methods":                    "GPT_section_METHODS_GPT_5.csv",
    "Results":                    "GPT_section_RESULTS_GPT_5.csv",
    "Discussion":                 "GPT_section_DISCUSSION_GPT_5.csv",
    # Two-section combos
    "Abstract+Methods":           "GPT_section_ABSTRACT_METHODS_GPT_5.csv",
    "Abstract+Results":           "GPT_section_ABSTRACT_RESULTS_GPT_5.csv",
    "Abstract+Discussion":        "GPT_section_ABSTRACT_DISCUSSION_GPT_5.csv",
    "Methods+Results":            "GPT_section_METHODS_RESULTS_GPT_5.csv",
    "Methods+Discussion":         "GPT_section_METHODS_DISCUSSION_GPT_5.csv",
    "Results+Discussion":         "GPT_section_RESULTS_DISCUSSION_GPT_5.csv",
    # Three-section combos
    "Abstract+Methods+Results":   "GPT_section_ABSTRACT_METHODS_RESULTS_GPT_5.csv",
    "Abstract+Methods+Discussion":"GPT_section_ABSTRACT_METHODS_DISCUSSION_GPT_5.csv",
    "Abstract+Results+Discussion":"GPT_section_ABSTRACT_RESULTS_DISCUSSION_GPT_5.csv",
    "Methods+Results+Discussion": "GPT_section_METHODS_RESULTS_DISCUSSION_GPT_5.csv",
    # Full text benchmark
    "Full Text":                  "whitematter_fulltext_GPT_5.csv",
}

BENCHMARK_SECTION = "Full Text"

# ── TEXT NORMALIZATION & HELPERS ──────────────────────────────────────────────
EMPTY_TOKENS = {
    "", "none", "n.a.", "na", "n a", "n/a", "null", "_", "-", "nan",
    "not reported", "unknown",
}

def normalize_text(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return " ".join(str(x).lower().strip().split())

def normalize_cmap_keys_values(cmap: Dict[str, str]) -> Dict[str, str]:
    return {normalize_text(k): normalize_text(v) for k, v in cmap.items()}

def is_empty_token(s: str) -> bool:
    return normalize_text(s) in EMPTY_TOKENS

def clean_split(x) -> List[str]:
    s = normalize_text(x)
    if is_empty_token(s):
        return []
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    return [normalize_text(p) for p in parts if p and not is_empty_token(p)]

def canonicalize(value: str, cmap: Dict[str, str]) -> str:
    return cmap.get(normalize_text(value), normalize_text(value))

def canonicalize_list(values: List[str], cmap: Dict[str, str]) -> List[str]:
    return [canonicalize(v, cmap) for v in values if not is_empty_token(v)]

def seq_sim(a: str, b: str) -> float:
    a, b = normalize_text(a), normalize_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def best_semantic_match(pred_raw: str, gold_list: List[str], cmap: Dict[str, str], threshold: float) -> Tuple[str, float]:
    p_norm = normalize_text(pred_raw)
    p_can = canonicalize(p_norm, cmap)
    if p_can in gold_list:
        return p_can, 1.0
    best_match, best_score = None, 0.0
    for g in gold_list:
        score = seq_sim(p_norm, g)
        if score > best_score:
            best_score = score
            best_match = g
    if best_score >= threshold:
        return best_match, best_score
    return None, 0.0

# Canonical maps...
canon_dti = normalize_cmap_keys_values({"yes": "yes", "no": "no"})
canon_human = normalize_cmap_keys_values({"yes": "yes", "no": "no"})
canon_dementia = normalize_cmap_keys_values({"yes": "yes", "no": "no"})
canon_type = normalize_cmap_keys_values({})
canon_disease = normalize_cmap_keys_values({
    "alzheimers disease": "alzheimers disease", "ad": "alzheimers disease",
    "parkinson disease": "parkinson disease",
})
canon_white_matter_tracts = normalize_cmap_keys_values({
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

# ── FIELD DEFINITIONS ─────────────────────────────────────────────────────────
binary_fields = [
    ("DTI Methodology [Y/N]", "DTI_gt", "DTI_pred", canon_dti, "yes"),
    ("Subject Species [Human/Other]", "Human_study_gt", "Human_study_pred", canon_human, "yes"),
    ("Dementia-Related[Y/N]", "Dementia_study_gt", "Dementia_study_pred", canon_dementia, "yes"),
]
multilabel_fields = [
    ("Study Design [Review/Original]", "Study_type_gt", "Study_type_pred", canon_type),
    ("Neurological Disorders", "Disease_study_gt", "Disease_study_pred", canon_disease),
    ("White Matter Tracts", "Whitematter_tracts_gt", "Whitematter_tracts_pred", canon_white_matter_tracts),
]

all_fields_for_norm = [(n, g, p, c) for (n, g, p, c, *_) in binary_fields] + [(n, g, p, c) for (n, g, p, c) in multilabel_fields]
ALL_FIELD_NAMES = [n for n, *_ in binary_fields] + [n for n, *_ in multilabel_fields]

def compute_binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, zero_division=0)

def compute_multilabel_f1(preds: List[List[str]], refs: List[List[str]]) -> float:
    if SKIP_EMPTY_GOLD:
        preds_f, refs_f = [], []
        for p, r in zip(preds, refs):
            if len(r) > 0:
                preds_f.append(p)
                refs_f.append(r)
        preds, refs = preds_f, refs_f
    all_labels = sorted(set(x for sub in preds + refs for x in sub))
    if not all_labels: return 0.0
    mlb = MultiLabelBinarizer(classes=all_labels)
    Y_true = mlb.fit_transform(refs)
    Y_pred = mlb.transform(preds)
    if Y_true.size == 0: return 0.0
    return f1_score(Y_true, Y_pred, average="micro", zero_division=0)

# ── GROUND TRUTH & PRED COLUMN RENAME MAPS ────────────────────────────────────
GT_RENAME = {
    "Is this DTI?": "DTI_gt",
    "Is this a single study or a review?": "Study_type_gt",
    "Human study or not?": "Human_study_gt",
    "Does this study dementia, alzheimers, or related disease? \n": "Dementia_study_gt",
    "Which one?": "Disease_study_gt",
    "What tracts were studied?": "Whitematter_tracts_gt",
}
PRED_RENAME = {
    "DTI_study": "DTI_pred", "study_type": "Study_type_pred", "Human_study": "Human_study_pred",
    "Dementia_study": "Dementia_study_pred", "Disease_study": "Disease_study_pred", "whitematter_tracts": "Whitematter_tracts_pred",
}


# ── RAW DATA EXTRACTION FOR BOOTSTRAPPING ───────────────────────────
def get_section_raw_data(section_label: str, pred_csv: str) -> Dict[str, Dict[str, any]]:
    pred_path = os.path.join(OUTPUT_DIR, pred_csv)
    golden_data = pd.read_csv(DATA_RAW)
    predicted_data = pd.read_csv(pred_path)

    df = pd.concat([golden_data.reset_index(drop=True), predicted_data.reset_index(drop=True)], axis=1, join='inner').copy()
    df = df.rename(columns={k: v for k, v in GT_RENAME.items() if k in df.columns})
    df = df.rename(columns={k: v for k, v in PRED_RENAME.items() if k in df.columns})

    for name, gt_col, pred_col, cmap in all_fields_for_norm:
        for col in [gt_col, pred_col]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x, _cmap=cmap: ", ".join(canonicalize_list(clean_split(x), _cmap)) if pd.notna(x) else "")

    raw_data = {}
    
    # Binary
    for name, gtf, prf, cmap, pos_label in binary_fields:
        if gtf not in df.columns or prf not in df.columns: continue
        y_true = np.array([1 if canonicalize(normalize_text(r.get(gtf)), cmap) == pos_label else 0 for _, r in df.iterrows()])
        y_pred = np.array([1 if canonicalize(normalize_text(r.get(prf)), cmap) == pos_label else 0 for _, r in df.iterrows()])
        raw_data[name] = {"type": "binary", "y_true": y_true, "y_pred": y_pred}

    # Multilabel
    for name, gtf, prf, cmap in multilabel_fields:
        if gtf not in df.columns or prf not in df.columns: continue
        all_preds, all_refs = [], []
        for _, row in df.iterrows():
            g_list = canonicalize_list(clean_split(row.get(gtf)), cmap)
            p_list_raw = [normalize_text(x) for x in clean_split(row.get(prf))]
            if USE_SEMANTIC_MATCHING and p_list_raw and g_list:
                p_mapped = []
                for p_raw in p_list_raw:
                    match_label, _ = best_semantic_match(p_raw, g_list, cmap, SIM_THRESH)
                    p_mapped.append(match_label if match_label is not None else normalize_text(p_raw))
            else:
                p_mapped = [normalize_text(p) for p in p_list_raw]
            if DEDUP_PER_SAMPLE: p_mapped = sorted(set(p_mapped))
            all_refs.append(g_list)
            all_preds.append(p_mapped)
        # Convert to numpy arrays of objects for easy indexing during bootstrap
        raw_data[name] = {"type": "multilabel", "y_true": np.array(all_refs, dtype=object), "y_pred": np.array(all_preds, dtype=object)}

    return raw_data


# ── EVALUATION & BOOTSTRAPPING LOGIC ──────────────────────────────────────────
def compute_f1_from_raw(field_data: Dict[str, any], indices: np.ndarray) -> float:
    if field_data["type"] == "binary":
        return compute_binary_f1(field_data["y_true"][indices], field_data["y_pred"][indices])
    else:
        # Multilabel
        preds = field_data["y_pred"][indices].tolist()
        refs = field_data["y_true"][indices].tolist()
        return compute_multilabel_f1(preds, refs)


def sig_symbol(p: float) -> str:
    """Map p-value to significance symbol."""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"   # Very Highly Significant (p < 0.1%)
    elif p < 0.01:
        return "**"    # Highly Significant     (p < 1%)
    elif p < 0.05:
        return "*"     # Statistically Significant (p < 5%)
    else:
        return ""      # Not significant



def bootstrap_and_test() -> pd.DataFrame:
    print("=" * 70)
    print(f"EXTRACTING RAW DATA FOR ALL 15 SECTIONS...")
    print("=" * 70)
    
    all_raw_data = {}
    for sec, csv_name in SECTION_CSVS.items():
        all_raw_data[sec] = get_section_raw_data(sec, csv_name)
        
    num_samples = len(all_raw_data[BENCHMARK_SECTION][ALL_FIELD_NAMES[0]]["y_true"])
    
    # Shared bootstrap indices — SAME resampled rows for every section in each iteration.
    # This is the paired bootstrap design: section and Full Text are evaluated on the
    # identical subset, so per-paper performance differences cancel out noise.
    np.random.seed(RANDOM_SEED)
    shared_boot_idx = [
        np.random.choice(num_samples, num_samples, replace=True) for _ in range(N_BOOT)
    ]
    
    print("\n" + "=" * 70)
    print(f"RUNNING BOOTSTRAP ({N_BOOT} iterations)...")
    print("=" * 70)
    
    results = []
    all_idx = np.arange(num_samples)
    
    # Dictionary to hold the bootstrap distributions for significance testing later
    boot_dists = {field: {} for field in ALL_FIELD_NAMES}
    
    for sec in SECTION_CSVS.keys():
        print(f"  Bootstrapping: {sec}")
        for field in ALL_FIELD_NAMES:
            raw = all_raw_data[sec][field]
            
            # Base F1
            base_f1 = compute_f1_from_raw(raw, all_idx)
            
            # Bootstrap F1 distribution (uses shared indices for proper paired comparison)
            f1_dist = np.array([compute_f1_from_raw(raw, idxs) for idxs in shared_boot_idx])
            boot_dists[field][sec] = f1_dist
            
            # Confidence Intervals
            ci_lower = np.percentile(f1_dist, 2.5)
            ci_upper = np.percentile(f1_dist, 97.5)
            
            error_lower = max(0.0, base_f1 - ci_lower)
            error_upper = max(0.0, ci_upper - base_f1)
            
            results.append({
                "section": sec,
                "field": field,
                "f1": base_f1,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "error_lower": error_lower,
                "error_upper": error_upper,
                "p_value": "",
                "p_value_raw": "",
                "sig_level": "",
                "sig_symbol": "ns",
            })
            
    df_res = pd.DataFrame(results)
    
    # ── PAIRED BOOTSTRAP P-VALUE VS FULL TEXT ──────────────────────────────
    # Method: in each bootstrap iteration the SAME paper-rows are used for both
    # the section and Full Text, so we compute per-iteration differences.
    # p-value tests whether the difference is significantly different from 0

    print("\n" + "=" * 70)
    print(f"PAIRED BOOTSTRAP P-VALUE VS '{BENCHMARK_SECTION}'")
    print("=" * 70)
    
    for field in ALL_FIELD_NAMES:
        bm_dist = boot_dists[field][BENCHMARK_SECTION]   # shape (N_BOOT,)
        
        # Collect raw p-values for this field across all sections (for FDR correction)
        raw_p_values = []
        sections_to_test = []
        c_i_bounds = []
        
        for sec in SECTION_CSVS.keys():
            if sec == BENCHMARK_SECTION:
                continue
            
            sec_dist = boot_dists[field][sec]             # shape (N_BOOT,)
            
            # Per-iteration paired difference
            diff = bm_dist - sec_dist
            
            # Two-sided p-value: fraction of iterations where diff crosses 0
            p_val = 2.0 * min(np.mean(diff <= 0), np.mean(diff >= 0))
            p_val = min(1.0, float(p_val))
            
            # 95% CI of the difference
            diff_ci_lo = float(np.percentile(diff, 2.5))
            diff_ci_hi = float(np.percentile(diff, 97.5))
            
            raw_p_values.append(p_val)
            sections_to_test.append(sec)
            c_i_bounds.append((diff_ci_lo, diff_ci_hi))
            
        # Apply FDR Correction across the 14 sections for this field
        # Here we apply it per-field across the 14 comparisons to the Full Text benchmark.
        rejected, adj_p_values = fdrcorrection(raw_p_values, method='indep')
        
        for sec, raw_p, adj_p, (ci_lo, ci_hi) in zip(sections_to_test, raw_p_values, adj_p_values, c_i_bounds):
            symb  = sig_symbol(adj_p)
            
            if adj_p < 0.001:
                level = "FDR q < 0.001 (Very Highly Significant)"
            elif adj_p < 0.01:
                level = "FDR q < 0.01  (Highly Significant)"
            elif adj_p < 0.05:
                level = "FDR q < 0.05  (Statistically Significant)"
            else:
                level = "FDR q ≥ 0.05  (Not Significant)"
            
            # Write back to DataFrame
            mask = (df_res['field'] == field) & (df_res['section'] == sec)
            df_res.loc[mask, 'p_value_raw']   = "<0.002" if raw_p < 0.002 else f"{raw_p:.3f}"
            df_res.loc[mask, 'p_value']   = "<0.002" if adj_p < 0.002 else f"{adj_p:.3f}"
            df_res.loc[mask, 'sig_level'] = level
            df_res.loc[mask, 'sig_symbol'] = symb
            
            print(f"  {field:15s} | {sec:30s} | diff_CI=[{ci_lo:+.3f},{ci_hi:+.3f}] | p_raw={raw_p:.4f} | q_fdr={adj_p:.4f} | {symb}")
            
    return df_res, boot_dists


# ── RUN PIPELINE & PLOTTING ────────────
# %%
results_df, boot_dists = bootstrap_and_test()

os.makedirs(TABLES_DIR, exist_ok=True)

# Save detailed CSV
results_csv_path = os.path.join(TABLES_DIR, "section_ablation.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\nSaved detailed summary to: {results_csv_path}")

# ── SAVE RAW BOOTSTRAP DISTRIBUTIONS TO CSV ───────────────────────────────────
print("\n" + "=" * 70)
print("SAVING BOOTSTRAP DISTRIBUTIONS (FOR PLOTTING)")
print("=" * 70)
for field in ALL_FIELD_NAMES:
    # Each field gets its own DataFrame where columns are sections and rows are bootstrap iterations
    field_dist_df = pd.DataFrame(boot_dists[field])
    
    # Map back to original short names to keep output filenames the same
    safe_name_map = {
        "DTI Methodology [Y/N]": "dti",
        "Subject Species [Human/Other]": "human",
        "Dementia-Related[Y/N]": "dementia",
        "Study Design [Review/Original]": "study_type",
        "Neurological Disorders": "diseases",
        "White Matter Tracts": "wm_tracts"
    }
    safe_name = safe_name_map.get(field, re.sub(r'[^a-z0-9]+', '_', field.lower()).strip('_'))
    
    dist_csv_path = os.path.join(TABLES_DIR, f"bootstrap_dist_{safe_name}.csv")
    field_dist_df.to_csv(dist_csv_path, index=False)
    print(f"  Saved: bootstrap_dist_{safe_name}.csv")
