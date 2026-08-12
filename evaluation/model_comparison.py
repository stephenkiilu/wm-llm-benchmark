"""F1 and accuracy per extraction variable for GPT-4o-mini vs GPT-5-mini.

Both models are evaluated on no-LUT predictions so neither gets the LUT
advantage. Reports bootstrap CIs and FDR-corrected paired significance tests.

Output: results/tables/model_comparison.csv
"""

import os
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from statsmodels.stats.multitest import fdrcorrection

SIM_THRESH = 0.95
USE_SEMANTIC_MATCHING = True   # set False to disable fuzzy mapping
DEDUP_PER_SAMPLE = False       # deduplicate predictions within each sample

# Drop rows with an empty gold reference: keeping them inflates subset accuracy,
# because an empty gold and an empty prediction count as an exact match.
SKIP_EMPTY_GOLD = True

N_BOOT = 1000
RANDOM_SEED = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "whitematter_dataset.csv")
DATA_GPT_4o_mini = os.path.join(BASE_DIR, "predictions", "no_lut", "whitematter_fulltext_GPT_4.csv")
DATA_GPT_5_mini = os.path.join(BASE_DIR, "predictions", "no_lut", "whitematter_fulltext_GPT_5.csv")
OUT_TABLE = os.path.join(BASE_DIR, "results", "tables", "model_comparison.csv")
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
            best_score = score
            best_ref = r_can
    if best_score >= thresh:
        return best_ref, best_score
    return None, best_score

# ── CANONICAL MAPS ────────────────────────────────────────────────────────────
canon_dti      = normalize_cmap_keys_values({"yes": "yes", "no": "no"})
canon_human    = normalize_cmap_keys_values({"yes": "yes", "no": "no", "human": "yes"})
canon_dementia = normalize_cmap_keys_values({"yes": "yes", "no": "no"})
canon_type     = normalize_cmap_keys_values({
    "single study": "single study", "single": "single study",
    "meta analysis": "meta analysis", "review": "review",
})
canon_disease  = normalize_cmap_keys_values({
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
    ("Does it use DTI?",                     "DTI_gt",           "DTI_pred",           canon_dti,      "yes"),
    ("Human_vs_non_human_study",             "Human_study_gt",   "Human_study_pred",   canon_human,    "yes"),
    ("Does it study dementia or related diseases?", "Dementia_study_gt", "Dementia_study_pred", canon_dementia, "yes"),
]

multilabel_fields = [
    ("Review or single study?",  "Study_type_gt",        "Study_type_pred",        canon_type),
    ("Which diseases are studied", "Disease_study_gt",   "Disease_study_pred",     canon_disease),
    ("WM tracts studied",        "Whitematter_tracts_gt", "Whitematter_tracts_pred", canon_white_matter_tracts),
]

all_fields_for_norm = [
    (n, g, p, c) for (n, g, p, c, *_) in binary_fields
] + multilabel_fields

# ── METRIC FUNCTIONS ──────────────────────────────────────────────────────────
def _jaccard_binary(y_true: List[int], y_pred: List[int]) -> float:
    vals = [1.0 if yt == yp else 0.0 for yt, yp in zip(y_true, y_pred)]
    return float(np.mean(vals)) if vals else 0.0

def compute_binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    return dict(
        accuracy  = accuracy_score(y_true, y_pred),
        precision = precision_score(y_true, y_pred, zero_division=0),
        recall    = recall_score(y_true, y_pred, zero_division=0),
        f1        = f1_score(y_true, y_pred, zero_division=0),
        jaccard   = _jaccard_binary(y_true, y_pred),
        support   = int(sum(y_true)),
        n_samples = len(y_true),
    )

def multilabel_binarize(preds: List[List[str]], refs: List[List[str]]):
    all_labels = sorted(set(x for sub in preds + refs for x in sub))
    if not all_labels:
        return np.zeros((0, 0), int), np.zeros((0, 0), int), []
    mlb = MultiLabelBinarizer(classes=all_labels)
    Y_true = mlb.fit_transform(refs)
    Y_pred = mlb.transform(preds)
    return Y_pred, Y_true, all_labels

def macro_labelwise(y_true: np.ndarray, y_pred: np.ndarray, scorer) -> float:
    vals = []
    for j in range(y_true.shape[1]):
        yt, yp = y_true[:, j], y_pred[:, j]
        if yt.sum() + yp.sum() == 0:
            continue
        vals.append(scorer(yt, yp, zero_division=0))
    return float(np.mean(vals)) if vals else 0.0

def _jaccard_samples(preds, refs) -> float:
    vals = []
    for p, r in zip(preds, refs):
        ps, rs = set(p), set(r)
        union = ps | rs
        vals.append(len(ps & rs) / len(union) if union else 1.0)
    return float(np.mean(vals)) if vals else 0.0

def filter_empty_gold(preds, refs):
    preds_f, refs_f = [], []
    for p, r in zip(preds, refs):
        if len(r) > 0:
            preds_f.append(p)
            refs_f.append(r)
    return preds_f, refs_f

def compute_multilabel_metrics(preds, refs) -> Dict:
    if SKIP_EMPTY_GOLD:
        preds, refs = filter_empty_gold(preds, refs)
    Y_pred, Y_true, labels = multilabel_binarize(preds, refs)
    if Y_true.size == 0:
        return dict(
            macro_p=0, macro_r=0, macro_f1=0,
            micro_p=0, micro_r=0, micro_f1=0,
            sample_p=0, sample_r=0, sample_f1=0,
            subset_acc=0, jaccard=0,
            labels=labels, n_samples=0,
        )
    m_p  = macro_labelwise(Y_true, Y_pred, precision_score)
    m_r  = macro_labelwise(Y_true, Y_pred, recall_score)
    m_f1 = macro_labelwise(Y_true, Y_pred, f1_score)
    mi_p  = precision_score(Y_true, Y_pred, average="micro", zero_division=0)
    mi_r  = recall_score(Y_true, Y_pred, average="micro", zero_division=0)
    mi_f1 = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
    sa_p  = precision_score(Y_true, Y_pred, average="samples", zero_division=0)
    sa_r  = recall_score(Y_true, Y_pred, average="samples", zero_division=0)
    sa_f1 = f1_score(Y_true, Y_pred, average="samples", zero_division=0)
    subset = float((Y_pred == Y_true).all(axis=1).mean())
    jacc   = _jaccard_samples(preds, refs)
    return dict(
        macro_p=m_p, macro_r=m_r, macro_f1=m_f1,
        micro_p=mi_p, micro_r=mi_r, micro_f1=mi_f1,
        sample_p=sa_p, sample_r=sa_r, sample_f1=sa_f1,
        subset_acc=subset, jaccard=jacc,
        labels=labels, n_samples=len(preds),
    )

# ── METRIC HELPERS FOR BOOTSTRAPPING ──────────────────────────────────────────
def compute_f1_binary(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, zero_division=0)

def compute_f1_multilabel(preds, refs) -> float:
    if SKIP_EMPTY_GOLD:
        preds, refs = filter_empty_gold(preds, refs)
    Y_pred, Y_true, labels = multilabel_binarize(preds, refs)
    if Y_true.size == 0:
        return 0.0
    return f1_score(Y_true, Y_pred, average="micro", zero_division=0)

def compute_f1_from_raw(field_data: Dict[str, any], indices: np.ndarray) -> float:
    if field_data["type"] == "binary":
        return compute_f1_binary(field_data["y_true"][indices], field_data["y_pred"][indices])
    else:
        # Multilabel
        preds = field_data["y_pred"][indices].tolist()
        refs = field_data["y_true"][indices].tolist()
        return compute_f1_multilabel(preds, refs)

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

# ── EVALUATE ONE MODEL ────────────────────────────────────────────────────────
def evaluate_model(pred_path: str, model_label: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Run the full evaluation pipeline for a single model and return:
    1. A DataFrame with baseline F1.
    2. A dictionary of raw arrays {field: {'type': 'binary' or 'multilabel', 'y_true': [...], 'y_pred': [...]}}
    """
    print(f"\n{'=' * 70}")
    print(f"Evaluating: {model_label}  ({pred_path})")
    print("=" * 70)

    golden_data    = pd.read_csv(DATA_RAW)
    predicted_data = pd.read_csv(pred_path)

    drop_cols = [
        "PMCID", 'Open Source?', 'Authors', 'Citation', 'First Author',
        "Which imaging modality was used? e.g electroencephalogram (EEG), Positron emission tomography (PET), Anatomical MRI, fMRI, diffusion MRI (dMRI) etc",
        'Journal/Book', 'Publication Year', 'Create Date', 'PMCID.1',
        'NIHMS ID', 'DOI', 'Group Difference Explored?',
        'Do they present results using x,y,z coordinates?', 'Other notes',
        'Unnamed: 27', 'Unnamed: 28'
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
        axis=1, join='inner'
    ).copy()

    rename_map = {
        "Is this DTI?": "DTI_gt",
        "Is this a single study or a review?": "Study_type_gt",
        "Human study or not?": "Human_study_gt",
        "Does this study dementia, alzheimers, or related disease? \n": "Dementia_study_gt",
        "Which one?": "Disease_study_gt",
        "How are results presented? T-tests, beta effect sizes, ANOVA, Chi test, regression etc": "Results_method_gt",
        "What analysis software libraries are being used e.g dipy.org, fsl, freesurfer, PRIDE ..?": "Analysis_software_gt",
        "What was the diffusion measure e.g Fractional Anisotropy (FA), Mean Diffusivity (MD), Axial diffusivity (AD), Radial diffusivity (RD)?": "Diffusion_measure_gt",
        "Were increases or decreases in WM integrity reported relative to the disease population? Eg: AD> HC= RD increase (Corpus Callosum, Cingulum) , AD<HC = FA decreases (cingulum). If Column T is NA enter NA here. (What it means, comparing Alzheimer's disease with Health Control, there was increase in Radial diffusivity in the Corpus Callosum and Cingulum white matter tracts, etc).": "WM_integrity_gt",
        "Brain Template/ Space e.g MNI, JHU'": "Brain_template_gt",
        "What tracts were studied?": "Whitematter_tracts_gt",
        "DTI_study": "DTI_pred",
        "study_type": "Study_type_pred",
        "Human_study": "Human_study_pred",
        "Dementia_study": "Dementia_study_pred",
        "Disease_study": "Disease_study_pred",
        "whitematter_tracts": "Whitematter_tracts_pred",
        "imaging_modalities": "Imaging_modalities_pred",
        "results_method": "Results_method_pred",
        "analysis_software": "Analysis_software_pred",
        "diffusion_measures": "Diffusion_measure_pred",
        "template_space": "Brain_template_pred",
        "question_of_study": "Question_of_study",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Normalize
    df_norm = df.copy()
    for name, gt_col, pred_col, cmap in all_fields_for_norm:
        for col in [gt_col, pred_col]:
            if col in df_norm.columns:
                df_norm[col] = df_norm[col].apply(
                    lambda x, _cmap=cmap: ", ".join(canonicalize_list(clean_split(x), _cmap)) if pd.notna(x) else ""
                )
    df = df_norm

    # ── Binary evaluation ─────────────────────────────────────────────────────
    binary_predictions = {name: {"y_true": [], "y_pred": []}
                          for name, *_ in binary_fields}
    for _, row in df.iterrows():
        for name, gtf, prf, cmap, pos_label in binary_fields:
            if gtf not in df.columns or prf not in df.columns:
                continue
            g_val = canonicalize(normalize_text(row.get(gtf)), cmap)
            p_val = canonicalize(normalize_text(row.get(prf)), cmap)
            binary_predictions[name]["y_true"].append(1 if g_val == pos_label else 0)
            binary_predictions[name]["y_pred"].append(1 if p_val == pos_label else 0)

    binary_results = {}
    for name, *_ in binary_fields:
        d = binary_predictions[name]
        binary_results[name] = compute_binary_metrics(d["y_true"], d["y_pred"])

    # ── Multilabel evaluation ─────────────────────────────────────────────────
    multilabel_predictions = {name: {"references": [], "predictions": []}
                              for name, *_ in multilabel_fields}
    for _, row in df.iterrows():
        for name, gtf, prf, cmap in multilabel_fields:
            if gtf not in df.columns or prf not in df.columns:
                continue
            g_list = canonicalize_list(clean_split(row.get(gtf)), cmap)
            p_list_raw = [normalize_text(x) for x in clean_split(row.get(prf))]
            if USE_SEMANTIC_MATCHING and p_list_raw and g_list:
                p_mapped = []
                for p_raw in p_list_raw:
                    match_label, _ = best_semantic_match(p_raw, g_list, cmap, SIM_THRESH)
                    p_mapped.append(match_label if match_label is not None else normalize_text(p_raw))
            else:
                p_mapped = [normalize_text(p) for p in p_list_raw]
            if DEDUP_PER_SAMPLE:
                p_mapped = sorted(set(p_mapped))
            multilabel_predictions[name]["references"].append(g_list[:])
            multilabel_predictions[name]["predictions"].append(p_mapped[:])

    multilabel_results = {}
    for name, *_ in multilabel_fields:
        d = multilabel_predictions[name]
        multilabel_results[name] = compute_multilabel_metrics(d["predictions"], d["references"])

    # ── Build combined F1 rows ────────────────────────────────────────────────
    rows = []
    for name, r in binary_results.items():
        rows.append({"field": name, "f1": round(r["f1"], 3), "model": model_label})
        print(f"  [binary]     {name:45s}  F1={r['f1']:.3f}")
    for name, r in multilabel_results.items():
        rows.append({"field": name, "f1": round(r["micro_f1"], 3), "model": model_label})
        print(f"  [multilabel] {name:45s}  F1={r['micro_f1']:.3f}")

    raw_data_dict = {}
    for name, *_ in binary_fields:
        raw_data_dict[name] = {"type": "binary", "y_true": np.array(binary_predictions[name]["y_true"]), "y_pred": np.array(binary_predictions[name]["y_pred"])}
    for name, *_ in multilabel_fields:
        # Save as object array to allow integer indexing over lists
        raw_data_dict[name] = {
            "type": "multilabel", 
            "y_true": np.empty(len(multilabel_predictions[name]["references"]), dtype=object),
            "y_pred": np.empty(len(multilabel_predictions[name]["predictions"]), dtype=object)
        }
        raw_data_dict[name]["y_true"][:] = multilabel_predictions[name]["references"]
        raw_data_dict[name]["y_pred"][:] = multilabel_predictions[name]["predictions"]

    return pd.DataFrame(rows), raw_data_dict

# ── EVALUATE BOTH MODELS ──────────────────────────────────────────────────────
results_gpt4, raw_gpt4 = evaluate_model(DATA_GPT_4o_mini, "GPT-4")
results_gpt5, raw_gpt5 = evaluate_model(DATA_GPT_5_mini,  "GPT-5")

comparison_df = pd.concat([results_gpt4, results_gpt5], ignore_index=True)


# ── PAIRED BOOTSTRAP SIGNIFICANCE TESTING ─────────────────────────────────────
def paired_bootstrap_testing(raw_gpt4: Dict, raw_gpt5: Dict, comparison_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    # Extract field names
    fields = comparison_df["field"].unique().tolist()
    
    # Num samples (assumes all fields evaluated over the same document count)
    num_samples = len(raw_gpt4[fields[0]]["y_true"])
    
    np.random.seed(RANDOM_SEED)
    shared_boot_idx = [
        np.random.choice(num_samples, num_samples, replace=True) for _ in range(N_BOOT)
    ]
    
    print("\n" + "=" * 70)
    print(f"RUNNING PAIRED BOOTSTRAP ({N_BOOT} iterations)...")
    print("=" * 70)
    
    boot_stats = []
    raw_p_values = []
    
    # For error bars in plotting
    dist_info = {m: {} for m in ["GPT-4", "GPT-5"]}
    
    for field in fields:
        raw4 = raw_gpt4[field]
        raw5 = raw_gpt5[field]
        
        f1_dist_4 = np.array([compute_f1_from_raw(raw4, idxs) for idxs in shared_boot_idx])
        f1_dist_5 = np.array([compute_f1_from_raw(raw5, idxs) for idxs in shared_boot_idx])
        
        dist_info["GPT-4"][field] = f1_dist_4
        dist_info["GPT-5"][field] = f1_dist_5
        
        # Per-iteration paired difference (baseline: GPT-4, new: GPT-5)
        diff = f1_dist_5 - f1_dist_4
        
        diff_ci_lo = float(np.percentile(diff, 2.5))
        diff_ci_hi = float(np.percentile(diff, 97.5))
        
        p_val = 2.0 * min(np.mean(diff <= 0), np.mean(diff >= 0))
        p_val = min(1.0, float(p_val))
        
        raw_p_values.append(p_val)
        
        boot_stats.append({
            "field": field,
            "diff_mean": float(np.mean(diff)),
            "diff_ci_lower": diff_ci_lo,
            "diff_ci_upper": diff_ci_hi,
            "p_value_raw": "<0.002" if p_val < 0.002 else f"{p_val:.3f}"
        })
        
    rejected, adj_p_values = fdrcorrection(raw_p_values, method='indep')
    
    for idx, stat in enumerate(boot_stats):
        stat["p_value_fdr"] = "<0.002" if adj_p_values[idx] < 0.002 else f"{adj_p_values[idx]:.3f}"
        stat["sig_symbol"] = sig_symbol(adj_p_values[idx])
        
        print(f"  {stat['field']:40s} | diff_CI=[{stat['diff_ci_lower']:+.3f},{stat['diff_ci_upper']:+.3f}] | p_raw={stat['p_value_raw']} | q_fdr={stat['p_value_fdr']} | {stat['sig_symbol']}")
        
    stats_df = pd.DataFrame(boot_stats)
    
    # Merge exact CI errors into the comparison_df for plotting
    ci_rows = []
    all_idx = np.arange(num_samples)
    for model, dists in [("GPT-4", dist_info["GPT-4"]), ("GPT-5", dist_info["GPT-5"])]:
        for field, dist in dists.items():
            ci_lo = np.percentile(dist, 2.5)
            ci_hi = np.percentile(dist, 97.5)
            ci_rows.append({"model": model, "field": field, "ci_lower": ci_lo, "ci_upper": ci_hi})
            
    ci_df = pd.DataFrame(ci_rows)
    final_df = comparison_df.merge(ci_df, on=["model", "field"])
    
    return final_df, stats_df, dist_info

final_df, stats_df, dist_info = paired_bootstrap_testing(raw_gpt4, raw_gpt5, comparison_df)

# Merging FDR stats so they are stored in the CSV
export_df = final_df.merge(stats_df, on="field", how="left")
export_df.to_csv(OUT_TABLE, index=False)


print("\n" + "=" * 70)
print("F1 COMPARISON — GPT-4 vs GPT-5")
print("=" * 70)
print(comparison_df.pivot(index="field", columns="model", values="f1").to_string())
