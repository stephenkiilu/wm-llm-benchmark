"""Gold-label complexity score for each of the six extraction variables.

Complexity is derived only from the ground-truth labels, so the
F1-vs-complexity relationship is not circular. The composite index is the mean
of four min-max-normalised gold-derived features: label-space size, label
cardinality, Shannon entropy of the label distribution, and answer length.

Output: results/tables/task_complexity.csv
"""

import os
from collections import Counter
from math import log2

import numpy as np
import pandas as pd

from evaluation import model_comparison as E
from evaluation import plot_style as S

S.apply_style()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_TABLE = os.path.join(BASE_DIR, "results", "tables", "task_complexity.csv")

raw = E.raw_gpt4  # gold (y_true) is identical across models
fields = list(raw.keys())


def gold_sets(fd):
    """Return the gold label set per paper (binary -> {'yes'|'no'})."""
    if fd["type"] == "binary":
        return [["yes"] if int(v) == 1 else ["no"] for v in fd["y_true"]]
    sets = [list(g) for g in fd["y_true"]]
    if E.SKIP_EMPTY_GOLD:                      # match the metric's filtering
        sets = [s for s in sets if s]
    return sets


rows = []
for f in fields:
    sets = gold_sets(raw[f])
    all_labels = [lbl for s in sets for lbl in s]
    counts = Counter(all_labels)
    total = sum(counts.values())
    entropy = -sum((n / total) * log2(n / total) for n in counts.values()) if total else 0.0
    rows.append({
        "field": f,
        "label_space": len(counts),                                       # # distinct labels
        "cardinality": float(np.mean([len(s) for s in sets])),            # labels / paper
        "entropy": entropy,                                               # bits
        "answer_len": float(np.mean([len(", ".join(s)) for s in sets])),  # chars
    })

comp = pd.DataFrame(rows).set_index("field")

# Min-max normalize each feature across the six variables, then average.
feat_cols = ["label_space", "cardinality", "entropy", "answer_len"]
norm = comp[feat_cols].copy()
for c in feat_cols:
    lo, hi = norm[c].min(), norm[c].max()
    norm[c] = 0.0 if hi == lo else (norm[c] - lo) / (hi - lo)
comp["complexity"] = norm.mean(axis=1)

# F1 per model from the pipeline.
f1_piv = E.final_df.pivot(index="field", columns="model", values="f1")
comp["f1_GPT4"] = f1_piv["GPT-4"]
comp["f1_GPT5"] = f1_piv["GPT-5"]

comp_sorted = comp.sort_values("complexity")
comp_sorted.to_csv(OUT_TABLE)
print(comp_sorted[["label_space", "cardinality", "entropy", "answer_len", "complexity",
                   "f1_GPT4", "f1_GPT5"]].round(3).to_string())
