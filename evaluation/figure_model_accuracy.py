"""GPT-4o-mini vs GPT-5-mini accuracy per extraction variable, no-LUT.

Uses the same bootstrap scheme and seed as the F1 pipeline, so the two figures
are directly comparable.

Output: results/tables/model_accuracy.csv, results/figures/model_accuracy.png (+ .svg)
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from statsmodels.stats.multitest import fdrcorrection

from evaluation import comparison_barplot as U
from evaluation import model_comparison as E

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_TABLE = os.path.join(BASE_DIR, "results", "tables", "model_accuracy.csv")
OUT_PNG = os.path.join(BASE_DIR, "results", "figures", "model_accuracy.png")

raw_gpt4, raw_gpt5 = E.raw_gpt4, E.raw_gpt5
N_BOOT, RANDOM_SEED, SKIP_EMPTY_GOLD = E.N_BOOT, E.RANDOM_SEED, E.SKIP_EMPTY_GOLD

# Multi-label fields that are conceptually single-valued -> exact-match accuracy.
FORCE_ACCURACY_FIELDS = {"Review or single study?"}
fields = list(raw_gpt4.keys())


def acc_metric(field_name, fd, indices) -> float:
    if fd["type"] == "binary":
        return accuracy_score(fd["y_true"][indices], fd["y_pred"][indices])
    preds = fd["y_pred"][indices].tolist()
    refs = fd["y_true"][indices].tolist()
    if SKIP_EMPTY_GOLD:
        preds, refs = E.filter_empty_gold(preds, refs)
    if field_name in FORCE_ACCURACY_FIELDS:
        vals = [1.0 if set(p) == set(r) else 0.0 for p, r in zip(preds, refs)]
        return float(np.mean(vals)) if vals else 0.0
    return E._jaccard_samples(preds, refs)


# ── Paired bootstrap (same scheme/seed as the F1 pipeline) ───────────────────
num_samples = len(raw_gpt4[fields[0]]["y_true"])
np.random.seed(RANDOM_SEED)
boot_idx = [np.random.choice(num_samples, num_samples, replace=True) for _ in range(N_BOOT)]
all_idx = np.arange(num_samples)

rows, raw_p = [], []
print("\n" + "=" * 70)
print("ACCURACY — GPT-4 vs GPT-5")
print("=" * 70)
for f in fields:
    r4, r5 = raw_gpt4[f], raw_gpt5[f]
    pt4, pt5 = acc_metric(f, r4, all_idx), acc_metric(f, r5, all_idx)
    d4 = np.array([acc_metric(f, r4, ix) for ix in boot_idx])
    d5 = np.array([acc_metric(f, r5, ix) for ix in boot_idx])
    for model, pt, dist in [("GPT-4", pt4, d4), ("GPT-5", pt5, d5)]:
        rows.append({"field": f, "model": model, "score": round(float(pt), 3),
                     "ci_lower": float(np.percentile(dist, 2.5)),
                     "ci_upper": float(np.percentile(dist, 97.5))})
    diff = d5 - d4
    raw_p.append(min(1.0, 2.0 * min(float(np.mean(diff <= 0)), float(np.mean(diff >= 0)))))
    print(f"  {f:45s}  GPT-4={pt4:.3f}  GPT-5={pt5:.3f}")

_, adj_p = fdrcorrection(raw_p, method="indep")
sig = {f: E.sig_symbol(adj_p[i]) for i, f in enumerate(fields)}

acc_df = pd.DataFrame(rows)
# Full stats per field (same columns as the F1 CSV) so the table is fully backed.
stats_df = pd.DataFrame({
    "field": fields,
    "p_value_raw": [round(float(raw_p[i]), 3) for i in range(len(fields))],
    "p_value_fdr": [round(float(adj_p[i]), 3) for i in range(len(fields))],
    "sig_symbol": [sig[f] for f in fields],
})
acc_df.merge(stats_df, on="field", how="left").to_csv(
    OUT_TABLE, index=False
)

piv = acc_df.pivot(index="field", columns="model", values="score")
lo = acc_df.pivot(index="field", columns="model", values="ci_lower")
hi = acc_df.pivot(index="field", columns="model", values="ci_upper")

# Shared field order: sorted by GPT-4 F1 (same order as the F1 figure).
f1_piv = E.final_df.pivot(index="field", columns="model", values="f1")
order = f1_piv["GPT-4"].sort_values(ascending=False).index.tolist()

U.plot_comparison(
    order, piv, lo, hi, sig,
    xlabel="Extraction Variable",
    ylabel="Accuracy Score",
    title="GPT-4o-mini vs GPT-5-mini Accuracy Performance",
    out_png=OUT_PNG,
)
