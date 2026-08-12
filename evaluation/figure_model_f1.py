"""GPT-4o-mini vs GPT-5-mini F1 per extraction variable, no-LUT.

Output: results/figures/model_f1.png (+ .svg)
"""

import os

from evaluation import comparison_barplot as U
from evaluation import model_comparison as E

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PNG = os.path.join(BASE_DIR, "results", "figures", "model_f1.png")

f1_df = E.final_df.rename(columns={"f1": "score"})
piv = f1_df.pivot(index="field", columns="model", values="score")
lo = f1_df.pivot(index="field", columns="model", values="ci_lower")
hi = f1_df.pivot(index="field", columns="model", values="ci_upper")
sig = dict(zip(E.stats_df["field"], E.stats_df["sig_symbol"]))

# Shared field order with the accuracy figure: sorted by GPT-4 F1, descending.
order = piv["GPT-4"].sort_values(ascending=False).index.tolist()

U.plot_comparison(
    order,
    piv,
    lo,
    hi,
    sig,
    xlabel="Extraction Variable",
    ylabel="F1 Score",
    title="GPT-4o-mini vs GPT-5-mini F1 Performance",
    out_png=OUT_PNG,
)
