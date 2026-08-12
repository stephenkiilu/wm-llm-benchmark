"""F1 by source section for each extraction variable, GPT-5-mini, no-LUT.

Reads the bootstrapped table written by section_ablation.py.

Output: results/figures/section_ablation.png (+ .svg)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from evaluation import plot_style as S

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "results", "tables", "section_ablation.csv")
OUT_PATH = os.path.join(BASE_DIR, "results", "figures", "section_ablation.png")

S.apply_style()


# ---- config -----------------------------------------------------------------
SECTION_ORDER = [
    "Abstract", "Methods", "Results", "Discussion",
    "Abstract+Methods", "Abstract+Results", "Abstract+Discussion",
    "Methods+Results", "Methods+Discussion", "Results+Discussion",
    "Abstract+Methods+Results", "Abstract+Methods+Discussion",
    "Abstract+Results+Discussion", "Methods+Results+Discussion",
    "Full Text",
]
FIELDS = [
    "DTI Methodology [Y/N]", "Subject Species [Human/Other]",
    "Dementia-Related[Y/N]", "Study Design [Review/Original]",
    "Neurological Disorders", "White Matter Tracts",
]
SHORT = {
    "DTI Methodology [Y/N]": "DTI Methodology",
    "Subject Species [Human/Other]": "Subject Species",
    "Dementia-Related[Y/N]": "Dementia-Related",
    "Study Design [Review/Original]": "Study Design",
    "Neurological Disorders": "Disorders Studied",
    "White Matter Tracts": "White Matter Tracts",
}
COLOR = "#2C6E91"        # section combinations
FT_COLOR = "#C0392B"     # Full Text reference
REFERENCE_SECTION = "Full Text"
PANEL_LETTERS = "abcdef"   # panel labels, in FIELDS order

TITLE = "Variable-Level Extraction F1 by Source Section"
XLABEL = "F1 Score (95% CI)"
YLABEL = "Source Section(s)"

# F1 axis range. Data spans ~0.28-1.0, so starting at 0.2 keeps the differences
# between section combinations readable; ticks are min / midpoint / max.
X_MIN, X_MAX = 0.2, 1.0

# ---- load -------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
df = df[df["section"].isin(SECTION_ORDER)].copy()

# ---- plot -------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(13, 9), sharey=True)
axes = axes.ravel()
y = np.arange(len(SECTION_ORDER))

for i, (ax, field) in enumerate(zip(axes, FIELDS)):
    sub = df[df["field"] == field].set_index("section").reindex(SECTION_ORDER)
    f1 = sub["f1"].values
    lo = sub["ci_lower"].values
    hi = sub["ci_upper"].values
    xerr = np.vstack([f1 - lo, hi - f1])          # asymmetric CI
    cols = [FT_COLOR if s == REFERENCE_SECTION else COLOR for s in SECTION_ORDER]

    for j in range(len(y)):
        ax.errorbar(f1[j], y[j], xerr=xerr[:, j:j + 1], fmt="o", ms=4.5,
                    color=cols[j], ecolor=cols[j], elinewidth=1.2,
                    capsize=2.5, capthick=1.0)

    # Panel titles stay regular weight; only the figure title is bold.
    S.label_axes(ax, title=SHORT[field], panel=True)
    # F1 axis stops exactly at 1.0, ticks at 0.2 / 0.6 / 1.
    S.three_ticks(ax, X_MIN, X_MAX, axis="x")
    S.square_axes(ax)
    ax.grid(axis="x")
    ax.tick_params(length=3)

    S.panel_letter(ax, PANEL_LETTERS[i])

# Section axis is categorical, so it keeps all 15 labels (the 3-tick rule is for
# numeric axes). SECTION_ORDER reads bottom-to-top: Abstract at the bottom,
# Full Text at the top.
for i, ax in enumerate(axes):
    if i % 3 == 0:                                 # y labels only on left column
        ax.set_yticks(y)
        ax.set_yticklabels(SECTION_ORDER, fontsize=S.FS_TICK - 2)
    if i >= 3:                                     # x label only on bottom row
        ax.set_xlabel(XLABEL, fontsize=S.FS_LABEL)

fig.supylabel(YLABEL, fontsize=S.FS_LABEL, fontweight="normal")

handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR, ms=7,
           label="Section combination"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=FT_COLOR, ms=7,
           label="Full Text (reference)"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.01))

fig.suptitle(TITLE, y=0.985)
fig.tight_layout(rect=[0.02, 0.03, 1, 0.97])
S.save_figure(fig, OUT_PATH)
