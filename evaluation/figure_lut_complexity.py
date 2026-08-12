"""Combined LUT and task-complexity figure, three panels.

    a  White-matter-tract F1 with and without the look-up table, per model
    b  F1 vs task complexity, no LUT
    c  F1 vs task complexity, with the LUT supplied

Both halves are computed live by task_complexity.py and lut_ablation.py rather
than transcribed, so an upstream change cannot leave this figure stale. The LUT
only renames tracts, so panel c differs from panel b in exactly two points.

Output: results/figures/lut_complexity.png (+ .svg)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

from evaluation import comparison_barplot as U
from evaluation import lut_ablation as L
from evaluation import plot_style as S
from evaluation import task_complexity as CX

# Pin the backend: the platform default renders this canvas one pixel narrower.
plt.switch_backend("Agg")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PNG = os.path.join(BASE_DIR, "results", "figures", "lut_complexity.png")

comp = CX.comp
short = U.SHORT_NAMES
var_colors = U.VARIABLE_COLORS
# "GPT-4"/"GPT-5" are the pipeline's keys; the figure shows the full model names,
# matching the captions. Same mapping the bar figures use.
display = U.DISPLAY_NAMES
markers = S.MODEL_MARKERS
WMT = "WM tracts studied"

# ---------------------------------------------------------------------------
# Panel A numbers, and the cross-check that panel B is the same no-LUT run.
# ---------------------------------------------------------------------------
NO_LUT = {"GPT-4": L.f1_gpt4_no_lut, "GPT-5": L.f1_gpt5_no_lut}
WITH_LUT = {"GPT-4": L.f1_gpt4_lut, "GPT-5": L.f1_gpt5_lut}
CI_NO = {"GPT-4": (L.ci_lo_4_no, L.ci_hi_4_no),
         "GPT-5": (L.ci_lo_5_no, L.ci_hi_5_no)}
CI_LUT = {"GPT-4": (L.ci_lo_4_lut, L.ci_hi_4_lut),
          "GPT-5": (L.ci_lo_5_lut, L.ci_hi_5_lut)}
SIG = {"GPT-4": L.sig_4, "GPT-5": L.sig_5}

for m, key in (("GPT-4", "f1_GPT4"), ("GPT-5", "f1_GPT5")):
    a, b = NO_LUT[m], comp.loc[WMT, key]
    if abs(a - b) > 5e-3:
        raise SystemExit(
            f"{m} no-LUT WMT F1 disagrees between panels: eval_lut says {a:.3f}, "
            f"the complexity pipeline says {b:.3f}. One of them is reading a LUT "
            f"run - see Evaluation/evaluation_gpt4_vs_gpt5.py inputs."
        )

# Panel C = panel B with the WMT point replaced by its with-LUT value.
F1_NO_LUT = {"GPT-4": comp["f1_GPT4"].copy(), "GPT-5": comp["f1_GPT5"].copy()}
F1_WITH_LUT = {m: s.copy() for m, s in F1_NO_LUT.items()}
for m in ("GPT-4", "GPT-5"):
    F1_WITH_LUT[m][WMT] = WITH_LUT[m]

# ---------------------------------------------------------------------------
# Typography. 1.9 is figure_AB_final's scale, so the two figures carry identical
# type. Sized for the page rather than the screen: dropped into a Google Doc (or
# any single-column layout) the figure is scaled to the ~6.5 in text width, which
# divides every point size by 2.4 - so 1.45 printed 6.7 pt ticks against the
# document's 11 pt body text, and 1.9 prints 8.8 pt. Enlarging the canvas would
# make that worse, not better: the wider the canvas, the harder the page scales it
# down. What matters is points per canvas inch.
# ---------------------------------------------------------------------------
FONT_SCALE = 1.9
FONT_STACK = [f for f in S.FONT_STACK if f != "Helvetica Neue"]   # Arial, as figure AB
FS_TITLE = S.FS_TITLE * FONT_SCALE
FS_PANEL_TITLE = S.FS_PANEL_TITLE * FONT_SCALE
FS_LABEL = S.FS_LABEL * FONT_SCALE
FS_TICK = S.FS_TICK * FONT_SCALE
FS_LEGEND = S.FS_LEGEND * FONT_SCALE
FS_VALUE = S.FS_VALUE * FONT_SCALE
FS_ANNOT = S.FS_ANNOT * FONT_SCALE
FS_NOTE = S.FS_NOTE * FONT_SCALE
FS_PANEL_LETTER = S.FS_PANEL_LETTER * FONT_SCALE

S.apply_style()
plt.rcParams.update({
    "font.sans-serif": FONT_STACK,
    "font.size": FS_TICK,
    "axes.titlesize": FS_TITLE,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "legend.title_fontsize": FS_LEGEND,
})

LUT_HATCH = "///"
# Scatter `s` is area, so the printed symbol diameter goes as its square root:
# 520 is 2x the linear size of the 130 used before (sqrt(520/130) = 2), which is
# what it takes for the shape (square vs circle) to stay legible once the page
# scales the figure to column width. LEGEND_MS is the matching diameter in points,
# kept in step so the key and the data show the same symbol.
MARKER_SIZE = 520
LEGEND_MS = 20

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 7.4))


def label_panel(ax, title, xlabel, ylabel):
    S.label_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel, panel=True)
    ax.title.set_fontsize(FS_PANEL_TITLE)
    ax.xaxis.label.set_fontsize(FS_LABEL)
    ax.yaxis.label.set_fontsize(FS_LABEL)


# ===========================================================================
# Panel A: WM tract F1, LUT vs no LUT.
# ===========================================================================
models = ["GPT-4", "GPT-5"]
x = np.arange(len(models))
BAR_W = 0.3


def errs(vals, cis):
    lo = np.array([v - c[0] for v, c in zip(vals, cis)])
    hi = np.array([c[1] - v for v, c in zip(vals, cis)])
    return [np.maximum(0, lo), np.maximum(0, hi)]


vals_no = [NO_LUT[m] for m in models]
vals_lut = [WITH_LUT[m] for m in models]
colors = [S.MODEL_COLORS[m] for m in models]

bars_no = axA.bar(x - BAR_W / 2, vals_no, BAR_W, color=colors, edgecolor="white",
                  yerr=errs(vals_no, [CI_NO[m] for m in models]), capsize=6,
                  error_kw={"elinewidth": 1.8, "capthick": 1.8, "ecolor": S.GREY})
bars_lut = axA.bar(x + BAR_W / 2, vals_lut, BAR_W, color=colors, edgecolor="black",
                   hatch=LUT_HATCH,
                   yerr=errs(vals_lut, [CI_LUT[m] for m in models]), capsize=6,
                   error_kw={"elinewidth": 1.8, "capthick": 1.8, "ecolor": S.GREY})

for bars in (bars_no, bars_lut):
    for bar in bars:
        v = bar.get_height()
        axA.text(bar.get_x() + bar.get_width() / 2, max(v - 0.045, 0.02),
                 f"{v:.2f}", ha="center", va="top", fontsize=FS_VALUE,
                 color="white")

for i, m in enumerate(models):
    if SIG[m]:
        top = max(CI_NO[m][1], CI_LUT[m][1])
        axA.text(x[i], top + 0.03, SIG[m], ha="center", va="bottom",
                 fontsize=FS_ANNOT * 1.2, color=S.GREY, clip_on=False)

label_panel(axA, "WM tract extraction: LUT effect", "Model", "F1 Score")
axA.set_xticks(x)
axA.set_xticklabels([display[m] for m in models])
S.three_ticks(axA, 0.0, 1.0, axis="y")
S.flush_category_axis(axA, x[0] - BAR_W, x[-1] + BAR_W)
S.square_axes(axA)
axA.grid(axis="y", linestyle="--", alpha=0.3)
# Colour is the model (also on the x-axis); the hatch is the only thing the
# legend has to decode, so it is small enough to sit inside the panel.
axA.legend(handles=[Patch(facecolor="#8A8A8A", edgecolor="white", label="No LUT"),
                    Patch(facecolor="#8A8A8A", edgecolor="black",
                          hatch=LUT_HATCH, label="With LUT")],
           loc="upper left", frameon=False, handlelength=1.6, labelspacing=0.35,
           borderaxespad=0.3)

# ===========================================================================
# Panels B and C: F1 vs complexity, without and with the LUT.
# ===========================================================================
xfull = comp["complexity"].values
rho = {}

for ax, f1, tag, panel_title in (
        (axB, F1_NO_LUT, "no LUT", "F1 vs task complexity, no LUT"),
        (axC, F1_WITH_LUT, "with LUT", "F1 vs task complexity, with LUT")):
    for model in models:
        y = f1[model].loc[comp.index].values
        r, p = spearmanr(xfull, y)
        rho[(tag, model)] = (r, p)
        # No fitted line: with n = 6 variables clustered at the low-complexity end,
        # an OLS slope is set by the two open-vocabulary points and would imply a
        # linear dose-response the design cannot support. The points carry the claim.
        for field in comp.index:
            ax.scatter(comp.loc[field, "complexity"], f1[model][field],
                       s=MARKER_SIZE, marker=markers[model],
                       color=var_colors[field], edgecolor="white", linewidth=1.4,
                       zorder=3, clip_on=False)

    # No in-plot variable labels: every variable is already named in the shared
    # legend below, and repeating two of them on the points only added text that
    # crowded the marker cluster.
    label_panel(ax, panel_title, "Task complexity", "F1 Score")
    S.three_ticks(ax, 0.0, 1.0, axis="both")
    S.share_origin_label(ax, keep="y")
    S.square_axes(ax)
    ax.grid(axis="y", alpha=0.25, lw=0.6)

# Printed as a diagnostic only. The no-LUT values are the ones the caption quotes;
# the with-LUT rho is deliberately not reported - see the note in the docstring.
for tag in ("no LUT", "with LUT"):
    for m in models:
        r, p = rho[(tag, m)]
        print(f"{tag:9s} {m}: Spearman rho={r:+.3f}, p={p:.4f}")

# ---------------------------------------------------------------------------
# One shared legend for B and C instead of a copy inside each: at this size two
# six-entry legends would take more of those boxes than the data does.
# ---------------------------------------------------------------------------
var_handles = [Patch(facecolor=var_colors[f], edgecolor="none", label=short[f])
               for f in comp.sort_values("complexity").index]
model_handles = [
    Line2D([0], [0], linestyle="none", marker=markers[m], ms=LEGEND_MS,
           markerfacecolor="#B0B0B0", markeredgecolor="white", markeredgewidth=1.4,
           label=display[m])
    for m in models
]

fig.suptitle("Look-up table grounding and task complexity", fontsize=FS_TITLE,
             y=0.985)
fig.tight_layout(rect=[0, 0.16, 1, 0.95])   # room for the shared legend strip
fig.canvas.draw()

# Centred under panels B and C, which are what it describes.
bb = axB.get_position(), axC.get_position()
legend = fig.legend(handles=var_handles + model_handles,
                    loc="upper center", ncol=3, frameon=False,
                    bbox_to_anchor=((bb[0].x0 + bb[1].x1) / 2, 0.155),
                    handlelength=1.6, columnspacing=1.6, labelspacing=0.4,
                    fontsize=FS_LEGEND)
legend.set_title("",
                 prop={"size": FS_LEGEND})

# *** is the only mark the figure prints, so the key defines only that one. The
# threshold is 0.002 rather than 0.001 because the bootstrap has 1,000 resamples
# and a two-sided p of 2*min(...), so 0.002 is the smallest p it can resolve -
# quoting a tighter bound than the resampling supports would overstate it.
note = fig.text(axA.get_position().x0, 0.055,
                "*** p<0.002",
                fontsize=FS_NOTE, style="italic", color=S.MUTED, va="top")


def panel_letter(ax, letter, dx_in=0.62, dy_in=0.42):
    pos = ax.get_position()
    return fig.text(pos.x0 - dx_in / fig.get_figwidth(),
                    pos.y1 + dy_in / fig.get_figheight(), letter,
                    fontsize=FS_PANEL_LETTER, fontweight=S.PANEL_LETTER_WEIGHT,
                    va="bottom", ha="left")


for ax, letter in ((axA, "a"), (axB, "b"), (axC, "c")):
    panel_letter(ax, letter)

S.save_figure(fig, OUT_PNG)
