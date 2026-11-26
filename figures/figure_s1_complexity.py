"""Supplementary Fig. S1: extraction F1 against task complexity, one panel per model.

Full-text runs without the vocabulary in the prompt;
WM tracts are scored on vocabulary ids as everywhere else. Colour is the field,
the panel is the model. x is log (complexity spans 0.0005-1), y is linear and
starts at 0.5. Prints Spearman rho per model with exact permutation p (n = 6).

Input:  results/tables/task_complexity.csv (analysis/complexity.py)
Output: results/figures/figure_s1_complexity.png, .svg
"""
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import style as S                         # noqa: E402
from paths import FIGURES, TABLES, WITHOUT  # noqa: E402

S.apply_style()

ARM = WITHOUT
WMT = "WM tracts studied"

# x starts at the smallest complexity value and stops at the largest (rule 8);
# y stops at 1 and steps down in halves. Both middle ticks are pinned, because
# three_ticks computes an arithmetic midpoint and that is meaningless on a log
# axis.
# Round decades, labelled as powers of ten. Ticking the data minimum instead
# gave "0.0005" and "0.02" - long strings that crowd a 2.9 in panel and read as
# arbitrary. 10^-4 is one decade below the smallest variable (0.00055), so the
# leftmost point sits just inside the box rather than on the spine.
XLO, XMID, XHI = 1e-4, 1e-2, 1e0
# y starts at 0.5, not 0: nothing scores below 0.523, so a 0.25 or 0 floor left
# half of every one of six panels empty. This IS a truncated axis - it is chosen
# to fill the panels, and the caption should say where the axis starts.
#
# y is LINEAR while x is log. Over 0.5-1 - a single octave - a log y shifts a
# point by at most 8.5% of the panel height against linear, so it would be an
# axis labelled "log scale" that a reader measuring the 0.75 tick would find
# sitting essentially at the midpoint. x spans 3.3 decades and genuinely needs
# the log; y does not, so it does not claim it.
YLO, YHI = 0.5, 1.0

VAR_COLORS = {
    "Use of DTI":                            "#E0A03C",   # amber
    "Study population":                    "#2C6FAD",   # blue
    "Article type":                     "#3E9C86",   # teal
    "Dementia focus": "#C4699B",   # mauve
    "Disorders studied":                  "#7FB9E8",   # light blue
    WMT:                                        "#D4552B",   # orange-red
}
# Display names, keyed by the gold column each variable is scored from. The
# legend prints them in complexity order (the order KEYED is sorted in), which
# is also the order the points run left to right - not the order they are
# written here.
VAR_NAMES = {
    "Use of DTI": "Use of DTI",
    "Study population": "Study population",
    "Article type": "Article type",
    "Dementia focus": "Dementia focus",
    "Disorders studied": "Disorders studied",
    WMT: "White matter tracts studied",
}
ORDER = ["gpt-4o-mini", "gpt-5-mini", "gpt-5.4-nano", "gpt-5.4-mini",
         "gpt-5.6-luna"]

NCOL, NROW = 3, 2

# ── TYPOGRAPHY ───────────────────────────────────────────────────────────────
# One multiplier for every size, so style.py's hierarchy survives the scaling.
# Small panels on a modest canvas is exactly the case where type has to be
# scaled UP: printed size is points per canvas inch, and each panel here is only
# ~2.9 in wide.
# Panel titles, ticks, axis names and legend take BODY_SCALE. No figure title:
# the caption carries it.
FONT_SCALE = 1.55
BODY_SCALE = FONT_SCALE * 1.10
FS_PANEL_TITLE = S.FS_PANEL_TITLE * BODY_SCALE
FS_LABEL = S.FS_LABEL * BODY_SCALE
FS_TICK = S.FS_TICK * BODY_SCALE
FS_LEGEND = S.FS_LEGEND * BODY_SCALE

MS = 15 * 1.10                            # marker size, matched to the type

plt.rcParams.update({
    "font.size": FS_TICK,
    "axes.titlesize": FS_PANEL_TITLE,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    # style.py's 10 pt title pad assumes data that stops short of the top
    # spine. Here the three easiest variables sit at F1 0.98-0.998, so their
    # markers reach the top of the box and the model name lands on them.
    "axes.titlepad": 20,
})

# ── DATA ─────────────────────────────────────────────────────────────────────
comp = pd.read_csv(TABLES / "task_complexity.csv", index_col="field")   # sorted by complexity
KEYED = list(comp.index)
x = comp["complexity"]
f1 = pd.DataFrame({m: comp[f"f1_{m}_{ARM}"] for m in ORDER})

# ── FIGURE ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(NROW, NCOL, figsize=(11.4, 7.2))
flat = axes.ravel()

for k, m in enumerate(ORDER):
    ax = flat[k]
    row, col = divmod(k, NCOL)
    ax.set_xscale("log")

    for f in KEYED:
        ax.plot(x[f], f1.loc[f, m], "o", color=VAR_COLORS[f], ms=MS,
                markeredgecolor="white", markeredgewidth=1.6,
                zorder=3, clip_on=False)

    S.label_axes(ax, title=m, panel=True)
    ax.title.set_fontsize(FS_PANEL_TITLE)
    S.three_ticks(ax, XLO, XHI, axis="x", mid=XMID,
                  fmt=lambda v: f"$10^{{{int(round(np.log10(v)))}}}$")
    S.three_ticks(ax, YLO, YHI, axis="y")     # midpoint 0.75, arithmetic
    ax.xaxis.set_minor_locator(plt.NullLocator())
    S.square_axes(ax)
    ax.grid(axis="y", alpha=0.25, lw=0.8)

    # Tick labels only on the outer edge. "Bottom of its column" is not the same
    # as "bottom row": the third column ends at row 0, because the cell beneath
    # it holds the key.
    # Every panel prints its own y ticks (0.5 / 0.75 / 1), so a reader can read a
    # value off any panel without tracking back to the left column.
    #
    # x is different: its labels are long ("0.0005"), and on the inner panels
    # they would sit directly under the panel below's title. Hiding the numbers
    # but keeping the marks would leave unlabelled dashes, so length=0 strips
    # the marks on the same panels.
    bottom_of_column = (k + NCOL) >= len(ORDER)
    if not bottom_of_column:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)

# The empty sixth cell becomes the key.
key_ax = flat[len(ORDER)]
key_ax.axis("off")
key_ax.legend(handles=[Line2D([0], [0], color="none", marker="o",
                              markerfacecolor=VAR_COLORS[f],
                              markeredgecolor="white", markeredgewidth=1.4,
                              ms=MS, label=VAR_NAMES[f]) for f in KEYED],
              loc="center", frameon=False, handletextpad=0.6,
              labelspacing=0.85, fontsize=FS_LEGEND)

# Figure-level axis labels: every panel shares one scale, so labelling each of
# them six times would spend space repeating the same two strings.
# One label per axis for the whole figure: all five panels share both scales.
fig.supxlabel("Task complexity (log scale)", fontsize=FS_LABEL, fontweight="normal")
supy = fig.supylabel("F$_1$ Score", fontsize=FS_LABEL, fontweight="normal")

fig.tight_layout()
fig.canvas.draw()

# supylabel defaults to the far left edge of the canvas, which on a 3x2 grid
# leaves it a long way from the numbers it describes. Pull it in to sit just
# left of the y tick labels, measured after the draw because square_axes fixes
# the panel geometry only at draw time.
_tick_left = min(lb.get_window_extent().x0
                 for k in (0, NCOL) for lb in flat[k].get_yticklabels())
supy.set_x(_tick_left / fig.get_figwidth() / fig.dpi - 0.30 / fig.get_figwidth())

# A centred outer tick label sits half outside the box; turn the two ends in.
for k in range(len(ORDER)):
    if (k + NCOL) >= len(ORDER):
        flat[k].get_xticklabels()[0].set_ha("left")
        flat[k].get_xticklabels()[-1].set_ha("right")

# Diagnostics only - deliberately not drawn. At n = 6 the rank correlation is
# the weakest number available here (near the top only 1.0, 0.943, 0.886, 0.829
# are attainable, and even rho = -1 cannot reach p below 2/720), so the figure
# makes its claim with points. p is exact by enumeration, not scipy's large-n
# t-approximation, which is anti-conservative by ~3.5x at this n.
_NULL = np.array([spearmanr(np.arange(len(KEYED)), np.array(pm)).statistic
                  for pm in itertools.permutations(range(len(KEYED)))])
for i, f in enumerate(KEYED, start=1):
    print(f"{i}  {VAR_NAMES[f]:<20s} complexity={x[f]:.3f}")
for m in ORDER:
    r = spearmanr(x.values, f1[m].values).statistic
    print(f"{ARM:8s} {m:<13s} rho={r:+.3f}, exact p="
          f"{np.mean(np.abs(_NULL) >= abs(r) - 1e-12):.4f}")

S.save_figure(fig, str(FIGURES / "figure_s1_complexity.png"))
