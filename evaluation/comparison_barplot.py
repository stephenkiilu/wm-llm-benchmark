"""Shared per-field comparison bar plot, used by the F1 and accuracy figures.

Keeps both figures visually identical: same colors, sizing, bar width, value
labels, significance stars and field order. Typography, spines and PNG+SVG
output come from plot_style.py.
"""

import matplotlib.pyplot as plt
import numpy as np

from evaluation import plot_style as S
from evaluation.plot_style import (
    GREY,
    MODEL_COLORS,
    apply_style,
    flush_category_axis,
    save_figure,
    sig_note,
    three_ticks,
)

apply_style()
FONT_SCALE = 1.58
FONT_STACK = [f for f in S.FONT_STACK if f != "Helvetica Neue"]

FS_TITLE = S.FS_TITLE * FONT_SCALE
FS_LABEL = S.FS_LABEL * FONT_SCALE
FS_TICK = S.FS_TICK * FONT_SCALE
FS_LEGEND = S.FS_LEGEND * FONT_SCALE
FS_VALUE = S.FS_VALUE * FONT_SCALE
FS_ANNOT = S.FS_ANNOT * FONT_SCALE
FS_NOTE = S.FS_NOTE * FONT_SCALE

# Applied per figure via rc_context rather than to the global rcParams: other
# scripts import this module only for SHORT_NAMES / VARIABLE_COLORS, and should not
# inherit these figures' type scale just by importing it.
RC = {
    "font.sans-serif": FONT_STACK,
    "font.size": FS_TICK,
    "axes.titlesize": FS_TITLE,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "legend.title_fontsize": FS_LEGEND,
}

# Clean display names for the x-axis (shared by every comparison figure).
SHORT_NAMES = {
    "Human_vs_non_human_study": "Subject Species",
    "Does it use DTI?": "DTI Methodology",
    "Review or single study?": "Study Design",
    "Does it study dementia or related diseases?": "Dementia-Related",
    "Which diseases are studied": "Disorders Studied",
    "WM tracts studied": "White Matter Tracts",
}

MODELS = ["GPT-4", "GPT-5"]
DISPLAY_NAMES = {"GPT-4": "GPT-4o-mini", "GPT-5": "GPT-5-mini"}
BAR_WIDTH = 0.36
GROUP_GAP = BAR_WIDTH / 3
GROUP_SPACING = 2 * BAR_WIDTH + GROUP_GAP
BOX_ASPECT = 0.55
FIGSIZE = (12.5, 8.5)

VARIABLE_COLORS = {
    "Human_vs_non_human_study": "#0072B2",                    # blue
    "Does it use DTI?": "#E69F00",                            # orange
    "Review or single study?": "#009E73",                     # green
    "Does it study dementia or related diseases?": "#CC79A7",  # magenta
    "Which diseases are studied": "#56B4E9",                  # sky
    "WM tracts studied": "#D55E00",                           # vermillion
}


def _two_lines(name):
    """Break a field name into at most two lines, at the break nearest its middle.

    At this type size a rotated "White Matter Tracts" is wider than the space one
    group of bars gets, so the labels ran into each other. Two upright lines fit,
    and upright labels are easier to read than angled ones.
    """
    points = [i for i, ch in enumerate(name) if ch in " -"]
    if not points:
        return name
    i = min(points, key=lambda p: abs(p - len(name) / 2))
    head = name[:i + 1] if name[i] == "-" else name[:i]
    return f"{head}\n{name[i + 1:]}"


def plot_comparison(order, piv, lo, hi, sig, ylabel, title, out_png,
                    xlabel="Extraction Variable", box_aspect=BOX_ASPECT,
                    note="auto"):
    """Render one grouped GPT-4 vs GPT-5 bar plot and save PNG + SVG.

    order : list of field keys (left -> right)
    piv, lo, hi : DataFrames indexed by field, columns = models (score / CI bounds)
    sig  : dict field -> significance symbol ("", "*", "**", "***")
    note : "auto" keys only the marks present, None omits it, or pass a string
    """
    if note == "auto":
        note = S.sig_key(sig.values())
    with plt.rc_context(RC):
        return _draw(order, piv, lo, hi, sig, ylabel, title, out_png, xlabel,
                     box_aspect, note)


def _draw(order, piv, lo, hi, sig, ylabel, title, out_png, xlabel, box_aspect,
          note):
    x = np.arange(len(order)) * GROUP_SPACING
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for i, model in enumerate(MODELS):
        vals = piv.loc[order, model].values
        errs = [np.maximum(0, vals - lo.loc[order, model].values),
                np.maximum(0, hi.loc[order, model].values - vals)]
        bars = ax.bar(x + i * BAR_WIDTH, vals, BAR_WIDTH,
                      label=DISPLAY_NAMES.get(model, model),
                      color=MODEL_COLORS[model], yerr=errs, capsize=6,
                      ecolor=GREY, error_kw={"elinewidth": 1.8,
                                             "capthick": 1.8})
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, max(v - 0.055, 0.02),
                    f"{v:.3f}", ha="center", va="top",
                    fontsize=FS_VALUE, color="white")

    for idx, f in enumerate(order):
        s = sig.get(f, "")
        if isinstance(s, str) and s.strip():
            mh = max(hi.loc[f, "GPT-4"], hi.loc[f, "GPT-5"])
            # clip_on=False: the y-axis stops at 1.0, so a star above a near-
            # ceiling bar sits just outside the box rather than stretching it.
            ax.text(x[idx] + BAR_WIDTH / 2, mh + 0.02, s, ha="center", va="bottom",
                    fontsize=FS_ANNOT * 1.2, color=GREY, clip_on=False)

    ax.set_xticks(x + BAR_WIDTH / 2)
    ax.set_xticklabels([_two_lines(SHORT_NAMES.get(f, f)) for f in order],
                       rotation=0, ha="center", fontsize=FS_TICK,
                       linespacing=1.15)
    S.label_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    ax.title.set_fontsize(FS_TITLE)
    ax.xaxis.label.set_fontsize(FS_LABEL)
    ax.yaxis.label.set_fontsize(FS_LABEL)
    ax.xaxis.labelpad = 16
    three_ticks(ax, 0.0, 1.0, axis="y")
    flush_category_axis(ax, -BAR_WIDTH / 2, x[-1] + 1.5 * BAR_WIDTH)
    ax.set_box_aspect(box_aspect)   
    ax.legend(title="Model", loc="upper right", frameon=False,
              handlelength=1.4, labelspacing=0.35, borderaxespad=0.2)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    if note:
        fig.canvas.draw()
        bb = ax.xaxis.label.get_window_extent().transformed(
            ax.transAxes.inverted())
        n = sig_note(ax, note, y=bb.y0 - 0.015)
        n.set_fontsize(FS_NOTE)

    fig.tight_layout()
    return save_figure(fig, out_png)
