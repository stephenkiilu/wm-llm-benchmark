"""Figure 2: what the vocabulary buys, at what cost, and which sections each field needs.

a  WM tract F1 without (light) and with (full colour) the vocabulary, per model,
   95% bootstrap CIs; stars are Benjamini-Hochberg q-values across the models.
b  Estimated cost of extracting the 622 articles with the vocabulary.
c  F1 of GPT-5.6-luna by input section, six fields x seven inputs; gold boxes
   mark the best input per field (all 15 inputs are in Table S2).

Inputs: results/tables/vocabulary_effect.csv, tract_results.csv, efficiency.csv,
        section_f1.csv
Output: results/figures/figure2.png (600 dpi), .svg, .pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import style as S  # noqa: E402
from page import Page, check_page, legend_style, style_chart  # noqa: E402
from paths import FIGURES, TABLES, WITH, WITHOUT  # noqa: E402

# ── SETTINGS ─────────────────────────────────────────────────────────────────
# The manuscript's full text width (6.5 in), so all three panels share one row.
WIDTH_MM = 165
HEIGHT_MM = 64
SCALE = 1.0          # printed size: 165 x 64 mm (6.50 x 2.52 in)
DPI = 600
TEXT_PT = 9
LETTER_PT = 10
AXIS_LW = 0.8
# Print legibility: Helvetica Neue Regular's thin strokes read faint once Word
# scales the figure, so this figure uses Arial in pure black, with 0.8 pt lines.
FONT = ["Arial", "Helvetica", "DejaVu Sans"]
BLACK = "#000000"
ARM, BASE_ARM = WITH, WITHOUT

# Cheapest first, the order used in every model panel.
MODELS = ["gpt-4o-mini", "gpt-5.6-luna", "gpt-5.4-nano", "gpt-5-mini", "gpt-5.4-mini"]
LABELS = {"gpt-4o-mini": "GPT-4o-mini", "gpt-5.6-luna": "GPT-5.6-luna",
          "gpt-5.4-nano": "GPT-5.4-nano", "gpt-5-mini": "GPT-5-mini",
          "gpt-5.4-mini": "GPT-5.4-mini"}
# The model colours of the earlier LUT and cost figures, kept so readers who
# saw those see the same mapping here.
MODEL_COLORS = {"gpt-4o-mini": "#777777", "gpt-5.6-luna": "#CC79A7",
                "gpt-5.4-nano": "#009E73", "gpt-5-mini": "#E69F00",
                "gpt-5.4-mini": "#0072B2"}
Q_ALPHA = 0.05
F1_AXIS = (0.4, 0.72)          # cropped, figures4papers style; ticks at 0.4 / 0.55 / 0.7
COST_MAX = 9

# Panel c: rows are fields, columns are inputs.
FIELDS = [("Article type", "Type"), ("Study population", "Population"),
          ("Use of DTI", "DTI use"), ("Dementia focus", "Dementia"),
          ("Disorders studied", "Disorders"), ("WM tracts studied", "Tracts")]
# The single sections define their letter in the label, so A+D and A+M+R
# read without the caption. Seven of the 15 evaluated inputs; none of the eight
# left out is the best input for any field (all 15 are in the Supplement).
INPUTS = [("A", "Abstract (A)"), ("M", "Methods (M)"), ("R", "Results (R)"),
          ("D", "Discussion (D)"), ("A+D", "A+D"), ("A+M+R", "A+M+R"),
          ("Full text", "Full text")]
HEAT_CMAP, HEAT_MIN, HEAT_MAX = "Blues", 0.3, 1.0
BEST_EDGE = "#D4A017"

# ── LAYOUT (mm from the top-left corner) ────────────────────────────────────
# All three plots share one top (9 mm) and one height (30 mm). c's column
# labels hang below it, beside the model legend, so c does not push the row down.
PANEL_TOP, PANEL_H = 9, 30
CHART_BOX = {"a": (12.5, PANEL_TOP, 42, PANEL_H), "b": (76, PANEL_TOP, 21, PANEL_H)}
CELL_W = 6.7                   # wide enough for "1.00" at 9 pt
HEAT_RIGHT = 163
MODEL_LEGEND = (4, 42.5)       # x, y of the legend under a and b

LEGEND_STYLE = legend_style(TEXT_PT)


# ── INPUTS ───────────────────────────────────────────────────────────────────
def load():
    effect = pd.read_csv(TABLES / "vocabulary_effect.csv").set_index("model").loc[MODELS]
    all_results = pd.read_csv(TABLES / "tract_results.csv").set_index(["arm", "model"])
    results = all_results.loc[ARM].loc[MODELS]
    baseline = all_results.loc[BASE_ARM].loc[MODELS]
    cost = (pd.read_csv(TABLES / "efficiency.csv").query("arm == @ARM")
            .set_index("model").loc[MODELS, "cost"])
    sections = pd.read_csv(TABLES / "section_f1.csv")
    heat = (sections.pivot(index="field", columns="config", values="f1")
            .reindex(index=[f for f, _ in FIELDS], columns=[c for c, _ in INPUTS]))
    if heat.isna().any().any():
        raise ValueError(f"section_f1.csv lacks cells:\n{heat[heat.isna().any(axis=1)]}")
    lowest = min(baseline.ci_lower.min(), results.ci_lower.min())
    if lowest < F1_AXIS[0]:
        raise ValueError(f"A CI reaches {lowest:.3f}, below panel a's cropped axis.")
    if cost.max() > COST_MAX:
        raise ValueError("A model costs more than the top of panel b's axis.")
    return effect, baseline, results, cost, heat


# ── PANELS ───────────────────────────────────────────────────────────────────
def stars(q):
    """Significance stars for a BH q-value; blank when not significant."""
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < Q_ALPHA else ""


def tint(color, amount=0.62):
    """The model colour mixed toward white, for the without-vocabulary bar."""
    rgb = np.array(to_rgb(color))
    return tuple(rgb + (1 - rgb) * amount)


def model_bars(ax):
    """Shared x axis for a and b: one slot per model, no tick labels (the legend
    names the models, as in the figures4papers bar charts)."""
    ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ax.set_xticks([])


def panel_vocabulary(ax, effect, baseline, results):
    """Tract F1 without (light) and with (full colour) the vocabulary, per model."""
    width = 0.38
    for i, model in enumerate(MODELS):
        color = MODEL_COLORS[model]
        for arm, dx, fill in [(baseline, -width / 2, tint(color)), (results, width / 2, color)]:
            r = arm.loc[model]
            ax.bar(i + dx, r.f1, width=width, color=fill,
                   yerr=[[r.f1 - r.ci_lower], [r.ci_upper - r.f1]], capsize=1.5,
                   error_kw=dict(elinewidth=AXIS_LW, capthick=AXIS_LW, ecolor=BLACK))
        mark = stars(effect.loc[model, "q_value"])
        if mark:
            ax.text(i, results.loc[model, "ci_upper"] + 0.008, mark, ha="center",
                    va="bottom", fontsize=TEXT_PT)
    model_bars(ax)
    ax.set_ylim(*F1_AXIS)
    mid = (F1_AXIS[0] + 0.7) / 2
    ax.set_yticks([F1_AXIS[0], mid, 0.7], [f"{F1_AXIS[0]:g}", f"{mid:g}", "0.7"])
    ax.set_ylabel("WM tract F$_1$")


def panel_cost(ax, cost):
    """Cost per model, in the same colours and order as panel a."""
    ax.bar(np.arange(len(MODELS)), [cost[m] for m in MODELS], width=0.7,
           color=[MODEL_COLORS[m] for m in MODELS])
    model_bars(ax)
    ax.set_ylim(0, COST_MAX)
    ax.set_yticks([0, COST_MAX / 2, COST_MAX], ["0", f"{COST_MAX / 2:g}", f"{COST_MAX:g}"])
    ax.set_ylabel("Cost in USD")


def apa(value):
    """F1 to two decimals without the leading zero (".98"), APA style for a
    quantity that cannot exceed 1; "0.98" is wider than a cell."""
    text = f"{value:.2f}"
    return text[1:] if text.startswith("0") else text


def panel_sections(page, heat):
    """F1 per field and input, best input per field boxed; same height as a and b."""
    n_rows, n_cols = heat.shape
    width = CELL_W * n_cols
    ax = page.axes(HEAT_RIGHT - width, PANEL_TOP, width, PANEL_H)
    values = heat.to_numpy()
    ax.imshow(values, cmap=HEAT_CMAP, vmin=HEAT_MIN, vmax=HEAT_MAX, aspect="auto")
    for i in range(n_rows):
        for j in range(n_cols):
            v = values[i, j]
            dark = (v - HEAT_MIN) / (HEAT_MAX - HEAT_MIN) > 0.6
            ax.text(j, i, apa(v), ha="center", va="center",
                    fontsize=TEXT_PT, color="white" if dark else BLACK)
        # Best input on unrounded scores; ties all boxed.
        for j in np.flatnonzero(values[i] == values[i].max()):
            ax.add_patch(Rectangle((j - 0.46, i - 0.44), 0.92, 0.88, fill=False,
                                   ec=BEST_EDGE, lw=1.0))
    # Vertical and below the cells, so the heatmap's top lines up with a and b.
    ax.set_xticks(range(n_cols), [short for _, short in INPUTS], rotation=90)
    ax.set_yticks(range(n_rows), [short for _, short in FIELDS])
    ax.tick_params(length=0, pad=1.4, labelsize=TEXT_PT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def model_legend(page):
    """The models, named once for a and b. The caption says what the light bars mean."""
    handles = [Patch(color=MODEL_COLORS[m], label=LABELS[m]) for m in MODELS]
    x, y = MODEL_LEGEND
    # Two columns keep the legend under a; a third ran under b and read as
    # b's axis title.
    page.fig.legend(handles=handles, loc="upper left", ncol=2,
                    bbox_to_anchor=(x / page.w, 1 - y / page.h),
                    **{**LEGEND_STYLE, "columnspacing": 0.9})


# ── ASSEMBLY ─────────────────────────────────────────────────────────────────
def build_figure(effect, baseline, results, cost, heat):
    page = Page(WIDTH_MM, HEIGHT_MM, scale=SCALE, dpi=DPI, text_pt=TEXT_PT,
                letter_pt=LETTER_PT, axis_lw=AXIS_LW, panel_pos={"a": (0, 0)})
    charts = {k: page.axes(*CHART_BOX[k]) for k in CHART_BOX}

    panel_vocabulary(charts["a"], effect, baseline, results)
    panel_cost(charts["b"], cost)
    for ax in charts.values():
        style_chart(ax, TEXT_PT, AXIS_LW)
    heat_ax = panel_sections(page, heat)
    model_legend(page)

    page.panel("a", "effect of controlled vocabulary")
    page.panel_over("b", "cost", charts["b"])
    page.panel_over("c", "input sections", heat_ax)

    check_page(page.fig)
    return page.fig


def main():
    S.apply_style()
    matplotlib.rcParams.update({"font.sans-serif": FONT, "text.color": BLACK,
                                "axes.edgecolor": BLACK, "axes.labelcolor": BLACK,
                                "xtick.color": BLACK, "ytick.color": BLACK})
    fig = build_figure(*load())
    S.save_figure(fig, str(FIGURES / "figure2.png"), dpi=DPI, also_pdf=True)


if __name__ == "__main__":
    main()
