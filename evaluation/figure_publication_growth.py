"""fMRI vs dMRI publication growth in PubMed: two-panel motivation figure.

Output: results/figures/publication_growth.png (+ .svg)
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from evaluation import plot_style as S

S.apply_style()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COUNTS_CSV = os.path.join(BASE_DIR, "data", "publication_counts.csv")
OUT_PNG = os.path.join(BASE_DIR, "results", "figures", "publication_growth.png")

# ---------------------------------------------------------------------------
# Typography for this figure: the shared scale in plot_style.py, stepped up for
# print. Scaling here rather than in plot_style keeps every other figure as is.
# ---------------------------------------------------------------------------
FONT_SCALE = 1.9
FS_TITLE = S.FS_TITLE * FONT_SCALE
FS_PANEL_TITLE = S.FS_PANEL_TITLE * FONT_SCALE
FS_LABEL = S.FS_LABEL * FONT_SCALE
FS_TICK = S.FS_TICK * FONT_SCALE
FS_LEGEND = S.FS_LEGEND * FONT_SCALE
FS_PANEL_LETTER = S.FS_PANEL_LETTER * FONT_SCALE
# The inset carries the small text of the figure, so it is pinned to the main tick
# size rather than to plot_style's FS_INSET (8pt, which stayed illegible even
# scaled) - a little smaller than the axis ticks, but in the same league.
FS_INSET = S.FS_TICK * FONT_SCALE * 0.82
FS_INSET_TITLE = S.FS_TICK * FONT_SCALE * 0.95

# This figure stays in Arial. The shared stack in plot_style.py now leads with
# Helvetica Neue; dropping that one entry leaves the original Arial-first stack,
# so the fallbacks stay in step with the template.
FONT_STACK = [f for f in S.FONT_STACK if f != "Helvetica Neue"]

plt.rcParams.update({
    "font.sans-serif": FONT_STACK,
    "font.size": FS_TICK,
    "axes.titlesize": FS_TITLE,
    "figure.titlesize": FS_TITLE,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND,
    "legend.title_fontsize": FS_LEGEND,
})

# Line weights in Panel B: the two modality curves and the inset trace carry the
# figure, so they are the heaviest strokes on it.
LW_CURVE = 4.0
LW_INSET = 3.4


def label_panel(ax, title, xlabel, ylabel):
    """S.label_axes with this figure's enlarged sizes."""
    S.label_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel, panel=True)
    ax.title.set_fontsize(FS_PANEL_TITLE)
    ax.xaxis.label.set_fontsize(FS_LABEL)
    ax.yaxis.label.set_fontsize(FS_LABEL)
    return ax


def panel_letter_at(ax, letter, dx_in=0.75, dy_in=0.55):
    """Panel letter a fixed distance in inches off the top-left of the plotting box.

    S.panel_letter measures its offset in axes fractions, which shrinks with the
    box: at this type size the letter landed on the panel title. Working in inches
    keeps the same visual gap whatever the panel size. Call after the layout is
    final (tight_layout + a draw), since the square boxes settle at draw time.
    """
    pos = ax.get_position()
    return fig.text(pos.x0 - dx_in / fig.get_figwidth(),
                    pos.y1 + dy_in / fig.get_figheight(),
                    letter, fontsize=FS_PANEL_LETTER,
                    fontweight=S.PANEL_LETTER_WEIGHT, va="bottom", ha="left")

# ---------------------------------------------------------------------------
# Canonical palette — defined ONCE and reused everywhere so a database always
# has the same color across every panel/figure.
# ---------------------------------------------------------------------------
# Database identity (Panel A): one fixed color per source.
DB_COLORS = {
    "PubMed": "#1baf7a",          # teal
    "Web of Science": "#7f77dd",  # purple
}
# Modality identity (Panel B): one fixed color per imaging modality.
MODALITY_COLORS = {
    "fMRI": "#2a78d6",  # blue
    "dMRI": "#d85a30",  # coral
}

# Both panels stay square (house rule 6) but are no longer the same size: Panel B
# carries 41 years x 2 series plus an inset, Panel A carries four numbers, so B
# gets the wider column. Panel A is anchored north so the two panel letters and
# titles still line up.
#
# What matters in a manuscript is the size the text reaches AFTER the figure is
# scaled to the column width, and that depends only on the ratio of point size to
# canvas inches - so the canvas is kept as small as the labels, legends and inset
# will tolerate. 13.4 in was past that limit at this type size (letters landed on
# titles, legends on data); 15.0 is the smallest that still lays out cleanly.
fig, (axA, axB) = plt.subplots(1, 2, figsize=(15.0, 8.8),
                               gridspec_kw={"width_ratios": [1, 1.6]})
axA.set_anchor("N")
axB.set_anchor("N")   # slack from the square boxes falls below, where savefig crops it

thousands = lambda v: f"{v:,.0f}"

# ===========================================================================
# Panel A: total publications 1985-2025, PubMed vs Web of Science.
# ===========================================================================
categories = ["dMRI", "fMRI"]
pubmed_totals = [73423, 107219]     # from pubmed_counts.csv, 1985-2025
wos_totals = [53927, 117212]        # from Web of Science, 1985-2025

# Vertical layout of Panel A, written as fractions of the panel height so the
# result is independent of how large the panel ends up being.
#
# BAR_FRAC is the height of one bar. WHITE_FRAC is then whatever is left over,
# split equally between the bands above the top pair, between the two pairs and
# below the bottom pair - so the modalities read as evenly spaced instead of as two
# slabs with a hole between them, at any bar thickness.
BAR_FRAC = 0.075
WHITE_FRAC = (1.0 - len(categories) * 2 * BAR_FRAC) / (len(categories) + 1)

y_pos = np.arange(len(categories))
# Convert those fractions into data units, where one row step is 1.
ROW_PITCH = 2 * BAR_FRAC + WHITE_FRAC   # centre-to-centre, as a panel fraction
Y_RANGE = 1.0 / ROW_PITCH               # axis length, in row steps
bar_h = BAR_FRAC * Y_RANGE
Y_CENTRE = (len(categories) - 1) / 2.0

axA.barh(y_pos + bar_h / 2, pubmed_totals, height=bar_h,
         color=DB_COLORS["PubMed"], label="PubMed")
axA.barh(y_pos - bar_h / 2, wos_totals, height=bar_h,
         color=DB_COLORS["Web of Science"], label="Web of Science")

axA.set_yticks(y_pos)
axA.set_yticklabels(categories)
label_panel(axA, "Total publications by database",
            "Number of publications (1985-2025)", "Imaging modality")
# Count axis: 0 -> 120,000 exactly, ticks at 0 / 60,000 / 120,000.
S.three_ticks(axA, 0, 120_000, axis="x", fmt=thousands)
# Modality axis is categorical; equal margins above, between and below the rows.
S.flush_category_axis(axA, Y_CENTRE - Y_RANGE / 2, Y_CENTRE + Y_RANGE / 2, axis="y")
# The spine stops at the top of the fMRI pair instead of running on up through the
# empty band above it, which read as the panel being taller than the data needs.
# It still reaches down to the x-axis, so the corner stays joined.
axA.spines["left"].set_bounds(Y_CENTRE - Y_RANGE / 2, y_pos[-1] + bar_h)
S.square_axes(axA)
# No "Database" heading on either legend: at this type size the heading costs a
# whole line of height, and "PubMed" / "Web of Science" already say what they are.
axA.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.15, 0.02),
           handlelength=1.4, labelspacing=0.35, borderaxespad=0.0)
axA.grid(axis="x", linestyle="--", alpha=0.3)

# ===========================================================================
# Panel B: PubMed annual counts, fMRI vs dMRI, with milestone markers.
# ===========================================================================
years, fmri, dmri = [], [], []
with open(COUNTS_CSV) as f:
    for row in csv.DictReader(f):
        y = int(row["year"])
        if y == 2026:  # drop the partial year
            continue
        years.append(y)
        fmri.append(int(row["fmri_count"]))
        dmri.append(int(row["dmri_count"]))

years = np.array(years)
fmri = np.array(fmri)
dmri = np.array(dmri)

axB.plot(years, fmri, color=MODALITY_COLORS["fMRI"], lw=LW_CURVE, label="fMRI")
axB.plot(years, dmri, color=MODALITY_COLORS["dMRI"], lw=LW_CURVE, label="dMRI")
axB.fill_between(years, fmri, dmri, color=MODALITY_COLORS["fMRI"], alpha=0.12,
                 label="fMRI - dMRI gap")

label_panel(axB, "Annual PubMed publications by modality",
            "Year", "Publications per year")
# Year axis ends exactly on the last year; counts run 0 -> 7,000 (max = 6,471).
Y_MAX = 7_000
S.three_ticks(axB, int(years.min()), int(years.max()), axis="x")
S.three_ticks(axB, 0, Y_MAX, axis="y", fmt=thousands)
S.square_axes(axB)
axB.grid(axis="y", linestyle="--", alpha=0.3)
# "1985" is centred on the origin by default, which puts half of it underneath the
# y-axis "0". Left-aligning just that label starts it at the spine instead. (Rule 9
# does not apply here - the two axes do not share an origin value, so both numbers
# have to stay.)
axB.get_xticklabels()[0].set_ha("left")

# Bottom right: both curves are high on the right-hand side, so the area under
# them is the clearest space on the panel. Nudged right of the corner as well, so
# the handles clear the rising dMRI curve rather than starting right against it.
axB.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.05, -0.015),
           handlelength=1.4, labelspacing=0.35, borderaxespad=0.0)

# Inset: fMRI/dMRI ratio, top left. Both curves are near zero until ~2000, so the
# upper left is the empty quarter of the panel. Square too, same three-tick rule.
#
# What bounds INSET_SIDE is the fMRI curve below and right of the inset: the wider
# the box, the further right its bottom-right corner reaches and the steeper the
# curve is by then. Buying room on the other three sides is what lets the box grow
# - it is pushed as far left and as high as the inset's own tick labels and title
# allow, so the corner can move right without dropping onto the curve:
#     right edge  0.08 + 0.46 = 0.54  ->  year 2007, curve at 0.34 of the panel
#     bottom edge 0.92 - 0.46 = 0.46  ->  0.12 of clear panel above the curve,
#                                         about half of which the tick labels use
# INSET_LEFT cannot go much below 0.08 either: the inset's own y tick labels hang
# off its left edge and start crowding the main panel's y-axis. Much larger than
# 0.46 and the bottom-right corner lands on the curve however the box is placed.
INSET_SIDE = 0.46
INSET_LEFT = 0.08
mask = (years >= 2000) & (dmri > 0)
ry = years[mask]
ratio = fmri[mask] / dmri[mask]
INSET_TOP = 0.92   # as high as it goes with the inset title still inside the panel
axins = axB.inset_axes([INSET_LEFT, INSET_TOP - INSET_SIDE, INSET_SIDE, INSET_SIDE])
axins.plot(ry, ratio, color=S.GREY, lw=LW_INSET)
axins.axhline(1.0, color="#898781", lw=1.4, ls=":")  # parity
# The inset title names the plotted quantity and its x span, so the small axes
# stay uncluttered instead of carrying their own axis labels. Same family, weight
# and colour as the panel titles - it was set in plot_style's INK grey, which at
# this size read as a different typeface next to the black panel titles.
axins.set_title("fMRI / dMRI ratio by year", fontsize=FS_INSET_TITLE,
                fontfamily=plt.rcParams["font.family"], fontweight="normal",
                color="black", pad=5)
# Midpoint of 2000-2025 is 2012.5, so pin the middle tick to a whole year.
S.three_ticks(axins, int(ry.min()), int(ry.max()), axis="x", mid=2013)
S.three_ticks(axins, 0, 4, axis="y")
S.square_axes(axins)
axins.tick_params(labelsize=FS_INSET, length=3)
# Same origin overlap as the main panel ("2000" centred on the inset's own "0"), and
# at the other end a centred "2025" overhangs the box far enough to reach the fMRI
# curve outside it. Aligning the end labels inwards keeps both inside the inset.
axins.get_xticklabels()[0].set_ha("left")
axins.get_xticklabels()[-1].set_ha("right")

fig.suptitle("Growth of the fMRI and dMRI Literature, 1985-2025", y=0.98,
             fontsize=FS_TITLE)
fig.tight_layout(rect=[0, 0, 1, 0.96])

# The square boxes only take their final size at draw time, so the layout has to
# be resolved before the panel letters can be positioned relative to them.
fig.canvas.draw()
panel_letter_at(axA, "a")
panel_letter_at(axB, "b")

S.save_figure(fig, OUT_PNG)
