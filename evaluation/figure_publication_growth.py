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
FS_INSET = S.FS_TICK * FONT_SCALE * 0.82
FS_INSET_TITLE = S.FS_TICK * FONT_SCALE * 0.95

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

# Database identity (Panel A): one fixed color per source.
DB_COLORS = {
    "PubMed": "#1baf7a",          # teal
    "Web of Science": "#7f77dd",  # purple
}

MODALITY_COLORS = {
    "fMRI": "#2a78d6",  # blue
    "dMRI": "#d85a30",  # coral
}

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

BAR_FRAC = 0.075
WHITE_FRAC = (1.0 - len(categories) * 2 * BAR_FRAC) / (len(categories) + 1)

y_pos = np.arange(len(categories))
ROW_PITCH = 2 * BAR_FRAC + WHITE_FRAC 
Y_RANGE = 1.0 / ROW_PITCH            
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
S.three_ticks(axA, 0, 120_000, axis="x", fmt=thousands)
S.flush_category_axis(axA, Y_CENTRE - Y_RANGE / 2, Y_CENTRE + Y_RANGE / 2, axis="y")
axA.spines["left"].set_bounds(Y_CENTRE - Y_RANGE / 2, y_pos[-1] + bar_h)
S.square_axes(axA)
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
Y_MAX = 7_000
S.three_ticks(axB, int(years.min()), int(years.max()), axis="x")
S.three_ticks(axB, 0, Y_MAX, axis="y", fmt=thousands)
S.square_axes(axB)
axB.grid(axis="y", linestyle="--", alpha=0.3)
axB.get_xticklabels()[0].set_ha("left")

axB.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.05, -0.015),
           handlelength=1.4, labelspacing=0.35, borderaxespad=0.0)

INSET_SIDE = 0.46
INSET_LEFT = 0.08
mask = (years >= 2000) & (dmri > 0)
ry = years[mask]
ratio = fmri[mask] / dmri[mask]
INSET_TOP = 0.92 
axins = axB.inset_axes([INSET_LEFT, INSET_TOP - INSET_SIDE, INSET_SIDE, INSET_SIDE])
axins.plot(ry, ratio, color=S.GREY, lw=LW_INSET)
axins.axhline(1.0, color="#898781", lw=1.4, ls=":")  # parity

axins.set_title("fMRI / dMRI ratio by year", fontsize=FS_INSET_TITLE,
                fontfamily=plt.rcParams["font.family"], fontweight="normal",
                color="black", pad=5)
S.three_ticks(axins, int(ry.min()), int(ry.max()), axis="x", mid=2013)
S.three_ticks(axins, 0, 4, axis="y")
S.square_axes(axins)
axins.tick_params(labelsize=FS_INSET, length=3)

axins.get_xticklabels()[0].set_ha("left")
axins.get_xticklabels()[-1].set_ha("right")

fig.suptitle("Growth of the fMRI and dMRI Literature, 1985-2025", y=0.98,
             fontsize=FS_TITLE)
fig.tight_layout(rect=[0, 0, 1, 0.96])

fig.canvas.draw()
panel_letter_at(axA, "a")
panel_letter_at(axB, "b")

S.save_figure(fig, OUT_PNG)
