"""Figure 1: a growing dMRI literature with inconsistent tract names.

a  Unique brain fMRI and dMRI papers in PubMed and Web of Science, 1985-2025.
b  Brain fMRI and dMRI papers per year in PubMed.
c  fMRI / dMRI publication ratio in PubMed, 2000-2025.
d  SLF I-III and the arcuate fasciculus (TractSeg renders, left and right
   hemispheres): the bundles some authors call SLF I-III + AF, others SLF I-IV.

Inputs: data/literature/*.csv, data/brain_renders/*.png
Output: results/figures/figure1.png (600 dpi), .svg, .pdf
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import style as S  # noqa: E402
from page import Page, align_end_ticks, check_page, legend_style, style_chart  # noqa: E402
from paths import DATA, FIGURES  # noqa: E402

COUNTS = DATA / "literature"
BRAINS = DATA / "brain_renders"

# ── SETTINGS ─────────────────────────────────────────────────────────────────
# Layout is written on a 121 mm grid and printed at SCALE of it, so the whole
# figure can be resized in one place. Type is set separately, in printed points,
# so shrinking the figure does not shrink the text by the same amount.
WIDTH_MM = 121
HEIGHT_MM = 88
SCALE = 0.8          # printed size: 96.8 x 70.4 mm (3.81 x 2.77 in)
DPI = 600
TEXT_PT = 8
LETTER_PT = 9
AXIS_LW = 0.6          # spines and ticks, pt
DATA_LW = 1.0          # chart lines, pt

START_YEAR, END_YEAR = 1985, 2025
RATIO_START, RATIO_MID = 2000, 2013   # printed midpoint; the true one is 2012.5
COUNT_MAX = 100_000
ANNUAL_MAX = 7_000
RATIO_MAX = 5

COLORS = {
    "PubMed": "#008C86",
    "Web of Science": "#8061A5",
    "fMRI": "#176BB7",
    "dMRI": "#D65F18",
    "ratio": "#45484B",
    # Sampled from the TractSeg renders, so the key matches the bundles exactly.
    "slf1": "#0072B2",
    "slf2": "#56B4E9",
    "slf3": "#009E73",
    "af": "#D55E00",
}

BRAIN_FILES = {"L": "599469_left.png",
               "R": "856766_right.png"}   # hemisphere, spelled out in the caption

# Panel d legend. The naming conflict (AF as its own tract, or as SLF IV) is
# stated in the caption, not the key.
LEGEND = [("slf1", "SLF I"), ("slf2", "SLF II"), ("slf3", "SLF III"), ("af", "AF")]

# ── LAYOUT (mm from the top-left corner of the page) ────────────────────────
# Two columns, two rows:  a b / c d.  Charts are wider than tall so each fills
# its column; a square box here would leave a third of every column empty.
PANEL_POS = {"a": (0, 0), "b": (61, 0), "c": (0, 46), "d": (61, 46)}  # letter x, y
CHART_BOX = {"a": (10.5, 6.5, 45.5, 24),     # x, y, width, height
             "b": (73, 6.5, 46.5, 24),
             "c": (10.5, 52.5, 45.5, 24)}
CHART_LEGEND_TOP = 39.5      # one-line legends under a and b
# Panel d fills the bottom-right cell: two brains side by side, legend beneath.
BRAIN_TOP = 52
BRAIN_WIDTH = 29
BRAIN_LEFT = {"L": 62, "R": 92}
D_CENTRE = 91.5


# ── INPUTS ───────────────────────────────────────────────────────────────────
def read_csv(path) -> list[dict]:
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def load_publication_counts():
    """Annual brain papers per modality, plus the unique 1985-2025 totals per database."""
    rows = [r for r in read_csv(COUNTS / "annual_counts_brain.csv")
            if START_YEAR <= int(r["year"]) <= END_YEAR]
    years = np.array([int(r["year"]) for r in rows])
    if not np.array_equal(years, np.arange(START_YEAR, END_YEAR + 1)):
        raise ValueError("annual_counts_brain.csv does not cover every year exactly once.")

    annual = {m: np.array([int(r[f"{m.lower()}_brain"]) for r in rows])
              for m in ["fMRI", "dMRI"]}
    rows = {r["measure"]: int(r["unique_papers_1985_2025"])
            for r in read_csv(COUNTS / "totals_unique.csv")}
    pubmed = {m: rows[f"{m.lower()}_brain"] for m in ["fMRI", "dMRI"]}
    # A paper can carry two [dp] years, so the yearly counts may only add up to more.
    if any(pubmed[m] > annual[m].sum() for m in pubmed):
        raise ValueError("A unique total exceeds the sum of its annual counts.")
    wos = {r["modality"]: int(r["count"]) for r in read_csv(COUNTS / "web_of_science_counts.csv")
           if r["measure"].startswith("total")}
    totals = {"PubMed": [pubmed["fMRI"], pubmed["dMRI"]],
              "Web of Science": [wos["fmri"], wos["dmri"]]}
    if max(max(t) for t in totals.values()) > COUNT_MAX:
        raise ValueError("A total runs past the last tick of panel a.")
    return years, annual, totals


def load_brain(path):
    """The render without its baked-in key, cropped to the glass brain.

    The key is a strip of swatches under the brain, separated from it by white
    rows; everything from the last white gap down is dropped. The remaining
    image is cropped to its non-white bounding box.
    """
    image = np.asarray(Image.open(path).convert("RGB"))
    ink = (image < 245).any(axis=2)
    rows = ink.any(axis=1)
    # Walk up from the bottom: past the key, then past the white gap above it.
    bottom = len(rows) - 1
    while not rows[bottom]:
        bottom -= 1
    while rows[bottom]:
        bottom -= 1
    while not rows[bottom]:
        bottom -= 1
    if bottom > 0.95 * len(rows) or bottom < 0.7 * len(rows):
        raise ValueError(f"{path.name}: could not find the key strip to crop.")
    ink[bottom + 1:] = False
    ys, xs = np.nonzero(ink)
    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


# ── LEGENDS ──────────────────────────────────────────────────────────────────
LEGEND_STYLE = legend_style(TEXT_PT)


def swatches(entries):
    return [Patch(color=COLORS[key], label=label) for key, label in entries]


def legend_under_chart(page, ax, handles, **style):
    """One-line legend centred under a chart."""
    box = ax.get_position()
    return ax.legend(handles=handles, loc="upper center", ncol=len(handles),
                     bbox_to_anchor=((box.x0 + box.x1) / 2, 1 - CHART_LEGEND_TOP / page.h),
                     bbox_transform=page.fig.transFigure,
                     **{**LEGEND_STYLE, "columnspacing": 1.0, **style})


# ── PANELS a-c ───────────────────────────────────────────────────────────────
def panel_publications_by_database(page, ax, totals):
    """Paired bars per modality, one colour per database."""
    height, offset = 0.3, 0.16
    for database, shift in [("PubMed", offset), ("Web of Science", -offset)]:
        ax.barh(np.array([1, 0]) + shift, totals[database], height=height,
                color=COLORS[database])
    legend_under_chart(page, ax, swatches([("PubMed", "PubMed"),
                                           ("Web of Science", "Web of Science")]))
    ax.set(yticks=[0, 1], yticklabels=["dMRI", "fMRI"], ylim=(-0.55, 1.55),
           xlim=(0, COUNT_MAX), xlabel="Publications")
    ticks = [0, COUNT_MAX // 2, COUNT_MAX]
    ax.set_xticks(ticks, [f"{t // 1000}k" if t else "0" for t in ticks])
    ax.tick_params(axis="y", length=0)


def panel_annual_publications(page, ax, years, annual):
    """Annual PubMed counts, one line per modality."""
    for modality, counts in annual.items():
        ax.plot(years, counts, color=COLORS[modality], lw=DATA_LW)
    lines = [Line2D([], [], color=COLORS[m], lw=DATA_LW, label=m) for m in annual]
    legend_under_chart(page, ax, lines, handlelength=1.6, columnspacing=1.2)
    ax.set(xlim=(START_YEAR, END_YEAR), ylim=(0, ANNUAL_MAX),
           xlabel="Year", ylabel="Publications")
    ax.set_xticks([START_YEAR, (START_YEAR + END_YEAR) // 2, END_YEAR])
    ax.set_yticks([0, ANNUAL_MAX / 2, ANNUAL_MAX],
                  ["0", f"{ANNUAL_MAX / 2000:g}k", f"{ANNUAL_MAX // 1000}k"])


def panel_ratio(ax, years, annual):
    """fMRI publications per dMRI publication, from 2000 onwards."""
    covered = (years >= RATIO_START) & (annual["dMRI"] > 0)
    ratio = annual["fMRI"][covered] / annual["dMRI"][covered]
    ax.plot(years[covered], ratio, color=COLORS["ratio"], lw=DATA_LW)
    ax.axhline(1, color=S.MUTED, ls=":", lw=0.8)   # equal counts; named in the caption
    ax.set(xlim=(RATIO_START, END_YEAR), ylim=(0, RATIO_MAX), xlabel="Year", ylabel="Ratio")
    ax.set_xticks([RATIO_START, RATIO_MID, END_YEAR])
    ax.set_yticks([0, RATIO_MAX / 2, RATIO_MAX], ["0", f"{RATIO_MAX / 2:g}", f"{RATIO_MAX:g}"])
    return ratio


# ── PANEL d ──────────────────────────────────────────────────────────────────
def panel_brains(page, brains):
    """Both hemispheres at one scale, each marked L or R in its corner."""
    mm_per_px = BRAIN_WIDTH / max(b.shape[1] for b in brains.values())
    placed = {}
    for name, image in brains.items():
        w = image.shape[1] * mm_per_px
        h = image.shape[0] * mm_per_px
        x = BRAIN_LEFT[name] + (BRAIN_WIDTH - w) / 2
        ax = page.axes(x, BRAIN_TOP, w, h)
        ax.set_gid(f"panel-d-{name.lower()}")
        ax.imshow(image, interpolation="lanczos")
        ax.axis("off")
        # Hemisphere letter in the render's empty top-left corner: no row of its own.
        page.text(x + 0.3, BRAIN_TOP + 0.3, name, ha="left")
        placed[name] = (x, w, h)
    return placed


def panel_legend(page, brain_height):
    """Square swatches in two columns, centred under both brains."""
    page.fig.legend(handles=swatches(LEGEND), loc="upper center", ncol=2,
                    bbox_to_anchor=(D_CENTRE / page.w,
                                    1 - (BRAIN_TOP + brain_height + 2) / page.h),
                    **LEGEND_STYLE)


# ── ASSEMBLY ─────────────────────────────────────────────────────────────────
def build_figure(years, annual, totals, brains):
    page = Page(WIDTH_MM, HEIGHT_MM, scale=SCALE, dpi=DPI, text_pt=TEXT_PT,
                letter_pt=LETTER_PT, axis_lw=AXIS_LW, panel_pos=PANEL_POS)
    fig = page.fig

    titles = {"a": "Total publications",
              "b": "Publications per year",
              "c": "fMRI/dMRI ratio"}
    charts = {}
    for letter in "abc":
        page.panel(letter, titles[letter])
        charts[letter] = page.axes(*CHART_BOX[letter])

    panel_publications_by_database(page, charts["a"], totals)
    panel_annual_publications(page, charts["b"], years, annual)
    ratio = panel_ratio(charts["c"], years, annual)
    for ax in charts.values():
        style_chart(ax, TEXT_PT, AXIS_LW)
        align_end_ticks(ax)

    page.panel("d", "Inconsistent SLF and AF naming")
    placed = panel_brains(page, brains)
    brain_height = max(h for _, _, h in placed.values())
    if BRAIN_TOP + brain_height > HEIGHT_MM:
        raise AssertionError(f"Brains end at {BRAIN_TOP + brain_height:.1f} mm, off the page.")
    panel_legend(page, brain_height)

    check_page(fig)
    return fig, ratio


def main():
    S.apply_style()
    years, annual, totals = load_publication_counts()
    brains = {name: load_brain(BRAINS / file) for name, file in BRAIN_FILES.items()}
    fig, _ = build_figure(years, annual, totals, brains)
    S.save_figure(fig, str(FIGURES / "figure1.png"), dpi=DPI, also_pdf=True)


if __name__ == "__main__":
    main()
