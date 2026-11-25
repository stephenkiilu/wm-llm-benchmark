"""Supplementary Fig. S2: the pipeline, from PubMed retrieval to structured output.

Three rows: literature retrieval; LLM extraction from article text, schema and
tract vocabulary; one JSON record per article. Geometry is in tenths of an inch.

Output: results/figures/figure_s2_pipeline.png, .svg
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import style as S  # noqa: E402
from paths import FIGURES  # noqa: E402

S.apply_style()

# ---------------------------------------------------------------------------
# Palette. Okabe-Ito accents (colour-blind safe, and the set most journals'
# figures converge on) over a cool grey scaffold. Only three accents are used,
# one per schema category, and they are reused wherever that category appears.
# ---------------------------------------------------------------------------
INK = "#1F2933"          # all body text
INK_SOFT = "#5A6875"     # secondary text
RULE = "#C6D0DA"         # card borders
ROW_BG = "#F4F7FA"       # stage row background
CARD_BG = "#FFFFFF"
CHIP = "#3D5A6C"         # stage number chips
ENGINE = "#22333F"       # the LLM box - the only dark fill

BIBLIO = "#0072B2"       # blue    - bibliographic
TECH = "#009E73"         # green   - technical
ANAT = "#D55E00"         # vermillion - anatomical
ARROW = "#8B9AA7"

# ---------------------------------------------------------------------------
# Type scale. One knob: FONT_SCALE multiplies the whole ladder. What survives
# scaling to a column width is the ratio of point size to canvas inches, so the
# canvas below is only as large as the text needs - 14.8 in at this scale.
# ---------------------------------------------------------------------------
FONT_SCALE = 1.35
FS_TITLE = 17 * FONT_SCALE
FS_STAGE_NUM = 21 * FONT_SCALE    # the 1 / 2 / 3 in the chips
FS_STAGE = 15 * FONT_SCALE        # stage name in the chip
FS_EYEBROW = 13 * FONT_SCALE      # INPUT 1 - ARTICLE TEXT
FS_CARD = 14 * FONT_SCALE         # card headers
FS_BODY = 12.5 * FONT_SCALE       # labels inside cards
FS_MOCK = 10.5 * FONT_SCALE       # mock article text and JSON

W, H = 148.0, 109.0      # canvas in tenths of an inch
fig = plt.figure(figsize=(W / 10, H / 10))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def box(x, y, w, h, fc=CARD_BG, ec=RULE, lw=1.0, r=1.3, z=2, **kw):
    """Rounded rectangle placed by its bottom-left corner."""
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
                       mutation_aspect=1, **kw)
    ax.add_patch(p)
    return p


def label(x, y, s, size=FS_BODY, color=INK, weight="normal", ha="left",
          va="center", style="normal", z=5, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va=va, style=style, zorder=z, **kw)


def eyebrow(x, y, s, color=INK_SOFT):
    """Small letterspaced label above a card, naming what it is."""
    return label(x, y, s, size=FS_EYEBROW, color=color, weight="bold",
                 va="baseline")


def arrow(x0, y0, x1, y1, color=ARROW, lw=2.6, scale=20):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                        mutation_scale=scale, linewidth=lw, color=color,
                        shrinkA=0, shrinkB=0, joinstyle="miter", zorder=4)
    ax.add_patch(a)
    return a


def spread(x0, x1, widths):
    """Left edges for `widths` laid across x0..x1 with equal gaps between them.

    Returns (positions, gap). Card widths are chosen from their content; the gap
    is the leftover, so it shrinks visibly when a card is widened too far.
    """
    gap = (x1 - x0 - sum(widths)) / (len(widths) - 1)
    xs, x = [], x0
    for w in widths:
        xs.append(x)
        x += w + gap
    return xs, gap


def stage(x, y, w, h, number, lines):
    """Numbered chip naming a stage, at the left edge of its row.

    The number and its label are centred in the chip as one block, so a tall row
    does not leave the text stranded at the top.
    """
    box(x, y, w, h, fc=CHIP, ec="none", r=1.8, z=3)
    step = FS_STAGE * 0.24
    block_h = FS_STAGE_NUM * 0.16 + 1.8 + len(lines) * step
    if block_h > h:      # a chip too short for its own contents would clip them
        raise ValueError(f"stage {number}: chip is {h:.1f} tall, needs "
                         f"{block_h:.1f} - raise that row's height")
    top = y + h / 2 + block_h / 2
    label(x + w / 2, top, number, size=FS_STAGE_NUM, color="white",
          weight="bold", ha="center", va="top", z=6)
    for i, line in enumerate(lines):
        label(x + w / 2, top - FS_STAGE_NUM * 0.16 - 1.8 - i * step, line,
              size=FS_STAGE, color="white", ha="center", va="top", z=6)


def card_header(x, y, w, h, text, color, size=FS_CARD):
    """Card with a filled header strip carrying its title."""
    strip = size * 0.30
    box(x, y, w, h)
    box(x, y + h - strip, w, strip, fc=color, ec="none", r=1.3, z=3)
    box(x, y + h - strip, w, strip / 2, fc=color, ec="none", r=0, z=3)
    label(x + w / 2, y + h - strip / 2, text, size=size, color="white",
          weight="bold", ha="center", z=6)
    return y + h - strip     # y of the strip's underside, i.e. usable top


# ===========================================================================
# Rows
# ===========================================================================
ROW_X, ROW_W = 2.0, W - 4.0
CHIP_X, CHIP_W = 3.5, 15.0
CONTENT_X = 21.5
CONTENT_R = W - 3.5
SPAN = CONTENT_R - CONTENT_X

# Row 1 is the shallowest row, so it is what sets the floor on row height: its
# chip still has to hold a number plus two lines of label. At 19.0 the number was
# clipped off the top of the chip.
R1_Y, R1_H = 81.0, 21.5     # retrieval - one row of steps
R2_Y, R2_H = 34.5, 43.0     # extraction - three inputs, so the deepest
R3_Y, R3_H = 5.0, 26.0      # output - six lines of JSON to hold
CHIP_PAD = 1.6

for y, h in ((R1_Y, R1_H), (R2_Y, R2_H), (R3_Y, R3_H)):
    box(ROW_X, y, ROW_W, h, fc=ROW_BG, ec="none", r=2.2, z=1)

for y, h, num, lines in ((R1_Y, R1_H, "1", ["Literature", "retrieval"]),
                         (R2_Y, R2_H, "2", ["LLM-based", "extraction"]),
                         (R3_Y, R3_H, "3", ["Structured", "output"])):
    stage(CHIP_X, y + CHIP_PAD, CHIP_W, h - 2 * CHIP_PAD, num, lines)

# ---------------------------------------------------------------------------
# Stage 1: query -> corpus -> full-text recovery -> structured text
# ---------------------------------------------------------------------------
CARD_H1 = 14.5
r1_bot = R1_Y + (R1_H - CARD_H1) / 2
r1_top = r1_bot + CARD_H1
(qx, cx, tx, sx), _ = spread(CONTENT_X, CONTENT_R, [24.0, 28.0, 27.0, 24.0])

# Query. The query string is set on two lines: on one it is wider than any card
# this row can spare.
top = card_header(qx, r1_bot, 24.0, CARD_H1, "PubMed query", CHIP)
ax.add_patch(Circle((qx + 3.6, top - 3.4), 1.5, facecolor="none",
                    edgecolor=BIBLIO, linewidth=1.9, zorder=5))
ax.plot([qx + 4.8, qx + 6.2], [top - 4.5, top - 5.9], color=BIBLIO, lw=1.9,
        solid_capstyle="round", zorder=5)
label(qx + 8.4, top - 2.6, '"MRI" · "DTI"', size=FS_MOCK)
label(qx + 8.4, top - 5.6, '"Dementia"', size=FS_MOCK)

# Corpus.
top = card_header(cx, r1_bot, 28.0, CARD_H1, "Corpus", CHIP)
for i, dx in enumerate((2.4, 1.2, 0.0)):
    box(cx + 2.6 + dx, top - 7.6 - dx * 0.5, 5.4, 7.2, fc=CARD_BG,
        ec=INK_SOFT, lw=0.9, r=0.5, z=4 + i)
label(cx + 12.4, top - 2.8, "n = 622", size=FS_BODY, weight="bold")
label(cx + 12.4, top - 6.2, "research articles", size=FS_MOCK, color=INK_SOFT)

# Full-text recovery.
top = card_header(tx, r1_bot, 27.0, CARD_H1, "Full-text recovery", CHIP)
for i, (nm, note, col) in enumerate([("Pubget", "open access", TECH),
                                     ("Grobid", "paywalled", BIBLIO)]):
    yy = top - 2.9 - i * 4.0
    ax.add_patch(Circle((tx + 3.4, yy), 1.4, facecolor=col, edgecolor="none",
                        zorder=5))
    label(tx + 6.2, yy, nm, size=FS_MOCK)
    label(tx + 15.0, yy, note, size=FS_MOCK, color=INK_SOFT)

# Structured text.
top = card_header(sx, r1_bot, 24.0, CARD_H1, "Structured text", CHIP)
label(sx + 12.0, top - 2.6, "JSON / XML", size=FS_BODY, weight="bold",
      ha="center")
label(sx + 12.0, top - 5.4, "abstract · methods", size=FS_MOCK,
      color=INK_SOFT, ha="center")
label(sx + 12.0, top - 8.0, "results · discussion", size=FS_MOCK,
      color=INK_SOFT, ha="center")

for x0, w0, x1 in ((qx, 24.0, cx), (cx, 28.0, tx), (tx, 27.0, sx)):
    arrow(x0 + w0 + 1.2, r1_bot + CARD_H1 / 2, x1 - 1.2, r1_bot + CARD_H1 / 2)

# ---------------------------------------------------------------------------
# Stage 2: three inputs into one LLM call
# ---------------------------------------------------------------------------
# The three stage-2 columns share a top and a bottom edge, so the row reads as
# one band of inputs rather than three floating boxes.
BAND_BOT = R2_Y + 1.2
BAND_TOP = R2_Y + R2_H - 5.2      # the eyebrow labels live above this
# The right column is the widest: it carries the longest lines in the figure (the
# schema's entity lists and the look-up table's field names side by side).
AW, LW_, RW = 35.0, 27.0, 49.0
(ax1, lx, sx2), _ = spread(CONTENT_X, CONTENT_R, [AW, LW_, RW])

# Input 1 - article text, drawn as a page of a paper: a dog-eared sheet with a
# title, a shaded abstract and a two-column body. Body copy is greeked (grey
# bars) - only the section headings are words, since those are what stage 1
# recovers and what the model is handed.
ay1, ah1 = BAND_BOT, BAND_TOP - BAND_BOT
eyebrow(ax1, ay1 + ah1 + 2.0, "INPUT 1 · ARTICLE TEXT")
PAPER_RULE = "#9AA8B4"   # the page edge - darker than a card, so it reads as a sheet
GREEK = "#D5DDE4"        # greeked body copy
FOLD = 3.4
px0, px1, pt = ax1, ax1 + AW, ay1 + ah1
ax.add_patch(Polygon([(px0 + 0.6, ay1 - 0.6), (px1 + 0.6, ay1 - 0.6),
                      (px1 + 0.6, pt - FOLD - 0.6), (px1 - FOLD + 0.6, pt - 0.6),
                      (px0 + 0.6, pt - 0.6)],
                     closed=True, facecolor="#DFE5EB", edgecolor="none", zorder=1.5))
ax.add_patch(Polygon([(px0, ay1), (px1, ay1), (px1, pt - FOLD),
                      (px1 - FOLD, pt), (px0, pt)], closed=True,
                     facecolor=CARD_BG, edgecolor=PAPER_RULE, linewidth=1.0,
                     joinstyle="miter", zorder=2))
ax.add_patch(Polygon([(px1 - FOLD, pt), (px1 - FOLD, pt - FOLD), (px1, pt - FOLD)],
                     closed=True, facecolor="#E8EDF2", edgecolor=PAPER_RULE,
                     linewidth=1.0, joinstyle="miter", zorder=3))


def greek(x, y, w, lines, step=1.55, last=0.6):
    """Grey bars standing in for `lines` lines of text, the last one short."""
    for i in range(lines):
        lw_ = w * (last if i == lines - 1 else 1.0)
        box(x, y - i * step - 0.35, lw_, 0.7, fc=GREEK, ec="none", r=0.3, z=4)


m = 2.6                                   # page margin
cx0, cw = px0 + m, AW - 2 * m             # text block
label(cx0, pt - 3.4, "White matter changes in mild", size=FS_MOCK,
      weight="bold", va="baseline")
label(cx0, pt - 5.8, "cognitive impairment and AD", size=FS_MOCK,
      weight="bold", va="baseline")
greek(cx0, pt - 7.7, cw * 0.55, 1)        # author line
ax.plot([cx0, cx0 + cw], [pt - 9.2, pt - 9.2], color=RULE, lw=0.8, zorder=4)

# Abstract, on a tint as journals set it.
box(cx0 - 0.8, pt - 17.6, cw + 1.6, 7.4, fc=ROW_BG, ec="none", r=0.6, z=3)
label(cx0, pt - 12.0, "Abstract", size=FS_MOCK, weight="bold", va="baseline")
greek(cx0, pt - 13.9, cw, 3)

# Two-column body.
gut = 1.6
colw = (cw - gut) / 2
c2 = cx0 + colw + gut
for x0, top_y, name, n in ((cx0, pt - 20.6, "Methods", 3),
                           (cx0, pt - 28.6, "Results", 3),
                           (c2, pt - 28.6, "Discussion", 3)):
    label(x0, top_y, name, size=FS_MOCK, weight="bold", va="baseline")
    greek(x0, top_y - 1.9, colw, n)
# A figure in the right column - three bars on an axis, so it reads as a plot.
# Grey, not the category hues: colour in this figure means a schema category.
fig_y0, fig_y1 = pt - 24.2, pt - 18.7
box(c2, fig_y0, colw, fig_y1 - fig_y0, fc=ROW_BG, ec=RULE, lw=0.7, r=0.4, z=4)
bw, base = colw / 5, fig_y0 + 0.9
for j, hh in enumerate((0.85, 0.5, 0.65)):
    box(c2 + bw * (0.8 + 1.3 * j), base, bw, (fig_y1 - base - 0.8) * hh,
        fc=ARROW, ec="none", r=0.1, z=5)
ax.plot([c2 + bw * 0.5, c2 + colw - bw * 0.5], [base, base], color=INK_SOFT,
        lw=0.9, zorder=6)
greek(c2, fig_y0 - 1.2, colw * 0.8, 1)    # caption

# The LLM.
lh = 19.0
ly = (BAND_BOT + BAND_TOP) / 2 - lh / 2
box(lx, ly, LW_, lh, fc=ENGINE, ec="none", r=1.8, z=3)
# A chip glyph, drawn rather than imported so the figure has no asset dependency.
gx, gy = lx + LW_ / 2, ly + lh - 5.0
box(gx - 3.6, gy - 2.7, 7.2, 5.4, fc="none", ec="white", lw=1.9, r=0.6, z=6)
for dx in (-1.8, 0.0, 1.8):
    ax.plot([gx + dx, gx + dx], [gy + 2.7, gy + 4.2], color="white", lw=1.7,
            solid_capstyle="round", zorder=6)
    ax.plot([gx + dx, gx + dx], [gy - 2.7, gy - 4.2], color="white", lw=1.7,
            solid_capstyle="round", zorder=6)
for dy in (-1.35, 1.35):
    ax.plot([gx - 3.6, gx - 5.1], [gy + dy, gy + dy], color="white", lw=1.7,
            solid_capstyle="round", zorder=6)
    ax.plot([gx + 3.6, gx + 5.1], [gy + dy, gy + dy], color="white", lw=1.7,
            solid_capstyle="round", zorder=6)
ax.add_patch(Circle((gx, gy), 1.4, facecolor=TECH, edgecolor="none", zorder=7))

label(gx, ly + 7.4, "LLM", size=FS_CARD, color="white",
      weight="bold", ha="center", z=6)
ax.plot([lx + 2.8, lx + LW_ - 2.8], [ly + 5.9, ly + 5.9], color="#4A5B67",
        lw=1.0, zorder=6)
label(gx, ly + 3.9, '"You are an expert in', size=FS_MOCK, color="#D3DBE2",
      ha="center", style="italic", z=6)
label(gx, ly + 1.6, 'neuroimaging extraction …"', size=FS_MOCK,
      color="#D3DBE2", ha="center", style="italic", z=6)

# Input 2 - schema. Categories run as rows, not columns: at this type size three
# columns of entity names do not fit the width, and a row can use all of it.
# 16.0 for the schema leaves the look-up table below it deep enough for three
# tract pills; any taller and the third pill drops out of its card.
s2_h = 16.0
s2_y = BAND_TOP - s2_h
eyebrow(sx2, s2_y + s2_h + 2.0, "INPUT 2 · SCHEMA")
top = card_header(sx2, s2_y, RW, s2_h, "14 target variables · 3 categories",
                  CHIP, size=FS_MOCK)
CATEGORIES = [
    ("Bibliographic", BIBLIO, "PMID · study type · …"),
    ("Technical", TECH, "imaging method · measures · …"),
    ("Anatomical", ANAT, "WM tracts · effects · …"),
]
for i, (name, col, items) in enumerate(CATEGORIES):
    yy = top - 2.9 - i * 3.9
    label(sx2 + 2.6, yy, name, size=FS_MOCK, color=col, weight="bold")
    label(sx2 + 18.5, yy, items, size=FS_MOCK, color=INK_SOFT)

# Input 3 - the controlled vocabulary, shown as a few of its tract names. Pills
# are sized to their text and flow left to right, wrapping at the card edge; the
# vocabulary's size follows the last pill as plain grey text.
s3_y = BAND_BOT
s3_h = s2_y - 5.6 - s3_y
eyebrow(sx2, s3_y + s3_h + 2.0, "INPUT 3 · CONTROLLED VOCABULARY")
box(sx2, s3_y, RW, s3_h)
TRACTS = ["Cingulum bundle", "Corticospinal tract",
          "Superior longitudinal fasciculus"]
PILL_PAD, PILL_GAP, PILL_H, ROW_STEP = 1.8, 1.4, 3.4, 4.6
renderer = fig.canvas.get_renderer()
to_data = ax.transData.inverted()


def text_w(s, **kw):
    t = ax.text(0, 0, s, fontsize=FS_MOCK, **kw)
    w = t.get_window_extent(renderer).transformed(to_data).width
    t.remove()
    return w


# Lay the rows out first, then centre the block vertically in the card.
x_lo, x_hi = sx2 + 2.6, sx2 + RW - 2.6
rows, x = [[]], x_lo
for tract in TRACTS:
    w = text_w(tract) + 2 * PILL_PAD
    if x + w > x_hi and rows[-1]:
        rows.append([])
        x = x_lo
    rows[-1].append((x, w, tract))
    x += w + PILL_GAP
count = "… 122 tracts"
if x + text_w(count) > x_hi:
    rows.append([])
    x = x_lo
rows[-1].append((x, None, count))

row0 = s3_y + s3_h / 2 + (len(rows) - 1) * ROW_STEP / 2
for i, row in enumerate(rows):
    ty = row0 - i * ROW_STEP
    for x, w, text in row:
        if w is None:
            label(x, ty, text, size=FS_MOCK, color=INK_SOFT)
            continue
        box(x, ty - PILL_H / 2, w, PILL_H, fc="#FDF0E7", ec=ANAT, lw=1.0,
            r=1.7, z=3)
        label(x + PILL_PAD, ty, text, size=FS_MOCK, color=ANAT, z=5)

arrow(ax1 + AW + 1.4, ly + lh / 2, lx - 1.4, ly + lh / 2)
arrow(sx2 - 1.4, ly + lh / 2, lx + LW_ + 1.4, ly + lh / 2)

# ---------------------------------------------------------------------------
# Stage 3: one JSON record per article
# ---------------------------------------------------------------------------
ox, ow = CONTENT_X, SPAN
oy, oh = R3_Y + 2.6, R3_H - 5.6
eyebrow(ox, oy + oh + 1.8, "ONE RECORD PER ARTICLE")
box(ox, oy, ow, oh)

arrow(lx + LW_ / 2, ly - 1.4, lx + LW_ / 2, oy + oh + 1.4)

JSON_LINES = [
    [("{", TECH, "bold")],
    [('  "pmid"', BIBLIO, "normal"), (': "16005548",  ', INK, "normal"),
     ('"study_type"', BIBLIO, "normal"), (': ["original"],  ', INK, "normal"),
     ('"subjects"', BIBLIO, "normal"), (': ["human"],', INK, "normal")],
    [('  "diseases"', BIBLIO, "normal"),
     (': ["Alzheimer\'s disease", "mild cognitive impairment"],', INK,
      "normal")],
    [('  "imaging_modality"', TECH, "normal"), (': ["DTI"],  ', INK, "normal"),
     ('"software"', TECH, "normal"), (': ["SPM99"],  ', INK, "normal"),
     ('"diffusion_measures"', TECH, "normal"), (': ["FA", "AD"],', INK,
                                                "normal")],
    [('  "wm_tracts"', ANAT, "normal"),
     (': ["superior longitudinal fasciculus", "cingulum bundle", '
      '"corticospinal tract"]', INK, "normal")],
    [("}", TECH, "bold")],
]
yy = oy + oh - 3.2
for line in JSON_LINES:
    xx = ox + 2.8
    for text, col, wt in line:
        t = label(xx, yy, text, size=FS_MOCK, color=col, weight=wt)
        # Chain the runs by measuring each one, so the coloured keys and the
        # black values sit on one continuous line.
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(ax.transData.inverted())
        xx = bb.x1
    yy -= (oh - 5.6) / (len(JSON_LINES) - 1)

fig.suptitle("From literature to structured white-matter records",
             fontsize=FS_TITLE, y=0.988)

out = S.save_figure(fig, str(FIGURES / "figure_s2_pipeline.png"))
