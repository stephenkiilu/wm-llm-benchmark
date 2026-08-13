"""Shared figure template for every publication figure in this project.

House rules:
    1. every figure carries a title
    2. both axes are labelled
    3. one font family and one size scale across all figures
    4. the title is the ONLY bold text
    5. every figure is written as PNG
    6. the plotting box is square - x and y axis lengths are equal
    7. numeric axes show exactly three ticks: min, midpoint, max
    8. numeric axes stop exactly at the maximum - no overhang past the last tick
    9. a shared origin value is printed once, not on both axes

Rules 1-5 come from ``apply_style`` / ``label_axes`` / ``save_figure``;
rules 6-9 from ``square_axes`` / ``three_ticks`` / ``share_origin_label``.

Colour convention:
    GPT-5 -> blue
    GPT-4 -> red

Import from any evaluation/plotting script:
    from plot_style import MODEL_COLORS, model_color, apply_style, save_figure
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

#MODEL COLOURS
GPT5_BLUE = "#5B8FF9"
GPT4_RED = "#EF553B"

MODEL_COLORS = {
    "GPT-5": GPT5_BLUE,
    "GPT-4": GPT4_RED,
}

MODEL_MARKERS = {
    "GPT-4": "s",   # square
    "GPT-5": "o",   # circle
}

INK = "#333333"
GREY = "#555555"
MUTED = "#777777"
GRID = "#DDDDDD"
HELVETICA_TTC = "/System/Library/Fonts/HelveticaNeue.ttc"
_TTC_FACES = {0: "Regular", 1: "Bold", 2: "Italic", 3: "BoldItalic"}

def _register_helvetica_neue(ttc: str = HELVETICA_TTC) -> bool:
    """Register real Helvetica Neue weights with matplotlib. True if available."""
    if not os.path.exists(ttc):
        return False
    try:
        from fontTools.ttLib import TTCollection

        cache = os.path.join(mpl.get_cachedir(), "helvetica-neue-faces")
        os.makedirs(cache, exist_ok=True)
        faces = {i: os.path.join(cache, f"HelveticaNeue-{n}.ttf")
                 for i, n in _TTC_FACES.items()}
        missing = {i: p for i, p in faces.items() if not os.path.exists(p)}
        if missing:
            collection = TTCollection(ttc)
            for index, path in missing.items():
                collection.fonts[index].save(path)
        for path in faces.values():
            mpl.font_manager.fontManager.addfont(path)
        return True
    except Exception:
        # fontTools missing, collection laid out differently, unwritable cache:
        # any of these just means falling back to Arial.
        return False


HELVETICA_NEUE = _register_helvetica_neue()

FONT_STACK = ["Helvetica Neue", "Arial", "Helvetica", "Liberation Sans",
              "DejaVu Sans"]
if not HELVETICA_NEUE:
    FONT_STACK = FONT_STACK[1:]

FS_TITLE = 15         # figure / axes title  (the only bold text)
FS_LABEL = 13         # axis labels
FS_PANEL_TITLE = 12   # subplot titles in multi-panel figures
FS_TICK = 11          # tick labels
FS_LEGEND = 11        # legend entries
FS_VALUE = 10         # in-bar value labels
FS_ANNOT = 10         # significance stars, callouts
FS_NOTE = 9.5         # footnotes (p-value key, etc.)
FS_INSET = 8          # inset axes: ticks, title, in-plot milestone labels
PANEL_LETTER_WEIGHT = "bold"
FS_PANEL_LETTER = 14


def model_color(label: str) -> str:
    """Return the canonical color for a model label.

    Matches on whether "5" or "4" appears in the label so variants like
    "GPT-5 Full", "GPT5", "gpt-4o-mini" all resolve correctly.
    """
    if label in MODEL_COLORS:
        return MODEL_COLORS[label]
    text = label.lower()
    if "5" in text:
        return GPT5_BLUE
    if "4" in text:
        return GPT4_RED
    raise KeyError(f"No model color mapping for label: {label!r}")


def apply_style() -> None:
    """Install the house rcParams. Call once, before creating any figure."""
    plt.rcParams.update({
        # -- font: one family, one size scale 
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": FS_TICK,
        "font.weight": "normal",
        "mathtext.default": "regular",

        # -- titles: the only bold text 
        "axes.titlesize": FS_TITLE,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
        "figure.titlesize": FS_TITLE,
        "figure.titleweight": "bold",

        # -- everything else stays regular weight -
        "axes.labelsize": FS_LABEL,
        "axes.labelweight": "normal",
        "axes.labelcolor": "black",
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "legend.title_fontsize": FS_LEGEND,
        "figure.labelsize": FS_LABEL,
        "figure.labelweight": "normal",

        # -- frame / grid 
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # -- output: crisp raster + editable vector 
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",   
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def label_axes(ax, title=None, xlabel=None, ylabel=None, panel=False):
    """Set title + both axis labels at the house sizes.

    ``panel=True`` uses the smaller, non-bold subplot-title size, reserving bold
    for the figure-level title.
    """
    if title is not None:
        if panel:
            ax.set_title(title, fontsize=FS_PANEL_TITLE, fontweight="normal")
        else:
            ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    return ax


def square_axes(ax):
    """Rule 6 - make the plotting box square, so the x and y axes are the same
    physical length regardless of the data ranges on them.
    """
    ax.set_box_aspect(1)
    return ax


def _tick_label(v):
    """Format a tick value without trailing zeros: 0, 0.5, 1 (not 0.0, 0.5, 1.0)."""
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def three_ticks(ax, lo, hi, axis="both", mid=None, fmt=None):
    """Rules 7 + 8 - show exactly [lo, midpoint, hi] and clamp the limit to
    (lo, hi) so the axis ends precisely at the maximum.

    Only for numeric axes; leave categorical axes (model names, section names)
    to their own tick labels.

    mid : override the midpoint. Use when the true midpoint is not a sensible
          value to print - e.g. a 2000-2025 year axis, whose midpoint is 2012.5.
    fmt : callable(value) -> str for the tick labels, e.g. thousands separators.
    """
    ticks = [lo, (lo + hi) / 2.0 if mid is None else mid, hi]
    labels = [fmt(t) if fmt else _tick_label(t) for t in ticks]
    if axis in ("x", "both"):
        ax.set_xlim(lo, hi)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    if axis in ("y", "both"):
        ax.set_ylim(lo, hi)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
    return ax


def share_origin_label(ax, keep="y"):
    """Rule 9 - when both axes start at the same value, print that value once.

    Two "0"s meeting at the corner reads as a duplicate, so blank one of them.
    keep="y" drops the x-axis first label; keep="x" drops the y-axis first label.
    """
    if keep == "y":
        labels = [t.get_text() for t in ax.get_xticklabels()]
        ax.set_xticklabels([""] + labels[1:])
    else:
        labels = [t.get_text() for t in ax.get_yticklabels()]
        ax.set_yticklabels([""] + labels[1:])
    return ax


def flush_category_axis(ax, first_edge, last_edge, axis="x"):
    """Rule 8 for a categorical bar axis: the outer bar edges sit exactly on the
    spines, with no empty margin before the first or after the last bar.
    """
    if axis == "x":
        ax.set_xlim(first_edge, last_edge)
    else:
        ax.set_ylim(first_edge, last_edge)
    return ax


def panel_letter(ax, letter, x=-0.08, y=1.12):
    """Stamp a panel identifier (A, B, C ...) in axes coordinates."""
    return ax.text(x, y, letter, transform=ax.transAxes,
                   fontsize=FS_PANEL_LETTER, fontweight=PANEL_LETTER_WEIGHT,
                   va="top", ha="left")


# *** is 0 of N_BOOT=1000 two-sided resamples, i.e. below the 2/1000 resolution.
SIG_LEVELS = (("*", "q<0.05"), ("**", "q<0.01"), ("***", "q<0.002"))


def sig_key(symbols, suffix="(FDR-adjusted)"):
    """Key covering only the marks a figure actually prints; None if it prints none."""
    present = {s for s in symbols if isinstance(s, str) and set(s) == {"*"}}
    parts = [f"{m} {t}" for m, t in SIG_LEVELS if m in present]
    return "    ".join(parts + [suffix]) if parts else None


def sig_note(ax, text, y=-0.14):
    """Place the significance-key footnote below an axes, consistently."""
    return ax.annotate(text, xy=(0.0, y), xycoords="axes fraction",
                       ha="left", va="top",
                       fontsize=FS_NOTE, style="italic", color=MUTED)


def save_figure(fig, out_path, dpi=300, also_pdf=False, close=True):
    """Save ``fig`` as PNG *and* SVG (house rule 5). Returns written paths.

    ``out_path`` may carry any extension (or none); it is used as the stem.
    """
    stem = os.path.splitext(out_path)[0]
    parent = os.path.dirname(stem)
    if parent:
        os.makedirs(parent, exist_ok=True)

    written = []
    for ext in (".png", ".svg") + ((".pdf",) if also_pdf else ()):
        path = stem + ext
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)

    if close:
        plt.close(fig)
    print("saved " + " + ".join(os.path.basename(p) for p in written)
          + f"  ->  {parent or '.'}")
    return written

# Applying the style at import time means any script that pulls MODEL_COLORS
# from here also picks up the shared typography.
apply_style()

__all__ = [
    "MODEL_COLORS", "MODEL_MARKERS", "GPT4_RED", "GPT5_BLUE", "model_color",
    "share_origin_label",
    "INK", "GREY", "MUTED", "GRID",
    "FONT_STACK", "FS_TITLE", "FS_LABEL", "FS_PANEL_TITLE", "FS_TICK",
    "FS_LEGEND", "FS_VALUE", "FS_ANNOT", "FS_NOTE", "FS_INSET", "FS_PANEL_LETTER",
    "apply_style", "label_axes", "panel_letter", "sig_note", "sig_key", "save_figure",
    "square_axes", "three_ticks", "flush_category_axis",
]
