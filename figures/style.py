"""Shared typography, colours and saving for every figure.

Helvetica Neue if the system has it (macOS), otherwise Arial or another sans
serif; every figure is saved as PNG and SVG (and PDF where asked).
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

# Neutral greys used for error bars, grids, significance marks and footnotes.
INK = "#333333"
MUTED = "#777777"
GRID = "#DDDDDD"

# ── TYPOGRAPHY ───────────────────────────────────────────────────────────────
# One family, one size scale. Helvetica Neue first, then Arial and the rest as
# fallbacks so the figures still render on machines without it.
#
# macOS ships Helvetica Neue as a .ttc *collection*, and matplotlib registers only
# the first face in a collection - so a bold request silently comes back with the
# regular face and every "bold" renders at normal weight, which quietly breaks
# bold text. _register_helvetica_neue splits the faces we use out of the
# collection, into matplotlib's own cache directory (never into the repo: the font
# is Apple-licensed), and registers each one so real weights are available.
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

def apply_style() -> None:
    """Install the house rcParams. Call once, before creating any figure."""
    plt.rcParams.update({
        # -- font: one family, one size scale ------------------------------
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": FS_TICK,
        "font.weight": "normal",
        "mathtext.default": "regular",

        # -- titles: the only bold text ------------------------------------
        "axes.titlesize": FS_TITLE,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
        "figure.titlesize": FS_TITLE,
        "figure.titleweight": "bold",

        # -- everything else stays regular weight --------------------------
        "axes.labelsize": FS_LABEL,
        "axes.labelweight": "normal",
        "axes.labelcolor": "black",
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "legend.title_fontsize": FS_LEGEND,
        "figure.labelsize": FS_LABEL,
        "figure.labelweight": "normal",

        # -- frame / grid --------------------------------------------------
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # -- output: crisp raster + editable vector ------------------------
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",   # keep SVG text as text (editable in Illustrator)
        "svg.hashsalt": "wm-llm-benchmark",   # stable ids, so rebuilt SVGs do not change
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
    """Square plotting box."""
    ax.set_box_aspect(1)
    return ax


def _tick_label(v):
    """Format a tick value without trailing zeros: 0, 0.5, 1 (not 0.0, 0.5, 1.0)."""
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def three_ticks(ax, lo, hi, axis="both", mid=None, fmt=None):
    """Ticks at [lo, mid, hi] only, with the axis ending exactly at hi."""
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


def save_figure(fig, out_path, dpi=300, also_pdf=False, close=True):
    """Save ``fig`` as PNG and SVG (and PDF); ``out_path``'s extension is ignored."""
    stem = os.path.splitext(out_path)[0]
    parent = os.path.dirname(stem)
    if parent:
        os.makedirs(parent, exist_ok=True)

    written = []
    for ext in (".png", ".svg") + ((".pdf",) if also_pdf else ()):
        path = stem + ext
        # no creation dates in the files, so an unchanged figure rebuilds byte-identical
        meta = {".svg": {"Date": None}, ".pdf": {"CreationDate": None}}.get(ext)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", metadata=meta)
        written.append(path)

    if close:
        plt.close(fig)
    print("saved " + " + ".join(os.path.basename(p) for p in written)
          + f"  ->  {parent or '.'}")
    return written


apply_style()
