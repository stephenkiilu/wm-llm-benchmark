"""Page layout shared by the manuscript figures (figure1.py, figure2.py).

Both figures are laid out on a millimetre grid measured from the top-left
corner, then printed at ``scale`` of that grid. Type is given in printed points,
so resizing a figure does not resize its text by the same factor.

The checks at the bottom fail the build rather than let a figure ship with a
label off the page or two labels on top of each other; those faults are easy
to miss in code and obvious in print.
"""
from __future__ import annotations

import warnings

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle


class Page:
    """Places axes and text in layout millimetres, from the top-left corner."""

    def __init__(self, width_mm, height_mm, *, scale, dpi, text_pt, letter_pt,
                 axis_lw, panel_pos):
        self.w, self.h = width_mm, height_mm      # layout grid, before scale
        self.text_pt, self.letter_pt = text_pt, letter_pt
        self.panel_pos = panel_pos
        plt.rcParams.update({"font.size": text_pt, "axes.linewidth": axis_lw,
                             "svg.fonttype": "none", "svg.hashsalt": "wm-llm-benchmark", "pdf.fonttype": 42,
                             "savefig.pad_inches": 0})
        self.fig = plt.figure(figsize=(width_mm * scale / 25.4, height_mm * scale / 25.4),
                              dpi=dpi)
        # A transparent sheet over the whole page, so the tight bounding box used
        # by S.save_figure is the page itself and the export keeps its exact size.
        self.fig.add_artist(Rectangle((0, 0), 1, 1, transform=self.fig.transFigure,
                                      fill=False, lw=0, zorder=-10))

    def axes(self, x, y, w, h):
        return self.fig.add_axes([x / self.w, 1 - (y + h) / self.h, w / self.w, h / self.h])

    def text(self, x, y, string, size=None, **kw):
        kw.setdefault("va", "top")
        return self.fig.text(x / self.w, 1 - y / self.h, string,
                             fontsize=size or self.text_pt, **kw)

    def panel(self, letter, title):
        """Bold lowercase letter, then a bold heading at text size."""
        x, y = self.panel_pos[letter]
        self.text(x, y, letter, self.letter_pt, weight="bold")
        self.text(x + 4, y + 0.3, title, weight="bold")

    def panel_over(self, letter, title, ax, y=0, gap_mm=3.5):
        """Heading centred over ``ax``, its letter ``gap_mm`` to the left of it."""
        box = ax.get_position()
        head = self.fig.text((box.x0 + box.x1) / 2, 1 - (y + 0.3) / self.h, title,
                             ha="center", va="top", fontsize=self.text_pt, weight="bold")
        self.fig.canvas.draw()
        left = head.get_window_extent(self.fig.canvas.get_renderer()).x0
        left_mm = left / self.fig.bbox.width * self.w
        self.text(left_mm - gap_mm, y, letter, self.letter_pt, weight="bold")


def legend_style(text_pt):
    """One legend style for every panel: square swatch (or a line, where the line
    style is the thing being keyed), label to the right, no frame."""
    return dict(frameon=False, fontsize=text_pt, handlelength=1.0,
                handleheight=1.0, handletextpad=0.5, columnspacing=1.6,
                labelspacing=0.35, borderaxespad=0, borderpad=0)


def style_chart(ax, text_pt, axis_lw):
    ax.tick_params(width=axis_lw, length=2, pad=1.6, labelsize=text_pt)
    for spine in ax.spines.values():
        spine.set_linewidth(axis_lw)
    ax.xaxis.label.set_size(text_pt)
    ax.yaxis.label.set_size(text_pt)
    ax.xaxis.labelpad = 1.6
    ax.yaxis.labelpad = 1.6


def align_end_ticks(ax):
    """Pull the outer x tick labels inwards so they do not overhang the box."""
    labels = ax.get_xticklabels()
    labels[0].set_ha("left")
    labels[-1].set_ha("right")


# ── CHECKS ───────────────────────────────────────────────────────────────────
def _visible_text(fig):
    return [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]


def assert_text_inside(fig):
    """Fail rather than ship a figure with a label cropped by the page edge."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for t in _visible_text(fig):
        box = t.get_window_extent(renderer)
        if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.x1 or box.y1 > fig.bbox.y1:
            outside.append(t.get_text())
    if outside:
        raise AssertionError(f"Text outside the page: {outside}")


def assert_no_text_overlap(fig):
    """No two labels may touch."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in _visible_text(fig)]
    clashes = [(a, b) for i, (a, ba) in enumerate(boxes) for b, bb in boxes[i + 1:]
               if ba.overlaps(bb)]
    if clashes:
        raise AssertionError(f"Overlapping text: {clashes}")


def assert_no_legend_overlap(fig):
    """Text checks miss a legend's handles, so compare whole legend boxes too."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [lg.get_window_extent(renderer) for lg in fig.findobj(matplotlib.legend.Legend)]
    if any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:]):
        raise AssertionError("Two legends overlap.")


def check_page(fig):
    """Fail on a layout fault, unless the intended font is missing on this machine:
    the layout is tuned for it, so a wider fallback font only earns a warning."""
    try:
        assert_text_inside(fig)
        assert_no_text_overlap(fig)
        assert_no_legend_overlap(fig)
    except AssertionError as fault:
        family = plt.rcParams["font.sans-serif"][0]
        try:
            font_manager.findfont(font_manager.FontProperties(family=family),
                                  fallback_to_default=False)
        except ValueError:
            warnings.warn(f"{fault} ({family} is not installed; layout was tuned for it)")
            return
        raise
