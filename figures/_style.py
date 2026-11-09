"""Shared house style for all RTE paper figures (matches fig_drift_age / fig_passn / fig_weighting)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
_fp = fm.FontProperties(fname=FONT)
fm.fontManager.addfont(FONT)

def apply():
    plt.rcParams.update({
        "font.family": _fp.get_name(),
        "font.weight": "bold",
        "font.size": 11,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "figure.titleweight": "bold",
        "axes.linewidth": 1.3,
        "xtick.major.width": 1.1, "ytick.major.width": 1.1,
        "xtick.direction": "in", "ytick.direction": "in",
        "axes.grid": False, "pdf.fonttype": 42,
        "mathtext.default": "bf",  # bold math too
    })

# House palette (matplotlib tab family, as used by the existing figures)
BLUE, DARK, RED, GREY = "#1f77b4", "#0d3b66", "#d62728", "#8a8a8a"
ORANGE, GREEN, PURPLE = "#ff7f0e", "#2ca02c", "#9467bd"
# Role assignments, fixed across every figure
BACKBONE = {"pi0": BLUE, "pi05": ORANGE, "groot": GREEN, "dp": PURPLE}
RTE = RED          # our method, emphasis color (matches passn/weighting)
FRESHEST = BLUE
BASE = GREY

def finish(ax, bold_ticks=True):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.3); sp.set_color("black")
    if bold_ticks:
        for lab in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            lab.set_fontweight("bold")


# Panels that sit side by side must share one canvas size, or LaTeX scales them to the same
# width and their heights disagree. PANEL is that shared canvas, in inches; it is large enough
# to contain the tight bounding box of every panel, so no label is ever cropped.
PANEL = (3.62, 2.86)

# The three teaser panels are scaled to one width in LaTeX, so their type must be
# specified once and reused. Three roles only: axis labels/titles, tick labels, and
# anything annotating inside the axes (legends, callouts, value labels).
T_AXIS, T_TICK, T_ANNOT = 10.5, 9.0, 8.5

# Same canvas is not enough: the axes rectangle inside it must match too, or the
# plot frames still sit at different heights once LaTeX lines the panels up.
# (left, bottom, width, height) in figure coordinates. The left margin is set by
# the widest y-tick labels of the three (the capture panel's suite names), the top
# by the tick labels alone (panel titles are dropped in favour of the (a)/(b)/(c)
# subcaptions), the bottom by the x label.
TEASER_RECT = (0.280, 0.245, 0.690, 0.720)


def place(ax):
    """Pin a teaser panel's axes to the shared rectangle. Call last, after every
    label and title is set, since tight_layout/subplots_adjust would override it."""
    ax.set_position(TEASER_RECT)

def save_panel(fig, path, png=None):
    """Save on the shared PANEL canvas, centred on the figure's own tight bounding box."""
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    cx, cy = 0.5 * (tb.x0 + tb.x1), 0.5 * (tb.y0 + tb.y1)
    w, h = PANEL
    if tb.width > w or tb.height > h:
        raise ValueError(f"tight bbox {tb.width:.2f}x{tb.height:.2f} exceeds PANEL {w}x{h}")
    box = Bbox.from_extents(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    fig.savefig(path, bbox_inches=box)
    if png:
        fig.savefig(png, dpi=300, bbox_inches=box)
