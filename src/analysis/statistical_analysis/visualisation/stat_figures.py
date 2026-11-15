from contextlib import contextmanager
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.font_manager import FontProperties
from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import expit
from ..stat_models import ModelResults
from ..config import MODEL_DISPLAY, SOURCE_DISPLAY, CONTROL_VARS, REGION_COLUMN

from .helpers import (
    save_figure,
    setup_plot_style,
    pretty_dep_label,
    relabel_model_source,
    pretty_model,
    SOURCE_COLUMN,
    MODEL_COLUMN,
    OUTCOME_COLUMN,
)


# Sign-encoding palette shared by every coefficient / effect figure in this
# module (coefplots, the per-source grid, the marginal-effects bars). Change it
# here and the whole set follows; the same two hues drive the score-distribution
# ramp in ``config.SENTIMENT_COLORS`` so the paper reads as one system.
_POSITIVE_COLOR = "#0097b2"  # teal
_NEGATIVE_COLOR = "#cea90d"  # gold
_INSIGNIFICANT_COLOR = "#999999"

_POSITIVE_MARKER = "^"
_NEGATIVE_MARKER = "v"
_INSIGNIFICANT_MARKER = "o"

_POINT_SIZE = 32


# Type sizes in points, and the one piece of arithmetic that makes them mean
# something on the page.
#
# _PAPER_FIG_WIDTH is ACM sigconf's \textwidth, but main.tex includes these
# floats at _PAPER_INCLUDE_FRACTION of it, so LaTeX scales the artwork *and its
# type* by that factor: a size below reaches the page at roughly
# size * _PAPER_INCLUDE_FRACTION. The labels are therefore authored well above
# body size to print a little under it (11pt -> ~6.9pt against 9pt body text).
#
# Keep _PAPER_INCLUDE_FRACTION in sync with the width= in main.tex. Raising it
# there without lowering these sizes prints the figures with type larger than
# the running text; lowering it shrinks the labels back toward illegibility.
_PAPER_INCLUDE_FRACTION = 0.6

# The coefficient labels are the anchor; everything else steps off them. Change
# _LABEL_FONTSIZE and the hierarchy follows.
_LABEL_FONTSIZE = 11.0     # y-axis coefficient / covariate names
# Clearance between the widest y-tick label and the axes, on top of the tick
# mark and tick pad the labels are already offset by.
_LABEL_MARGIN_PAD_IN = 0.14
_TICK_FONTSIZE = 10.0      # x-axis numerals
_XLABEL_FONTSIZE = 11.0
_TITLE_FONTSIZE = 12.0     # per-panel titles
_SUPTITLE_FONTSIZE = 13.0
_LEGEND_FONTSIZE = 10.0


# "Which side means what" strip drawn under the x-tick labels of a
# differential panel. A coefplot of a difference is the one place where the
# sign carries information the axis cannot show — a reader has to remember
# which source the differential was taken from — so the two directions are
# spelled out where the eye already is, under the axis.
_DIRECTION_ARROW_ROW_IN = 0.24   # the arrow's own row, above the labels
_DIRECTION_LINE_IN = 0.22        # per line of label text
_DIRECTION_ARROW_IN = 0.10       # arrow's drop below the band's top edge
_DIRECTION_ARROW_COLOR = "#8c8c8c"
_DIRECTION_TEXT_COLOR = "#4a4a4a"
_DIRECTION_FONTSIZE = 10.0
_DIRECTION_LABEL_GAP_IN = 0.20   # clear space between the two labels
# Footnotes are qualifications, read only if the reader wants them, so they are
# set below the smallest type that carries information.
_FOOTNOTE_SIZE_RATIO = 0.78


def _direction_band_in(labels: Optional[Tuple[str, str]],
                       caption: Optional[str] = None,
                       drawn_arrow: bool = True) -> float:
    """Height the direction strip needs: its caption, the arrow's row, its labels."""
    if not labels:
        return 0.0
    lines = max(len(label.splitlines()) for label in labels)
    if caption:
        lines += len(caption.splitlines())
    return (_DIRECTION_ARROW_ROW_IN if drawn_arrow else 0.0) + lines * _DIRECTION_LINE_IN

_ZERO_LINE_COLOR = "#9a9a9a"
_ZERO_LINE_WIDTH = 0.7

# Line weights for the coefplot chrome, matching the prototype sheet in
# results/final2.0/figures/prototype/refit_and_plot.py: one hairline guide per
# coefficient row, a dashed rule wherever the term group changes, a faint dotted
# x grid and a single light bottom spine. Everything here is deliberately
# lighter than the data — the guides locate a row, they are not read.
_ROW_GUIDE_COLOR = "#e6e6e6"
_ROW_GUIDE_WIDTH = 0.45
_GROUP_RULE_COLOR = "#b4b4b4"
_GROUP_RULE_WIDTH = 0.5
_GROUP_RULE_STYLE = (0, (2.5, 2.5))
# Extra gutter opened at a panel-group boundary, in inches. Taken out of the
# panels rather than added to the figure, so the figure's width is unchanged.
_GROUP_GUTTER_IN = 0.20
_XGRID_COLOR = "#c8c8c8"
_XGRID_WIDTH = 0.35
_XGRID_ALPHA = 0.55
_SPINE_COLOR = "#c8c8c8"

# Confidence whiskers: a thin line with short end caps, in points, rather than
# cap ticks drawn in data units (those scaled with the row pitch, so tightening
# the rows turned them into full-height bars).
#
# Two weights, because the two coefplot layouts print at different scales. The
# multi-panel grid splits _PAPER_FIG_WIDTH between its panels, so each panel is
# about half as wide as a single-panel coefplot and the prototype's hairline
# weight reads correctly there. On a single panel the same line spans twice the
# distance and looks starved, so it carries the heavier set below.
_CI_LINE_WIDTH = 0.9
_CI_CAP_SIZE = 1.7
_CI_CAP_WIDTH = 0.7
_CI_LINE_WIDTH_SINGLE = 1.6
_CI_CAP_SIZE_SINGLE = 2.6
_CI_CAP_WIDTH_SINGLE = 1.2
_CI_ALPHA = 0.95


# Monochrome, single-channel (typographic) cue distinguishing term types on the
# y-axis: ideology = italic, source = regular, model = bold. Interaction terms
# (e.g. "Model_Grok*immig") are styled per component, so each side of the "*"
# keeps its own term-type formatting. The "Model_" prefix is dropped from model
# labels at display time.
_TERM_FONT_KW = {
    "ideology": dict(style="italic", weight="normal"),
    "model":    dict(style="normal", weight="bold"),
    "source":   dict(style="normal", weight="normal"),
}

_DEFAULT_LABEL_COLOR = "black"
_MARKER_EDGE_COLOR = "black"
_BASELINE_LABELS = ("Intercept", "Baseline E[ΔY]", "Baseline E[Y]")
_BASELINE_COLOR = "#aaaaaa"  # faded grey for intercept / baseline rows

# Control covariates: their own group (after interactions, before the intercept),
# shown in muted grey so they read as non-focal nuisance terms. Continuous
# controls match by exact (pretty) label; categorical-dummy controls such as the
# regional fixed effects match by a "<var>_" prefix (e.g. "region_Anglophone").
_CONTROL_LABELS = {pretty_dep_label(v) for v in CONTROL_VARS}
_CONTROL_LABEL_PREFIXES = (f"{REGION_COLUMN}_",)
_CONTROL_COLOR = "#777777"


def _is_control_label(label: str) -> bool:
    return label in _CONTROL_LABELS or label.startswith(_CONTROL_LABEL_PREFIXES)

_SOURCE_LABEL_TOKENS = set(SOURCE_DISPLAY.values()) | {"Source"}
_MODEL_LABEL_TOKENS = set(MODEL_DISPLAY.values())


def _segment_category(seg: str) -> str:
    """Classify one component of a (pretty) coefficient label as model/source/ideology."""
    # Exact match on model tokens avoids the "Grok" ⊂ "Grokipedia" clash.
    if "Model" in seg or seg in _MODEL_LABEL_TOKENS:
        return "model"
    if seg in _SOURCE_LABEL_TOKENS or any(tok in seg for tok in SOURCE_DISPLAY.values()):
        return "source"
    return "ideology"


def _styled_label_segments(label: str, fontsize: float,
                           display: Optional[str] = None,
                           italic_terms: bool = True):
    """Split a label on '*' into [(text, FontProperties, color), ...].

    Each component is styled by its own term type; the '*' separators stay neutral
    so an interaction such as "A*B" preserves A's and B's individual formatting.
    The "Model_" prefix is stripped from model components for display.

    `display` replaces the drawn text while `label` still decides the styling;
    it is only honoured for single-component (non-interaction) labels, which is
    what the display renames cover.
    """
    if display is not None and display != label and "*" not in label:
        category = _label_category(label)
        kw = dict(_TERM_FONT_KW.get(category, _TERM_FONT_KW["ideology"]))
        if not italic_terms:
            kw["style"] = "normal"
        color = _CONTROL_COLOR if category == "control" else _DEFAULT_LABEL_COLOR
        if label in _BASELINE_LABELS:
            kw, color = dict(style="italic", weight="normal"), _BASELINE_COLOR
        return [(display, FontProperties(size=fontsize, **kw), color)]
    if label in _BASELINE_LABELS:
        return [(label, FontProperties(size=fontsize, style="italic"), _BASELINE_COLOR)]
    if _is_control_label(label):
        # Drop the "<var>_" prefix from categorical-dummy controls for display
        # (e.g. "region_Anglophone" -> "Anglophone").
        display = label.split("_", 1)[1] if label.startswith(_CONTROL_LABEL_PREFIXES) else label
        return [(display, FontProperties(size=fontsize, style="normal"), _CONTROL_COLOR)]

    segments = []
    for i, part in enumerate(label.split("*")):
        if i > 0:
            segments.append(("*", FontProperties(size=fontsize), _DEFAULT_LABEL_COLOR))
        category = _segment_category(part)
        kw = dict(_TERM_FONT_KW[category])
        if not italic_terms:
            kw["style"] = "normal"
        text = part.replace("Model_", "") if category == "model" else part
        segments.append((text, FontProperties(size=fontsize, **kw), _DEFAULT_LABEL_COLOR))
    return segments


def _style_term_ticklabels(ax: plt.Axes, labels: Sequence[str],
                           fontsize: float = _LABEL_FONTSIZE,
                           display_labels: Optional[Sequence[str]] = None,
                           italic_terms: bool = True) -> None:
    """Overlay y-tick labels styled per term type (monochrome, single-channel).

    The plain tick labels are kept but made transparent so the layout still
    reserves space for them; styled composites are drawn on top, right-anchored
    at the axis exactly like normal tick labels.

    `display_labels` (parallel to `labels`) supplies the text to draw while the
    term-type styling keeps keying off the short labels in `labels`.
    """
    for tick in ax.get_yticklabels():
        tick.set_alpha(0)

    pad_pts = plt.rcParams["ytick.major.size"] + plt.rcParams["ytick.major.pad"]
    transform = ax.get_yaxis_transform()  # x: axes fraction, y: data coordinates
    shown = list(display_labels) if display_labels is not None else list(labels)
    for y, (label, display) in enumerate(zip(labels, shown)):
        children = [
            TextArea(text, textprops=dict(fontproperties=fp, color=color))
            for text, fp, color in _styled_label_segments(
                label, fontsize, display=display, italic_terms=italic_terms,
            )
        ]
        packed = HPacker(children=children, align="baseline", pad=0, sep=0)
        ab = AnnotationBbox(
            packed, (0, y), xycoords=transform,
            xybox=(-pad_pts, 0), boxcoords="offset points",
            box_alignment=(1.0, 0.5), frameon=False, pad=0,
            annotation_clip=False,
        )
        ax.add_artist(ab)


_COEFFICIENT_LABEL_ORDER = [
    "culincl",
    "immig",
    "lgbt",
    "womlab",
    "anti-client",
    "plural",
    "anti-popul",
]

# Group ordering for the coefficient axis: ideologies first, then controls, then
# sources, then models, then interaction terms. (Intercept / baseline rows are
# appended last by the extract functions, so they always sit at the bottom.)
_CATEGORY_RANK = {"ideology": 0, "control": 1, "source": 2, "model": 3, "interaction": 4}


def _label_category(label: str) -> str:
    """Category of a whole coefficient label.

    Controls take precedence over the term-type classification; an interaction is
    any label that spans a '*'.
    """
    if _is_control_label(label):
        return "control"
    if "*" in label:
        return "interaction"
    return _segment_category(label)


def _label_group(label: str) -> str:
    """Group a label belongs to for the dashed separator rules.

    Same partition as `_label_category`, except the intercept / baseline rows
    form their own group so they are always ruled off from the terms above.
    """
    if label in _BASELINE_LABELS:
        return "baseline"
    return _label_category(label)


def _label_sort_indices(labels: List[str]) -> List[int]:
    """Indices that reorder `labels` for coefficient plots.

    Primary order is by term group — ideologies, then sources, then models, then
    interactions. Within ideologies the explicit `_COEFFICIENT_LABEL_ORDER` is
    honoured; all other ties keep their original relative order.
    """
    ideology_rank = {name: i for i, name in enumerate(_COEFFICIENT_LABEL_ORDER)}

    def key(i: int):
        lbl = labels[i]
        category = _label_category(lbl)
        group = _CATEGORY_RANK.get(category, _CATEGORY_RANK["ideology"])
        within = ideology_rank.get(lbl, len(ideology_rank)) if category == "ideology" else 0
        return (group, within, i)

    return sorted(range(len(labels)), key=key)


def _coef_styles_and_labels(coefs, pvalues, names, alpha: float):
    """Map (coef, p-value) → (color, marker); labels pass through unchanged."""
    colors, markers = [], []
    for c, p in zip(coefs, pvalues):
        if p >= alpha:
            colors.append(_INSIGNIFICANT_COLOR)
            markers.append(_INSIGNIFICANT_MARKER)
        elif c > 0:
            colors.append(_POSITIVE_COLOR)
            markers.append(_POSITIVE_MARKER)
        else:
            colors.append(_NEGATIVE_COLOR)
            markers.append(_NEGATIVE_MARKER)
    return colors, markers, list(names)


def _scatter_by_marker(ax, xs, ys, colors, markers, **scatter_kwargs):
    """Scatter points grouped by marker (matplotlib's scatter takes one marker per call)."""
    by_marker: dict = {}
    for x, y, c, m in zip(xs, ys, colors, markers):
        by_marker.setdefault(m, ([], [], []))
        by_marker[m][0].append(x)
        by_marker[m][1].append(y)
        by_marker[m][2].append(c)
    for marker, (mx, my, mc) in by_marker.items():
        ax.scatter(mx, my, c=mc, marker=marker, **scatter_kwargs)


# Ink for a light page is the default throughout this module. A figure dropped
# on a dark poster board needs the opposite: the marks stay (they carry the
# data), everything that is chrome or type flips to light. The swap is done by
# rebinding these module names for the duration of one figure — they are read
# inside the drawing functions, so nothing else has to thread a palette through
# — plus the rcParams matplotlib uses for tick labels, titles and legend text.
_LIGHT_INK = {
    "_DEFAULT_LABEL_COLOR": "#f5f5f5",
    "_BASELINE_COLOR": "#9a9a9a",
    "_CONTROL_COLOR": "#b0b0b0",
    "_FIT_STATS_COLOR": "#c8c8c8",
    "_DIRECTION_TEXT_COLOR": "#e0e0e0",
    "_DIRECTION_ARROW_COLOR": "#b0b0b0",
    "_ROW_GUIDE_COLOR": "#4a4a4a",
    "_GROUP_RULE_COLOR": "#8a8a8a",
    "_ZERO_LINE_COLOR": "#9a9a9a",
    "_XGRID_COLOR": "#6a6a6a",
    "_SPINE_COLOR": "#8a8a8a",
    "_MARKER_EDGE_COLOR": "#1a1a1a",
}
_LIGHT_INK_RC = {
    "text.color": "#f5f5f5",
    "axes.titlecolor": "#f5f5f5",
    "legend.labelcolor": "#f5f5f5",
    "axes.labelcolor": "#f5f5f5",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#8a8a8a",
}


@contextmanager
def _ink(light: bool):
    """Draw with light-on-dark chrome and type inside this block."""
    if not light:
        yield
        return
    previous = {name: globals()[name] for name in _LIGHT_INK}
    globals().update(_LIGHT_INK)
    try:
        with plt.rc_context(_LIGHT_INK_RC):
            yield
    finally:
        globals().update(previous)



def _coefficient_legend_handles() -> List[Line2D]:
    common = dict(color="none", markeredgecolor=_MARKER_EDGE_COLOR,
                  markersize=8, linestyle="None")
    return [
        Line2D([0], [0], marker=_POSITIVE_MARKER, markerfacecolor=_POSITIVE_COLOR, label="Positive", **common),
        Line2D([0], [0], marker=_NEGATIVE_MARKER, markerfacecolor=_NEGATIVE_COLOR, label="Negative", **common),
        Line2D([0], [0], marker=_INSIGNIFICANT_MARKER, markerfacecolor=_INSIGNIFICANT_COLOR, label="n.s.", **common),
    ]


_LEGEND_PAD_IN = 0.03  # floor for the legend's distance from the figure bottom
# Clearance between the x-tick numerals and whatever is hung under them (the
# direction strip, the sign key), measured from the numerals rather than
# reserved, so it stays the same however tall they are.
_BELOW_AXIS_GAP_IN = 0.13


def plot_coefficient_legend(
    save_path: Optional[Path] = None,
    ncol: int = 3,
    fontsize: float = _LEGEND_FONTSIZE,
    formats: Sequence[str] = ("svg", "png", "pdf"),
) -> plt.Figure:
    """The Positive / Negative / n.s. key on its own, as a standalone figure.

    Same handles as the key drawn inside the coefplots
    (:func:`_coefficient_legend_handles`), so the two cannot drift apart. Meant
    to pair with ``plot_coefficients_grid(..., show_legend=False)``: on a slide
    the key is placed once and the panels are revealed under it, which the
    baked-in legend cannot do.

    SVG leads ``formats`` because the file is a few vector glyphs that will be
    scaled and repositioned by hand in the deck; a raster of a 3-marker key is
    either fuzzy or needlessly large.
    """
    setup_plot_style()
    # The figsize is a starting canvas, not the output size: the legend is
    # attached to the FIGURE and no axes is added, so the bbox_inches="tight"
    # crop below has only the key to measure and the file comes out at the
    # key's own size. An axes spanning the canvas would set the tight bbox
    # itself and leave the key floating in whitespace.
    fig = plt.figure(figsize=(4.0, 0.5))
    fig.legend(
        handles=_coefficient_legend_handles(), loc="center",
        ncol=ncol, fontsize=fontsize,
        frameon=False, handlelength=1.2, handleheight=1.0,
        columnspacing=1.8, borderaxespad=0.0,
    )
    if save_path is not None:
        save_path = Path(save_path)
        for ext in formats:
            save_figure(fig, save_path.with_suffix(f".{ext}"), dpi=300,
                        pad_inches=_SAVE_PAD_IN,
                        transparent=_TRANSPARENT_BACKGROUND)
    return fig


def _legend_width_in(fig: plt.Figure, fontsize: float, markerscale: float) -> float:
    """Width the sign key needs, in inches, without leaving it on the figure."""
    legend = fig.legend(
        handles=_coefficient_legend_handles(), loc="center",
        bbox_to_anchor=(0.5, 0.5), ncol=3, fontsize=fontsize,
        frameon=False, handlelength=1.2, handleheight=1.0,
        columnspacing=1.8, borderaxespad=0.0, borderpad=0.0,
        markerscale=markerscale,
    )
    width = legend.get_window_extent(renderer=fig.canvas.get_renderer()).width
    legend.remove()
    return width / fig.dpi


def _place_legend_in_width(fig: plt.Figure, x_center: float, y_center: float,
                           max_width_in: float, fontsize: float,
                           markerscale: float, ncol: int = 3,
                           loc: str = "center"):
    """Draw the sign key centred on (x_center, y_center), shrunk to fit a width.

    The key's width depends on the type metrics of three labels and their
    handles, so it is measured rather than predicted: drawn, measured, and
    redrawn a size smaller while it overruns. Without this it silently overflows
    whatever it was centred in — the tight crop then widens the whole figure
    rather than clipping, which is how it hides.
    """
    renderer = fig.canvas.get_renderer()
    for _ in range(6):
        legend = fig.legend(
            handles=_coefficient_legend_handles(), loc=loc,
            bbox_to_anchor=(x_center, y_center),
            ncol=ncol, fontsize=fontsize,
            frameon=False, handlelength=1.2, handleheight=1.0,
            columnspacing=1.8, borderaxespad=0.0, borderpad=0.0,
            markerscale=markerscale,
        )
        width_in = legend.get_window_extent(renderer=renderer).width / fig.dpi
        if width_in <= max_width_in:
            return legend
        legend.remove()
        fontsize *= max(0.75, max_width_in / width_in)
        markerscale *= max(0.75, max_width_in / width_in)
    return legend


def _add_coefficient_legend(fig: plt.Figure, ax: plt.Axes) -> None:
    """Park the sign legend directly under the axes' x-tick labels.

    Hung off the measured bottom of the tick labels, as in
    :func:`plot_coefficients_grid`: the band reserved below the axes is an
    upper bound on what the numerals need, so subtracting it left a gap the
    size of whatever they did not use. Whatever blank is left below the legend
    is removed by the ``bbox_inches="tight"`` crop.
    """
    pos = ax.get_position()
    x_center = pos.x0 + pos.width / 2
    fig_height = fig.get_figheight()
    fig.legend(
        handles=_coefficient_legend_handles(), loc="upper center",
        bbox_to_anchor=(
            x_center, _lowest_ink_frac(fig, [ax]) - _BELOW_AXIS_GAP_IN / fig_height,
        ),
        ncol=3, fontsize=_LEGEND_FONTSIZE,
        frameon=False, handlelength=1.2, handleheight=1.0,
        columnspacing=1.8, borderaxespad=0.0, borderpad=0.0,
    )


def _draw_confidence_intervals(
    ax: plt.Axes,
    coefs: Sequence[float],
    y_positions: Sequence[float],
    conf_ints: Optional[Sequence],
    colors: Sequence[str],
    line_width: float = _CI_LINE_WIDTH,
    cap_size: float = _CI_CAP_SIZE,
    cap_width: float = _CI_CAP_WIDTH,
) -> None:
    """Horizontal 95% CI whiskers with short end caps, one per coefficient.

    Weights default to the multi-panel set; single-panel callers pass the
    heavier ``*_SINGLE`` values.
    """
    if conf_ints is None:
        return
    for x, y, ci, color in zip(coefs, y_positions, conf_ints, colors):
        if ci is None or np.isnan(ci[0]):
            continue
        ax.errorbar(
            x, y, xerr=[[x - ci[0]], [ci[1] - x]], fmt="none", ecolor=color,
            elinewidth=line_width, capsize=cap_size,
            capthick=cap_width, alpha=_CI_ALPHA, zorder=2,
        )


def _style_coefficient_axes(ax: plt.Axes, labels: Sequence[str],
                            scale: float = 1.0) -> None:
    """Draw the coefplot chrome: row guides, group rules, zero line, grid, spines.

    Replaces seaborn's whitegrid background (whose y grid is one heavy line per
    row) with the lighter vocabulary from the prototype sheet. Call after the
    y ticks are set — the guides are placed at tick positions 0..n-1.
    """
    n_rows = len(labels)
    ax.grid(False)  # drop the style default before drawing our own

    for y in range(n_rows):
        ax.axhline(y, color=_ROW_GUIDE_COLOR, lw=_ROW_GUIDE_WIDTH * scale, zorder=0)

    # A dashed rule wherever the term group changes (ideologies | models |
    # intercept), so the reader sees the blocks without extra labelling.
    for i in range(1, n_rows):
        if _label_group(labels[i]) != _label_group(labels[i - 1]):
            ax.axhline(
                i - 0.5, color=_GROUP_RULE_COLOR, lw=_GROUP_RULE_WIDTH * scale,
                ls=_GROUP_RULE_STYLE, zorder=1,
            )

    ax.axvline(0, color=_ZERO_LINE_COLOR, lw=_ZERO_LINE_WIDTH * scale, zorder=1)
    ax.grid(
        True, axis="x", color=_XGRID_COLOR, lw=_XGRID_WIDTH * scale, ls=":",
        alpha=_XGRID_ALPHA, zorder=0,
    )
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(_SPINE_COLOR)
    ax.tick_params(axis="y", length=0)


def _draw_coefficient_points(
    ax: plt.Axes,
    coefs: np.ndarray,
    pvalues: np.ndarray,
    labels: List[str],
    colors: List[str],
    markers: List[str],
    conf_ints: Optional[List],
    alpha: float,
    title: str,
    xlabel: str = "",
) -> None:
    """Coefplot-style: point estimate as a marker, CI as a horizontal whisker."""
    y_pos = np.arange(len(coefs))

    _draw_confidence_intervals(
        ax, coefs, y_pos, conf_ints, colors,
        line_width=_CI_LINE_WIDTH_SINGLE,
        cap_size=_CI_CAP_SIZE_SINGLE,
        cap_width=_CI_CAP_WIDTH_SINGLE,
    )
    _scatter_by_marker(
        ax, coefs, y_pos, colors, markers,
        s=_POINT_SIZE, zorder=3, edgecolor="black", linewidth=0.4,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=_LABEL_FONTSIZE)
    _style_term_ticklabels(ax, labels)
    ax.tick_params(axis="x", labelsize=_TICK_FONTSIZE)
    ax.invert_yaxis()
    _set_coefficient_ylim(ax, len(coefs))
    if title:
        ax.set_title(title, fontweight="bold", fontsize=_TITLE_FONTSIZE, pad=6)
    _style_coefficient_axes(ax, labels)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=_XLABEL_FONTSIZE)


def _fit_stats_text(results) -> Optional[str]:
    """One-line fit summary for a coefplot corner: R² (or an analog).

    Resolution order:
    - ``fit_stats_label`` attribute (e.g. the MixedLM adapter supplies a
      marginal/conditional R² string),
    - OLS ``rsquared``,
    - discrete-model ``prsquared`` (McFadden pseudo-R²).
    Returns ``None`` when nothing applicable is available.

    Adjusted R² is deliberately omitted: with n in the thousands and a dozen
    predictors it never differs from R² before the third decimal, so it only
    lengthened the corner annotation. The regression tables still carry it.
    """
    label = getattr(results, "fit_stats_label", None)
    if label:
        return label

    r2 = getattr(results, "rsquared", None)
    if r2 is not None:
        return f"$R^2$ = {r2:.3f}"

    pr2 = getattr(results, "prsquared", None)
    if pr2 is None:
        return None
    return f"pseudo $R^2$ = {pr2:.3f}"


_FIT_STATS_COLOR = "#6b6b6b"
_FIT_STATS_FONTSIZE = 9.5
# How far below the top of the axes the fit-stats line is dropped, in points, so
# it does not sit flush against the panel title. Points rather than axes
# fraction: this module's coefplots come in several panel heights, and a
# fractional offset would mean a different gap in each. The ceiling is the
# topmost row guide, which the paper grid puts ~4.7pt below the text's bounding
# box, so keep this well under that or the line lands on the first coefficient
# and its whisker.
_FIT_STATS_DROP_PT = 2.5


def _annotate_fit_stats(ax: plt.Axes, results,
                        fontsize: float = _FIT_STATS_FONTSIZE) -> bool:
    """Print the fit-stats line in the top-right corner of the axes.

    Top-right rather than bottom-right, unboxed: the bottom row is the
    intercept, so the old placement sat the text on that row's guide line and
    needed a white patch punched through the chrome to stay legible. The top
    corner is free of chrome and needs no mask — provided the caller reserved
    ``_FIT_STATS_ROW_PAD`` extra rows of in-axes headroom for it, otherwise the
    text lands on the topmost coefficient and its whisker.

    Returns whether anything was drawn.
    """
    text = _fit_stats_text(results)
    if not text:
        return False
    ax.annotate(
        text, xy=(0.985, 1.0), xycoords=ax.transAxes,
        xytext=(0, -_FIT_STATS_DROP_PT), textcoords="offset points",
        ha="right", va="top", fontsize=fontsize,
        color=_FIT_STATS_COLOR, style="italic", zorder=4,
        annotation_clip=False,
    )
    return True


def _clean_ols_param_name(name: str) -> str:
    """Normalize a statsmodels OLS parameter name so it matches a predictor key."""
    return (
        str(name)
        .replace("C(", "").replace(", Sum", "").replace(")", "")
        .replace("[T.", "_").replace("[S.", "_").replace("]", "")
        .replace(":", "_x_")
        .replace("_1", "")
    )


def _extract_ols_rows(ols_res: ModelResults, alpha: float, show_intercept: bool = True):
    """Return (coefs, pvalues, labels, conf_ints) for every OLS term, with Intercept last."""
    params_series = ols_res.results.params
    pvalues_series = ols_res.results.pvalues
    conf_int_df = ols_res.results.conf_int(alpha=alpha) if hasattr(ols_res.results, "conf_int") else None

    coefs: List[float] = []
    pvalues: List[float] = []
    labels: List[str] = []
    conf_ints: List[tuple] = []

    def _append_row(idx: int, label: str) -> None:
        coefs.append(params_series.iloc[idx])
        pvalues.append(pvalues_series.iloc[idx])
        labels.append(label)
        if conf_int_df is not None:
            row = conf_int_df.iloc[idx]
            conf_ints.append((row.iloc[0], row.iloc[1]))
        else:
            conf_ints.append((np.nan, np.nan))

    for idx, ols_name in enumerate(params_series.index):
        if ols_name == "Intercept":
            continue
        _append_row(idx, pretty_dep_label(_clean_ols_param_name(ols_name)))

    order = _label_sort_indices(labels)
    coefs = [coefs[i] for i in order]
    pvalues = [pvalues[i] for i in order]
    labels = [labels[i] for i in order]
    conf_ints = [conf_ints[i] for i in order]

    if show_intercept and "Intercept" in params_series.index:
        _append_row(params_series.index.get_loc("Intercept"), "Intercept")

    return np.array(coefs), np.array(pvalues), labels, conf_ints


def _extract_ordinal_rows(
    ordinal_res: ModelResults,
    difference_outcome: bool = False,
    show_baseline: bool = True,
):
    """Return (coefs, pvalues, labels, conf_ints) for predictors + derived baseline E[Y].

    `difference_outcome=True` marks regressions whose outcome is a difference
    (e.g. wiki−groki), so the baseline row is labelled E[ΔY] instead of E[Y].
    `show_baseline=False` drops that derived row, leaving only estimated terms
    (the ordinal cutpoints carry no slope interpretation).
    """
    n_preds = len(ordinal_res.predictors)
    coefs = np.array(ordinal_res.results.params[:n_preds])

    if hasattr(ordinal_res.results, "pvalues"):
        pvalues = np.array(ordinal_res.results.pvalues[:n_preds])
    elif hasattr(ordinal_res.results, "pvvalues"):
        pvalues = np.array(ordinal_res.results.pvvalues[:n_preds])
    else:
        pvalues = np.zeros(len(coefs))
        print("⚠ Warning: Could not extract p-values from ordinal model. All shown as significant.")

    labels = [pretty_dep_label(p) for p in ordinal_res.predictors]

    conf_ints: List[Optional[tuple]] = [None] * n_preds
    if hasattr(ordinal_res.results, "conf_int"):
        try:
            ci_df = ordinal_res.results.conf_int()[:n_preds]
            conf_ints = [(ci_df.iloc[i, 0], ci_df.iloc[i, 1]) for i in range(n_preds)]
        except Exception:
            pass

    order = _label_sort_indices(labels)
    coefs = coefs[order]
    pvalues = pvalues[order]
    labels = [labels[i] for i in order]
    conf_ints = [conf_ints[i] for i in order]

    if not show_baseline:
        return coefs, pvalues, labels, conf_ints

    # Derive the baseline expected outcome E[Y | x=0] from the ordinal thresholds.
    try:
        all_params = ordinal_res.results.params.values
        thresholds = ordinal_res.results.model.transform_threshold_params(all_params)
        # transform_threshold_params brackets the cutpoints with ±inf; drop those
        # so the explicit 0/1 bounds below yield exactly one prob per category.
        finite_thresholds = thresholds[np.isfinite(thresholds)]
        cum_probs = expit(finite_thresholds)
        cat_probs = np.diff(np.concatenate([[0.0], cum_probs, [1.0]]))
        unique_cats = np.unique(np.sort(ordinal_res.results.model.endog.astype(int)))
        baseline_ey = float(np.sum(unique_cats * cat_probs))

        coefs = np.append(coefs, baseline_ey)
        pvalues = np.append(pvalues, 0.0)  # Derived quantity — no formal p-value.
        labels.append("Baseline E[ΔY]" if difference_outcome else "Baseline E[Y]")
        conf_ints.append(None)
    except Exception as exc:
        print(f"Warning: Could not compute ordinal baseline E[ΔY]: {exc}")

    return coefs, pvalues, labels, conf_ints


# Vertical budget for the coefplots, in absolute inches. The figure height is
# built up from the parts that actually need room — one strip per coefficient
# row, plus fixed bands for the x-tick labels, the sign legend and a little
# headroom — rather than a row allowance plus one lump constant. That keeps the
# page area proportional to the number of coefficients: a figure carrying 11
# rows no longer reserves the whitespace of one carrying 20.
_PAPER_FIG_WIDTH = 7.0
_PAPER_ROW_HEIGHT = 0.225    # row pitch (11pt labels occupy ~0.15 in of it)
_PAPER_MIN_PANEL_IN = 1.7   # floor so a 3-row coefplot still reads as a panel
_XTICK_BAND_IN = 0.32       # x-tick labels below the axes
_LEGEND_BAND_IN = 0.34      # the one-row Positive/Negative/n.s. legend
_LEGEND_GAP_IN = 0.06       # clearance between tick labels and legend
_TOP_BAND_IN = 0.10         # headroom above the topmost row
_PANEL_TITLE_BAND_IN = 0.34  # per-panel bold title (grid only)
# Border left by the bbox_inches="tight" crop. Matplotlib's 0.1 in default adds
# 0.2 in of page margin on top of the whitespace LaTeX already puts around a
# float, so these are authored with a hairline border instead.
_SAVE_PAD_IN = 0.02

# Coefplots are written without their background patches, so a deck's own
# background shows through instead of a white card sitting on it. Against the
# paper's white page it makes no visible difference, so the whole coefplot
# family shares the one setting rather than splitting slide and paper paths.
_TRANSPARENT_BACKGROUND = True

# Padding inside the axes, in row units. The top pad clears the R² annotation
# in the top-right corner; the bottom pad separates the intercept row from the
# axis spine, as on the prototype sheet.
_Y_PAD_TOP = 0.75
_Y_PAD_BOTTOM = 0.7

# Extra rows of headroom added on top of _Y_PAD_TOP when the fit-stats line is
# drawn. _Y_PAD_TOP alone leaves the annotation sitting on the topmost
# coefficient and its whisker. Cut as fine as it goes: with the text flush to
# the (invisible) top spine, this clears the topmost marker by half a point.
_FIT_STATS_ROW_PAD = 0.15

# What the single-panel coefplot reserves below the axes. Only the legend band
# lives here: ``tight_layout`` fits the x-tick labels inside the rect itself, so
# counting them twice is what pushed the legend off the bottom of the figure.
_BOTTOM_RESERVE_IN = _LEGEND_BAND_IN + _LEGEND_GAP_IN


def _paper_panel_height(n_rows: int, extra_top_rows: float = 0.0,
                        row_height: float = _PAPER_ROW_HEIGHT,
                        min_panel_in: float = _PAPER_MIN_PANEL_IN) -> float:
    """Axes height for `n_rows` coefficients, including the in-axes y padding."""
    return max(
        min_panel_in,
        (n_rows + _Y_PAD_TOP + extra_top_rows + _Y_PAD_BOTTOM) * row_height,
    )


def _set_coefficient_ylim(ax: plt.Axes, n_rows: int, extra_top_rows: float = 0.0) -> None:
    """Pad the (inverted) y axis by the row-unit margins above.

    `extra_top_rows` is the fit-stats strip; pass the same value used for the
    panel height so the reserved inches and the reserved data range agree.
    """
    ax.set_ylim(n_rows - 1 + _Y_PAD_BOTTOM, -(_Y_PAD_TOP + extra_top_rows))


def _paper_figsize(n_rows: int, extra_top_rows: float = 0.0) -> tuple:
    height = (
        _paper_panel_height(n_rows, extra_top_rows)
        + _XTICK_BAND_IN
        + _BOTTOM_RESERVE_IN
        + _TOP_BAND_IN
    )
    return (_PAPER_FIG_WIDTH, height)


def _bottom_margin_frac(fig: plt.Figure) -> float:
    """Reserve a fixed-inch strip at the bottom, expressed as a figure fraction."""
    return min(0.45, _BOTTOM_RESERVE_IN / fig.get_figheight())


def _top_margin_frac(fig: plt.Figure) -> float:
    """Headroom above the axes, expressed as a figure fraction."""
    return 1 - _TOP_BAND_IN / fig.get_figheight()


def plot_ols_coefficients(
    ols_res: ModelResults,
    save_path: Optional[Path] = None,
    alpha: float = 0.05,
    show_intercept: bool = True,
) -> plt.Figure:
    """Coefplot of OLS coefficients with 95% CIs, sized for a two-column paper."""
    setup_plot_style()

    coefs, pvalues, labels, conf_ints = _extract_ols_rows(ols_res, alpha=alpha, show_intercept=show_intercept)
    colors, markers, annotated_labels = _coef_styles_and_labels(coefs, pvalues, labels, alpha)

    fit_pad = _FIT_STATS_ROW_PAD if _fit_stats_text(ols_res.results) else 0.0
    fig, ax = plt.subplots(figsize=_paper_figsize(len(coefs), fit_pad))
    _draw_coefficient_points(
        ax, coefs, pvalues, annotated_labels, colors, markers, conf_ints, alpha,
        title="",
    )
    _annotate_fit_stats(ax, ols_res.results)
    # Re-set: _draw_coefficient_points sizes the y range without the fit-stats strip.
    _set_coefficient_ylim(ax, len(coefs), fit_pad)
    plt.tight_layout(rect=(0.0, _bottom_margin_frac(fig), 1.0, _top_margin_frac(fig)))
    _add_coefficient_legend(fig, ax)
    save_figure(fig, save_path, dpi=300, pad_inches=_SAVE_PAD_IN,
                transparent=_TRANSPARENT_BACKGROUND)
    if save_path is not None:
        save_figure(fig, Path(save_path).with_suffix(".pdf"), dpi=300,
                    pad_inches=_SAVE_PAD_IN, transparent=_TRANSPARENT_BACKGROUND)
    plt.close()
    return fig


def plot_ordinal_coefficients(
    ordinal_res: ModelResults,
    save_path: Optional[Path] = None,
    alpha: float = 0.05,
    reference_model_name: Optional[str] = None,
    difference_outcome: bool = False,
) -> plt.Figure:
    """Coefplot of ordinal coefficients + derived baseline E[Y], sized for a two-column paper."""
    setup_plot_style()

    coefs, pvalues, labels, conf_ints = _extract_ordinal_rows(ordinal_res, difference_outcome)
    colors, markers, annotated_labels = _coef_styles_and_labels(coefs, pvalues, labels, alpha)

    fit_pad = _FIT_STATS_ROW_PAD if _fit_stats_text(ordinal_res.results) else 0.0
    fig, ax = plt.subplots(figsize=_paper_figsize(len(coefs), fit_pad))
    title = "Ordinal Regression Coefficients"
    if reference_model_name:
        title += f"  (ref: {pretty_model(reference_model_name)})"
    fig.suptitle(title, fontsize=_SUPTITLE_FONTSIZE, fontweight="bold", y=0.995)
    _draw_coefficient_points(
        ax, coefs, pvalues, annotated_labels, colors, markers, conf_ints, alpha,
        title="",
    )
    _annotate_fit_stats(ax, ordinal_res.results)
    # Re-set: _draw_coefficient_points sizes the y range without the fit-stats strip.
    _set_coefficient_ylim(ax, len(coefs), fit_pad)
    plt.tight_layout(rect=(0.0, _bottom_margin_frac(fig), 1.0, _top_margin_frac(fig)))
    _add_coefficient_legend(fig, ax)
    save_figure(fig, save_path, dpi=300, pad_inches=_SAVE_PAD_IN,
                transparent=_TRANSPARENT_BACKGROUND)
    plt.close()
    return fig


def _extract_panel_rows(
    res: ModelResults,
    alpha: float,
    show_intercept: bool,
    difference_outcome: bool,
):
    """Extract coefplot rows from either an OLS or an ordinal fit."""
    if res.model_type == "ordinal":
        return _extract_ordinal_rows(
            res,
            difference_outcome=difference_outcome,
            show_baseline=show_intercept,
        )
    return _extract_ols_rows(res, alpha=alpha, show_intercept=show_intercept)



def _lowest_ink_frac(fig: plt.Figure, axes: Sequence[plt.Axes]) -> float:
    """Bottom of the lowest ink in `axes` — x-tick labels included — as a figure fraction.

    The bands below the axes are sized in advance from type sizes, which is
    what positions the axes, but they are deliberate over-estimates: reserving
    what the numerals *might* need and then hanging the legend off the figure's
    bottom edge leaves a visible gap between the two. Measuring after the draw
    puts whatever goes below the axes at a fixed distance from the numerals
    instead, and the tight crop takes back the unused reserve.
    """
    renderer = fig.canvas.get_renderer()
    y0 = min(ax.get_tightbbox(renderer).y0 for ax in axes)
    return y0 / (fig.get_figheight() * fig.dpi)


def _shrink_text_to_width(fig: plt.Figure, text: str, max_width_in: float,
                          fontsize: float, min_fontsize: float) -> float:
    """Largest size up to `fontsize` at which `text` fits `max_width_in`.

    A direction label sits under one half of one panel, narrow enough that a
    phrase as ordinary as "More favourable in Grokipedia" overruns it and
    collides with its opposite number. Line breaks stay the caller's, since the
    band's reserved height is counted from them.
    """
    renderer = fig.canvas.get_renderer()
    artist = fig.text(0, 0, text, fontsize=fontsize)
    width = artist.get_window_extent(renderer=renderer).width / fig.dpi
    artist.remove()
    if width <= max_width_in:
        return fontsize
    return max(min_fontsize, fontsize * max_width_in / width)


def _draw_direction_band(fig: plt.Figure, ax: plt.Axes,
                         left_label: str, right_label: str,
                         band_top_frac: float,
                         caption: Optional[str] = None,
                         fontsize: float = _DIRECTION_FONTSIZE,
                         scale: float = 1.0,
                         drawn_arrow: bool = True) -> float:
    """Two outward arrows from the zero line, one labelled per direction.

    Anchored at x=0 rather than at the panel's midpoint, so the arrows start
    exactly where the sign flips — the zero rule the reader is already using to
    read the coefficients. Drawn in figure coordinates after the axes are
    positioned, hanging from `band_top_frac` (a figure fraction, measured from
    the bottom of the x-tick labels rather than from a reserved band, so the
    strip sits under the numerals at a fixed distance whatever they occupy).

    `caption` is the wording the two directions share, set once across the whole
    panel above the arrow, leaving each side's label short enough to fit its own
    half of a narrow panel at full size ("More favourable in" over
    "Grokipedia | Wikipedia").

    Returns the figure fraction its lowest ink reaches, so the next thing down
    can hang off the labels actually drawn rather than off a nominal band.
    """
    fig_height = fig.get_figheight()
    pos = ax.get_position()
    lo, hi = ax.get_xlim()
    x_zero_fig = fig.transFigure.inverted().transform(
        ax.transData.transform((min(max(0.0, lo), hi), 0.0))
    )[0]

    # One size for the whole strip: the caption gets the panel's width, each
    # label half of it, and the tightest of those fits decides. Sizing them
    # independently left the caption large and the names under it small.
    fig_width = fig.get_figwidth()
    fitted = [
        _shrink_text_to_width(
            fig, label,
            abs(x_end - x_zero_fig) * fig_width - _DIRECTION_LABEL_GAP_IN * scale,
            fontsize=fontsize, min_fontsize=fontsize * 0.6,
        )
        for x_end, label in ((pos.x0, left_label), (pos.x1, right_label))
        if abs(x_end - x_zero_fig) >= 1e-3
    ]
    if caption:
        fitted.append(_shrink_text_to_width(
            fig, caption, (pos.x1 - pos.x0) * fig_width,
            fontsize=fontsize, min_fontsize=fontsize * 0.6,
        ))
    label_fs = min(fitted, default=fontsize)

    caption_artist = None
    if caption:
        caption_artist = fig.text(
            (pos.x0 + pos.x1) / 2, band_top_frac, caption,
            ha="center", va="top", fontsize=label_fs,
            color=_DIRECTION_TEXT_COLOR,
        )
        band_top_frac = (
            caption_artist.get_window_extent(renderer=fig.canvas.get_renderer()).y0
            / (fig_height * fig.dpi)
        ) - 0.04 * scale / fig_height

    if drawn_arrow:
        arrow_y = band_top_frac - _DIRECTION_ARROW_IN * scale / fig_height
        text_y = arrow_y - 0.09 * scale / fig_height
    else:
        # No arrow row: the direction rides on the labels as a glyph, and the
        # strip costs one line of type instead of a line plus a rule.
        arrow_y = None
        text_y = band_top_frac

    drawn = []
    for x_end, label in ((pos.x0, left_label), (pos.x1, right_label)):
        # Nothing to point at when the zero line sits on that edge of the panel.
        if abs(x_end - x_zero_fig) < 1e-3:
            continue
        if arrow_y is None:
            label = f"\u2190 {label}" if x_end < x_zero_fig else f"{label} \u2192"
            drawn.append(fig.text(
                (x_zero_fig + x_end) / 2, text_y, label,
                ha="center", va="top", fontsize=label_fs,
                color=_DIRECTION_TEXT_COLOR, linespacing=1.25,
            ))
            continue
        fig.add_artist(FancyArrowPatch(
            (x_zero_fig, arrow_y), (x_end, arrow_y),
            transform=fig.transFigure, arrowstyle="-|>",
            mutation_scale=11 * scale, color=_DIRECTION_ARROW_COLOR,
            lw=1.0 * scale, shrinkA=0, shrinkB=0,
            joinstyle="miter", capstyle="butt",
        ))
        drawn.append(fig.text(
            (x_zero_fig + x_end) / 2, text_y, label,
            ha="center", va="top", fontsize=label_fs,
            color=_DIRECTION_TEXT_COLOR, linespacing=1.25,
        ))

    if not drawn:
        return band_top_frac
    renderer = fig.canvas.get_renderer()
    lowest = min(t.get_window_extent(renderer=renderer).y0 for t in drawn)
    return lowest / (fig_height * fig.dpi)


def _drop_hidden_rows(coefs, pvalues, labels, conf_ints, hidden):
    """Filter coefplot rows whose term category is in `hidden`.

    Display-side only: the terms stay in the fit, so every other panel — the
    remaining coefficients, their CIs and the reported R^2 — is exactly the one
    the full model produced. Use it for terms that are estimated as controls
    but are not what the figure is about, e.g. the LLM rater dummies.
    """
    keep = [i for i, lbl in enumerate(labels) if _label_category(lbl) not in hidden]
    return (
        np.array([coefs[i] for i in keep]),
        np.array([pvalues[i] for i in keep]),
        [labels[i] for i in keep],
        [conf_ints[i] for i in keep] if conf_ints is not None else None,
    )


def _widest_label_width_in(fig: plt.Figure, labels: Sequence[str],
                           fontsize: float = _LABEL_FONTSIZE,
                           italic_terms: bool = True) -> float:
    """Width, in inches, of the widest styled y-tick label in `labels`.

    The styled labels are drawn as unclipped annotations, so a left margin that
    is too small does not crop them — it widens the saved figure past
    `fig_width`. With display names long enough for a slide the 1.44 in default
    is no longer enough, so it is measured instead of assumed.
    """
    renderer = fig.canvas.get_renderer()
    widest = 0.0
    for label in labels:
        width = 0.0
        for text, fp, _color in _styled_label_segments(
            label, fontsize, italic_terms=italic_terms,
        ):
            artist = fig.text(0, 0, text, fontproperties=fp)
            width += artist.get_window_extent(renderer=renderer).width
            artist.remove()
        widest = max(widest, width)
    return widest / fig.dpi


def plot_coefficients_grid(
    panels: Sequence[Tuple[str, ModelResults]],
    save_path: Optional[Path] = None,
    alpha: float = 0.05,
    show_intercept: bool = False,
    suptitle: Optional[str] = None,
    shared_x: bool = False,
    difference_outcome: bool = False,
    show_fit_stats: bool = True,
    shared_x_groups: Optional[Sequence[Sequence[int]]] = None,
    group_boundaries: Optional[Sequence[int]] = None,
    show_legend: bool = True,
    extra_formats: Sequence[str] = (),
    label_display: Optional[Mapping[str, str]] = None,
    hide_categories: Sequence[str] = (),
    direction_labels: Optional[Tuple[str, str]] = None,
    direction_caption: Optional[str] = None,
    direction_panel: int = 0,
    direction_placement: str = "below",
    legend_position: str = "below",
    light_ink: bool = False,
    italic_terms: bool = True,
    footnote: Optional[str] = None,
    left_margin_in: Optional[float] = None,
    type_scale: float = 1.0,
    fig_width: Optional[float] = None,
    row_height_in: Optional[float] = None,
) -> plt.Figure:
    """Side-by-side coefplots sharing a y-axis of predictor labels.

    `panels` is an ordered sequence of (panel_title, model_results), rendered
    left-to-right; each panel may be an OLS or an ordinal fit. Predictor labels
    are shown only on the leftmost panel.

    `show_fit_stats=False` drops the per-panel fit annotation. Use it wherever a
    panel would report McFadden's pseudo R^2, which reads as a variance-explained
    figure next to an OLS R^2 without being one.

    `shared_x` ties every panel to one x axis. `shared_x_groups` is the partial
    form: each inner sequence lists panel indices that get a common x range
    (the union of their autoscaled ranges), leaving the panels outside it on
    their own scale. Use it when only some panels are commensurable — the
    per-source coefficients read against each other, the differential does not
    share their units and would only be squeezed by their range.

    `group_boundaries` marks panels that begin a new group: each index gets a
    wider gutter to its left with a dashed rule down it. That is the same
    vocabulary `_style_coefficient_axes` already uses horizontally for the term
    blocks, so a dashed rule reads as "group boundary" throughout the figure.
    The extra gutter is taken out of the panels, not added to the figure, so the
    width does not change.

    `show_legend=False` drops the Positive/Negative/n.s. key and the band of
    height reserved for it, so the figure ends at the x-tick labels. Use it for
    a deck, where the key is a separate build step on the slide — see
    :func:`plot_coefficient_legend`, which writes the same key on its own.

    `extra_formats` writes further siblings of `save_path` (e.g. ``("svg",)``)
    beyond the PNG/PDF pair always written, from the same crop.

    `direction_labels` is a (left, right) pair spelling out what each side of
    the zero line means, drawn as two arrows out of zero in their own band
    below the x-tick labels. Meant for a differential panel, where the sign is
    the whole reading — see :func:`_draw_direction_band`. `direction_panel`
    picks which panel it sits under (the leftmost by default); the band belongs
    to that one axis, so it is only meaningful where a single panel carries the
    differential. `direction_caption` is wording the two share, set once across
    the panel above the arrow, which keeps each side's own label short.

    `type_scale` multiplies the type sizes, the marks and every band of height
    that holds them, leaving the figure's width alone. Above 1 the type grows
    relative to the plot area, which is what a figure read at poster distance
    needs and what a figure set in a paper's column does not. `fig_width`
    overrides the authored width in inches (the panels absorb the difference),
    and `row_height_in` overrides the row pitch, which is what sets the height.
    Type scale and row pitch move together by default; separating them is how a
    figure gets flatter without its labels shrinking, down to the point where
    consecutive labels would touch.

    `direction_placement` is ``"below"`` (its own strip under the x-tick
    labels), ``"one_line"`` (the same strip with the arrow carried as a glyph on
    the labels, one line of type instead of three) or ``"inside_top"`` (in headroom opened inside the panels, under the
    fit-stats line). Inside, it costs panel height instead of figure furniture,
    and the three panels' x axes then form one flush baseline.

    `italic_terms=False` sets the coefficient names upright. The italic is the
    cue separating ideology terms from model and source ones, so it only comes
    off where those groups are not drawn.

    `footnote` is small print set at the bottom right, on the direction strip's
    own line where there is one, so it costs no height. Mark whatever it
    qualifies with the same asterisk it starts with.

    `light_ink` flips chrome and type to light-on-dark for a figure placed on a
    dark board; the data marks are unchanged.

    `legend_position` places the sign key: ``"below"`` on its own line centred
    under the panels; ``"above"`` centred over the panel titles;
    ``"bottom_right"`` in the small-print slot at the bottom right, on the
    direction strip's line and at the footnote's size, costing no height; ``"strip_line"`` centred under the panels but level with
    the direction strip, saving that line's height; ``"label_column"`` in the
    blank block under the y-tick labels, also level with the strip. The last two
    exist because two stacked strips under the axes cost a third of the figure's
    height on a wide figure. `left_margin_in` sets the label column's width in
    inches (it is otherwise measured from the labels), which is what makes room
    for a full-size key in the column.

    `hide_categories` drops whole term groups from the y axis — any of
    ``"ideology"``, ``"control"``, ``"source"``, ``"model"``, ``"interaction"``.
    The model is unchanged: the terms are still estimated and still absorbed by
    the reported R^2, they are simply not drawn. Pass ``("model",)`` for a
    figure about the ideology coefficients, where the LLM rater dummies are
    nuisance terms that only cost rows.

    `label_display` renames y-tick labels at display time only, keyed by the
    short label `pretty_dep_label` produces (e.g. ``{"culincl": "Cultural
    Inclusivity"}``). Row ordering, the term-block rules and the italic/bold
    term-type styling all still key off the short labels, and the left margin
    grows to fit whatever the renames are. Use it for a deck — see
    :data:`SLIDE_IDEOLOGY_LABELS`.
    """
    with _ink(light_ink):
        return _plot_coefficients_grid(
            panels=panels, save_path=save_path, alpha=alpha,
            show_intercept=show_intercept, suptitle=suptitle, shared_x=shared_x,
            difference_outcome=difference_outcome, show_fit_stats=show_fit_stats,
            shared_x_groups=shared_x_groups, group_boundaries=group_boundaries,
            show_legend=show_legend, extra_formats=extra_formats,
            label_display=label_display, hide_categories=hide_categories,
            direction_labels=direction_labels, direction_caption=direction_caption,
            direction_panel=direction_panel,
            direction_placement=direction_placement,
            legend_position=legend_position, left_margin_in=left_margin_in,
            type_scale=type_scale, fig_width=fig_width,
            row_height_in=row_height_in, light_ink=light_ink,
            italic_terms=italic_terms, footnote=footnote,
        )


def _plot_coefficients_grid(
    panels,
    save_path=None,
    alpha=0.05,
    show_intercept=False,
    suptitle=None,
    shared_x=False,
    difference_outcome=False,
    show_fit_stats=True,
    shared_x_groups=None,
    group_boundaries=None,
    show_legend=True,
    extra_formats=(),
    label_display=None,
    hide_categories=(),
    direction_labels=None,
    direction_caption=None,
    direction_panel=0,
    direction_placement="below",
    legend_position="below",
    left_margin_in=None,
    type_scale=1.0,
    fig_width=None,
    row_height_in=None,
    light_ink=False,
    italic_terms=True,
    footnote=None,
) -> plt.Figure:
    """Body of :func:`plot_coefficients_grid`; see there for the arguments."""
    setup_plot_style()
    if light_ink:
        # After the style, not before: seaborn's set_style rewrites the very
        # colour rcParams the light palette is setting. The enclosing
        # rc_context restores them when the figure is done.
        plt.rcParams.update(_LIGHT_INK_RC)

    # Everything that carries or holds type scales together; the plot area is
    # whatever is left of the width. Scaling type alone would run the labels
    # into each other, and scaling the whole figure instead would leave the
    # type-to-plot ratio — the thing that decides legibility at distance —
    # exactly where it started.
    ts = type_scale
    label_fs = _LABEL_FONTSIZE * ts
    tick_fs = _TICK_FONTSIZE * ts
    title_fs = _TITLE_FONTSIZE * ts
    suptitle_fs = _SUPTITLE_FONTSIZE * ts
    legend_fs = _LEGEND_FONTSIZE * ts
    fit_stats_fs = _FIT_STATS_FONTSIZE * ts
    direction_fs = _DIRECTION_FONTSIZE * ts
    row_height_in = row_height_in or _PAPER_ROW_HEIGHT * ts
    xtick_band_in = _XTICK_BAND_IN * ts
    legend_band_in = _LEGEND_BAND_IN * ts
    legend_gap_in = _LEGEND_GAP_IN * ts
    direction_band_in = _direction_band_in(
        direction_labels, direction_caption,
        drawn_arrow=direction_placement != "one_line",
    ) * ts
    title_band_in = _PANEL_TITLE_BAND_IN * ts
    top_band_in = _TOP_BAND_IN * ts

    extracted = [
        (
            title,
            *_extract_panel_rows(
                res, alpha=alpha, show_intercept=show_intercept,
                difference_outcome=difference_outcome,
            ),
        )
        for title, res in panels
    ]

    if hide_categories:
        hidden = set(hide_categories)
        extracted = [
            (title, *_drop_hidden_rows(coefs, pvalues, labels, conf_ints, hidden))
            for title, coefs, pvalues, labels, conf_ints in extracted
        ]

    unified_labels: List[str] = []
    seen = set()
    for _, _, _, labels, _ in extracted:
        for lbl in labels:
            if lbl not in seen:
                seen.add(lbl)
                unified_labels.append(lbl)

    display_labels = [
        (label_display or {}).get(lbl, lbl) for lbl in unified_labels
    ]

    n_rows = len(unified_labels)
    n_panels = len(panels)

    # Inch-based vertical budget, from the same bands as the single-panel
    # coefplot, so the figure-level legend never collides with the x-tick
    # labels regardless of figure height.
    # Unlike the single-panel plot this one positions the axes explicitly via
    # subplots_adjust, so the band has to cover the x-tick labels as well.
    legend_bottom_in = _LEGEND_PAD_IN       # bottom edge of color legend
    legend_reserve_in = (
        legend_bottom_in + legend_band_in + legend_gap_in if show_legend else 0.0
    )
    strip_inside = direction_placement == "inside_top" and direction_labels
    drawn_arrow = direction_placement != "one_line"
    direction_reserve_in = 0.0 if strip_inside else direction_band_in
    bottom_total_in = legend_reserve_in + direction_reserve_in + xtick_band_in
    # Each panel carries a bold title; a suptitle adds one more line above it,
    # and a key set above the titles takes a line of its own.
    top_in = (
        top_band_in + title_band_in
        + (0.28 * ts if suptitle else 0.0)
        + (legend_band_in + legend_gap_in
           if show_legend and legend_position == "above" else 0.0)
    )

    fig_width = fig_width or max(_PAPER_FIG_WIDTH, 2.6 * n_panels + 1.8)
    fit_pad = (
        _FIT_STATS_ROW_PAD
        if show_fit_stats and any(_fit_stats_text(res.results) for _, res in panels)
        else 0.0
    )
    # Inside placement buys its room from the panels: enough rows of headroom
    # for the fit-stats line and the strip, on top of the usual padding.
    fit_stats_drop_in = fit_stats_fs / 72 + 0.10 * ts
    strip_rows = (
        (direction_band_in + fit_stats_drop_in + 0.06 * ts) / row_height_in
        if strip_inside else 0.0
    )
    top_rows = fit_pad + strip_rows
    panel_area_in = _paper_panel_height(
        n_rows, top_rows, row_height=row_height_in,
        min_panel_in=_PAPER_MIN_PANEL_IN * ts,
    )
    fig_height = panel_area_in + bottom_total_in + top_in

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(fig_width, fig_height),
        sharey=True,
        sharex=shared_x,
    )
    if n_panels == 1:
        axes = [axes]

    label_to_y = {lbl: i for i, lbl in enumerate(unified_labels)}

    for ax, (panel_title, coefs, pvalues, labels, conf_ints), (_, panel_res) in zip(axes, extracted, panels):
        colors, markers, _annotated = _coef_styles_and_labels(coefs, pvalues, labels, alpha)
        y_idx = np.array([label_to_y[lbl] for lbl in labels])

        _draw_confidence_intervals(
            ax, coefs, y_idx, conf_ints, colors,
            line_width=_CI_LINE_WIDTH * ts, cap_size=_CI_CAP_SIZE * ts,
            cap_width=_CI_CAP_WIDTH * ts,
        )
        _scatter_by_marker(
            ax, coefs, y_idx, colors, markers,
            s=_POINT_SIZE * ts ** 2, zorder=3, edgecolor=_MARKER_EDGE_COLOR,
            linewidth=0.4 * ts,
        )

        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(display_labels, fontsize=label_fs)
        if ax is axes[0]:  # sharey: only the leftmost panel shows y-labels
            _style_term_ticklabels(
                ax, unified_labels, fontsize=label_fs,
                display_labels=display_labels, italic_terms=italic_terms,
            )
        ax.tick_params(axis="x", labelsize=tick_fs)
        ax.set_title(panel_title, fontweight="bold", fontsize=title_fs, pad=6 * ts)
        if show_fit_stats:
            _annotate_fit_stats(ax, panel_res.results, fontsize=fit_stats_fs)
        _style_coefficient_axes(ax, unified_labels, scale=ts)

    # Common x range per group, before the ylim below: reading get_xlim() is
    # what forces the autoscale whose ranges are being unified here.
    for group in shared_x_groups or []:
        group_axes = [axes[i] for i in group if 0 <= i < n_panels]
        if len(group_axes) < 2:
            continue
        limits = [ax.get_xlim() for ax in group_axes]
        lo, hi = min(l for l, _ in limits), max(h for _, h in limits)
        for ax in group_axes:
            ax.set_xlim(lo, hi)

    axes[0].invert_yaxis()
    _set_coefficient_ylim(axes[0], n_rows, top_rows)  # sharey: the rest follow
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=suptitle_fs, fontweight="bold",
                     y=1 - 0.18 * ts / fig_height)

    # Enough for the longest styled y-tick label ("anti-client", "DeepSeek"
    # bold) at _LABEL_FONTSIZE. These labels are drawn as unclipped annotations,
    # so a margin that is too small does not crop them — it makes the saved
    # figure wider than `fig_width`. Renamed labels are measured, since their
    # length is not known in advance.
    left_in = left_margin_in or 1.44 * ts
    if left_margin_in is None and (label_display or ts != 1.0):
        left_in = max(
            left_in,
            _widest_label_width_in(fig, display_labels, fontsize=label_fs,
                                   italic_terms=italic_terms)
            + _LABEL_MARGIN_PAD_IN * ts,
        )
    if show_legend and legend_position == "label_column":
        # The key lives in this margin, so it is the margin's other lower bound;
        # otherwise it would be shrunk to fit the longest label's width.
        left_in = max(
            left_in,
            _legend_width_in(fig, legend_fs, ts) + 2 * _LABEL_MARGIN_PAD_IN * ts,
        )

    fig.subplots_adjust(
        left=left_in / fig_width,
        right=1 - 0.12 / fig_width,
        bottom=bottom_total_in / fig_height,
        top=1 - top_in / fig_height,
        # Wide enough that adjacent panels' outermost x-tick labels do not
        # touch when neighbouring panels share an x range.
        wspace=0.16,
    )

    # Panel-group boundaries: open a wider gutter before each boundary panel and
    # run a dashed rule down it, the same cue _style_coefficient_axes uses for
    # the horizontal term blocks. The extra gutter is taken out of the panels'
    # widths, so the span from the first panel's left edge to the last panel's
    # right edge — and with it fig_width and the legend centering below — is
    # unchanged.
    boundaries = sorted({b for b in (group_boundaries or []) if 0 < b < n_panels})
    if boundaries:
        pos = [ax.get_position().frozen() for ax in axes]
        left_edge = pos[0].x0
        gutter = pos[1].x0 - pos[0].x1
        extra = _GROUP_GUTTER_IN / fig_width
        width = pos[0].width - extra * len(boundaries) / n_panels
        for i, ax in enumerate(axes):
            shift = extra * sum(1 for b in boundaries if b <= i)
            ax.set_position([
                left_edge + i * (width + gutter) + shift,
                pos[i].y0, width, pos[i].height,
            ])
        # Drawn once the bottom of the figure is settled, below: the rule runs
        # down to whatever the boundary panel's own chrome reaches, and the
        # direction strip is not placed yet.
        boundary_xs = [
            axes[b - 1].get_position().x1 + (gutter + extra) / 2 for b in boundaries
        ]
        panel_y0, panel_y1 = pos[0].y0, pos[0].y1

    # Everything below the axes hangs off the x-tick labels' measured bottom,
    # one gap at a time, so no band's spare reserve turns into a hole.
    gap_frac = _BELOW_AXIS_GAP_IN * ts / fig_height
    below_frac = _lowest_ink_frac(fig, axes)

    rule_bottom_frac = panel_y0 if boundaries else 0.0
    band_top_frac = below_frac - gap_frac
    if direction_labels:
        strip_ax = axes[min(direction_panel, n_panels - 1)]
        if strip_inside:
            # In the headroom opened inside the panel, below the fit-stats line.
            _draw_direction_band(
                fig, strip_ax, *direction_labels,
                band_top_frac=(
                    strip_ax.get_position().y1 - fit_stats_drop_in / fig_height
                ),
                caption=direction_caption, fontsize=direction_fs, scale=ts,
                drawn_arrow=drawn_arrow,
            )
        else:
            below_frac = _draw_direction_band(
                fig, strip_ax, *direction_labels,
                band_top_frac=band_top_frac, caption=direction_caption,
                fontsize=direction_fs, scale=ts, drawn_arrow=drawn_arrow,
            )
            # The strip belongs to its panel, so the group rule runs past the
            # axis and down beside it; stopping at the spine would leave the
            # caption reading as if it sat under the whole figure.
            rule_bottom_frac = below_frac

    if footnote:
        # Bottom right, on the strip's line: small print for a caveat that
        # belongs to the figure rather than to any one panel.
        note = fig.text(
            axes[-1].get_position().x1,
            band_top_frac if direction_labels else below_frac - gap_frac,
            footnote, ha="right", va="top",
            fontsize=direction_fs * _FOOTNOTE_SIZE_RATIO,
            color=_FIT_STATS_COLOR, linespacing=1.3,
        )
        note_bottom = (
            note.get_window_extent(renderer=fig.canvas.get_renderer()).y0
            / (fig_height * fig.dpi)
        )
        below_frac = min(below_frac, note_bottom)

    for x in (boundary_xs if boundaries else []):
        fig.add_artist(Line2D(
            [x, x], [rule_bottom_frac, panel_y1],
            transform=fig.transFigure, color=_GROUP_RULE_COLOR,
            lw=_GROUP_RULE_WIDTH * ts, ls=_GROUP_RULE_STYLE, zorder=0,
        ))

    # Center under the panels' xlabels (between the leftmost and rightmost panel),
    # not the figure — the y-tick labels on the leftmost panel push the plotting
    # area right of figure-center.
    if show_legend:
        panels_x0 = axes[0].get_position().x0
        panels_x1 = axes[-1].get_position().x1
        strip_line_y = (
            (band_top_frac + below_frac) / 2 if direction_labels else band_top_frac
        )
        legend_loc = "center"
        legend_fs_used, marker_scale_used = legend_fs, ts
        if legend_position == "bottom_right":
            # In the small-print slot at the bottom right, on the direction
            # strip's line, so it costs no height of its own.
            legend_loc = "upper right"
            legend_fs_used = legend_fs * _FOOTNOTE_SIZE_RATIO
            marker_scale_used = ts * _FOOTNOTE_SIZE_RATIO
            x_center = panels_x1
            y_center = (
                band_top_frac if direction_labels else below_frac - gap_frac
            )
            max_width_in = (panels_x1 - panels_x0) * fig_width
        elif legend_position == "above":
            # A line of its own above the panel titles: the key reads as the
            # figure's, and the space it leaves below lets the three x axes
            # finish on one flush baseline.
            x_center = (panels_x0 + panels_x1) / 2
            y_center = 1 - (top_band_in + legend_band_in / 2) / fig_height
            max_width_in = (panels_x1 - panels_x0) * fig_width
        elif legend_position == "label_column":
            x_center, y_center = panels_x0 / 2, strip_line_y
            max_width_in = panels_x0 * fig_width - _LABEL_MARGIN_PAD_IN * ts
        elif legend_position == "strip_line":
            x_center, y_center = (panels_x0 + panels_x1) / 2, strip_line_y
            max_width_in = (panels_x1 - panels_x0) * fig_width
        else:
            x_center = (panels_x0 + panels_x1) / 2
            y_center = below_frac - gap_frac - _LEGEND_BAND_IN * ts / 2 / fig_height
            max_width_in = (panels_x1 - panels_x0) * fig_width
        _place_legend_in_width(
            fig, x_center=x_center, y_center=y_center, max_width_in=max_width_in,
            fontsize=legend_fs_used, markerscale=marker_scale_used,
            loc=legend_loc,
        )

    save_figure(fig, save_path, dpi=300, pad_inches=_SAVE_PAD_IN,
                transparent=_TRANSPARENT_BACKGROUND)
    if save_path is not None:
        save_figure(fig, Path(save_path).with_suffix(".pdf"), dpi=300,
                    pad_inches=_SAVE_PAD_IN, transparent=_TRANSPARENT_BACKGROUND)
        for ext in extra_formats:
            save_figure(fig, Path(save_path).with_suffix(f".{ext}"), dpi=300,
                        pad_inches=_SAVE_PAD_IN,
                        transparent=_TRANSPARENT_BACKGROUND)
    plt.close()
    return fig


def plot_overview_for_dep_var(
    df: pd.DataFrame,
    dep_var: str,
    dep_var_label: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Create overview plots for a single dependent variable.

    This figure focuses on panels that vary by the dependent variable.
    Panels that are identical across dependent variables are handled in
    a separate, shared overview figure.
    """
    label = pretty_dep_label(dep_var_label or dep_var)
    setup_plot_style()
    fig = plt.figure(figsize=(16, 8))

    # 1. Boxplot by dep_var
    ax1 = plt.subplot(2, 3, 1)
    if dep_var in df.columns:
        sns.boxplot(data=df, x=dep_var, y=OUTCOME_COLUMN)
        ax1.set_title(f"By {label}", fontweight="bold")
        ax1.set_xlabel(label)
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # 2. Heatmap: Model × dep_var
    ax2 = plt.subplot(2, 3, 2)
    if dep_var in df.columns:
        pivot = df.groupby([MODEL_COLUMN, dep_var], observed=False)[OUTCOME_COLUMN].mean().unstack()
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax2)
        ax2.set_title(f"Model × {label}", fontweight="bold")
        ax2.set_xlabel(label)
        relabel_model_source(ax2)

    # 3. Heatmap: Source × dep_var
    ax3 = plt.subplot(2, 3, 3)
    if dep_var in df.columns:
        pivot = df.groupby([SOURCE_COLUMN, dep_var], observed=False)[OUTCOME_COLUMN].mean().unstack()
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax3)
        ax3.set_title(f"Source × {label}", fontweight="bold")
        ax3.set_xlabel(label)
        relabel_model_source(ax3)

    # 4. Violin: Model × dep_var
    ax4 = plt.subplot(2, 3, 4)
    if dep_var in df.columns:
        sns.violinplot(data=df, x=MODEL_COLUMN, y=OUTCOME_COLUMN, hue=dep_var, ax=ax4, split=True)
        ax4.set_title(f"Violin: Model × {label}", fontweight="bold")
        leg4 = ax4.get_legend()
        if leg4 is not None:
            leg4.set_title(label)
        ax4.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        relabel_model_source(ax4)

    # 5. Point plot with CI
    ax5 = plt.subplot(2, 3, 5)
    if dep_var in df.columns:
        sns.pointplot(data=df, x=MODEL_COLUMN, y=OUTCOME_COLUMN, hue=dep_var, ax=ax5, dodge=True, capsize=0.1)
        ax5.set_title(f"Mean ± CI: Model × {label}", fontweight="bold")
        leg5 = ax5.get_legend()
        if leg5 is not None:
            leg5.set_title(label)
        ax5.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        relabel_model_source(ax5)

    plt.suptitle(f"Overview (DV-specific): {label}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    save_figure(fig, save_path)
    plt.close()



def plot_absolute_bias_for_dep_var(
    df: pd.DataFrame,
    dep_var: str,
    dep_var_label: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Absolute-bias plots for a single dependent variable.

    Mirrors `plot_overview_for_dep_var` but uses |neutrality_assessment| so
    that positive and negative bias do not cancel each other out.
    """
    label = pretty_dep_label(dep_var_label or dep_var)
    setup_plot_style()
    abs_col = "abs_sentiment"
    df = df.copy()
    df[abs_col] = df[OUTCOME_COLUMN].abs()

    fig = plt.figure(figsize=(16, 8))

    # 1. Boxplot by dep_var
    ax1 = plt.subplot(2, 3, 1)
    if dep_var in df.columns:
        sns.boxplot(data=df, x=dep_var, y=abs_col, ax=ax1)
        ax1.set_title(f"Bias magnitude by {label}", fontweight="bold")
        ax1.set_xlabel(label)
        ax1.set_ylabel("|Neutrality Assessment|")

    # 2. Heatmap: Model × dep_var (mean absolute)
    ax2 = plt.subplot(2, 3, 2)
    if dep_var in df.columns:
        pivot = df.groupby([MODEL_COLUMN, dep_var], observed=False)[abs_col].mean().unstack()
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax2)
        ax2.set_title(f"Mean |Sent|: Model × {label}", fontweight="bold")
        ax2.set_xlabel(label)
        relabel_model_source(ax2)

    # 3. Heatmap: Source × dep_var (mean absolute)
    ax3 = plt.subplot(2, 3, 3)
    if dep_var in df.columns:
        pivot = df.groupby([SOURCE_COLUMN, dep_var], observed=False)[abs_col].mean().unstack()
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax3)
        ax3.set_title(f"Mean |Sent|: Source × {label}", fontweight="bold")
        ax3.set_xlabel(label)
        relabel_model_source(ax3)

    # 4. Violin: Model × dep_var
    ax4 = plt.subplot(2, 3, 4)
    if dep_var in df.columns:
        sns.violinplot(data=df, x=MODEL_COLUMN, y=abs_col, hue=dep_var, ax=ax4, split=True)
        ax4.set_title(f"Violin |Sent|: Model × {label}", fontweight="bold")
        ax4.set_ylabel("|Neutrality Assessment|")
        leg4 = ax4.get_legend()
        if leg4 is not None:
            leg4.set_title(label)
        relabel_model_source(ax4)

    # 5. Point plot with CI
    ax5 = plt.subplot(2, 3, 5)
    if dep_var in df.columns:
        sns.pointplot(data=df, x=MODEL_COLUMN, y=abs_col, hue=dep_var, ax=ax5, dodge=True, capsize=0.1)
        ax5.set_title(f"Mean ± CI |Sent|: Model × {label}", fontweight="bold")
        ax5.set_ylabel("|Neutrality Assessment|")
        leg5 = ax5.get_legend()
        if leg5 is not None:
            leg5.set_title(label)
        relabel_model_source(ax5)

    # 6. Bar: mean absolute bias per dep_var group
    ax6 = plt.subplot(2, 3, 6)
    if dep_var in df.columns:
        means = df.groupby(dep_var, observed=False)[abs_col].mean()
        ax6.bar(
            means.index.astype(str), means.values,
            color="coral", alpha=0.8, edgecolor="black",
        )
        for i, (idx, val) in enumerate(means.items()):
            ax6.text(i, val + 0.01, f"{val:.2f}", ha="center", fontsize=9)
        ax6.set_title(f"Mean |Sent| by {label}", fontweight="bold")
        ax6.set_xlabel(label)
        ax6.set_ylabel("|Neutrality Assessment|")
        ax6.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        f"Bias Magnitude (|Neutrality Assessment|): {label}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close()




def plot_wald_tests(ordinal_res: ModelResults, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot Wald test statistics for ordinal model."""
    setup_plot_style()
    n_preds = len(ordinal_res.predictors)
    wald_stats = ordinal_res.results.tvalues[:n_preds].values
    pvals = ordinal_res.results.pvalues[:n_preds].values

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, n_preds * 0.2)))
    fig.suptitle("Wald Tests (Ordinal Model)", fontsize=14, fontweight="bold", y=1.02)

    # Z-stats
    ax1 = axes[0]
    colors = ["C0" if p < 0.05 else "gray" for p in pvals]
    ax1.barh(range(n_preds), wald_stats, color=colors, alpha=0.8)
    ax1.set_yticks(range(n_preds))
    ax1.set_yticklabels(
        [pretty_dep_label(p) for p in ordinal_res.predictors],
        fontsize=7,
    )
    ax1.set_xlabel("Wald z")
    ax1.axvline(x=0, color="black", linewidth=0.8)
    ax1.grid(True, alpha=0.3, axis="x")

    # P-values
    ax2 = axes[1]
    neg_log_p = -np.log10(np.clip(pvals, 1e-10, 1))
    colors = ["red" if p < 0.05 else "gray" for p in pvals]
    ax2.barh(range(n_preds), neg_log_p, color=colors, alpha=0.8)
    ax2.axvline(x=-np.log10(0.05), color="black", linestyle="--", label="α=0.05")
    ax2.set_yticks(range(n_preds))
    ax2.set_yticklabels(
        [pretty_dep_label(p) for p in ordinal_res.predictors],
        fontsize=7,
    )
    ax2.set_xlabel("-log₁₀(p)")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)
    return fig


def plot_marginal_effects(
    ordinal_res: ModelResults,
    outcome_categories: Optional[List] = None,
    save_path: Optional[Path] = None,
    at: str = "overall",
    alpha: float = 0.05,
) -> Optional[plt.Figure]:
    """
    Plot average marginal effects (AME) from the ordinal logit model.

    Left panel: heatmap of dP(Y=j)/dX for each predictor × outcome category.
    Right panel: bar chart of the marginal effect on E[Y] = Σ j·dP(Y=j)/dX.
    """
    if ordinal_res is None or ordinal_res.marginal_effects is None:
        print("No marginal effects available.")
        return None

    me = ordinal_res.marginal_effects

    try:
        margeff = np.array(me.margeff)       # (n_pred, n_cat)
        margeff_se = np.array(me.margeff_se) # (n_pred, n_cat)
        pvalues = np.array(me.pvalues)       # (n_pred, n_cat)
    except AttributeError as exc:
        print(f"Could not extract marginal effects arrays: {exc}")
        return None

    predictor_names = [pretty_dep_label(p) for p in ordinal_res.predictors]
    n_pred = len(predictor_names)

    if margeff.ndim == 1:
        margeff = margeff[:, np.newaxis]
        margeff_se = margeff_se[:, np.newaxis]
        pvalues = pvalues[:, np.newaxis]

    n_cat = margeff.shape[1]
    if outcome_categories is None or len(outcome_categories) != n_cat:
        outcome_categories = list(range(n_cat))
    cat_labels = [str(c) for c in outcome_categories]

    setup_plot_style()
    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, max(8, n_pred * 0.45)),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    fig.suptitle("Average Marginal Effects — Ordinal Logit", fontsize=14, fontweight="bold")

    # ── Left: heatmap dP(Y=j)/dX ─────────────────────────────────────────────
    ax1 = axes[0]
    me_df = pd.DataFrame(margeff, index=predictor_names, columns=cat_labels)
    pv_df = pd.DataFrame(pvalues, index=predictor_names, columns=cat_labels)

    vmax = float(np.abs(margeff).max())
    sns.heatmap(
        me_df, ax=ax1,
        cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
        annot=True, fmt=".3f", linewidths=0.4, linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "dP/dX"},
    )
    # Overlay significance stars
    for i in range(n_pred):
        for j in range(n_cat):
            p = pv_df.iloc[i, j]
            star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            if star:
                ax1.text(
                    j + 0.87, i + 0.82, star,
                    ha="right", va="bottom", fontsize=7, color="black", fontweight="bold",
                )

    ax1.set_title("dP(Y=j) / dX per Outcome Category", fontweight="bold", fontsize=11)
    ax1.set_xlabel("Outcome Category (Likert Score)")
    ax1.set_ylabel("")

    # ── Right: marginal effect on E[Y] = Σ j·dP(Y=j)/dX ────────────────────
    ax2 = axes[1]
    cat_values = np.array([int(c) for c in cat_labels])
    ey_me = margeff @ cat_values                                # (n_pred,)
    ey_se = np.sqrt((margeff_se ** 2) @ (cat_values ** 2))     # delta-method SE

    # Summary p-value: smallest p across categories (optimistic but informative)
    min_pvals = pvalues.min(axis=1)
    bar_colors = [_POSITIVE_COLOR if v > 0 else _NEGATIVE_COLOR for v in ey_me]
    bar_alphas = [1.0 if p < alpha else 0.35 for p in min_pvals]

    for i, (v, c, a) in enumerate(zip(ey_me, bar_colors, bar_alphas)):
        ax2.barh(i, v, color=c, alpha=a, edgecolor="black", linewidth=0.3)
    for i, (v, se, p) in enumerate(zip(ey_me, ey_se, min_pvals)):
        if p < alpha:
            ax2.plot([v - 1.96 * se, v + 1.96 * se], [i, i], color="black", linewidth=1.2)

    ax2.set_yticks(range(n_pred))
    ax2.set_yticklabels(predictor_names, fontsize=9)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_title("ME on E[Y]", fontweight="bold", fontsize=11)
    ax2.set_xlabel("∑ j · (dP/dX)")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.grid(True, alpha=0.3, axis="x", linestyle=":")

    legend_elements = [
        Patch(facecolor=_POSITIVE_COLOR, edgecolor="black", label="Positive effect"),
        Patch(facecolor=_NEGATIVE_COLOR, edgecolor="black", label="Negative effect"),
        Patch(facecolor="#888888", alpha=0.35, edgecolor="black", label=f"p ≥ {alpha}"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center", bbox_to_anchor=(0.5, -0.01),
        ncol=3, fontsize=9, frameon=False,
    )
    fig.text(0.5, 0.005, "* p<0.05, ** p<0.01, *** p<0.001",
             ha="center", fontsize=8, style="italic", alpha=0.7)

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    save_figure(fig, save_path, dpi=300)
    plt.close()
    return fig


def plot_anova(ols_res: ModelResults, save_path: Optional[Path] = None) -> plt.Figure:
    """Plot ANOVA F-stats and p-values."""
    setup_plot_style()
    if ols_res.anova_table is None:
        raise ValueError("No ANOVA table")

    anova = ols_res.anova_table.dropna(subset=["F"])

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(anova) * 0.3)))
    fig.suptitle("ANOVA (OLS Model)", fontsize=14, fontweight="bold", y=1.02)

    # F-stats
    ax1 = axes[0]
    terms = anova.index.tolist()
    pretty_terms = [pretty_dep_label(t) for t in terms]
    f_vals = anova["F"].values
    colors = ["C0" if p < 0.05 else "gray" for p in anova["PR(>F)"]]
    ax1.barh(range(len(terms)), f_vals, color=colors, alpha=0.8)
    ax1.set_yticks(range(len(terms)))
    ax1.set_yticklabels(pretty_terms, fontsize=10)
    ax1.set_xlabel("F")
    ax1.grid(True, alpha=0.3, axis="x")

    # P-values
    ax2 = axes[1]
    neg_log_p = -np.log10(np.clip(anova["PR(>F)"].values, 1e-10, 1))
    colors = ["red" if p < 0.05 else "gray" for p in anova["PR(>F)"]]
    ax2.barh(range(len(terms)), neg_log_p, color=colors, alpha=0.8)
    ax2.axvline(x=-np.log10(0.05), color="black", linestyle="--", label="α=0.05")
    ax2.set_yticks(range(len(terms)))
    ax2.set_yticklabels(pretty_terms, fontsize=10)
    ax2.set_xlabel("-log₁₀(p)")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)
    return fig
