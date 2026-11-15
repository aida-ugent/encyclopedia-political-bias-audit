from itertools import combinations
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from ..config import MODEL_DISPLAY, SCORE_LABELS, SENTIMENT_COLORS
# The light-on-dark palette lives with the coefplots, which is where it is
# defined and kept in step; a poster's figures have to agree on it.
from .stat_figures import _LIGHT_INK_RC, _ink
from .helpers import (
    MODEL_COLUMN,
    OUTCOME_COLUMN,
    SOURCE_COLUMN,
    compute_article_length,
    pretty_dep_label,
    pretty_model,
    pretty_source,
    relabel_model_source,
    save_figure,
    setup_plot_style,
)

SOURCE_WIKI = "wikipedia"
SOURCE_GROKI = "grokipedia_0.2"
SCORE_RANGE = range(-2, 3)


_SHORT_SCORE_LABELS = {
    -2: "Strongly biased against (−2)",
    -1: "Biased against (−1)",
    0:  "Neutral (0)",
    1:  "Biased in favor (+1)",
    2:  "Strongly biased in favor (+2)",
}

# Compact variant for single-row legends at full text width; "against / in
# favor" stays unambiguous under the legend title ("Neutrality score").
_COMPACT_SCORE_LABELS = {
    -2: "Strongly against (−2)",
    -1: "Against (−1)",
    0:  "Neutral (0)",
    1:  "In favor (+1)",
    2:  "Strongly in favor (+2)",
}

# Single-row legend for the narrow preset. Spelling out all five categories
# needs two rows in one column, and that second row costs ~0.09 in of the
# figure's height — the scarcest resource here, since the inline-label test is
# a test on segment height. Naming only the poles and the midpoint keeps the
# scale readable (it is ordinal and evenly spaced, so the intermediate steps
# follow from the ends) and fits on one row, which buys the axes that height
# back and with it two more labelled segments.
# Poster legend: the poles are named, the intermediate steps are not. The scale
# is ordinal and evenly spaced, so "-1" between "Strongly against (-2)" and
# "Neutral (0)" needs no wording of its own — and the two words it drops were
# most of the legend's width, which was most of the figure's.
_POSTER_SCORE_LABELS = {
    -2: "Strongly against (−2)",
    -1: "−1",
    0:  "Neutral (0)",
    1:  "+1",
    2:  "Strongly in favor (+2)",
}

_SCALE_SCORE_LABELS = {
    -2: "−2 strongly against",
    -1: "−1",
    0:  "0 neutral",
    1:  "+1",
    2:  "+2 strongly in favor",
}


def _label_text_color(facecolor) -> str:
    """Black or white, whichever reads better on the given segment color."""
    r, g, b = mcolors.to_rgb(facecolor)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.45 else "black"


def _annotate_stacked_segments(
    ax: plt.Axes,
    value_label_fontsize: int,
    value_label_min: float,
    label_small_outside: bool = True,
    adaptive_fontsize: bool = False,
    skip_smallest_per_bar: bool = False,
    small_label_fontsize: Optional[float] = None,
    value_label_suffix: str = "%",
    min_inline_fontsize: float = 6.5,
    outside_label_floor_pt: float = 0.0,
    outside_room: float = 0.55,
) -> None:
    """Label every non-empty segment of a 100%-stacked bar chart.

    Segments tall enough to hold text get a centered label (white on dark
    fills); thinner segments are labeled outside the bar, to the right, with
    a leader line — stacked apart so neighbouring tiny segments stay legible.
    Set ``label_small_outside=False`` to drop those outside labels entirely
    (useful for narrow, single-column layouts where leader lines would clutter
    the panel or spill into a neighbour).

    ``adaptive_fontsize`` shrinks a segment's inline label (down to
    ``min_inline_fontsize``) so it still fits inside a thin segment, rather
    than pushing it outside — this keeps far more labels in place on narrow
    bars. ``min_inline_fontsize`` is that floor, and it is what buys extra
    labels at no cost in page height: a segment too thin for the authored
    size may still clear the floor. 6pt is the practical floor for print. ``skip_smallest_per_bar`` drops the single smallest segment that
    still would not fit inline; since a 100%-stacked bar sums to 100, that one
    value stays trivially inferable, and it removes the worst leader line.

    ``value_label_suffix`` is appended to each value; pass ``""`` on narrow
    (single-column) layouts, where the "%" glyph alone can be wider than the
    slack between the number and the bar edge and the y-axis label already
    states the unit.
    """
    bars = {}
    for container in ax.containers:
        for rect in container:
            if rect.get_height() > 0.05:
                bars.setdefault(round(rect.get_x(), 6), []).append(rect)

    small_fontsize = (
        small_label_fontsize if small_label_fontsize is not None
        else max(7.0, value_label_fontsize - 2)
    )
    min_inline = min_inline_fontsize
    # Font points -> y-data units (y spans ~103), so stacked outside labels
    # can be spaced to not overlap. Axes geometry is approximate this early
    # in the layout, hence the generous multiplier.
    axes_height_in = ax.get_position().height * ax.figure.get_figheight()
    pts_per_unit = max(axes_height_in * 72.0, 1e-9) / 103.0
    min_gap = 1.8 * small_fontsize / pts_per_unit

    def _inline_fontsize(height: float):
        """Largest font (<= requested) that fits this segment, or None."""
        capacity = height * pts_per_unit / 0.85
        if adaptive_fontsize:
            fs = min(value_label_fontsize, capacity)
            return fs if (fs >= min_inline and height >= value_label_min) else None
        fits = capacity >= value_label_fontsize
        return value_label_fontsize if (fits and height >= value_label_min) else None

    needs_right_room = None
    for x_left, rects in sorted(bars.items()):
        # Optionally drop the single smallest segment that can't be labelled
        # inline, so it becomes the (inferable) omitted value on this bar.
        drop_idx = None
        if skip_smallest_per_bar:
            unfit = [i for i, r in enumerate(rects)
                     if _inline_fontsize(r.get_height()) is None]
            if unfit:
                drop_idx = min(unfit, key=lambda i: rects[i].get_height())

        small = []
        for idx, rect in enumerate(rects):
            if idx == drop_idx:
                continue
            height = rect.get_height()
            y_center = rect.get_y() + height / 2
            fs = _inline_fontsize(height)
            if fs is not None:
                ax.text(
                    rect.get_x() + rect.get_width() / 2, y_center,
                    f"{height:.1f}{value_label_suffix}",
                    ha="center", va="center", fontsize=fs,
                    color=_label_text_color(rect.get_facecolor()),
                )
            else:
                small.append((y_center, height))
        if not small or not label_small_outside:
            continue

        small.sort()
        ys = [y for y, _ in small]
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + min_gap)
        ys[-1] = min(ys[-1], 101.0)
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - min_gap)
        # Lift the lowest label clear of the x axis. A 1-2% tail has its centre
        # almost on the spine, so a label centred on it lands on top of the x
        # tick labels below; the leader line spans the offset instead.
        if ys:
            # Floor given in points and converted here, not in data units: the
            # bar is a percentage scale, so a floor in percent slides closer to
            # the axis as the axes get shorter and the clearance it was tuned
            # for quietly disappears.
            ys[0] = max(ys[0], outside_label_floor_pt / pts_per_unit)
            for i in range(1, len(ys)):
                ys[i] = max(ys[i], ys[i - 1] + min_gap)

        x_right = x_left + rects[0].get_width()
        for (y_center, height), y_label in zip(small, ys):
            ax.annotate(
                f"{height:.1f}{value_label_suffix}",
                xy=(x_right, y_center),
                xytext=(x_right + 0.09, y_label),
                ha="left", va="center", fontsize=small_fontsize, color="black",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="0.35",
                                shrinkA=0, shrinkB=0),
                annotation_clip=False,
            )
        needs_right_room = x_right + outside_room

    # Make sure outside labels on the rightmost bar stay inside the axes.
    if needs_right_room is not None:
        left, right = ax.get_xlim()
        ax.set_xlim(left, max(right, needs_right_room))


def _stacked_score_distribution(
    ax: plt.Axes,
    df: pd.DataFrame,
    group_col: str,
    group_title: str,
    show_ylabel: bool,
    value_label_fontsize: int = 10,
    value_label_min: float = 2.5,
    score_labels: Optional[dict] = None,
    bar_width: float = 0.62,
    label_small_outside: bool = True,
    adaptive_fontsize: bool = False,
    skip_smallest_per_bar: bool = False,
    small_label_fontsize: Optional[float] = None,
    value_label_suffix: str = "%",
    min_inline_fontsize: float = 6.5,
    outside_label_floor_pt: float = 0.0,
    outside_room: float = 0.55,
) -> None:
    crosstab = pd.crosstab(df[group_col], df[OUTCOME_COLUMN], normalize="index") * 100
    for val in SCORE_RANGE:
        if val not in crosstab.columns:
            crosstab[val] = 0
    crosstab = crosstab.reindex(columns=sorted(crosstab.columns))

    if group_col == MODEL_COLUMN:
        ordered = [m for m in MODEL_DISPLAY if m in crosstab.index]
        ordered += [m for m in crosstab.index if m not in ordered]
        crosstab = crosstab.reindex(index=ordered)

    labeled = crosstab.rename(columns=score_labels or SCORE_LABELS)
    print(f"\nScore distribution — {group_title} (% of articles):")
    print(labeled.round(1).to_string())
    labeled.plot(
        kind="bar", stacked=True, ax=ax, color=SENTIMENT_COLORS,
        edgecolor="black", linewidth=0.4, width=bar_width, legend=False,
    )

    _annotate_stacked_segments(
        ax, value_label_fontsize, value_label_min,
        label_small_outside=label_small_outside,
        adaptive_fontsize=adaptive_fontsize,
        skip_smallest_per_bar=skip_smallest_per_bar,
        small_label_fontsize=small_label_fontsize,
        value_label_suffix=value_label_suffix,
        min_inline_fontsize=min_inline_fontsize,
        outside_label_floor_pt=outside_label_floor_pt,
        outside_room=outside_room,
    )

    ax.set_title(group_title, fontweight="bold", fontsize=13, pad=10)
    ax.set_xlabel(group_col, fontsize=12)
    ax.set_ylabel("Share of articles (%)" if show_ylabel else "", fontsize=12)
    ax.set_ylim(0, 103)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(axis="x", which="both", rotation=0, labelsize=11, top=False)
    ax.tick_params(axis="y", which="both", labelsize=10, right=False)
    ax.grid(True, alpha=0.25, axis="y", linestyle=":", linewidth=0.6)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    relabel_model_source(ax)


def _plot_single_score_distribution(
    df: pd.DataFrame,
    group_col: str,
    group_title: str,
    save_path: Optional[Path],
):
    """Paper-ready figure: single 100%-stacked bar chart grouped by `group_col`.

    Sized for a one-column paper (~6.3 in text width): wide and short, with all
    five legend entries on a single row below the axes.
    """
    setup_plot_style()

    fig, ax = plt.subplots(1, 1, figsize=(6.3, 3.4))

    _stacked_score_distribution(
        ax, df, group_col, group_title, show_ylabel=True,
        value_label_fontsize=12, value_label_min=3.0,
        score_labels=_COMPACT_SCORE_LABELS, bar_width=0.45,
    )
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_xticklabels([t.get_text().capitalize() for t in ax.get_xticklabels()])

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="Neutrality score",
        loc="lower center", bbox_to_anchor=(0.5, -0.01),
        ncol=5, fontsize=9, title_fontsize=10, frameon=False,
        columnspacing=0.9, handlelength=1.2, handletextpad=0.4,
    )

    plt.tight_layout(rect=(0.0, 0.20, 1.0, 1.0))
    save_figure(fig, save_path, dpi=300)
    if save_path is not None:
        save_figure(fig, Path(save_path).with_suffix(".pdf"), dpi=300)
    plt.close(fig)


# Layout presets for the combined Source/Model figure. Both target an ACM
# two-column page; they differ only in how much of it the figure claims:
#
#   "column"   3.33 x 2.6 in — fits \columnwidth, include as \begin{figure}
#              + width=\columnwidth. ~2.6 column-inches of page.
#   "textwidth" 7.0 x 3.4 in — spans both columns, include as \begin{figure*}
#              + width=\textwidth. ~7 column-inches of page.
#
# Both are authored at their final printed size, so \includegraphics must not
# rescale them: any width= other than the matching one shrinks the text below
# its authored point size, which is exactly how this figure became illegible.
_SCORE_DIST_LAYOUTS = {
    "column": dict(
        # Width calibrated so the saved file measures 3.333 in — ACM sigconf's
        # \columnwidth — after the bbox_inches="tight" crop and the
        # save_pad_inches border below. \includegraphics[width=\columnwidth]
        # then rescales by 1.00 and the 6/6.5pt text prints at its authored
        # size. Re-measure the output PDF's MediaBox if the fonts or legend
        # change; over-wide output is what silently shrinks the text.
        #
        # Height saves 3.333 x 2.48 in. Two floors sit just below it, both set
        # by Mistral, the bar with the two thinnest labelled segments:
        #   2.403 in — its -2 (3.55%) stops fitting inline and needs a side
        #     label, but it is an INTERIOR bar, so that label lands in the gap
        #     and collides with DeepSeek's inline "4.8". Side labels only work
        #     where the bar's right flank is free (Wikipedia, DeepSeek).
        #   2.659 in — its +2 (3.16%) fits inline. Held until we chose to trade
        #     that one label for the height; it is now the inferable value in
        #     its bar. Raise the height back above 2.659 to print it again.
        figsize=(3.42, 2.42),
        # 6pt is the practical floor for print; ACM captions are 8pt.
        base_fontsize=6.5,
        value_fontsize=6.5,
        # Category names are the width-critical text: six of them share the
        # column, and at 6.5pt "Grokipedia"/"Wikipedia" and "Mistral"/"DeepSeek"
        # end up touching. 6pt buys each pair ~0.05 in of clearance.
        tick_fontsize=6.0,
        # Wider bars than the textwidth preset: at this width the bar has to be
        # thick enough to hold a 5-character value label ("48.5").
        bar_width=0.72,
        # Panel (a) gets slightly more than its 2/6 share of the width so
        # "Grokipedia"/"Wikipedia" clear each other at 6.5pt.
        source_panel_weight=2.5,
        # "%" on every label costs ~0.09 in per label — more than the slack
        # between the number and the bar edge here. The y-axis states the unit.
        value_label_suffix="",
        # A segment too thin for 6.5pt still gets its label if it can hold 6pt,
        # which is the print floor. Together with the bands below this puts the
        # inline cut at 3.52% of the bar, which keeps Mistral's -2 (3.55%) —
        # the thinnest inline label in the figure, and by only 0.03 points, so
        # do not shave the height further without re-checking it. Do not lower
        # the floor either; below 6pt the digits stop being readable in print,
        # and the caption's stated cut-off moves with it.
        min_inline_fontsize=6.0,
        # Poles-only wording so the five entries fit on ONE row; see
        # _SCALE_SCORE_LABELS. legend_ncol must stay 5 to match it.
        score_labels=_SCALE_SCORE_LABELS,
        # Inches the legend hangs below the figure box: with the axes taking
        # 0.86 of the height there is no slack left inside it, and at 0 the
        # legend's caps collided with the descender of "Grokipedia" in the x
        # tick labels. The tight crop reclaims the overhang, so this costs
        # ~0.02 in of height and buys ~1.8pt of ink clearance.
        legend_gap_in=0.080,
        # Push the still-unlabelled tails out to the side on a leader line, all
        # but one per bar (skip_smallest_per_bar). Leaving exactly one out is
        # what makes the figure self-contained: a 100%-stacked bar sums to 100,
        # so a single missing value is recoverable by subtraction. Only
        # Wikipedia and DeepSeek were missing BOTH tails, and so were the only
        # two bars where neither could be recovered; they are the two that get
        # a side label here.
        label_small_outside=True,
        # Lifts DeepSeek's -2 label (segment centre at 0.95% of the bar) off the
        # x axis so it clears the tick labels; the leader line covers the
        # offset. 7.5pt leaves ~1.2pt between the label's box and the
        # "DeepSeek" tick label below it.
        outside_label_floor_pt=7.5,
        # 0.45 of a category slot of extra x range, which is what keeps the
        # rightmost bar's side label inside the crop so the saved file stays at
        # \columnwidth. Less and the label pushes the crop wider (LaTeX then
        # rescales the whole figure); more and panel (a) compresses until
        # "Grokipedia" and "Wikipedia" collide.
        outside_room=0.45,
        legend_ncol=5,
        legend_fontsize=5.5,
        legend_columnspacing=0.5,
        save_pad_inches=0.01,
        # Band under the axes for the 6pt x tick labels, and above it for the
        # panel titles. Both in inches; see the subplots_adjust call. Together
        # they are the only overhead in the height, so the axes get
        # 2.42 - 0.348 = 2.07 in, or 0.86 of the figure. The two-row legend this
        # replaced needed 0.20 of the height on its own, which left 0.75.
        # Lowering bottom_band_in further starts clipping the x tick labels.
        bottom_band_in=0.241,
        top_band_in=0.107,
        margins=dict(left=0.125, right=0.985, wspace=0.08),
    ),
    "textwidth": dict(
        figsize=(7.0, 3.4),
        base_fontsize=9,
        value_fontsize=9,
        tick_fontsize=9,
        bar_width=0.6,
        source_panel_weight=None,  # exact bar-count ratio
        value_label_suffix="%",
        # At 3.4in tall every segment >=3% already clears 9pt, so no floor is
        # needed here; keep it at the authored size.
        min_inline_fontsize=9.0,
        # Room for the full wording on one row at this width.
        score_labels=_COMPACT_SCORE_LABELS,
        legend_gap_in=0.0,
        label_small_outside=False,
        outside_label_floor_pt=0.0,
        outside_room=0.55,
        legend_ncol=5,
        legend_fontsize=9,
        legend_columnspacing=1.0,
        save_pad_inches=None,
        bottom_band_in=0.544,
        top_band_in=0.204,
        margins=dict(left=0.105, right=0.99, wspace=0.06),
    ),
    "poster": dict(
        # Read from a metre or two away, on a dark board, next to the poster's
        # coefplot. Wide and shallow to sit as a band; type sized against the
        # figure's own width rather than a page's column.
        # Six bars and a one-row key; the type is what the width has to serve,
        # not the other way round. Narrower than the poster coefplot on
        # purpose — matching that figure's width only stretched the bars.
        figsize=(9.2, 4.9),
        base_fontsize=19,
        value_fontsize=19,
        # The category names are the width-critical text, six of them sharing
        # the figure: at 19pt "Mistral" and "DeepSeek" touch whatever the panel
        # weights are. Everything else stays at 19.
        tick_fontsize=16,
        bar_width=0.66,
        # More than panel (a)'s 2/6 share of the width: at 19pt "Grokipedia"
        # and "Wikipedia" are each wider than their bar's slot and run into
        # each other on the exact ratio.
        source_panel_weight=2.5,
        # The y-axis states the unit; a "%" on every one of eighteen labels is
        # eighteen repetitions of it.
        value_label_suffix="",
        # A segment thinner than this holds no label at all rather than one
        # that overruns its own band.
        min_inline_fontsize=15.0,
        score_labels=_POSTER_SCORE_LABELS,
        # This is the legend's BOTTOM edge below the figure box, so it has to
        # clear the legend's own height (~0.35 in at 17pt with handles) before
        # it clears anything else; less than that and the key rides up into the
        # x tick labels. The tight crop reclaims the overhang either way.
        legend_gap_in=0.46,
        # No leader lines: at this distance the thin tails read as "almost
        # nothing", which is the whole of what they say, and each rescued
        # decimal costs clarity. The values are in the paper.
        label_small_outside=False,
        outside_label_floor_pt=0.0,
        outside_room=0.12,
        legend_ncol=5,
        # The key is the width-critical element: at 17pt it overran the
        # authored 9.2 in and the tight crop widened the whole figure to 10.5.
        legend_fontsize=14,
        legend_columnspacing=1.0,
        save_pad_inches=0.02,
        bottom_band_in=0.42,
        top_band_in=0.10,
        margins=dict(left=0.105, right=0.995, wspace=0.05),
        # The panels are told apart by their own bar names, so the tags are
        # two more things to read and nothing to learn from.
        panel_tags=False,
        # Dark ink: this one goes on a light area of the board. The background
        # stays transparent either way, so the board shows through.
        light_ink=False,
        transparent=True,
        extra_formats=("svg",),
    ),
}


def plot_score_distribution_by_source_and_model(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    layout: str = "column",
):
    """Combined two-panel figure: score distribution by Source (a) and Model (b).

    Panels sit side by side sharing the y-axis (ticks on the left panel only),
    panel widths proportional to their bar counts, one legend below in score
    order. ``layout`` selects a preset from :data:`_SCORE_DIST_LAYOUTS` —
    ``"column"`` (default, fits \\columnwidth) or ``"textwidth"`` (spans both
    columns); see that table for the matching ``\\includegraphics`` width.

    Label policy: a segment is labelled inline only when it is thick enough to
    hold the label at its authored point size (roughly >= 4% of the bar at
    either preset). The thinner slices — the +-2 categories, ~1-4% — are left
    unlabelled rather than pushed outside the bar on leader lines, which is what
    made the earlier, taller version cluttered. Their exact values are printed
    to stdout by ``_stacked_score_distribution`` and belong in the running text
    or an appendix table.
    """
    try:
        cfg = _SCORE_DIST_LAYOUTS[layout]
    except KeyError:
        raise ValueError(
            f"unknown layout {layout!r}; expected one of "
            f"{sorted(_SCORE_DIST_LAYOUTS)}"
        ) from None

    with _ink(cfg.get("light_ink", False)):
        setup_plot_style()
        if cfg.get("light_ink", False):
            # After the style: seaborn's set_style rewrites these very colours.
            plt.rcParams.update(_LIGHT_INK_RC)
        return _draw_score_distribution_by_source_and_model(df, save_path, cfg)


def _draw_score_distribution_by_source_and_model(df, save_path, cfg):
    """Body of :func:`plot_score_distribution_by_source_and_model`."""
    base_fs = cfg["base_fontsize"]
    n_sources = df[SOURCE_COLUMN].nunique()
    n_models = df[MODEL_COLUMN].nunique()
    source_weight = cfg["source_panel_weight"] or n_sources
    wspace = cfg["margins"]["wspace"]
    fig, axes = plt.subplots(
        1, 2, figsize=cfg["figsize"], sharey=True,
        gridspec_kw={"width_ratios": [source_weight, n_models], "wspace": wspace},
    )
    # Applied here, not after the panel loop: the inline-label fit test measures
    # ax.get_position() to convert segment percentages into points, so it has to
    # run against the final axes box. Left until after plotting, it sized every
    # label against matplotlib's default 0.11-0.88 placeholder — 2.06 in of axes
    # where this preset really renders 2.30 in — and mislabelled the borderline
    # segments in both directions.
    # bottom/top come from the preset in INCHES, converted here. The band under
    # the axes holds the x tick labels and the band above holds the panel
    # titles; both are fixed-size type, so as fractions they only ever agreed
    # with the type at one figure height. Expressing them in inches is what lets
    # figsize's height be changed without the tick labels walking into the
    # legend below them.
    margins = dict(cfg["margins"])
    fig_h = cfg["figsize"][1]
    margins["bottom"] = cfg["bottom_band_in"] / fig_h
    margins["top"] = 1.0 - cfg["top_band_in"] / fig_h
    fig.subplots_adjust(**margins)

    show_tags = cfg.get("panel_tags", True)
    panels = [(SOURCE_COLUMN, "(a)"), (MODEL_COLUMN, "(b)")]
    for ax, (col, panel_label) in zip(axes, panels):
        _stacked_score_distribution(
            ax, df, col, panel_label, show_ylabel=(ax is axes[0]),
            value_label_fontsize=cfg["value_fontsize"], value_label_min=0.0,
            score_labels=cfg["score_labels"], bar_width=cfg["bar_width"],
            label_small_outside=cfg["label_small_outside"],
            adaptive_fontsize=True,
            skip_smallest_per_bar=cfg["label_small_outside"],
            small_label_fontsize=cfg["value_fontsize"],
            value_label_suffix=cfg["value_label_suffix"],
            min_inline_fontsize=cfg["min_inline_fontsize"],
            outside_label_floor_pt=cfg["outside_label_floor_pt"],
            outside_room=cfg["outside_room"],
        )
        ax.set_title("")  # clear the bold center title set by the helper
        if show_tags:
            ax.set_title(panel_label, loc="left", fontsize=base_fs + 1,
                         fontweight="normal", pad=2)
        ax.set_xlabel("")
        # "ratings", not "articles": a panel (a) bar pools every judge's rating
        # of that source (1,394 articles x 4 judges), so its denominator is
        # ratings. Only in panel (b) does each article contribute exactly one
        # row, and "ratings" is correct there too — so it is the honest label
        # for a shared axis.
        ax.set_ylabel(
            "Share of ratings (%)" if ax is axes[0] else "", fontsize=base_fs,
        )
        # Tick labels already come out display-cased via MODEL_DISPLAY /
        # SOURCE_DISPLAY (relabel_model_source), so no .capitalize() here — that
        # would flatten names such as "DeepSeek" to "Deepseek".
        # length=0: the x axis is categorical, so the tick marks add nothing the
        # names don't already say. grid(False): every segment carries its value,
        # which makes y gridlines pure ink.
        ax.tick_params(axis="x", labelsize=cfg["tick_fontsize"], pad=1.5,
                       length=0)
        ax.tick_params(axis="y", labelsize=cfg["tick_fontsize"], pad=1.5)
        ax.grid(False)
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    # Panel (b) keeps its left spine (ticks and labels off) on purpose: it
    # divides the two panels, which matters because they are not the same
    # quantity — (a) pools judges, (b) pools sources.
    axes[1].spines["left"].set_visible(True)

    # Legend in score order (−2 → +2) so it reads as the ordinal scale it is.
    # The title is dropped (the caption carries "neutrality score"). At
    # \textwidth all five entries fit on one row and need no reordering. In a
    # single column they take two rows, and matplotlib fills legend columns
    # top-to-bottom, so the permutation below is what makes the grid read
    # monotonically left-to-right, top-to-bottom:
    #     row 1: −2  −1  0
    #     row 2: +1  +2
    handles, labels = axes[0].get_legend_handles_labels()
    if cfg["legend_ncol"] == 3:
        order = [0, 3, 1, 4, 2]
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]
    fig.legend(
        handles, labels,
        # Anchor expressed in inches below the figure box, converted here.
        # Inches rather than a figure fraction: the fraction meant the clearance
        # scaled with figure height, so shrinking the figure silently walked the
        # legend up into the x tick labels.
        loc="lower center",
        bbox_to_anchor=(0.5, -cfg["legend_gap_in"] / cfg["figsize"][1]),
        ncol=cfg["legend_ncol"], fontsize=cfg["legend_fontsize"], frameon=False,
        columnspacing=cfg["legend_columnspacing"], handlelength=1.1,
        handletextpad=0.35,
        labelspacing=0.25, handleheight=0.8,
    )

    # Gap between the legend and the panels is governed by these two knobs
    # (the figure is saved with bbox_inches="tight", so outer margins are
    # cropped and only the panel↔legend distance shows):
    #   • bbox_to_anchor=(0.5, Y) above — raise Y to move the legend UP toward
    #     the panels (smaller gap); lower it to push the legend down.
    #   • bottom= in the preset — reserved space under the axes; lower it to
    #     pull the panels + x-tick labels down toward the legend.
    pad = cfg["save_pad_inches"]
    transparent = cfg.get("transparent", False)
    save_figure(fig, save_path, dpi=300, pad_inches=pad, transparent=transparent)
    if save_path is not None:
        for ext in ("pdf",) + tuple(cfg.get("extra_formats", ())):
            save_figure(fig, Path(save_path).with_suffix(f".{ext}"), dpi=300,
                        pad_inches=pad, transparent=transparent)
    plt.close(fig)



def plot_score_distribution_by_model(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
):
    """Paper-ready figure: 100%-stacked bar chart of neutrality scores by Model."""
    return _plot_single_score_distribution(
        df,
        MODEL_COLUMN,
        "By Model",
        save_path=save_path,
    )


def plot_score_distribution_by_source(
    df: pd.DataFrame,
    save_path: Optional[Path],
):
    """Paper-ready figure: 100%-stacked bar chart of neutrality scores by Source."""
    return _plot_single_score_distribution(
        df,
        SOURCE_COLUMN,
        "By Source",
        save_path=save_path,
    )


def plot_score_distribution_by_model_source_combo(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
):
    """Paper-ready figure: 100%-stacked bars for each Model × Source combination."""
    setup_plot_style()

    fig, ax = plt.subplots(1, 1, figsize=(12, 6.8))

    combo_col = "Model × Source"
    df_combo = df.assign(**{
        combo_col: (
            df[MODEL_COLUMN].astype(str).map(pretty_model)
            + " · "
            + df[SOURCE_COLUMN].astype(str).map(pretty_source)
        )
    })
    _stacked_score_distribution(
        ax, df_combo, combo_col, "By Model × Source", show_ylabel=True,
    )
    ax.tick_params(axis="x", labelrotation=15)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="Neutrality score",
        loc="lower center", bbox_to_anchor=(0.5, -0.01),
        ncol=5, fontsize=10, title_fontsize=11, frameon=False,
    )

    plt.tight_layout(rect=(0.0, 0.09, 1.0, 0.95))
    save_figure(fig, save_path, dpi=300)
    plt.close(fig)


_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def _compute_neutral_rate_stats(df: pd.DataFrame) -> dict:
    """Compute per-cell neutral rates + per-model, per-source, and interaction tests."""
    work = df.copy()
    work["is_neutral"] = (work[OUTCOME_COLUMN] == 0).astype(int)
    work["_model_s"] = work[MODEL_COLUMN].astype(str)
    work["_source_s"] = work[SOURCE_COLUMN].astype(str)

    models = sorted(work["_model_s"].unique())
    sources = sorted(work["_source_s"].unique())

    rows = []
    for m in models:
        for s in sources:
            cell = work[(work["_model_s"] == m) & (work["_source_s"] == s)]
            n = len(cell)
            k = int(cell["is_neutral"].sum())
            if n:
                rate = k / n
                lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
            else:
                rate, lo, hi = np.nan, np.nan, np.nan
            rows.append({
                "Model": pretty_model(m),
                "Source": pretty_source(s),
                "_model_raw": m,
                "_source_raw": s,
                "n": n,
                "n_neutral": k,
                "neutral_rate": rate,
                "ci_low": lo,
                "ci_high": hi,
            })
    summary = pd.DataFrame(rows)

    def _cell(m: str, s: str) -> pd.Series:
        return summary[
            (summary["_model_raw"] == m) & (summary["_source_raw"] == s)
        ].iloc[0]

    def _two_prop(a_k: int, a_n: int, b_k: int, b_n: int) -> dict:
        if min(a_n, b_n) == 0:
            return {"diff": np.nan, "se": np.nan, "ci_low": np.nan,
                    "ci_high": np.nan, "p_value": np.nan}
        p1, p2 = a_k / a_n, b_k / b_n
        diff = p1 - p2
        se = np.sqrt(p1 * (1 - p1) / a_n + p2 * (1 - p2) / b_n)
        _, pval = proportions_ztest([a_k, b_k], [a_n, b_n])
        return {"diff": diff, "se": se, "ci_low": diff - 1.96 * se,
                "ci_high": diff + 1.96 * se, "p_value": pval}

    per_model_tests = []
    if len(sources) == 2:
        s1, s2 = sources
        for m in models:
            r1, r2 = _cell(m, s1), _cell(m, s2)
            t = _two_prop(int(r1["n_neutral"]), int(r1["n"]),
                          int(r2["n_neutral"]), int(r2["n"]))
            per_model_tests.append({
                "Model": pretty_model(m), "_model_raw": m,
                "comparison": f"{pretty_source(s1)} vs {pretty_source(s2)}",
                **t,
            })

    # Per-source pairwise tests across every unordered model pair (generalises
    # cleanly from 2 to N raters).
    per_source_tests = []
    for m1, m2 in combinations(models, 2):
        for s in sources:
            r1, r2 = _cell(m1, s), _cell(m2, s)
            t = _two_prop(int(r1["n_neutral"]), int(r1["n"]),
                          int(r2["n_neutral"]), int(r2["n"]))
            per_source_tests.append({
                "Source": pretty_source(s), "_source_raw": s,
                "comparison": f"{pretty_model(m1)} vs {pretty_model(m2)}",
                **t,
            })

    interaction = {
        "chi2": np.nan, "dof": np.nan, "p_value": np.nan,
        "diff": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan,
    }
    if len(models) >= 2 and len(sources) >= 2:
        k = {(m, s): int(_cell(m, s)["n_neutral"]) for m in models for s in sources}
        n = {(m, s): int(_cell(m, s)["n"]) for m in models for s in sources}
        # Δ-of-Δs only well-defined for the 2×2 case.
        if len(models) == 2 and len(sources) == 2 and all(v > 0 for v in n.values()):
            m1, m2 = models
            s1, s2 = sources
            p = {key: k[key] / n[key] for key in k}
            diff = (p[(m1, s1)] - p[(m1, s2)]) - (p[(m2, s1)] - p[(m2, s2)])
            variance = sum(p[key] * (1 - p[key]) / n[key] for key in p)
            se = np.sqrt(variance)
            interaction.update({
                "diff": diff, "se": se,
                "ci_low": diff - 1.96 * se, "ci_high": diff + 1.96 * se,
            })
        # χ² contingency test generalises to any model × source table.
        table = np.array([[k[(m, s)] for s in sources] for m in models])
        if table.sum() > 0 and (table > 0).all():
            chi2, pval, dof, _ = chi2_contingency(table, correction=False)
            interaction.update({"chi2": chi2, "dof": dof, "p_value": pval})

    return {
        "models": models,
        "sources": sources,
        "summary": summary,
        "per_model_tests": per_model_tests,
        "per_source_tests": per_source_tests,
        "interaction": interaction,
    }


def _save_neutral_rate_tables(table_path: Path, stats_bundle: dict) -> None:
    table_path = Path(table_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    summary = stats_bundle["summary"]
    summary_cols = ["Model", "Source", "n", "n_neutral", "neutral_rate", "ci_low", "ci_high"]
    summary[summary_cols].to_csv(table_path, index=False)

    interaction = stats_bundle["interaction"]
    test_rows = []
    for t in stats_bundle["per_model_tests"]:
        test_rows.append({
            "test": "per_model", "group": t["Model"], "comparison": t["comparison"],
            "diff": t["diff"], "ci_low": t["ci_low"], "ci_high": t["ci_high"],
            "statistic": np.nan, "dof": np.nan, "p_value": t["p_value"],
        })
    for t in stats_bundle["per_source_tests"]:
        test_rows.append({
            "test": "per_source", "group": t["Source"], "comparison": t["comparison"],
            "diff": t["diff"], "ci_low": t["ci_low"], "ci_high": t["ci_high"],
            "statistic": np.nan, "dof": np.nan, "p_value": t["p_value"],
        })
    test_rows.append({
        "test": "interaction_chi2", "group": "",
        "comparison": "model × source (neutral counts)",
        "diff": interaction["diff"],
        "ci_low": interaction["ci_low"], "ci_high": interaction["ci_high"],
        "statistic": interaction["chi2"], "dof": interaction["dof"],
        "p_value": interaction["p_value"],
    })
    tests_path = table_path.with_name(table_path.stem + "_tests.csv")
    pd.DataFrame(test_rows).to_csv(tests_path, index=False)


def _fmt_group_tests(label_key: str, tests: List[dict]) -> str:
    parts = []
    for t in tests:
        if np.isnan(t["p_value"]):
            parts.append(f"{t[label_key]}: n/a")
        else:
            parts.append(
                f"{t[label_key]} ({t['comparison']}): "
                f"Δ = {t['diff'] * 100:+.1f}pp, p = {t['p_value']:.3g}"
            )
    return "   |   ".join(parts)


def _draw_slope_panel(
    ax: plt.Axes, stats_bundle: dict, annotate_slopes: bool = True,
    legend_above: bool = False,
) -> None:
    summary = stats_bundle["summary"]
    models = stats_bundle["models"]
    sources = stats_bundle["sources"]
    per_model_tests = stats_bundle["per_model_tests"]

    x = np.arange(len(sources))
    x_offsets = np.zeros(max(len(models), 1))

    per_model_by_raw = {t["_model_raw"]: t for t in per_model_tests}

    # Rank models by their mid-line neutral rate so the Δ/p annotations of
    # vertically adjacent lines are pushed in opposite directions (and don't
    # land on top of each other, e.g. Grok vs DeepSeek).
    mid_ys = {}
    for m in models:
        d = summary[summary["_model_raw"] == m].set_index("_source_raw").reindex(sources)
        yv = d["neutral_rate"].to_numpy() * 100
        if len(yv) == 2 and not np.isnan(yv).any():
            mid_ys[m] = (yv[0] + yv[1]) / 2
    y_rank = {m: r for r, m in enumerate(sorted(mid_ys, key=mid_ys.get))}

    def _model_color(raw: str) -> str:
        name = str(raw).lower()
        if "grok" in name:
            return "#1A1A1A"  # xAI black
        if "claude" in name:
            return "#D97757"  # Anthropic clay
        if "deepseek" in name:
            return "#4D6BFE"  # DeepSeek blue
        # Deterministic fallback so additional raters get distinct colours.
        return _PALETTE[hash(name) % len(_PALETTE)]

    for i, m in enumerate(models):
        color = _model_color(m)
        data = summary[summary["_model_raw"] == m].set_index("_source_raw").reindex(sources)
        y = data["neutral_rate"].to_numpy() * 100
        lo = data["ci_low"].to_numpy() * 100
        hi = data["ci_high"].to_numpy() * 100
        xp = x + x_offsets[i]
        ax.errorbar(
            xp, y, yerr=[y - lo, hi - y], capsize=4,
            marker="o", markersize=10, linewidth=2.2, color=color,
            label=pretty_model(m).capitalize(), markerfacecolor=color,
            markeredgecolor="black", markeredgewidth=0.6, zorder=3,
        )
        for point_idx, (xi, yi) in enumerate(zip(xp, y)):
            # Place the left-source percentage label *below* the line to free
            # up horizontal space for the slope-Δ annotation; the right-source
            # label stays above where it doesn't collide with anything.
            if point_idx == 0:
                xytext = (6, -6)
                va = "top"
            else:
                xytext = (6, 6)
                va = "baseline"
            ax.annotate(
                f"{yi:.1f}%", xy=(xi, yi),
                xytext=xytext, textcoords="offset points",
                fontsize=17, color=color, fontweight="bold",
                va=va,
            )

        if annotate_slopes and m in per_model_by_raw and len(sources) == 2:
            t = per_model_by_raw[m]
            if not np.isnan(t["p_value"]):
                mid_x = (x[0] + x[1]) / 2 + x_offsets[i]
                mid_y = (y[0] + y[1]) / 2
                # Left-source percentage labels now sit below the line, so
                # there's room to push every non-Grok slope label well to the
                # left of the line midpoint without overlapping them.
                name_l = str(m).lower()
                if "grok" in name_l:
                    x_text_offset = -8
                elif "deepseek" in name_l:
                    x_text_offset = -150
                else:
                    x_text_offset = -150
                y_offset = 18 if y_rank.get(m, i) % 2 == 0 else -26
                ax.annotate(
                    f"Δ={t['diff'] * 100:+.1f}pp, p={t['p_value']:.3g}",
                    xy=(mid_x, mid_y),
                    xytext=(x_text_offset, y_offset), textcoords="offset points",
                    fontsize=15, color=color,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec=color, lw=0.6, alpha=0.92),
                )

    ax.set_xticks(x)
    ax.set_xticklabels([pretty_source(s).capitalize() for s in sources], fontsize=17)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xlim(-0.5, len(sources) - 0.5)
    ax.set_xlabel("Source", fontsize=18)
    ax.set_ylabel("Neutral rate (score = 0, %)", fontsize=18)
    y_max = np.nanmax(summary["ci_high"].to_numpy() * 100) if len(summary) else 100
    ax.set_ylim(0, min(y_max + 22, 110))

    # Order legend entries by MODEL_DISPLAY (Claude, Grok, DeepSeek), with any
    # unexpected models appended after, regardless of the line-plotting order.
    handles, labels = ax.get_legend_handles_labels()
    display_rank = {pretty_model(m).capitalize(): i for i, m in enumerate(MODEL_DISPLAY)}
    order = sorted(range(len(labels)), key=lambda i: display_rank.get(labels[i], len(display_rank) + i))
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    if legend_above:
        ax.legend(
            handles, labels,
            title="Model", frameon=False,
            loc="lower center", bbox_to_anchor=(0.5, 1.02),
            ncol=max(len(models), 1),
            fontsize=16, title_fontsize=16,
            columnspacing=1.6, handlelength=1.6, handletextpad=0.5,
        )
    else:
        ax.legend(handles, labels, title="Model", frameon=False, loc="upper right",
                  fontsize=16, title_fontsize=16)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_neutral_rate_by_model_source(
    df: pd.DataFrame,
    plot_path: Optional[Path] = None,
    table_path: Optional[Path] = None,
) -> plt.Figure:
    """Grouped-bar view: neutral rate per Model × Source with 95% Wilson CIs.

    Tests:
      (1) Per model, S1 vs S2 — two-proportion z-test.
      (2) Per source, A vs B — two-proportion z-test.
      (3) 2×2 χ² on neutral counts (rows = model, cols = source).
    """
    setup_plot_style()
    stats_bundle = _compute_neutral_rate_stats(df)
    summary = stats_bundle["summary"]
    models = stats_bundle["models"]
    sources = stats_bundle["sources"]
    interaction = stats_bundle["interaction"]

    fig, ax = plt.subplots(figsize=(10, 6.8))
    x_positions = np.arange(len(sources))
    width = 0.36

    for i, m in enumerate(models):
        data = summary[summary["_model_raw"] == m].set_index("_source_raw").reindex(sources)
        heights = data["neutral_rate"].to_numpy() * 100
        lows = data["ci_low"].to_numpy() * 100
        highs = data["ci_high"].to_numpy() * 100
        offset = (i - (len(models) - 1) / 2) * width
        bar_x = x_positions + offset
        ax.bar(
            bar_x, heights, width,
            yerr=[heights - lows, highs - heights], capsize=5,
            label=pretty_model(m), color=_PALETTE[i % len(_PALETTE)],
            edgecolor="black", linewidth=0.5,
        )
        for xi, hi_val, h in zip(bar_x, highs, heights):
            ax.text(xi, hi_val + 1.2, f"{h:.1f}%", ha="center", fontsize=9)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([pretty_source(s) for s in sources])
    ax.set_xlabel("Source", fontsize=12)
    ax.set_ylabel("Neutral rate (score = 0, %)", fontsize=12)
    ax.set_title(
        "Neutral Verdict Rate by Model × Source (95% Wilson CI)",
        fontweight="bold", fontsize=13, pad=12,
    )
    y_max = np.nanmax(summary["ci_high"].to_numpy() * 100) if len(summary) else 100
    ax.set_ylim(0, min(y_max + 18, 110))
    ax.legend(title="Model", frameon=False, loc="upper left")
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    footnote_lines = []
    if stats_bundle["per_model_tests"]:
        footnote_lines.append("Per model — " + _fmt_group_tests("Model", stats_bundle["per_model_tests"]))
    if stats_bundle["per_source_tests"]:
        footnote_lines.append("Per source — " + _fmt_group_tests("Source", stats_bundle["per_source_tests"]))
    if not np.isnan(interaction["p_value"]):
        footnote_lines.append(
            f"Interaction χ²({int(interaction['dof'])}) = {interaction['chi2']:.2f}, "
            f"p = {interaction['p_value']:.3g}"
        )
    fig.text(
        0.5, 0.02, "\n".join(footnote_lines),
        ha="center", fontsize=9, style="italic", alpha=0.85,
    )

    plt.tight_layout(rect=(0, 0.12, 1, 1))
    save_figure(fig, plot_path, dpi=300)
    plt.close(fig)

    if table_path is not None:
        _save_neutral_rate_tables(table_path, stats_bundle)

    return fig


def plot_neutral_rate_slope(
    df: pd.DataFrame,
    plot_path: Optional[Path] = None,
    table_path: Optional[Path] = None,
):
    """Interaction (slope) plot: one line per model across sources, annotated with
    within-model Δ and p. Per-source and χ² interaction tests shown in the footnote.
    """
    setup_plot_style()
    stats_bundle = _compute_neutral_rate_stats(df)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    _draw_slope_panel(ax, stats_bundle, annotate_slopes=True, legend_above=True)
    ax.set_xlabel("")
    if (legend := ax.get_legend()) is not None:
        legend.set_title("")

    plt.tight_layout(rect=(0, 0.04, 1, 0.9))
    if plot_path is not None:
        save_figure(fig, plot_path, dpi=300)
        save_figure(fig, Path(plot_path).with_suffix(".pdf"))
    plt.close(fig)

    if table_path is not None:
        _save_neutral_rate_tables(table_path, stats_bundle)


def save_neutral_rate_abs_score_table(
    df: pd.DataFrame,
    table_path: Path,
) -> pd.DataFrame:
    """Single Model × Source table combining the two bias-magnitude views:
    neutral rate (score = 0, with 95% Wilson CI) and mean |score|.

    Tabular companion to ``plot_neutral_rate_slope`` and
    ``plot_mean_abs_score_heatmap``.
    """
    stats_bundle = _compute_neutral_rate_stats(df)
    summary = stats_bundle["summary"]

    work = df.copy()
    work["_abs"] = work[OUTCOME_COLUMN].abs()
    mean_abs = (
        work.groupby(
            [work[MODEL_COLUMN].astype(str), work[SOURCE_COLUMN].astype(str)],
            observed=False,
        )["_abs"]
        .mean()
        .rename("mean_abs_score")
        .reset_index()
        .rename(columns={MODEL_COLUMN: "_model_raw", SOURCE_COLUMN: "_source_raw"})
    )

    table = summary.merge(mean_abs, on=["_model_raw", "_source_raw"], how="left")

    # Match the heatmap's row order (MODEL_DISPLAY first, then any extras)
    model_order = [m for m in MODEL_DISPLAY if m in table["_model_raw"].unique()]
    model_order += [m for m in table["_model_raw"].unique() if m not in model_order]
    table["_model_rank"] = table["_model_raw"].map({m: i for i, m in enumerate(model_order)})
    table = table.sort_values(["_model_rank", "_source_raw"])

    out_cols = [
        "Model", "Source",
        "n_neutral", "neutral_rate", "ci_low", "ci_high",
        "mean_abs_score",
    ]
    table = table[out_cols]

    # Per-source average across models. CI bounds are left empty: the same
    # articles are rated by every model, so a pooled binomial CI would be
    # misleadingly narrow.
    avg = (
        table.groupby("Source", as_index=False)[["n_neutral", "neutral_rate", "mean_abs_score"]]
        .mean()
        .assign(Model="Average", ci_low=np.nan, ci_high=np.nan)
    )
    table = pd.concat([table, avg[out_cols]], ignore_index=True)

    table_path = Path(table_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)
    return table


_SCORE_LABELS = list(range(-2, 3))


def _agreement_metrics(y1: np.ndarray, y2: np.ndarray, n_bootstrap: int, seed: int) -> dict:
    n = len(y1)
    if n == 0:
        return {
            "n": 0, "exact": np.nan, "adjacent": np.nan,
            "kappa": np.nan, "kappa_lo": np.nan, "kappa_hi": np.nan,
            "kappa_unweighted": np.nan, "mean_abs_diff": np.nan,
            "pearson_r": np.nan, "spearman_rho": np.nan,
        }
    diff = np.abs(y1 - y2)
    exact = float((y1 == y2).mean())
    adjacent = float((diff <= 1).mean())
    mean_abs_diff = float(diff.mean())
    # Correlation alongside κ: r is blind to a rater's constant offset, so a
    # pair that ranks articles identically but sits a scale point apart scores
    # high on r and low on κ. Undefined (nan) if either rater is constant.
    if n < 2 or y1.std() == 0 or y2.std() == 0:
        pearson_r = spearman_rho = np.nan
    else:
        pearson_r = float(np.corrcoef(y1, y2)[0, 1])
        spearman_rho = float(spearmanr(y1, y2).statistic)
    kappa = float(cohen_kappa_score(y1, y2, weights="quadratic", labels=_SCORE_LABELS))
    kappa_unweighted = float(cohen_kappa_score(y1, y2, labels=_SCORE_LABELS))

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        try:
            b = cohen_kappa_score(y1[idx], y2[idx], weights="quadratic", labels=_SCORE_LABELS)
            if not np.isnan(b):
                boots.append(b)
        except ValueError:
            continue
    if boots:
        lo = float(np.percentile(boots, 2.5))
        hi = float(np.percentile(boots, 97.5))
    else:
        lo = hi = np.nan
    return {
        "n": n, "exact": exact, "adjacent": adjacent,
        "kappa": kappa, "kappa_lo": lo, "kappa_hi": hi,
        "kappa_unweighted": kappa_unweighted, "mean_abs_diff": mean_abs_diff,
        "pearson_r": pearson_r, "spearman_rho": spearman_rho,
    }


def _krippendorff_alpha_ordinal(items_by_raters: np.ndarray) -> float:
    """Krippendorff's α with the ordinal distance metric.

    ``items_by_raters`` is shape (n_items, n_raters); missing ratings are NaN.
    Items with fewer than 2 raters are excluded automatically. Returns NaN
    when expected disagreement is zero (e.g. constant data).
    """
    M = np.asarray(items_by_raters, dtype=float)
    if M.size == 0:
        return float("nan")

    valid_vals = M[~np.isnan(M)]
    if valid_vals.size == 0:
        return float("nan")

    categories = sorted(np.unique(valid_vals).tolist())
    K = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}

    coincidence = np.zeros((K, K), dtype=float)
    for row in M:
        row = row[~np.isnan(row)]
        m_i = len(row)
        if m_i < 2:
            continue
        idxs = [cat_idx[v] for v in row]
        for a in range(m_i):
            for b in range(m_i):
                if a == b:
                    continue
                coincidence[idxs[a], idxs[b]] += 1.0 / (m_i - 1)

    n_c = coincidence.sum(axis=1)
    n_total = float(n_c.sum())
    if n_total < 2:
        return float("nan")

    def _delta_ord(c: int, k: int) -> float:
        if c == k:
            return 0.0
        lo, hi = (c, k) if c < k else (k, c)
        between = n_c[lo:hi + 1].sum() - (n_c[lo] + n_c[hi]) / 2.0
        return float(between * between)

    do_sum = 0.0
    de_sum = 0.0
    for c in range(K):
        for k in range(K):
            d = _delta_ord(c, k)
            if d == 0.0:
                continue
            do_sum += coincidence[c, k] * d
            de_sum += n_c[c] * n_c[k] * d

    Do = do_sum / n_total
    De = de_sum / (n_total * (n_total - 1))
    if De == 0.0:
        return float("nan")
    return 1.0 - Do / De


def _suffix_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}__{suffix}{path.suffix}")


def _safe_model_slug(name: str) -> str:
    return name.replace("/", "-").replace(" ", "_")


def _agreement_for_pair(
    work: pd.DataFrame,
    sources: List[str],
    m1: str,
    m2: str,
    n_bootstrap: int,
    seed: int,
) -> Tuple[List[dict], dict]:
    """Compute per-source confusion + metrics for a single (m1, m2) rater pair."""
    per_source = []
    for s in sources:
        sub = work[work["_source_s"] == s]
        paired = (
            sub.pivot_table(
                index="Name", columns="_model_s", values=OUTCOME_COLUMN,
                aggfunc="first", observed=True,
            )
        )
        if m1 not in paired.columns or m2 not in paired.columns:
            continue
        paired = paired.dropna(subset=[m1, m2])
        if paired.empty:
            continue
        y1 = paired[m1].astype(int).to_numpy()
        y2 = paired[m2].astype(int).to_numpy()
        metrics = _agreement_metrics(y1, y2, n_bootstrap=n_bootstrap, seed=seed)
        cm = confusion_matrix(y1, y2, labels=_SCORE_LABELS)
        per_source.append({
            "source_raw": s, "source_pretty": pretty_source(s),
            "y1": y1, "y2": y2, "cm": cm, **metrics,
        })

    pooled: dict = {}
    if per_source:
        pooled_y1 = np.concatenate([d["y1"] for d in per_source])
        pooled_y2 = np.concatenate([d["y2"] for d in per_source])
        pooled = _agreement_metrics(pooled_y1, pooled_y2, n_bootstrap=n_bootstrap, seed=seed)
    return per_source, pooled


def _render_agreement_figure(
    per_source: List[dict],
    m1: str,
    m2: str,
    plot_path: Optional[Path],
) -> Optional[plt.Figure]:
    """Render the per-source confusion-matrix heatmap row for one (m1, m2) pair."""
    if not per_source:
        return None
    n_panels = len(per_source)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 6.6))
    if n_panels == 1:
        axes = [axes]

    for ax, d in zip(axes, per_source):
        cm = d["cm"]
        total = cm.sum()
        cm_pct = cm / total * 100 if total > 0 else cm.astype(float)
        annot = np.empty(cm.shape, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{cm[i, j]}\n{cm_pct[i, j]:.1f}%"
        sns.heatmap(
            cm_pct, annot=annot, fmt="", cmap="Blues", ax=ax,
            xticklabels=_SCORE_LABELS, yticklabels=_SCORE_LABELS,
            cbar_kws={"label": "% of paired articles"},
            linewidths=0.4, linecolor="white",
        )
        ax.set_xlabel(f"{pretty_model(m2)} score", fontsize=11)
        ax.set_ylabel(f"{pretty_model(m1)} score", fontsize=11)
        ax.set_title(
            f"{d['source_pretty']}  (n={d['n']})\n"
            f"exact = {d['exact'] * 100:.1f}%   |   ±1 = {d['adjacent'] * 100:.1f}%\n"
            f"κ = {d['kappa']:.2f} [{d['kappa_lo']:.2f}, {d['kappa_hi']:.2f}]   |   "
            f"mean |Δ| = {d['mean_abs_diff']:.2f}",
            fontsize=10, pad=10,
        )

    fig.suptitle(
        f"Inter-Model Agreement on Ordinal Neutrality Score "
        f"(rows = {pretty_model(m1)}, cols = {pretty_model(m2)})",
        fontsize=13, fontweight="bold", y=0.995,
    )
    plt.tight_layout(rect=(0, 0.02, 1, 0.95))
    if plot_path is not None:
        save_figure(fig, plot_path, dpi=300)
    plt.close(fig)
    return fig


def plot_model_agreement_by_source(
    df: pd.DataFrame,
    plot_path: Optional[Path] = None,
    table_path: Optional[Path] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Optional[plt.Figure]:
    """Per-source agreement between rater pairs on the ordinal neutrality scale.

    Reports per source (and a pooled row) for every unordered model pair:
      - Exact agreement rate
      - Adjacent agreement rate (|Δ| ≤ 1)
      - Cohen's weighted (quadratic) κ with bootstrap 95% CI
      - Mean |Δ|

    With 2 raters, writes one figure to ``plot_path`` (legacy filename). With
    >2 raters, writes one figure per unordered pair, suffixing ``plot_path``
    with ``__{m1}_vs_{m2}``. The CSV at ``table_path`` always contains every
    pair × source combination (plus pooled rows), tagged with ``model_a`` /
    ``model_b`` columns.
    """
    setup_plot_style()

    if "Name" not in df.columns:
        raise ValueError("Expected 'Name' column to identify articles for pairing.")

    work = df.copy()
    work["_model_s"] = work[MODEL_COLUMN].astype(str)
    work["_source_s"] = work[SOURCE_COLUMN].astype(str)

    models = sorted(work["_model_s"].unique())
    sources = sorted(work["_source_s"].unique())
    if len(models) < 2:
        raise ValueError(f"Need at least 2 models, got {len(models)}: {models}")

    pairs = list(combinations(models, 2))
    plot_path = Path(plot_path) if plot_path is not None else None

    rows: List[dict] = []
    last_fig: Optional[plt.Figure] = None
    for m1, m2 in pairs:
        per_source, pooled = _agreement_for_pair(
            work, sources, m1, m2, n_bootstrap=n_bootstrap, seed=seed,
        )
        if plot_path is not None:
            pair_path = (
                plot_path
                if len(pairs) == 1
                else _suffix_path(
                    plot_path,
                    f"{_safe_model_slug(m1)}_vs_{_safe_model_slug(m2)}",
                )
            )
        else:
            pair_path = None
        fig = _render_agreement_figure(per_source, m1, m2, pair_path)
        if fig is not None:
            last_fig = fig

        for d in per_source:
            rows.append({
                "model_a": m1, "model_b": m2,
                "source": d["source_pretty"],
                "n_paired": d["n"],
                "exact_agree_pct": d["exact"] * 100,
                "adjacent_agree_pct": d["adjacent"] * 100,
                "weighted_kappa": d["kappa"],
                "kappa_ci_low": d["kappa_lo"],
                "kappa_ci_high": d["kappa_hi"],
                "mean_abs_diff": d["mean_abs_diff"],
            })
        if pooled:
            rows.append({
                "model_a": m1, "model_b": m2,
                "source": "pooled",
                "n_paired": pooled["n"],
                "exact_agree_pct": pooled["exact"] * 100,
                "adjacent_agree_pct": pooled["adjacent"] * 100,
                "weighted_kappa": pooled["kappa"],
                "kappa_ci_low": pooled["kappa_lo"],
                "kappa_ci_high": pooled["kappa_hi"],
                "mean_abs_diff": pooled["mean_abs_diff"],
            })

    if table_path is not None:
        table_path = Path(table_path)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(table_path, index=False)

    return last_fig


def _inter_rater_summary_by_source(
    work: pd.DataFrame, models: List[str], sources: List[str]
) -> pd.DataFrame:
    """One row per source (plus a pooled row) summarising multi-rater agreement.

    All statistics are genuinely multi-rater (no pair averaging), computed on
    the articles scored by every rater:

      * ``krippendorff_alpha_ordinal`` — chance-corrected ordinal α, the headline
        agreement number, defined for any number of raters.
      * ``exact_agreement_pct`` — share of articles where *all* raters gave the
        identical score.
      * ``within_1_agreement_pct`` — share where all raters fall within a 1-point
        spread (max − min ≤ 1).

    Because the percentages require unanimity, adding a rater can only hold or
    lower them, unlike a pooled-pairwise average.
    """
    def _summary_row(
        label: str, scope_df: pd.DataFrame, index_cols, nonneutral_only: bool = False
    ) -> Optional[dict]:
        pivot = scope_df.pivot_table(
            index=index_cols, columns="_model_s",
            values=OUTCOME_COLUMN, aggfunc="first", observed=True,
        )
        cols = [m for m in models if m in pivot.columns]
        if len(cols) < 2:
            return None
        # Complete cases only: articles scored by every rater.
        full = pivot[cols].dropna()
        if nonneutral_only:
            # Keep articles where at least one rater scored non-neutral (≠ 0).
            full = full[(full != 0).any(axis=1)]
        if full.empty:
            return None

        spread = full.max(axis=1) - full.min(axis=1)
        return {
            "source": label,
            "n_items": len(full),
            "krippendorff_alpha_ordinal": _krippendorff_alpha_ordinal(full.to_numpy()),
            "exact_agreement_pct": float((spread == 0).mean() * 100),
            "within_1_agreement_pct": float((spread <= 1).mean() * 100),
        }

    rows: List[dict] = []
    for s in sources:
        if (row := _summary_row(pretty_source(s), work[work["_source_s"] == s], "Name")) is not None:
            rows.append(row)

    # Pooled across sources: index by (Name, source) so an article scored in
    # both sources stays two distinct items rather than collapsing via "first".
    if (pooled := _summary_row("Pooled", work, ["Name", "_source_s"])) is not None:
        rows.append(pooled)
    # Pooled, restricted to articles flagged non-neutral by ≥1 rater.
    if (pooled_nn := _summary_row(
        "Pooled (≥1 non-neutral)", work, ["Name", "_source_s"], nonneutral_only=True
    )) is not None:
        rows.append(pooled_nn)
    return pd.DataFrame(rows)


def plot_inter_rater_agreement(
    df: pd.DataFrame,
    plot_path: Optional[Path] = None,
    table_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """Multi-rater agreement summary: per-source pairwise weighted κ heatmap.

    For each source, builds a raters × raters matrix of Cohen's quadratic-
    weighted κ (computed on the articles both raters scored) and renders it as
    a heatmap. The panel title reports Krippendorff's ordinal α — the single-
    number multi-rater statistic — together with the rate at which every
    rater agreed exactly. A pooled panel aggregates across sources.

    The CSV at ``table_path`` is a paper-ready wide table: per source, one row
    per rater pair (exact %, ±1 adjacent %, weighted κ, Pearson r and Spearman
    ρ) followed by an "All raters" row holding the n-way exact / ±1 agreement
    and Krippendorff's ordinal α.

    r/ρ sit next to κ because they answer a different question: correlation is
    invariant to a rater's constant offset, so a pair that orders the articles
    the same way but sits a scale point apart scores high on r and low on κ.
    They are pairwise by construction, hence blank on the "All raters" row (as
    κ is), where α is the multi-rater statistic.
    """
    setup_plot_style()

    if "Name" not in df.columns:
        raise ValueError("Expected 'Name' column to identify articles for pairing.")

    work = df.copy()
    work["_model_s"] = work[MODEL_COLUMN].astype(str)
    work["_source_s"] = work[SOURCE_COLUMN].astype(str)

    models = sorted(work["_model_s"].unique())
    sources = sorted(work["_source_s"].unique())
    if len(models) < 2:
        raise ValueError(f"Need at least 2 raters, got {len(models)}: {models}")

    def _panel_for(scope_label: str, scope_df: pd.DataFrame) -> Optional[dict]:
        # Index by (Name, Source) so the pooled panel keeps a leader's
        # Wikipedia and Grokipedia observations as distinct items rather than
        # silently collapsing them via aggfunc="first". For per-source panels
        # the Source level is constant, so this is equivalent to indexing by
        # Name alone.
        pivot = scope_df.pivot_table(
            index=["Name", "_source_s"], columns="_model_s",
            values=OUTCOME_COLUMN, aggfunc="first", observed=True,
        )
        # Restrict to the rater columns we know about, preserving order.
        cols = [m for m in models if m in pivot.columns]
        if len(cols) < 2:
            return None
        pivot = pivot[cols]

        # Pairwise weighted κ on the intersection where both raters scored.
        kappa_mat = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
        n_mat = pd.DataFrame(0, index=cols, columns=cols, dtype=int)
        pair_rows: List[dict] = []
        for m1, m2 in combinations(cols, 2):
            paired = pivot[[m1, m2]].dropna()
            n = len(paired)
            if n == 0:
                continue
            y1 = paired[m1].astype(int).to_numpy()
            y2 = paired[m2].astype(int).to_numpy()
            metrics = _agreement_metrics(y1, y2, n_bootstrap=0, seed=0)
            kappa_mat.loc[m1, m2] = metrics["kappa"]
            kappa_mat.loc[m2, m1] = metrics["kappa"]
            n_mat.loc[m1, m2] = n
            n_mat.loc[m2, m1] = n
            pair_rows.append({
                "scope": scope_label,
                "metric": "pairwise_weighted_kappa",
                "rater_a": pretty_model(m1),
                "rater_b": pretty_model(m2),
                "n_paired": n,
                "value": metrics["kappa"],
                "exact_agree_pct": metrics["exact"] * 100,
                "adjacent_agree_pct": metrics["adjacent"] * 100,
                "pearson_r": metrics["pearson_r"],
                "spearman_rho": metrics["spearman_rho"],
            })
        for m in cols:
            kappa_mat.loc[m, m] = 1.0

        # Multi-rater α uses items rated by ≥2 raters.
        alpha_pivot = pivot.dropna(thresh=2)
        n_alpha_items = len(alpha_pivot)
        alpha = (
            _krippendorff_alpha_ordinal(alpha_pivot.to_numpy())
            if n_alpha_items > 0 else float("nan")
        )

        # N-way agreement: items where every rater scored. Exact = all identical;
        # within-1 = all raters fall in a 1-point band (max − min ≤ 1).
        full = pivot.dropna()
        if len(full) > 0:
            n_full_items = len(full)
            full_agree_pct = float((full.nunique(axis=1) == 1).mean() * 100.0)
            full_within1_pct = float(
                ((full.max(axis=1) - full.min(axis=1)) <= 1).mean() * 100.0
            )
        else:
            n_full_items = 0
            full_agree_pct = float("nan")
            full_within1_pct = float("nan")

        return {
            "scope": scope_label,
            "cols": cols,
            "kappa_mat": kappa_mat,
            "n_mat": n_mat,
            "alpha": alpha,
            "n_alpha_items": n_alpha_items,
            "n_full_items": n_full_items,
            "full_agree_pct": full_agree_pct,
            "full_within1_pct": full_within1_pct,
            "pair_rows": pair_rows,
        }

    panels: List[dict] = []
    for s in sources:
        p = _panel_for(pretty_source(s), work[work["_source_s"] == s])
        if p is not None:
            panels.append(p)
    pooled_panel = _panel_for("pooled", work)
    if pooled_panel is not None:
        panels.append(pooled_panel)

    if not panels:
        return None

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 4.4), squeeze=False)
    axes = axes[0]

    for ax, panel in zip(axes, panels):
        cols = panel["cols"]
        kappa_mat = panel["kappa_mat"]
        n_mat = panel["n_mat"]
        labels = [pretty_model(m) for m in cols]

        annot = np.empty(kappa_mat.shape, dtype=object)
        for i, ri in enumerate(cols):
            for j, rj in enumerate(cols):
                if i == j:
                    annot[i, j] = "—"
                else:
                    k = kappa_mat.loc[ri, rj]
                    n_pair = n_mat.loc[ri, rj]
                    annot[i, j] = (
                        f"{k:.2f}\n(n={n_pair})" if not np.isnan(k) else "n/a"
                    )

        sns.heatmap(
            kappa_mat.astype(float),
            ax=ax,
            cmap="RdYlGn",
            vmin=-0.2, vmax=1.0,
            annot=annot, fmt="",
            xticklabels=labels, yticklabels=labels,
            linewidths=0.6, linecolor="white",
            cbar_kws={"label": "Weighted κ"},
            square=True,
            annot_kws={"fontsize": 9},
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_title(
            f"{panel['scope']}\n"
            f"Krippendorff α(ord) = {panel['alpha']:.2f}   |   "
            f"all-agree = {panel['full_agree_pct']:.1f}% (n={panel['n_full_items']})",
            fontsize=10, pad=8,
        )

    fig.suptitle(
        f"Inter-Rater Agreement on Ordinal Neutrality Score ({len(models)} raters)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if plot_path is not None:
        save_figure(fig, plot_path, dpi=300)
    plt.close(fig)

    if table_path is not None:
        # Paper-ready wide layout: per source, one row per rater pair (exact,
        # ±1 adjacent, weighted κ) followed by an "All raters" row carrying the
        # n-way exact / ±1 agreement and Krippendorff's α. Blank cells (e.g. α
        # on pair rows, κ on the All-raters row) are intentionally left empty.
        rows: List[dict] = []
        for panel in panels:
            source = str(panel["scope"]).capitalize()
            for pr in panel["pair_rows"]:
                rows.append({
                    "source": source,
                    "raters": f"{pr['rater_a']} -- {pr['rater_b']}",
                    "n": pr["n_paired"],
                    "exact_agree_pct": pr["exact_agree_pct"],
                    "adjacent_agree_pct": pr["adjacent_agree_pct"],
                    "weighted_kappa": pr["value"],
                    "pearson_r": pr["pearson_r"],
                    "spearman_rho": pr["spearman_rho"],
                    "krippendorff_alpha": np.nan,
                })
            n_raters = len(panel["cols"])
            rows.append({
                "source": source,
                "raters": f"All {n_raters} raters",
                "n": panel["n_full_items"],
                "exact_agree_pct": panel["full_agree_pct"],
                "adjacent_agree_pct": panel["full_within1_pct"],
                "weighted_kappa": np.nan,
                # r is a pairwise statistic; the n-way row carries α instead.
                "pearson_r": np.nan,
                "spearman_rho": np.nan,
                "krippendorff_alpha": panel["alpha"],
            })
        table_path = Path(table_path)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(table_path, index=False)

        # Compact per-source summary (one row per source): exact / ±1 agreement,
        # Cohen's κ, quadratic-weighted κ and mean |Δ|, pooled over rater pairs.
        summary = _inter_rater_summary_by_source(work, models, sources)
        summary_path = _suffix_path(table_path, "summary")
        summary.to_csv(summary_path, index=False)

    return fig


def _compute_did_results(
    model_df: pd.DataFrame,
    bool_vars: List[str],
) -> pd.DataFrame:
    """Difference-in-differences for each boolean condition for a single model."""
    rows = []
    for bool_col in bool_vars:
        if bool_col not in model_df.columns:
            continue

        def mean_for(flag: int, source: str) -> float:
            mask = (model_df[bool_col] == flag) & (model_df[SOURCE_COLUMN] == source)
            return model_df.loc[mask, OUTCOME_COLUMN].mean()

        diff = (
            (mean_for(1, SOURCE_WIKI) - mean_for(1, SOURCE_GROKI))
            - (mean_for(0, SOURCE_WIKI) - mean_for(0, SOURCE_GROKI))
        )
        rows.append({"bool_column": bool_col.replace("_bool", ""), "neutrality_diff": diff})

    return pd.DataFrame(rows, columns=["bool_column", "neutrality_diff"]).sort_values("neutrality_diff", key=abs, ascending=False)


def _draw_did_panel(ax: plt.Axes, results_df: pd.DataFrame, model: str) -> None:
    diffs = results_df["neutrality_diff"].to_numpy()
    colors = [
        "#2ca02c" if x > 0.3 else "#1f77b4" if x > 0 else
        "#d62728" if x < -0.3 else "#ff7f0e" if x < 0 else "#7f7f7f"
        for x in diffs
    ]
    y_pos = np.arange(len(results_df))

    ax.barh(y_pos, diffs, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Difference-in-Differences (Wiki − Grokipedia)", fontsize=9)
    ax.set_ylabel("Condition", fontsize=9)
    ax.set_title(f"Condition Effect on Bias (DiD) — {pretty_model(model)}", fontweight="bold", fontsize=10)

    max_val = float(np.abs(diffs).max()) if len(diffs) else 1.0
    ax.set_xlim(-max_val * 1.2, max_val * 1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_df["bool_column"], fontsize=8)
    ax.grid(True, axis="x", alpha=0.3, linestyle=":")

    for i, val in enumerate(diffs):
        offset = 0.02 * max_val if val >= 0 else -0.04 * max_val
        ax.text(
            val + offset, i, f"{val:+.2f}",
            va="center", fontsize=8,
            fontweight="bold" if abs(val) > 0.5 else "normal",
        )


def plot_mean_score_by_model_source(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple = (3.3, 2.4),
) -> plt.Figure:
    """Paper-ready heatmap of mean neutrality score per Model × Source.

    Defaults are tuned for a single column of a two-column paper
    (~3.3 in wide, 8 pt body, embedded TrueType fonts).
    """
    setup_plot_style()

    required_cols = [MODEL_COLUMN, SOURCE_COLUMN, OUTCOME_COLUMN]
    if missing := [c for c in required_cols if c not in df.columns]:
        raise ValueError(f"Missing required columns: {missing}")

    pivot = (
        df.groupby([MODEL_COLUMN, SOURCE_COLUMN], observed=False)[OUTCOME_COLUMN]
        .mean()
        .unstack()
    )
    pivot.index = [pretty_model(m) for m in pivot.index]
    pivot.columns = [pretty_source(s) for s in pivot.columns]

    vmax = max(float(pivot.abs().to_numpy().max()), 0.2)

    rc = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.5,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-vmax,
            vmax=vmax,
            ax=ax,
            cbar_kws={"label": "Mean neutrality"},
            annot_kws={"fontsize": 9},
            linewidths=0.4,
            linecolor="white",
        )
        ax.set_xlabel("Source")
        ax.set_ylabel("Model")
        ax.tick_params(axis="x", rotation=0, length=2)
        ax.tick_params(axis="y", rotation=0, length=2)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=7.5, length=2)
        cbar.outline.set_linewidth(0.4)
        cbar.ax.yaxis.label.set_size(8)

        fig.tight_layout(pad=0.3)
        if save_path is not None:
            save_figure(fig, save_path, dpi=300)

    return fig


def plot_mean_abs_score_heatmap(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple = (3.3, 2.4),
) -> plt.Figure:
    """Paper-ready heatmap of mean |neutrality score| per Model × Source.

    Bias *magnitude* (non-negative): higher = more biased. Sibling to
    ``plot_mean_score_by_model_source`` that avoids +/− cancellation.
    """
    setup_plot_style()

    required_cols = [MODEL_COLUMN, SOURCE_COLUMN, OUTCOME_COLUMN]
    if missing := [c for c in required_cols if c not in df.columns]:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    work["_abs"] = work[OUTCOME_COLUMN].abs()
    pivot = (
        work.groupby([MODEL_COLUMN, SOURCE_COLUMN], observed=False)["_abs"]
        .mean()
        .unstack()
    )
    ordered = [m for m in MODEL_DISPLAY if m in pivot.index]
    ordered += [m for m in pivot.index if m not in ordered]
    pivot = pivot.reindex(index=ordered)
    pivot.index = [pretty_model(m) for m in pivot.index]
    pivot.columns = [pretty_source(s) for s in pivot.columns]

    vmax = max(float(pivot.to_numpy().max()), 0.2)

    rc = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.5,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            vmin=0,
            vmax=vmax,
            ax=ax,
            cbar_kws={"label": "Mean |bias score|"},
            annot_kws={"fontsize": 9},
            linewidths=0.4,
            linecolor="white",
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=0, length=2)
        ax.tick_params(axis="y", rotation=0, length=2)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=7.5, length=2)
        cbar.outline.set_linewidth(0.4)
        cbar.ax.yaxis.label.set_size(8)

        fig.tight_layout(pad=0.3)
        if save_path is not None:
            save_figure(fig, save_path, dpi=300)
            save_figure(fig, Path(save_path).with_suffix(".pdf"), dpi=300)

    return fig


_BIAS_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.linewidth": 0.5,
}


def plot_non_neutral_rate_by_model_source(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple = (3.3, 2.6),
) -> plt.Figure:
    """Paper-ready grouped bars: non-neutral rate (score ≠ 0) per Model × Source
    with Wilson 95% CIs.
    """
    setup_plot_style()

    required_cols = [MODEL_COLUMN, SOURCE_COLUMN, OUTCOME_COLUMN]
    if missing := [c for c in required_cols if c not in df.columns]:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    work["_model_s"] = work[MODEL_COLUMN].astype(str)
    work["_source_s"] = work[SOURCE_COLUMN].astype(str)
    work["_nonneutral"] = (work[OUTCOME_COLUMN] != 0).astype(int)

    models = sorted(work["_model_s"].unique())
    sources = sorted(work["_source_s"].unique())

    cells = {}
    for m in models:
        for s in sources:
            cell = work[(work["_model_s"] == m) & (work["_source_s"] == s)]
            n = len(cell)
            k = int(cell["_nonneutral"].sum())
            if n:
                rate = k / n
                lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
            else:
                rate, lo, hi = np.nan, np.nan, np.nan
            cells[(m, s)] = (rate, lo, hi, n)

    with plt.rc_context(_BIAS_RC):
        fig, ax = plt.subplots(figsize=figsize)
        x_positions = np.arange(len(sources))
        width = 0.36

        for i, m in enumerate(models):
            heights = np.array([cells[(m, s)][0] for s in sources]) * 100
            lows = np.array([cells[(m, s)][1] for s in sources]) * 100
            highs = np.array([cells[(m, s)][2] for s in sources]) * 100
            offset = (i - (len(models) - 1) / 2) * width
            bar_x = x_positions + offset
            ax.bar(
                bar_x, heights, width,
                yerr=[heights - lows, highs - heights],
                capsize=2.5, error_kw={"linewidth": 0.6, "ecolor": "black"},
                label=pretty_model(m),
                color=_PALETTE[i % len(_PALETTE)],
                edgecolor="black", linewidth=0.4,
            )
            for xi, h, hi in zip(bar_x, heights, highs):
                ax.text(xi, hi + 1.2, f"{h:.1f}%", ha="center", fontsize=7)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([pretty_source(s) for s in sources])
        ax.set_xlabel("Source")
        ax.set_ylabel("Non-neutral rate (score ≠ 0, %)")
        ax.set_ylim(0, 108)
        ax.legend(title="Model", frameon=False, loc="lower right",
                  handlelength=1.2, handletextpad=0.4, borderpad=0.2)
        ax.grid(True, axis="y", alpha=0.25, linestyle=":", linewidth=0.4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        fig.tight_layout(pad=0.3)
        if save_path is not None:
            save_figure(fig, save_path, dpi=300)

    return fig


def plot_bias_direction_stacked_by_model_source(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize: tuple = (3.5, 2.9),
) -> plt.Figure:
    """Paper-ready 100%-stacked bars: against / neutral / in favor per
    Model × Source combo. Shows direction *and* avoids cancellation.
    """
    setup_plot_style()

    required_cols = [MODEL_COLUMN, SOURCE_COLUMN, OUTCOME_COLUMN]
    if missing := [c for c in required_cols if c not in df.columns]:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    work["_bucket"] = np.where(
        work[OUTCOME_COLUMN] < 0, "Against",
        np.where(work[OUTCOME_COLUMN] > 0, "In favor", "Neutral"),
    )
    work["_combo"] = (
        work[MODEL_COLUMN].astype(str).map(pretty_model)
        + "\n"
        + work[SOURCE_COLUMN].astype(str).map(pretty_source)
    )

    order = ["Against", "Neutral", "In favor"]
    crosstab = pd.crosstab(work["_combo"], work["_bucket"], normalize="index") * 100
    for col in order:
        if col not in crosstab.columns:
            crosstab[col] = 0.0
    crosstab = crosstab[order]

    colors = {"Against": "#C44E52", "Neutral": "#ABABAA", "In favor": "#4C72B0"}

    with plt.rc_context({**_BIAS_RC, "xtick.labelsize": 7}):
        fig, ax = plt.subplots(figsize=figsize)
        crosstab.plot(
            kind="bar", stacked=True, ax=ax,
            color=[colors[c] for c in order],
            edgecolor="black", linewidth=0.4, width=0.7, legend=False,
        )
        for container in ax.containers:
            heights = [rect.get_height() for rect in container]
            ax.bar_label(
                container,
                labels=[f"{h:.0f}%" if h >= 4 else "" for h in heights],
                label_type="center", fontsize=7, color="black",
            )
        ax.set_ylabel("Share of articles (%)")
        ax.set_xlabel("")
        ax.set_ylim(0, 103)
        ax.tick_params(axis="x", rotation=0, length=2)
        ax.tick_params(axis="y", length=2)
        ax.grid(True, axis="y", alpha=0.25, linestyle=":", linewidth=0.4)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(
            title=None, loc="lower center",
            bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
            handlelength=1.2, handletextpad=0.4, columnspacing=1.0,
        )
        fig.tight_layout(pad=0.3)
        if save_path is not None:
            save_figure(fig, save_path, dpi=300)

    return fig


def plot_common_overview(
    df: pd.DataFrame,
    bool_independent_vars: List[str],
    save_path: Optional[Path] = None,
    model_order: Optional[List[str]] = None,
) -> plt.Figure:
    """Overview dashboard: global distributions, Model×Source means, per-model DiD panels."""
    setup_plot_style()

    required_cols = [MODEL_COLUMN, SOURCE_COLUMN, OUTCOME_COLUMN]
    if missing := [c for c in required_cols if c not in df.columns]:
        raise ValueError(f"Missing required columns: {missing}")

    models = model_order or df[MODEL_COLUMN].unique().tolist()
    n_models = len(models)

    n_total_plots = 4 + n_models
    n_cols = 3
    n_rows = int(np.ceil(n_total_plots / n_cols))

    fig = plt.figure(figsize=(18, 6 * n_rows))

    ax1 = plt.subplot(n_rows, n_cols, 1)
    sns.boxplot(data=df, x=MODEL_COLUMN, y=OUTCOME_COLUMN, ax=ax1, palette="Set2", legend=False, hue=MODEL_COLUMN)
    ax1.set_title("Distribution by Model", fontweight="bold", fontsize=11)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax1.set_ylabel("Neutrality Assessment")
    relabel_model_source(ax1)

    ax2 = plt.subplot(n_rows, n_cols, 2)
    sns.boxplot(data=df, x=SOURCE_COLUMN, y=OUTCOME_COLUMN, ax=ax2, palette="Set2", hue=SOURCE_COLUMN, legend=False)
    ax2.set_title("Distribution by Source", fontweight="bold", fontsize=11)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax2.set_ylabel("")
    relabel_model_source(ax2)

    ax3 = plt.subplot(n_rows, n_cols, 3)
    counts = df[OUTCOME_COLUMN].value_counts().sort_index()
    norm = plt.Normalize(vmin=-2, vmax=2)
    colors = [plt.cm.RdYlGn(norm(x)) for x in counts.index]
    ax3.bar(counts.index, counts.values, color=colors, alpha=0.8, edgecolor="black")
    ax3.set_title("Overall Score Distribution", fontweight="bold", fontsize=11)
    ax3.set_xlabel("Neutrality Assessment")
    ax3.set_ylabel("Count")
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    ax4 = plt.subplot(n_rows, n_cols, 4)
    pivot = df.groupby([MODEL_COLUMN, SOURCE_COLUMN], observed=False)[OUTCOME_COLUMN].mean().unstack()
    vmax = max(pivot.abs().to_numpy().max(), 0.2)
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-vmax, vmax=vmax,
                ax=ax4, cbar_kws={"label": "Mean Neutrality"})
    ax4.set_title("Mean Score: Model × Source", fontweight="bold", fontsize=11)
    relabel_model_source(ax4)

    for idx, model in enumerate(models):
        model_df = df[df[MODEL_COLUMN] == model]
        results_df = _compute_did_results(model_df, bool_independent_vars)
        ax = plt.subplot(n_rows, n_cols, 5 + idx)
        _draw_did_panel(ax, results_df, model)

    plt.suptitle(
        "Neutrality Assessment Overview Dashboard",
        fontsize=16, fontweight="bold", y=0.995,
    )

    fig.text(
        0.5, 0.01,
        "Effect = (Wiki−Groki when condition=1) − (Wiki−Groki when condition=0).",
        ha="center", fontsize=8, style="italic", alpha=0.7,
    )

    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    save_figure(fig, save_path)
    plt.close()


def plot_absolute_bias_common(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Overview plots using |neutrality_assessment| to show bias magnitude."""
    setup_plot_style()
    abs_col = "abs_sentiment"
    df = df.copy()
    df[abs_col] = df[OUTCOME_COLUMN].abs()

    fig = plt.figure(figsize=(16, 10))

    ax1 = plt.subplot(2, 3, 1)
    sns.boxplot(data=df, x=MODEL_COLUMN, y=abs_col, ax=ax1)
    ax1.set_title("Bias magnitude by Model", fontweight="bold")
    ax1.set_ylabel("|Neutrality Assessment|")
    relabel_model_source(ax1)

    ax2 = plt.subplot(2, 3, 2)
    sns.boxplot(data=df, x=SOURCE_COLUMN, y=abs_col, ax=ax2)
    ax2.set_title("Bias magnitude by Source", fontweight="bold")
    ax2.set_ylabel("|Neutrality Assessment|")
    relabel_model_source(ax2)

    ax3 = plt.subplot(2, 3, 3)
    counts = df[abs_col].value_counts().sort_index()
    ax3.bar(counts.index, counts.values, color="steelblue", alpha=0.7, edgecolor="black")
    ax3.set_title("|Sentiment| Distribution", fontweight="bold")
    ax3.set_xlabel("|Neutrality Assessment|")
    ax3.grid(True, alpha=0.3, axis="y")

    ax4 = plt.subplot(2, 3, 4)
    pivot = df.groupby([MODEL_COLUMN, SOURCE_COLUMN], observed=False)[abs_col].mean().unstack()
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax4)
    ax4.set_title("Mean |Sentiment|: Model × Source", fontweight="bold")
    relabel_model_source(ax4)

    ax5 = plt.subplot(2, 3, 5)
    means = df.groupby(MODEL_COLUMN, observed=False)[abs_col].mean().sort_values(ascending=False)
    ax5.bar(means.index, means.values, color="coral", alpha=0.8, edgecolor="black")
    ax5.set_title("Mean |Sentiment| by Model", fontweight="bold")
    ax5.set_ylabel("|Neutrality Assessment|")
    ax5.tick_params(axis="x", rotation=45)
    ax5.grid(True, alpha=0.3, axis="y")
    relabel_model_source(ax5)

    ax6 = plt.subplot(2, 3, 6)
    combo_data = (
        df.groupby([MODEL_COLUMN, SOURCE_COLUMN], observed=False)[abs_col]
        .mean()
        .reset_index()
    )
    combo_data["Combo"] = (
        combo_data[MODEL_COLUMN].astype(str).map(pretty_model)
        + " / "
        + combo_data[SOURCE_COLUMN].astype(str).map(pretty_source)
    )
    combo_data = combo_data.sort_values(abs_col)
    colors_map = ["red" if x < 0 else "green" if x > 0 else "gray" for x in combo_data[abs_col]]
    ax6.barh(range(len(combo_data)), combo_data[abs_col], color=colors_map, alpha=0.7)
    ax6.set_yticks(range(len(combo_data)))
    ax6.set_yticklabels(combo_data["Combo"], fontsize=8)
    ax6.set_xlabel("Mean |Sentiment| Score")
    ax6.set_title("Mean |Sentiment| by Model × Source", fontsize=11, fontweight="bold")
    ax6.axvline(x=0, color="black", linestyle="-", linewidth=0.8)
    ax6.grid(True, alpha=0.3, axis="x")

    plt.suptitle(
        "Bias Magnitude (|Neutrality Assessment|): Common Overview",
        fontsize=14, fontweight="bold", y=1.02,
    )

    save_figure(fig, save_path)
    plt.close()


def _plot_paired_heatmaps(
    df: pd.DataFrame,
    groups: List[str],
    save_path: Path,
    *,
    group_col: str,
    pivot_col: str,
    x_key: str,
    y_key: str,
    suptitle: Optional[str],
    title_pretty: Callable[[str], str],
    cbar_label: str = "Number of Articles",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    also_save_pdf: bool = False,
    font_scale: float = 1.0,
) -> None:
    """Side-by-side contingency heatmaps of paired neutrality scores."""
    title_fs = 14 * font_scale
    label_fs = 12 * font_scale
    tick_fs = 10 * font_scale
    annot_fs = 10 * font_scale
    cbar_label_fs = 12 * font_scale
    cbar_tick_fs = 10 * font_scale
    legend_fs = 8 * font_scale

    n_groups = max(len(groups), 1)
    fig, axes = plt.subplots(1, n_groups, figsize=(8 * n_groups, 7))
    if n_groups == 1:
        axes = [axes]
    if suptitle:
        fig.suptitle(suptitle, fontsize=18 * font_scale, fontweight="bold", y=1.02)

    for ax_idx, group in enumerate(groups):
        subset = df[df[group_col] == group]
        pivoted = subset.pivot_table(
            index="name",
            columns=pivot_col,
            values=OUTCOME_COLUMN,
            aggfunc="first",
            observed=False,
        )
        if x_key not in pivoted.columns or y_key not in pivoted.columns:
            continue
        pivoted = pivoted.dropna(subset=[x_key, y_key])

        heatmap_data = pd.crosstab(pivoted[y_key], pivoted[x_key])
        heatmap_data = heatmap_data.reindex(sorted(heatmap_data.index))
        heatmap_data = heatmap_data[sorted(heatmap_data.columns)]

        sns.heatmap(
            heatmap_data,
            ax=axes[ax_idx],
            cmap="Blues",
            annot=True,
            fmt="d",
            linewidths=0.5,
            linecolor="gray",
            cbar_kws={"label": cbar_label},
            annot_kws={"fontsize": annot_fs},
        )
        axes[ax_idx].set_title(
            title_pretty(group).capitalize(), fontweight="bold", pad=15, fontsize=title_fs,
        )
        if xlabel is not None:
            axes[ax_idx].set_xlabel(xlabel, fontsize=label_fs)
        if ylabel is not None:
            axes[ax_idx].set_ylabel(ylabel, fontsize=label_fs)
        axes[ax_idx].tick_params(axis="both", labelsize=tick_fs)
        cbar = axes[ax_idx].collections[0].colorbar
        if cbar is not None:
            cbar.ax.yaxis.label.set_size(cbar_label_fs)
            cbar.ax.tick_params(labelsize=cbar_tick_fs)
        axes[ax_idx].plot(
            [-0.5, 4.5], [-0.5, 4.5],
            "k--", linewidth=1, alpha=0.3, label="Perfect agreement",
        )
        axes[ax_idx].legend(loc="upper right", fontsize=legend_fs)

    save_figure(fig, save_path)
    if also_save_pdf:
        save_figure(fig, Path(save_path).with_suffix(".pdf"))
    plt.close(fig)


def plot_ideology_correlation_matrix(
    df: pd.DataFrame,
    ideology_vars: List[str],
    save_path: Path,
    method: str = "pearson",
) -> Optional[plt.Figure]:
    """Heatmap of pairwise correlations between ideology scores.

    Dedupes to one row per leader before correlating, since ideology scores
    are leader-level and repeat across Model × Source rows.
    """
    setup_plot_style()
    cols = [c for c in ideology_vars if c in df.columns]
    if len(cols) < 2:
        print(f"Skipping correlation matrix: need >=2 ideology columns, got {cols}")
        return None

    id_col = "name" if "name" in df.columns else "Name"
    work = df.drop_duplicates(subset=[id_col])[cols].apply(pd.to_numeric, errors="coerce")
    corr = work.corr(method=method)

    labels = [pretty_dep_label(c) for c in corr.columns]
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 2, 0.9 * len(cols) + 1.5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1, vmax=1, center=0,
        square=True,
        cbar_kws={"label": f"{method.capitalize()} r"},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(f"Ideology score correlations ({method}, n={len(work.dropna())})")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)
    return fig


def plot_per_model_figures(df: pd.DataFrame, models: List[str], save_path: Path) -> None:
    _plot_paired_heatmaps(
        df, models, save_path,
        group_col=MODEL_COLUMN,
        pivot_col=SOURCE_COLUMN,
        x_key=SOURCE_WIKI,
        y_key=SOURCE_GROKI,
        suptitle=None,
        title_pretty=pretty_model,
        xlabel="Wikipedia",
        ylabel="Grokipedia",
        also_save_pdf=True,
        font_scale=1.5,
    )


def plot_per_source_figures(df: pd.DataFrame, sources: List[str], save_path: Path) -> None:
    """Per-source contingency heatmaps for every unordered model pair.

    With 2 raters, writes one file at ``save_path`` (legacy behaviour). With
    >2 raters, writes one file per pair, suffixing ``save_path`` with
    ``__{m1}_vs_{m2}``.
    """
    save_path = Path(save_path)
    models = sorted(df[MODEL_COLUMN].dropna().astype(str).unique())
    if len(models) < 2:
        return
    pairs = list(combinations(models, 2))
    for m1, m2 in pairs:
        pair_path = (
            save_path
            if len(pairs) == 1
            else _suffix_path(
                save_path,
                f"{_safe_model_slug(m1)}_vs_{_safe_model_slug(m2)}",
            )
        )
        _plot_paired_heatmaps(
            df, sources, pair_path,
            group_col=SOURCE_COLUMN,
            pivot_col=MODEL_COLUMN,
            x_key=m1,
            y_key=m2,
            suptitle=None,
            title_pretty=pretty_source,
            cbar_label="Number of Articles",
            xlabel=pretty_model(m1),
            ylabel=pretty_model(m2),
            also_save_pdf=True,
            font_scale=1.5,
        )


def plot_outcome_vs_article_length_by_source(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
    source: str = "grokipedia",
    source_text: str = "grokipedia_0_2",
    model: str = "gpt-4",
) -> Optional[plt.Figure]:
    """Outcome against article length with a fitted line per source."""
    setup_plot_style()
    work = df[(df[SOURCE_COLUMN] == source) & (df[MODEL_COLUMN] == model)].copy()
    work["article_length"] = work[source_text].apply(compute_article_length)
    plot_df = work.dropna(subset=[OUTCOME_COLUMN, SOURCE_COLUMN, "article_length"])

    if plot_df.empty:
        print("No data to plot.")
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=plot_df, x="article_length", y=OUTCOME_COLUMN, ax=ax, alpha=0.6)
    sns.regplot(
        data=plot_df, x="article_length", y=OUTCOME_COLUMN,
        scatter=False, ax=ax, line_kws={"color": "red", "lw": 2},
    )

    ax.set_title(f"{OUTCOME_COLUMN} vs Article Length ({source})")
    ax.set_xlabel("Article Length")
    ax.set_ylabel(OUTCOME_COLUMN)

    save_figure(fig, save_path)
    plt.close(fig)
