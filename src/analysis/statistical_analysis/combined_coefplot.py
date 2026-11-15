#!/usr/bin/env python3
"""Single three-panel coefficient figure for the paper.

Replaces the two separate paper figures (the differential coefplot and the
two-panel per-source grid) with one full-width figure of three panels, ordered
differential | Wikipedia | Grokipedia. Same specifications as
:mod:`main`, refitted here so the figure cannot drift from them.

    python -m src.analysis.statistical_analysis.combined_coefplot
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from .config import (
    CLUSTER_COLUMN, CONTINUOUS_INDEPENDENT_VARS, DEFAULT_LEADERS_PATH,
    PER_SOURCE_BASELINE_MODEL,
)
from .data_loader import load_and_prepare_data
from .run_identification import build_differential
from .stat_models import fit_all_models
from .visualisation.helpers import SLIDE_IDEOLOGY_LABELS
from .visualisation.stat_figures import (
    plot_coefficient_legend, plot_coefficients_grid,
)

# Δy = Wikipedia − Grokipedia (see :func:`run_identification.build_differential`)
# on a −2 "biased against" .. +2 "biased in favour" outcome, so a term pushing
# the differential right makes the politician read more favourably on
# Wikipedia, and one pushing it left more favourably on Grokipedia.
# The shared wording is set once across the panel, so each side's own label is
# one word and fits its half of the axis at full size.
DELTA_DIRECTION_CAPTION = "More favourable in"
DELTA_DIRECTION_LABELS = ("Grokipedia", "Wikipedia")

# The poster figure is authored at this width and read from a metre or two
# away, not held at reading distance, so its type is set larger relative to the
# plot area than the paper's. Vector output, so the poster layout is free to
# place it at any size; what is fixed here is the ratio.
POSTER_FIG_WIDTH_IN = 13.0
POSTER_TYPE_SCALE = 1.7
# Rows tighter than the type scale would make them, so the figure stays a
# landscape band rather than growing into a square as the type grows. Roughly
# 1.4 label heights, which keeps consecutive names clearly apart.
POSTER_ROW_HEIGHT_IN = 0.28
# Bottom right, level with the direction strip and set at the small-print size,
# so the key costs no height. Under the label column it looked detached from
# the figure; level with the strip but centred it landed on the arrow.
POSTER_LEGEND_POSITION = "bottom_right"

# One x axis serves all three panels, which is what makes them comparable at a
# glance, but the differential is a difference of scores and the other two are
# levels. Said once, in small print, rather than left for the reader to catch.
POSTER_FOOTNOTE = (
    "* Differential is a difference of scores; the source panels are levels.\n"
    "  One shared x axis, so the three panels' zero lines align."
)

PANEL_ORDER = ["delta", "wikipedia", "grokipedia_0.2"]
PANEL_TITLES = {
    "delta": r"Differential ($\Delta y$)",
    "wikipedia": "Wikipedia",
    "grokipedia_0.2": "Grokipedia",
}


def build_panels(df, ideology_vars: List[str], cluster_col: Optional[str]):
    fits = {}

    fits["delta"] = fit_all_models(
        build_differential(df),
        independent_vars=ideology_vars,
        cluster_col=cluster_col,
        model_col="Model",
        use_source=False,
        use_model=False,
    )

    for source in df["Source"].unique():
        sub = df[df["Source"] == source].copy()
        cats = list(sub["Model"].cat.categories)
        if PER_SOURCE_BASELINE_MODEL in cats:
            sub["Model"] = sub["Model"].cat.reorder_categories(
                [PER_SOURCE_BASELINE_MODEL]
                + [m for m in cats if m != PER_SOURCE_BASELINE_MODEL]
            )
        fits[source] = fit_all_models(
            sub,
            independent_vars=ideology_vars,
            cluster_col=cluster_col,
            model_col="Model",
            use_source=False,
            use_model=True,
            use_model_dep_interactions=False,
        )

    return [
        (PANEL_TITLES[key], fits[key].ols)
        for key in PANEL_ORDER
        if key in fits and fits[key].ols is not None
    ]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=Path, default=Path("results/final2.0"))
    parser.add_argument("--experiment", default="ideology")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("results/figures/ols/combined.pdf"),
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help=(
            "Also write the figure split up for a slide deck: "
            "coefplot_delta (the differential panel), "
            "coefplot_sources (Wikipedia | Grokipedia), and "
            "coefplot_legend (the Positive/Negative/n.s. key on its own). "
            "Both panels carry the key as well; the standalone file is there "
            "for a deck that would rather place it by hand. PNG/PDF/SVG each."
        ),
    )
    parser.add_argument(
        "--poster",
        action="store_true",
        help=(
            "Also write the three panels as one poster figure: long ideology "
            "names, a ruled separator between the differential regression and "
            "the two per-source ones, the direction key under the "
            "differential, and type sized to be read at poster distance. "
            "PNG/PDF/SVG in <results-path>/figures/poster."
        ),
    )
    parser.add_argument(
        "--poster-dir",
        type=Path,
        default=None,
        help="Directory for the --poster figure. Defaults to "
             "<results-path>/figures/poster.",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help=(
            "Directory for the --split PNGs. Defaults to the slide-figure "
            "tree alongside the other decks' figures, "
            "<results-path>/figures/slides/coefplot."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = load_and_prepare_data(
        results_path=args.results_path,
        leaders_path=DEFAULT_LEADERS_PATH,
        experiment=args.experiment,
        continuous_independent_vars=CONTINUOUS_INDEPENDENT_VARS,
    )

    panels = build_panels(df, CONTINUOUS_INDEPENDENT_VARS, CLUSTER_COLUMN)
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    plot_coefficients_grid(
        panels=panels,
        save_path=args.save_path,
        show_intercept=True,
        # The differential is a different quantity from the two level panels,
        # so its x axis is not tied to theirs.
        shared_x=False,
    )
    print(f"Wrote {args.save_path}")

    if args.split:
        split_dir = args.split_dir or (
            args.results_path / "figures" / "slides" / "coefplot"
        )
        delta = [(t, r) for t, r in panels if t == PANEL_TITLES["delta"]]
        sources = [(t, r) for t, r in panels if t != PANEL_TITLES["delta"]]
        for suffix, subset in (("delta", delta), ("sources", sources)):
            if not subset:
                continue
            out = split_dir / f"coefplot_{suffix}.png"
            plot_coefficients_grid(
                panels=subset,
                save_path=out,
                show_intercept=True,
                # Within the two-source figure the coefficients are
                # commensurable, so those panels share one x axis; the lone
                # differential panel has nothing to share with.
                shared_x=(suffix == "sources"),
                # The LLM rater dummies are still estimated (same regression,
                # same R^2); they are nuisance terms for a slide about the
                # ideology coefficients, so they are not drawn.
                hide_categories=("model",),
                extra_formats=("svg",),
                # A slide has the width for the full dimension names, and an
                # audience seeing the figure once cannot expand the paper's
                # abbreviations the way a reader with the variable table can.
                label_display=SLIDE_IDEOLOGY_LABELS,
                # As on the poster: names, not a term-type cue.
                italic_terms=False,
                # Only the differential panel needs it: Δy is Wikipedia minus
                # Grokipedia, so the sign says which of the two is the more
                # favourable, and that is not something the axis can show.
                direction_labels=(
                    DELTA_DIRECTION_LABELS if suffix == "delta" else None
                ),
                direction_caption=(
                    DELTA_DIRECTION_CAPTION if suffix == "delta" else None
                ),
            )
            print(f"Wrote {out}")

        plot_coefficient_legend(split_dir / "coefplot_legend.svg")

    if args.poster:
        poster_dir = args.poster_dir or (args.results_path / "figures" / "poster")
        out = poster_dir / "coefplot_poster.png"
        plot_coefficients_grid(
            panels=panels,
            save_path=out,
            show_intercept=True,
            # The differential is a different regression from the two level
            # panels and carries different units, so it keeps its own x axis
            # while Wikipedia and Grokipedia share theirs.
            # One x axis across all three: the zero lines and the tick
            # intervals then agree panel to panel, which is what makes the
            # side-by-side comparison readable at a glance. The three ranges
            # happen to be close enough in magnitude that nothing is squeezed.
            shared_x=True,
            # ... and the same split is drawn: a wider gutter with a dashed
            # rule down it, between the differential and the per-source pair.
            group_boundaries=[1],
            label_display=SLIDE_IDEOLOGY_LABELS,
            hide_categories=("model",),
            direction_labels=DELTA_DIRECTION_LABELS,
            direction_caption=DELTA_DIRECTION_CAPTION,
            direction_panel=PANEL_ORDER.index("delta"),
            legend_position=POSTER_LEGEND_POSITION,
            # The key goes above the titles and the strip carries its arrow as
            # a glyph on the labels instead of a drawn rule, so the bottom of
            # the figure is the strip alone and the three x axes finish on one
            # flush baseline. Inside the panels the strip cost more than it
            # saved (6.35 in tall against 5.38 in here), since it pushes the
            # rows down as well as needing its own room.
            direction_placement="one_line",
            # The poster board is dark, so chrome and type flip to light ink.
            light_ink=True,
            # The ideology names read as names, not as a term-type cue: the
            # italic distinguishes them from model and source terms, and those
            # are not drawn here.
            italic_terms=False,
            type_scale=POSTER_TYPE_SCALE,
            fig_width=POSTER_FIG_WIDTH_IN,
            row_height_in=POSTER_ROW_HEIGHT_IN,
            extra_formats=("svg",),
        )
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
