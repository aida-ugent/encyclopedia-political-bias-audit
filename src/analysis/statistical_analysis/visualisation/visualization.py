"""Visualization functions for statistical analysis."""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from ..stat_models import AllModelResults, print_all_model_summaries

from .helpers import pretty_dep_label, MODEL_COLUMN, SOURCE_COLUMN
from .descriptive_stats import (
    plot_ideology_correlation_matrix,
    plot_per_model_figures,
    plot_per_source_figures,
    plot_common_overview,
    plot_mean_score_by_model_source,
    plot_mean_abs_score_heatmap,
    plot_non_neutral_rate_by_model_source,
    plot_bias_direction_stacked_by_model_source,
    plot_absolute_bias_common,
    plot_outcome_vs_article_length_by_source,
    plot_score_distribution_by_model,
    plot_score_distribution_by_source,
    plot_score_distribution_by_source_and_model,
    plot_score_distribution_by_model_source_combo,
    plot_neutral_rate_by_model_source,
    plot_neutral_rate_slope,
    save_neutral_rate_abs_score_table,
    plot_model_agreement_by_source,
    plot_inter_rater_agreement,
)
from .stat_figures import (
    plot_ols_coefficients,
    plot_ordinal_coefficients,
    plot_wald_tests,
    plot_anova,
    plot_overview_for_dep_var,
    plot_absolute_bias_for_dep_var,
    plot_marginal_effects,
)

OLS_SUBDIR = "ols"
ORDINAL_SUBDIR = "ordinal"


sources = {
    "wikipedia": "wikipedia",
    "grokipedia_0.2": "grokipedia_0_2"
}


def _plot_ols_stat_figures(ols_res, figures_dir: Path, show_intercept: bool = True) -> None:
    """Save all OLS-specific statistical figures under `figures_dir`."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_ols_coefficients(
        ols_res,
        save_path=figures_dir / "00_coefficients.png",
        show_intercept=show_intercept,
    )
    if ols_res.anova_table is not None:
        plot_anova(ols_res, save_path=figures_dir / "00_anova.png")


def _plot_ordinal_stat_figures(
    ordinal_res,
    figures_dir: Path,
    reference_model_name: str = None,
    difference_outcome: bool = False,
) -> None:
    """Save all ordinal-specific statistical figures under `figures_dir`."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_ordinal_coefficients(
        ordinal_res,
        save_path=figures_dir / "00_coefficients.png",
        reference_model_name=reference_model_name,
        difference_outcome=difference_outcome,
    )
    plot_wald_tests(ordinal_res, save_path=figures_dir / "00_wald_tests.png")
    if ordinal_res.marginal_effects is not None:
        plot_marginal_effects(
            ordinal_res,
            outcome_categories=[-2, -1, 0, 1, 2],
            save_path=figures_dir / "00_marginal_effects.png",
        )


def plot_all_stat_figures(
    df: pd.DataFrame,
    all_results: AllModelResults,
    independent_vars: List[str],
    create_individual_overviews: bool,
    figures_dir: Path,
    reference_model_name: str = None,
    show_intercept: bool = True,
    difference_outcome: bool = False,
) -> None:
    """Create and save all figures, with OLS and ordinal outputs in separate subdirectories.

    Regression summary tables are written to `figures_dir / "tables"` alongside
    the figures (intercept always retained in tables, regardless of
    `show_intercept`).
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if all_results.ols:
        _plot_ols_stat_figures(all_results.ols, figures_dir / OLS_SUBDIR, show_intercept=show_intercept)

    if all_results.ordinal:
        _plot_ordinal_stat_figures(
            all_results.ordinal,
            figures_dir / ORDINAL_SUBDIR,
            reference_model_name=reference_model_name,
            difference_outcome=difference_outcome,
        )

    print_all_model_summaries(all_results, table_dir=figures_dir / "tables")

    if create_individual_overviews:
        # Create individual overview plots per dependent variable for descriptive stats
        for i, dep_var in enumerate(independent_vars, 1):
            # Short, human-friendly label for plotting / filenames
            dep_label = pretty_dep_label(dep_var)
            prefix = f"{i:02d}_{dep_label}"

            # Overview (descriptive stats - doesn't require separate models)
            plot_overview_for_dep_var(
                df,
                dep_var,
                dep_var_label=dep_label,
                save_path=figures_dir / f"{prefix}_overview.png"
            )

            # Absolute-bias overview per dep var
            plot_absolute_bias_for_dep_var(
                df,
                dep_var,
                dep_var_label=dep_label,
                save_path=figures_dir / f"{prefix}_abs_bias.png",
            )


def plot_common_figures(
    df: pd.DataFrame,
    bool_independent_vars: List[str],
    figures_dir: Path,
    continuous_independent_vars: List[str] = None,
    tables_dir: Optional[Path] = None,
)  -> None:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    # Tables default to a sibling 'tables/' folder so callers don't have to
    # pass both paths explicitly.
    tables_dir = Path(tables_dir) if tables_dir is not None else figures_dir.parent / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Common overview panels that are identical across dependent variables
    plot_common_overview(df, bool_independent_vars=bool_independent_vars, save_path=figures_dir / "00_overview_common.png")

    if continuous_independent_vars:
        plot_ideology_correlation_matrix(
            df,
            ideology_vars=continuous_independent_vars,
            save_path=figures_dir / "00_ideology_correlation_matrix.png",
        )

    # Standalone paper-ready heatmap (single-column, 2-column paper layout)
    plot_mean_score_by_model_source(
        df,
        save_path=figures_dir / "00_mean_score_model_source.png",
    )
    plot_mean_abs_score_heatmap(
        df,
        save_path=figures_dir / "00_mean_abs_score_model_source_heatmap.png",
    )
    # Bias-magnitude views that avoid the +/− cancellation of the mean-score plot
    plot_non_neutral_rate_by_model_source(
        df,
        save_path=figures_dir / "00_non_neutral_rate_model_source.png",
    )
    plot_bias_direction_stacked_by_model_source(
        df,
        save_path=figures_dir / "00_bias_direction_stacked_model_source.png",
    )

    # Dedicated paper-ready figures: score distribution by Model, by Source,
    # and broken out by each Model × Source combination.
    score_dist_dir = figures_dir / "score_distribution"
    plot_score_distribution_by_model(
        df, save_path=score_dist_dir / "00_score_distribution_by_model.png",
    )
    plot_score_distribution_by_source(
        df, save_path=score_dist_dir / "00_score_distribution_by_source.png",
    )
    plot_score_distribution_by_source_and_model(
        df, save_path=score_dist_dir / "00_score_distribution_by_source_and_model.png",
    )
    plot_score_distribution_by_model_source_combo(
        df, save_path=score_dist_dir / "00_score_distribution_by_model_source_combo.png",
    )

    # Neutral-rate comparison across Model × Source, with within-model and interaction tests
    plot_neutral_rate_by_model_source(
        df,
        plot_path=figures_dir / "00_neutral_rate_by_model_source.png",
        table_path=tables_dir / "00_neutral_rate_by_model_source.csv",
    )
    plot_neutral_rate_slope(
        df,
        plot_path=figures_dir / "00_neutral_rate_slope.png",
        table_path=tables_dir / "00_neutral_rate_slope.csv",
    )

    # Combined tabular view of the slope plot + mean-|score| heatmap
    save_neutral_rate_abs_score_table(
        df,
        table_path=tables_dir / "00_neutral_rate_and_mean_abs_score.csv",
    )

    # Inter-model agreement on the ordinal score (per source + pooled)
    plot_model_agreement_by_source(
        df,
        plot_path=figures_dir / "00_model_agreement_by_source.png",
        table_path=tables_dir / "00_model_agreement_by_source.csv",
    )

    # Multi-rater agreement summary (per-source pairwise κ heatmap + Krippendorff α)
    plot_inter_rater_agreement(
        df,
        plot_path=figures_dir / "00_inter_rater_agreement.png",
        table_path=tables_dir / "00_inter_rater_agreement.csv",
    )

    # Absolute-bias common overview
    plot_absolute_bias_common(df, save_path=figures_dir / "00_abs_bias_common.png")


    for source, souce_text in sources.items():
        for model in df[MODEL_COLUMN].unique():
            plot_outcome_vs_article_length_by_source(
                df,
                save_path=figures_dir / "length" / f"outcome_vs_article_length_by_source_{source}_{model}.png",
                source=source,
                source_text = souce_text,
                model=model
            )

    plot_per_model_figures(df, models = df[MODEL_COLUMN].unique(), save_path=figures_dir/ "per_model_paired_neutrality_heatmaps" )
    plot_per_source_figures(df, sources = df[SOURCE_COLUMN].unique(), save_path=figures_dir/ "per_source_paired_neutrality_heatmaps" )
    plt.close("all")
    print(f"\nAll figures saved to: {figures_dir}")
