#!/usr/bin/env python3
"""
Statistical analysis of sentiment predictions (Likert scale -2 to 2).

Supports multiple dependent variables and multiple model batches.

Output will be saved to results/{timestamp}_{run_id}/figures (created automatically).
"""

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

from .config import CONTINUOUS_INDEPENDENT_VARS, BOOL_INDEPENDENT_VARS, DEFAULT_LEADERS_PATH, BASELINE_MODEL, PER_SOURCE_BASELINE_MODEL, CONTROL_VARS, REGION_COLUMN, CLUSTER_COLUMN
from .data_loader import load_and_prepare_data
from .stat_models import fit_all_models
from .analysis import print_data_summary, print_all_summary_statistics
from .visualisation.visualization import plot_all_stat_figures, plot_common_figures
from .visualisation.stat_figures import plot_coefficients_grid
from .visualisation.helpers import compute_article_length
from .robustness import run_robustness_checks
from .mixed_models import run_mixed_effects_analysis
from .ordinal_validation import run_ordinal_validation
from .length_functional_form import run_length_functional_form

def run_analysis(
    results_path: Path,
    experiment: str,
    leaders_path: Path,
    bool_independent_vars: List[str],
    continuous_independent_vars: List[str],
    use_clustered_se: bool = True,
    output_dir: Optional[Path] = None,
    run_regressions: bool = True,
) -> Tuple:
    """Run the analysis pipeline.

    Loads the data, prints the summaries and generates the common/descriptive
    figures (e.g. the score-distribution plots). When ``run_regressions`` is
    False the pipeline stops there — all regression model fitting, robustness /
    mixed-effects checks and coefficient figures (the slow part) are skipped.
    """
    print(f"Results path: {results_path}")
    print(f"Experiment: {experiment}")
    print(f"Independent vars: {bool_independent_vars}")
    print(f"Continuous independent vars: {continuous_independent_vars}")
    print(f"Run regressions: {run_regressions}")
    # Set up output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    # Load data
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    df = load_and_prepare_data(
        results_path=results_path,
        leaders_path=leaders_path,
        experiment=experiment,
        continuous_independent_vars=continuous_independent_vars,
        tables_dir=table_dir,
    )
    print_data_summary(df)

    # Summary statistics
    print("\n")
    print_all_summary_statistics(df, independent_vars=bool_independent_vars)

    # Descriptive figures (score distributions, etc.) — always generated.
    print("\n" + "=" * 80)
    print("GENERATING DESCRIPTIVE FIGURES")
    print("=" * 80)
    plot_common_figures(
        df=df,
        bool_independent_vars=bool_independent_vars,
        continuous_independent_vars=continuous_independent_vars,
        figures_dir=figure_dir,
        tables_dir=table_dir,
    )

    if not run_regressions:
        print("\nSkipping regressions (run_regressions=False)")
        print("\n" + "=" * 80)
        print("COMPLETE")
        print("=" * 80)
        return

    run_regression_analysis(
        df=df,
        bool_independent_vars=bool_independent_vars,
        continuous_independent_vars=continuous_independent_vars,
        use_clustered_se=use_clustered_se,
        figure_dir=figure_dir,
        table_dir=table_dir,
    )

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


def run_regression_analysis(
    df,
    bool_independent_vars: List[str],
    continuous_independent_vars: List[str],
    use_clustered_se: bool,
    figure_dir: Path,
    table_dir: Path,
) -> None:
    """Fit every regression model and generate the coefficient figures.

    This is the slow half of the pipeline; ``run_analysis(run_regressions=False)``
    skips it entirely.
    """
    # Fit models
    print("\n" + "=" * 80)
    print("FITTING MODELS")
    print("=" * 80)

    # Cluster-robust SEs at the party level (see config.CLUSTER_COLUMN): V-Dem
    # ideology scores are party-level measures, so co-partisans share both
    # their predictors and much of their editorial treatment.
    cluster_col = CLUSTER_COLUMN if use_clustered_se else None
    continuous_results = fit_all_models(df, independent_vars=continuous_independent_vars, cluster_col=cluster_col, model_col="Model", source_col="is_Grokipedia")
    bool_results = fit_all_models(df, independent_vars=bool_independent_vars, cluster_col=cluster_col, model_col="Model", source_col="is_Grokipedia")

    # Build wiki-groki difference across all models so Model can serve as an interaction factor
    df_wiki_minus_groki = (
        df[df["Source"] == "wikipedia"]
        .merge(
            df[df["Source"] == "grokipedia_0.2"][['name', 'country_isocode', 'Model', 'neutrality_assessment']],
            on=['name', 'country_isocode', 'Model'],
            suffixes=('', '_groki')
        )
        .assign(neutrality_assessment=lambda x: x['neutrality_assessment'] - x['neutrality_assessment_groki'])
        .drop(columns=['neutrality_assessment_groki'])
    )

    # Controls for the differential regression:
    #  - article-length difference: word count (wikipedia − grokipedia_0_2),
    #    z-scored to match the standardized continuous predictors. The full
    #    article texts are leader-level columns already carried on every wiki row.
    #  - regional fixed effects (V-Dem regions). Only ~5 dummies, so unlike the
    #    per-source country FE these are cheap enough to include in both models.
    wiki_len = df_wiki_minus_groki["wikipedia"].apply(compute_article_length)
    groki_len = df_wiki_minus_groki["grokipedia_0_2"].apply(compute_article_length)
    length_diff = (wiki_len - groki_len).astype(float)
    df_wiki_minus_groki["article_length_diff"] = (length_diff - length_diff.mean()) / length_diff.std()
    # Controls disabled for the differential regression. Flip either flag to
    # True to restore that control; the code paths below stay intact.
    INCLUDE_REGION_FE = False        # regional (country) fixed effects
    INCLUDE_LENGTH_CONTROL = False   # article-length difference
    region_controls = [REGION_COLUMN] if INCLUDE_REGION_FE else []
    length_controls = CONTROL_VARS if INCLUDE_LENGTH_CONTROL else []
    diff_control_vars = length_controls + region_controls

    wiki_minus_groki_results = fit_all_models(
        df_wiki_minus_groki,
        independent_vars=continuous_independent_vars,
        cluster_col=cluster_col,
        model_col="Model",
        use_source=False,
        use_model=True,
        control_vars=diff_control_vars,
        sum_code_controls=region_controls,
    )
    # Pooled model (no model interactions) — clean overall ideology effects
    wiki_minus_groki_pooled_results = fit_all_models(
        df_wiki_minus_groki,
        independent_vars=continuous_independent_vars,
        cluster_col=cluster_col,
        model_col="Model",
        use_source=False,
        use_model=False,
        control_vars=diff_control_vars,
        sum_code_controls=region_controls,
    )

    # Source-term model (no difference outcome): keeps Source × ideology
    # interactions but drops every Model term (main effect and interactions),
    # pooling across LLM raters. Within-leader/cross-rater dependence is
    # absorbed by the party-level cluster-robust SEs.
    source_ideology_results = fit_all_models(
        df,
        independent_vars=continuous_independent_vars,
        cluster_col=cluster_col,
        model_col="Model",
        source_col="is_Grokipedia",
        use_source=True,
        use_model_dep_interactions=False,
        include_model_term=False,
    )

    # Cell-means reparameterisation of the Source × ideology model: no
    # overall intercept, full-rank Source dummies, ideology terms interacted
    # with each Source level -> per-source slopes appear as stand-alone rows
    # (no reference level).
    source_cell_means_results = fit_all_models(
        df,
        independent_vars=continuous_independent_vars,
        cluster_col=cluster_col,
        model_col="Model",
        source_col="is_Grokipedia",
        use_source=True,
        use_model_dep_interactions=False,
        include_model_term=False,
        source_cell_means=True,
    )

    per_model_source = {}

    for model in df["Model"].unique():
        for source in df["Source"].unique():
            per_model_source[(source, model)] = fit_all_models(df[(df["Source"] == source) & (df["Model"] == model)], independent_vars=continuous_independent_vars, cluster_col=cluster_col, use_source=False)

    # Per-source regressions: one OLS per Source. Source is fixed by the
    # subset, so it drops out of the design; Model is kept as a main-effect
    # control only (no Model × ideology interactions) — pooled ideology slopes
    # within each source, adjusted for rater-level intercept shifts. Regional
    # fixed effects (when enabled via INCLUDE_REGION_FE above) are added to the
    # OLS only (absorbing between-region variation; excluded from the ordinal
    # fit, which is too slow with many dummy levels).
    per_source = {}
    for source in df["Source"].unique():
        df_source = df[df["Source"] == source].copy()
        # Mistral as the dropped reference level for the per-source fits
        # (overrides the global BASELINE_MODEL category order).
        model_cats = list(df_source["Model"].cat.categories)
        if PER_SOURCE_BASELINE_MODEL in model_cats:
            df_source["Model"] = df_source["Model"].cat.reorder_categories(
                [PER_SOURCE_BASELINE_MODEL]
                + [m for m in model_cats if m != PER_SOURCE_BASELINE_MODEL]
            )
        per_source[source] = fit_all_models(
            df_source,
            independent_vars=continuous_independent_vars,
            cluster_col=cluster_col,
            model_col="Model",
            use_source=False,
            use_model=True,
            use_model_dep_interactions=False,
            ols_control_vars=region_controls,
        )

    # Ordinal-logit re-estimation of Eq. 1 and Eq. 2 (SM validation figure and
    # OLS-vs-ordinal comparison tables). Reuses the fits above, so it cannot
    # drift from the specifications it validates.
    run_ordinal_validation(
        delta_results=wiki_minus_groki_pooled_results,
        per_source_results=per_source,
        figure_dir=figure_dir,
        table_dir=table_dir,
    )

    # Robustness checks (Appendix Tables B & C, coefficient-stability figure)
    print("\n" + "=" * 80)
    print("ROBUSTNESS CHECKS")
    print("=" * 80)
    run_robustness_checks(
        df=df,
        df_wiki_minus_groki=df_wiki_minus_groki,
        ideology_vars=continuous_independent_vars,
        cluster_col=cluster_col,
        table_dir=table_dir,
        figure_dir=figure_dir,
    )

    # Article-length control on the log scale, plus ideology x length
    # interactions. Replaces the linear word-count difference of the robustness
    # ladder with L = log(W_G/W_W), reports how far the equal-length point sits
    # from the data, and tests whether the ideology effects strengthen where
    # Grokipedia expanded the article more.
    print("\n" + "=" * 80)
    print("LENGTH FUNCTIONAL FORM AND IDEOLOGY x LENGTH")
    print("=" * 80)
    run_length_functional_form(
        df_wiki_minus_groki=df_wiki_minus_groki,
        ideology_vars=continuous_independent_vars,
        cluster_col=cluster_col,
        table_dir=table_dir,
        figure_dir=figure_dir,
        df=df,
    )

    # Mixed-effects models: regional fixed effects + a per-country random
    # intercept, for the differentiation outcome and per-source. Saves coefplots,
    # summaries, and ranked per-country BLUPs under mixed_effects/.
    run_mixed_effects_analysis(
        df=df,
        df_wiki_minus_groki=df_wiki_minus_groki,
        ideology_vars=continuous_independent_vars,
        figure_dir=figure_dir,
        table_dir=table_dir,
    )

    # Coefficient figures
    print("\n" + "=" * 80)
    print("GENERATING REGRESSION FIGURES")
    print("=" * 80)

    plot_all_stat_figures(
        df=df,
        all_results=bool_results,
        create_individual_overviews=True,
        independent_vars=bool_independent_vars,
        figures_dir=figure_dir/ "boolean_ind_vars",
    )

    plot_all_stat_figures(
        df=df,
        all_results=continuous_results,
        create_individual_overviews=False,
        independent_vars=continuous_independent_vars,
        figures_dir=figure_dir/ "continuous_ind_vars"
    )

    # Determine which model is the reference (dropped by drop_first)
    wiki_groki_models = list(df_wiki_minus_groki["Model"].unique())
    reference_model = BASELINE_MODEL if BASELINE_MODEL in wiki_groki_models else sorted(wiki_groki_models)[0]

    # 1. Pooled model: clean overall ideology effects (no model interactions)
    plot_all_stat_figures(
        df=df_wiki_minus_groki,
        all_results=wiki_minus_groki_pooled_results,
        create_individual_overviews=False,
        independent_vars=continuous_independent_vars,
        figures_dir=figure_dir / "wiki_minus_groki" / "pooled",
        difference_outcome=True,
    )

    # 2. Interaction model: how LLMs diverge (label reference model)
    plot_all_stat_figures(
        df=df_wiki_minus_groki,
        all_results=wiki_minus_groki_results,
        create_individual_overviews=False,
        independent_vars=continuous_independent_vars,
        figures_dir=figure_dir / "wiki_minus_groki" / "interactions",
        reference_model_name=reference_model,
        difference_outcome=True,
    )

    # 3. Source-term model on full df: Source × ideology interactions,
    # no Model × ideology interactions.
    plot_all_stat_figures(
        df=df,
        all_results=source_ideology_results,
        create_individual_overviews=False,
        independent_vars=continuous_independent_vars,
        figures_dir=figure_dir / "source_ideology_interaction",
    )

    # 4. Cell-means reparameterisation: per-source ideology slopes appear as
    # stand-alone coefficients (no reference level).
    plot_all_stat_figures(
        df=df,
        all_results=source_cell_means_results,
        create_individual_overviews=False,
        independent_vars=continuous_independent_vars,
        figures_dir=figure_dir / "source_cell_means",
    )

    for model in df["Model"].unique():
        for source in df["Source"].unique():
            plot_all_stat_figures(
                df=df,
                all_results=per_model_source[(source, model)],
                create_individual_overviews=False,
                independent_vars=continuous_independent_vars,
                figures_dir=figure_dir/ f"per_model_source/{source}/{model}",
            )

    for source in df["Source"].unique():
        plot_all_stat_figures(
            df=df[df["Source"] == source],
            all_results=per_source[source],
            create_individual_overviews=False,
            independent_vars=continuous_independent_vars,
            figures_dir=figure_dir / f"per_source/{source}",
            show_intercept=False,
        )

    # Combined per-source OLS coefplot: Wikipedia (left) | Grokipedia (right)
    per_source_panel_order = [
        ("wikipedia", "Wikipedia"),
        ("grokipedia_0.2", "Grokipedia"),
    ]
    per_source_panels = [
        (panel_title, per_source[source].ols)
        for source, panel_title in per_source_panel_order
        if source in per_source and per_source[source].ols is not None
    ]
    if per_source_panels:
        plot_coefficients_grid(
            panels=per_source_panels,
            save_path=figure_dir / "per_source" / "ols_coefficients_grid.png",
            show_intercept=True,
            shared_x=True,
        )


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.analysis.statistical_analysis.main",
        description=(
            "Statistical analysis pipeline. By default it fits every regression "
            "model and generates all figures; use --descriptive-only to produce "
            "just the data summaries and common/descriptive figures (e.g. the "
            "score-distribution plots) without running the regressions."
        ),
    )
    parser.add_argument(
        "--descriptive-only", "--no-regressions",
        dest="descriptive_only", action="store_true",
        help="Skip all regression fitting, robustness/mixed-effects checks and "
             "coefficient figures; only produce descriptive outputs.",
    )
    parser.add_argument(
        "--results-path", type=Path, default=Path("results/final2.0"),
        help="Root results directory to load from (default: results/final2.0).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where figures/tables are written (default: --results-path).",
    )
    parser.add_argument(
        "--experiment", default="ideology",
        help="Experiment name (default: ideology).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    results_path = args.results_path
    experiment = args.experiment
    leaders_path = DEFAULT_LEADERS_PATH
    bool_independent_vars = BOOL_INDEPENDENT_VARS
    cont_independent_vars = CONTINUOUS_INDEPENDENT_VARS

    output_dir = args.output_dir if args.output_dir is not None else results_path

    run_analysis(
        results_path=results_path,
        experiment=experiment,
        leaders_path=leaders_path,
        continuous_independent_vars=cont_independent_vars,
        bool_independent_vars=bool_independent_vars,
        use_clustered_se=True,
        output_dir=output_dir,
        run_regressions=not args.descriptive_only,
    )

if __name__ == "__main__":
    main()
