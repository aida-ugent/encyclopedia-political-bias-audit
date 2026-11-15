"""Analysis and summary statistics functions."""

from typing import Dict, List

import pandas as pd


def compute_summary_statistics(
    df: pd.DataFrame,
    outcome_col: str = "neutrality_assessment",
    group_cols: List[str] = None,
    agg_funcs: List[str] = ["mean", "std", "count"],
) -> pd.DataFrame:
    """Compute summary statistics for outcome by grouping columns."""
    return df.groupby(group_cols, observed=True)[outcome_col].agg(agg_funcs).round(2)


def print_data_summary(df: pd.DataFrame, outcome_col: str = "neutrality_assessment") -> None:
    """Print data shape and distribution summary."""
    print(f"Data shape: {df.shape}")
    print(f"\n{outcome_col} distribution:\n{df[outcome_col].value_counts().sort_index()}")
    for col in ["Model", "Source"]:
        if col in df.columns:
            print(f"\n{col} distribution:\n{df[col].value_counts()}")


def compute_all_summary_statistics(
    df: pd.DataFrame,
    outcome_col: str = "neutrality_assessment",
    model_col: str = "Model",
    source_col: str = "Source",
    independent_vars: List[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Compute summary statistics for all factor combinations and dependent variables."""
    summaries = {}

    # Single factor: Model, Source
    for name, col in [("model", model_col), ("source", source_col)]:
        if col in df.columns:
            summaries[f"by_{name}"] = compute_summary_statistics(df, outcome_col, [col])

    # For each dependent variable
    for indep_var in independent_vars:
        if indep_var not in df.columns:
            continue
        short_name = indep_var.replace("_bool", "").replace("v2x", "").replace("pa_", "")

        # Single: by indep_var
        summaries[f"by_{short_name}"] = compute_summary_statistics(df, outcome_col, [indep_var])

        # Two-way combinations
        if model_col in df.columns:
            summaries[f"by_model_{short_name}"] = compute_summary_statistics(
                df, outcome_col, [model_col, indep_var]
            )
        if source_col in df.columns:
            summaries[f"by_source_{short_name}"] = compute_summary_statistics(
                df, outcome_col, [source_col, indep_var]
            )

    # Model × Source
    if model_col in df.columns and source_col in df.columns:
        summaries["by_model_source"] = compute_summary_statistics(
            df, outcome_col, [model_col, source_col]
        )

    return summaries


def print_all_summary_statistics(
    df: pd.DataFrame,
    outcome_col: str = "neutrality_assessment",
    model_col: str = "Model",
    source_col: str = "Source",
    independent_vars: List[str] = None,
) -> None:
    """Print all summary statistics."""
    print("=" * 80)
    print("SUMMARY STATISTICS BY FACTOR")
    print("=" * 80)

    summaries = compute_all_summary_statistics(
        df, outcome_col, model_col, source_col, independent_vars
    )

    for i, (key, table) in enumerate(summaries.items(), 1):
        print(f"\n{i}. {key}:")
        print(table)
