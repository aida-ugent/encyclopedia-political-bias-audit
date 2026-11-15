#!/usr/bin/env python3
"""Run only the identification checks in :mod:`identification`.

The full pipeline in :mod:`main` refits every model and regenerates every
figure. These checks need nothing but the prepared analysis frame, so this
entry point loads the data and runs them directly.

    python -m src.analysis.statistical_analysis.run_identification
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from .config import (
    CLUSTER_COLUMN, CONTINUOUS_INDEPENDENT_VARS, DEFAULT_LEADERS_PATH,
    get_project_root,
)
from .data_loader import load_and_prepare_data
from .identification import run_identification_checks


def build_differential(df):
    """Within-politician Wikipedia minus Grokipedia outcome, one row per judge."""
    return (
        df[df["Source"] == "wikipedia"]
        .merge(
            df[df["Source"] == "grokipedia_0.2"][
                ["name", "country_isocode", "Model", "neutrality_assessment"]
            ],
            on=["name", "country_isocode", "Model"],
            suffixes=("", "_groki"),
        )
        .assign(
            neutrality_assessment=lambda x: x["neutrality_assessment"]
            - x["neutrality_assessment_groki"]
        )
        .drop(columns=["neutrality_assessment_groki"])
    )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=Path, default=Path("results/final2.0"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--experiment", default="ideology")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    output_dir = args.output_dir or args.results_path
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_data(
        results_path=args.results_path,
        leaders_path=DEFAULT_LEADERS_PATH,
        experiment=args.experiment,
        continuous_independent_vars=CONTINUOUS_INDEPENDENT_VARS,
    )

    run_identification_checks(
        df=df,
        df_wiki_minus_groki=build_differential(df),
        ideology_vars=CONTINUOUS_INDEPENDENT_VARS,
        table_dir=table_dir,
        project_root=get_project_root(),
        cluster_col=CLUSTER_COLUMN,
    )


if __name__ == "__main__":
    main()
