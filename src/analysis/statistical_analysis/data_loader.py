"""Data loading and preparation functions for statistical analysis."""

import logging
from pathlib import Path
from typing import Dict, List, Optional
from statsmodels.stats.outliers_influence import variance_inflation_factor

import numpy as np
import pandas as pd

from .config import (
    LIKERT_MAPPING, SOURCE_NAME_MAPPING, BASELINE_MODEL,
    REGION_COLUMN, REGION_BASELINE, REGIONS_BY_ISO3, CLUSTER_COLUMN,
    RAW_SUFFIX,
)

logger = logging.getLogger(__name__)

# Canonical column names used throughout the analysis pipeline
_CANONICAL_COLUMNS = ["Name", "Source", "Model", "neutrality_assessment", "explanation"]

# Column renames applied when loading API results to align with batch format.
# Covers the older voting-style columns (Choice/Explanation) and the lowercase
# columns written by the synchronous ideology runner (name/source/model).
_API_COLUMN_RENAMES = {
    "Choice": "neutrality_assessment",
    "Explanation": "explanation",
    "name": "Name",
    "source": "Source",
    "model": "Model",
}

def _normalise_results(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Normalise a raw results DataFrame to canonical column names.

    Applies common transformations shared by both API and batch loaders:
    - Renames API-style columns (``Choice``/``Explanation`` and the lowercase
      ``name``/``source``/``model``) to match the batch format.
    - Ensures a ``Model`` column exists (filled with *model_name*).
    - Maps source identifiers via :data:`SOURCE_NAME_MAPPING`.
    - Drops ``custom_id`` if present (batch artefact).
    """
    df = df.copy()

    # Rename first so the canonical ``Model``/``Source`` columns exist before
    # the fallback and the source mapping below run.
    rename_map = {old: new for old, new in _API_COLUMN_RENAMES.items() if old in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    if "Model" not in df.columns:
        df["Model"] = model_name

    if "custom_id" in df.columns:
        df = df.drop(columns=["custom_id"])

    df["Source"] = df["Source"].replace(SOURCE_NAME_MAPPING)
    return df


def _load_results_csv(path: Path, model_name: str, label: str) -> Optional[pd.DataFrame]:
    """Read a single ``results.csv`` and normalise it, or return ``None``."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = _normalise_results(df, model_name)
    logger.info("Loaded %s: %s (%d rows)", label, model_name, len(df))
    return df


def load_model_results(experiment_path: Path, label: str) -> List[pd.DataFrame]:
    """Load results from all model subdirectories under ``{kind}/{experiment}/``.

    Used for both ``api/`` and ``local/`` results, which share the structure::

        {kind}/{experiment}/{model}/results.csv
    """
    if not experiment_path.exists():
        return []

    dfs: List[pd.DataFrame] = []
    for model_dir in sorted(experiment_path.iterdir()):
        if not model_dir.is_dir():
            continue
        df = _load_results_csv(
            model_dir / "results.csv", model_name=model_dir.name, label=label
        )
        if df is not None:
            dfs.append(df)
    return dfs


def load_batch_model_results(batch_experiment_path: Path) -> List[pd.DataFrame]:
    """Load results from all batch subdirectories under ``batch/{experiment}/``.

    Expected structure::

        batch/{experiment}/{provider}_{model}/{batch_id}/results.csv
    """
    if not batch_experiment_path.exists():
        return []

    dfs: List[pd.DataFrame] = []
    for model_dir in sorted(batch_experiment_path.iterdir()):
        if not model_dir.is_dir():
            continue
        for batch_id_dir in sorted(model_dir.iterdir()):
            if not batch_id_dir.is_dir():
                continue
            label = f"batch results ({batch_id_dir.name})"
            df = _load_results_csv(
                batch_id_dir / "results.csv", model_name=model_dir.name, label=label
            )
            if df is not None:
                dfs.append(df)
    return dfs


def discover_and_load_results(
    results_path: Path,
    experiment: str,
) -> pd.DataFrame:
    """Discover and load results from the ``api/``, ``batch/`` and ``local/`` subdirectories.

    Args:
        results_path: Root results directory (e.g. ``results/20260312_100420_…``).
        experiment: Experiment name (e.g. ``"ideology"``).

    Returns:
        Combined DataFrame with canonical columns:
        ``Name``, ``Source``, ``Model``, ``neutrality_assessment``, ``explanation``.
    """
    api_path = results_path / "api" / experiment
    batch_path = results_path / "batch" / experiment
    local_path = results_path / "local" / experiment

    logger.info("Scanning API results in: %s", api_path)
    logger.info("Scanning batch results in: %s", batch_path)
    logger.info("Scanning local results in: %s", local_path)

    all_dfs = (
        load_model_results(api_path, label="API results")
        + load_batch_model_results(batch_path)
        + load_model_results(local_path, label="local results")
    )
    if not all_dfs:
        raise FileNotFoundError(
            f"No results found under {results_path} for experiment '{experiment}'"
        )

    combined = pd.concat(all_dfs, ignore_index=True)
    keep_cols = [c for c in _CANONICAL_COLUMNS if c in combined.columns]
    return combined[keep_cols]


def apply_likert_mapping(
    df: pd.DataFrame,
    column: str = "neutrality_assessment",
    mapping: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """Apply Likert scale mapping to convert text labels to numeric values."""
    if mapping is None:
        mapping = LIKERT_MAPPING
    df = df.copy()
    df[column] = df[column].map(mapping)
    return df


def prepare_analysis_dataframe(
    merged_df: pd.DataFrame,
    choice_col: str = "neutrality_assessment",
) -> pd.DataFrame:
    """Set categorical types for Model, Source, dependent vars, and Choice."""
    df = merged_df.copy()
    df["is_Grokipedia"] = df["Source"].map({'wikipedia': 0, 'grokipedia_0.2': 1})

    df["Model"] = df["Model"].astype("category")
    # Put the baseline model first so it becomes the dropped reference level in
    # both the OLS C(Model) treatment coding and the ordinal get_dummies encoding.
    if BASELINE_MODEL in df["Model"].cat.categories:
        ordered = [BASELINE_MODEL] + [m for m in df["Model"].cat.categories if m != BASELINE_MODEL]
        df["Model"] = df["Model"].cat.reorder_categories(ordered)

    df["Source"] = df["Source"].astype("category")

    # Regional fixed-effects control: map ISO3 country codes to world regions and
    # put the baseline region first so it becomes the dropped reference level
    # (matching the Model/treatment-coding convention used above).
    if "country_isocode" in df.columns:
        df[REGION_COLUMN] = df["country_isocode"].map(REGIONS_BY_ISO3).astype("category")
        present = list(df[REGION_COLUMN].cat.categories)
        if REGION_BASELINE in present:
            ordered = [REGION_BASELINE] + [r for r in present if r != REGION_BASELINE]
            df[REGION_COLUMN] = df[REGION_COLUMN].cat.reorder_categories(ordered)

    # Clustering key for the cluster-robust standard errors. Party Facts ids are
    # globally unique per party, so they need no country prefix; where one is
    # missing we fall back to country + party name (a party name alone is not
    # unique across countries, e.g. "ps", "sp").
    if CLUSTER_COLUMN not in df.columns:
        pf = df["partyfacts_id"] if "partyfacts_id" in df.columns else pd.Series(index=df.index, dtype=float)
        fallback = (
            df["country_isocode"].astype(str) + "|" + df["party"].astype(str)
            if {"country_isocode", "party"} <= set(df.columns)
            else df["Name"].astype(str)
        )
        df[CLUSTER_COLUMN] = np.where(
            pf.notna(), "pf:" + pf.astype("Int64").astype(str), fallback
        )

    df["neutrality_assessment"] = df[choice_col].astype(int)
    return df

def process_leaders_df(df, continuous_independent_vars, z_score_standardization=True):
    # '''Make all dep vars be 0: conservative 1: liberal, and rename them to reflect this'''
    # transformations = [
    #         {'original': 'v2xpa_antiplural', 'new': 'v2xpa_plural', 'transform': lambda x: 1 - x},
    #         {'original': 'v2xpa_antiplural_bool', 'new': 'v2xpa_plural_bool', 'transform': lambda x: 1 - x},

    #         {'original': 'v2xpa_popul', 'new': 'v2xpa_antipopul', 'transform': lambda x: 1 - x},
    #         {'original': 'v2xpa_popul_bool', 'new': 'v2xpa_antipopul_bool', 'transform': lambda x: 1 - x},

    #         {'original': 'v2paclient_osp', 'new': 'v2paanticlient_osp', 'transform': lambda x: 4 - x},
    #         {'original': 'v2paclient_osp_bool', 'new': 'v2paanticlient_osp_bool', 'transform': lambda x: 1 - x},

    #         {'original': 'v2paclient_bool', 'new': 'v2paanticlient_bool', 'transform': lambda x: 1 - x},
    #     ]

    # for trans in transformations:
    #     df[trans['original']] = df[trans['original']].apply(trans['transform'])
    #     df = df.rename(columns={trans['original']: trans['new']})

    # Keep an untouched copy of every ideology score on its original codebook
    # scale (0-4 for the Likert items, 0-6 for v2pariglef, 0-1 for the two
    # indexes). The z-scoring below overwrites the column in place, which loses
    # the location of the codebook's own neutral category -- needed by the
    # magnitude models, where "ideologically extreme" means far from the
    # codebook midpoint and not far from the sample mean.
    for var in continuous_independent_vars:
        df[f"{var}{RAW_SUFFIX}"] = pd.to_numeric(df[var], errors="coerce")

    if z_score_standardization:
        # z-score standardization: z = (x - mu) / sigma
        for var in continuous_independent_vars:
            col = pd.to_numeric(df[var], errors="coerce")
            mu = col.mean()
            sigma = col.std()
            df[var] = (col - mu) / sigma

    return df

def run_vif_check(df: pd.DataFrame, independent_vars: List[str]) -> pd.DataFrame:
    """Compute VIF for independent variables.

    Continuous variables are used as-is; categorical variables (e.g. region) are
    one-hot encoded with the first category dropped, matching the reference coding
    used in the models, so the dummies are not perfectly collinear.
    """
    cols = [c for c in independent_vars if c in df.columns]
    if len(cols) < 2:
        return pd.DataFrame(columns=["variable", "VIF"])

    pieces = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            pieces.append(pd.to_numeric(df[c], errors="coerce").rename(c))
        else:
            pieces.append(pd.get_dummies(df[c], prefix=c, drop_first=True).astype(float))
    x = pd.concat(pieces, axis=1).dropna()

    # Drop constants (VIF undefined)
    constant_cols = [c for c in x.columns if x[c].nunique(dropna=True) <= 1]
    if constant_cols:
        logger.warning("Dropping constant columns from VIF: %s", constant_cols)
        x = x.drop(columns=constant_cols)

    if x.shape[1] < 2 or x.empty:
        return pd.DataFrame(columns=["variable", "VIF"])

    vif_df = pd.DataFrame(
        {
            "variable": x.columns,
            "VIF": [variance_inflation_factor(x.values, i) for i in range(x.shape[1])],
        }
    ).sort_values("VIF", ascending=False, ignore_index=True)

    logger.info("VIF results:\n%s", vif_df.to_string(index=False))
    return vif_df

def load_and_prepare_data(
    results_path: str | Path,
    leaders_path: str | Path,
    experiment: str,
    continuous_independent_vars: Optional[List[str]] = None,
    z_score_standardization: bool = True,
    tables_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Discover, load and prepare results from api/, batch/ and local/ subdirectories.

    Args:
        results_path: Root results directory containing api/, batch/ and local/ folders.
        leaders_path: Path to leaders metadata CSV.
        experiment: Experiment name (e.g. "ideology").
        independent_vars: List of column names for dependent variables.
        z_score_standardization: Whether to apply z-score standardization to continuous independent variables.
    Returns:
        Prepared DataFrame with all model results merged.
    """
    leaders_df = pd.read_csv(leaders_path)

    print(f"Loaded leaders metadata: {leaders_df.shape[0]} rows")
    leaders_df = process_leaders_df(leaders_df, continuous_independent_vars=continuous_independent_vars, z_score_standardization=z_score_standardization)

    results_path = Path(results_path)
    results = discover_and_load_results(results_path, experiment)

    merged = leaders_df.merge(
        results,
        left_on="name",
        right_on="Name",
        how="inner"
    )

    merged = apply_likert_mapping(merged)

    final = prepare_analysis_dataframe(merged)

    na = final[final[["Name", "Source", "Model", "neutrality_assessment"] + continuous_independent_vars].isna().any(axis=1)]
    print(f"Dropping {len(na)} rows with no flags")

    final = final.dropna(subset=["Name", "Source", "Model", "neutrality_assessment"] + continuous_independent_vars)

    vif_df = run_vif_check(final, continuous_independent_vars)
    print(vif_df)
    # VIF including the regional fixed effects alongside the ideology predictors.
    vif_regions_df = run_vif_check(final, continuous_independent_vars + [REGION_COLUMN])
    print(vif_regions_df)
    if tables_dir is not None:
        tables_dir = Path(tables_dir)
        tables_dir.mkdir(parents=True, exist_ok=True)
        vif_path = tables_dir / "vif.csv"
        vif_df.to_csv(vif_path, index=False)
        print(f"Saved VIF table to {vif_path}")
        vif_regions_path = tables_dir / "vif_regions_ideologies.csv"
        vif_regions_df.to_csv(vif_regions_path, index=False)
        print(f"Saved regions+ideologies VIF table to {vif_regions_path}")
    high_vif = vif_df[vif_df["VIF"] >= 5]
    print(f"VIF check: {len(high_vif)} variables with VIF >= 5")
    if not high_vif.empty:
      logger.warning("Potential multicollinearity (VIF >= 5):\n%s", high_vif.to_string(index=False))

    return final
