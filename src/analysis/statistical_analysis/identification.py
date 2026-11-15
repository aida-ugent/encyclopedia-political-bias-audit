"""Identification and measurement-error checks requested at review.

Four independent checks, each written to its own table under
``tables/identification/``:

* **country** — Eq. 1 and Eq. 2 re-estimated with country fixed effects
  (145 levels) instead of the six V-Dem regions, party-clustered SEs. Answers
  whether the ideology coefficients are within-country or between-country.
* **selection** — logistic model of surviving the coverage funnel (6,846
  annotated politicians -> 1,394 retained pairs) on ideology, region, sex and
  office, plus the ideology coefficients re-estimated on the subsample the
  funnel over-represents.
* **staleness** — the gap in years between a politician's WhoGov year and the
  V-Party election year their ideology score is taken from, added as a
  covariate and used to split the sample.
* **clustering** — the same equations with standard errors clustered at the
  politician, party and country levels. Party is nested in country here, so the
  country level is the conservative multiway answer.
* **multiplicity** — Holm and Benjamini-Hochberg adjustment of the nine
  ideology p-values in each equation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from .config import CLUSTER_COLUMN, REGION_COLUMN

logger = logging.getLogger(__name__)

COUNTRY_COLUMN = "country_isocode"
OUTCOME = "neutrality_assessment"

# Pre-funnel pool: every politician carrying a full set of V-Party annotations,
# before article availability, the 300-word floor and the identity checks.
POOL_PATH = "data/LeadersAndParties/leaders.csv"
PARTIES_PATH = "data/LeadersAndParties/parties.csv"

# The pre-funnel pool predates the sign-flipping renames applied in the analysis
# frame, so the three reversed dimensions are recreated here on the analysis
# convention (higher means more of the named position).
_POOL_FLIPS = {
    "v2xpa_antiplural": ("v2xpa_plural", lambda x: 1 - x),
    "v2xpa_popul": ("v2xpa_antipopul", lambda x: 1 - x),
    "v2paclient_osp": ("v2paanticlient_osp", lambda x: 4 - x),
}


def _stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def _cell(coef: float, se: float, p: float) -> str:
    return f"{coef:.3f}{_stars(p)} ({se:.3f})"


def _fit(formula: str, data: pd.DataFrame, cluster_col: Optional[str]):
    """OLS with party-clustered SEs when a cluster column is given."""
    model = smf.ols(formula, data=data)
    if cluster_col is None:
        return model.fit()
    return model.fit(
        cov_type="cluster", cov_kwds={"groups": data[cluster_col].astype(str)}
    )


def _per_source_baseline(sub: pd.DataFrame) -> pd.DataFrame:
    """Reorder the judge categories so the per-source intercepts match Eq. 2."""
    from .config import PER_SOURCE_BASELINE_MODEL

    cats = list(sub["Model"].cat.categories)
    if PER_SOURCE_BASELINE_MODEL not in cats:
        return sub
    sub = sub.copy()
    sub["Model"] = sub["Model"].cat.reorder_categories(
        [PER_SOURCE_BASELINE_MODEL]
        + [m for m in cats if m != PER_SOURCE_BASELINE_MODEL]
    )
    return sub


def _coef_frame(res, keep: Sequence[str], label: str) -> pd.DataFrame:
    rows = []
    for name in keep:
        if name not in res.params.index:
            continue
        rows.append(
            {
                "variable": name,
                "spec": label,
                "coef": res.params[name],
                "se": res.bse[name],
                "p": res.pvalues[name],
                "cell": _cell(res.params[name], res.bse[name], res.pvalues[name]),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Country fixed effects
# ---------------------------------------------------------------------------

def country_fixed_effects(
    df: pd.DataFrame,
    df_wiki_minus_groki: pd.DataFrame,
    ideology_vars: List[str],
    cluster_col: Optional[str] = CLUSTER_COLUMN,
) -> Dict[str, pd.DataFrame]:
    """Eq. 1 and Eq. 2 with no geography, region FE, and country FE.

    Country FE are absorbed as ordinary dummies (145 levels over 5,576
    observations), so the ideology coefficients are identified only by
    within-country variation between parties.
    """
    ideology = " + ".join(ideology_vars)
    out = {}

    specs = [
        ("(1) No geography", ""),
        ("(2) Region FE", f" + C({REGION_COLUMN}, Sum)"),
        ("(3) Country FE", f" + C({COUNTRY_COLUMN})"),
    ]

    delta_rows = []
    for label, extra in specs:
        res = _fit(f"{OUTCOME} ~ {ideology}{extra}", df_wiki_minus_groki, cluster_col)
        frame = _coef_frame(res, ["Intercept"] + ideology_vars, label)
        frame["r2"] = res.rsquared
        frame["n"] = int(res.nobs)
        delta_rows.append(frame)
    out["delta"] = pd.concat(delta_rows, ignore_index=True)

    for source, sub in df.groupby("Source", observed=True):
        rows = []
        sub = _per_source_baseline(sub)
        for label, extra in specs:
            res = _fit(
                f"{OUTCOME} ~ {ideology} + C(Model){extra}", sub, cluster_col
            )
            frame = _coef_frame(res, ["Intercept"] + ideology_vars, label)
            frame["r2"] = res.rsquared
            frame["n"] = int(res.nobs)
            rows.append(frame)
        out[f"per_source_{source}"] = pd.concat(rows, ignore_index=True)

    n_countries = df_wiki_minus_groki[COUNTRY_COLUMN].nunique()
    logger.info("Country FE: %d country levels", n_countries)
    out["meta"] = pd.DataFrame(
        [{"n_countries": n_countries, "n_parties": df_wiki_minus_groki[cluster_col].nunique()}]
    )
    return out


# ---------------------------------------------------------------------------
# 2. Funnel selection
# ---------------------------------------------------------------------------

def _load_pool(project_root: Path, ideology_vars: List[str]) -> pd.DataFrame:
    """The 6,846-politician annotated pool, on the analysis sign convention."""
    pool = pd.read_csv(project_root / POOL_PATH, low_memory=False)
    for original, (new, transform) in _POOL_FLIPS.items():
        if original in pool.columns:
            pool[new] = pool[original].apply(transform)
    missing = [v for v in ideology_vars if v not in pool.columns]
    if missing:
        raise KeyError(f"pool is missing ideology columns: {missing}")
    return pool


def funnel_selection(
    df: pd.DataFrame,
    ideology_vars: List[str],
    project_root: Path,
    df_wiki_minus_groki: Optional[pd.DataFrame] = None,
    cluster_col: Optional[str] = CLUSTER_COLUMN,
) -> Dict[str, pd.DataFrame]:
    """Logit of surviving the funnel on ideology, region, sex and office.

    Ideology is z-scored on the *pool*, not the retained sample, so a
    coefficient is the log-odds change in retention per pool standard deviation.

    When ``df_wiki_minus_groki`` is supplied, Eq. 1 is also re-estimated with
    inverse-probability weights from the retention model and separately on the
    two prominence strata, which is what decides whether the selection the logit
    detects actually moves the ideology coefficients.
    """
    from .config import REGIONS_BY_ISO3

    pool = _load_pool(project_root, ideology_vars)
    retained_names = set(df["name"].unique())

    pool = pool.dropna(subset=ideology_vars).copy()
    pool["retained"] = pool["name"].isin(retained_names).astype(int)
    pool[REGION_COLUMN] = pool[COUNTRY_COLUMN].map(REGIONS_BY_ISO3)
    pool["female"] = (pool["gender"].astype(str).str.lower() == "female").astype(int)
    pool["top_office"] = (
        pool["position"]
        .astype(str)
        .str.contains("president|prime min|chancellor|premier", case=False, regex=True)
        .astype(int)
    )

    for var in ideology_vars:
        col = pd.to_numeric(pool[var], errors="coerce")
        pool[var] = (col - col.mean()) / col.std()

    fit_df = pool.dropna(subset=[REGION_COLUMN]).copy()
    ideology = " + ".join(ideology_vars)

    specs = {
        "(1) Ideology": f"retained ~ {ideology}",
        "(2) + Region": f"retained ~ {ideology} + C({REGION_COLUMN})",
        "(3) + Region, sex, office": (
            f"retained ~ {ideology} + C({REGION_COLUMN}) + female + top_office"
        ),
    }

    rows = []
    full_fit = None
    for label, formula in specs.items():
        res = smf.logit(formula, data=fit_df).fit(disp=0)
        frame = _coef_frame(res, ideology_vars + ["female", "top_office"], label)
        frame["pseudo_r2"] = res.prsquared
        frame["n"] = int(res.nobs)
        rows.append(frame)
        full_fit = res

    out: Dict[str, pd.DataFrame] = {}
    if df_wiki_minus_groki is not None and full_fit is not None:
        fit_df["p_retain"] = full_fit.predict(fit_df)
        keys = fit_df.set_index("name")[["p_retain", "top_office"]]
        keys = keys[~keys.index.duplicated()]
        out["reweighted"] = _reweighted_delta(
            df_wiki_minus_groki, keys, ideology_vars, cluster_col
        )

    stages = pd.DataFrame(
        [
            {
                "stage": "annotated pool",
                "n": len(pool),
                "retention_rate": pool["retained"].mean(),
            },
            {
                "stage": "retained pairs",
                "n": int(pool["retained"].sum()),
                "retention_rate": 1.0,
            },
        ]
    )

    out["logit"] = pd.concat(rows, ignore_index=True)
    out["stages"] = stages
    out["descriptives"] = (
        pool.groupby("retained")[ideology_vars].mean().T.reset_index()
    )
    return out


def _reweighted_delta(
    df_wiki_minus_groki: pd.DataFrame,
    keys: pd.DataFrame,
    ideology_vars: List[str],
    cluster_col: Optional[str],
) -> pd.DataFrame:
    """Eq. 1 unweighted, inverse-probability weighted, and split by prominence.

    The IPW weights are ``1 / p_retain`` from the retention logit, so the
    retained sample is projected back onto the annotated pool. Weights are
    trimmed at the 1st and 99th percentiles to keep a handful of very unlikely
    retentions from dominating the fit.
    """
    data = df_wiki_minus_groki.merge(
        keys, left_on="name", right_index=True, how="inner"
    )
    lo, hi = data["p_retain"].quantile([0.01, 0.99])
    data["ipw"] = 1.0 / data["p_retain"].clip(lower=lo, upper=hi)

    ideology = " + ".join(ideology_vars)
    rows = []

    res = _fit(f"{OUTCOME} ~ {ideology}", data, cluster_col)
    frame = _coef_frame(res, ["Intercept"] + ideology_vars, "(1) Unweighted")
    frame["n"] = int(res.nobs)
    rows.append(frame)

    wls = smf.wls(f"{OUTCOME} ~ {ideology}", data=data, weights=data["ipw"]).fit(
        cov_type="cluster", cov_kwds={"groups": data[cluster_col].astype(str)}
    )
    frame = _coef_frame(wls, ["Intercept"] + ideology_vars, "(2) IPW")
    frame["n"] = int(wls.nobs)
    rows.append(frame)

    for value, label in [(1, "(3) Top office"), (0, "(4) Other office")]:
        subset = data[data["top_office"] == value]
        if subset["name"].nunique() < 50:
            continue
        res = _fit(f"{OUTCOME} ~ {ideology}", subset, cluster_col)
        frame = _coef_frame(res, ["Intercept"] + ideology_vars, label)
        frame["n"] = int(res.nobs)
        rows.append(frame)

    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Clustering level
# ---------------------------------------------------------------------------

def clustering_levels(
    df: pd.DataFrame,
    df_wiki_minus_groki: pd.DataFrame,
    ideology_vars: List[str],
) -> Dict[str, pd.DataFrame]:
    """Eq. 1 and Eq. 2 with standard errors clustered at four nesting levels.

    Party is perfectly nested in country in this sample, so a two-way
    party-and-country cluster reduces to the coarser of the two. Clustering on
    the country is therefore the conservative multiway answer, and it is
    reported here alongside the politician and party levels.
    """
    ideology = " + ".join(ideology_vars)
    levels = [
        ("Politician", "name"),
        ("Party", CLUSTER_COLUMN),
        ("Country", COUNTRY_COLUMN),
    ]

    nesting = pd.DataFrame(
        [
            {
                "parties": df_wiki_minus_groki[CLUSTER_COLUMN].nunique(),
                "countries": df_wiki_minus_groki[COUNTRY_COLUMN].nunique(),
                "parties_spanning_countries": int(
                    (
                        df_wiki_minus_groki.groupby(CLUSTER_COLUMN)[COUNTRY_COLUMN]
                        .nunique()
                        > 1
                    ).sum()
                ),
            }
        ]
    )

    out = {"nesting": nesting}

    rows = []
    for label, col in levels:
        res = _fit(f"{OUTCOME} ~ {ideology}", df_wiki_minus_groki, col)
        frame = _coef_frame(res, ["Intercept"] + ideology_vars, label)
        frame["n_clusters"] = df_wiki_minus_groki[col].nunique()
        rows.append(frame)
    out["delta"] = pd.concat(rows, ignore_index=True)

    for source, sub in df.groupby("Source", observed=True):
        sub = _per_source_baseline(sub)
        rows = []
        for label, col in levels:
            res = _fit(f"{OUTCOME} ~ {ideology} + C(Model)", sub, col)
            frame = _coef_frame(res, ["Intercept"] + ideology_vars, label)
            frame["n_clusters"] = sub[col].nunique()
            rows.append(frame)
        out[f"per_source_{source}"] = pd.concat(rows, ignore_index=True)

    return out


# ---------------------------------------------------------------------------
# 4. Ideology staleness
# ---------------------------------------------------------------------------

def _election_year_gap(df: pd.DataFrame, project_root: Path) -> pd.Series:
    """Years between a politician's WhoGov year and their V-Party election year.

    Reconstructs the nearest-year match used to build the dataset: for each
    politician, the party-election row (same Party Facts id) whose year is
    closest to theirs.
    """
    parties = pd.read_csv(project_root / PARTIES_PATH, low_memory=False)
    parties = parties[["pf_party_id", "year"]].dropna()
    parties["pf_party_id"] = parties["pf_party_id"].astype(float)

    by_party = parties.groupby("pf_party_id")["year"].apply(list).to_dict()

    gaps = []
    for pf, year in zip(df["partyfacts_id"], df["year"]):
        years = by_party.get(float(pf)) if pd.notna(pf) else None
        gaps.append(min(abs(y - year) for y in years) if years else np.nan)
    return pd.Series(gaps, index=df.index, dtype=float)


def ideology_staleness(
    df_wiki_minus_groki: pd.DataFrame,
    ideology_vars: List[str],
    project_root: Path,
    cluster_col: Optional[str] = CLUSTER_COLUMN,
) -> Dict[str, pd.DataFrame]:
    """Eq. 1 with the election-year gap as a covariate and as a sample split."""
    data = df_wiki_minus_groki.copy()
    data["year_gap"] = _election_year_gap(data, project_root)
    have_gap = data.dropna(subset=["year_gap"]).copy()
    have_gap["year_gap_z"] = (
        have_gap["year_gap"] - have_gap["year_gap"].mean()
    ) / have_gap["year_gap"].std()

    ideology = " + ".join(ideology_vars)
    median_gap = have_gap["year_gap"].median()

    specs = [
        ("(1) Baseline", f"{OUTCOME} ~ {ideology}", have_gap),
        ("(2) + Year gap", f"{OUTCOME} ~ {ideology} + year_gap_z", have_gap),
        (
            f"(3) Gap <= {median_gap:.0f}y",
            f"{OUTCOME} ~ {ideology}",
            have_gap[have_gap["year_gap"] <= median_gap],
        ),
    ]

    rows = []
    for label, formula, subset in specs:
        res = _fit(formula, subset, cluster_col)
        frame = _coef_frame(res, ideology_vars + ["year_gap_z", "Intercept"], label)
        frame["r2"] = res.rsquared
        frame["n"] = int(res.nobs)
        rows.append(frame)

    gap_summary = (
        have_gap.drop_duplicates("name")["year_gap"]
        .describe(percentiles=[0.25, 0.5, 0.75, 0.9])
        .to_frame("year_gap")
        .reset_index()
    )

    return {"fits": pd.concat(rows, ignore_index=True), "gap_summary": gap_summary}


# ---------------------------------------------------------------------------
# 5. Multiplicity
# ---------------------------------------------------------------------------

def multiplicity(
    df: pd.DataFrame,
    df_wiki_minus_groki: pd.DataFrame,
    ideology_vars: List[str],
    cluster_col: Optional[str] = CLUSTER_COLUMN,
) -> pd.DataFrame:
    """Holm and Benjamini-Hochberg adjustment across the nine ideology tests."""
    ideology = " + ".join(ideology_vars)
    frames = []

    fits = {"Eq. 1 (differential)": (f"{OUTCOME} ~ {ideology}", df_wiki_minus_groki)}
    for source, sub in df.groupby("Source", observed=True):
        fits[f"Eq. 2 ({source})"] = (
            f"{OUTCOME} ~ {ideology} + C(Model)", _per_source_baseline(sub)
        )

    for label, (formula, data) in fits.items():
        res = _fit(formula, data, cluster_col)
        p = np.array([res.pvalues[v] for v in ideology_vars])
        frames.append(
            pd.DataFrame(
                {
                    "equation": label,
                    "variable": ideology_vars,
                    "coef": [res.params[v] for v in ideology_vars],
                    "p_raw": p,
                    "p_holm": multipletests(p, method="holm")[1],
                    "p_bh": multipletests(p, method="fdr_bh")[1],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_identification_checks(
    df: pd.DataFrame,
    df_wiki_minus_groki: pd.DataFrame,
    ideology_vars: List[str],
    table_dir: Path,
    project_root: Path,
    cluster_col: Optional[str] = CLUSTER_COLUMN,
) -> None:
    out_dir = Path(table_dir) / "identification"
    out_dir.mkdir(parents=True, exist_ok=True)

    country = country_fixed_effects(df, df_wiki_minus_groki, ideology_vars, cluster_col)
    for key, frame in country.items():
        frame.to_csv(out_dir / f"country_fe_{key}.csv", index=False)

    selection = funnel_selection(
        df,
        ideology_vars,
        project_root,
        df_wiki_minus_groki=df_wiki_minus_groki,
        cluster_col=cluster_col,
    )
    for key, frame in selection.items():
        frame.to_csv(out_dir / f"selection_{key}.csv", index=False)

    stale = ideology_staleness(
        df_wiki_minus_groki, ideology_vars, project_root, cluster_col
    )
    for key, frame in stale.items():
        frame.to_csv(out_dir / f"staleness_{key}.csv", index=False)

    clusters = clustering_levels(df, df_wiki_minus_groki, ideology_vars)
    for key, frame in clusters.items():
        frame.to_csv(out_dir / f"clustering_{key}.csv", index=False)

    multiplicity(df, df_wiki_minus_groki, ideology_vars, cluster_col).to_csv(
        out_dir / "multiplicity.csv", index=False
    )

    logger.info("Identification checks written to %s", out_dir)
