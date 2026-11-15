"""Linear mixed-effects models with a per-country random intercept.

These complement the fixed-effects OLS/ordinal models in :mod:`stat_models` by
treating the leader's *country* as a random grouping factor instead of (or
alongside) regional fixed effects. Three models are fit, mirroring the existing
specifications:

* **differentiation** — outcome is the within-leader Wikipedia−Grokipedia
  difference (the ``df_wiki_minus_groki`` frame), pooled across raters. Mirrors
  ``wiki_minus_groki_pooled_results``.
* **per_source/<source>** — one model per encyclopedia, outcome is the raw
  neutrality assessment with a ``C(Model)`` rater main effect. Mirrors the
  per-source OLS panels behind ``ols_coefficients_grid``.

Every model carries **regional fixed effects** (one coefficient per region, same
coding as the corresponding OLS model) *and* a **random intercept for
``country_isocode``** — country is nested within region, so the region terms
capture between-region means while the random intercept captures the residual
country-to-country spread.

For each model we save the standard coefficient plot + summary table in a new
``mixed_effects/`` subfolder, then extract the estimated random effects (BLUPs)
per country, rank them, write them to CSV, and plot them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .config import REGION_COLUMN
from .stat_models import ModelResults
from .visualisation.helpers import save_figure, setup_plot_style
from .visualisation.stat_figures import plot_ols_coefficients, plot_coefficients_grid

logger = logging.getLogger(__name__)

OUTCOME_COLUMN = "neutrality_assessment"
COUNTRY_GROUP_COLUMN = "country_isocode"

# Countries with fewer politicians than this are dropped from the BLUP plot —
# their random intercepts are shrunk heavily towards zero and the rank is noise.
MIN_POLITICIANS_PER_COUNTRY = 10


def _politicians_per_country(df: pd.DataFrame) -> pd.Series:
    """Unique politicians per country (each leader counted once, not per rater)."""
    name_col = "name" if "name" in df.columns else "Name"
    return df.groupby(COUNTRY_GROUP_COLUMN)[name_col].nunique()


class _FixedEffectsView:
    """Adapter exposing only the *fixed-effect* params of a MixedLM result.

    The OLS coefficient plotter (`_extract_ols_rows`) reads ``params``,
    ``pvalues`` and ``conf_int`` and treats every row as a coefficient. A fitted
    MixedLM additionally carries the random-effect variance component (e.g.
    ``country_isocode Var``); this view drops it so the coefplot shows only the
    regression coefficients, exactly like the OLS figures.
    """

    def __init__(self, res):
        self._res = res
        self._fe_names = list(res.fe_params.index)

    @property
    def params(self) -> pd.Series:
        return self._res.fe_params

    @property
    def pvalues(self) -> pd.Series:
        return self._res.pvalues.loc[self._fe_names]

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        return self._res.conf_int(alpha=alpha).loc[self._fe_names]

    @property
    def fit_stats_label(self) -> Optional[str]:
        """Nakagawa marginal/conditional R² for the coefplot annotation.

        Marginal: share of variance explained by the fixed effects alone;
        conditional: fixed effects + the country random intercept. Uses
        ``fittedvalues`` (fixed-effects predictions only in MixedLM) for the
        fixed-effect variance, ``cov_re × scale`` for the RE variance and
        ``scale`` for the residual variance.
        """
        try:
            res = self._res
            var_fixed = float(np.var(np.asarray(res.fittedvalues)))
            var_re = float(np.asarray(res.cov_re).reshape(-1)[0]) * float(res.scale)
            var_resid = float(res.scale)
            total = var_fixed + var_re + var_resid
            if total <= 0:
                return None
            r2m = var_fixed / total
            r2c = (var_fixed + var_re) / total
            return f"marg. $R^2$ = {r2m:.3f}   cond. $R^2$ = {r2c:.3f}"
        except Exception:
            return None


def _as_ols_like(res) -> ModelResults:
    """Wrap a MixedLM result so it can flow through the OLS plotting helpers."""
    return ModelResults(model_type="ols", results=_FixedEffectsView(res), anova_table=None)


def _ideology_terms(df: pd.DataFrame, ideology_vars: Sequence[str]) -> List[str]:
    return [v for v in ideology_vars if v in df.columns]


def fit_country_mixed_model(
    df: pd.DataFrame,
    ideology_vars: Sequence[str],
    extra_fixed: Optional[Sequence[str]] = None,
    region_coding: str = "treatment",
    outcome_col: str = OUTCOME_COLUMN,
):
    """Fit ``outcome ~ extra_fixed + ideology + region`` with a country random intercept.

    Args:
        extra_fixed: raw patsy terms added as fixed effects (e.g. ``["C(Model)"]``).
        region_coding: ``"treatment"`` (``C(region)``; one coef per non-reference
            region) or ``"sum"`` (``C(region, Sum)``; deviations from the grand
            mean). Matches whichever coding the analogous OLS model uses.

    Returns the fitted ``MixedLMResults`` (or ``None`` if the data is unusable).
    """
    terms: List[str] = list(extra_fixed or [])
    terms += _ideology_terms(df, ideology_vars)

    needed = [outcome_col, COUNTRY_GROUP_COLUMN]
    if REGION_COLUMN in df.columns:
        region_term = f"C({REGION_COLUMN}, Sum)" if region_coding == "sum" else f"C({REGION_COLUMN})"
        terms.append(region_term)
        needed.append(REGION_COLUMN)
    else:
        logger.warning("No '%s' column; fitting mixed model without regional fixed effects.", REGION_COLUMN)

    if not terms:
        logger.warning("No fixed-effect terms available; skipping mixed model.")
        return None

    data = df.dropna(subset=needed).copy()
    if data.empty or data[COUNTRY_GROUP_COLUMN].nunique() < 2:
        logger.warning("Too few countries to fit a random intercept; skipping.")
        return None

    formula = f"{outcome_col} ~ " + " + ".join(terms)
    print(f"Mixed model formula:\n{formula}\n  groups = {COUNTRY_GROUP_COLUMN} "
          f"({data[COUNTRY_GROUP_COLUMN].nunique()} countries, {len(data)} rows)")

    model = smf.mixedlm(formula, data, groups=data[COUNTRY_GROUP_COLUMN])
    # re_formula defaults to a random intercept ("~1"). NB: the default "lbfgs"
    # optimizer spuriously collapses the random-effect variance to the boundary
    # (GroupVar→0) on this data; bfgs/cg/powell all recover the same non-zero
    # variance, so we try those in sequence instead.
    res = model.fit(method=["bfgs", "cg", "powell"], maxiter=200)
    return res


def extract_country_blups(res) -> pd.DataFrame:
    """Extract per-country random-intercept estimates (BLUPs), ranked ascending.

    Returns a DataFrame with columns ``country``, ``blup`` (the estimated random
    intercept = country-level deviation from the fixed-effect prediction), sorted
    from most-negative to most-positive.

    Returns an empty DataFrame if the random-effect variance collapsed to the
    boundary (≈0): statsmodels cannot invert the singular covariance, and there is
    no meaningful country-level spread left to rank.
    """
    try:
        re_dict = res.random_effects  # {country: Series indexed by RE design columns}
    except (np.linalg.LinAlgError, ValueError) as exc:
        logger.warning(
            "Country random-intercept variance ≈ 0 (singular covariance); no BLUPs "
            "to extract: %s", exc,
        )
        return pd.DataFrame(columns=["country", "blup"])

    rows = []
    for country, series in re_dict.items():
        # Random-intercept-only model -> a single value per group; sum() is robust
        # whether the intercept column is named "Group", "Intercept" or "1".
        rows.append({"country": str(country), "blup": float(np.asarray(series).sum())})
    return pd.DataFrame(rows).sort_values("blup", ascending=True, ignore_index=True)


def plot_country_blups(
    blup_df: pd.DataFrame,
    save_path: Path,
    title: str,
    re_sd: Optional[float] = None,
    min_politicians: int = 0,
) -> None:
    """Ranked horizontal bar chart of per-country BLUPs (random intercepts).

    Green = country reads more *biased-in-favour* than its region/fixed-effect
    baseline; red = more *biased-against*. Dashed lines mark ±1 random-effect SD.
    When *blup_df* carries an ``n_politicians`` column, the count is appended to
    each country label and countries below *min_politicians* are dropped.
    """
    if blup_df.empty:
        return
    setup_plot_style()

    ordered = blup_df.sort_values("blup", ascending=True, ignore_index=True)
    if "n_politicians" in ordered.columns:
        if min_politicians > 0:
            kept = ordered["n_politicians"].fillna(0) >= min_politicians
            dropped = int((~kept).sum())
            if dropped:
                print(f"BLUP plot: dropping {dropped} countries with < {min_politicians} politicians")
            ordered = ordered[kept].reset_index(drop=True)
            if ordered.empty:
                return
        labels = [f"{c} ({int(k)})" for c, k in zip(ordered["country"], ordered["n_politicians"])]
    else:
        labels = list(ordered["country"])
    colors = ["#C44E52" if v < 0 else "#55A868" for v in ordered["blup"]]

    n = len(ordered)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.18 * n)))
    ax.barh(np.arange(n), ordered["blup"], color=colors)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels, fontsize=6 if n > 60 else 8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    if re_sd is not None and np.isfinite(re_sd) and re_sd > 0:
        for s in (-re_sd, re_sd):
            ax.axvline(s, color="#888888", linestyle="--", linewidth=0.8)
        ax.text(re_sd, n - 0.5, "  ±1 SD", fontsize=7, color="#555555", va="top")
    ax.set_xlabel("Random intercept (BLUP)  — biased against   ·   + biased in favour")
    ax.set_ylabel("Country (ISO3, n politicians)" if "n_politicians" in ordered.columns else "Country (ISO3)")
    ax.set_title(title)
    ax.set_ylim(-0.5, n - 0.5)
    fig.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)


def _random_effect_sd(res) -> Optional[float]:
    """Estimated SD of the country random intercept (sqrt of the variance component)."""
    try:
        # cov_re is the random-effect covariance (1x1 for an intercept-only model),
        # scaled by the residual variance (res.scale).
        var = float(np.asarray(res.cov_re).reshape(-1)[0]) * float(res.scale)
        return float(np.sqrt(var)) if var > 0 else None
    except Exception:
        return None


def _intraclass_correlation(res) -> Optional[float]:
    """ICC = country variance / (country variance + residual variance)."""
    try:
        re_var = float(np.asarray(res.cov_re).reshape(-1)[0]) * float(res.scale)
        resid_var = float(res.scale)
        total = re_var + resid_var
        return re_var / total if total > 0 else None
    except Exception:
        return None


def plot_blup_distribution(
    blup_df: pd.DataFrame,
    save_path: Path,
    title: str,
    re_sd: Optional[float] = None,
    icc: Optional[float] = None,
    n_extreme_labels: int = 5,
) -> None:
    """Histogram + kernel density of the per-country BLUPs.

    Annotated with the model-estimated random-intercept SD (dashed ±1 SD lines)
    and the ICC (share of residual variance attributable to country). The
    *n_extreme_labels* most negative and most positive countries are labelled
    along the bottom of the plot.
    """
    if blup_df.empty:
        return
    setup_plot_style()

    values = blup_df["blup"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=min(30, max(10, len(values) // 4)), density=True,
            color="#4C72B0", alpha=0.55, edgecolor="white")
    if len(values) > 1 and np.std(values) > 0:
        from scipy.stats import gaussian_kde
        grid = np.linspace(values.min(), values.max(), 200)
        ax.plot(grid, gaussian_kde(values)(grid), color="#4C72B0", linewidth=1.5)

    ax.axvline(0.0, color="black", linewidth=0.8)
    if re_sd is not None and np.isfinite(re_sd) and re_sd > 0:
        for s in (-re_sd, re_sd):
            ax.axvline(s, color="#888888", linestyle="--", linewidth=0.8)

    # Label the most extreme countries in each tail: a rug tick at the BLUP
    # value plus the ISO code above it, staggered over two heights so adjacent
    # labels don't collide.
    if n_extreme_labels > 0 and len(blup_df) > 2 * n_extreme_labels:
        ordered = blup_df.sort_values("blup", ignore_index=True)
        extremes = pd.concat([ordered.head(n_extreme_labels), ordered.tail(n_extreme_labels)])
    else:
        extremes = blup_df
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.05)
    for i, row in enumerate(extremes.itertuples(index=False)):
        ax.plot([row.blup, row.blup], [0, ymax * 0.03], color="#C44E52" if row.blup < 0 else "#55A868", linewidth=1.2)
        ax.text(
            row.blup, ymax * (0.05 if i % 2 == 0 else 0.11), str(row.country),
            rotation=90, ha="center", va="bottom", fontsize=7, color="#333333",
        )

    annot_lines = [f"countries = {len(values)}"]
    if re_sd is not None and np.isfinite(re_sd):
        annot_lines.append(f"RE SD = {re_sd:.3f}")
    annot_lines.append(f"empirical SD = {np.std(values, ddof=1):.3f}")
    if icc is not None and np.isfinite(icc):
        annot_lines.append(f"ICC = {icc:.3f}")
    ax.text(
        0.02, 0.98, "\n".join(annot_lines), transform=ax.transAxes,
        va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    ax.set_xlabel("Random intercept (BLUP)  — biased against   ·   + biased in favour")
    ax.set_ylabel("Density")
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, save_path)
    plt.close(fig)


def _save_mixed_summary(res, table_path: Path, label: str) -> None:
    """Write the MixedLM summary (fixed effects + variance components) to disk."""
    table_path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "=" * 80,
        f"MIXED-EFFECTS MODEL — {label}",
        f"(random intercept: {COUNTRY_GROUP_COLUMN})",
        "=" * 80,
        "",
        str(res.summary()),
    ]
    table_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Saved mixed-model summary ({label}) to {table_path}")


def _run_one_mixed(
    df: pd.DataFrame,
    label: str,
    ideology_vars: Sequence[str],
    figure_dir: Path,
    table_dir: Path,
    extra_fixed: Optional[Sequence[str]] = None,
    region_coding: str = "treatment",
    outcome_col: str = OUTCOME_COLUMN,
):
    """Fit one mixed model and save coefplot, summary table, BLUP CSV + plot."""
    res = fit_country_mixed_model(
        df, ideology_vars, extra_fixed=extra_fixed,
        region_coding=region_coding, outcome_col=outcome_col,
    )
    if res is None:
        return None

    safe = label.replace("/", "_")
    fig_subdir = figure_dir / "mixed_effects" / label
    tbl_subdir = table_dir / "mixed_effects"
    fig_subdir.mkdir(parents=True, exist_ok=True)
    tbl_subdir.mkdir(parents=True, exist_ok=True)

    # Standard coefficient plot (fixed effects only) + summary table.
    plot_ols_coefficients(
        _as_ols_like(res),
        save_path=fig_subdir / "00_coefficients.png",
        show_intercept=True,
    )
    _save_mixed_summary(res, tbl_subdir / f"{safe}_summary.txt", label)

    # Random effects (BLUPs): extract, rank, save, plot.
    blups = extract_country_blups(res)
    if not blups.empty:
        blups["n_politicians"] = blups["country"].map(_politicians_per_country(df))
    blup_csv = tbl_subdir / f"{safe}_country_blups.csv"
    blups.to_csv(blup_csv, index=False)
    print(f"Saved country BLUPs ({label}) to {blup_csv}")

    plot_country_blups(
        blups,
        save_path=fig_subdir / "01_country_blups.png",
        title=f"Country random intercepts (BLUPs) — {label}",
        re_sd=_random_effect_sd(res),
        min_politicians=MIN_POLITICIANS_PER_COUNTRY,
    )

    # Distribution of the country intercepts (all countries, no min-n filter).
    plot_blup_distribution(
        blups,
        save_path=fig_subdir / "02_blup_distribution.png",
        title=f"Distribution of country random intercepts — {label}",
        re_sd=_random_effect_sd(res),
        icc=_intraclass_correlation(res),
    )
    return res


def run_mixed_effects_analysis(
    df: pd.DataFrame,
    df_wiki_minus_groki: pd.DataFrame,
    ideology_vars: Sequence[str],
    figure_dir: Path,
    table_dir: Path,
) -> None:
    """Fit the differentiation + per-source mixed models and emit all outputs.

    Files land under ``figure_dir/mixed_effects/`` and ``table_dir/mixed_effects/``.
    """
    figure_dir = Path(figure_dir)
    table_dir = Path(table_dir)

    print("\n" + "=" * 80)
    print("MIXED-EFFECTS MODELS (country random intercept)")
    print("=" * 80)

    # 1. Differentiation model (Wikipedia − Grokipedia difference), pooled across
    #    raters. Region sum-coded to match wiki_minus_groki_pooled_results.
    _run_one_mixed(
        df_wiki_minus_groki,
        label="differentiation",
        ideology_vars=ideology_vars,
        figure_dir=figure_dir,
        table_dir=table_dir,
        extra_fixed=None,
        region_coding="sum",
    )

    # 2. Per-source models (mirror the ols_coefficients_grid panels). Region
    #    treatment-coded and a C(Model) rater main effect, matching per_source OLS.
    per_source_panels: List[Tuple[str, ModelResults]] = []
    panel_order = [("wikipedia", "Wikipedia"), ("grokipedia_0.2", "Grokipedia")]
    for source, panel_title in panel_order:
        sub = df[df["Source"] == source]
        if sub.empty:
            continue
        res = _run_one_mixed(
            sub,
            label=f"per_source/{source}",
            ideology_vars=ideology_vars,
            figure_dir=figure_dir,
            table_dir=table_dir,
            extra_fixed=["C(Model)"],
            region_coding="treatment",
        )
        if res is not None:
            per_source_panels.append((panel_title, _as_ols_like(res)))

    # Combined side-by-side coefplot grid, mirroring per_source/ols_coefficients_grid.png.
    if per_source_panels:
        plot_coefficients_grid(
            panels=per_source_panels,
            save_path=figure_dir / "mixed_effects" / "per_source" / "ols_coefficients_grid.png",
            show_intercept=True,
            shared_x=True,
        )
