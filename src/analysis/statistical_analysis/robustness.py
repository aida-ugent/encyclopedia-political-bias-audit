"""Robustness checks (appendix tables and stability figure) for the ideology
regressions.

Three nested specifications are estimated for each equation:

  (1) Baseline           — the 9 ideology dimensions only
  (2) + Article length   — adds the article-length control
  (3) + Length + Region  — additionally adds V-Dem regional fixed effects

Eq. 1 is the differential (wiki − groki) regression; its length control is the
wiki−groki word-count difference. Eq. 2 is the per-source regression; its length
control is the (within-source z-scored) word count of that source's article. The
regional fixed effects are sum/deviation-coded in both equations, so each region
coefficient is a deviation from the grand mean and the intercept stays an
average over regions, comparable across the three specifications.

Everything here uses OLS — the conventional choice for a coefficient-comparison
appendix table, and directly interpretable on the outcome scale.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from .config import REGION_COLUMN, PER_SOURCE_BASELINE_MODEL, MODEL_DISPLAY
from .stat_models import fit_ols_model, ModelResults
from .visualisation.helpers import (
    pretty_dep_label, pretty_model, setup_plot_style, save_figure,
    compute_article_length,
)
from .visualisation.stat_figures import _clean_ols_param_name, _extract_ols_rows

# Coefficients highlighted in the stability figure (the "surviving" findings).
KEY_VARS = ["v2palgbt_osp", "v2xpa_antipopul", "v2pariglef_osp", "v2xpa_plural"]

_SPEC_LABELS = ("(1) Baseline", "(2) + Length", "(3) + Length + Region")
_STABILITY_COLORS = ("#1b9e77", "#7570b3", "#d95f02")  # one per specification


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _cell(coef: float, se: float, p: float) -> str:
    return f"{coef:.3f}{_stars(p)} ({se:.3f})"


def _display_label(label: str) -> str:
    """Strip the encoding prefix from dummy labels for a cleaner table row."""
    for prefix in (f"{REGION_COLUMN}_", "Model_"):
        if label.startswith(prefix):
            return label[len(prefix):]
    return label


def _ols_coef_map(res: ModelResults) -> dict:
    """Map pretty predictor label -> (coef, se, p) for an OLS fit."""
    r = res.results
    out = {}
    for name in r.params.index:
        label = "Intercept" if name == "Intercept" else pretty_dep_label(_clean_ols_param_name(name))
        out[label] = (r.params[name], r.bse[name], r.pvalues[name])
    return out


def _comparison_table(specs: Sequence[Tuple[str, ModelResults]]) -> pd.DataFrame:
    """Assemble a coefficient-comparison DataFrame (rows = predictors, cols = specs)."""
    # Canonical row order taken from the richest spec (it is a superset).
    _, _, ordered_labels, _ = _extract_ols_rows(specs[-1][1], alpha=0.05, show_intercept=True)
    coef_maps = [(label, _ols_coef_map(res)) for label, res in specs]

    rows = []
    for lbl in ordered_labels:
        row = {"Variable": _display_label(lbl)}
        for spec_label, cmap in coef_maps:
            row[spec_label] = _cell(*cmap[lbl]) if lbl in cmap else ""
        rows.append(row)

    # Fit statistics footer.
    n_row = {"Variable": "Observations"}
    r2_row = {"Variable": "R-squared"}
    for spec_label, res in specs:
        n_row[spec_label] = f"{int(res.results.nobs)}"
        r2_row[spec_label] = f"{res.results.rsquared:.3f}"
    rows.extend([n_row, r2_row])

    return pd.DataFrame(rows)


def _write_table(table: pd.DataFrame, title: str, notes: List[str], txt_path: Path, csv_path: Path) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [title, "=" * len(title), "", table.to_string(index=False), ""]
    lines.extend(notes)
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    table.to_csv(csv_path, index=False)
    print(f"Saved {txt_path}")
    print(f"Saved {csv_path}")


_TABLE_NOTES = [
    "Cells: OLS coefficient (cluster-robust SE in parentheses), clustered by party.",
    "Significance: * p<0.05, ** p<0.01, *** p<0.001.",
    "Regions are sum-coded: each coefficient is the deviation from the grand mean (no baseline level).",
    "Judge reference level: Mistral (as in the per-source table of the appendix).",
]

# Eq. 1 (differential) sum/deviation-codes the regional fixed effects, so each
# region coefficient is a deviation from the grand mean rather than a contrast
# against a dropped baseline level.
_EQ1_TABLE_NOTES = [
    "Cells: OLS coefficient (cluster-robust SE in parentheses), clustered by party.",
    "Significance: * p<0.05, ** p<0.01, *** p<0.001.",
    "Regions are sum-coded: each coefficient is the deviation from the grand mean (no baseline level).",
]


def _eq1_specs(df_wmg: pd.DataFrame, ideology_vars: List[str], cluster_col: Optional[str]) -> List[Tuple[str, ModelResults]]:
    """Eq. 1: differential (wiki − groki) pooled OLS, nested controls."""
    control_ladder = [[], ["article_length_diff"], ["article_length_diff", REGION_COLUMN]]
    specs = []
    for label, controls in zip(_SPEC_LABELS, control_ladder):
        res = fit_ols_model(
            df_wmg, ideology_vars, cluster_col=cluster_col, model_col="Model",
            use_source=False, use_model=False, control_vars=(controls or None),
            sum_code_controls=[REGION_COLUMN],
        )
        specs.append((label, res))
    return specs


def _eq2_specs(df: pd.DataFrame, source: str, ideology_vars: List[str], cluster_col: Optional[str]) -> List[Tuple[str, ModelResults]]:
    """Eq. 2: per-source OLS (Model main-effect control), nested controls."""
    sub = df[df["Source"] == source].copy()
    # Mistral as the dropped judge reference, mirroring the per-source fits in
    # main.py so the '(1) Baseline' column reproduces the per-source table of the
    # appendix (which the paper quotes) rather than a Grok-referenced variant.
    model_cats = list(sub["Model"].cat.categories)
    if PER_SOURCE_BASELINE_MODEL in model_cats:
        sub["Model"] = sub["Model"].cat.reorder_categories(
            [PER_SOURCE_BASELINE_MODEL]
            + [m for m in model_cats if m != PER_SOURCE_BASELINE_MODEL]
        )
    # Source's own article length, z-scored within the source.
    article_col = source.replace(".", "_")  # 'grokipedia_0.2' -> 'grokipedia_0_2'
    length = sub[article_col].apply(compute_article_length).astype(float)
    sub["article_length"] = (length - length.mean()) / length.std()

    control_ladder = [[], ["article_length"], ["article_length", REGION_COLUMN]]
    specs = []
    for label, controls in zip(_SPEC_LABELS, control_ladder):
        res = fit_ols_model(
            sub, ideology_vars, cluster_col=cluster_col, model_col="Model",
            use_source=False, use_model=True, use_model_dep_interactions=False,
            control_vars=(controls or None),
            sum_code_controls=[REGION_COLUMN],
        )
        specs.append((label, res))
    return specs


def plot_coefficient_stability(
    specs: Sequence[Tuple[str, ModelResults]],
    key_vars: Sequence[str],
    save_path: Path,
    title: str = "Coefficient stability across specifications (Eq. 1)",
) -> None:
    """Dot-and-whisker plot of the key coefficients across the nested specs.

    Filled marker = significant (p<0.05); open marker = not significant. Colour
    encodes the specification.
    """
    setup_plot_style()
    n = len(key_vars)
    offsets = np.linspace(0.22, -0.22, len(specs))

    fig, ax = plt.subplots(figsize=(7.0, max(2.6, 0.9 * n + 1.2)))
    for vi, var in enumerate(key_vars):
        y0 = n - 1 - vi  # first listed variable at the top
        for si, (spec_label, res) in enumerate(specs):
            r = res.results
            if var not in r.params.index:
                continue
            c, se, p = r.params[var], r.bse[var], r.pvalues[var]
            yy = y0 + offsets[si]
            color = _STABILITY_COLORS[si % len(_STABILITY_COLORS)]
            ax.plot([c - 1.96 * se, c + 1.96 * se], [yy, yy], color=color, lw=1.3, alpha=0.85, zorder=2)
            ax.scatter(
                [c], [yy], s=46, zorder=3, marker="o",
                facecolor=(color if p < 0.05 else "white"),
                edgecolor=color, linewidths=1.4,
            )

    ax.axvline(0, color="#9a9a9a", lw=0.8, linestyle="--")
    ax.set_yticks(range(n))
    ax.set_yticklabels([pretty_dep_label(v) for v in reversed(list(key_vars))], fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.grid(True, axis="x", alpha=0.3, linestyle=":", linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    from matplotlib.lines import Line2D
    spec_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_STABILITY_COLORS[i % len(_STABILITY_COLORS)],
               markeredgecolor=_STABILITY_COLORS[i % len(_STABILITY_COLORS)], markersize=8, label=lbl)
        for i, (lbl, _) in enumerate(specs)
    ]
    sig_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#444444", markeredgecolor="#444444", markersize=8, label="p<0.05"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="#444444", markersize=8, label="n.s."),
    ]
    ax.legend(handles=spec_handles + sig_handles, fontsize=9, loc="best", frameon=False, ncol=2)

    plt.tight_layout()
    # No background patches, as with the other coefplots — see
    # `stat_figures._TRANSPARENT_BACKGROUND`.
    save_figure(fig, save_path, dpi=300, transparent=True)
    if save_path is not None:
        save_figure(fig, Path(save_path).with_suffix(".pdf"), dpi=300,
                    transparent=True)
    plt.close()


def _region_increment_row(
    df: pd.DataFrame,
    label: str,
    ideology_vars: List[str],
    outcome_col: str,
    cluster_col: Optional[str],
    use_model: bool = False,
) -> dict:
    """Compare a baseline vs baseline+region OLS; return fit/ΔR²/partial-R²/F/Wald.

    ``use_model`` mirrors the equation's own specification so the reported R² are
    directly comparable to the appendix ladders: Eq. 2 (per source) carries judge
    dummies as a main-effect control in every spec, Eq. 1 (differential) carries
    none. Without this the baseline R² here would not match the ones quoted for
    the same equation elsewhere.

    The classic incremental F-test assumes i.i.d. errors; alongside it we report a
    cluster-robust joint Wald test (region dummies = 0), clustered by party, which
    respects the repeated-rater structure used throughout the analysis.
    """
    ideo_terms = [v for v in ideology_vars if v in df.columns]
    # Judges-only (Eq. 2) / intercept-only (Eq. 1) floor: the common reference the
    # ideology and region blocks are each measured against, so the two increments
    # quoted in the paper are on the same footing.
    f_floor = f"{outcome_col} ~ " + ("C(Model)" if use_model else "1")
    f_base = f_floor + " + " + " + ".join(ideo_terms)
    f_full = f"{f_base} + C({REGION_COLUMN})"
    # Ideology block on its own, without the judge dummies. The design is fully
    # balanced (every article rated by every judge), so the judge dummies are
    # orthogonal to the leader-level ideology scores and this R2 coincides with
    # Delta R2 (ideology) below. Reporting it lets the paper quote the ideology
    # contribution as a plain R2 rather than as an increment.
    f_ideo_only = f"{outcome_col} ~ " + " + ".join(ideo_terms)

    m_floor = smf.ols(f_floor, data=df).fit()
    m_base = smf.ols(f_base, data=df).fit()
    m_full = smf.ols(f_full, data=df).fit()
    m_ideo_only = smf.ols(f_ideo_only, data=df).fit()

    av = anova_lm(m_base, m_full)
    f_stat = float(av["F"].iloc[-1])
    f_p = float(av["Pr(>F)"].iloc[-1])
    df1 = int(av["df_diff"].iloc[-1])
    df2 = int(av["df_resid"].iloc[-1])
    delta_r2_ideo = m_base.rsquared - m_floor.rsquared
    delta_r2 = m_full.rsquared - m_base.rsquared
    partial_r2 = (m_base.ssr - m_full.ssr) / m_base.ssr

    av_ideo = anova_lm(m_floor, m_base)
    f_p_ideo = float(av_ideo["Pr(>F)"].iloc[-1])

    wald_chi2 = float("nan")
    wald_p = float("nan")
    if cluster_col is not None and cluster_col in df.columns:
        m_full_clu = smf.ols(f_full, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df[cluster_col]}
        )
        names = list(m_full_clu.params.index)
        region_params = [p for p in names if p.startswith(f"C({REGION_COLUMN})")]
        R = np.zeros((len(region_params), len(names)))
        for i, p in enumerate(region_params):
            R[i, names.index(p)] = 1.0
        wald = m_full_clu.wald_test(R, use_f=False)
        wald_chi2 = float(np.squeeze(wald.statistic))
        wald_p = float(np.squeeze(wald.pvalue))

    return {
        "Equation": label,
        "Judge dummies": "yes" if use_model else "no",
        "R2 (floor)": f"{m_floor.rsquared:.4f}",
        "R2 (ideology only)": f"{m_ideo_only.rsquared:.4f}",
        "R2 (baseline)": f"{m_base.rsquared:.4f}",
        "R2 (+region)": f"{m_full.rsquared:.4f}",
        "Delta R2 (ideology)": f"{delta_r2_ideo:.4f}{_stars(f_p_ideo)}",
        "Delta R2 (region)": f"{delta_r2:.4f}",
        "Partial R2": f"{partial_r2:.4f}",
        "F": f"{f_stat:.2f}",
        "df": f"({df1}, {df2})",
        "p(F)": f"{f_p:.2e}{_stars(f_p)}",
        "Wald chi2 (clustered)": f"{wald_chi2:.2f}",
        "p(Wald)": f"{wald_p:.2e}{_stars(wald_p)}",
    }


def run_region_increment_test(
    df: pd.DataFrame,
    df_wiki_minus_groki: pd.DataFrame,
    ideology_vars: List[str],
    cluster_col: Optional[str],
    outcome_col: str,
    table_dir: Path,
) -> None:
    """Incremental F-test / partial R² for the joint contribution of region dummies.

    Tests, for each equation, whether adding the regional dummies to an
    ideology-only OLS significantly improves fit — i.e. "do regions matter as a
    group?" rather than "does each specific region matter?".
    """
    # use_model follows each equation's own specification (see _eq1_specs /
    # _eq2_specs): no judge dummies in the differential, judge dummies as a
    # main-effect control in every per-source spec.
    rows = [
        _region_increment_row(
            df_wiki_minus_groki, "Eq. 1 (wiki - groki)",
            ideology_vars, outcome_col, cluster_col, use_model=False,
        ),
        # Same equation with judge dummies in the floor. Eq. 1 itself carries none,
        # since the outcome is already differenced within a judge, but the per-source
        # rows below do, so this variant is the one that puts all three equations on a
        # common floor. It is the row the paper's variance-decomposition table reports.
        _region_increment_row(
            df_wiki_minus_groki, "Eq. 1 (wiki - groki, judge floor)",
            ideology_vars, outcome_col, cluster_col, use_model=True,
        ),
    ]
    for source, pretty_source in (("wikipedia", "Wikipedia"), ("grokipedia_0.2", "Grokipedia")):
        if source in set(df["Source"].unique()):
            rows.append(
                _region_increment_row(
                    df[df["Source"] == source], f"Eq. 2 ({pretty_source})",
                    ideology_vars, outcome_col, cluster_col, use_model=True,
                )
            )

    table = pd.DataFrame(rows)
    title = "Incremental F-test / Partial R-squared - joint contribution of regional dummies"
    notes = [
        "Nested ladder: floor -> baseline (+ 9 ideology dims) -> + regional dummies (V-Dem).",
        "Floor = judge (Model) dummies for Eq. 2, intercept only for Eq. 1. Judge dummies are",
        "  carried in every Eq. 2 spec, matching _eq2_specs, so R2 (baseline) equals the",
        "  '(1) Baseline' R2 of Appendix Table C. No article-length control in any model.",
        "The 'judge floor' Eq. 1 row repeats that equation with judge dummies in the floor,",
        "  putting all three equations on a common floor; it is the row the paper reports.",
        "  Region's increment is nearly unchanged either way, only the R2 levels shift.",
        "R2 (ideology only) = the 9 ideology dims without judge dummies. The design is balanced",
        "  (every article rated by every judge), so the judge block is orthogonal to ideology and",
        "  this equals Delta R2 (ideology); it is the figure quoted in the main text.",
        "Delta R2 (ideology) = R2(baseline) - R2(floor); Delta R2 (region) = R2(+region) - R2(baseline).",
        "  Both are percentage points of total variance, measured against the same floor,",
        "  so the ideology and region blocks are directly comparable.",
        "Partial R2 = (SSR_base - SSR_full) / SSR_base (share of residual variance explained by region).",
        "F: classic incremental F-test (assumes i.i.d. errors).",
        "Wald chi2: joint test that all region dummies = 0, cluster-robust by party.",
        "Significance: * p<0.05, ** p<0.01, *** p<0.001.",
    ]
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    txt = "\n".join([title, "=" * len(title), "", table.to_string(index=False), ""] + notes) + "\n"
    (table_dir / "region_incremental_ftest.txt").write_text(txt, encoding="utf-8")
    table.to_csv(table_dir / "region_incremental_ftest.csv", index=False)
    print(f"Saved {table_dir / 'region_incremental_ftest.txt'}")
    print(f"Saved {table_dir / 'region_incremental_ftest.csv'}")


def run_robustness_checks(
    df: pd.DataFrame,
    df_wiki_minus_groki: pd.DataFrame,
    ideology_vars: List[str],
    cluster_col: Optional[str],
    table_dir: Path,
    figure_dir: Path,
) -> None:
    """Estimate and save Appendix Tables B & C and the coefficient-stability figure."""
    table_dir = Path(table_dir)
    figure_dir = Path(figure_dir)

    # --- Appendix Table B: Eq. 1 (differential) ---
    eq1_specs = _eq1_specs(df_wiki_minus_groki, ideology_vars, cluster_col)
    _write_table(
        _comparison_table(eq1_specs),
        title="Appendix Table B - Robustness Regression for Eq. 1 (wiki - groki)",
        notes=_EQ1_TABLE_NOTES,
        txt_path=table_dir / "appendix_table_B_eq1_robustness.txt",
        csv_path=table_dir / "appendix_table_B_eq1_robustness.csv",
    )

    # --- Appendix Table C: Eq. 2 (per-source) ---
    source_panels = {
        "wikipedia": "Wikipedia",
        "grokipedia_0.2": "Grokipedia",
    }
    c_sections = []
    for source, pretty_source in source_panels.items():
        if source not in set(df["Source"].unique()):
            continue
        specs = _eq2_specs(df, source, ideology_vars, cluster_col)
        table = _comparison_table(specs)
        c_sections.append(f"--- {pretty_source} ---")
        c_sections.append(table.to_string(index=False))
        c_sections.append("")
        table.to_csv(table_dir / f"appendix_table_C_eq2_robustness_{pretty_source.lower()}.csv", index=False)

    title_c = "Appendix Table C - Robustness Regression for Eq. 2 (per source)"
    c_lines = [title_c, "=" * len(title_c), ""] + c_sections + _TABLE_NOTES
    (table_dir / "appendix_table_C_eq2_robustness.txt").write_text("\n".join(c_lines) + "\n", encoding="utf-8")
    print(f"Saved {table_dir / 'appendix_table_C_eq2_robustness.txt'}")

    # --- Incremental F-test / partial R²: do regions matter as a group? ---
    run_region_increment_test(
        df=df, df_wiki_minus_groki=df_wiki_minus_groki, ideology_vars=ideology_vars,
        cluster_col=cluster_col, outcome_col="neutrality_assessment", table_dir=table_dir,
    )

    # --- Appendix Table D: Eq. 1 re-estimated per judge ---
    run_per_judge_eq1(
        df_wiki_minus_groki=df_wiki_minus_groki, ideology_vars=ideology_vars,
        cluster_col=cluster_col, table_dir=table_dir,
    )

    # --- Appendix Figure: coefficient stability for Eq. 1 ---
    plot_coefficient_stability(
        eq1_specs, KEY_VARS,
        save_path=figure_dir / "robustness" / "coefficient_stability_eq1.png",
    )


# Row order mirrors Appendix Table B / tab:rob_delta in the supplement so the
# per-judge panel can be read straight against the pooled ladder.
_EQ1_ROW_ORDER = [
    "v2paanticlient_osp", "v2paculsup_osp", "v2paimmig_osp", "v2palgbt_osp",
    "v2pawomlab_osp", "v2xpa_plural", "v2xpa_antipopul", "v2parelig_osp",
    "v2pariglef_osp",
]

_PER_JUDGE_NOTES = [
    "Cells: OLS coefficient (SE in parentheses). Significance: * p<0.05, ** p<0.01, *** p<0.001.",
    "Pooled column: all four judges (reproduces Appendix Table B, spec 1).",
    "All columns use cluster-robust SE by party: co-partisans are distinct observations",
    "  even within a single judge, so the clustering still binds per judge.",
    "Ideology scores are z-scored on the full sample, so all columns are on a common scale.",
    "Negative = Grokipedia portrays that ideology more favourably than Wikipedia.",
    "Homogeneity: joint Wald test that every judge x ideology interaction is zero, clustered by party.",
]


def _per_judge_eq1_fits(df_wmg: pd.DataFrame, ideology_vars: List[str], cluster_col: Optional[str]):
    """Fit Eq. 1 pooled and once per judge. Returns (ordered fits, judge labels)."""
    terms = [v for v in _EQ1_ROW_ORDER if v in ideology_vars]
    formula = "neutrality_assessment ~ " + " + ".join(terms)

    def _fit(data: pd.DataFrame):
        """OLS with party-clustered SEs where a cluster key is available.

        Unlike leader-level clustering, party clustering still binds within a
        single judge's column: co-partisans are separate observations there, so
        every column below gets the same cover as the pooled fit.
        """
        if cluster_col is not None and cluster_col in data.columns:
            return smf.ols(formula, data=data).fit(
                cov_type="cluster", cov_kwds={"groups": data[cluster_col]}
            )
        return smf.ols(formula, data=data).fit(cov_type="HC1")

    fits = [("Pooled", _fit(df_wmg))]

    # MODEL_DISPLAY order, so the columns read the same way as the figures.
    present = set(df_wmg["Model"].astype(str).unique())
    judges = [j for j in MODEL_DISPLAY if j in present] + sorted(present - set(MODEL_DISPLAY))
    for j in judges:
        fits.append((pretty_model(j), _fit(df_wmg[df_wmg["Model"].astype(str) == j])))
    return fits, terms


def _judge_homogeneity(df_wmg: pd.DataFrame, terms: List[str], cluster_col: Optional[str]) -> pd.DataFrame:
    """Joint and per-dimension Wald tests that the ideology vector is judge-invariant."""
    formula = "neutrality_assessment ~ C(Model) * (" + " + ".join(terms) + ")"
    kw = {"cov_type": "cluster", "cov_kwds": {"groups": df_wmg[cluster_col]}} \
        if cluster_col is not None and cluster_col in df_wmg.columns else {"cov_type": "HC1"}
    m = smf.ols(formula, data=df_wmg).fit(**kw)
    names = list(m.params.index)
    inter = [p for p in names if ":" in p]

    def wald(params):
        R = np.zeros((len(params), len(names)))
        for i, p in enumerate(params):
            R[i, names.index(p)] = 1.0
        w = m.wald_test(R, use_f=False, scalar=True)
        return float(w.statistic), float(w.pvalue), len(params)

    rows = []
    chi2, p, k = wald(inter)
    rows.append({"Test": "All judge x ideology interactions",
                 "chi2": f"{chi2:.1f}", "df": f"{k}", "p": f"{p:.2e}{_stars(p)}"})
    for v in terms:
        ii = [p_ for p_ in inter if p_.endswith(":" + v) or p_.startswith(v + ":")]
        if not ii:
            continue
        chi2, p, k = wald(ii)
        rows.append({"Test": pretty_dep_label(v), "chi2": f"{chi2:.1f}", "df": f"{k}",
                     "p": f"{p:.2e}{_stars(p)}"})
    return pd.DataFrame(rows)


def run_per_judge_eq1(
    df_wiki_minus_groki: pd.DataFrame,
    ideology_vars: List[str],
    cluster_col: Optional[str],
    table_dir: Path,
) -> None:
    """Appendix Table D: Eq. 1 re-estimated separately for each LLM judge.

    Answers "does the differential result survive rater by rater?" rather than
    only in the pooled fit. Each judge rates every politician once on each
    source, so a per-judge differential has one observation per politician —
    but several politicians per party, so the party-level clustering used
    throughout still applies within each judge's column.
    """
    table_dir = Path(table_dir)
    fits, terms = _per_judge_eq1_fits(df_wiki_minus_groki, ideology_vars, cluster_col)

    rows = []
    for v in terms + ["Intercept"]:
        row = {"Variable": "Intercept" if v == "Intercept" else pretty_dep_label(v)}
        for label, res in fits:
            row[label] = _cell(res.params[v], res.bse[v], res.pvalues[v])
        rows.append(row)
    rows.append({"Variable": "Observations", **{l: f"{int(r.nobs)}" for l, r in fits}})
    rows.append({"Variable": "R-squared", **{l: f"{r.rsquared:.3f}" for l, r in fits}})
    # Vector norm: the judge's overall "gain" on the ideology block, which is
    # what separates the columns once direction is accounted for.
    rows.append({"Variable": "||beta_ideology||",
                 **{l: f"{np.sqrt(sum(r.params[v] ** 2 for v in terms)):.3f}" for l, r in fits}})
    table = pd.DataFrame(rows)

    homog = _judge_homogeneity(df_wiki_minus_groki, terms, cluster_col)

    title = "Appendix Table D - Eq. 1 (wiki - groki) estimated separately per judge"
    lines = [title, "=" * len(title), "", table.to_string(index=False), "",
             "Judge homogeneity tests", "-" * 23, homog.to_string(index=False), ""]
    lines.extend(_PER_JUDGE_NOTES)
    (table_dir / "appendix_table_D_eq1_per_judge.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    table.to_csv(table_dir / "appendix_table_D_eq1_per_judge.csv", index=False)
    homog.to_csv(table_dir / "appendix_table_D_judge_homogeneity.csv", index=False)
    print(f"Saved {table_dir / 'appendix_table_D_eq1_per_judge.txt'}")
    print(f"Saved {table_dir / 'appendix_table_D_eq1_per_judge.csv'}")
    print(f"Saved {table_dir / 'appendix_table_D_judge_homogeneity.csv'}")
