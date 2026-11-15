"""Ordinal-logit validation of the two OLS ideology regressions.

The neutrality assessment is an ordered five-point Likert item, and the
differential is a difference of two such items. OLS treats those scores as
equally spaced interval measurements; a proportional-odds (ordinal logit) model
uses only their ordering. Re-estimating both equations that way therefore checks
that the reported ideology effects are not an artifact of the equal-spacing
assumption.

Two specifications are re-estimated, mirroring the paper term for term:

  Eq. 1 (differential) — ΔY (wiki − groki), pooled across judges, ideology
                         dimensions only, cutpoints in place of the intercept.
  Eq. 2 (per-source)   — Y within each source, judge dummies as main effects
                         only (no judge × ideology interactions).

Standard errors are cluster-robust by party in both, as in the OLS fits.

Outputs (relative to the run's results directory):
  figures/ordinal_validation/ordinal_coefficients_grid.{png,pdf}
  tables/ordinal_validation/ordinal_vs_ols_{delta,<source>}.{txt,csv}
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .config import SOURCE_DISPLAY
from .stat_models import AllModelResults, ModelResults
from .visualisation.helpers import pretty_dep_label
from .visualisation.stat_figures import (
    _clean_ols_param_name,
    _extract_ordinal_rows,
    plot_coefficients_grid,
)

# Panel order of the validation figure: differential first, then the two
# per-source models, matching the order of Eq. 1 and Eq. 2 in the paper.
_PER_SOURCE_ORDER = ("grokipedia_0.2", "wikipedia")

_ALPHA = 0.05


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


def _ols_coef_map(res: ModelResults) -> Dict[str, tuple]:
    """Map pretty term label -> (coef, se, p) for an OLS fit."""
    r = res.results
    return {
        ("Intercept" if name == "Intercept" else pretty_dep_label(_clean_ols_param_name(name))):
            (r.params[name], r.bse[name], r.pvalues[name])
        for name in r.params.index
    }


def _ordinal_coef_map(res: ModelResults) -> Dict[str, tuple]:
    """Map pretty term label -> (coef, se, p) for an ordinal fit (slopes only)."""
    r = res.results
    return {
        pretty_dep_label(name): (r.params.iloc[i], r.bse.iloc[i], r.pvalues.iloc[i])
        for i, name in enumerate(res.predictors)
    }


def _agreement(ols: Optional[tuple], ordinal: Optional[tuple]) -> str:
    """One-word verdict on whether the two fits tell the same story for a term.

    Sign is only compared where both fits reject the null: the sign of an
    estimate indistinguishable from zero carries no claim to contradict.
    """
    if ols is None or ordinal is None:
        return ""
    ols_sig, ord_sig = ols[2] < _ALPHA, ordinal[2] < _ALPHA
    if ols_sig != ord_sig:
        return "differs"
    if not ols_sig:
        return "same (n.s.)"
    return "same" if (ols[0] > 0) == (ordinal[0] > 0) else "sign flip"


def _comparison_table(results: AllModelResults, difference_outcome: bool) -> pd.DataFrame:
    """OLS vs ordinal estimates for one specification, one row per term."""
    _, _, ordered_labels, _ = _extract_ordinal_rows(
        results.ordinal, difference_outcome=difference_outcome, show_baseline=False
    )
    ols_map = _ols_coef_map(results.ols)
    ord_map = _ordinal_coef_map(results.ordinal)

    rows = []
    for label in ordered_labels:
        ols, ordinal = ols_map.get(label), ord_map.get(label)
        rows.append({
            "Variable": label.replace("Model_", ""),
            "OLS beta (SE)": _cell(*ols) if ols else "",
            "OLS p": f"{ols[2]:.4f}" if ols else "",
            "Ordinal beta (SE)": _cell(*ordinal) if ordinal else "",
            "Ordinal p": f"{ordinal[2]:.4f}" if ordinal else "",
            "Agreement": _agreement(ols, ordinal),
        })

    rows.append({
        "Variable": "Observations",
        "OLS beta (SE)": f"{int(results.ols.results.nobs)}",
        "Ordinal beta (SE)": f"{int(results.ordinal.results.nobs)}",
    })
    rows.append({
        "Variable": "Fit",
        "OLS beta (SE)": f"R2 = {results.ols.results.rsquared:.3f}",
        "Ordinal beta (SE)": f"pseudo R2 = {results.ordinal.results.prsquared:.3f}",
    })
    return pd.DataFrame(rows).fillna("")


_TABLE_NOTES = [
    "Cells: coefficient (cluster-robust SE in parentheses), clustered by party.",
    "Significance: * p<0.05, ** p<0.01, *** p<0.001.",
    "OLS coefficients are on the Likert scale; ordinal coefficients are log-odds,",
    "so magnitudes are not comparable across the two columns — signs and",
    "significance are.",
    "Agreement: 'same' = same sign and both significant at the 5% level;",
    "'same (n.s.)' = neither significant; 'differs' = significant in one fit only.",
]


def _write_table(table: pd.DataFrame, title: str, txt_path: Path, csv_path: Path) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [title, "=" * len(title), "", table.to_string(index=False), ""]
    lines.extend(_TABLE_NOTES)
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    table.to_csv(csv_path, index=False)
    print(f"Saved {txt_path}")
    print(f"Saved {csv_path}")


def run_ordinal_validation(
    delta_results: AllModelResults,
    per_source_results: Dict[str, AllModelResults],
    figure_dir: Path,
    table_dir: Path,
) -> None:
    """Render the ordinal-validation figure and OLS-vs-ordinal comparison tables.

    Takes the fits already produced for the paper — the pooled differential
    model (Eq. 1) and the per-source models (Eq. 2) — so the validation cannot
    drift from the specifications it validates.
    """
    print("\n" + "=" * 80)
    print("ORDINAL VALIDATION (Eq. 1 and Eq. 2 re-estimated as ordinal logits)")
    print("=" * 80)

    specs: List[Tuple[str, AllModelResults, bool]] = [
        ("Wikipedia $-$ Grokipedia ($\\Delta Y$)", delta_results, True)
    ]
    specs += [
        (SOURCE_DISPLAY.get(source, source), per_source_results[source], False)
        for source in _PER_SOURCE_ORDER
        if source in per_source_results
    ]

    panels: Sequence[Tuple[str, ModelResults]] = [
        (title, results.ordinal) for title, results, _ in specs if results.ordinal is not None
    ]
    if not panels:
        print("No ordinal fits available — skipping ordinal validation")
        return

    # Cutpoint rows carry no slope interpretation, so the baseline E[Y] row is
    # dropped (show_intercept=False) and every panel shows only estimated terms.
    # The fit annotation is dropped too: these fits report McFadden's pseudo R^2,
    # which sits at a different level than the OLS R^2 of the paper's figures and
    # invites reading the gap as a loss of fit.
    plot_coefficients_grid(
        panels=panels,
        save_path=figure_dir / "ordinal_validation" / "ordinal_coefficients_grid.png",
        show_intercept=False,
        shared_x=False,
        show_fit_stats=False,
    )

    for title, results, difference_outcome in specs:
        if results.ordinal is None or results.ols is None:
            continue
        slug = "delta" if difference_outcome else title.lower().replace(" ", "_")
        table = _comparison_table(results, difference_outcome)
        _write_table(
            table,
            title=f"ORDINAL VS OLS — {title}",
            txt_path=table_dir / "ordinal_validation" / f"ordinal_vs_ols_{slug}.txt",
            csv_path=table_dir / "ordinal_validation" / f"ordinal_vs_ols_{slug}.csv",
        )
