#!/usr/bin/env python3
"""Render the identification-check CSVs as LaTeX tables for the supplement.

Reads ``tables/identification/*.csv`` (written by
:mod:`identification`) plus the weighted-validation CSV from the survey
analysis, and writes one ``.tex`` fragment per table into the paper's
``tables/`` directory, so no number is transcribed by hand.

    python -m src.analysis.statistical_analysis.emit_identification_tables
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PRETTY = {
    "v2paanticlient_osp": r"\texttt{anti-client}",
    "v2paculsup_osp": r"\texttt{culincl}",
    "v2paimmig_osp": r"\texttt{immig}",
    "v2palgbt_osp": r"\texttt{lgbt}",
    "v2pawomlab_osp": r"\texttt{womlab}",
    "v2xpa_plural": r"\texttt{plural}",
    "v2xpa_antipopul": r"\texttt{anti-popul}",
    "v2parelig_osp": r"\texttt{secular}",
    "v2pariglef_osp": r"\texttt{rightwing}",
    "Intercept": "Intercept",
    "year_gap_z": "Election-year gap",
    "female": "Female",
    "top_office": "Head of state or government",
}
ROW_ORDER = [
    "v2paanticlient_osp", "v2paculsup_osp", "v2paimmig_osp", "v2palgbt_osp",
    "v2pawomlab_osp", "v2xpa_plural", "v2xpa_antipopul", "v2parelig_osp",
    "v2pariglef_osp",
]


def _phantom(cell: str) -> str:
    """Right-align the numeric part of a coefficient cell on the minus sign."""
    return cell if cell.startswith("-") else r"\phantom{-}" + cell


def _pivot(df: pd.DataFrame, specs: List[str], rows: List[str]) -> pd.DataFrame:
    piv = df.pivot(index="variable", columns="spec", values="cell")
    piv = piv.reindex([r for r in rows if r in piv.index])
    return piv[[s for s in specs if s in piv.columns]]


def _table(
    piv: pd.DataFrame,
    caption: str,
    label: str,
    footer: Dict[str, List[str]],
    col_headers: Optional[List[str]] = None,
    group_header: Optional[str] = None,
) -> str:
    headers = col_headers or list(piv.columns)
    ncol = len(headers)
    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        r"    \small",
        r"    \setlength{\tabcolsep}{4pt}",
        f"    \\begin{{tabular}}{{l{'c' * ncol}}}",
        r"        \toprule",
    ]
    if group_header:
        lines.append(f"        {group_header}")
    lines.append("        & " + " & ".join(headers) + r" \\")
    lines.append(r"        \midrule")
    for var, row in piv.iterrows():
        cells = [_phantom(str(v)) if pd.notna(v) else "" for v in row]
        lines.append(f"        {PRETTY.get(var, var)} & " + " & ".join(cells) + r" \\")
    if footer:
        lines.append(r"        \midrule")
        for name, values in footer.items():
            lines.append(f"        {name} & " + " & ".join(values) + r" \\")
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _fit_row(df: pd.DataFrame, specs: List[str], col: str, fmt: str) -> List[str]:
    stats = df.groupby("spec")[col].first()
    return [format(stats[s], fmt) if s in stats.index else "" for s in specs]


# ---------------------------------------------------------------------------

def country_table(src: Path) -> str:
    baseline, country = "(1) No geography", "(3) Country FE"
    frames = {
        "delta": pd.read_csv(src / "country_fe_delta.csv"),
        "wikipedia": pd.read_csv(src / "country_fe_per_source_wikipedia.csv"),
        "grokipedia": pd.read_csv(src / "country_fe_per_source_grokipedia_0.2.csv"),
    }
    rows = ROW_ORDER + ["Intercept"]
    pieces, r2, n = {}, [], []
    for key, frame in frames.items():
        piv = _pivot(frame, [baseline, country], rows)
        for spec in (baseline, country):
            pieces[f"{key}|{spec}"] = piv[spec]
        stats = frame.groupby("spec")[["r2", "n"]].first()
        r2 += [f"{stats.loc[s, 'r2']:.3f}" for s in (baseline, country)]
        n += [f"{int(stats.loc[s, 'n'])}" for s in (baseline, country)]

    piv = pd.DataFrame(pieces).reindex([r for r in rows if r in frames["delta"]["variable"].values])
    meta = pd.read_csv(src / "country_fe_meta.csv").iloc[0]

    caption = (
        "Ideology coefficients with and without country fixed effects. "
        "\\textit{Base} reproduces the specification reported in the main paper; "
        f"\\textit{{+Country}} adds a fixed effect for each of the {int(meta['n_countries'])} "
        "countries in the sample, so the ideology coefficients are identified only "
        "by variation between parties \\emph{within} a country. The differential "
        "columns are Eq.~1, the Wikipedia and Grokipedia columns Eq.~2 (judge "
        "indicators included, not shown). All columns use cluster-robust standard "
        f"errors at the party level ({int(meta['n_parties'])} clusters). "
        "$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$."
    )
    group = (
        r"& \multicolumn{2}{c}{Differential} & \multicolumn{2}{c}{Wikipedia}"
        r" & \multicolumn{2}{c}{Grokipedia} \\"
        "\n"
        r"        \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}"
    )
    return _table(
        piv,
        caption,
        "tab:country_fe",
        {"$R^2$": r2, "Observations": n},
        col_headers=["Base", "+Country"] * 3,
        group_header=group,
    )


def selection_tables(src: Path) -> str:
    logit = pd.read_csv(src / "selection_logit.csv")
    specs = ["(1) Ideology", "(2) + Region", "(3) + Region, sex, office"]
    rows = ROW_ORDER + ["female", "top_office"]
    piv = _pivot(logit, specs, rows)
    stages = pd.read_csv(src / "selection_stages.csv")
    pool_n = int(stages.loc[0, "n"])
    rate = 100 * stages.loc[0, "retention_rate"]

    caption = (
        "Logistic model of surviving the coverage funnel. The outcome is 1 for "
        f"the politicians retained in the final dataset and 0 otherwise, over the "
        f"{pool_n:,} politicians carrying a complete set of V-Party annotations "
        f"({rate:.1f}\\% retained). Ideology scores are $z$-scored on this pool, so "
        "a coefficient is the change in log-odds of retention per pool standard "
        "deviation. Coefficients are log-odds. "
        "$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$."
    )
    out = _table(
        piv,
        caption,
        "tab:selection_logit",
        {
            "Pseudo-$R^2$": _fit_row(logit, specs, "pseudo_r2", ".3f"),
            "Observations": _fit_row(logit, specs, "n", "d"),
        },
    )

    ipw = pd.read_csv(src / "selection_reweighted.csv")
    ipw_specs = ["(1) Unweighted", "(2) IPW", "(3) Top office", "(4) Other office"]
    piv2 = _pivot(ipw, ipw_specs, ROW_ORDER + ["Intercept"])
    caption2 = (
        "Eq.~1 re-estimated against the selection detected in "
        "Table~\\ref{tab:selection_logit}. Column (2) weights each politician by "
        "the inverse of their fitted retention probability from specification (3) "
        "of that table, projecting the retained sample back onto the annotated "
        "pool; weights are trimmed at the 1st and 99th percentiles. Columns (3) "
        "and (4) split the sample on the strongest single predictor of retention, "
        "whether the politician is a head of state or government. Party-clustered "
        "standard errors throughout. "
        "$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$."
    )
    out += "\n" + _table(
        piv2,
        caption2,
        "tab:selection_ipw",
        {"Observations": _fit_row(ipw, ipw_specs, "n", "d")},
    )
    return out


def staleness_table(src: Path) -> str:
    fits = pd.read_csv(src / "staleness_fits.csv")
    specs = list(dict.fromkeys(fits["spec"]))
    piv = _pivot(fits, specs, ROW_ORDER + ["year_gap_z", "Intercept"])
    gap = pd.read_csv(src / "staleness_gap_summary.csv").set_index("index")["year_gap"]

    caption = (
        "Eq.~1 against measurement error in the ideology scores. The "
        "election-year gap is the number of years between a politician's WhoGov "
        f"year and the V-Party election year their score is taken from (mean "
        f"{gap['mean']:.1f}, median {gap['50%']:.0f}, maximum {gap['max']:.0f} over the "
        f"{int(gap['count']):,} politicians whose party matches a V-Party election "
        "row on its Party Facts id). Column (2) adds the $z$-scored gap as a "
        "covariate, column (3) restricts the sample to gaps at or below the "
        "median. Party-clustered standard errors. "
        "$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$."
    )
    return _table(
        piv,
        caption,
        "tab:staleness",
        {
            "$R^2$": _fit_row(fits, specs, "r2", ".3f"),
            "Observations": _fit_row(fits, specs, "n", "d"),
        },
    )


def clustering_table(src: Path) -> str:
    levels = ["Politician", "Party", "Country"]
    frames = {
        "delta": pd.read_csv(src / "clustering_delta.csv"),
        "wikipedia": pd.read_csv(src / "clustering_per_source_wikipedia.csv"),
        "grokipedia": pd.read_csv(src / "clustering_per_source_grokipedia_0.2.csv"),
    }
    nesting = pd.read_csv(src / "clustering_nesting.csv").iloc[0]

    # Coefficients are identical across clustering levels, so only the standard
    # errors and the stars carry information; show the party and country columns
    # for the two per-source panels and all three for the differential.
    shown = levels[1:]
    pieces, counts = {}, []
    for key in ("delta", "wikipedia", "grokipedia"):
        piv = _pivot(frames[key], shown, ROW_ORDER + ["Intercept"])
        for lvl in shown:
            pieces[f"{key}|{lvl}"] = piv[lvl]
        counts += [
            f"{int(frames[key].groupby('spec')['n_clusters'].first()[lvl])}"
            for lvl in shown
        ]

    out = pd.DataFrame(pieces).reindex(
        [r for r in ROW_ORDER + ["Intercept"] if r in frames["delta"]["variable"].values]
    )

    caption = (
        "The same fits under standard errors clustered at three nesting levels. "
        f"Every one of the {int(nesting['parties'])} parties in the sample lies "
        f"within a single one of the {int(nesting['countries'])} countries "
        "(0 parties span more than one), so a two-way party-and-country cluster "
        "collapses onto the coarser level and the country column is the "
        "conservative multiway estimate. Point estimates are identical across "
        "columns by construction, so only the standard errors differ. The "
        "politician level, which the main paper argues against, is discussed in "
        "the text rather than shown. "
        "$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$."
    )
    group = (
        r"& \multicolumn{2}{c}{Differential} & \multicolumn{2}{c}{Wikipedia}"
        r" & \multicolumn{2}{c}{Grokipedia} \\"
        "\n"
        r"        \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}"
    )
    return _table(
        out,
        caption,
        "tab:clustering",
        {"Clusters": counts},
        col_headers=["Party", "Country"] * 3,
        group_header=group,
    )


def multiplicity_table(src: Path) -> str:
    m = pd.read_csv(src / "multiplicity.csv")
    order = {
        "Eq. 1 (differential)": "Differential",
        "Eq. 2 (grokipedia_0.2)": "Grokipedia",
        "Eq. 2 (wikipedia)": "Wikipedia",
    }

    def p(v: float) -> str:
        if v < 0.001:
            return r"$<$.001"
        return f"{v:.3f}".lstrip("0")

    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \caption{Multiplicity-adjusted inference over the nine ideology "
        r"dimensions. Each equation is one family of nine tests; $p_{\mathrm{Holm}}$ "
        r"controls the family-wise error rate and $p_{\mathrm{BH}}$ the false "
        r"discovery rate. Raw $p$-values are the party-clustered ones reported in "
        r"the main paper. Dimensions insignificant before adjustment are omitted; "
        r"no Wikipedia dimension appears, as none reaches $p<0.05$ unadjusted.}",
        r"    \label{tab:multiplicity}",
        r"    \small",
        r"    \begin{tabular}{llcccc}",
        r"        \toprule",
        r"        Equation & Dimension & $\hat\beta$ & $p$ & "
        r"$p_{\mathrm{Holm}}$ & $p_{\mathrm{BH}}$ \\",
        r"        \midrule",
    ]
    for eq, label in order.items():
        sub = m[(m["equation"] == eq) & (m["p_raw"] < 0.05)]
        for i, (_, r) in enumerate(sub.iterrows()):
            first = label if i == 0 else ""
            lines.append(
                f"        {first} & {PRETTY[r['variable']]} & "
                f"{_phantom(f'{r.coef:.3f}')} & {p(r.p_raw)} & "
                f"{p(r.p_holm)} & {p(r.p_bh)} \\\\"
            )
        if len(sub):
            lines.append(r"        \addlinespace")
    if lines[-1].strip() == r"\addlinespace":
        lines.pop()
    lines += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def weighted_validation_table(path: Path) -> str:
    d = pd.read_csv(path)
    names = {
        "claude-opus-4-6": "Claude",
        "grok-4": "Grok",
        "mistralai": "Mistral",
        "deepseek-v4-pro": "DeepSeek",
        "llm_median": "Panel median",
    }
    lines = [
        r"\begin{table}[t]",
        r"    \centering",
        r"    \caption{Correlation with the two-rater human mean, unweighted and "
        r"post-stratified. Weights are the ratio of the corpus share to the sample "
        r"share of each judge-disagreement stratum, so the weighted column projects "
        r"onto the length-eligible paired corpus rather than the difficulty-enriched "
        r"validation sample. The Kish effective sample size under these weights is "
        r"40.3 of 70 articles. The final row is the share of shown ratings an "
        r"annotator judged reasonable or partially reasonable.}",
        r"    \label{tab:weighted_validation}",
        r"    \small",
        r"    \begin{tabular}{lcc}",
        r"        \toprule",
        r"        Rater & $r$ & $r$ (weighted) \\",
        r"        \midrule",
    ]
    for key, label in names.items():
        row = d[d["rater"] == key]
        if row.empty:
            continue
        r = row.iloc[0]
        lines.append(
            f"        {label} & {r.pearson_r:.2f} & {r.pearson_r_w:.2f} \\\\"
        )
    reas = d[d["rater"].str.startswith("shown_llm")].iloc[0]
    lines += [
        r"        \midrule",
        f"        Reasonable or partial (\\%) & {reas.pearson_r:.1f} & "
        f"{reas.pearson_r_w:.1f} \\\\",
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--identification-dir",
        type=Path,
        default=Path("results/final2.0/tables/identification"),
    )
    parser.add_argument(
        "--weighted-validation",
        type=Path,
        default=Path(
            "survey/result_analysis/analyses_results/"
            "results-survey918273_analysis/weighted_validation.csv"
        ),
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("results/tables")
    )
    args = parser.parse_args(argv)

    src = args.identification_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written = {
        "country_fe.tex": country_table(src),
        "selection.tex": selection_tables(src),
        "staleness.tex": staleness_table(src),
        "clustering.tex": clustering_table(src),
        "multiplicity.tex": multiplicity_table(src),
        "weighted_validation.tex": weighted_validation_table(args.weighted_validation),
    }
    for name, text in written.items():
        (args.out_dir / name).write_text(text)
        print(f"Wrote {args.out_dir / name}")


if __name__ == "__main__":
    main()
