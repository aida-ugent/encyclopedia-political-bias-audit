"""Paired significance tests for the two descriptive tables of the paper.

Both encyclopedias are rated for the *same* politicians by the *same* judges, so
the Wikipedia-against-Grokipedia contrasts in Table 1 are paired, not two
independent samples. The tests here respect that pairing:

Every cell of Table 1 is tested the same way, by a two-sided politician-level
bootstrap of the paired within-politician difference. Per-judge rows carry one
difference per politician; the ``Average`` row carries four, which the resampling
keeps together because the four judges rate the same politicians and their
contributions are not independent. One procedure for the whole table keeps the
caption to a single clause and matches how the intervals in Table 2 are built.

Wilcoxon signed-rank (magnitude) and McNemar (neutral rate) are retained as a
cross-check on the per-judge rows. They agree with the bootstrap on every star
band, so the choice of test does not carry any verdict in the paper.

For Table 2 the same logic gives the Wikipedia-against-Grokipedia contrast in
Krippendorff's alpha a politician-level bootstrap interval. Agreement
coefficients are not usefully tested against zero (nobody doubts the judges beat
chance), so intervals rather than p-values are reported for those.

    python -m src.analysis.statistical_analysis.descriptive_tests
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

from .config import (
    CONTINUOUS_INDEPENDENT_VARS, DEFAULT_LEADERS_PATH, MODEL_DISPLAY,
)
from .data_loader import load_and_prepare_data
from .visualisation.descriptive_stats import _krippendorff_alpha_ordinal

logger = logging.getLogger(__name__)

WIKI, GROKI = "wikipedia", "grokipedia_0.2"
BOOT_B = 5000
BOOT_SEED = 20260821


def _paired_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per politician and judge, with both sources side by side."""
    w = df[df["Source"] == WIKI][["name", "Model", "neutrality_assessment"]]
    g = df[df["Source"] == GROKI][["name", "Model", "neutrality_assessment"]]
    m = w.merge(g, on=["name", "Model"], suffixes=("_wiki", "_groki"))
    m["abs_wiki"] = m["neutrality_assessment_wiki"].abs()
    m["abs_groki"] = m["neutrality_assessment_groki"].abs()
    m["neutral_wiki"] = (m["neutrality_assessment_wiki"] == 0).astype(int)
    m["neutral_groki"] = (m["neutrality_assessment_groki"] == 0).astype(int)
    return m


def _cluster_boot_p(values: np.ndarray, names: np.ndarray) -> float:
    """Two-sided bootstrap p-value for a zero mean, resampling politicians.

    Used for the pooled ``Average`` row, where each politician contributes four
    correlated differences.
    """
    rng = np.random.default_rng(BOOT_SEED)
    uniq, inv = np.unique(names, return_inverse=True)
    groups = [values[inv == i] for i in range(len(uniq))]
    obs = values.mean()
    idx = np.arange(len(groups))
    centred = []
    for _ in range(BOOT_B):
        pick = rng.choice(idx, idx.size)
        draw = np.concatenate([groups[i] for i in pick])
        centred.append(draw.mean() - obs)
    centred = np.asarray(centred)
    return float(min(1.0, 2 * min((centred >= -obs).mean(), (centred <= -obs).mean())))


def table1_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap tests for the two Delta columns, with rank tests alongside.

    ``p_abs`` and ``p_neutral`` are the bootstrap p-values the paper reports.
    ``p_abs_wilcoxon`` and ``p_neutral_mcnemar`` are the cross-check.
    """
    paired = _paired_frame(df)
    rows = []
    for model, sub in paired.groupby("Model", observed=True):
        # Magnitude: Wilcoxon on the within-politician difference.
        d = (sub["abs_groki"] - sub["abs_wiki"]).to_numpy(float)
        nz = d[d != 0]
        p_mag = (
            float(stats.wilcoxon(nz, alternative="two-sided").pvalue)
            if nz.size
            else np.nan
        )
        # Neutral rate: McNemar on the discordant pairs.
        b = int(((sub["neutral_wiki"] == 1) & (sub["neutral_groki"] == 0)).sum())
        c = int(((sub["neutral_wiki"] == 0) & (sub["neutral_groki"] == 1)).sum())
        p_neu = float(mcnemar([[0, b], [c, 0]], exact=(b + c) < 25).pvalue)
        rows.append(
            {
                "judge": MODEL_DISPLAY.get(model, model),
                "n_pairs": len(sub),
                "delta_abs": float(sub["abs_groki"].mean() - sub["abs_wiki"].mean()),
                "p_abs": _cluster_boot_p(
                    (sub["abs_groki"] - sub["abs_wiki"]).to_numpy(float),
                    sub["name"].to_numpy(),
                ),
                "p_abs_wilcoxon": p_mag,
                "delta_neutral_pp": 100
                * float(sub["neutral_groki"].mean() - sub["neutral_wiki"].mean()),
                "p_neutral": _cluster_boot_p(
                    (sub["neutral_groki"] - sub["neutral_wiki"]).to_numpy(float),
                    sub["name"].to_numpy(),
                ),
                "p_neutral_mcnemar": p_neu,
                "discordant_b": b,
                "discordant_c": c,
            }
        )

    # Average row: pooled over judges, politician-level cluster bootstrap.
    names = paired["name"].to_numpy()
    rows.append(
        {
            "judge": "Average",
            "n_pairs": len(paired),
            "delta_abs": float((paired["abs_groki"] - paired["abs_wiki"]).mean()),
            "p_abs": _cluster_boot_p(
                (paired["abs_groki"] - paired["abs_wiki"]).to_numpy(float), names
            ),
            "p_abs_wilcoxon": np.nan,
            "delta_neutral_pp": 100
            * float((paired["neutral_groki"] - paired["neutral_wiki"]).mean()),
            "p_neutral": _cluster_boot_p(
                (paired["neutral_groki"] - paired["neutral_wiki"]).to_numpy(float),
                names,
            ),
            "p_neutral_mcnemar": np.nan,
            "discordant_b": np.nan,
            "discordant_c": np.nan,
        }
    )
    return pd.DataFrame(rows)


def _alpha_by_source(df: pd.DataFrame, source: str, names: List[str]) -> float:
    piv = (
        df[(df["Source"] == source) & (df["name"].isin(names))]
        .pivot_table(index="name", columns="Model", values="neutrality_assessment",
                     aggfunc="first")
    )
    return _krippendorff_alpha_ordinal(piv.to_numpy())


def table2_alpha_interval(df: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap interval for alpha per source and for the source difference."""
    names = sorted(df["name"].unique())
    rng = np.random.default_rng(BOOT_SEED)

    piv = {
        s: df[df["Source"] == s]
        .pivot_table(index="name", columns="Model", values="neutrality_assessment",
                     aggfunc="first")
        .reindex(names)
        for s in (WIKI, GROKI)
    }
    point = {s: _krippendorff_alpha_ordinal(piv[s].to_numpy()) for s in piv}
    obs_diff = point[WIKI] - point[GROKI]

    reps = {WIKI: [], GROKI: [], "diff": []}
    idx = np.arange(len(names))
    for _ in range(500):  # alpha is O(n^2) in categories, so fewer reps
        pick = rng.choice(idx, idx.size)
        a = {s: _krippendorff_alpha_ordinal(piv[s].to_numpy()[pick]) for s in piv}
        for s in piv:
            reps[s].append(a[s])
        reps["diff"].append(a[WIKI] - a[GROKI])

    def ci(v):
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    # Pooled alpha treats each article as an item; a politician contributes one
    # per source, so the bootstrap resamples politicians and keeps both.
    pooled_point = _krippendorff_alpha_ordinal(
        np.vstack([piv[WIKI].to_numpy(), piv[GROKI].to_numpy()])
    )
    pooled_reps = []
    for _ in range(500):
        pick = rng.choice(idx, idx.size)
        pooled_reps.append(
            _krippendorff_alpha_ordinal(
                np.vstack([piv[WIKI].to_numpy()[pick], piv[GROKI].to_numpy()[pick]])
            )
        )

    out = []
    for key, val, reps_v in [
        (WIKI, point[WIKI], reps[WIKI]),
        (GROKI, point[GROKI], reps[GROKI]),
        ("pooled", pooled_point, pooled_reps),
        ("diff", obs_diff, reps["diff"]),
    ]:
        lo, hi = ci(reps_v)
        out.append({"quantity": key, "alpha": val, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(out)


PAIRS = [
    ("Claude", "DeepSeek"), ("Claude", "Grok"), ("Claude", "Mistral"),
    ("DeepSeek", "Grok"), ("DeepSeek", "Mistral"), ("Grok", "Mistral"),
]


def table2_all_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap intervals for every cell of Table 2, to size the layout.

    Reports the interval half-width per statistic so it is clear which columns
    would actually gain from carrying one.
    """
    from sklearn.metrics import cohen_kappa_score

    inv = {v: k for k, v in MODEL_DISPLAY.items()}
    labels = [-2, -1, 0, 1, 2]
    rng = np.random.default_rng(BOOT_SEED)
    rows = []

    for source_label, source in [("Wikipedia", WIKI), ("Grokipedia", GROKI),
                                 ("Pooled", None)]:
        sub = df if source is None else df[df["Source"] == source]
        idx_cols = ["name", "Source"] if source is None else ["name"]
        piv = sub.pivot_table(index=idx_cols, columns="Model",
                              values="neutrality_assessment", aggfunc="first").dropna()
        n = len(piv)
        draws = [rng.integers(0, n, n) for _ in range(500)]

        for a, b in PAIRS:
            y1 = piv[inv[a]].to_numpy(int); y2 = piv[inv[b]].to_numpy(int)
            d = np.abs(y1 - y2)
            stat = {
                "exact": 100 * (d == 0).mean(),
                "within1": 100 * (d <= 1).mean(),
                "kappa": cohen_kappa_score(y1, y2, weights="quadratic", labels=labels),
                "r": float(np.corrcoef(y1, y2)[0, 1]),
            }
            reps = {k: [] for k in stat}
            for pick in draws:
                dd = np.abs(y1[pick] - y2[pick])
                reps["exact"].append(100 * (dd == 0).mean())
                reps["within1"].append(100 * (dd <= 1).mean())
                reps["kappa"].append(
                    cohen_kappa_score(y1[pick], y2[pick], weights="quadratic",
                                      labels=labels))
                reps["r"].append(float(np.corrcoef(y1[pick], y2[pick])[0, 1]))
            for k, v in stat.items():
                lo, hi = np.percentile(reps[k], [2.5, 97.5])
                rows.append({"source": source_label, "raters": f"{a}--{b}",
                             "stat": k, "value": v, "ci_low": lo, "ci_high": hi,
                             "half_width": (hi - lo) / 2})
    return pd.DataFrame(rows)


def table2_r_and_alpha(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson r and ordinal Krippendorff alpha, with intervals, for every row.

    Alpha is defined for any number of raters, so the pairwise rows get one on
    the same footing as the all-four rows, which makes the column comparable
    down the whole table in a way quadratically weighted kappa is not.
    """
    inv = {v: k for k, v in MODEL_DISPLAY.items()}
    rng = np.random.default_rng(BOOT_SEED)
    rows = []

    scopes = [("Wikipedia", WIKI), ("Grokipedia", GROKI), ("Pooled", None)]
    for label, source in scopes:
        sub = df if source is None else df[df["Source"] == source]
        idx = ["name", "Source"] if source is None else ["name"]
        piv = sub.pivot_table(index=idx, columns="Model",
                              values="neutrality_assessment",
                              aggfunc="first").dropna()
        n = len(piv)
        draws = [rng.integers(0, n, n) for _ in range(500)]
        cols_all = [inv[m] for m in ("Claude", "Grok", "Mistral", "DeepSeek")]

        combos = [(f"{a}--{b}", [inv[a], inv[b]]) for a, b in PAIRS]
        combos.append(("All 4 raters", cols_all))

        for raters, cols in combos:
            m = piv[cols].to_numpy(int)
            a_pt = _krippendorff_alpha_ordinal(m)
            r_pt = (
                float(np.corrcoef(m[:, 0], m[:, 1])[0, 1]) if m.shape[1] == 2 else np.nan
            )
            a_reps, r_reps = [], []
            for pick in draws:
                mm = m[pick]
                a_reps.append(_krippendorff_alpha_ordinal(mm))
                if mm.shape[1] == 2:
                    r_reps.append(float(np.corrcoef(mm[:, 0], mm[:, 1])[0, 1]))
            a_lo, a_hi = np.percentile(a_reps, [2.5, 97.5])
            if r_reps:
                r_lo, r_hi = np.percentile(r_reps, [2.5, 97.5])
            else:
                r_lo = r_hi = np.nan
            d = np.abs(m[:, 0] - m[:, 1]) if m.shape[1] == 2 else (
                m.max(axis=1) - m.min(axis=1))
            rows.append({
                "source": label, "raters": raters, "n": n,
                "exact_pct": 100 * float((d == 0).mean()),
                "within1_pct": 100 * float((d <= 1).mean()),
                "r": r_pt, "r_lo": r_lo, "r_hi": r_hi,
                "alpha": a_pt, "alpha_lo": a_lo, "alpha_hi": a_hi,
            })
    return pd.DataFrame(rows)


def _icc(m: np.ndarray) -> Dict[str, float]:
    """Two-way ICCs for an items x raters matrix.

    ``consistency`` is ICC(3,1), the multi-rater analogue of Pearson's r: it
    treats a rater's constant offset as a nuisance and measures whether raters
    order the items alike. ``agreement`` is ICC(2,1), which charges that offset
    as disagreement, so it plays the role Krippendorff's alpha plays pairwise.
    For two raters the two reduce to r and to an agreement coefficient.
    """
    x = m.astype(float)
    n, k = x.shape
    grand = x.mean()
    ss_rows = k * ((x.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((x.mean(axis=0) - grand) ** 2).sum()
    ss_tot = ((x - grand) ** 2).sum()
    ss_err = ss_tot - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    consistency = (ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err)
    agreement = (ms_rows - ms_err) / (
        ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    )
    return {"consistency": float(consistency), "agreement": float(agreement)}


def table2_multirater_r(df: pd.DataFrame) -> pd.DataFrame:
    """Candidate all-four-rater analogues of Pearson's r, with intervals."""
    inv = {v: k for k, v in MODEL_DISPLAY.items()}
    cols_all = [inv[m] for m in ("Claude", "Grok", "Mistral", "DeepSeek")]
    rng = np.random.default_rng(BOOT_SEED)
    rows = []

    for label, source in [("Wikipedia", WIKI), ("Grokipedia", GROKI),
                          ("Pooled", None)]:
        sub = df if source is None else df[df["Source"] == source]
        idx = ["name", "Source"] if source is None else ["name"]
        m = (sub.pivot_table(index=idx, columns="Model",
                             values="neutrality_assessment", aggfunc="first")
             .dropna()[cols_all].to_numpy(int))
        n = len(m)

        def mean_pairwise_r(mat):
            c = np.corrcoef(mat.T)
            iu = np.triu_indices(mat.shape[1], 1)
            return float(c[iu].mean())

        point = {"mean_r": mean_pairwise_r(m), **_icc(m)}
        reps = {k: [] for k in point}
        for _ in range(500):
            mm = m[rng.integers(0, n, n)]
            reps["mean_r"].append(mean_pairwise_r(mm))
            ic = _icc(mm)
            reps["consistency"].append(ic["consistency"])
            reps["agreement"].append(ic["agreement"])
        for k, v in point.items():
            lo, hi = np.percentile(reps[k], [2.5, 97.5])
            rows.append({"source": label, "stat": k, "value": v,
                         "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=Path, default=Path("results/final2.0"))
    parser.add_argument("--experiment", default="ideology")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = load_and_prepare_data(
        results_path=args.results_path,
        leaders_path=DEFAULT_LEADERS_PATH,
        experiment=args.experiment,
        continuous_independent_vars=CONTINUOUS_INDEPENDENT_VARS,
    )

    out_dir = args.results_path / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    t1 = table1_tests(df)
    t1.to_csv(out_dir / "table1_paired_tests.csv", index=False)
    print("\nTable 1 paired tests\n" + t1.to_string(index=False))

    t2 = table2_alpha_interval(df)
    t2.to_csv(out_dir / "table2_alpha_intervals.csv", index=False)
    print("\nTable 2 alpha intervals\n" + t2.to_string(index=False))

    t4 = table2_r_and_alpha(df)
    t4.to_csv(out_dir / "table2_r_alpha.csv", index=False)
    print("\nTable 2 with r and pairwise alpha")
    print(t4.to_string(index=False))

    t5 = table2_multirater_r(df)
    t5.to_csv(out_dir / "table2_multirater_r.csv", index=False)
    print("\nAll-four-rater analogues of r")
    print(t5.to_string(index=False))

    t3 = table2_all_intervals(df)
    t3.to_csv(out_dir / "table2_all_intervals.csv", index=False)
    print("\nTypical interval half-width by statistic")
    print(t3.groupby("stat")["half_width"].describe()[["mean", "min", "max"]]
          .to_string())


if __name__ == "__main__":
    main()
