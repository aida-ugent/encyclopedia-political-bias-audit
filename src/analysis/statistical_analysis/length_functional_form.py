#!/usr/bin/env python3
"""Functional form of the article-length control in Eq. 1, and whether length
moderates the ideology effects.

Eq. 1 regresses the within-pair neutrality differential
``dY = Y_wikipedia - Y_grokipedia`` on the nine z-scored ideology dimensions.
The paper's length control there is the z-scored raw word-count difference
``W_W - W_G``, entered linearly. Two things are wrong with that. A linear
control in raw words is the wrong scale for a disparity that is multiplicative
(Grokipedia articles run several times longer), and its zero point is the
sample mean difference, an arbitrary quantity. This module replaces it with

    L = log(W_G / W_W),

the log length ratio. L is symmetric in the two encyclopedias, additive in the
multiplicative disparity, and its zero point is the substantively meaningful
one, namely a pair whose two articles are the same length. A negative intercept
at L = 0 is then a real answer to the length objection rather than an artefact
of where the sample mean happens to sit, provided the reader can see how far
L = 0 is from the observed data. ``support_table`` reports exactly that, and
``linearity_table`` checks the log-linearity that the extrapolation rests on.

The second half tests a prediction the length-artefact story does not make. If
the ideological gap reflects contested subject matter rather than a byproduct of
verbosity, the ideology effect should be *larger* in pairs where Grokipedia
expanded more, so each ideology x L interaction should carry the same sign as
its own main effect (amplification) rather than the opposite sign
(attenuation). That direction is what ``interaction_table`` reports, per
dimension and jointly, with the rightwing dimension broken out across the range
of L in ``marginal_effects_table``.

Run standalone::

    python -m src.analysis.statistical_analysis.length_functional_form
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .config import (
    CLUSTER_COLUMN, CONTINUOUS_INDEPENDENT_VARS, DEFAULT_LEADERS_PATH,
)
from .visualisation.helpers import (
    compute_article_length, pretty_dep_label, save_figure, setup_plot_style,
)

WIKI, GROKI = "wikipedia", "grokipedia_0.2"
ARTICLE_COLUMN = {WIKI: "wikipedia", GROKI: "grokipedia_0_2"}
OUTCOME = "neutrality_assessment"

RIGHTWING = "v2pariglef_osp"
LGBT = "v2palgbt_osp"
CONFIRMATORY = (RIGHTWING, LGBT)

RATIO = "log_length_ratio"          # L = log(W_G / W_W)
LOG_WIKI = "log_len_wiki"
LOG_GROKI = "log_len_groki"
LINEAR_DIFF = "article_length_diff"  # z-scored (W_W - W_G), the paper's control

N_BINS = 5
_ME_QUANTILES = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0)


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


def _marginal_effect(res, var: str, terms: Dict[str, float]) -> Tuple[float, float]:
    """Effect of *var* with each interaction term evaluated at a given value."""
    names = list(res.params.index)
    contrast = np.zeros(len(names))
    contrast[names.index(var)] = 1.0
    for term, value in terms.items():
        contrast[names.index(term)] = value
    est = float(contrast @ res.params.values)
    se = float(np.sqrt(contrast @ res.cov_params().values @ contrast))
    return est, se


def _p_two_sided(z: float) -> float:
    from scipy.stats import norm
    return float(2.0 * (1.0 - norm.cdf(abs(z))))


def _fit(formula: str, data: pd.DataFrame, cluster_col: Optional[str]):
    model = smf.ols(formula, data=data)
    if cluster_col is not None and cluster_col in data.columns:
        return model.fit(cov_type="cluster", cov_kwds={"groups": data[cluster_col]})
    return model.fit(cov_type="HC1")


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Differential frame (one row per politician x judge), as built in main.py."""
    return (
        df[df["Source"] == WIKI]
        .merge(
            df[df["Source"] == GROKI][["name", "country_isocode", "Model", OUTCOME]],
            on=["name", "country_isocode", "Model"],
            suffixes=("", "_groki"),
        )
        .assign(**{OUTCOME: lambda x: x[OUTCOME] - x[f"{OUTCOME}_groki"]})
        .drop(columns=[f"{OUTCOME}_groki"])
    )


def add_length_vars(pairs: pd.DataFrame) -> pd.DataFrame:
    """Add both length parameterisations and the quantile bins of L."""
    pairs = pairs.copy()
    wiki_len = pairs[ARTICLE_COLUMN[WIKI]].apply(compute_article_length).astype(float)
    groki_len = pairs[ARTICLE_COLUMN[GROKI]].apply(compute_article_length).astype(float)
    pairs["len_wiki"], pairs["len_groki"] = wiki_len, groki_len
    pairs = pairs.dropna(subset=["len_wiki", "len_groki"])
    pairs = pairs[(pairs["len_wiki"] > 0) & (pairs["len_groki"] > 0)].copy()

    pairs[LOG_WIKI] = np.log(pairs["len_wiki"])
    pairs[LOG_GROKI] = np.log(pairs["len_groki"])
    pairs[RATIO] = pairs[LOG_GROKI] - pairs[LOG_WIKI]

    # The paper's control, rebuilt here so the ladder below is self-contained.
    diff = pairs["len_wiki"] - pairs["len_groki"]
    pairs[LINEAR_DIFF] = (diff - diff.mean()) / diff.std()

    pairs["len_bin"] = pd.qcut(pairs[RATIO], N_BINS, labels=False).astype(int)
    return pairs


def support_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Where the data on L actually sit, and how far L = 0 is from them.

    Reported at the politician level (one row per pair, not per judge) since L is
    a property of the article pair and every pair is repeated across judges.
    """
    pol = pairs.drop_duplicates("name")
    L = pol[RATIO]
    rows = [
        ("politicians (pairs)", f"{len(pol)}"),
        ("rating rows (pairs x judges)", f"{len(pairs)}"),
        ("median words, Wikipedia", f"{pol['len_wiki'].median():.0f}"),
        ("median words, Grokipedia", f"{pol['len_groki'].median():.0f}"),
        ("median expansion factor W_G/W_W", f"{np.exp(L.median()):.2f}x"),
        ("mean L = log(W_G/W_W)", f"{L.mean():.3f}"),
        ("sd L", f"{L.std():.3f}"),
        ("min L", f"{L.min():.3f}"),
        ("p1 L", f"{L.quantile(0.01):.3f}"),
        ("p5 L", f"{L.quantile(0.05):.3f}"),
        ("q25 L", f"{L.quantile(0.25):.3f}"),
        ("median L", f"{L.median():.3f}"),
        ("q75 L", f"{L.quantile(0.75):.3f}"),
        ("p95 L", f"{L.quantile(0.95):.3f}"),
        ("max L", f"{L.max():.3f}"),
        ("pairs with L <= 0 (Wikipedia at least as long)", f"{int((L <= 0).sum())}"),
        ("share with L <= 0", f"{(L <= 0).mean() * 100:.1f}%"),
        ("pairs with L <= 0.1", f"{int((L <= 0.1).sum())}"),
        ("distance from mean L to 0, in sd of L", f"{L.mean() / L.std():.2f}"),
        ("L = 0 inside observed range", "yes" if L.min() <= 0 <= L.max() else "no"),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value"])


def spec_ladder(
    pairs: pd.DataFrame,
    ideology_vars: Sequence[str],
    cluster_col: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Eq. 1 under four length specifications, coefficients side by side.

    (1) no length control, (2) the paper's linear z-scored word-count
    difference, (3) L entered linearly, (4) L plus a rightwing x L interaction.
    """
    ideo = [v for v in ideology_vars if v in pairs.columns]
    base = " + ".join(ideo)
    specs = {
        "(1) no length": f"{OUTCOME} ~ {base}",
        "(2) linear dW (paper)": f"{OUTCOME} ~ {base} + {LINEAR_DIFF}",
        "(3) log ratio L": f"{OUTCOME} ~ {base} + {RATIO}",
        "(4) L + rightwing x L": f"{OUTCOME} ~ {base} + {RATIO} + {RIGHTWING}:{RATIO}",
    }
    fits = {label: _fit(f, pairs, cluster_col) for label, f in specs.items()}

    row_order = ["Intercept"] + list(ideo) + [LINEAR_DIFF, RATIO, f"{RIGHTWING}:{RATIO}"]
    rows = []
    for name in row_order:
        label = "Intercept" if name == "Intercept" else pretty_dep_label(name)
        if ":" in name:
            label = " x ".join(pretty_dep_label(p) for p in name.split(":"))
        row = {"Variable": label}
        for spec_label, res in fits.items():
            row[spec_label] = (
                _cell(res.params[name], res.bse[name], res.pvalues[name])
                if name in res.params.index else ""
            )
        rows.append(row)

    rows.append({"Variable": "Observations",
                 **{k: f"{int(r.nobs)}" for k, r in fits.items()}})
    rows.append({"Variable": "R-squared",
                 **{k: f"{r.rsquared:.3f}" for k, r in fits.items()}})
    # The intercept of spec (3) is the equal-length counterfactual; report its
    # interval explicitly, since the whole extrapolation argument rests on it.
    ci = fits["(3) log ratio L"].conf_int()
    rows.append({"Variable": "Intercept 95% CI",
                 "(3) log ratio L": f"[{ci.loc['Intercept', 0]:.3f}, {ci.loc['Intercept', 1]:.3f}]"})
    return pd.DataFrame(rows), fits


def equal_length_table(
    pairs: pd.DataFrame,
    ideology_vars: Sequence[str],
    cluster_col: Optional[str],
) -> pd.DataFrame:
    """The intercept read at three points on L, at mean ideology.

    Under spec (3) the intercept is the expected differential for a pair of
    equally long articles, which is out of sample. The same fit evaluated at the
    observed mean and median of L stays inside it, so the three rows together
    show how much of the negative intercept is extrapolation.
    """
    ideo = [v for v in ideology_vars if v in pairs.columns]
    res = _fit(f"{OUTCOME} ~ {' + '.join(ideo)} + {RATIO}", pairs, cluster_col)
    L = pairs[RATIO]
    names = list(res.params.index)

    rows = []
    for label, value in (
        ("L = 0 (equal length)", 0.0),
        ("L = observed min", float(L.min())),
        ("L = p5", float(L.quantile(0.05))),
        ("L = sample mean", float(L.mean())),
        ("L = median", float(L.median())),
    ):
        contrast = np.zeros(len(names))
        contrast[names.index("Intercept")] = 1.0
        contrast[names.index(RATIO)] = value
        est = float(contrast @ res.params.values)
        se = float(np.sqrt(contrast @ res.cov_params().values @ contrast))
        near = pairs.loc[(L - value).abs() <= 0.25, "name"].nunique()
        rows.append({
            "evaluated at": label,
            "L": f"{value:.3f}",
            "W_G/W_W": f"{np.exp(value):.2f}x",
            "predicted dY": f"{est:.3f}",
            "se": f"{se:.3f}",
            "95% CI": f"[{est - 1.96 * se:.3f}, {est + 1.96 * se:.3f}]",
            "pairs within +-0.25": f"{near}",
            "in support": "yes" if L.min() <= value <= L.max() else "no (extrapolation)",
        })
    return pd.DataFrame(rows)


def linearity_table(
    pairs: pd.DataFrame,
    ideology_vars: Sequence[str],
    cluster_col: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Is the length effect log-linear? Quantile bins of L against the line.

    The extrapolation to L = 0 is only as good as the linearity assumption, so
    the linear term is compared with a saturated set of quintile dummies. The
    joint Wald test on the dummies is the test of log-linearity; the bin means
    show where any departure sits.
    """
    ideo = [v for v in ideology_vars if v in pairs.columns]
    base = " + ".join(ideo)
    linear = _fit(f"{OUTCOME} ~ {base} + {RATIO}", pairs, cluster_col)
    binned = _fit(f"{OUTCOME} ~ {base} + C(len_bin)", pairs, cluster_col)
    both = _fit(f"{OUTCOME} ~ {base} + {RATIO} + C(len_bin)", pairs, cluster_col)

    names = list(both.params.index)
    dummies = [n for n in names if n.startswith("C(len_bin)")]
    R = np.zeros((len(dummies), len(names)))
    for i, name in enumerate(dummies):
        R[i, names.index(name)] = 1.0
    wald = both.wald_test(R, use_f=False, scalar=True)

    b_slope = float(linear.params[RATIO])
    b0 = float(linear.params["Intercept"])
    rows = []
    for b, sub in pairs.groupby("len_bin"):
        centre = float(sub[RATIO].median())
        fitted = float(binned.params["Intercept"])
        key = f"C(len_bin)[T.{b}]"
        if key in binned.params.index:
            fitted += float(binned.params[key])
        rows.append({
            "bin": f"Q{b + 1}",
            "L range": f"[{sub[RATIO].min():.2f}, {sub[RATIO].max():.2f}]",
            "median L": f"{centre:.3f}",
            "median W_G/W_W": f"{np.exp(centre):.2f}x",
            "n rows": int(len(sub)),
            "n pairs": int(sub["name"].nunique()),
            "binned dY at mean ideology": f"{fitted:.3f}",
            "linear-in-L prediction": f"{b0 + b_slope * centre:.3f}",
        })
    bins = pd.DataFrame(rows)

    test = pd.DataFrame([
        {"test": "bin dummies add nothing beyond linear L (log-linearity)",
         "chi2": f"{float(wald.statistic):.2f}", "df": f"{len(dummies)}",
         "p": f"{float(wald.pvalue):.3f}{_stars(float(wald.pvalue))}"},
        {"test": "R2, linear L", "chi2": "", "df": "", "p": f"{linear.rsquared:.4f}"},
        {"test": "R2, quintile dummies", "chi2": "", "df": "", "p": f"{binned.rsquared:.4f}"},
    ])
    return bins, test


def separate_slopes_table(
    pairs: pd.DataFrame,
    ideology_vars: Sequence[str],
    cluster_col: Optional[str],
) -> pd.DataFrame:
    """Relax the ratio restriction: log W_W and log W_G with their own slopes.

    Entering L alone imposes equal and opposite slopes on the two log lengths.
    Fitting them freely and testing that restriction says whether the ratio is
    the right one-dimensional summary of the pair's lengths.
    """
    ideo = [v for v in ideology_vars if v in pairs.columns]
    res = _fit(
        f"{OUTCOME} ~ {' + '.join(ideo)} + {LOG_WIKI} + {LOG_GROKI}",
        pairs, cluster_col,
    )
    names = list(res.params.index)
    R = np.zeros((1, len(names)))
    R[0, names.index(LOG_WIKI)] = 1.0
    R[0, names.index(LOG_GROKI)] = 1.0
    wald = res.wald_test(R, use_f=False, scalar=True)

    rows = [
        {"term": "log W_W (Wikipedia)",
         "coef": _cell(res.params[LOG_WIKI], res.bse[LOG_WIKI], res.pvalues[LOG_WIKI])},
        {"term": "log W_G (Grokipedia)",
         "coef": _cell(res.params[LOG_GROKI], res.bse[LOG_GROKI], res.pvalues[LOG_GROKI])},
        {"term": "Intercept", "coef": _cell(res.params["Intercept"], res.bse["Intercept"],
                                            res.pvalues["Intercept"])},
        {"term": "ratio restriction b_W + b_G = 0",
         "coef": f"chi2={float(wald.statistic):.2f}, p={float(wald.pvalue):.3f}"
                 f"{_stars(float(wald.pvalue))}"},
        {"term": "R-squared", "coef": f"{res.rsquared:.4f}"},
    ]
    return pd.DataFrame(rows)


def _interaction_formula(ideo: Sequence[str], var: str, moderators: Sequence[str]) -> str:
    """Eq. 1 plus *moderators* as main effects and as interactions with *var*."""
    return (
        f"{OUTCOME} ~ {' + '.join(ideo)} + {' + '.join(moderators)} + "
        + " + ".join(f"{var}:{m}" for m in moderators)
    )


def _joint_wald(res, terms: Sequence[str]):
    names = list(res.params.index)
    R = np.zeros((len(terms), len(names)))
    for i, t in enumerate(terms):
        R[i, names.index(t)] = 1.0
    w = res.wald_test(R, use_f=False, scalar=True)
    return float(w.statistic), float(w.pvalue), len(terms)


def interaction_table(
    pairs: pd.DataFrame,
    ideology_vars: Sequence[str],
    cluster_col: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ideology x L, one dimension at a time and all nine at once.

    ``direction`` is the substantive read. Contestedness predicts that a pair
    Grokipedia expanded more shows a *larger* ideology effect, so the
    interaction should share the sign of the main effect ("amplifies").
    The opposite sign ("attenuates") is what a verbosity artefact would give.
    """
    ideo = [v for v in ideology_vars if v in pairs.columns]
    base = " + ".join(ideo)

    single_rows = []
    for var in ideo:
        res = _fit(_interaction_formula(ideo, var, [RATIO]), pairs, cluster_col)
        term = f"{var}:{RATIO}"
        main, inter = float(res.params[var]), float(res.params[term])
        single_rows.append({
            "dimension": pretty_dep_label(var),
            "main effect (at L=0)": _cell(res.params[var], res.bse[var], res.pvalues[var]),
            "x L": _cell(res.params[term], res.bse[term], res.pvalues[term]),
            "p(interaction)": f"{res.pvalues[term]:.3f}",
            "direction": ("amplifies" if main * inter > 0 else "attenuates"),
            "confirmatory": "yes" if var in CONFIRMATORY else "",
        })
    single = pd.DataFrame(single_rows)

    # All nine interactions together, plus the joint test.
    inter_terms = " + ".join(f"{v}:{RATIO}" for v in ideo)
    full = _fit(f"{OUTCOME} ~ {base} + {RATIO} + {inter_terms}", pairs, cluster_col)
    terms = [f"{v}:{RATIO}" for v in ideo]
    chi2, p, k = _joint_wald(full, terms)

    joint_rows = [{"test": "all 9 ideology x L = 0", "chi2": f"{chi2:.2f}",
                   "df": f"{k}", "p": f"{p:.4f}{_stars(p)}"}]
    for var in CONFIRMATORY:
        term = f"{var}:{RATIO}"
        if term not in full.params.index:
            continue
        joint_rows.append({
            "test": f"{pretty_dep_label(var)} x L (in the all-nine fit)",
            "chi2": f"{(full.params[term] / full.bse[term]) ** 2:.2f}",
            "df": "1",
            "p": f"{full.pvalues[term]:.4f}{_stars(float(full.pvalues[term]))}",
        })
    joint_rows.append({"test": "R2, all-nine interaction fit", "chi2": "", "df": "",
                       "p": f"{full.rsquared:.4f}"})
    return single, pd.DataFrame(joint_rows)


def interaction_decomposition_table(
    pairs: pd.DataFrame,
    ideology_vars: Sequence[str],
    cluster_col: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split each ideology x L interaction into its two halves.

    A large L means Grokipedia wrote a lot *or* Wikipedia wrote little, and the
    two carry different stories. The first is Grokipedia expanding on a subject;
    the second is thin Wikipedia coverage of an obscure one. Letting log W_W and
    log W_G each carry their own interaction separates them, and the
    equal-and-opposite test says whether L was an adequate summary at all. The
    x log W_G column is the contestedness test proper: how the ideology effect
    moves with how much Grokipedia wrote, holding Wikipedia's coverage of the
    same politician fixed.
    """
    ideo = [v for v in ideology_vars if v in pairs.columns]
    mods = [LOG_WIKI, LOG_GROKI]

    rows = []
    for var in ideo:
        res = _fit(_interaction_formula(ideo, var, mods), pairs, cluster_col)
        t_w, t_g = f"{var}:{LOG_WIKI}", f"{var}:{LOG_GROKI}"
        names = list(res.params.index)
        R = np.zeros((1, len(names)))
        R[0, names.index(t_w)] = 1.0
        R[0, names.index(t_g)] = 1.0
        w = res.wald_test(R, use_f=False, scalar=True)
        # Read the main effect at the mean of both log lengths: the raw
        # coefficient sits at log W = 0, i.e. a one-word article.
        at_mean = {f"{var}:{m}": float(pairs[m].mean()) for m in mods}
        main, se_main = _marginal_effect(res, var, at_mean)
        inter_g = float(res.params[t_g])
        rows.append({
            "dimension": pretty_dep_label(var),
            "main effect (at mean lengths)":
                _cell(main, se_main, _p_two_sided(main / se_main)),
            "x log W_W": _cell(res.params[t_w], res.bse[t_w], res.pvalues[t_w]),
            "x log W_G": _cell(res.params[t_g], res.bse[t_g], res.pvalues[t_g]),
            "direction (W_G)": "amplifies" if main * inter_g > 0 else "attenuates",
            "L restriction (sum = 0)":
                f"chi2={float(w.statistic):.2f}, p={float(w.pvalue):.3f}"
                f"{_stars(float(w.pvalue))}",
            "confirmatory": "yes" if var in CONFIRMATORY else "",
        })
    single = pd.DataFrame(rows)

    # All nine, both moderators at once: joint tests on each block.
    inter_terms = " + ".join(f"{v}:{m}" for v in ideo for m in mods)
    full = _fit(
        f"{OUTCOME} ~ {' + '.join(ideo)} + {' + '.join(mods)} + {inter_terms}",
        pairs, cluster_col,
    )
    tests = []
    for label, terms in (
        ("all 9 ideology x log W_G = 0", [f"{v}:{LOG_GROKI}" for v in ideo]),
        ("all 9 ideology x log W_W = 0", [f"{v}:{LOG_WIKI}" for v in ideo]),
        ("all 18 interactions = 0", [f"{v}:{m}" for v in ideo for m in mods]),
    ):
        chi2, p, k = _joint_wald(full, terms)
        tests.append({"test": label, "chi2": f"{chi2:.2f}", "df": f"{k}",
                      "p": f"{p:.4f}{_stars(p)}"})
    for var in CONFIRMATORY:
        term = f"{var}:{LOG_GROKI}"
        if term not in full.params.index:
            continue
        tests.append({
            "test": f"{pretty_dep_label(var)} x log W_G (in the all-18 fit)",
            "chi2": f"{(full.params[term] / full.bse[term]) ** 2:.2f}", "df": "1",
            "p": f"{full.pvalues[term]:.4f}{_stars(float(full.pvalues[term]))}",
        })
    tests.append({"test": "R2, all-18 interaction fit", "chi2": "", "df": "",
                  "p": f"{full.rsquared:.4f}"})
    return single, pd.DataFrame(tests)


def marginal_effects_table(
    pairs: pd.DataFrame,
    ideology_vars: Sequence[str],
    cluster_col: Optional[str],
    var: str = RIGHTWING,
    moderator: str = RATIO,
) -> pd.DataFrame:
    """The effect of one ideology dimension across the observed range of *moderator*.

    With ``moderator=LOG_GROKI`` the fit also carries log W_W and its own
    interaction, held at its sample mean, so the column varies how much
    Grokipedia wrote with Wikipedia's coverage of the same politician fixed.
    """
    ideo = [v for v in ideology_vars if v in pairs.columns]
    mods = [RATIO] if moderator == RATIO else [LOG_WIKI, LOG_GROKI]
    res = _fit(_interaction_formula(ideo, var, mods), pairs, cluster_col)
    held = {f"{var}:{m}": float(pairs[m].mean()) for m in mods if m != moderator}

    x = pairs[moderator]
    points = [("equal length" if moderator == RATIO else "1,000 words",
               0.0 if moderator == RATIO else float(np.log(1000)))]
    points += [(f"p{int(q * 100)}", float(x.quantile(q))) for q in _ME_QUANTILES]

    rows = []
    for label, value in points:
        est, se = _marginal_effect(res, var, {**held, f"{var}:{moderator}": value})
        row = {
            "point": label,
            moderator: f"{value:.3f}",
            "words" if moderator != RATIO else "W_G/W_W":
                f"{np.exp(value):.0f}" if moderator != RATIO else f"{np.exp(value):.2f}x",
            f"d dY / d {pretty_dep_label(var)}": f"{est:.3f}",
            "se": f"{se:.3f}",
            "95% CI": f"[{est - 1.96 * se:.3f}, {est + 1.96 * se:.3f}]",
            "p": f"{_p_two_sided(est / se):.3f}",
            "in support": "yes" if x.min() <= value <= x.max() else "no",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def per_judge_interaction_table(
    pairs: pd.DataFrame,
    ideology_vars: Sequence[str],
    cluster_col: Optional[str],
    variables: Sequence[str] = CONFIRMATORY,
) -> pd.DataFrame:
    """Is the x log W_G interaction one judge's quirk, or all four?"""
    from .config import MODEL_DISPLAY
    ideo = [v for v in ideology_vars if v in pairs.columns]
    present = set(pairs["Model"].astype(str).unique())
    judges = [("Pooled", pairs)] + [
        (MODEL_DISPLAY.get(j, j), pairs[pairs["Model"].astype(str) == j])
        for j in list(MODEL_DISPLAY) + sorted(present - set(MODEL_DISPLAY))
        if j in present
    ]
    rows = []
    for var in variables:
        row = {"dimension": pretty_dep_label(var), "term": "x log W_G"}
        for label, sub in judges:
            res = _fit(_interaction_formula(ideo, var, [LOG_WIKI, LOG_GROKI]),
                       sub, cluster_col)
            t = f"{var}:{LOG_GROKI}"
            row[label] = _cell(res.params[t], res.bse[t], res.pvalues[t])
        rows.append(row)
    return pd.DataFrame(rows)


def controls_robustness_table(
    pairs: pd.DataFrame,
    ideology_vars: Sequence[str],
    cluster_col: Optional[str],
    variables: Sequence[str] = CONFIRMATORY,
) -> pd.DataFrame:
    """The x log W_G interaction with region fixed effects, and per-source.

    The within-source rows re-ask the question on ratings in levels rather than
    on the differential: does the ideology slope of a source steepen with the
    length of the article that source wrote, holding the other encyclopedia's
    length (a prominence proxy the judge never saw) fixed. That is the same test
    as the differential row, localised to one side of the pair.
    """
    from .config import REGION_COLUMN
    ideo = [v for v in ideology_vars if v in pairs.columns]
    rows = []
    for var in variables:
        base = _interaction_formula(ideo, var, [LOG_WIKI, LOG_GROKI])
        t = f"{var}:{LOG_GROKI}"
        specs = [("differential, no controls", base, pairs)]
        if REGION_COLUMN in pairs.columns:
            specs.append(("differential, + region FE",
                          f"{base} + C({REGION_COLUMN})", pairs))
        for label, formula, data in specs:
            res = _fit(formula, data, cluster_col)
            rows.append({"dimension": pretty_dep_label(var), "spec": label,
                         "x log W_G": _cell(res.params[t], res.bse[t], res.pvalues[t]),
                         "n": f"{int(res.nobs)}"})
    return pd.DataFrame(rows)


def per_source_interaction_table(
    df: pd.DataFrame,
    ideology_vars: Sequence[str],
    cluster_col: Optional[str],
    variables: Sequence[str] = CONFIRMATORY,
) -> pd.DataFrame:
    """Ratings in levels: ideology x own log length within each source.

    ``log_other`` (the counterpart encyclopedia's length) is carried as a
    prominence proxy the judge never read, so the own-length interaction is a
    property of the rated text rather than of the subject's notability.
    """
    ideo = [v for v in ideology_vars if v in df.columns]
    rows = []
    for source, label in ((WIKI, "Wikipedia"), (GROKI, "Grokipedia")):
        sub = df[df["Source"] == source].copy()
        own = sub[ARTICLE_COLUMN[source]].apply(compute_article_length).astype(float)
        other_src = GROKI if source == WIKI else WIKI
        other = sub[ARTICLE_COLUMN[other_src]].apply(compute_article_length).astype(float)
        sub["log_own"], sub["log_other"] = np.log(own), np.log(other)
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["log_own", "log_other"])
        for var in variables:
            formula = (
                f"{OUTCOME} ~ {' + '.join(ideo)} + log_own + log_other"
                f" + {var}:log_own + C(Model)"
            )
            res = _fit(formula, sub, cluster_col)
            t = f"{var}:log_own"
            main, se_main = _marginal_effect(
                res, var, {t: float(sub["log_own"].mean())})
            rows.append({
                "source": label,
                "dimension": pretty_dep_label(var),
                "main effect (at mean own length)":
                    _cell(main, se_main, _p_two_sided(main / se_main)),
                "x log own length": _cell(res.params[t], res.bse[t], res.pvalues[t]),
                "direction": "amplifies" if main * float(res.params[t]) > 0 else "attenuates",
                "n": f"{int(res.nobs)}",
            })
    return pd.DataFrame(rows)


def expansion_bins_table(
    pairs: pd.DataFrame,
    ideology_vars: Sequence[str],
    cluster_col: Optional[str],
    variables: Sequence[str] = CONFIRMATORY,
    n_bins: int = 3,
) -> pd.DataFrame:
    """The same moderation without the log-linear form: tertiles of log W_G.

    The ideology dimension is interacted with bin dummies instead of with log
    W_G itself, so a monotone pattern across the bins is evidence the linear
    interaction is not an artefact of the functional form or of the long tail.
    log W_W and its interaction stay in, holding Wikipedia's coverage fixed.
    """
    ideo = [v for v in ideology_vars if v in pairs.columns]
    data = pairs.copy()
    data["groki_bin"] = pd.qcut(data[LOG_GROKI], n_bins, labels=False).astype(int)
    rows = []
    for var in variables:
        res = _fit(
            f"{OUTCOME} ~ {' + '.join(ideo)} + {LOG_WIKI} + C(groki_bin)"
            f" + {var}:{LOG_WIKI} + {var}:C(groki_bin)",
            data, cluster_col,
        )
        held = {f"{var}:{LOG_WIKI}": float(data[LOG_WIKI].mean())}
        for b, sub in data.groupby("groki_bin"):
            # The reference bin has no dummy: its effect is the main effect.
            term = next(
                (t for t in (f"{var}:C(groki_bin)[{b}]", f"{var}:C(groki_bin)[T.{b}]")
                 if t in res.params.index),
                None,
            )
            terms = dict(held) if term is None else {**held, term: 1.0}
            est, se = _marginal_effect(res, var, terms)
            rows.append({
                "dimension": pretty_dep_label(var),
                "bin": f"T{b + 1}",
                "W_G words": f"{sub['len_groki'].min():.0f}-{sub['len_groki'].max():.0f}",
                "median W_G": f"{sub['len_groki'].median():.0f}",
                "n pairs": int(sub["name"].nunique()),
                "effect on dY": f"{est:.3f}",
                "se": f"{se:.3f}",
                "p": f"{_p_two_sided(est / se):.3f}",
            })
    return pd.DataFrame(rows)


def plot_marginal_effect(
    pairs: pd.DataFrame,
    ideology_vars: Sequence[str],
    cluster_col: Optional[str],
    save_path: Path,
    var: str = RIGHTWING,
) -> None:
    """Effect of *var* on dY across L (left) and across log W_G (right)."""
    ideo = [v for v in ideology_vars if v in pairs.columns]
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 4.6), sharex="col", sharey="row",
                             gridspec_kw={"height_ratios": [4, 1], "hspace": 0.1,
                                          "wspace": 0.12})
    panels = (
        (RATIO, [RATIO], r"$L = \log(W_G / W_W)$", "log length ratio"),
        (LOG_GROKI, [LOG_WIKI, LOG_GROKI], r"$\log W_G$ (Wikipedia length held fixed)",
         "Grokipedia article length"),
    )
    for col, (moderator, mods, xlabel, title) in enumerate(panels):
        res = _fit(_interaction_formula(ideo, var, mods), pairs, cluster_col)
        held = {f"{var}:{m}": float(pairs[m].mean()) for m in mods if m != moderator}
        x = pairs.drop_duplicates("name")[moderator]
        lo = min(0.0, x.min()) - 0.05 if moderator == RATIO else x.min()
        grid = np.linspace(lo, x.max(), 200)
        est, se = zip(*(
            _marginal_effect(res, var, {**held, f"{var}:{moderator}": g}) for g in grid
        ))
        est, se = np.asarray(est), np.asarray(se)

        ax, ax_rug = axes[0, col], axes[1, col]
        ax.fill_between(grid, est - 1.96 * se, est + 1.96 * se,
                        color="#7570b3", alpha=0.22, lw=0)
        ax.plot(grid, est, color="#7570b3", lw=1.8)
        ax.axhline(0, color="#9a9a9a", lw=0.8, ls="--")
        if moderator == RATIO:
            ax.axvline(0, color="#d95f02", lw=1.0, ls=":")
        ax.set_title(title, fontsize=10)
        if col == 0:
            ax.set_ylabel(rf"effect of {pretty_dep_label(var)} on $\Delta y$", fontsize=10)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        ax_rug.plot(x, np.zeros(len(x)), "|", color="#444444", alpha=0.35, ms=8)
        ax_rug.set_yticks([])
        ax_rug.set_xlabel(xlabel, fontsize=10)
        if moderator == RATIO:
            ax_rug.axvline(0, color="#d95f02", lw=1.0, ls=":")
        for spine in ("top", "right", "left"):
            ax_rug.spines[spine].set_visible(False)

    fig.suptitle(
        f"{pretty_dep_label(var)} effect on the differential, by article length",
        fontsize=11, fontweight="bold",
    )
    save_figure(fig, save_path, dpi=300)
    save_figure(fig, Path(save_path).with_suffix(".pdf"), dpi=300)
    plt.close()


def run(
    df_wiki_minus_groki: pd.DataFrame,
    ideology_vars: Sequence[str] = tuple(CONTINUOUS_INDEPENDENT_VARS),
    cluster_col: Optional[str] = CLUSTER_COLUMN,
    df: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Return every table keyed by output name, plus the prepared pair frame."""
    pairs = add_length_vars(df_wiki_minus_groki)
    ladder, _ = spec_ladder(pairs, ideology_vars, cluster_col)
    bins, linearity = linearity_table(pairs, ideology_vars, cluster_col)
    single, joint = interaction_table(pairs, ideology_vars, cluster_col)
    decomp, decomp_joint = interaction_decomposition_table(pairs, ideology_vars, cluster_col)
    tables = {
        "support": support_table(pairs),
        "spec_ladder": ladder,
        "equal_length_prediction": equal_length_table(pairs, ideology_vars, cluster_col),
        "log_linearity_bins": bins,
        "log_linearity_test": linearity,
        "separate_log_slopes": separate_slopes_table(pairs, ideology_vars, cluster_col),
        "ideology_x_length_single": single,
        "ideology_x_length_joint": joint,
        "interaction_decomposition": decomp,
        "interaction_decomposition_joint": decomp_joint,
        "rightwing_me_by_ratio": marginal_effects_table(
            pairs, ideology_vars, cluster_col, RIGHTWING, RATIO),
        "rightwing_me_by_groki_length": marginal_effects_table(
            pairs, ideology_vars, cluster_col, RIGHTWING, LOG_GROKI),
        "lgbt_me_by_ratio": marginal_effects_table(
            pairs, ideology_vars, cluster_col, LGBT, RATIO),
        "lgbt_me_by_groki_length": marginal_effects_table(
            pairs, ideology_vars, cluster_col, LGBT, LOG_GROKI),
        "per_judge_interaction": per_judge_interaction_table(
            pairs, ideology_vars, cluster_col),
        "controls_robustness": controls_robustness_table(
            pairs, ideology_vars, cluster_col),
        "expansion_bins": expansion_bins_table(pairs, ideology_vars, cluster_col),
    }
    if df is not None:
        tables["per_source_interaction"] = per_source_interaction_table(
            df, ideology_vars, cluster_col)
    return tables, pairs


def run_length_functional_form(
    df_wiki_minus_groki: pd.DataFrame,
    ideology_vars: Sequence[str],
    cluster_col: Optional[str],
    table_dir: Path,
    figure_dir: Path,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """Pipeline entry point: write the tables and the marginal-effect figures."""
    tables, pairs = run(df_wiki_minus_groki, ideology_vars, cluster_col, df=df)

    out_dir = Path(table_dir) / "length_functional_form"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "Article-length functional form and ideology x length interaction (Eq. 1)",
        "=" * 72,
        "",
        "L = log(W_G / W_W), the log Grokipedia/Wikipedia word-count ratio.",
        "Outcome dY = Y_wikipedia - Y_grokipedia; negative means Grokipedia is rated",
        "  to portray the politician more favourably.",
        "All fits are OLS with cluster-robust SEs by party. Ideology dims are z-scored,",
        "  so a main effect is read at the sample ideology mean and the length terms",
        "  are in natural logs of words (1 unit = one e-fold, 0.69 = one doubling).",
        "Significance: * p<0.05, ** p<0.01, *** p<0.001.",
        "",
    ]
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
        lines += [f"--- {name} ---", table.to_string(index=False), ""]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out_dir / 'summary.txt'}")

    fig_dir = Path(figure_dir) / "length_functional_form"
    for var, stem in ((RIGHTWING, "rightwing"), (LGBT, "lgbt")):
        plot_marginal_effect(pairs, ideology_vars, cluster_col,
                             save_path=fig_dir / f"{stem}_by_length.png", var=var)
    return tables


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=Path, default=Path("results/final2.0"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--experiment", default="ideology")
    args = parser.parse_args(argv)

    from .data_loader import load_and_prepare_data

    df = load_and_prepare_data(
        results_path=args.results_path,
        leaders_path=DEFAULT_LEADERS_PATH,
        experiment=args.experiment,
        continuous_independent_vars=list(CONTINUOUS_INDEPENDENT_VARS),
    )
    out_root = args.output_dir or args.results_path
    tables = run_length_functional_form(
        df_wiki_minus_groki=build_pairs(df),
        ideology_vars=list(CONTINUOUS_INDEPENDENT_VARS),
        cluster_col=CLUSTER_COLUMN,
        table_dir=out_root / "tables",
        figure_dir=out_root / "figures",
        df=df,
    )
    for name, table in tables.items():
        print(f"\n--- {name}")
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
