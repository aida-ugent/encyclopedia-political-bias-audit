"""Reliability / ceiling / LLM-vs-consensus validation for the neutrality survey.

The complement to agreement.py's kappa-based tables: the classical
annotation-study apparatus (ordinal Krippendorff's alpha, ICC, the attenuation
ceiling, rank correlations against the human consensus, and plain interpretable
agreement), all on the same articles, with one cluster bootstrap shared by every
statistic so the differences are paired.

What is computed, in the order of the validation subsection it feeds:

1. Human reliability -- ordinal Krippendorff's alpha on the n_articles x 2 human
   matrix (the coefficient annotation-study readers expect; handles the nested
   design natively, since the 140 ratings come from 140 distinct raters), plus
   the interval alpha for continuity with agreement.py's ``human_irr`` and
   ICC(1,1) on the same matrix.  Report alpha in the text; ICC drives the
   ceiling arithmetic below, because Spearman-Brown is exact for ICC and only
   approximate for alpha.  The two should land close together -- ``alpha_icc_gap``
   is printed so a divergence is not missed.

2. The ceiling -- ICC(1,2) = 2*ICC(1,1) / (1 + ICC(1,1)) is the reliability of
   the 2-rater MEAN; no measure whatsoever can correlate with that mean better
   than ``ceiling_r`` = sqrt(ICC(1,2)).  (Sanity check, verified in
   ``_selftest``: ICC(1,1) = 0.35 -> ICC(1,2) = 0.52 -> ceiling = 0.72.)

3. LLM vs consensus -- Pearson r (the coefficient the attenuation argument is
   about), Spearman rho (what readers expect) and Kendall tau_b (the cleaner
   rank statistic under the heavy ties a 5-point scale produces) between each
   model's rating and the human mean, plus ``r_over_ceiling`` = r / ceiling_r:
   the share of the ACHIEVABLE signal attained.  The ratio is reported instead
   of a disattenuated coefficient on purpose -- corrected correlations can
   exceed 1.0 and read as inflation.

4. Substitution check -- ordinal alpha between the model and each human
   SEPARATELY, averaged over the two, against alpha(H1, H2): "could the model
   stand in for one annotator?", a more demanding question than 3.  The average
   over both humans is the symmetrization, and it is mandatory rather than
   cosmetic: with one rater per response there is no such thing as "the first
   human", so which rating of an article lands in column 1 of the export is
   arbitrary.

5. Raw agreement -- exact %, within-1 %, MAE and MEAN SIGNED difference, for
   human-human and for human-LLM, plus SD of model ratings vs SD of human
   ratings.  The signed difference and the SD ratio are the midpoint-compression
   diagnostics: LLM Likert ratings tend to pile on the scale midpoint and avoid
   the extremes, which suppresses correlation without being a correlation
   problem per se (a monotone rescaling recovers it).

6. Bootstrap -- ~5000 resamples of the ARTICLES (each cluster carrying its two
   human ratings and every model's rating), all statistics recomputed per
   replicate by the very same function that produces the point estimates, so
   percentile CIs are consistent with them by construction.  Because the draw is
   shared, the CI on alpha(H1,H2) - alpha(H,LLM) is a PAIRED interval: article
   difficulty cancels within the replicate, making it tighter than anything
   obtainable from the two marginal CIs.

7. Several models -- per-model correlations in one table, inter-LLM ordinal alpha
   next to the mean human-LLM alpha, and the model x model correlation matrix.
   If the models agree with each other far more than with humans that is shared
   model bias, and it is stated rather than left for a reviewer to find.

Outputs (into the analysis dir agreement.py already creates):
    reliability_human.csv     -- rows 1-2 and the human side of 5
    llm_vs_consensus.csv      -- rows 3-5 per model, with paired diffs from 4
    inter_llm.csv             -- row 7 (alpha) + inter_llm_corr.csv (matrix)
    validation_table.csv      -- the paper table (HH | H-LLM | Ceiling), and
    validation_table_all_models.csv -- the same with one column per model

Run as part of agreement.py, or standalone once that has produced the tidy CSV:
    python survey/result_analysis/reliability.py
    python survey/result_analysis/reliability.py --selftest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
SURVEY = HERE.parent
sys.path.insert(0, str(SURVEY))

from agreement import panel_median  # noqa: E402

LEVELS = np.array([-2, -1, 0, 1, 2])       # the neutrality scale
L = len(LEVELS)
DELTA2_INTERVAL = (LEVELS[:, None] - LEVELS[None, :]).astype(float) ** 2

BOOT_B = 5000            # article resamples for the percentile CIs
BOOT_SEED = 20260729

PRIMARY_MODEL = "shown_llm"   # the model the prose paragraph is written about
RATERS_PER_ARTICLE = 2        # the matrix analysed (ICC, substitution, raw HH)


# --------------------------------------------------------------------------
# Krippendorff's alpha, coincidence-matrix form
# --------------------------------------------------------------------------
def unit_coincidences(units: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """Per-unit contributions to the coincidence matrix, as a (U, L*L) array.

    Row u is (outer(c, c) - diag(c)) / (m_u - 1) flattened, where c is unit u's
    count of ratings per scale level and m_u its number of ratings; NaN ratings
    are dropped and units left with < 2 ratings contribute a zero row (they are
    unpairable, which is exactly how alpha treats them).

    Keeping the per-unit pieces separate is what makes the cluster bootstrap
    cheap: a resample's coincidence matrix is just ``counts @ mats``.
    """
    mats = np.zeros((len(units), L, L))
    for u, g in enumerate(units):
        g = np.asarray(g, float)
        g = g[~np.isnan(g)]
        if len(g) < 2:
            continue
        c = np.bincount(np.searchsorted(LEVELS, g.astype(int)),
                        minlength=L).astype(float)
        mats[u] = (np.outer(c, c) - np.diag(c)) / (len(g) - 1)
    return mats.reshape(len(units), L * L)


def delta2_ordinal(n_c: np.ndarray) -> np.ndarray:
    """Krippendorff's ORDINAL difference function, squared.

    delta(c, k) = sum_{g=c..k} n_g - (n_c + n_k)/2  for c <= k, symmetric,
    where n_g are the coincidence-matrix marginals -- i.e. rank distance in
    OBSERVED-frequency units, so how far apart two categories are depends on how
    much of the sample lies between them.  This is why it must be recomputed
    inside every bootstrap replicate rather than fixed once.
    """
    cum = np.cumsum(n_c)
    pre = cum - n_c                                   # exclusive prefix sums
    m = cum[None, :] - pre[:, None] - (n_c[:, None] + n_c[None, :]) / 2
    up = np.triu(m)                                   # valid where c <= k
    return (up + up.T) ** 2                           # diagonal is 0 by design


def alpha_from_coincidence(o: np.ndarray, ordinal: bool = True) -> float:
    """alpha = 1 - D_o/D_e from a coincidence matrix ``o`` (L x L)."""
    n_c = o.sum(1)
    n = n_c.sum()
    if n < 2:
        return np.nan
    d2 = delta2_ordinal(n_c) if ordinal else DELTA2_INTERVAL
    d_o = float((o * d2).sum()) / n
    d_e = float((np.outer(n_c, n_c) * d2).sum()) / (n * (n - 1))
    if d_e == 0:
        return np.nan
    return 1 - d_o / d_e


def krippendorff_alpha(units: list[np.ndarray], ordinal: bool = True) -> float:
    """Krippendorff's alpha over units (articles) with any number of raters."""
    return alpha_from_coincidence(
        unit_coincidences(units).sum(0).reshape(L, L), ordinal)


# --------------------------------------------------------------------------
# ICC(1,1) / ICC(1,k) and the attenuation ceiling
# --------------------------------------------------------------------------
def icc_oneway(mat: np.ndarray) -> tuple[float, float, float]:
    """One-way random-effects ICC on a balanced (n_units x k_raters) matrix.

    Returns (ICC(1,1), ICC(1,k), ceiling), where ICC(1,1) is the reliability of
    a SINGLE rating, ICC(1,k) = k*ICC11 / (1 + (k-1)*ICC11) the Spearman-Brown
    reliability of the k-rater MEAN (2*ICC11/(1+ICC11) at k = 2), and ceiling =
    sqrt(ICC(1,k)) the largest correlation any measure can have with that mean.

    ICC(1,*) is the right family here: raters are not crossed with articles
    (each response comes from a different Prolific participant), so there is no
    rater main effect to remove -- rater differences stay in the error term.
    """
    n, k = mat.shape
    if n < 2 or k < 2:
        return (np.nan, np.nan, np.nan)
    row = mat.mean(1)
    msb = k * float(((row - mat.mean()) ** 2).sum()) / (n - 1)      # between articles
    msw = float(((mat - row[:, None]) ** 2).sum()) / (n * (k - 1))  # within articles
    denom = msb + (k - 1) * msw
    if denom <= 0:
        return (np.nan, np.nan, np.nan)
    icc11 = (msb - msw) / denom
    icc1k = k * icc11 / (1 + (k - 1) * icc11) if (1 + (k - 1) * icc11) != 0 else np.nan
    return (icc11, icc1k, float(np.sqrt(icc1k)) if icc1k > 0 else 0.0)


# --------------------------------------------------------------------------
# correlations and raw agreement
# --------------------------------------------------------------------------
def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho as Pearson on tie-averaged ranks (identical, much faster)."""
    return _pearson(stats.rankdata(a), stats.rankdata(b))


def _kendall(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall tau_b -- the tie-corrected variant, which is what a 5-point scale
    with two raters needs."""
    if len(a) < 3:
        return np.nan
    return float(stats.kendalltau(a, b, variant="b").statistic)


def pair_agreement(a: np.ndarray, b: np.ndarray, signed: bool = True) -> dict:
    """Exact %, within-1 %, MAE and mean signed difference b - a over rating pairs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    out = {"exact_pct": 100 * float(np.mean(d == 0)),
           "within1_pct": 100 * float(np.mean(np.abs(d) <= 1)),
           "mae": float(np.mean(np.abs(d)))}
    # For human-human the labelling of the two raters is arbitrary, so the mean
    # signed difference is 0 by symmetry and carries no information; only the
    # human->model direction is reported.
    out["mean_signed_diff"] = float(np.mean(d)) if signed else 0.0
    return out


# --------------------------------------------------------------------------
# the statistic bundle (point estimate and every bootstrap replicate)
# --------------------------------------------------------------------------
class Sample:
    """Article-level arrays + the precomputed per-article alpha pieces.

    One instance is built from the responses; ``stats_for(idx)`` then evaluates
    the whole battery on any (re)sample of article indices.  Point estimates use
    idx = 0..U-1, bootstrap replicates use a draw with replacement -- same code
    path, so a CI can never be computed from a different definition than the
    estimate it brackets.
    """

    def __init__(self, H: np.ndarray, models: dict[str, np.ndarray],
                 llm_only: list[str]):
        self.H = H                      # (U, 2) int human ratings
        self.models = models            # name -> (U,) float ratings, NaN allowed
        self.llm_only = llm_only        # real models (excl. shown_llm/llm_median)
        self.U = len(H)
        self.hmean = H.mean(1)
        # alpha pieces: human-human, model-vs-each-human, and inter-LLM
        self.A_hh = unit_coincidences(list(H))
        self.A_sub = {m: [unit_coincidences([np.array([h, v]) for h, v in zip(H[:, j], val)])
                          for j in range(H.shape[1])]
                      for m, val in models.items()}
        self.A_llm = unit_coincidences([np.array([models[m][u] for m in llm_only])
                                        for u in range(self.U)])
        self.ok = {m: ~np.isnan(v) for m, v in models.items()}

    def stats_for(self, idx: np.ndarray) -> dict:
        H, w = self.H[idx], np.bincount(idx, minlength=self.U).astype(float)
        d = {}
        # ---- 1. human reliability
        o_hh = (w @ self.A_hh).reshape(L, L)
        d["alpha_ord_hh"] = alpha_from_coincidence(o_hh, ordinal=True)
        d["alpha_int_hh"] = alpha_from_coincidence(o_hh, ordinal=False)
        icc11, icc12, ceiling = icc_oneway(H.astype(float))
        # ---- 2. the ceiling
        d.update({"icc11": icc11, "icc12": icc12, "ceiling_r": ceiling,
                  "alpha_icc_gap": d["alpha_ord_hh"] - icc11})
        # ---- 5. raw agreement, human side
        d.update({f"hh_{k}": v for k, v in
                  pair_agreement(H[:, 0], H[:, 1], signed=False).items()})
        # sd_human is over all INDIVIDUAL ratings, which is the like-for-like
        # comparison for a model's article-level SD (both are then the marginal
        # SD of one rating of one article). sd_human_mean -- the SD of the 2-rater
        # consensus, with half the rater noise -- is the stricter benchmark.
        d["sd_human"] = float(np.std(H.ravel().astype(float), ddof=1))
        d["sd_human_mean"] = float(np.std(H.mean(1), ddof=1))
        # ---- 7. inter-LLM reliability, and the human-LLM alphas to compare it to
        d["alpha_ord_llm_llm"] = alpha_from_coincidence(
            (w @ self.A_llm).reshape(L, L), ordinal=True)

        for m, v in self.models.items():
            vi, ok = v[idx], self.ok[m][idx]
            if ok.sum() < 3:
                continue
            hm, hs, vv = self.hmean[idx][ok], H[ok], vi[ok]
            # ---- 3. against the human consensus (the 2-rater mean)
            r = _pearson(vv, hm)
            d[f"{m}|pearson_r"] = r
            d[f"{m}|spearman_rho"] = _spearman(vv, hm)
            d[f"{m}|kendall_tau_b"] = _kendall(vv, hm)
            d[f"{m}|r_over_ceiling"] = r / ceiling if ceiling > 0 else np.nan
            # ---- 4. substitution: alpha(model, each human), averaged, vs
            # alpha(H1, H2) on the SAME articles (this model's non-NaN ones), so
            # the difference below is a like-for-like paired contrast.
            wm = w * self.ok[m]
            subs = [alpha_from_coincidence((wm @ A).reshape(L, L), ordinal=True)
                    for A in self.A_sub[m]]
            a_sub = float(np.mean(subs))
            a_hh_m = alpha_from_coincidence((wm @ self.A_hh).reshape(L, L), ordinal=True)
            d[f"{m}|alpha_ord_sub"] = a_sub
            d[f"{m}|alpha_ord_hh_matched"] = a_hh_m
            d[f"{m}|alpha_hh_minus_sub"] = a_hh_m - a_sub
            # ---- 5. raw agreement, model side: every individual human rating
            # against the model (the level HH's pairs live on), plus the
            # consensus-level errors and the SD comparison.
            flat_h = hs.ravel().astype(float)
            flat_v = np.repeat(vv, hs.shape[1])
            d.update({f"{m}|hl_{k}": val for k, val in
                      pair_agreement(flat_h, flat_v).items()})
            # |model - consensus| is not the mean of |model - each rater|; the
            # SIGNED version is (by linearity), so it is not repeated here.
            d[f"{m}|hl_mae_vs_mean"] = float(np.mean(np.abs(vv - hm)))
            d[f"{m}|sd_llm"] = float(np.std(vv, ddof=1))
            d[f"{m}|sd_ratio_llm_over_human"] = (
                d[f"{m}|sd_llm"] / d["sd_human"] if d["sd_human"] > 0 else np.nan)
        return d


def bootstrap(sample: Sample, B: int = BOOT_B, seed: int = BOOT_SEED
              ) -> tuple[dict, pd.DataFrame]:
    """Point estimates + the B x statistics replicate table (articles resampled
    with replacement, each carrying its two human ratings and all model ratings).

    Every statistic is recomputed from the same draw, so any difference of two
    of them (notably alpha_hh - alpha_sub) has its article-sampling variance
    cancelled within the replicate.
    """
    point = sample.stats_for(np.arange(sample.U))
    rng = np.random.default_rng(seed)
    reps = [sample.stats_for(rng.choice(sample.U, sample.U)) for _ in range(B)]
    return point, pd.DataFrame(reps)


def _ci(reps: pd.DataFrame, key: str) -> tuple[float, float]:
    if key not in reps.columns:
        return (np.nan, np.nan)
    x = reps[key].to_numpy(float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))


def _r3(x) -> float:
    return np.nan if x is None or not np.isfinite(x) else round(float(x), 3)


# --------------------------------------------------------------------------
# assembly / reporting
# --------------------------------------------------------------------------
def build_sample(resp: pd.DataFrame, votes: pd.DataFrame) -> Sample:
    """The n_articles x RATERS_PER_ARTICLE human matrix plus one rating vector
    per model, aligned on article (taskid).

    Restricted to articles with EXACTLY RATERS_PER_ARTICLE usable human ratings:
    ICC, the substitution alphas and the raw HH numbers all need a fixed rater
    count to be comparable to each other and to the ceiling arithmetic.  Any
    off-target article is reported (agreement.py's ``rating_coverage`` lists
    them) and alpha over ALL articles with >= 2 ratings is reported separately in
    reliability_human.csv as ``alpha_ord_hh_all_articles``.
    """
    counts = resp.groupby("taskid")["human_rating_int"].count()
    tids = counts[counts == RATERS_PER_ARTICLE].index.to_numpy()
    if len(tids) < 3:
        raise ValueError(f"only {len(tids)} articles with exactly "
                         f"{RATERS_PER_ARTICLE} ratings -- nothing to analyse")
    H = np.array([resp.loc[resp["taskid"] == t, "human_rating_int"].to_numpy(int)
                  for t in tids])
    llm_only = list(votes.columns)
    models = {m: votes[m].reindex(tids).to_numpy(float) for m in llm_only}
    # the rating each respondent actually saw (constant within an article) and
    # the across-model median, for continuity with agreement_by_model.csv
    models["shown_llm"] = (resp.groupby("taskid")["llm_rating_int"].first()
                           .reindex(tids).to_numpy(float))
    models["llm_median"] = panel_median(votes).reindex(tids).to_numpy(float)
    return Sample(H, models, llm_only)


HUMAN_COLS = ["alpha_ord_hh", "alpha_int_hh", "icc11", "icc12", "ceiling_r",
              "alpha_icc_gap", "hh_exact_pct", "hh_within1_pct", "hh_mae",
              "sd_human", "sd_human_mean"]

MODEL_COLS = ["pearson_r", "spearman_rho", "kendall_tau_b", "r_over_ceiling",
              "alpha_ord_sub", "alpha_ord_hh_matched", "alpha_hh_minus_sub",
              "hl_exact_pct", "hl_within1_pct", "hl_mae", "hl_mean_signed_diff",
              "hl_mae_vs_mean", "sd_llm", "sd_ratio_llm_over_human"]


def _with_ci(point: dict, reps: pd.DataFrame, keys: list[str],
             prefix: str = "") -> dict:
    """{stat, stat_lo, stat_hi} for each key, stripping the model prefix."""
    row = {}
    for k in keys:
        full = f"{prefix}{k}" if prefix else k
        lo, hi = _ci(reps, full)
        row[k] = _r3(point.get(full))
        row[f"{k}_lo"], row[f"{k}_hi"] = _r3(lo), _r3(hi)
    return row


def human_reliability_table(sample: Sample, resp: pd.DataFrame,
                            point: dict, reps: pd.DataFrame) -> pd.DataFrame:
    counts = resp.groupby("taskid")["human_rating_int"].count()
    multi = [resp.loc[resp["taskid"] == t, "human_rating_int"].to_numpy(int)
             for t in counts[counts >= 2].index]
    row = {"n_articles": sample.U,
           "n_ratings": int(sample.U * RATERS_PER_ARTICLE),
           "raters_per_article": RATERS_PER_ARTICLE,
           **_with_ci(point, reps, HUMAN_COLS),
           "n_articles_ge2": len(multi),
           "alpha_ord_hh_all_articles": _r3(krippendorff_alpha(multi, ordinal=True)),
           "alpha_int_hh_all_articles": _r3(krippendorff_alpha(multi, ordinal=False)),
           "boot_B": BOOT_B}
    return pd.DataFrame([row])


def llm_table(sample: Sample, point: dict, reps: pd.DataFrame) -> pd.DataFrame:
    """Per model: correlations against the human mean, the achieved share of the
    ceiling, the substitution alphas and the raw/compression numbers.

    ``r_over_ceiling_{lo,hi}`` is the JOINT interval -- r and the ceiling
    re-estimated together in each replicate -- and is right-skewed and wide,
    because a replicate whose human matrix happens to look unreliable puts a
    small number in the denominator.  ``r_over_ceiling_fixed_{lo,hi}`` rescales
    r's own interval by the point-estimate ceiling instead: it isolates the
    model's sampling variability at a ceiling taken as known.  Quote the fixed
    one for "x% of the achievable signal" and the joint one when the claim has
    to carry the ceiling's uncertainty too.
    """
    rows = []
    ceiling = point["ceiling_r"]
    for m in sample.models:
        if f"{m}|pearson_r" not in point:
            continue
        row = {"model": m,
               "n_articles": int(sample.ok[m].sum()),
               "ceiling_r": _r3(ceiling),
               **_with_ci(point, reps, MODEL_COLS, prefix=f"{m}|")}
        r_lo, r_hi = _ci(reps, f"{m}|pearson_r")
        row["r_over_ceiling_fixed_lo"] = _r3(r_lo / ceiling) if ceiling > 0 else np.nan
        row["r_over_ceiling_fixed_hi"] = _r3(r_hi / ceiling) if ceiling > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def inter_llm_table(sample: Sample, point: dict, reps: pd.DataFrame
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inter-LLM ordinal alpha vs the mean human-LLM alpha (shared model bias
    shows up as the former being much larger), plus the model x model Pearson
    matrix on the article-level ratings."""
    subs = [point[f"{m}|alpha_ord_sub"] for m in sample.llm_only
            if f"{m}|alpha_ord_sub" in point]
    mean_hl = float(np.mean(subs)) if subs else np.nan
    lo, hi = _ci(reps, "alpha_ord_llm_llm")
    row = {"n_models": len(sample.llm_only),
           "models": " ".join(sample.llm_only),
           "alpha_ord_llm_llm": _r3(point["alpha_ord_llm_llm"]),
           "alpha_ord_llm_llm_lo": _r3(lo), "alpha_ord_llm_llm_hi": _r3(hi),
           "alpha_ord_hh": _r3(point["alpha_ord_hh"]),
           "mean_alpha_ord_human_llm": _r3(mean_hl),
           "llm_llm_minus_human_llm": _r3(point["alpha_ord_llm_llm"] - mean_hl)}
    mat = pd.DataFrame({m: sample.models[m] for m in sample.llm_only}).corr().round(3)
    return pd.DataFrame([row]), mat.reset_index(names="model")


# (label, HH key, H-LLM key, Ceiling key, decimals)
PAPER_ROWS = [
    ("Ordinal alpha [95% CI]", "alpha_ord_hh", "alpha_ord_sub", None, 2),
    ("Pearson r w/ consensus", None, "pearson_r", "ceiling_r", 2),
    ("Spearman rho w/ consensus", None, "spearman_rho", None, 2),
    ("Kendall tau_b w/ consensus", None, "kendall_tau_b", None, 2),
    ("Exact %", "hh_exact_pct", "hl_exact_pct", None, 1),
    ("Within-1 %", "hh_within1_pct", "hl_within1_pct", None, 1),
    ("MAE", "hh_mae", "hl_mae", None, 2),
    ("Mean signed diff. (LLM - human)", None, "hl_mean_signed_diff", None, 2),
    ("SD of ratings", "sd_human", "sd_llm", None, 2),
]


def paper_table(point: dict, reps: pd.DataFrame,
                models: str | list[str]) -> pd.DataFrame:
    """The paper's validation table: HH | H-LLM (one column per model) | Ceiling,
    every cell "value [95% CI]".

    The HH alpha and the H-LLM alpha are the substitution pair from step 4 (the
    latter is alpha(model, human) averaged over the two humans), so that row's
    columns answer "how interchangeable are two humans?" and "how
    interchangeable is the model with a human?" on one metric.
    """
    models = [models] if isinstance(models, str) else models

    def cell(key: str | None, dec: int, prefix: str = "") -> str:
        if key is None:
            return "--"
        full = f"{prefix}{key}"
        v = point.get(full)
        if v is None or not np.isfinite(v):
            return "--"
        lo, hi = _ci(reps, full)
        if np.isfinite(lo) and np.isfinite(hi):
            return f"{v:.{dec}f} [{lo:.{dec}f}, {hi:.{dec}f}]"
        return f"{v:.{dec}f}"

    rows = []
    for label, hh, hl, ceil_, dec in PAPER_ROWS:
        row = {"metric": label, "HH": cell(hh, dec)}
        for m in models:
            col = "H-LLM" if len(models) == 1 else m
            row[col] = cell(hl, dec, prefix=f"{m}|")
        row["Ceiling"] = cell(ceil_, dec)
        rows.append(row)
    return pd.DataFrame(rows)


def _md(tbl: pd.DataFrame) -> str:
    head = "| " + " | ".join(tbl.columns) + " |"
    rule = "|" + "|".join("---" for _ in tbl.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in r) + " |"
            for r in tbl.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def prose(point: dict, reps: pd.DataFrame, model: str) -> str:
    """The two or three sentences the validation subsection needs: the human
    alpha, the correlation against its ceiling, the substitution contrast and the
    direction of the bias.  A draft to edit, not a finished paragraph -- but
    every number in it is one of the computed statistics, so it cannot drift from
    the tables."""
    a_lo, a_hi = _ci(reps, "alpha_ord_hh")
    r, ceiling = point.get(f"{model}|pearson_r", np.nan), point["ceiling_r"]
    r_lo, r_hi = _ci(reps, f"{model}|pearson_r")     # fixed-ceiling ratio interval
    sd_ratio = point.get(f"{model}|sd_ratio_llm_over_human", np.nan)
    signed = point.get(f"{model}|hl_mean_signed_diff", np.nan)
    d = point.get(f"{model}|alpha_hh_minus_sub", np.nan)
    d_lo, d_hi = _ci(reps, f"{model}|alpha_hh_minus_sub")
    direction = "more in-favor (less neutral)" if signed > 0 else "more biased-against"
    spread = ("compressed toward the scale midpoint (the usual LLM-Likert "
              "failure mode)" if sd_ratio < 0.95 else
              "wider than the human ratings" if sd_ratio > 1.05 else
              "spread like the human ratings")
    # A negative alpha_hh_minus_sub means the model is MORE interchangeable with
    # a human rater than a second human is -- worth saying in those words.
    sub = (f"substituting it for one annotator does not lose reliability: it "
           f"gains {-d:.2f} alpha [{-d_hi:.2f}, {-d_lo:.2f}]" if d < 0 else
           f"substituting it for one annotator costs {d:.2f} alpha "
           f"[{d_lo:.2f}, {d_hi:.2f}]")
    over = (" -- above the estimated ceiling, which on 70 articles means the "
            "ceiling itself is uncertain rather than that the model beat it"
            if r > ceiling else "")
    return (
        f"Two human raters per article reach ordinal Krippendorff's alpha = "
        f"{point['alpha_ord_hh']:.2f} [95% CI {a_lo:.2f}, {a_hi:.2f}] "
        f"(ICC(1,1) = {point['icc11']:.2f}), so their mean has reliability "
        f"ICC(1,2) = {point['icc12']:.2f} and no measure whatsoever can correlate "
        f"with it above {ceiling:.2f}. {model} attains r = {r:.2f} against that "
        f"mean, i.e. {100 * r / ceiling:.0f}% of the achievable signal "
        f"[{100 * r_lo / ceiling:.0f}%, {100 * r_hi / ceiling:.0f}%, ceiling "
        f"held fixed]{over}; {sub}. Its ratings are {spread} (SD ratio "
        f"{sd_ratio:.2f}) and run {abs(signed):.2f} scale points {direction} "
        f"than the humans'.")


def reliability_report(resp: pd.DataFrame, votes: pd.DataFrame, out_dir: Path,
                       B: int = BOOT_B) -> None:
    """Compute everything and write the four CSVs into ``out_dir``."""
    sample = build_sample(resp, votes)
    print(f"\nReliability / ceiling analysis: {sample.U} articles x "
          f"{RATERS_PER_ARTICLE} human ratings, {len(sample.models)} rating "
          f"columns, {B} article resamples")
    point, reps = bootstrap(sample, B=B)

    hum = human_reliability_table(sample, resp, point, reps)
    hum.to_csv(out_dir / "reliability_human.csv", index=False)
    print(f"\nWrote {out_dir / 'reliability_human.csv'}  "
          f"(human reliability + the attenuation ceiling)\n")
    print(hum[["n_articles", "alpha_ord_hh", "alpha_ord_hh_lo", "alpha_ord_hh_hi",
               "alpha_int_hh", "icc11", "icc11_lo", "icc11_hi", "icc12",
               "ceiling_r", "alpha_icc_gap"]].to_string(index=False))
    gap = float(hum["alpha_icc_gap"].iloc[0])
    if abs(gap) > 0.05:
        print(f"  NOTE: ordinal alpha and ICC(1,1) differ by {gap:+.3f}; the "
              f"ordinal metric's frequency-based distances are pulling away "
              f"from the interval/variance view -- worth a look before quoting "
              f"the two interchangeably.")

    llm = llm_table(sample, point, reps)
    llm.to_csv(out_dir / "llm_vs_consensus.csv", index=False)
    print(f"\nWrote {out_dir / 'llm_vs_consensus.csv'}  "
          f"(per model vs the 2-rater human mean; paired alpha contrast)\n")
    print(llm[["model", "n_articles", "pearson_r", "pearson_r_lo", "pearson_r_hi",
               "spearman_rho", "kendall_tau_b", "ceiling_r", "r_over_ceiling",
               "r_over_ceiling_fixed_lo", "r_over_ceiling_fixed_hi",
               "r_over_ceiling_lo", "r_over_ceiling_hi"]].to_string(index=False))
    print("  (r_over_ceiling = share of the achievable signal; *_fixed_* CI holds "
          "the ceiling at its point estimate,\n   the plain *_lo/_hi CI "
          "re-estimates the ceiling in every replicate and is wide and "
          "right-skewed by construction)")
    print("\nSubstitution check -- ordinal alpha(model, each human) averaged over "
          "the two humans, against alpha(H1, H2) on the same articles\n"
          "(alpha_hh_minus_sub > 0 = a second human is more interchangeable "
          "with a human than the model is; CI is paired):\n")
    print(llm[["model", "alpha_ord_sub", "alpha_ord_sub_lo", "alpha_ord_sub_hi",
               "alpha_ord_hh_matched", "alpha_hh_minus_sub",
               "alpha_hh_minus_sub_lo", "alpha_hh_minus_sub_hi"]].to_string(index=False))
    print("\nRaw agreement and midpoint compression (HH: exact "
          f"{point['hh_exact_pct']:.1f}%, within-1 {point['hh_within1_pct']:.1f}%, "
          f"MAE {point['hh_mae']:.3f}, SD {point['sd_human']:.3f}):\n")
    print(llm[["model", "hl_exact_pct", "hl_within1_pct", "hl_mae",
               "hl_mean_signed_diff", "hl_mean_signed_diff_lo",
               "hl_mean_signed_diff_hi", "hl_mae_vs_mean", "sd_llm",
               "sd_ratio_llm_over_human"]].to_string(index=False))
    print("  (hl_mean_signed_diff = mean(model - human): positive = the model "
          "rates articles MORE in-favor;\n   sd_ratio < 1 = midpoint "
          "compression, which suppresses correlation and is fixable by a "
          "monotone rescaling)")

    inter, corr = inter_llm_table(sample, point, reps)
    inter.to_csv(out_dir / "inter_llm.csv", index=False)
    corr.to_csv(out_dir / "inter_llm_corr.csv", index=False)
    print(f"\nWrote {out_dir / 'inter_llm.csv'} and "
          f"{out_dir / 'inter_llm_corr.csv'}\n")
    print(inter.to_string(index=False))
    print("\nModel x model Pearson r (article-level ratings):")
    print(corr.to_string(index=False))
    d_llm = float(inter["llm_llm_minus_human_llm"].iloc[0])
    if np.isfinite(d_llm) and d_llm > 0.1:
        print(f"  NOTE: the models agree with EACH OTHER {d_llm:+.3f} alpha more "
              f"than with humans -- shared model bias; report it explicitly.")

    tbl = paper_table(point, reps, PRIMARY_MODEL)
    tbl.to_csv(out_dir / "validation_table.csv", index=False)
    wide = paper_table(point, reps, list(sample.models))
    wide.to_csv(out_dir / "validation_table_all_models.csv", index=False)
    print(f"\nWrote {out_dir / 'validation_table.csv'}  "
          f"(paper table, model = {PRIMARY_MODEL}) and "
          f"{out_dir / 'validation_table_all_models.csv'} (one column per model)\n")
    print(_md(tbl))
    print("\n" + prose(point, reps, PRIMARY_MODEL) + "\n")


# --------------------------------------------------------------------------
# self-tests: the estimators, against independent implementations
# --------------------------------------------------------------------------
def _alpha_bruteforce(units: list[np.ndarray], ordinal: bool) -> float:
    """Definition-following alpha (explicit pairwise sums, no matrix algebra) --
    an independent check on the coincidence-matrix implementation."""
    pooled = np.concatenate([np.asarray(u, float) for u in units])
    n = len(pooled)
    counts = np.bincount(np.searchsorted(LEVELS, pooled.astype(int)), minlength=L)

    def delta2(a: float, b: float) -> float:
        if not ordinal:
            return (a - b) ** 2
        i, j = sorted((int(np.searchsorted(LEVELS, a)), int(np.searchsorted(LEVELS, b))))
        return (counts[i:j + 1].sum() - (counts[i] + counts[j]) / 2) ** 2

    d_o = sum(sum(delta2(g[i], g[j]) for i in range(len(g)) for j in range(len(g)) if i != j)
              / (len(g) - 1) for g in units) / n
    d_e = sum(delta2(a, b) for i, a in enumerate(pooled) for j, b in enumerate(pooled)
              if i != j) / (n * (n - 1))
    return 1 - d_o / d_e


def _selftest() -> None:
    rng = np.random.default_rng(7)
    units = [rng.choice(LEVELS, size=rng.integers(2, 5)) for _ in range(40)]
    for ordinal in (True, False):
        got, want = krippendorff_alpha(units, ordinal), _alpha_bruteforce(units, ordinal)
        assert abs(got - want) < 1e-9, (ordinal, got, want)
        print(f"alpha (ordinal={ordinal}) matches brute force: {got:.6f}")

    # interval alpha must equal agreement.py's independent implementation
    from agreement import krippendorff_alpha as agr_alpha
    got = krippendorff_alpha(units, ordinal=False)
    want = agr_alpha(units, lambda x, y: (x - y) ** 2)
    assert abs(got - want) < 1e-9, (got, want)
    print(f"interval alpha matches agreement.py: {got:.6f}")

    # a perfectly reliable and a pure-noise matrix
    perfect = np.tile(rng.choice(LEVELS, 50)[:, None], (1, 2))
    assert abs(icc_oneway(perfect.astype(float))[0] - 1.0) < 1e-12
    assert abs(krippendorff_alpha(list(perfect), ordinal=True) - 1.0) < 1e-12
    print("perfect agreement -> ICC(1,1) = alpha = 1")

    # the worked example: ICC(1,1) = 0.35 -> ICC(1,2) = 0.52 -> ceiling = 0.72.
    # First the arithmetic itself, exactly (2i/(1+i) and its square root)...
    i = 0.35
    assert round(2 * i / (1 + i), 2) == 0.52
    assert round(np.sqrt(2 * i / (1 + i)), 2) == 0.72
    # ...then that icc_oneway recovers a matrix built to have that ICC.
    icc_target = 0.35
    t = rng.normal(0, np.sqrt(icc_target), 20000)
    mat = t[:, None] + rng.normal(0, np.sqrt(1 - icc_target), (20000, 2))
    i11, i12, ceiling = icc_oneway(mat)
    assert abs(i11 - icc_target) < 0.02, i11
    assert abs(i12 - 2 * i11 / (1 + i11)) < 1e-12
    assert abs(np.sqrt(i12) - ceiling) < 1e-12
    print(f"worked example: ICC(1,1)={i11:.2f} -> ICC(1,2)={i12:.2f} -> "
          f"ceiling={ceiling:.2f}  (expected 0.35 / 0.52 / 0.72)")

    # correlations against scipy
    a, b = rng.integers(-2, 3, 60).astype(float), rng.integers(-2, 3, 60).astype(float)
    assert abs(_spearman(a, b) - stats.spearmanr(a, b).statistic) < 1e-12
    assert abs(_pearson(a, b) - stats.pearsonr(a, b).statistic) < 1e-12
    print("Pearson / Spearman match scipy (ties included)")

    # a known-signal end-to-end run: the ratio must sit inside [0, ~1.05] and
    # the paired contrast must reproduce the difference of the point estimates
    U = 70
    truth = rng.normal(0, 1, U)
    H = np.clip(np.rint(truth[:, None] + rng.normal(0, 1.0, (U, 2))), -2, 2)
    v = np.clip(np.rint(truth + rng.normal(0, 0.8, U)), -2, 2)
    s = Sample(H.astype(int), {"m": v}, ["m"])
    p, r = bootstrap(s, B=200, seed=1)
    assert abs(p["m|alpha_hh_minus_sub"]
               - (p["m|alpha_ord_hh_matched"] - p["m|alpha_ord_sub"])) < 1e-12
    assert 0 < p["m|r_over_ceiling"] < 1.2, p["m|r_over_ceiling"]
    lo, hi = _ci(r, "m|alpha_hh_minus_sub")
    print(f"end-to-end: alpha_hh={p['alpha_ord_hh']:.3f} icc11={p['icc11']:.3f} "
          f"ceiling={p['ceiling_r']:.3f} r={p['m|pearson_r']:.3f} "
          f"ratio={p['m|r_over_ceiling']:.3f} "
          f"alpha_hh-sub={p['m|alpha_hh_minus_sub']:.3f} [{lo:.3f}, {hi:.3f}]")
    print("\nAll self-tests passed.")


def main() -> None:
    """Standalone entry point: reuse agreement.py's loading + APPROVED gate."""
    if "--selftest" in sys.argv:
        _selftest()
        return
    import agreement as agr
    responses_file = HERE / "survey_results" / agr.RESPONSES_FILE
    tasks = pd.read_csv(SURVEY / "tasks.csv")
    out_dir = HERE / "analyses_results" / f"{responses_file.stem}_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    resp = agr.load_responses(responses_file, tasks, out_dir)
    resp = resp.dropna(subset=["human_rating_int", "taskid"])
    from demographics import included_pids
    inc = included_pids()
    if inc is not None:
        resp = resp[resp["prolific_pid"].isin(inc)].copy()
        print(f"Prolific status gate: kept {len(resp)} responses (APPROVED only)")
    reliability_report(resp, agr.llm_votes(tasks), out_dir)


if __name__ == "__main__":
    main()
