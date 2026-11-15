"""Human-LLM agreement metrics for the validation survey.

Reads the tidy responses (export_responses.py) and the per-article ratings of
ALL four LLMs (results/final2.0, joined by article name+source via tasks.csv),
then writes agreement_by_model.csv with, per model:

  * qwk            -- quadratic-weighted Cohen's kappa, human rating vs that
                      model's rating of the same article (one pair per human
                      rating; articles with several raters contribute several)
  * exact_pct      -- share of ratings identical on the 5-point scale
  * within1_pct    -- share agreeing to within one scale point (|human - model|
                      <= 1), i.e. exact matches plus off-by-one pairs
  * n              -- number of human ratings paired
  * rating_sd      -- spread of that judge's OWN ratings over the validation
                      articles (see ``rating_sd``); a compressed judge can post a
                      decent within1_pct simply by never leaving the midpoint

Rows: each LLM, the rating actually SHOWN in the survey ("shown_llm"), and the
4-model median ("llm_median").

Each agreement metric also has a ``_w`` twin: post-stratification-weighted on
the disagreement bin (corpus share / sample share -- see ``task_weights``), so
unweighted columns describe the difficulty-enriched sample while ``_w`` columns
project to the length-eligible corpus. ``n_eff`` is the Kish effective sample
size under those weights.

Only APPROVED Prolific submissions are analysed: the demographic export (see
demographics.py) is read up front and any response whose participant is not
APPROVED (demographics.INCLUDED_STATUSES), or is on the manual
demographics.BLACKLIST_PIDS, is dropped from every metric. If no export is
present the gate is skipped (a warning is printed) and all responses are used.

No parameters: drop the CSV downloaded from the LimeSurvey admin UI into
survey/result_analysis/survey_results/, put its FILENAME in RESPONSES_FILE
below, and run

    python survey/result_analysis/agreement.py

Also writes human_irr.csv: inter-rater agreement between human annotators on
articles rated by >=2 participants (pairwise exact/within-1 agreement, mean
absolute difference, QWK over pairs, Krippendorff's alpha, interval metric).

Writes articles_off_target_ratings.csv: study articles whose number of
(APPROVED) human ratings is not exactly RATINGS_TARGET (2) -- i.e. under- or
over-covered articles, including any fielded but not yet rated.

Also links the Prolific demographic export (demographics.py) to the responses
and writes participant descriptives + exploratory demographic x outcome tables
under a demographics/ subdirectory.

Writes ceiling_diff.csv (and merges it into agreement_vs_ceiling.csv): the
cluster bootstrap of kappa_model - kappa_human_human with BOTH terms recomputed
on the same resampled articles, at two matched aggregation levels, plus a
chance-correction-free robustness column. See paired_ceiling_ci -- this is the
interval the "model meets the human ceiling" claim should rest on, because
comparing two independently-bootstrapped CIs double-counts the shared
article-sampling variance.

Also runs reliability.py's battery, which reports the same validation on the
classical annotation-study footing: ordinal Krippendorff's alpha and ICC(1,1) on
the articles x 2 human matrix, the attenuation ceiling sqrt(ICC(1,2)),
Pearson/Spearman/Kendall of each model against the 2-rater human mean with the
share of the achievable signal attained, the "could the model replace one
annotator" substitution contrast, and the exact/within-1/MAE/signed-difference/SD
numbers that expose midpoint compression -- all with one shared 5000-draw article
bootstrap (reliability_human.csv, llm_vs_consensus.csv, inter_llm.csv,
inter_llm_corr.csv, validation_table.csv).

Writes agreement_by_disagreement.csv: human-LLM agreement broken out by the
sampling stratum (sd of the four judges' ratings, DEFAULT_DISAGREEMENT_BINS
equal-width bins). Judge consensus behaves as a confidence signal -- agreement
with humans is highest where the judges agree with each other -- while the
human-human gap on the same articles barely moves across bins, so the bins are
not merely tracking task difficulty. See agreement_by_disagreement.

Outputs (responses_tidy.csv, agreement_by_model.csv, human_irr.csv,
ceiling_diff.csv, agreement_by_disagreement.csv, alignment_by_politics.csv,
reliability_human.csv, llm_vs_consensus.csv, inter_llm*.csv,
validation_table.csv, demographics/*.csv) go to
    survey/result_analysis/analyses_results/<input-name>_analysis/
"""
from __future__ import annotations

import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
SURVEY = HERE.parent
sys.path.insert(0, str(SURVEY))
from build_tasks import (DEFAULT_DISAGREEMENT_BINS, DEFAULT_EXPERIMENT,  # noqa: E402
                         DEFAULT_LEADERS_PATH, DEFAULT_RESULTS_PATH,
                         LIKERT_MAPPING, MAX_WORDS, MIN_WORDS,
                         SOURCE_TO_TEXT_COL, _assign_bin, _bin_edges,
                         discover_and_load_results, load_merged_results)
from power_analysis import qwk  # noqa: E402  (quadratic-weighted kappa, -2..2)


def _to_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return LIKERT_MAPPING.get(str(x).strip())


def task_weights(tasks: pd.DataFrame) -> pd.Series:
    """Post-stratification weight per taskid, projecting the equal-allocation
    disagreement-bin sample onto the length-eligible paired corpus.

    weight = (corpus share of the article's disagreement bin) /
             (sample share of that bin), normalized to mean 1 over tasks.
    Corpus shares are recomputed from the full results the same way
    build_tasks.py builds its pool (one unit per ARTICLE, article within the
    [MIN_WORDS, MAX_WORDS] window, same pooled bin edges), so weighted
    estimates generalize to that eligible corpus -- NOT past the length window."""
    sources = sorted(tasks["source"].unique())
    df = load_merged_results(DEFAULT_RESULTS_PATH, DEFAULT_LEADERS_PATH, DEFAULT_EXPERIMENT)
    df = df[df["Source"].isin(sources)].copy()
    df["v"] = df["neutrality_assessment"].map(_to_int)
    df = df.dropna(subset=["v"])
    dis = (df.groupby(["Name", "Source"])["v"]
           .agg(lambda s: float(s.std(ddof=0))).unstack("Source"))
    lead = df.drop_duplicates("Name").set_index("Name")

    # Per-article disagreement over all length-eligible articles of both sources.
    parts = []
    for s in sources:
        d = dis[s].dropna()
        words = lead.reindex(d.index)[SOURCE_TO_TEXT_COL[s]].astype(str).str.split().str.len()
        parts.append(d[(words >= MIN_WORDS) & (words <= MAX_WORDS)])
    d = pd.concat(parts)
    edges = _bin_edges(d, DEFAULT_DISAGREEMENT_BINS)
    corpus = pd.Series(_assign_bin(d, edges)).value_counts(normalize=True)
    sample = tasks["disagreement_bin"].value_counts(normalize=True)
    w_bin = (corpus / sample).reindex(sample.index).fillna(0.0)
    w = tasks.set_index("taskid")["disagreement_bin"].map(w_bin)
    return w / w.mean()


def llm_votes(tasks: pd.DataFrame) -> pd.DataFrame:
    """taskid x model -> rating int, for every model that rated the pool's articles."""
    res = discover_and_load_results(DEFAULT_RESULTS_PATH, DEFAULT_EXPERIMENT)
    res = res.assign(v=res["neutrality_assessment"].map(_to_int)).dropna(subset=["v"])
    res["Source"] = res["Source"].replace({"grokipedia_0_2": "grokipedia_0.2"})
    merged = tasks[["taskid", "name", "source"]].merge(
        res, left_on=["name", "source"], right_on=["Name", "Source"], how="left")
    return merged.pivot_table(index="taskid", columns="Model", values="v", aggfunc="first")


def panel_median(votes: pd.DataFrame) -> pd.Series:
    """The panel aggregate the paper reports: the per-article median of the four
    judges, rounded back to the integer scale.

    With an even number of judges the median falls on a half point whenever the
    two middle ratings differ, which is 21 of the 70 validated articles, so the
    tie rule is not a detail.  Ties are broken TOWARD ZERO (numpy's default is
    half-to-even, which sends +-0.5 to neutral but pushes +-1.5 out to the
    extreme +-2 -- asymmetric, and an accident of the default rather than a
    choice).  Rounding toward zero never credits the panel with more bias than
    the judges agreed on, and is the conservative direction for a paper whose
    claim is that Grokipedia reads as less neutral.
    """
    med = votes.median(axis=1)
    return np.sign(med) * np.ceil(np.abs(med) - 0.5)


def within1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pair adjacent agreement: True where the two ratings are equal or one
    scale point apart."""
    return np.abs(a - b) <= 1


BOOT_B = 5000       # bootstrap resamples for the 95% percentile CIs
BOOT_SEED = 20260723


def _cluster_boot_ci(groups: list[np.ndarray], stat, B: int = BOOT_B,
                     seed: int = BOOT_SEED) -> tuple[float, float]:
    """95% percentile CI for ``stat`` under resampling of CLUSTERS (articles)
    with replacement -- the right unit of resampling here, because an article
    with two human raters contributes two correlated rating-pairs. ``groups`` is
    one array per cluster; ``stat`` maps a list of (resampled) cluster arrays to
    a scalar. Returns (nan, nan) if under-determined."""
    if len(groups) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(groups))
    est = [s for _ in range(B)
           if not np.isnan(s := stat([groups[i] for i in rng.choice(idx, idx.size)]))]
    if len(est) < 2:
        return (np.nan, np.nan)
    return (round(float(np.percentile(est, 2.5)), 3),
            round(float(np.percentile(est, 97.5)), 3))


def _qwk_pooled_pairs(groups: list[np.ndarray]) -> float:
    """QWK over the pooled (human, model) rating-pairs of the given clusters;
    each cluster array has columns [human, model]."""
    if not groups:
        return np.nan
    arr = np.concatenate(groups)
    if len(arr) < 2:
        return np.nan
    return qwk(arr[:, 0].astype(int), arr[:, 1].astype(int))


PAIRED_SEED = 20260727    # separate stream for the joint (paired-difference) bootstrap


def _wdisag(a: np.ndarray, b: np.ndarray) -> float:
    """Mean quadratic-weighted DISAGREEMENT -- the numerator of QWK, with no
    chance correction: mean((a-b)^2) / 16 on the -2..2 scale (16 = max squared
    difference). Denominator-free, so a difference of two _wdisag values is not
    sensitive to the differing marginals that QWK's expected-agreement term
    uses; serves as the robustness check on the kappa differences."""
    return float(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)) / 16.0


def _consensus(g: np.ndarray) -> int:
    """Pooled human label for one article: mean of its ratings, rounded
    half-away-from-zero back onto the -2..2 scale."""
    m = float(np.mean(g))
    return int(np.sign(m) * np.floor(abs(m) + 0.5))


def paired_ceiling_ci(resp: pd.DataFrame, votes: pd.DataFrame,
                      B: int = BOOT_B, seed: int = PAIRED_SEED) -> pd.DataFrame:
    """Cluster bootstrap of the DIFFERENCE (kappa_model - kappa_human_human) with
    both terms recomputed on the SAME resampled articles.

    Why paired: the two kappas share the articles and the human ratings, so a
    resample that happens to draw a widely-spread article set inflates both
    (their bootstrap correlation is ~0.6-0.7 for the stronger models). Comparing
    two independently-bootstrapped intervals -- or subtracting their endpoints --
    double-counts that shared article-sampling variance and gives a needlessly
    wide interval for the contrast the paper actually claims. Resampling once
    and differencing within the draw cancels it.

    Restricted to articles with >=2 human ratings (the only ones that yield a
    human-human pair), so kappa_model here is computed on that subset and can
    differ slightly from the all-responses `qwk` in agreement_by_model.csv.

    Two aggregation levels, each internally matched:
      * _ind  -- each INDIVIDUAL human rating vs the model, benchmarked against
                 one human vs another human (kappa_hh).
      * _cons -- the 2-rater pooled human label (_consensus) vs the model,
                 benchmarked against the Spearman-Brown reliability of that
                 2-rater mean, 2k/(1+k) -- the matched ceiling for a consensus
                 reference. Using the _ind ceiling here would under-credit the
                 model; using this one at the _ind level would over-penalise it.

    Also bootstraps ``ddisag`` = _wdisag(human, model) - _wdisag(human, human):
    the chance-correction-free version of the same contrast (negative = the
    model is CLOSER to a human rating than a second human is).

    ``p_boot`` is the bootstrap share of draws with difference <= 0 (a one-sided
    plausibility measure, not an exact test); read the interval, not the p."""
    counts = resp.groupby("taskid")["human_rating_int"].count()
    tids = counts[counts >= 2].index.to_numpy()
    if len(tids) < 2:
        return pd.DataFrame()
    H = [resp.loc[resp["taskid"] == t, "human_rating_int"].to_numpy(int) for t in tids]
    # within-article human pairs (both raters of an article, all pairs if >2)
    HP = [np.array([(g[i], g[j]) for i in range(len(g)) for j in range(i + 1, len(g))])
          for g in H]
    cons = np.array([_consensus(g) for g in H])

    mv = {m: votes[m].reindex(tids).to_numpy(float) for m in votes.columns}
    mv["shown_llm"] = resp.groupby("taskid")["llm_rating_int"].first() \
                          .reindex(tids).to_numpy(float)
    mv["llm_median"] = panel_median(votes).reindex(tids).to_numpy(float)

    rng = np.random.default_rng(seed)
    draws = rng.choice(len(tids), size=(B, len(tids)))
    acc = {m: {"ind": [], "cons": [], "dd": []} for m in mv}

    for d in draws:
        hp = np.concatenate([HP[i] for i in d])
        khh = qwk(hp[:, 0], hp[:, 1])
        if np.isnan(khh):
            continue
        ksb = 2 * khh / (1 + khh)          # Spearman-Brown: reliability of a 2-rater mean
        do_hh = _wdisag(hp[:, 0], hp[:, 1])
        for m, v in mv.items():
            sel = [i for i in d if not np.isnan(v[i])]
            if len(sel) < 2:
                continue
            h = np.concatenate([H[i] for i in sel])
            mm = np.concatenate([np.full(len(H[i]), v[i]) for i in sel]).astype(int)
            k_ind = qwk(h, mm)
            k_cons = qwk(cons[sel], v[sel].astype(int))
            if not np.isnan(k_ind):
                acc[m]["ind"].append(k_ind - khh)
                acc[m]["dd"].append(_wdisag(h, mm) - do_hh)
            if not np.isnan(k_cons):
                acc[m]["cons"].append(k_cons - ksb)

    # point estimates on the full sample (same restriction / definitions)
    hp0 = np.concatenate(HP)
    khh0 = qwk(hp0[:, 0], hp0[:, 1])
    ksb0 = 2 * khh0 / (1 + khh0)
    do_hh0 = _wdisag(hp0[:, 0], hp0[:, 1])

    def _ci(x, lo=2.5, hi=97.5):
        return ((round(float(np.percentile(x, lo)), 3),
                 round(float(np.percentile(x, hi)), 3)) if len(x) > 1
                else (np.nan, np.nan))

    rows = []
    for m, v in mv.items():
        sel = np.where(~np.isnan(v))[0]
        h = np.concatenate([H[i] for i in sel])
        mm = np.concatenate([np.full(len(H[i]), v[i]) for i in sel]).astype(int)
        k_ind, k_cons = qwk(h, mm), qwk(cons[sel], v[sel].astype(int))
        lo_i, hi_i = _ci(acc[m]["ind"])
        lo_c, hi_c = _ci(acc[m]["cons"])
        lo_d, hi_d = _ci(acc[m]["dd"])
        rows.append({
            "model": m,
            "n_articles_paired": len(sel),
            "qwk_ind": round(k_ind, 3), "ceiling_ind": round(khh0, 3),
            "diff_ind": round(k_ind - khh0, 3), "diff_ind_lo": lo_i, "diff_ind_hi": hi_i,
            "diff_ind_excl_0": bool(lo_i > 0 or hi_i < 0),
            "p_boot_ind": round(float(np.mean(np.array(acc[m]["ind"]) <= 0)), 4),
            "qwk_cons": round(k_cons, 3), "ceiling_cons": round(ksb0, 3),
            "diff_cons": round(k_cons - ksb0, 3), "diff_cons_lo": lo_c, "diff_cons_hi": hi_c,
            "diff_cons_excl_0": bool(lo_c > 0 or hi_c < 0),
            "ddisag": round(_wdisag(h, mm) - do_hh0, 4),
            "ddisag_lo": lo_d, "ddisag_hi": hi_d,
        })
    return pd.DataFrame(rows)


def metrics(human: np.ndarray, model: np.ndarray, w: np.ndarray | None = None,
            groups: np.ndarray | None = None) -> dict:
    """Agreement metrics; with post-stratification weights ``w`` the keys get a
    ``_w`` suffix and ``n`` becomes ``n_eff`` (Kish effective sample size).

    In the unweighted branch, passing ``groups`` (a cluster id -- the taskid --
    per rating) adds a cluster bootstrap 95% CI on QWK (qwk_lo/qwk_hi),
    resampling ARTICLES rather than rating-pairs so the interval respects the
    two-raters-per-article correlation. ``mean_signed_err`` is mean(human -
    model): positive = humans rate more in-favor (less neutral) than the model."""
    ok = ~np.isnan(model)
    h, m = human[ok].astype(int), model[ok].astype(int)
    if w is None:
        if len(h) == 0:
            return {"n": 0, "qwk": np.nan, "qwk_lo": np.nan, "qwk_hi": np.nan,
                    "exact_pct": np.nan, "within1_pct": np.nan,
                    "mean_signed_err": np.nan}
        d = {
            "n": len(h),
            "qwk": round(qwk(h, m), 3),
            "exact_pct": round(100 * np.mean(h == m), 1),
            "within1_pct": round(100 * np.mean(within1(h, m)), 1),
            "mean_signed_err": round(float(np.mean(h - m)), 3),
        }
        if groups is not None:
            g = np.asarray(groups)[ok]
            by = [np.column_stack([h[g == u], m[g == u]]) for u in pd.unique(g)]
            d["qwk_lo"], d["qwk_hi"] = _cluster_boot_ci(by, _qwk_pooled_pairs)
        else:
            d["qwk_lo"], d["qwk_hi"] = np.nan, np.nan
        return d
    ww = w[ok]
    if len(h) == 0 or ww.sum() == 0:
        return {"n_eff": 0.0, "qwk_w": np.nan, "exact_pct_w": np.nan,
                "within1_pct_w": np.nan}
    return {
        "n_eff": round(ww.sum() ** 2 / (ww ** 2).sum(), 1),
        "qwk_w": round(qwk(h, m, ww), 3),
        "exact_pct_w": round(100 * np.average(h == m, weights=ww), 1),
        "within1_pct_w": round(100 * np.average(within1(h, m), weights=ww), 1),
    }


def rating_sd(per_task: pd.Series, taskids: np.ndarray) -> float:
    """SD of one judge's own ratings across the validation-set articles.

    ``per_task`` maps taskid -> that judge's rating; ``taskids`` is the taskid of
    every analysed human rating. The SD is taken over the DISTINCT articles the
    judge rated (ddof=1), not over the rating-pairs, so a double-rated article
    counts once and the number is the marginal spread of one rating of one
    article -- directly comparable across judges, and to reliability.py's
    ``sd_human`` (0.80 on this sample) as the human benchmark.

    Read it alongside the agreement columns: a judge that parks on the midpoint
    scores well on within1_pct while carrying almost no signal, and that shows up
    here as a small SD rather than in kappa."""
    v = per_task.reindex(pd.unique(taskids)).dropna().to_numpy(float)
    return round(float(np.std(v, ddof=1)), 3) if len(v) > 1 else np.nan


def krippendorff_alpha(groups: list[np.ndarray], metric) -> float:
    """Krippendorff's alpha over units (articles) with >=2 ratings each.
    `metric(a, b)` is the squared-distance function between two ratings."""
    pooled = np.concatenate(groups)
    n_pairable = len(pooled)
    if n_pairable < 2:
        return np.nan
    do = sum(sum(metric(g[i], g[j]) for i in range(len(g)) for j in range(len(g)) if i != j)
             / (len(g) - 1) for g in groups) / n_pairable
    de = sum(metric(a, b) for i, a in enumerate(pooled) for j, b in enumerate(pooled)
             if i != j) / (n_pairable * (n_pairable - 1))
    if de == 0:
        return np.nan
    return 1 - do / de


def human_irr(resp: pd.DataFrame) -> pd.DataFrame:
    """Inter-rater agreement between human annotators on co-rated articles.

    Uses every article with >=2 human ratings: all within-article rater pairs
    for pairwise stats, plus Krippendorff's alpha (interval metric), which is
    designed for this unbalanced any-raters-per-item setting."""
    counts = resp.groupby("taskid")["human_rating_int"].count()
    multi = counts[counts >= 2].index
    groups = [resp.loc[resp["taskid"] == t, "human_rating_int"].to_numpy(int)
              for t in multi]
    pairs = [(g[i], g[j]) for g in groups
             for i in range(len(g)) for j in range(i + 1, len(g))]
    row = {
        "n_articles_total": resp["taskid"].nunique(),
        "n_articles_multirated": len(multi),
        "n_rater_pairs": len(pairs),
    }
    if pairs:
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        # Cluster bootstrap CI on qwk_pairs: resample the multi-rated ARTICLES
        # (each rebuilds its own within-article pairs), matching the human-LLM
        # CI so the ceiling and the observed agreements are comparable intervals.
        def _qwk_hh(gs: list[np.ndarray]) -> float:
            pr = [(g[i], g[j]) for g in gs
                  for i in range(len(g)) for j in range(i + 1, len(g))]
            if len(pr) < 2:
                return np.nan
            return qwk(np.array([p[0] for p in pr]), np.array([p[1] for p in pr]))
        lo, hi = _cluster_boot_ci(groups, _qwk_hh)
        row.update({
            "exact_pct": round(100 * np.mean(a == b), 1),
            "within1_pct": round(100 * np.mean(within1(a, b)), 1),
            "mean_abs_diff": round(np.mean(np.abs(a - b)), 3),
            "qwk_pairs": round(qwk(a, b), 3) if len(pairs) >= 2 else np.nan,
            "qwk_pairs_lo": lo,
            "qwk_pairs_hi": hi,
            "krippendorff_alpha": round(
                krippendorff_alpha(groups, lambda x, y: (x - y) ** 2), 3),
        })
    else:
        row.update({"exact_pct": np.nan, "within1_pct": np.nan,
                    "mean_abs_diff": np.nan, "qwk_pairs": np.nan,
                    "qwk_pairs_lo": np.nan, "qwk_pairs_hi": np.nan,
                    "krippendorff_alpha": np.nan})
    return pd.DataFrame([row])


def load_responses(path: Path, tasks: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Tidy responses from `path`: already-tidy CSVs pass through; a raw
    LimeSurvey download is converted (tidy saved into `out_dir`)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "human_rating_int" in df.columns:      # already tidy
        return df
    from export_responses import load_raw_csv, tidy
    resp = tidy(load_raw_csv(path), tasks)
    tidy_path = out_dir / "responses_tidy.csv"
    resp.to_csv(tidy_path, index=False)
    print(f"Converted raw LimeSurvey export -> {tidy_path} ({len(resp)} completed responses)")
    return resp


RATINGS_TARGET = 2   # each article should be rated exactly this many times


def rating_coverage(resp: pd.DataFrame, resp_all: pd.DataFrame,
                    tasks: pd.DataFrame, out_dir: Path) -> None:
    """Write articles_off_target_ratings.csv: every study article whose number of
    usable (APPROVED) human ratings is not exactly RATINGS_TARGET -- including
    articles fielded but not yet rated (n_ratings = 0).

    Two counts per article:
      * n_ratings   -- APPROVED ratings (the analysis basis; must equal target)
      * n_submitted -- ALL completed submissions regardless of Prolific status
                       (approved + awaiting-review + rejected/...), i.e. what the
                       allocator actually collected. The gap n_submitted -
                       n_ratings is ratings pending approval.

    Off-target is judged on n_ratings. The study's article universe is the
    fielded wave(s): tasks whose wave appears among the rated taskids (all tasks
    when there is no wave column)."""
    counts = resp.groupby("taskid").size()
    counts_all = resp_all.groupby("taskid").size()
    if "wave" in tasks.columns:
        fielded = tasks.loc[tasks["taskid"].isin(resp_all["taskid"]), "wave"].unique()
        study = tasks[tasks["wave"].isin(fielded)]
    else:
        study = tasks
    cov = study[["taskid", "name", "source"]].copy()
    if "wave" in study.columns:
        cov["wave"] = study["wave"].values
    cov["n_ratings"] = cov["taskid"].map(counts).fillna(0).astype(int)
    cov["n_submitted"] = cov["taskid"].map(counts_all).fillna(0).astype(int)
    cov["target"] = RATINGS_TARGET
    cov["delta"] = cov["n_ratings"] - RATINGS_TARGET
    off = cov[cov["n_ratings"] != RATINGS_TARGET].sort_values(["n_ratings", "taskid"])
    out_path = out_dir / "articles_off_target_ratings.csv"
    off.to_csv(out_path, index=False)
    under = int((off["delta"] < 0).sum())
    over = int((off["delta"] > 0).sum())
    print(f"\nWrote {out_path}  ({len(off)}/{len(cov)} articles off the "
          f"{RATINGS_TARGET}-approved-rating target: {under} under, {over} over; "
          f"total submitted {int(cov['n_submitted'].sum())} vs approved {int(cov['n_ratings'].sum())})")
    if len(off):
        print(off.to_string(index=False))


RESPONSES_FILE = "results-survey918273.csv"      # <- the download to analyse


def main() -> None:
    # build_tasks logs which results.csv files it merges and whether it served
    # the combined frame from its cache -- surface that instead of swallowing it.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    responses_file = HERE / "survey_results" / RESPONSES_FILE
    if not responses_file.exists():
        raise SystemExit(f"{responses_file} not found -- set RESPONSES_FILE in this script.")
    tasks = pd.read_csv(SURVEY / "tasks.csv")
    out_dir = HERE / "analyses_results" / f"{responses_file.stem}_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Analysing {responses_file.name} -> {out_dir}")

    resp = load_responses(responses_file, tasks, out_dir)
    resp = resp.dropna(subset=["human_rating_int", "taskid"])
    resp_all = resp.copy()   # all completed responses, before the APPROVED gate
    # Quality gate: keep only participants whose Prolific submission is APPROVED
    # (demographics.INCLUDED_STATUSES) in EVERY metric below. Skipped only when
    # no Prolific export is present to read status from.
    from demographics import included_pids
    inc = included_pids()
    if inc is not None:
        before = len(resp)
        resp = resp[resp["prolific_pid"].isin(inc)].copy()
        print(f"Prolific status gate: kept {len(resp)}/{before} responses (APPROVED only)")
    else:
        print("(no Prolific export -- APPROVED-status gate skipped; using all responses)")
    votes = llm_votes(tasks)
    # Post-stratification weights (disagreement bin, corpus/sample): unweighted
    # columns describe the difficulty-enriched sample; *_w columns project to
    # the length-eligible corpus (n_eff = Kish effective sample size). With a
    # pre-committed wave-2 reserve, sample bin shares must come from the waves
    # actually FIELDED (wave-2 taskids only circulate once the allocator's N is
    # raised), so restrict to waves that appear in the responses.
    if "wave" in tasks.columns:
        fielded = tasks.loc[tasks["taskid"].isin(resp["taskid"]), "wave"].unique()
        w_task = task_weights(tasks[tasks["wave"].isin(fielded)])
    else:
        w_task = task_weights(tasks)
    w = resp["taskid"].map(w_task).to_numpy(float)

    def reasonable(sub: pd.DataFrame) -> dict:
        """Participants' verdict on the SHOWN rating (only exists for the model
        that was displayed): share judged yes / partially / no, plus the
        post-stratification-weighted 'yes' share."""
        v = sub["llm_reasonable"].dropna()
        if len(v) == 0:
            return {"n_shown": 0, "reasonable_yes_pct": np.nan,
                    "reasonable_partial_pct": np.nan, "reasonable_no_pct": np.nan,
                    "reasonable_yes_pct_w": np.nan}
        ws = sub.loc[v.index, "taskid"].map(w_task).to_numpy(float)
        return {
            "n_shown": len(v),
            "reasonable_yes_pct": round(100 * (v == "yes").mean(), 1),
            "reasonable_partial_pct": round(100 * (v == "partially").mean(), 1),
            "reasonable_no_pct": round(100 * (v == "no").mean(), 1),
            "reasonable_yes_pct_w": round(100 * np.average(v == "yes", weights=ws), 1),
        }

    human = resp["human_rating_int"].to_numpy(float)
    grp = resp["taskid"].to_numpy()   # cluster id per rating for the bootstrap CI
    rows = []
    for model in votes.columns:
        mv = resp["taskid"].map(votes[model]).to_numpy(float)
        rows.append({"model": model,
                     **metrics(human, mv, groups=grp), **metrics(human, mv, w),
                     "rating_sd": rating_sd(votes[model], grp),
                     **reasonable(resp[resp["llm_model"] == model])})
    # the rating each respondent actually saw (reasonable = all responses)
    shown = resp["llm_rating_int"].to_numpy(float)
    rows.append({"model": "shown_llm",
                 **metrics(human, shown, groups=grp), **metrics(human, shown, w),
                 # one shown rating per article, so the by-article series is well
                 # defined even though the displayed model varies across articles
                 "rating_sd": rating_sd(
                     resp.groupby("taskid")["llm_rating_int"].first(), grp),
                 **reasonable(resp)})
    # 4-model median per article (rounded to the scale; no reasonableness verdict)
    med = panel_median(votes)
    mv = resp["taskid"].map(med).to_numpy(float)
    rows.append({"model": "llm_median",
                 **metrics(human, mv, groups=grp), **metrics(human, mv, w),
                 "rating_sd": rating_sd(med, grp),
                 **reasonable(resp.iloc[0:0])})

    out = pd.DataFrame(rows, columns=["model", "n", "qwk", "qwk_lo", "qwk_hi",
                                      "exact_pct", "within1_pct", "mean_signed_err",
                                      "rating_sd",
                                      "n_eff", "qwk_w", "exact_pct_w", "within1_pct_w",
                                      "n_shown", "reasonable_yes_pct",
                                      "reasonable_partial_pct", "reasonable_no_pct",
                                      "reasonable_yes_pct_w"])
    out_path = out_dir / "agreement_by_model.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}\n")
    print(out.to_string(index=False))

    # Human inter-rater ceiling (same QWK metric) + the model-vs-ceiling table.
    irr = human_irr(resp)
    irr_path = out_dir / "human_irr.csv"
    irr.to_csv(irr_path, index=False)
    print(f"\nWrote {irr_path}\n")
    print(irr.to_string(index=False))
    # Paired (joint-resample) differences vs the ceiling -- the headline contrast.
    paired = paired_ceiling_ci(resp, votes)
    if not paired.empty:
        paired.to_csv(out_dir / "ceiling_diff.csv", index=False)
    agreement_vs_ceiling(out, irr, out_dir, paired)

    # Reliability / ceiling / consensus-correlation battery (reliability.py):
    # ordinal alpha, ICC, sqrt(ICC(1,2)) ceiling, Pearson/Spearman/Kendall vs the
    # 2-rater human mean, the substitution contrast and the compression
    # diagnostics -- one shared cluster bootstrap, so its differences are paired.
    from reliability import reliability_report
    reliability_report(resp, votes, out_dir)

    rating_coverage(resp, resp_all, tasks, out_dir)
    agreement_by_disagreement(resp, votes, tasks, out_dir)
    politics_alignment(resp, votes, out_dir)


def agreement_vs_ceiling(out: pd.DataFrame, irr: pd.DataFrame, out_dir: Path,
                         paired: pd.DataFrame | None = None) -> None:
    """Focused paper table: each model's human-LLM QWK (with cluster-bootstrap
    95% CI) next to the human-human QWK ceiling on the same metric, plus the gap
    and whether the model's CI clears the ceiling point estimate.

    ``ci_clears_ceiling`` compares an interval to a POINT estimate, ignoring the
    ceiling's own +-0.19 uncertainty; it is kept for continuity but the headline
    claim should rest on the paired columns from ``paired_ceiling_ci`` (merged in
    when supplied), which bootstrap kappa_model - kappa_hh on the same resampled
    articles and so account for both uncertainties and their correlation."""
    ceil = float(irr["qwk_pairs"].iloc[0])
    c_lo, c_hi = float(irr["qwk_pairs_lo"].iloc[0]), float(irr["qwk_pairs_hi"].iloc[0])
    tbl = out[["model", "n", "qwk", "qwk_lo", "qwk_hi", "mean_signed_err"]].copy()
    tbl["qwk_ceiling"] = round(ceil, 3)
    tbl["gap_vs_ceiling"] = (tbl["qwk"] - ceil).round(3)
    # "meets ceiling" = model CI lower bound is not below the ceiling point est.
    tbl["ci_clears_ceiling"] = tbl["qwk_lo"] >= ceil
    ceiling_row = pd.DataFrame([{
        "model": "human_ceiling", "n": int(irr["n_rater_pairs"].iloc[0]),
        "qwk": round(ceil, 3), "qwk_lo": c_lo, "qwk_hi": c_hi,
        "mean_signed_err": np.nan, "qwk_ceiling": round(ceil, 3),
        "gap_vs_ceiling": 0.0, "ci_clears_ceiling": True}])
    tbl = pd.concat([tbl, ceiling_row], ignore_index=True)
    if paired is not None and not paired.empty:
        tbl = tbl.merge(paired.drop(columns=["qwk_ind", "ceiling_ind"]),
                        on="model", how="left")
    out_path = out_dir / "agreement_vs_ceiling.csv"
    tbl.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  (human-LLM QWK vs human-human ceiling, "
          f"cluster-bootstrap 95% CIs)\n")
    print(tbl.to_string(index=False))
    if paired is not None and not paired.empty:
        cols = ["model", "diff_ind", "diff_ind_lo", "diff_ind_hi", "p_boot_ind",
                "diff_cons", "diff_cons_lo", "diff_cons_hi", "ddisag",
                "ddisag_lo", "ddisag_hi"]
        print(f"\nPaired difference vs the human ceiling ({BOOT_B} joint cluster "
              f"resamples; ceiling_ind={paired['ceiling_ind'].iloc[0]}, "
              f"ceiling_cons={paired['ceiling_cons'].iloc[0]}):")
        print("  _ind  = individual human rating vs model, minus human-vs-human")
        print("  _cons = 2-rater pooled label vs model, minus Spearman-Brown ceiling")
        print("  ddisag = weighted disagreement gap, no chance correction "
              "(negative = model closer to a human than a 2nd human is)\n")
        print(paired[cols].to_string(index=False))


def agreement_by_disagreement(resp: pd.DataFrame, votes: pd.DataFrame,
                              tasks: pd.DataFrame, out_dir: Path) -> None:
    """Does human-LLM agreement track how much the JUDGES disagree with each other?

    The sample is stratified on judge disagreement (build_tasks: sd of the four
    models' integer ratings, cut into DEFAULT_DISAGREEMENT_BINS equal-width bins
    from 0 to the eligible-corpus maximum), so the strata are already in
    ``tasks.disagreement_bin``. One row per bin plus an ``all`` row:

      * n_articles, n_ratings  -- articles in the bin, human ratings of them
      * mean_disagreement      -- mean sd of the four judges' ratings
      * exact_pct, within1_pct, mae -- pooled over ALL FOUR judges x every human
        rating of the article, so an article contributes up to 4 x 2 = 8 pairs.
        exact_pct and mae carry cluster-bootstrap 95% CIs (articles resampled,
        since those 8 pairs are not independent).
      * med_exact_pct, med_within1_pct, med_mae -- the same against the rounded
        4-model median, i.e. the panel aggregate the paper reports.
      * hh_exact_pct, hh_within1_pct, hh_gap -- the HUMAN-human comparison on the
        same articles. This is the control for "those articles are simply
        harder": if judge disagreement only tracked task difficulty, human
        annotators would lose agreement in step with the judges.

    The ``all`` row additionally carries bin-free trend statistics over articles,
    which therefore do not depend on the bin edges:

      * rho_llm, p_llm -- Spearman of judge disagreement vs |human - panel
        median| per article; positive = the judges agree with humans LESS on the
        articles where they disagree with each other.
      * rho_hh, p_hh   -- the same against the human-human gap.

    Treat the per-bin rows as descriptive. With ~14 articles per bin, adjacent
    bins cannot be ordered, which is why the trend is reported on the pooled
    articles instead."""
    if "disagreement_bin" not in tasks.columns:
        print("\n(no disagreement_bin in tasks.csv -- agreement-by-disagreement skipped)")
        return
    meta = tasks.drop_duplicates("taskid").set_index("taskid")[
        ["llm_disagreement", "disagreement_bin"]]
    r = resp.join(meta, on="taskid").dropna(subset=["disagreement_bin"])
    if r.empty:
        print("\n(no rated articles carry a disagreement bin -- skipped)")
        return
    med = panel_median(votes)                   # matches the llm_median row above
    r = r.assign(e_med=(r["human_rating_int"].astype(float)
                        - r["taskid"].map(med).astype(float)).abs())

    def judge_pairs(sub: pd.DataFrame) -> list[np.ndarray]:
        """One [human, judge] pair-array per article, pooling all four judges."""
        out = []
        for tid, g in sub.groupby("taskid"):
            jv = (votes.loc[tid].dropna().to_numpy(float)
                  if tid in votes.index else np.empty(0))
            if not len(jv):
                continue
            h = g["human_rating_int"].to_numpy(float)
            out.append(np.column_stack([np.repeat(h, len(jv)), np.tile(jv, len(h))]))
        return out

    def _pooled(groups, fn):
        if not groups:
            return np.nan
        a = np.concatenate(groups)
        return float(fn(np.abs(a[:, 0] - a[:, 1])))

    _exact = lambda gs: _pooled(gs, lambda d: 100 * np.mean(d == 0))   # noqa: E731
    _w1 = lambda gs: _pooled(gs, lambda d: 100 * np.mean(d <= 1))      # noqa: E731
    _mae = lambda gs: _pooled(gs, np.mean)                             # noqa: E731

    def hh_gaps(sub: pd.DataFrame) -> np.ndarray:
        """All within-article |rating_i - rating_j| over human raters."""
        g = [abs(a - b) for _, s in sub.groupby("taskid")
             for a, b in combinations(s["human_rating_int"].astype(float), 2)]
        return np.asarray(g, float)

    def row(label, sub: pd.DataFrame) -> dict:
        gs, hh = judge_pairs(sub), hh_gaps(sub)
        e_lo, e_hi = _cluster_boot_ci(gs, _exact)
        m_lo, m_hi = _cluster_boot_ci(gs, _mae)
        e_med = sub["e_med"].dropna()
        return {
            "bin": label,
            "n_articles": int(sub["taskid"].nunique()),
            "n_ratings": int(len(sub)),
            "n_judge_pairs": int(sum(len(g) for g in gs)),
            "mean_disagreement": round(float(sub["llm_disagreement"].mean()), 3),
            "exact_pct": round(_exact(gs), 1),
            "exact_pct_lo": round(e_lo, 1), "exact_pct_hi": round(e_hi, 1),
            "within1_pct": round(_w1(gs), 1),
            "mae": round(_mae(gs), 3),
            "mae_lo": round(m_lo, 3), "mae_hi": round(m_hi, 3),
            "med_exact_pct": round(100 * float((e_med == 0).mean()), 1),
            "med_within1_pct": round(100 * float((e_med <= 1).mean()), 1),
            "med_mae": round(float(e_med.mean()), 3),
            "hh_exact_pct": round(100 * float((hh == 0).mean()), 1) if len(hh) else None,
            "hh_within1_pct": round(100 * float((hh <= 1).mean()), 1) if len(hh) else None,
            "hh_gap": round(float(hh.mean()), 3) if len(hh) else None,
        }

    rows = [row(int(b), sub) for b, sub in r.groupby("disagreement_bin")]
    rows.append(row("all", r))

    # Bin-free trends over articles: judge disagreement vs the human gap, and the
    # same against the human-human gap (the difficulty control).
    art = r.groupby("taskid").agg(dis=("llm_disagreement", "first"),
                                  gap=("e_med", "mean"))
    hh_art = pd.Series({t: g for t, s in r.groupby("taskid")
                        for g in [np.mean([abs(a - b) for a, b in combinations(
                            s["human_rating_int"].astype(float), 2)] or [np.nan])]})
    if len(art) > 2:
        rho, p = stats.spearmanr(art["dis"], art["gap"])
        rows[-1].update(rho_llm=round(float(rho), 3), p_llm=round(float(p), 4))
        both = pd.concat([art["dis"], hh_art.rename("hh")], axis=1).dropna()
        if len(both) > 2:
            rho2, p2 = stats.spearmanr(both["dis"], both["hh"])
            rows[-1].update(rho_hh=round(float(rho2), 3), p_hh=round(float(p2), 4))

    out = pd.DataFrame(rows)
    out_path = out_dir / "agreement_by_disagreement.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  (per-bin n is small: descriptive, see rho_* on the "
          f"'all' row for the bin-free trend)\n")
    print(out.to_string(index=False))


def politics_alignment(resp: pd.DataFrame, votes: pd.DataFrame, out_dir: Path) -> None:
    """Does rater ideology (1-10 left-right self-placement, WVS E033) predict alignment
    with each LLM? Per model:

      * spearman_align -- rank correlation between ideology and CLOSENESS to the
        model (-|human - model|): positive = right-leaning raters agree more.
      * spearman_signed -- rank correlation between ideology and the signed gap
        (human - model): positive = right-leaning raters rate articles as MORE
        in-favor than the model does.
      * mean |human-model| for left (1-4), centre (5-6), right (7-10) raters.

    Exploratory by design (few raters per cell); opt-outs are excluded."""
    r = resp.dropna(subset=["econ_leftright"]).copy()
    if r.empty:
        print("\n(no political self-placement data yet -- alignment-by-politics skipped)")
        return
    e = r["econ_leftright"].astype(float)
    band = pd.cut(e, bins=[0.5, 4.5, 6.5, 10.5], labels=["left", "centre", "right"])
    rows = []
    model_cols = list(votes.columns) + ["shown_llm"]
    for model in model_cols:
        mv = (r["llm_rating_int"].astype(float) if model == "shown_llm"
              else r["taskid"].map(votes[model]).astype(float))
        ok = mv.notna()
        h, m_, ee, bb = r["human_rating_int"][ok].astype(float), mv[ok], e[ok], band[ok]
        if len(h) < 3:
            continue
        gap = h - m_
        closeness = -gap.abs()
        row = {"model": model, "n": len(h),
               "spearman_align": round(pd.Series(ee.values).corr(
                   pd.Series(closeness.values), method="spearman"), 3),
               "spearman_signed": round(pd.Series(ee.values).corr(
                   pd.Series(gap.values), method="spearman"), 3)}
        for b in ("left", "centre", "right"):
            sel = bb == b
            row[f"absdiff_{b}"] = round(gap[sel].abs().mean(), 2) if sel.any() else None
            row[f"n_{b}"] = int(sel.sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    out_path = out_dir / "alignment_by_politics.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  (exploratory: small per-cell n)\n")
    print(out.to_string(index=False))

    # Participant demographics (Prolific export) -> demographics/ subdirectory.
    from demographics import demographics_report
    demographics_report(resp, out_dir)


if __name__ == "__main__":
    main()
