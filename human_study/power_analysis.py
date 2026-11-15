"""Precision analysis for the human-validation survey: how many articles / raters
are needed for acceptably tight confidence intervals on the validation metrics.

Metrics simulated (Monte Carlo, generative model below):
  * QWK(human, LLM)  -- quadratic-weighted kappa between pooled human ratings and
    the LLM rating shown for the same article (the headline validation number)
  * QWK(human, human) -- inter-rater agreement between the two raters of an article

Generative model: each article has a discrete true bias ``t`` drawn from the
empirical LLM-rating distribution of the task pool (tasks.csv by default). Every
rating (LLM or human) is ``round(clip(t + N(0, sigma)))`` on the -2..2 scale.
``sigma_llm`` is fixed; a grid of ``sigma_human`` values sweeps optimistic ->
pessimistic true agreement. For each (n_articles, raters, sigma_human) cell the
script reports the implied TRUE kappas and the 95% CI half-width of the estimate
(1.96 x Monte-Carlo SD), i.e. the precision you can expect at that sample size.

The proportion-based metric (share of LLM ratings judged "reasonable") has a
closed form and is printed analytically, with a cluster design effect for
multiple ratings per article.

Run:
    python survey/power_analysis.py
    python survey/power_analysis.py --n-grid 50 100 150 200 --raters 2 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LEVELS = np.array([-2, -1, 0, 1, 2])
W = (LEVELS[:, None] - LEVELS[None, :]) ** 2 / (LEVELS[-1] - LEVELS[0]) ** 2


def qwk(a: np.ndarray, b: np.ndarray, w: np.ndarray | None = None) -> float:
    """Quadratic-weighted kappa for ratings in {-2..2}. Optional per-pair sample
    weights ``w`` (e.g. post-stratification weights) scale each pair's
    contribution to the observed and expected agreement matrices."""
    obs = np.zeros((5, 5))
    np.add.at(obs, (a + 2, b + 2), 1.0 if w is None else w)
    obs /= len(a) if w is None else w.sum()
    exp = obs.sum(1)[:, None] * obs.sum(0)[None, :]
    denom = (W * exp).sum()
    if denom == 0:
        return np.nan
    return 1 - (W * obs).sum() / denom


def rate(truth: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.clip(np.rint(truth + rng.normal(0, sigma, truth.shape)), -2, 2).astype(int)


def simulate_cell(
    n_articles: int,
    n_raters: int,
    sigma_h: float,
    sigma_llm: float,
    probs: np.ndarray,
    nsim: int,
    rng: np.random.Generator,
) -> dict:
    k_hl, k_hh = [], []
    for _ in range(nsim):
        truth = rng.choice(LEVELS, size=n_articles, p=probs)
        llm = rate(truth, sigma_llm, rng)
        humans = [rate(truth, sigma_h, rng) for _ in range(n_raters)]
        k_hl.append(qwk(np.concatenate(humans), np.tile(llm, n_raters)))
        k_hh.append(np.mean([qwk(humans[i], humans[j])
                             for i in range(n_raters) for j in range(i + 1, n_raters)]))
    k_hl, k_hh = np.array(k_hl), np.array(k_hh)
    return {
        "n_articles": n_articles,
        "raters": n_raters,
        "sigma_h": sigma_h,
        "true_qwk_hl": np.nanmean(k_hl),
        "hw_hl": 1.96 * np.nanstd(k_hl),
        "true_qwk_hh": np.nanmean(k_hh),
        "hw_hh": 1.96 * np.nanstd(k_hh),
    }


def proportion_table(n_grid: list[int], raters: list[int], icc: float) -> pd.DataFrame:
    """95% CI half-width for the share of LLM ratings judged 'reasonable',
    with a design effect for m ratings clustered per article."""
    rows = []
    for n in n_grid:
        for m in raters:
            deff = 1 + (m - 1) * icc
            for p in (0.7, 0.8, 0.9):
                hw = 1.96 * np.sqrt(p * (1 - p) * deff / (n * m))
                rows.append({"n_articles": n, "raters": m, "p": p, "hw": hw})
    return pd.DataFrame(rows).pivot_table(index=["n_articles", "raters"], columns="p", values="hw")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", type=Path, default=HERE / "tasks.csv",
                    help="task pool whose llm_int column gives the article truth distribution")
    ap.add_argument("--n-grid", type=int, nargs="+", default=[50, 75, 100, 150, 200, 300])
    ap.add_argument("--raters", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--sigma-human", type=float, nargs="+", default=[0.5, 0.8, 1.1],
                    help="human noise SD grid: ~optimistic / moderate / pessimistic agreement")
    ap.add_argument("--sigma-llm", type=float, default=0.4,
                    help="LLM noise SD around article truth (~mean inter-model std in results)")
    ap.add_argument("--icc", type=float, default=0.3,
                    help="within-article correlation of 'reasonable' verdicts (design effect)")
    ap.add_argument("--probs", type=float, nargs=5, default=None, metavar="P",
                    help="override the truth distribution for -2..2 (sensitivity analysis); "
                         "normalised, e.g. --probs 1 1 1 1 1 for uniform")
    ap.add_argument("--nsim", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.probs is not None:
        probs = np.array(args.probs) / np.sum(args.probs)
        print("Article truth distribution (from --probs):")
    else:
        tasks = pd.read_csv(args.tasks)
        counts = tasks["llm_int"].value_counts().reindex(LEVELS, fill_value=0)
        probs = (counts / counts.sum()).to_numpy()
        print(f"Article truth distribution (from {args.tasks.name}, n={len(tasks)}):")
    print("  " + ", ".join(f"{v:+d}: {p:.0%}" for v, p in zip(LEVELS, probs)))

    rng = np.random.default_rng(args.seed)
    rows = [
        simulate_cell(n, m, sh, args.sigma_llm, probs, args.nsim, rng)
        for sh in args.sigma_human for m in args.raters for n in args.n_grid
    ]
    res = pd.DataFrame(rows)

    print(f"\n95% CI half-widths for QWK (sigma_llm={args.sigma_llm}, {args.nsim} sims/cell)")
    print("hw_hl = human-vs-LLM kappa, hw_hh = human-vs-human kappa\n")
    for sh, grp in res.groupby("sigma_h"):
        t = grp.iloc[0]
        print(f"sigma_human={sh}  (true QWK human-LLM ~{t.true_qwk_hl:.2f}, "
              f"human-human ~{t.true_qwk_hh:.2f})")
        print(grp.pivot_table(index="n_articles", columns="raters",
                              values=["hw_hl", "hw_hh"]).round(3).to_string(), "\n")

    print(f"95% CI half-width for prop. 'reasonable' (analytic, ICC={args.icc}):")
    print(proportion_table(args.n_grid, args.raters, args.icc).round(3).to_string())


if __name__ == "__main__":
    main()
