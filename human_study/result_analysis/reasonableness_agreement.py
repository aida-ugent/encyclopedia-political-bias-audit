"""Do the two annotators of an article agree on whether the judge was reasonable?

The table reports the share of shown ratings annotators called reasonable
(87.9%), but with no human baseline it is uninterpretable: the same problem the
neutrality ratings had before alpha(H,H) = 0.20 gave them a reference. Both
annotators of an article saw the SAME judge's rating and explanation -- judge
assignment is per article, not per annotator -- so the two reasonableness
verdicts are a genuine human-human pair on an identical stimulus, and the
baseline is computable.

Scale: no < partially < yes, treated as ordinal (3 levels). Reported alongside
the binary collapse (yes/partially vs no), which is the form the paper's 87.9%
uses.

Everything is bootstrapped over ARTICLES (the pair is the unit; the two
annotators of an article are not independent observations), matching the rest of
the battery.

Outputs:
  reasonableness_agreement.csv   one row per metric

Usage: python reasonableness_agreement.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import agreement as agr
from demographics import included_pids

HERE = Path(__file__).resolve().parent
SURVEY = HERE.parent
ANALYSIS_DIR = HERE / "analyses_results" / f"{Path(agr.RESPONSES_FILE).stem}_analysis"

# no < partially < yes: an ordinal scale, so a no/partially split is a smaller
# disagreement than a no/yes one and alpha should be told so.
CODES = {"no": 0, "partially": 1, "yes": 2}
BOOT_B, BOOT_SEED = 5000, 20260723


def delta2_ordinal(n_c: np.ndarray) -> np.ndarray:
    """Krippendorff's squared ordinal difference function (reliability.py's, but
    for an arbitrary number of levels rather than the fixed 5-point scale)."""
    cum = np.cumsum(n_c)
    pre = cum - n_c
    m = cum[None, :] - pre[:, None] - (n_c[:, None] + n_c[None, :]) / 2
    up = np.triu(m)
    return (up + up.T) ** 2


def alpha_ordinal(pairs: np.ndarray, n_levels: int) -> float:
    """Ordinal Krippendorff's alpha over (n_units x 2) integer-coded pairs."""
    o = np.zeros((n_levels, n_levels))
    for a, b in pairs:
        o[a, b] += 1
        o[b, a] += 1
    n_c = o.sum(1)
    n = n_c.sum()
    if n < 2:
        return np.nan
    d2 = delta2_ordinal(n_c)
    d_o = float((o * d2).sum()) / n
    d_e = float((np.outer(n_c, n_c) * d2).sum()) / (n * (n - 1))
    return 1 - d_o / d_e if d_e else np.nan


def metrics(pairs: np.ndarray, rating_gap: np.ndarray) -> dict:
    a, b = pairs[:, 0], pairs[:, 1]
    binar = (pairs >= 1).astype(int)            # yes/partially vs no
    both_ok = (binar.sum(1) == 2)
    out = {
        "exact_pct": 100 * float(np.mean(a == b)),
        "alpha_ordinal": alpha_ordinal(pairs, len(CODES)),
        "alpha_binary": alpha_ordinal(binar, 2),
        "exact_pct_binary": 100 * float(np.mean(binar[:, 0] == binar[:, 1])),
        "both_reasonable_pct": 100 * float(np.mean(both_ok)),
        "both_no_pct": 100 * float(np.mean(binar.sum(1) == 0)),
        "split_pct": 100 * float(np.mean(binar.sum(1) == 1)),
        "both_yes_pct": 100 * float(np.mean((a == 2) & (b == 2))),
    }
    # Does endorsing the judge go with the two annotators agreeing with EACH
    # OTHER? If a pair that disagrees on the article still endorses the judge,
    # the endorsement is not merely a restatement of agreement.
    ok = ~np.isnan(rating_gap)
    if ok.sum() > 2:
        out["mean_rating_gap_both_ok"] = float(
            np.mean(rating_gap[ok & both_ok])) if (ok & both_ok).any() else np.nan
        out["mean_rating_gap_not_both_ok"] = float(
            np.mean(rating_gap[ok & ~both_ok])) if (ok & ~both_ok).any() else np.nan
    return out


def main() -> None:
    tasks = pd.read_csv(SURVEY / "tasks.csv")
    resp = agr.load_responses(HERE / "survey_results" / agr.RESPONSES_FILE,
                              tasks, ANALYSIS_DIR)
    resp = resp.dropna(subset=["human_rating_int", "taskid"])
    inc = included_pids()
    if inc is not None:
        resp = resp[resp["prolific_pid"].isin(inc)].copy()

    resp["reas_code"] = resp["llm_reasonable"].map(CODES)
    rows, gaps = [], []
    dropped = 0
    for tid, g in resp.groupby("taskid"):
        c = g["reas_code"].dropna()
        if len(c) != 2:
            dropped += 1
            continue
        rows.append(c.to_numpy(int)[:2])
        r = g["human_rating_int"].to_numpy(float)
        gaps.append(abs(r[0] - r[1]) if len(r) >= 2 else np.nan)
    pairs, gap = np.array(rows), np.array(gaps)
    print(f"{len(pairs)} articles with two reasonableness verdicts "
          f"({dropped} dropped for missing a verdict)")
    # Sanity check on the premise: both annotators must have seen the same judge.
    same = resp.groupby("taskid")["llm_model"].nunique()
    print(f"articles where both annotators saw the same judge: "
          f"{(same == 1).sum()}/{len(same)}")

    point = metrics(pairs, gap)
    rng = np.random.default_rng(BOOT_SEED)
    reps = [metrics(pairs[i], gap[i])
            for i in rng.choice(len(pairs), size=(BOOT_B, len(pairs)))]
    boot = pd.DataFrame(reps)

    print("\nHUMAN--HUMAN AGREEMENT ON THE SAME EXPLANATION")
    out = []
    for k, v in point.items():
        s = boot[k].replace([np.inf, -np.inf], np.nan).dropna()
        lo, hi = np.percentile(s, [2.5, 97.5]) if len(s) > 1 else (np.nan, np.nan)
        print(f"  {k:<28} {v:7.3f}  [{lo:.3f}, {hi:.3f}]")
        out.append({"metric": k, "value": v, "ci_lo": lo, "ci_hi": hi,
                    "n_articles": len(pairs)})
    pd.DataFrame(out).to_csv(ANALYSIS_DIR / "reasonableness_agreement.csv",
                             index=False)
    print(f"\nwrote {ANALYSIS_DIR / 'reasonableness_agreement.csv'}")


if __name__ == "__main__":
    main()
