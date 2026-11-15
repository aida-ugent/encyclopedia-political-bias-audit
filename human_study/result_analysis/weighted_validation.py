#!/usr/bin/env python3
"""Post-stratified twins of the validation headline numbers.

``agreement.py`` already reports weighted QWK, exact and within-1 agreement
(the ``_w`` columns). The main paper leads on the Pearson correlation with the
human consensus and on the reasonableness share, neither of which has a
weighted twin there. This script supplies both, using the same
disagreement-bin post-stratification weights (``agreement.task_weights``), so
the numbers project from the difficulty-enriched sample onto the
length-eligible paired corpus.

    python survey/result_analysis/weighted_validation.py

Writes ``weighted_validation.csv`` next to the other analysis outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agreement import (  # noqa: E402
    HERE, RESPONSES_FILE, SURVEY, llm_votes, load_responses, panel_median,
    task_weights,
)
from demographics import included_pids  # noqa: E402

MODELS = ["claude-opus-4-6", "grok-4", "mistralai", "deepseek-v4-pro"]


def _weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    w = w / w.sum()
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    vx = np.average((x - mx) ** 2, weights=w)
    vy = np.average((y - my) ** 2, weights=w)
    return float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else np.nan


def _load_gated_responses(tasks: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Tidy responses under the same APPROVED-only gate ``agreement.py`` uses."""
    resp = load_responses(HERE / "survey_results" / RESPONSES_FILE, tasks, out_dir)
    resp = resp.dropna(subset=["human_rating_int", "taskid"])
    inc = included_pids()
    if inc is not None:
        resp = resp[resp["prolific_pid"].isin(inc)].copy()
    return resp


def main() -> None:
    tasks = pd.read_csv(SURVEY / "tasks.csv")
    out_dir = HERE / "analyses_results" / f"{Path(RESPONSES_FILE).stem}_analysis"
    tidy = _load_gated_responses(tasks, out_dir)
    weights = task_weights(tasks)
    votes = llm_votes(tasks)

    # Human consensus per article: the mean of its two raters.
    consensus = tidy.groupby("taskid")["human_rating_int"].mean()
    tids = consensus.index
    w = weights.reindex(tids).astype(float)
    h = consensus.to_numpy(float)

    rows = []
    panel = votes.reindex(tids)[MODELS]
    series = {m: panel[m].to_numpy(float) for m in MODELS}
    # Rounded median, matching the ``llm_median`` row in reliability.py.
    series["llm_median"] = panel_median(panel).to_numpy(float)

    for name, m in series.items():
        ok = ~np.isnan(m) & ~np.isnan(h)
        rows.append(
            {
                "rater": name,
                "n_articles": int(ok.sum()),
                "pearson_r": round(_weighted_pearson(h[ok], m[ok], np.ones(ok.sum())), 3),
                "pearson_r_w": round(_weighted_pearson(h[ok], m[ok], w.to_numpy()[ok]), 3),
            }
        )

    # Reasonableness of the rating each annotator was actually shown, weighted
    # on the article the rating belonged to.
    shown = tidy.dropna(subset=["llm_reasonable"]).copy()
    shown["w"] = shown["taskid"].map(weights).astype(float)
    verdict = shown["llm_reasonable"].astype(str).str.lower()
    combined = verdict.isin(["yes", "partially"]).to_numpy(float)
    rows.append(
        {
            "rater": "shown_llm (reasonable or partial)",
            "n_articles": int(len(shown)),
            "pearson_r": round(100 * combined.mean(), 1),
            "pearson_r_w": round(
                100 * np.average(combined, weights=shown["w"].to_numpy()), 1
            ),
        }
    )

    out = pd.DataFrame(rows)
    dest = out_dir / "weighted_validation.csv"
    out.to_csv(dest, index=False)
    print(out.to_string(index=False))
    # Article-level effective sample size; agreement.py reports the
    # rating-level twin, which is twice this at two raters per article.
    print(f"\nKish n_eff = {w.sum() ** 2 / (w ** 2).sum():.1f} of {len(w)} articles")
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
