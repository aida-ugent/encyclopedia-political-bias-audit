"""Joint distribution of the two human ratings of an article, as one heatmap.

human_descriptives.py already draws this as panel (a) of the three-panel
figures/human_agreement.png, but in the SORTED (lo, hi) form: only the upper
triangle can be populated, so the matrix is half empty and the axes read
"lower rating" / "higher rating" rather than the rating scale itself. This
module renders the same pairs on the full -2..+2 x -2..+2 grid, matching the
layout of human_vs_panel_median_heatmap.py so the human-human and the
human-panel matrices can be read side by side.

The two annotators of an article are interchangeable (140 ratings from 140
distinct raters, nothing orders them), so there is no rater 1 and rater 2 to
put on the axes. The matrix is therefore built RATING-wise and is symmetric by
construction: cell (i, j) counts the ratings equal to i whose partner rating
was j, so every one of the 140 ratings appears exactly once, the row and column
margins are both the rating distribution, and the diagonal holds the ratings
whose partner agreed exactly (twice the number of agreeing articles). Reading
"n at (i, j)" as "n annotators said i while their co-rater said j" is exact,
and no arbitrary ordering of the pair is imposed.

Outputs, into the same analysis directory as the other survey figures:
  figures/human_rater_pair_heatmap.png / .pdf
  human_rater_pair_heatmap.csv   the 25 cells + the pair-level margins

Usage: python rater_pair_heatmap.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns

import human_descriptives as hd

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = hd.ANALYSIS_DIR
FIG_DIR = hd.FIG_DIR

SCORES = list(range(-2, 3))
# Same orange as the other survey figures: the validated contrast partner to
# the Blues ramp, used here only to ring the exact-agreement diagonal.
EXACT = "#eb6834"


def symmetric_matrix(pairs: pd.DataFrame) -> np.ndarray:
    """cell (i, j) = ratings of value i whose partner rated j. Sums to 2*n_pairs."""
    cm = np.zeros((5, 5), dtype=int)
    for lo, hi in zip(pairs["lo"].astype(int), pairs["hi"].astype(int)):
        cm[lo + 2, hi + 2] += 1
        cm[hi + 2, lo + 2] += 1
    return cm


def render(cm: np.ndarray, paths: list[Path]) -> None:
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(6.8, 6.0))

    total = cm.sum()
    cm_pct = cm / total * 100
    annot = np.array([[f"{cm[i, j]}\n{cm_pct[i, j]:.1f}%" if cm[i, j] else ""
                       for j in range(5)] for i in range(5)], dtype=object)
    sns.heatmap(cm_pct, annot=annot, fmt="", cmap="Blues", ax=ax, square=True,
                xticklabels=SCORES, yticklabels=SCORES,
                cbar_kws={"label": "% of ratings", "shrink": 0.82},
                linewidths=0.4, linecolor="white")
    for k in range(5):
        ax.add_patch(Rectangle((k, k), 1, 1, fill=False, edgecolor=EXACT,
                               lw=2.4, zorder=5))

    ax.set_xlabel("Co-rater's score for the same article", fontsize=11)
    ax.set_ylabel("Annotator's score", fontsize=11)
    ax.tick_params(labelrotation=0)
    fig.tight_layout()
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = hd.load()
    pairs = hd.rater_pairs(df)
    cm = symmetric_matrix(pairs)

    paths = [FIG_DIR / f"human_rater_pair_heatmap.{ext}" for ext in ("png", "pdf")]
    render(cm, paths)

    row = {
        "n_articles": len(pairs),
        "n_ratings": int(cm.sum()),
        "exact_agree_pct": round((pairs["abs_diff"] == 0).mean() * 100, 1),
        "adjacent_agree_pct": round((pairs["abs_diff"] <= 1).mean() * 100, 1),
        "mean_abs_diff": round(pairs["abs_diff"].mean(), 3),
        "n_pairs_opposite_sign": int(((pairs["lo"] < 0) & (pairs["hi"] > 0)).sum()),
    }
    summary_cols = list(row)
    for i, a in enumerate(SCORES):
        for j, b in enumerate(SCORES):
            row[f"n_r{a}_c{b}"] = int(cm[i, j])
    table = pd.DataFrame([row])
    table_path = ANALYSIS_DIR / "human_rater_pair_heatmap.csv"
    table.to_csv(table_path, index=False)

    print(f"\nwrote {paths[0]}\nwrote {paths[1]}\nwrote {table_path}\n")
    print(table[summary_cols].to_string(index=False))
    print("\nmatrix (rows = annotator, cols = co-rater)")
    print(pd.DataFrame(cm, index=SCORES, columns=SCORES).to_string())


if __name__ == "__main__":
    main()
