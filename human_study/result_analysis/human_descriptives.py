"""Descriptive statistics for the human neutrality ratings, plus an agreement figure.

Reads the de-identified analysis frame (responses_tidy_anon.csv -- APPROVED
submissions only, two ratings per article) and writes:

  human_descriptives/rating_distribution.csv   overall + per-source rating counts
  human_descriptives/rating_by_source.csv      centre/spread per source, article-level test
  human_descriptives/criteria_summary.csv      the seven Likert statements
  human_descriptives/sample_profile.csv        familiarity, politics, LLM-reasonableness
  human_descriptives/rater_pairs.csv           one row per article: the two ratings
  figures/human_agreement.png                  three-panel agreement figure

Ratings are on a 5-point bias scale (-2 strongly biased against .. +2 strongly
biased in favour); 0 is neutral, so the *sign* is a direction and the *magnitude*
is severity. Statements crit0-crit6 are agreement scales (-2 strongly disagree ..
+2 strongly agree) where higher = more neutral.

Usage: python human_descriptives.py [--no-figure]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE / "analyses_results" / "results-survey918273_analysis"
ANON = ANALYSIS_DIR / "responses_tidy_anon.csv"
TABLE_DIR = ANALYSIS_DIR / "human_descriptives"
FIG_DIR = ANALYSIS_DIR / "figures"

SOURCE_DISPLAY = {"wikipedia": "Wikipedia", "grokipedia_0.2": "Grokipedia"}
SCALE = [-2, -1, 0, 1, 2]
SCALE_LABELS = {-2: "Strongly\nagainst", -1: "Biased\nagainst", 0: "Neutral",
                1: "Biased\nin favour", 2: "Strongly\nin favour"}
CRITERIA = {
    "viewpoints_fair_int": "Presents viewpoints fairly",
    "avoids_sides_int": "Avoids taking sides",
    "factual_language_int": "Fact-focused language",
    "facts_vs_opinions_int": "Distinguishes facts/opinions",
    "no_fringe_exaggeration_int": "Avoids exaggerating fringe views",
    "no_contested_as_fact_int": "Avoids contested-as-fact",
    "impartial_tone_int": "Impartial tone",
}

# Validated against the dataviz six checks (light #fcfcfb and dark surfaces):
# adjacent CVD dE 24.7 protan, normal-vision dE 33.6, both >= 3:1 on surface.
BLUE, ORANGE = "#2a78d6", "#eb6834"
SOURCE_COLORS = {"Wikipedia": BLUE, "Grokipedia": ORANGE}
SEQ_CMAP = sns.light_palette(BLUE, as_cmap=True)   # single hue, light -> dark
INK, MUTED = "#0b0b0b", "#52514e"


def load() -> pd.DataFrame:
    df = pd.read_csv(ANON, encoding="utf-8-sig")
    df["source_label"] = df["source"].map(SOURCE_DISPLAY).fillna(df["source"])
    return df


def rater_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per article holding its two ratings.

    The two raters are interchangeable -- nothing distinguishes "rater 1" from
    "rater 2" -- so the pair is stored sorted (lo <= hi). An unsorted pair would
    make the joint distribution depend on row order, which is meaningless here.
    """
    rows = []
    for (taskid, name, source), g in df.groupby(["taskid", "article_name", "source_label"]):
        vals = sorted(g["human_rating_int"].dropna().astype(int))
        if len(vals) != 2:
            continue
        rows.append({"taskid": taskid, "article_name": name, "source": source,
                     "lo": vals[0], "hi": vals[1], "abs_diff": vals[1] - vals[0],
                     "mean_rating": np.mean(vals)})
    return pd.DataFrame(rows).sort_values("taskid").reset_index(drop=True)


def write_tables(df: pd.DataFrame, pairs: pd.DataFrame) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    dist = (df.groupby(["source_label", "human_rating_int"]).size()
              .unstack(fill_value=0).reindex(columns=SCALE, fill_value=0))
    dist.loc["All"] = dist.sum()
    dist_pct = dist.div(dist.sum(axis=1), axis=0).mul(100).round(1)
    dist_pct.columns = [f"pct_{c}" for c in dist_pct.columns]
    dist.columns = [f"n_{c}" for c in dist.columns]
    out = pd.concat([dist, dist_pct], axis=1)
    out.to_csv(TABLE_DIR / "rating_distribution.csv")

    # Rating-level SD/mean, but the group comparison is run on ARTICLE means:
    # the two ratings of an article are not independent, and each article
    # belongs to exactly one source, so the article is the unit of analysis.
    per_source = df.groupby("source_label")["human_rating_int"].agg(
        n_ratings="count", mean="mean", sd="std", median="median")
    art = pairs.groupby("source")["mean_rating"]
    per_source["n_articles"] = art.count()
    per_source["article_mean"] = art.mean()
    per_source["article_sd"] = art.std()
    per_source["pct_nonneutral"] = (df.assign(nn=df["human_rating_int"] != 0)
                                    .groupby("source_label")["nn"].mean().mul(100))
    wiki = pairs.loc[pairs["source"] == "Wikipedia", "mean_rating"]
    grok = pairs.loc[pairs["source"] == "Grokipedia", "mean_rating"]
    u, p = stats.mannwhitneyu(wiki, grok, alternative="two-sided")
    per_source["mannwhitney_u_articles"] = u
    per_source["mannwhitney_p_articles"] = p
    per_source.round(3).to_csv(TABLE_DIR / "rating_by_source.csv")

    crit = []
    for col, label in CRITERIA.items():
        row = {"statement": label, "column": col,
               "mean": df[col].mean(), "sd": df[col].std(),
               "pct_agree": (df[col] > 0).mean() * 100,
               "pct_disagree": (df[col] < 0).mean() * 100}
        for src, g in df.groupby("source_label"):
            row[f"mean_{src.lower()}"] = g[col].mean()
        # rater-pair gap on this statement, same articles as the headline rating
        gaps = (df.groupby("taskid")[col]
                  .apply(lambda s: s.max() - s.min() if s.notna().sum() == 2 else np.nan))
        row["mean_rater_gap"] = gaps.mean()
        crit.append(row)
    pd.DataFrame(crit).round(3).to_csv(TABLE_DIR / "criteria_summary.csv", index=False)

    profile = []
    for col in ["knew_subject", "llm_reasonable"]:
        vc = df[col].value_counts(dropna=False)
        for k, v in vc.items():
            profile.append({"variable": col, "value": k, "n": v,
                            "pct": round(v / len(df) * 100, 1)})
    pol = df["econ_leftright"].dropna()
    profile.append({"variable": "econ_leftright", "value": "mean", "n": len(pol),
                    "pct": round(pol.mean(), 2)})
    profile.append({"variable": "econ_leftright", "value": "median", "n": len(pol),
                    "pct": float(pol.median())})
    profile.append({"variable": "econ_leftright", "value": "missing", "n": int(df["econ_leftright"].isna().sum()),
                    "pct": round(df["econ_leftright"].isna().mean() * 100, 1)})
    pd.DataFrame(profile).to_csv(TABLE_DIR / "sample_profile.csv", index=False)

    pairs.to_csv(TABLE_DIR / "rater_pairs.csv", index=False)
    print(f"wrote 5 tables -> {TABLE_DIR.relative_to(HERE)}")


def figure(df: pd.DataFrame, pairs: pd.DataFrame) -> None:
    """Three panels on one question: do the two raters of an article agree?"""
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 9, "axes.labelcolor": INK,
                         "text.color": INK, "axes.edgecolor": "#d8d7d3"})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1),
                             gridspec_kw={"width_ratios": [1.15, 0.85, 1.1]})

    # (a) joint distribution of the two ratings. Sequential single hue = counts.
    ax = axes[0]
    mat = (pairs.groupby(["hi", "lo"]).size()
           .unstack(fill_value=0).reindex(index=SCALE[::-1], columns=SCALE, fill_value=0))
    # Cells where lo > hi cannot exist once the pair is sorted. Drawn as "0"
    # they read as observed-and-empty, so blank them out instead.
    impossible = pd.DataFrame({c: [c > r for r in mat.index] for c in mat.columns},
                              index=mat.index)
    sns.heatmap(mat, ax=ax, cmap=SEQ_CMAP, annot=True, fmt="d", cbar=False,
                mask=impossible, linewidths=2, linecolor="#fcfcfb", square=True,
                annot_kws={"fontsize": 9, "color": INK})
    ax.set_xlabel("Lower of the two ratings", color=MUTED)
    ax.set_ylabel("Higher of the two ratings", color=MUTED)
    ax.set_xticklabels([SCALE_LABELS[v].replace("\n", " ") for v in SCALE],
                       rotation=35, ha="right", fontsize=7.5, color=MUTED)
    ax.set_yticklabels([SCALE_LABELS[v].replace("\n", " ") for v in SCALE[::-1]],
                       rotation=0, fontsize=7.5, color=MUTED)
    # The diagonal is exact agreement -- ring it so it reads without counting.
    for i, v in enumerate(SCALE[::-1]):
        ax.add_patch(plt.Rectangle((SCALE.index(v), i), 1, 1, fill=False,
                                   edgecolor=ORANGE, linewidth=2.2, zorder=5))
    ax.set_title("(a) The two ratings per article\n"
                 f"ringed cells = exact agreement ({(pairs.abs_diff == 0).mean() * 100:.0f}%)",
                 fontsize=9.5, color=INK, pad=10)

    # (b) how far apart the pair is
    ax = axes[1]
    gap = pairs["abs_diff"].value_counts().reindex(range(5), fill_value=0)
    gap_pct = gap / gap.sum() * 100
    ax.bar(gap.index, gap_pct.values, color=BLUE, width=0.62, zorder=3)
    for x, pct, n in zip(gap.index, gap_pct.values, gap.values):
        if n:
            ax.text(x, pct + 1.4, f"{pct:.0f}%\n(n={n})", ha="center",
                    fontsize=8, color=MUTED, linespacing=1.3)
    ax.set_xticks(range(5))
    ax.set_xlabel("Scale points between the two raters", color=MUTED)
    ax.set_ylabel("Share of articles (%)", color=MUTED)
    ax.set_ylim(0, max(gap_pct) * 1.32)
    ax.set_title(f"(b) Size of the disagreement\nmean gap {pairs.abs_diff.mean():.2f} points",
                 fontsize=9.5, color=INK, pad=10)
    ax.grid(axis="x", visible=False)

    # (c) is agreement better on some statements than on the headline rating?
    ax = axes[2]
    gaps = {}
    for col, label in CRITERIA.items():
        g = df.groupby("taskid")[col].apply(
            lambda s: s.max() - s.min() if s.notna().sum() == 2 else np.nan)
        gaps[label] = g.mean()
    gaps["Overall neutrality rating"] = pairs["abs_diff"].mean()
    ser = pd.Series(gaps).sort_values()
    colors = [ORANGE if k == "Overall neutrality rating" else BLUE for k in ser.index]
    ax.barh(range(len(ser)), ser.values, color=colors, height=0.66, zorder=3)
    for i, v in enumerate(ser.values):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(ser)))
    ax.set_yticklabels(ser.index, fontsize=8, color=MUTED)
    ax.set_xlabel("Mean gap between the two raters (scale points)", color=MUTED)
    ax.set_xlim(0, max(ser.values) * 1.22)
    ax.set_title("(c) Agreement by question\nlower = raters closer together",
                 fontsize=9.5, color=INK, pad=10)
    ax.grid(axis="y", visible=False)

    fig.suptitle("How the two human raters of each article compare "
                 f"(n = {len(pairs)} articles, {len(df)} ratings)",
                 fontsize=11, color=INK, y=1.02)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "human_agreement.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    print(f"wrote {out.relative_to(HERE)}")


def report(df: pd.DataFrame, pairs: pd.DataFrame) -> None:
    print(f"\n{len(df)} ratings over {pairs.taskid.nunique()} articles, 2 per article\n")
    print("Neutrality rating (-2 biased against .. +2 biased in favour)")
    for v in SCALE:
        n = (df.human_rating_int == v).sum()
        print(f"  {v:+d} {SCALE_LABELS[v].replace(chr(10), ' '):<20} {n:>4}  {n / len(df) * 100:>5.1f}%  {'#' * round(n / len(df) * 60)}")
    print(f"  mean {df.human_rating_int.mean():+.3f}   SD {df.human_rating_int.std():.3f}   "
          f"median {df.human_rating_int.median():+.1f}")

    print("\nBy source (article-level means, n=35 each)")
    for src, g in pairs.groupby("source"):
        print(f"  {src:<12} mean {g.mean_rating.mean():+.3f}   SD {g.mean_rating.std():.3f}")
    wiki = pairs.loc[pairs["source"] == "Wikipedia", "mean_rating"]
    grok = pairs.loc[pairs["source"] == "Grokipedia", "mean_rating"]
    u, p = stats.mannwhitneyu(wiki, grok, alternative="two-sided")
    print(f"  Mann-Whitney U = {u:.1f}, p = {p:.4f}")

    print("\nRater agreement")
    print(f"  exact match      {(pairs.abs_diff == 0).mean() * 100:>5.1f}%")
    print(f"  within 1 point   {(pairs.abs_diff <= 1).mean() * 100:>5.1f}%")
    print(f"  mean gap         {pairs.abs_diff.mean():>5.2f} scale points")
    opposed = pairs[(pairs.lo < 0) & (pairs.hi > 0)]
    print(f"  opposite signs   {len(opposed):>5} articles ({len(opposed) / len(pairs) * 100:.1f}%)")
    print(f"  both non-neutral {len(pairs[(pairs.lo != 0) & (pairs.hi != 0)]):>5} articles")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    df = load()
    pairs = rater_pairs(df)
    write_tables(df, pairs)
    if not args.no_figure:
        figure(df, pairs)
    report(df, pairs)


if __name__ == "__main__":
    main()
