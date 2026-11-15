"""Build the task pool for the neutrality-rating survey.

Each *task* is one article that a human rater will (a) rate for bias on the same
Likert scale the LLMs used and (b) then compare against ONE randomly chosen LLM's
rating + explanation **for the same article**.

Input  : the LLM results spread across ``api/``, ``batch/`` and ``local/`` under a
          results root, collected and merged the same way as
          ``src/analysis/statistical_analysis/data_loader.py`` (one row per article
          x model), then joined to the leaders metadata for the full article text.
          (A single pre-merged CSV can still be passed with ``--input``.)
Output : tasks.csv  -- one row per task, consumed by build_survey.py
          tasks_preview.md -- human-readable preview (article text truncated)

Run:
    python survey/build_tasks.py --n 10 --seed 42
"""
from __future__ import annotations

import argparse
import logging
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Article-subject country column (from the leaders metadata).
COUNTRY_COL = "country_isocode"

# Mirrors src/analysis/statistical_analysis/config.py::LIKERT_MAPPING
LIKERT_MAPPING = {
    "strongly_biased_against": -2,
    "somewhat_biased_against": -1,
    "fully_neutral": 0,
    "somewhat_biased_in_favor": 1,
    "strongly_biased_in_favor": 2,
}
INT_TO_LABEL = {v: k for k, v in LIKERT_MAPPING.items()}
LIKERT_INTS = tuple(sorted(INT_TO_LABEL))  # (-2, -1, 0, 1, 2)

# Mirrors src/analysis/statistical_analysis/config.py::SOURCE_NAME_MAPPING
SOURCE_NAME_MAPPING = {
    "wikipedia": "wikipedia",
    "grokipedia_0_2": "grokipedia_0.2",
    "grokipedia_0_1": "grokipedia_0.1",
}

# Source value (canonical, post-mapping) -> article text column in the leaders CSV.
SOURCE_TO_TEXT_COL = {
    "wikipedia": "wikipedia",
    "grokipedia_0.2": "grokipedia_0_2",
    "grokipedia_0.1": "grokipedia_0_1",
}

# Disagreement among the LLMs for an article = std of their integer ratings (0 = full
# agreement). To get a *spread* of disagreement in the sample -- rather than mirroring
# the input (which is concentrated at low values) -- the observed article-disagreement
# range [0, max] is cut into equal-width bins and articles are drawn evenly across
# them. No disagreement value is forced; the bins just spread the selection out.
DEFAULT_DISAGREEMENT_BINS = 5

# Rater-fairness eligibility window: an article is only selectable if it is
# within [MIN_WORDS, MAX_WORDS] (a 6-fold length band, bounding the estimated
# task at roughly 11-36 minutes; ~51% of Wikipedia and ~75% of Grokipedia
# articles eligible). This deliberately excludes stubs and very long (typically
# very prominent) articles; disclose the cap in the methods text.
MIN_WORDS = 1083
MAX_WORDS = 6500


def _bin_edges(values: "pd.Series | np.ndarray", n_bins: int) -> np.ndarray:
    """Equal-width bin edges over [0, max(values)] (with a sane fallback when flat)."""
    hi = float(np.max(values)) if len(values) else 1.0
    if hi <= 0:
        hi = 1.0
    return np.linspace(0.0, hi, n_bins + 1)


def _assign_bin(values: "pd.Series | np.ndarray", edges: np.ndarray) -> np.ndarray:
    """Map disagreement values to a 1..K bin index (right-edge inclusive, clipped)."""
    idx = np.digitize(np.asarray(values, dtype=float), edges[1:-1], right=False) + 1
    return np.clip(idx, 1, len(edges) - 1).astype(int)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_PATH = REPO_ROOT / "results" / "final2.0"
DEFAULT_LEADERS_PATH = REPO_ROOT / "data" / "articles" / "LeadersAndParties" / "leaders_processed.csv"
DEFAULT_EXPERIMENT = "ideology"
OUT_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Results discovery + merge.
#
# This mirrors src/analysis/statistical_analysis/data_loader.py so the survey
# task pool is built from the SAME api/ + batch/ + local/ results as the
# statistical analysis (rather than a single batch-only merged.csv). Kept as a
# self-contained copy so the survey tool depends only on pandas.
# --------------------------------------------------------------------------- #
_CANONICAL_COLUMNS = ["Name", "Source", "Model", "neutrality_assessment", "explanation"]

# Column renames applied when loading API/local results to align with batch format.
_API_COLUMN_RENAMES = {
    "Choice": "neutrality_assessment",
    "Explanation": "explanation",
    "name": "Name",
    "source": "Source",
    "model": "Model",
}


def _normalise_results(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Normalise a raw results DataFrame to canonical column names."""
    df = df.copy()
    rename_map = {old: new for old, new in _API_COLUMN_RENAMES.items() if old in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
    if "Model" not in df.columns:
        df["Model"] = model_name
    if "custom_id" in df.columns:
        df = df.drop(columns=["custom_id"])
    df["Source"] = df["Source"].replace(SOURCE_NAME_MAPPING)
    return df


def _load_results_csv(path: Path, model_name: str, label: str) -> Optional[pd.DataFrame]:
    """Read a single ``results.csv`` and normalise it, or return ``None``."""
    if not path.exists():
        return None
    df = _normalise_results(pd.read_csv(path), model_name)
    logger.info("Loaded %s: %s (%d rows)", label, model_name, len(df))
    return df


def load_model_results(experiment_path: Path, label: str) -> List[pd.DataFrame]:
    """Load ``{kind}/{experiment}/{model}/results.csv`` (used for api/ and local/)."""
    if not experiment_path.exists():
        return []
    dfs: List[pd.DataFrame] = []
    for model_dir in sorted(experiment_path.iterdir()):
        if not model_dir.is_dir():
            continue
        df = _load_results_csv(model_dir / "results.csv", model_name=model_dir.name, label=label)
        if df is not None:
            dfs.append(df)
    return dfs


def load_batch_model_results(batch_experiment_path: Path) -> List[pd.DataFrame]:
    """Load ``batch/{experiment}/{model}/{batch_id}/results.csv``."""
    if not batch_experiment_path.exists():
        return []
    dfs: List[pd.DataFrame] = []
    for model_dir in sorted(batch_experiment_path.iterdir()):
        if not model_dir.is_dir():
            continue
        for batch_id_dir in sorted(model_dir.iterdir()):
            if not batch_id_dir.is_dir():
                continue
            df = _load_results_csv(
                batch_id_dir / "results.csv",
                model_name=model_dir.name,
                label=f"batch results ({batch_id_dir.name})",
            )
            if df is not None:
                dfs.append(df)
    return dfs


# Cache for the combined api/ + batch/ + local/ frame. Concatenating the
# per-model results.csv files is pure I/O that yields the same frame every run,
# so it is written once to CACHE_DIRNAME/combined_{experiment}.csv and reused.
# The cache is NOT trusted blindly: it is rebuilt whenever any source
# results.csv is newer than it (so a newly added model run is never silently
# ignored) or its columns no longer look canonical. Pass use_cache=False (CLI:
# --no-results-cache) to force a rebuild.
CACHE_DIRNAME = "cache"
_CACHE_KEY_COLUMNS = ["Name", "Source", "Model", "neutrality_assessment"]


def results_cache_path(results_path: Path, experiment: str) -> Path:
    """Where the combined results frame for ``experiment`` is cached."""
    return results_path / CACHE_DIRNAME / f"combined_{experiment}.csv"


def _source_results_files(results_path: Path, experiment: str) -> List[Path]:
    """Every results.csv that ``discover_and_load_results`` would read."""
    return sorted(
        list((results_path / "api" / experiment).glob("*/results.csv"))
        + list((results_path / "batch" / experiment).glob("*/*/results.csv"))
        + list((results_path / "local" / experiment).glob("*/results.csv"))
    )


def _read_results_cache(cache_path: Path, sources: List[Path]) -> Optional[pd.DataFrame]:
    """Return the cached combined frame, or ``None`` if missing/stale/unusable."""
    if not cache_path.exists():
        return None
    cache_mtime = cache_path.stat().st_mtime
    newer = [p for p in sources if p.stat().st_mtime > cache_mtime]
    if newer:
        logger.info(
            "Results cache %s is stale (%d source file(s) newer, e.g. %s) -- rebuilding",
            cache_path, len(newer), newer[0],
        )
        return None
    try:
        df = pd.read_csv(cache_path)
    except Exception as exc:  # unreadable/truncated cache -> just rebuild it
        logger.warning("Could not read results cache %s (%s) -- rebuilding", cache_path, exc)
        return None
    missing = [c for c in _CACHE_KEY_COLUMNS if c not in df.columns]
    if missing or df.empty:
        logger.warning(
            "Results cache %s looks wrong (missing %s, %d rows) -- rebuilding",
            cache_path, missing, len(df),
        )
        return None
    return df


def discover_and_load_results(
    results_path: Path, experiment: str, use_cache: bool = True
) -> pd.DataFrame:
    """Discover and merge results from the api/, batch/ and local/ subdirectories.

    The merged frame is cached at ``results_cache_path(results_path, experiment)``
    and reused on later runs unless a source results.csv has changed since; set
    ``use_cache=False`` to always re-merge (and not write the cache).
    """
    cache_path = results_cache_path(results_path, experiment)
    sources = _source_results_files(results_path, experiment)
    if use_cache:
        cached = _read_results_cache(cache_path, sources)
        if cached is not None:
            logger.info(
                "Loaded combined results from cache %s (%d rows, %d models)",
                cache_path, len(cached), cached["Model"].nunique(),
            )
            return cached

    all_dfs = (
        load_model_results(results_path / "api" / experiment, label="API results")
        + load_batch_model_results(results_path / "batch" / experiment)
        + load_model_results(results_path / "local" / experiment, label="local results")
    )
    if not all_dfs:
        raise FileNotFoundError(
            f"No results found under {results_path} for experiment '{experiment}'"
        )
    combined = pd.concat(all_dfs, ignore_index=True)
    keep = [c for c in _CANONICAL_COLUMNS if c in combined.columns]
    combined = combined[keep]
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".csv.tmp")  # write atomically
        combined.to_csv(tmp, index=False)
        tmp.replace(cache_path)
        logger.info(
            "Cached combined results -> %s (%d rows from %d file(s))",
            cache_path, len(combined), len(sources),
        )
    return combined


def load_merged_results(
    results_path: Path, leaders_path: Path, experiment: str, use_cache: bool = True
) -> pd.DataFrame:
    """Merge discovered LLM results with the leaders metadata (for article text).

    Returns one row per (article x model) with the canonical result columns plus
    the leaders columns (including the per-source article-text columns).
    """
    results = discover_and_load_results(results_path, experiment, use_cache=use_cache)
    leaders = pd.read_csv(leaders_path)
    merged = leaders.merge(results, left_on="name", right_on="Name", how="inner")
    logger.info(
        "Merged %d result rows with leaders -> %d rows (%d unique articles)",
        len(results), len(merged), merged[["Name", "Source"]].drop_duplicates().shape[0],
    )
    return merged


def _label_to_int(value) -> int:
    """The assessment may be stored as an int (-2..2) or as a text label."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return LIKERT_MAPPING[str(value).strip()]


def _allocate(weights: Dict[str, int], total: int) -> Dict[str, int]:
    """Largest-remainder (Hamilton) proportional allocation of ``total`` slots to
    categories by ``weights``, never giving a category more than its weight (the
    number of articles actually available for it)."""
    keys = list(weights)
    cap = {k: int(weights[k]) for k in keys}
    total = min(total, sum(cap.values()))
    if total <= 0:
        return {k: 0 for k in keys}
    w = np.array([weights[k] for k in keys], dtype=float)
    quota = {k: q for k, q in zip(keys, w / w.sum() * total)}
    alloc = {k: min(int(np.floor(quota[k])), cap[k]) for k in keys}
    # Distribute the leftover by descending fractional remainder, respecting caps.
    while sum(alloc.values()) < total:
        candidates = sorted(
            (k for k in keys if alloc[k] < cap[k]),
            key=lambda k: quota[k] - alloc[k],
            reverse=True,
        )
        if not candidates:
            break
        alloc[candidates[0]] += 1
    return alloc


def _pick_country_proportional(
    pool: pd.DataFrame, k: int, rng: np.random.RandomState
) -> List[int]:
    """Pick ``k`` row indices from ``pool`` with countries represented proportionally."""
    if k <= 0 or pool.empty:
        return []
    alloc = _allocate(pool["country"].value_counts().to_dict(), k)
    chosen: List[int] = []
    for country, quota in alloc.items():
        if quota > 0:
            cdf = pool[pool["country"] == country]
            chosen += cdf.sample(n=quota, random_state=rng).index.tolist()
    return chosen


def _pick_balanced_model(available: List[str], counts: Dict[str, int],
                         rng: np.random.RandomState) -> str:
    """Pick the least-used model among ``available`` (random tie-break), so the model
    shown across all tasks is as evenly distributed as possible."""
    fewest = min(counts.get(m, 0) for m in available)
    candidates = sorted(m for m in available if counts.get(m, 0) == fewest)
    pick = candidates[rng.randint(len(candidates))]
    counts[pick] = counts.get(pick, 0) + 1
    return pick


def _assign_models(task_models: List[Dict[str, int]], rng: np.random.RandomState) -> List[str]:
    """Choose which LLM's rating to show for each task.

    ``task_models[i]`` maps every model that rated task ``i`` to the Likert int it gave.
    First a coverage pass guarantees each Likert value is shown at least once -- handling
    the rarest-surfaceable values first and, among the options, taking the least-used
    model. The remaining tasks are then filled with the least-used available model, so the
    models stay as evenly distributed as the coverage constraint allows.
    """
    n = len(task_models)
    assigned: List[Optional[str]] = [None] * n
    counts: Dict[str, int] = {}
    providers = {v: [i for i, tm in enumerate(task_models) if v in tm.values()] for v in LIKERT_INTS}

    for v in sorted(LIKERT_INTS, key=lambda x: (len(providers[x]), x)):
        if any(assigned[i] is not None and task_models[i][assigned[i]] == v for i in range(n)):
            continue  # already shown by an earlier assignment
        free = [i for i in providers[v] if assigned[i] is None]
        if not free:
            if not providers[v]:
                logger.warning("Likert value %d (%s) is not shown: no selected task surfaces it.",
                               v, INT_TO_LABEL[v])
            continue
        best_i = best_m = None
        for i in free:
            for m in sorted(mm for mm, iv in task_models[i].items() if iv == v):
                if best_m is None or counts.get(m, 0) < counts.get(best_m, 0):
                    best_i, best_m = i, m
        assigned[best_i] = best_m
        counts[best_m] = counts.get(best_m, 0) + 1

    for i in range(n):
        if assigned[i] is None:
            assigned[i] = _pick_balanced_model(sorted(task_models[i]), counts, rng)
    return assigned  # type: ignore[return-value]


def _select_stratified(pool: pd.DataFrame, k: int, edges: np.ndarray,
                       rng: np.random.RandomState) -> pd.DataFrame:
    """Pick ``k`` rows from ``pool`` (which has ``country`` and ``disagreement`` columns)
    so their disagreement is spread evenly across the bins defined by ``edges``.

    Disagreement is the *primary* stratifier: ``k`` is split evenly across the K bins so
    the selection spans low-, mid- and high-disagreement items -- regardless of how the
    input is distributed. Within each bin, countries are represented proportionally. If a
    bin has fewer rows than its share, the shortfall is topped up from the other bins
    (still country-proportional). The returned frame keeps the assigned ``dbin`` column.
    """
    pool = pool.assign(dbin=_assign_bin(pool["disagreement"], edges))
    n_bins = len(edges) - 1

    # Even split; hand any remainder to the lower bins (the more common, easiest to fill).
    base = k // n_bins
    targets = {b: base for b in range(1, n_bins + 1)}
    for i in range(k - base * n_bins):
        targets[i + 1] += 1

    chosen: List[int] = []
    shortfall = 0
    for b in range(1, n_bins + 1):
        bin_pool = pool[pool["dbin"] == b]
        take = min(targets[b], len(bin_pool))
        shortfall += targets[b] - take
        chosen += _pick_country_proportional(bin_pool, take, rng)

    # Top up any shortfall (a bin smaller than its share) from the remaining rows.
    if shortfall > 0:
        remaining = pool.drop(index=chosen)
        chosen += _pick_country_proportional(remaining, shortfall, rng)

    return pool.loc[chosen]


def build_tasks(df: pd.DataFrame, n: int, seed: int, sources: list[str],
                n_bins: int = DEFAULT_DISAGREEMENT_BINS, reserve: int = 0) -> pd.DataFrame:
    """Build the task pool: ``n`` articles split evenly between the two sources, every
    article about a DIFFERENT person (so all ``n`` are independent units -- no
    within-person clustering).

    Within each source, articles are selected so countries are represented
    proportionally and the articles' LLM disagreement (std of the models' integer
    ratings) is spread evenly across ``n_bins`` equal-width bins from 0 to the maximum.
    Bin edges are computed over the eligible articles of BOTH sources, so bins are
    comparable across sources.

    ``reserve`` > 0 additionally draws that many wave-2 articles (taskids
    ``n+1..n+reserve``, ``wave`` column = 2) from the remaining pool in the SAME
    seeded run. Pre-committing the expansion before any responses exist is what
    keeps a later "add more articles" launch a fixed probability sample: wave 1
    alone is the standard design, and wave 1+2 pooled is one pre-registered draw
    (post-stratification weights are recomputed from the realized bin counts of
    whichever waves were fielded)."""
    required = {"Name", "Source", "Model", "neutrality_assessment", "explanation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"results are missing required columns: {sorted(missing)}")
    if COUNTRY_COL not in df.columns:
        raise ValueError(f"results are missing the country column {COUNTRY_COL!r} (needed for "
                         "proportional country representation).")
    if len(sources) != 2:
        raise ValueError(f"the design needs exactly 2 sources, got {sources!r}.")
    if n % 2:
        raise ValueError(f"--n must be even (articles are split 50/50 by source), got {n}.")

    df = df[df["Source"].isin(sources)].copy()
    df["llm_int"] = df["neutrality_assessment"].map(_label_to_int)
    df = df[df["llm_int"].isin(INT_TO_LABEL)]  # drop "error"/unmapped rows

    # Per-article (= Name x Source) disagreement among the LLMs: std of their integer
    # ratings (0 when they fully agree). Pivot to one column per source.
    article_dis = (
        df.groupby(["Name", "Source"])["llm_int"].agg(lambda s: float(s.std(ddof=0)))
        .unstack("Source")
    )
    country = df.groupby("Name")[COUNTRY_COL].first()
    lead = df.drop_duplicates("Name").set_index("Name")  # per-person article-text columns

    # One row per ARTICLE (Name x Source) with country, disagreement and length.
    arts = []
    for source in sources:
        for name, dis in article_dis[source].dropna().items():
            arts.append({
                "Name": name, "source": source, "country": country.get(name),
                "disagreement": float(dis),
                "n_words": len(str(lead.at[name, SOURCE_TO_TEXT_COL[source]]).split()),
            })
    articles = pd.DataFrame(arts)
    articles = articles[articles["country"].notna()].reset_index(drop=True)

    # Length-fairness filter: keep only articles inside the [MIN_WORDS, MAX_WORDS]
    # window (per ARTICLE -- the person's other article plays no role).
    before = len(articles)
    articles = articles[articles["n_words"].between(MIN_WORDS, MAX_WORDS)].reset_index(drop=True)
    logger.info("Length window %d..%d words: %d of %d articles eligible",
                MIN_WORDS, MAX_WORDS, len(articles), before)

    rng = np.random.RandomState(seed)
    edges = _bin_edges(articles["disagreement"], n_bins)

    # Per wave, per source: stratified selection. People already selected (in any
    # wave or source) are excluded, so no person appears twice (that would
    # reintroduce a within-person cluster). Wave 2's odd remainder goes to the
    # first source.
    per_wave = {1: [n // 2, n // 2]}
    if reserve:
        per_wave[2] = [reserve - reserve // 2, reserve // 2]

    # Each wave is selected AND laid out completely before the next wave touches
    # the RNG, so wave 1 is byte-identical to a reserve-free build of the same
    # seed (adding --reserve provably does not alter the fielded study). Wave 1
    # gets taskids 1..n, wave 2 n+1.. (wave-2 taskids only enter circulation when
    # the allocator's N is raised). Within each wave: choose which LLM rating to
    # show per task so that every surfaceable Likert value appears at least once
    # and the models stay as balanced as possible IN THAT WAVE (wave 1 must stand
    # alone if wave 2 never runs), then shuffle the task order (seeded,
    # reproducible) so taskids don't follow the source/bin layout: the allocator
    # hands out least-covered taskids in order, and without this the first raters
    # would get all-wikipedia articles walking the disagreement bins low to high.
    rows = []
    taken: set = set()
    for wave, quotas in sorted(per_wave.items()):
        picked = []
        for source, k in zip(sources, quotas):
            pool = articles[(articles["source"] == source) & ~articles["Name"].isin(taken)]
            if k > len(pool):
                raise ValueError(f"Wave {wave} requests {k} {source} articles but only "
                                 f"{len(pool)} eligible remain "
                                 f"(within {MIN_WORDS}..{MAX_WORDS} words).")
            sel = _select_stratified(pool, k, edges, rng)
            picked.append(sel)
            taken |= set(sel["Name"])
        selected = pd.concat(picked, ignore_index=True)

        task_specs = []  # (article row, sub, article_text)
        task_models: List[Dict[str, int]] = []
        for art in selected.itertuples(index=False):
            sub = df[(df["Name"] == art.Name) & (df["Source"] == art.source)]
            article_text = str(sub.iloc[0][SOURCE_TO_TEXT_COL[art.source]])
            task_specs.append((art, sub, article_text))
            task_models.append({m: int(i) for m, i in zip(sub["Model"], sub["llm_int"])})
        assigned = _assign_models(task_models, rng)
        order = rng.permutation(len(task_specs))
        task_specs = [task_specs[i] for i in order]
        assigned = [assigned[i] for i in order]

        for (art, sub, article_text), model in zip(task_specs, assigned):
            chosen = sub[sub["Model"] == model].iloc[0]
            llm_int = int(chosen["llm_int"])
            rows.append(
                {
                    "taskid": len(rows) + 1,
                    "wave": wave,
                    "name": art.Name,
                    "source": art.source,
                    "country": art.country,
                    "llm_disagreement": round(float(art.disagreement), 3),
                    # Consensus valence: mean of ALL models' ratings of this article
                    # (<0 = biased against the person, >0 = in favor).
                    "llm_mean": round(float(sub["llm_int"].mean()), 3),
                    "disagreement_bin": int(art.dbin),
                    "article_col": SOURCE_TO_TEXT_COL[art.source],
                    "n_words": len(article_text.split()),
                    "article_text": article_text,
                    "llm_model": chosen["Model"],
                    "llm_int": llm_int,
                    "llm_label": INT_TO_LABEL[llm_int],
                    "llm_explanation": str(chosen["explanation"]),
                }
            )

    return pd.DataFrame(rows)


def write_preview(tasks: pd.DataFrame, path: Path) -> None:
    lines = ["# Task pool preview", ""]
    for t in tasks.itertuples(index=False):
        lines += [
            f"## Task {t.taskid} [wave {getattr(t, 'wave', 1)}]: "
            f"{t.name} ({t.source}, {t.country}, {t.n_words} words)",
            f"- **LLM disagreement:** {t.llm_disagreement} (bin {t.disagreement_bin}); "
            f"consensus mean rating {t.llm_mean:+.2f}",
            f"- **LLM shown:** `{t.llm_model}` -> **{t.llm_label}** ({t.llm_int})",
            f"- **LLM explanation:** {t.llm_explanation}",
            "",
            "> " + textwrap.shorten(t.article_text, width=400, placeholder=" ..."),
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-path", type=Path, default=DEFAULT_RESULTS_PATH,
        help="results root containing api/ batch/ local/ (collected + merged like data_loader.py)",
    )
    ap.add_argument(
        "--leaders-path", type=Path, default=DEFAULT_LEADERS_PATH,
        help="leaders metadata CSV providing the per-source article text",
    )
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="experiment name under each kind/")
    ap.add_argument(
        "--input", type=Path, default=None,
        help="optional pre-merged results CSV; if given, used instead of discovering api/batch/local",
    )
    ap.add_argument("--n", type=int, default=10,
                    help="number of ARTICLES (even; split 50/50 by source, each about a different person)")
    ap.add_argument("--reserve", type=int, default=0,
                    help="wave-2 articles drawn NOW in the same run (taskids n+1..n+reserve); "
                         "field them later by swapping articles_full.json onto Drive and "
                         "raising the allocator's N")
    ap.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    ap.add_argument(
        "--sources",
        nargs="+",
        default=["wikipedia", "grokipedia_0.2"],
        help="the two Source values to sample from (exactly two, n/2 articles each)",
    )
    ap.add_argument("--disagreement-bins", type=int, default=DEFAULT_DISAGREEMENT_BINS,
                    help="equal-width bins over the disagreement range; articles are drawn evenly across them")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "tasks.csv")
    ap.add_argument(
        "--no-results-cache", action="store_true",
        help=f"always re-merge api/batch/local instead of using "
             f"{CACHE_DIRNAME}/combined_<experiment>.csv (the cache is otherwise "
             f"refreshed automatically when a source results.csv changes)",
    )
    args = ap.parse_args()

    if args.input is not None:
        df = pd.read_csv(args.input)
    else:
        df = load_merged_results(args.results_path, args.leaders_path, args.experiment,
                                 use_cache=not args.no_results_cache)

    tasks = build_tasks(df, args.n, args.seed, args.sources, args.disagreement_bins,
                        reserve=args.reserve)
    tasks.to_csv(args.out, index=False)
    write_preview(tasks, args.out.with_name("tasks_preview.md"))

    print(f"Wrote {len(tasks)} tasks (articles, one per person) -> {args.out}")
    if args.reserve:
        print(f"Waves: {tasks['wave'].value_counts().sort_index().to_dict()} "
              f"(wave 2 = pre-committed reserve, taskids {args.n + 1}..{args.n + args.reserve})")
    print(f"Preview            -> {args.out.with_name('tasks_preview.md')}")
    print("\nBy source:", tasks["source"].value_counts().to_dict())
    print("By country:", tasks["country"].value_counts().to_dict())
    print("By LLM model:", tasks["llm_model"].value_counts().to_dict())
    print("LLM label distribution:", tasks["llm_label"].value_counts().to_dict())
    bin_counts = tasks["disagreement_bin"].value_counts().sort_index()
    print("Disagreement spread (articles per bin):", bin_counts.to_dict(),
          f"| range {tasks.llm_disagreement.min()}..{tasks.llm_disagreement.max()}")
    print("Consensus valence (mean rating): "
          f"{(tasks.llm_mean < 0).sum()} negative, {(tasks.llm_mean == 0).sum()} neutral, "
          f"{(tasks.llm_mean > 0).sum()} positive")
    print(f"Article length (words): min={tasks.n_words.min()} "
          f"median={int(tasks.n_words.median())} max={tasks.n_words.max()}")
    avg_words = tasks.groupby("source")["n_words"].mean().round(0).astype(int)
    print("Avg words per encyclopedia:", avg_words.to_dict())

    run_article_pipeline()


def run_article_pipeline() -> None:
    """After writing tasks.csv: create the article set (explode_articles, reusing
    cleaned files from previous sets + MANIFEST) and build articles.json + .lss
    from it (build_survey). Article FORMATTING is deliberately NOT automatic --
    run `python survey/clean_articles.py` yourself for any fresh files it lists,
    then re-run build_survey.py. Deploy: replace articles.json on Drive (same
    file id) and re-import the .lss."""
    import subprocess
    import sys
    here = Path(__file__).resolve().parent
    for step in ("explode_articles.py", "build_survey.py"):
        print(f"\n=== {step} ===", flush=True)
        r = subprocess.run([sys.executable, "-u", str(here / step)])
        if r.returncode != 0:
            raise SystemExit(f"{step} failed (rc={r.returncode}); pipeline stopped.")
    print("\nDone. If the set contains fresh unformatted files (listed above), run "
          "`python survey/clean_articles.py` and then `python survey/build_survey.py` "
          "again before deploying (Drive articles.json + .lss import).")


if __name__ == "__main__":
    main()
