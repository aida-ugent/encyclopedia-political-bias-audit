"""Participant demographics for the validation survey.

Links the Prolific demographic export (one row per submission, keyed by
``Participant id`` = the survey's ``prolific_pid``) to the tidy survey responses
and writes descriptive statistics -- plus exploratory demographic x outcome
relations -- as CSVs under a ``demographics/`` subdirectory of the analysis
folder.

Each participant rates exactly one article, so participant-level and
response-level are the same thing here; the demographic *distributions* are
de-duplicated by ``prolific_pid`` defensively.

Outputs (all under ``<analysis>/demographics/``):
  * coverage.csv            -- how many responses matched a demographic row
  * participant_profile.csv -- the merged per-participant table (demographics +
                               the response's key outcomes, incl. a ``flag_fast``
                               column); the base for the summaries below and for
                               any ad-hoc analysis
  * flagged_fast.csv        -- participants whose completion time is > FAST_SD
                               (default 3) SDs BELOW the mean -- a low-effort /
                               speeder check (empty when none qualify)
  * summary_categorical.csv -- per categorical variable x level: n and % of
                               participants (a ``<missing>`` level is included
                               when some rows lack the field, so % sums to 100)
  * summary_numeric.csv     -- per numeric variable (age, completion time): n,
                               mean, sd, min, p25, median, p75, max
  * outcomes_by_group.csv   -- EXPLORATORY. Per categorical variable x level:
                               n plus mean ideology, mean |human-shown LLM|,
                               exact-match %, within-1-point % and
                               reasonable-"yes" %. Tiny per-cell n -- purely
                               descriptive, not inferential.

Standalone:  python survey/result_analysis/demographics.py
(uses the same RESPONSES_FILE / DEMOGRAPHICS_FILE as agreement.py); normally it
is invoked automatically at the end of agreement.py's run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# Prolific export placed next to the LimeSurvey download. The filename Prolific
# generates is opaque; set it explicitly, or leave the glob to pick the newest.
DEMOGRAPHICS_FILE = "demographics.csv"

# Prolific submission statuses eligible for analysis (allowlist). Only APPROVED
# submissions are analysed; awaiting-review / rejected / returned / timed-out /
# active are all excluded. To also analyse not-yet-reviewed submissions, add
# "AWAITING REVIEW" here.
INCLUDED_STATUSES = {"APPROVED"}

# Manual blacklist: participant ids excluded from the analysis REGARDLESS of
# Prolific status (e.g. flagged low-effort/duplicate on inspection). Applied on
# top of INCLUDED_STATUSES, so these are dropped even when APPROVED.
# Supply the two original exclusions privately, or equivalent pseudonymous IDs.
import json
_EXCLUSIONS = HERE / "excluded_participants.json"
BLACKLIST_PIDS = set(json.loads(_EXCLUSIONS.read_text())) if _EXCLUSIONS.exists() else set()

# Values Prolific writes for returned/expired submissions instead of real data.
_SENTINELS = {"CONSENT_REVOKED", "DATA_EXPIRED", "Not Applicable", "NA", "N/A", ""}

# Prolific column -> tidy snake_case name. Only these are described.
CATEGORICAL = {
    "Sex": "sex",
    "Highest education level completed": "education",
    "Ethnicity simplified": "ethnicity",
    "Country of residence": "country_residence",
    "Country of birth": "country_birth",
    "Nationality": "nationality",
    "Student status": "student_status",
    "Employment status": "employment_status",
}
NUMERIC = {"Age": "age", "Time taken": "time_taken_s"}

# Speeder flag: participants whose completion time is this many SDs BELOW the
# mean (computed over the analysed set) are flagged as implausibly fast.
FAST_SD = 3.0
# Derived from "Fluent languages" (a comma-joined list -> too many unique combos
# to tabulate raw). ``multilingual`` is described as a category; ``n_fluent``
# as a numeric. The raw string is kept in the profile as ``fluent_languages``.
_DERIVED_CATEGORICAL = ["multilingual"]
_DERIVED_NUMERIC = ["n_fluent"]


def _clean(series: pd.Series) -> pd.Series:
    """Blank out Prolific's sentinel strings so they count as missing."""
    return series.astype(str).str.strip().where(
        lambda s: ~s.isin(_SENTINELS) & (s.str.lower() != "nan"), other=np.nan)


def _resolve_path(demographics_path: Path | None) -> Path | None:
    """The configured export, else the newest matching one in survey_results/."""
    if demographics_path is None:
        demographics_path = HERE / "survey_results" / DEMOGRAPHICS_FILE
    if demographics_path.exists():
        return demographics_path
    hits = sorted((HERE / "survey_results").glob("prolific_demographic_export_*.csv"))
    return hits[-1] if hits else None


def included_pids(demographics_path: Path | None = None) -> set[str] | None:
    """Prolific participant ids eligible for analysis: those with a submission
    whose Status is in ``INCLUDED_STATUSES`` (APPROVED only, by default), minus
    any id in ``BLACKLIST_PIDS`` (dropped regardless of status). Returns None if
    no export is found, so the caller skips filtering rather than silently
    dropping everyone."""
    path = _resolve_path(demographics_path)
    if path is None:
        return None
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    status = raw["Status"].astype(str).str.strip().str.upper()
    incl = {s.upper() for s in INCLUDED_STATUSES}
    return set(raw.loc[status.isin(incl), "Participant id"].dropna()) - BLACKLIST_PIDS


def load_demographics(path: Path) -> pd.DataFrame:
    """Read the Prolific export, keeping one clean row per participant.

    Returns a frame keyed by ``prolific_pid`` with the categorical/numeric
    fields renamed to snake_case; sentinel values become NaN. When a
    participant has several submission rows (e.g. a return then an approval),
    the row with the most non-missing fields is kept."""
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    raw = raw.rename(columns={"Participant id": "prolific_pid"})
    keep = ["prolific_pid", "Status", "Fluent languages", *CATEGORICAL, *NUMERIC]
    df = raw[[c for c in keep if c in raw.columns]].copy()
    for col in list(CATEGORICAL) + list(NUMERIC) + ["Status", "Fluent languages"]:
        if col in df.columns:
            df[col] = _clean(df[col])
    df = df.rename(columns={**CATEGORICAL, **NUMERIC, "Fluent languages": "fluent_languages"})
    for col in NUMERIC.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Derive language features from the comma-joined "Fluent languages" list.
    if "fluent_languages" in df.columns:
        toks = df["fluent_languages"].str.split(",")
        df["n_fluent"] = toks.map(lambda x: len([t for t in x if t.strip()]) if isinstance(x, list) else np.nan)
        df["multilingual"] = df["n_fluent"].map(
            lambda n: np.nan if pd.isna(n) else ("multilingual" if n > 1 else "monolingual"))
    # Prefer the most-complete row per participant.
    df["_filled"] = df.notna().sum(axis=1)
    df = (df.sort_values("_filled", ascending=False)
            .drop_duplicates("prolific_pid").drop(columns="_filled"))
    return df.reset_index(drop=True)


def _outcomes(resp: pd.DataFrame) -> pd.DataFrame:
    """Per-response outcome columns used for the demographic x outcome tables."""
    o = pd.DataFrame({"prolific_pid": resp["prolific_pid"]})
    h = resp["human_rating_int"].astype(float)
    m = resp["llm_rating_int"].astype(float)     # the rating actually shown
    o["ideology"] = pd.to_numeric(resp.get("econ_leftright"), errors="coerce")
    o["abs_diff"] = (h - m).abs()
    o["exact"] = (h == m).astype(float)
    o["within1"] = ((h - m).abs() <= 1).astype(float)
    o["reasonable_yes"] = (resp.get("llm_reasonable") == "yes").astype(float)
    return o


def _summary_categorical(profile: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    n_total = len(profile)
    rows = []
    for var in variables:
        counts = profile[var].value_counts(dropna=False)
        for level, n in counts.items():
            label = "<missing>" if (pd.isna(level) or level is np.nan) else level
            rows.append({"variable": var, "level": label, "n": int(n),
                         "pct": round(100 * n / n_total, 1) if n_total else np.nan})
    return pd.DataFrame(rows, columns=["variable", "level", "n", "pct"])


def _summary_numeric(profile: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    rows = []
    for var in variables:
        s = pd.to_numeric(profile[var], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append({"variable": var, "n": int(s.size),
                     "mean": round(s.mean(), 1), "sd": round(s.std(ddof=1), 1) if s.size > 1 else np.nan,
                     "min": round(s.min(), 1), "p25": round(s.quantile(.25), 1),
                     "median": round(s.median(), 1), "p75": round(s.quantile(.75), 1),
                     "max": round(s.max(), 1)})
    return pd.DataFrame(rows, columns=["variable", "n", "mean", "sd", "min",
                                       "p25", "median", "p75", "max"])


def _outcomes_by_group(merged: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """EXPLORATORY: mean outcomes within each demographic level (tiny cells)."""
    rows = []
    for var in variables:
        for level, g in merged.groupby(var, dropna=True):
            rows.append({
                "variable": var, "level": level, "n": len(g),
                "mean_ideology": round(g["ideology"].mean(), 2) if g["ideology"].notna().any() else None,
                "mean_abs_diff": round(g["abs_diff"].mean(), 2) if g["abs_diff"].notna().any() else None,
                "exact_pct": round(100 * g["exact"].mean(), 1) if g["exact"].notna().any() else None,
                "within1_pct": round(100 * g["within1"].mean(), 1) if g["within1"].notna().any() else None,
                "reasonable_yes_pct": round(100 * g["reasonable_yes"].mean(), 1) if g["reasonable_yes"].notna().any() else None,
            })
    return pd.DataFrame(rows, columns=["variable", "level", "n", "mean_ideology",
                                       "mean_abs_diff", "exact_pct", "within1_pct",
                                       "reasonable_yes_pct"])


def _flag_fast(profile: pd.DataFrame, n_sd: float = FAST_SD) -> tuple[pd.DataFrame, dict]:
    """Add a boolean ``flag_fast`` column: True when the participant's completion
    time is more than ``n_sd`` SDs BELOW the mean (a low-effort/speeder check).
    The mean and SD are computed over the participants in ``profile`` (the
    analysed set). Returns (profile-with-column, threshold-info)."""
    t = pd.to_numeric(profile.get("time_taken_s"), errors="coerce")
    info = {"n_sd": n_sd, "n_with_time": int(t.notna().sum()),
            "mean_s": None, "sd_s": None, "threshold_s": None, "n_flagged": 0}
    profile = profile.copy()
    if t.notna().sum() < 2:
        profile["flag_fast"] = False
        return profile, info
    mean, sd = t.mean(), t.std(ddof=1)
    thr = mean - n_sd * sd
    profile["flag_fast"] = (t < thr).fillna(False)
    info.update(mean_s=round(mean, 1), sd_s=round(sd, 1), threshold_s=round(thr, 1),
                n_flagged=int(profile["flag_fast"].sum()))
    return profile, info


def demographics_report(resp: pd.DataFrame, out_dir: Path,
                        demographics_path: Path | None = None) -> None:
    """Link demographics to responses and write the CSVs under ``out_dir/demographics``."""
    demographics_path = _resolve_path(demographics_path)
    if demographics_path is None:
        print("\n(no Prolific demographic export found -- demographics skipped; "
              "drop it in survey_results/ and set DEMOGRAPHICS_FILE)")
        return

    demo = load_demographics(demographics_path)
    resp = resp.dropna(subset=["prolific_pid"]).copy()
    outcomes = _outcomes(resp)

    dg = out_dir / "demographics"
    dg.mkdir(parents=True, exist_ok=True)

    # Coverage: every consenting response should map to a demographic row.
    resp_ids = set(resp["prolific_pid"])
    demo_ids = set(demo["prolific_pid"])
    matched = resp_ids & demo_ids
    unmatched = sorted(resp_ids - demo_ids)
    coverage = pd.DataFrame([{
        "n_responses": len(resp_ids),
        "n_demographic_rows": len(demo),
        "n_matched": len(matched),
        "n_unmatched_responses": len(unmatched),
        "unmatched_pids": ";".join(unmatched),
    }])
    coverage.to_csv(dg / "coverage.csv", index=False)

    # Per-participant profile = demographics + that participant's outcomes.
    profile = (resp[["prolific_pid", "taskid", "source", "llm_model"]]
               .merge(outcomes, on="prolific_pid", how="left")
               .merge(demo.drop(columns=[c for c in ["Status"] if c in demo.columns]),
                      on="prolific_pid", how="left"))
    profile = profile.drop_duplicates("prolific_pid").reset_index(drop=True)

    # Flag implausibly fast completions (> FAST_SD SDs below the mean time).
    profile, fast_info = _flag_fast(profile)
    profile.to_csv(dg / "participant_profile.csv", index=False)

    flagged = profile[profile["flag_fast"]][
        ["prolific_pid", "taskid", "time_taken_s"]].copy()
    flagged["mean_s"] = fast_info["mean_s"]
    flagged["sd_s"] = fast_info["sd_s"]
    flagged["threshold_s"] = fast_info["threshold_s"]
    flagged["n_sd_below"] = fast_info["n_sd"]
    flagged.to_csv(dg / "flagged_fast.csv", index=False)

    cat_vars = [v for v in list(CATEGORICAL.values()) + _DERIVED_CATEGORICAL
                if v in profile.columns]
    num_vars = [v for v in list(NUMERIC.values()) + _DERIVED_NUMERIC
                if v in profile.columns]

    _summary_categorical(profile, cat_vars).to_csv(dg / "summary_categorical.csv", index=False)
    _summary_numeric(profile, num_vars).to_csv(dg / "summary_numeric.csv", index=False)
    _outcomes_by_group(profile, cat_vars).to_csv(dg / "outcomes_by_group.csv", index=False)

    print(f"\nWrote demographics/ ({len(matched)}/{len(resp_ids)} responses matched "
          f"a demographic row) -> {dg}")
    if unmatched:
        print(f"  unmatched responses (no demographic row): {len(unmatched)}")
    if fast_info["threshold_s"] is not None:
        print(f"  fast-completion flag (< mean - {fast_info['n_sd']:g} SD = "
              f"{fast_info['threshold_s']:g}s): {fast_info['n_flagged']} flagged")
    num = _summary_numeric(profile, num_vars)
    if not num.empty:
        print(num.to_string(index=False))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE.parent))
    from agreement import RESPONSES_FILE, load_responses
    responses_file = HERE / "survey_results" / RESPONSES_FILE
    tasks = pd.read_csv(HERE.parent / "tasks.csv")
    out_dir = HERE / "analyses_results" / f"{responses_file.stem}_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    resp = load_responses(responses_file, tasks, out_dir)
    resp = resp.dropna(subset=["human_rating_int", "taskid"])
    inc = included_pids()
    if inc is not None:
        resp = resp[resp["prolific_pid"].isin(inc)].copy()
    demographics_report(resp, out_dir)
