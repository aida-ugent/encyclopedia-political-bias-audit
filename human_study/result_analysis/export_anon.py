"""Write responses_tidy_anon.csv: the analysis rows, minus the Prolific ids.

``responses_tidy.csv`` (written by agreement.py) holds every completed response
regardless of Prolific status, and carries the participant/study/session ids.
This script derives the shareable version: keep only submissions whose Prolific
status is APPROVED, and drop every column that joins a row back to a person --
the participant/study/session ids and the LimeSurvey ``response_id``.

Rows are then shuffled (seeded, so the output is reproducible). Without this the
Nth row of the anon file would still be the Nth APPROVED row of the tidy file,
which re-creates the link that dropping ``response_id`` was meant to break.

Caveat: ``submitdate`` survives, and at second resolution it is near-unique (138
distinct values over 140 rows). It joins to "Completed at" in the Prolific
export, so this file remains re-identifiable by anyone holding that export. Pass
--drop-submitdate for a version without it.

Usage: python export_anon.py [--apply-blacklist] [--drop-submitdate] [--dry-run]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import demographics as dg

HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE / "analyses_results" / "results-survey918273_analysis"
TIDY = ANALYSIS_DIR / "responses_tidy.csv"
ANON = ANALYSIS_DIR / "responses_tidy_anon.csv"

ID_COLUMNS = ["response_id", "prolific_pid", "prolific_study_id", "prolific_session_id"]
SHUFFLE_SEED = 20260731


def approved_pids() -> set[str]:
    """Participant ids with an APPROVED submission, blacklist NOT applied."""
    path = dg._resolve_path(None)
    if path is None:
        raise SystemExit("no Prolific demographic export found in survey_results/")
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    status = raw["Status"].astype(str).str.strip().str.upper()
    incl = {s.upper() for s in dg.INCLUDED_STATUSES}
    return set(raw.loc[status.isin(incl), "Participant id"].dropna())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--apply-blacklist",
        action="store_true",
        help="also drop demographics.BLACKLIST_PIDS, matching what the analyses use",
    )
    ap.add_argument(
        "--drop-submitdate",
        action="store_true",
        help="also drop submitdate, which is near-unique and joins to the Prolific export",
    )
    ap.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    args = ap.parse_args()

    tidy = pd.read_csv(TIDY, encoding="utf-8-sig")
    keep = approved_pids()
    if args.apply_blacklist:
        keep -= dg.BLACKLIST_PIDS

    drop_cols = list(ID_COLUMNS) + (["submitdate"] if args.drop_submitdate else [])
    mask = tidy["prolific_pid"].isin(keep)
    dropped = tidy.loc[~mask]
    anon = (tidy.loc[mask]
            .drop(columns=[c for c in drop_cols if c in tidy.columns])
            .sample(frac=1, random_state=SHUFFLE_SEED)
            .reset_index(drop=True))

    print(f"{TIDY.name}: {len(tidy)} completed responses")
    print(f"dropped {len(dropped)} non-APPROVED"
          f"{'/blacklisted' if args.apply_blacklist else ''} response(s)")
    print(f"dropped columns: {', '.join(drop_cols)}")
    print(f"shuffled rows (seed {SHUFFLE_SEED})")
    print(f"-> {len(anon)} rows, {len(anon.columns)} columns")

    per_article = anon.groupby("taskid").size().value_counts().sort_index()
    print("ratings per article:")
    for n, count in per_article.items():
        print(f"  {n} rating(s): {count} article(s)")

    if args.dry_run:
        print("\ndry run - file unchanged")
        return

    anon.to_csv(ANON, index=False)
    print(f"\nwrote {ANON.relative_to(HERE)}")


if __name__ == "__main__":
    main()
