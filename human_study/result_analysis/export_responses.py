"""Tidy LimeSurvey responses (helper for agreement.py -- not a CLI).

For every submitted response it figures out which article was shown (the hidden
``taskid``), pulls the human rating + explanation and the reasonableness
judgement, and joins the task metadata (article name, source, the LLM model/
label/explanation that was displayed) from tasks.csv. Input is the CSV
downloaded from the LimeSurvey admin UI (load_raw_csv handles both heading
styles); tidy() produces the analysis DataFrame.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import sys

HERE = Path(__file__).resolve().parent
SURVEY = HERE.parent
sys.path.insert(0, str(SURVEY))

REPO_ROOT = SURVEY.parent
PROMPTS_PATH = REPO_ROOT / "src" / "llm_experiments" / "experiments" / "prompts" / "neutrality_prompts.py"

LIKERT_CODE_TO_INT = {"SBA": -2, "SWA": -1, "NEU": 0, "SWF": 1, "SBF": 2}
REASONABLE_CODE_TO_TEXT = {"YES": "yes", "PART": "partially", "NO": "no"}
# Prior familiarity with the subject (mirrors FAMILIAR_CODES in build_survey.py).
FAMILIAR_CODE_TO_TEXT = {"FKNW": "knew", "FHRD": "heard_of", "FNO": "unknown"}
# Per-statement agreement scale (mirrors AGREE_CODES in build_survey.py).
AGREE_CODE_TO_INT = {"ASD": -2, "AD": -1, "AN": 0, "AA": 1, "ASA": 2}


def load_likert_question_keys() -> list[str]:
    """The per-statement Likert keys, in the same order build_survey.py renders them."""
    from build_survey import NEUTRALITY_STATEMENTS  # single source of truth
    return [key for key, _ in NEUTRALITY_STATEMENTS]


def tidy(raw: pd.DataFrame, tasks: pd.DataFrame) -> pd.DataFrame:
    tasks_idx = tasks.set_index("taskid")
    likert_keys = load_likert_question_keys()

    out = []
    for _, r in raw.iterrows():
        try:
            k = int(float(r.get("taskid")))
        except (TypeError, ValueError):
            continue  # unfinished / no task assigned
        if k not in tasks_idx.index:
            continue
        meta = tasks_idx.loc[k]
        # Generic question titles (one shared block for all tasks); the per-task
        # rate{k}/ok{k}/... titles are accepted as fallback for legacy responses.
        def get(col: str):
            v = r.get(col)
            return v if v is not None else r.get(f"{col}{k}")
        rate_code = get("rate")
        ok_code = get("ok")
        row = {
            "response_id": r.get("id"),
            "prolific_pid": r.get("ppid"),
            "prolific_study_id": r.get("studyid"),
            "prolific_session_id": r.get("sessionid"),
            "consent": r.get("consent"),
            "submitdate": r.get("submitdate"),
            "taskid": k,
            "article_name": meta["name"],
            "source": meta["source"],
            "human_rating_code": rate_code,
            "human_rating_int": LIKERT_CODE_TO_INT.get(rate_code),
            "human_explanation": get("expl"),
            "llm_model": meta["llm_model"],
            "llm_label": meta["llm_label"],
            "llm_rating_int": meta["llm_int"],
            "llm_explanation": meta["llm_explanation"],
            "llm_reasonable": REASONABLE_CODE_TO_TEXT.get(ok_code, ok_code),
            "llm_reasonable_explanation": get("why"),
            "knew_subject": FAMILIAR_CODE_TO_TEXT.get(get("known"), get("known")),
            # 0-10 left-right self-placement; empty = prefer not to say
            "econ_leftright": (int(str(get("econlr"))[1:])
                               if str(get("econlr") or "").startswith("L") else None),
        }
        # Per-statement agreement responses (crit{i} -> <key>_code / <key>_int).
        for i, key in enumerate(likert_keys):
            code = r.get(f"crit{i}")
            if code is None:
                code = r.get(f"crit{i}q{k}")
            row[f"{key}_code"] = code
            row[f"{key}_int"] = AGREE_CODE_TO_INT.get(code)
        out.append(row)
    df = pd.DataFrame(out)
    if not df.empty:
        df["human_vs_llm_diff"] = df["human_rating_int"] - df["llm_rating_int"]
    return df


def load_raw_csv(path: Path) -> pd.DataFrame:
    """Load a CSV downloaded from the LimeSurvey admin UI (Responses -> Export),
    for hosted instances without RemoteControl API access. Handles both heading
    styles: "question code" exports pass through directly; "full question text"
    exports are mapped back to codes by matching the known question texts (from
    build_survey.py), and answer label texts are mapped back to answer codes."""
    from build_survey import (AGREE_CODES, CONSENT_CODES, FAMILIAR_CODES,
                              LIKERT_CODES, NEUTRALITY_STATEMENTS, REASONABLE_CODES)
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "taskid" not in df.columns:      # full-question-text headings
        mapping = {}
        for c in df.columns:
            cl = str(c)
            if cl.startswith("Response ID"):
                mapping[c] = "id"
            elif cl.startswith("Date submitted"):
                mapping[c] = "submitdate"
            elif "Prolific participant" in cl:
                mapping[c] = "ppid"
            elif "Prolific study" in cl:
                mapping[c] = "studyid"
            elif "Prolific session" in cl:
                mapping[c] = "sessionid"
            elif "Assigned task" in cl:
                mapping[c] = "taskid"
            elif cl.startswith("Please read this before"):
                mapping[c] = "consent"
            elif cl.startswith("How would you rate the neutrality"):
                mapping[c] = "rate"
            elif cl.startswith("Briefly explain your neutrality"):
                mapping[c] = "expl"
            elif cl.startswith("Do you think this rating"):
                mapping[c] = "ok"
            elif "explain why you (dis)agree" in cl:
                mapping[c] = "why"
            elif "did you know of the person" in cl:
                mapping[c] = "known"
            elif cl.startswith("In political matters") or cl.startswith("In economic matters"):
                mapping[c] = "econlr"
            else:
                for i, (_key, stmt) in enumerate(NEUTRALITY_STATEMENTS):
                    if cl.strip().startswith(stmt[:40]):
                        mapping[c] = f"crit{i}"
        df = df.rename(columns=mapping)
        # answer label text -> answer code (rate labels are matched by their
        # leading label part, before the em-dash description)
        from build_survey import ECON_CODES
        text_maps = {
            "consent": {t: c for c, t in CONSENT_CODES},
            "econlr": {t: c for c, t in ECON_CODES},
            "ok": {t: c for c, t in REASONABLE_CODES},
            "known": {t: c for c, t in FAMILIAR_CODES},
        }
        for col, tmap in text_maps.items():
            if col in df.columns:
                df[col] = df[col].map(lambda v, m=tmap: m.get(str(v).strip(), v))
        agree = {t: c for c, t in AGREE_CODES}
        for i in range(len(NEUTRALITY_STATEMENTS)):
            col = f"crit{i}"
            if col in df.columns:
                df[col] = df[col].map(lambda v: agree.get(str(v).strip(), v))
        if "rate" in df.columns:
            def rate_code(v):
                s = str(v).strip().lower()
                for code, label, _ in LIKERT_CODES:
                    if s.startswith(label.replace("_", " ").lower()):
                        return code
                return v
            df["rate"] = df["rate"].map(rate_code)
    # completed responses only (matches the RC export's 'complete' filter)
    if "submitdate" in df.columns:
        df = df[df["submitdate"].notna() & (df["submitdate"].astype(str).str.strip() != "")]
    return df
