"""Aggregate audit. Prints counts and means only. Never prints a participant row."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from scog.data.card import load_card, parse_eyes_itemised, prepare_card
from scog.data.channel4 import clean_channel4, load_channel4
from scog.data.eu_emotion import (
    VIDEO_EXTENSIONS,
    load_eu_emotion,
    match_videos,
    stimulus_table,
    summarise_stimuli,
)
from scog.data.ybt import TEXT_TO_CODE, load_ybt
from scog.metrics.divergence import js_divergence
from scog.scoring.short_forms import (
    AQ10_ITEMS,
    EQ10_ITEMS,
    SPQ10_ITEMS,
    SQR10_ITEMS,
    texts_match,
)

MIN_CELL = 10


def _rel(got: float, expected: float) -> dict:
    scale = abs(expected) if expected != 0 else 1.0
    rel = abs(got - expected) / scale
    return {
        "got": got,
        "expected": expected,
        "relative_diff": rel,
        "mismatch_over_1pct": bool(rel > 0.01),
    }


def _cell_counts(series: pd.Series) -> dict:
    counts = series.fillna("(missing)").astype(str).value_counts()
    shown = {str(k): int(v) for k, v in counts.items() if int(v) >= MIN_CELL}
    hidden = int(counts[counts < MIN_CELL].sum()) if (counts < MIN_CELL).any() else 0
    return {"cells_n_at_least_10": shown, "n_in_cells_below_10": hidden}


def audit_channel4(path: Path) -> dict:
    raw = load_channel4(path)
    cleaned, audit = clean_channel4(raw)
    primary_autistic = cleaned["autistic"] & ~cleaned["ambiguous_subtype"]
    primary_control = cleaned["no_diagnosis"] & ~cleaned["ambiguous_subtype"]

    def aq_block(mask: pd.Series) -> dict:
        scores = cleaned.loc[mask, "aq10_binary"].dropna()
        if len(scores) == 0:
            raise RuntimeError("Channel 4 AQ block has zero rows")
        return {
            "n": int(len(scores)),
            "mean": float(scores.mean()),
            "sd": float(scores.std(ddof=1)),
            "pct_below_6": float((scores < 6).mean() * 100),
            "pct_at_or_above_6": float((scores >= 6).mean() * 100),
        }

    autistic = aq_block(primary_autistic)
    control = aq_block(primary_control)
    checks = {
        "n_autistic": _rel(audit["n_autistic"], 14700),
        "n_repeat": _rel(audit["n_repeat_dropped"], 26897),
        "n_identical": _rel(audit["n_identical_items_dropped"], 356),
        "n_age0": _rel(audit["n_age0_set_missing"], 1178),
        "n_age_over_90": _rel(audit["n_age_over90_set_missing"], 157),
        "n_ambiguous": _rel(audit["n_ambiguous"], 29167),
        "aq_autistic_mean": _rel(autistic["mean"], 6.95),
        "aq_autistic_sd": _rel(autistic["sd"], 2.35),
        "aq_control_mean": _rel(control["mean"], 3.06),
        "aq_control_sd": _rel(control["sd"], 2.15),
        "pct_autistic_below_6": _rel(autistic["pct_below_6"], 26.6),
        "pct_control_at_or_above_6": _rel(control["pct_at_or_above_6"], 14.1),
    }
    demo = cleaned.loc[primary_autistic | primary_control]
    return {
        "n_raw_rows": int(len(raw)),
        "exclusions": audit,
        "aq10_binary": {"autistic": autistic, "no_diagnosis": control},
        "checks": checks,
        "sex": _cell_counts(demo["sex"]),
        "age_mean": float(demo["age"].mean()),
        "age_n": int(demo["age"].notna().sum()),
    }


def _token_summary(series: pd.Series) -> dict:
    lengths: Counter = Counter()
    first: Counter = Counter()
    for raw in series.dropna():
        text = str(raw).strip()
        if not text:
            continue
        parts = [part.strip() for part in text.split(",")]
        lengths[len(parts)] += 1
        token = parts[0] if parts else ""
        if token in {"0", "1", "2", "3", "4", "-1"}:
            first[token] += 1
        else:
            first["other"] += 1
    return {
        "n": int(sum(lengths.values())),
        "length_counts": {str(k): int(v) for k, v in sorted(lengths.items())},
        "first_token_counts": {str(k): int(v) for k, v in sorted(first.items())},
    }


def audit_card(path: Path, *, seed: int = 20261029) -> dict:
    raw = load_card(path)
    work = prepare_card(raw)
    n_volunteers = int(work["VolunteerID"].nunique())
    tests = {str(k): int(v) for k, v in work["TestName"].value_counts().items()}
    itemised = work["Itemised Score"].fillna("").astype(str).str.strip().ne("")
    itemised_by_test = {
        str(name): int(itemised.loc[work["TestName"].eq(name)].sum())
        for name in work["TestName"].dropna().unique()
    }
    score = pd.to_numeric(work["Score"], errors="coerce")
    score_sd = {}
    for name, idx in work.groupby("TestName").groups.items():
        values = score.loc[idx].dropna()
        if len(values) >= MIN_CELL:
            score_sd[str(name)] = {"n": int(len(values)), "sd": float(values.std(ddof=1))}
    both = set(work.loc[work["TestName"].eq("AQ"), "VolunteerID"]) & set(
        work.loc[work["TestName"].eq("EQ"), "VolunteerID"]
    ) & set(work.loc[work["TestName"].eq("SQ"), "VolunteerID"])
    eyes_ids = set(work.loc[work["TestName"].eq("EyesTest"), "VolunteerID"])
    aq_ids = set(work.loc[work["TestName"].eq("AQ"), "VolunteerID"])
    eyes = work.loc[work["TestName"].eq("EyesTest")].copy()
    parsed_rows = []
    n_fail = 0
    n_header_mismatch = 0
    for raw_item, asc in zip(eyes["Itemised Score"], eyes["ASC diagnosis"]):
        try:
            parsed = parse_eyes_itemised(raw_item)
        except ValueError:
            n_fail += 1
            continue
        if parsed["header_n_correct"] is not None and not parsed["header_matches_key"]:
            n_header_mismatch += 1
        parsed_rows.append(
            {
                "asc": asc,
                "choices": parsed["choices"],
                "n_missing": parsed["n_missing"],
                "excluded": parsed["excluded_missing"],
            }
        )
    kept = [row for row in parsed_rows if not row["excluded"]]
    asc_counts = Counter(str(row["asc"]) for row in kept)
    # Group labels are whatever the export uses. Report cells n >= 10 only.
    choice_report = _eyes_js(kept, seed=seed)
    return {
        "n_rows": int(len(work)),
        "n_volunteers": n_volunteers,
        "test_counts": tests,
        "n_itemised_by_test": itemised_by_test,
        "score_sd_by_test": score_sd,
        "n_volunteers_aq_eq_sq": int(len(both)),
        "n_volunteers_eyes_and_aq": int(len(eyes_ids & aq_ids)),
        "sex": _cell_counts(work.drop_duplicates("VolunteerID")["Sex_norm"]),
        "eyes": {
            "n_parsed": len(parsed_rows),
            "n_parse_fail": n_fail,
            "n_header_mismatch": n_header_mismatch,
            "n_kept": len(kept),
            "asc_counts_n_at_least_10": {
                k: int(v) for k, v in asc_counts.items() if v >= MIN_CELL
            },
            "js": choice_report,
        },
        "aq_itemised_tokens": _token_summary(work.loc[work["TestName"].eq("AQ"), "Itemised Score"]),
        "checks": {
            "n_rows": _rel(int(len(work)), 69369),
            "n_volunteers": _rel(n_volunteers, 22800),
            "n_aq_eq_sq": _rel(int(len(both)), 15004),
            "n_eyes_and_aq": _rel(int(len(eyes_ids & aq_ids)), 2905),
        },
    }


def _eyes_js(rows: list[dict], *, seed: int) -> dict:
    if len(rows) < 20:
        raise RuntimeError("Eyes Test JS: fewer than 20 kept respondents")
    # Two most common ASC labels, if both have n >= 10.
    counts = Counter(str(row["asc"]) for row in rows)
    common = [label for label, n in counts.most_common() if n >= MIN_CELL]
    if len(common) < 2:
        return {"status": "fewer than two ASC labels with n >= 10"}
    label_a, label_b = common[0], common[1]
    def counts_for(label: str) -> np.ndarray:
        chosen = [row["choices"] for row in rows if str(row["asc"]) == label]
        mat = np.zeros((36, 4), dtype=float)
        for choices in chosen:
            for item, choice in enumerate(choices):
                if choice in (1, 2, 3, 4):
                    mat[item, int(choice) - 1] += 1
        return mat

    group_js = []
    a = counts_for(label_a)
    b = counts_for(label_b)
    for item in range(36):
        if a[item].sum() == 0 or b[item].sum() == 0:
            raise RuntimeError(f"Eyes item {item + 1} has an empty choice distribution")
        group_js.append(js_divergence(a[item], b[item]))
    # Random halves of the larger group.
    larger = label_a if counts[label_a] >= counts[label_b] else label_b
    pool = [row["choices"] for row in rows if str(row["asc"]) == larger]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))
    mid = len(pool) // 2
    def counts_from(subset) -> np.ndarray:
        mat = np.zeros((36, 4), dtype=float)
        for choices in subset:
            for item, choice in enumerate(choices):
                if choice in (1, 2, 3, 4):
                    mat[item, int(choice) - 1] += 1
        return mat
    left = counts_from([pool[i] for i in order[:mid]])
    right = counts_from([pool[i] for i in order[mid:]])
    half_js = [js_divergence(left[i], right[i]) for i in range(36)]
    return {
        "larger_group_label_is_first_common": True,
        "n_group_first": int(counts[label_a]),
        "n_group_second": int(counts[label_b]),
        "group_first": label_a,
        "group_second": label_b,
        "mean_js_groups": float(np.mean(group_js)),
        "mean_js_random_halves": float(np.mean(half_js)),
        "p95_js_random_halves": float(np.quantile(half_js, 0.95)),
        "checks": {
            "mean_js_groups": _rel(float(np.mean(group_js)), 0.0056),
            "mean_js_halves": _rel(float(np.mean(half_js)), 0.0015),
            "p95_js_halves": _rel(float(np.quantile(half_js, 0.95)), 0.0018),
        },
    }


def audit_ybt(path: Path) -> dict:
    df, questions = load_ybt(path)
    header = list(pd.read_csv(path, nrows=0).columns)
    # Question text is instrument wording. Compare published stems to those cells.
    question_by_name = {}
    # questions aligns to the raw header, before the Q311 rename. Re-read names.
    import csv
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        raw_header = next(reader)
        raw_questions = next(reader)
    for name, question in zip(raw_header, raw_questions):
        question_by_name.setdefault(name, question)
    finished = df["Finished"]
    if finished.dtype == object:
        n_finished = int(finished.astype(str).str.strip().isin(["1", "True", "TRUE", "true"]).sum())
    else:
        n_finished = int(pd.to_numeric(finished, errors="coerce").fillna(0).eq(1).sum())
    item_cols = (
        [f"aq_{i}" for i in range(1, 11)]
        + [f"eq10_{i}" for i in range(1, 11)]
        + [f"sq10_{i}" for i in range(1, 11)]
    )
    missing_items = [col for col in item_cols if col not in df.columns]
    if missing_items:
        raise RuntimeError(f"YourBrainType missing item columns: {missing_items}")
    unknown_labels: Counter = Counter()
    coded_cols = {}
    for col in item_cols:
        text = df[col].astype(str).str.strip().str.lower()
        mapped = text.map(TEXT_TO_CODE)
        missing = df[col].isna() | text.isin(["", "nan", "none"])
        unknown = ~missing & mapped.isna()
        if int(unknown.sum()):
            unknown_labels.update(text.loc[unknown].value_counts().to_dict())
        coded_cols[col] = mapped.mask(missing)
    coded = pd.DataFrame(coded_cols)
    n_complete = int(coded.notna().all(axis=1).sum())
    unknown_report = (
        {label: int(n) for label, n in unknown_labels.items() if n >= MIN_CELL}
        if len(unknown_labels) <= 8
        else {"n_unknown_labels": len(unknown_labels), "n_unknown_cells": int(sum(unknown_labels.values()))}
    )
    diagnosis = df["diagnosis"] if "diagnosis" in df.columns else pd.Series([""] * len(df))
    text = diagnosis.fillna("").astype(str)
    n_autism_mention = int(text.str.contains("autism", case=False).sum())
    order = {
        "aq": texts_match([question_by_name.get(f"aq_{i}", "") for i in range(1, 11)], AQ10_ITEMS),
        "eq": texts_match([question_by_name.get(f"eq10_{i}", "") for i in range(1, 11)], EQ10_ITEMS),
        "sqr": texts_match([question_by_name.get(f"sq10_{i}", "") for i in range(1, 11)], SQR10_ITEMS),
        "spq": texts_match(
            [question_by_name.get(f"spq_{i}", question_by_name.get(f"spq10_{i}", "")) for i in range(1, 11)],
            SPQ10_ITEMS,
        ),
    }
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_finished": n_finished,
        "n_complete_30": n_complete,
        "unknown_likert_labels": unknown_report,
        "n_autism_mention": n_autism_mention,
        "published_item_order_matches": order,
        "checks": {
            "n_rows": _rel(int(len(df)), 24203),
            "n_finished": _rel(n_finished, 4487),
            "n_complete_30": _rel(n_complete, 14167),
            "n_autism_mention": _rel(n_autism_mention, 758),
        },
    }


def audit_eu(path: Path, video_dir: Path) -> dict:
    raw = load_eu_emotion(path)
    table = stimulus_table(raw)
    summary = summarise_stimuli(table)
    videos = [p for p in video_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]
    matched = match_videos(table["stimulus_id"].tolist(), videos)
    summary["video_match"] = {
        "n_video_files": matched["n_video_files"],
        "n_matched": matched["n_matched"],
        "n_missing_stimuli": len(matched["missing_stimuli"]),
        "n_extra_files": len(matched["extra_files"]),
        "n_colliding_stems": len(matched["collisions"]),
        "missing_stimuli_sample": matched["missing_stimuli"][:15],
        "collision_count": len(matched["collisions"]),
    }
    summary["checks"] = {
        "n_stimuli": _rel(summary["n_stimuli"], 484),
        "mean_target": _rel(summary["mean_target_proportion"], 0.731),
        "mean_entropy": _rel(summary["mean_entropy_bits"], 1.01),
        "sd_entropy": _rel(summary["sd_entropy_bits"], 0.49),
        "n_modal_not_target": _rel(summary["n_modal_not_target"], 46),
    }
    return summary


def main() -> None:
    import os
    root = Path(os.environ["SCOG_DATA_ROOT"])
    video = Path(os.environ["SCOG_EU_VIDEO_DIR"])
    report = {
        "channel4": audit_channel4(root / "data_c4_raw.csv"),
        "card": audit_card(root / "CARD_Jul2026(Sheet1).csv"),
        "ybt": audit_ybt(root / "YBT.csv"),
        "eu_emotion": audit_eu(root / "eu_emotions_oreilly_item_level.xlsx", video),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
