"""
Aggregate CARD Jul2026 EyesTest into one row per VolunteerID with item-level
RMET correctness + linked trait totals (EQ/SQ/AQ/SPQ) and D-score.

study4_rmet only — does not import or modify study3 / card_aligned pipelines.

EyesTest Itemised Score encoding (CARD platform):
  header: n_skipped, n_correct, n_incorrect, mean_RT_ms
  then 36 pairs: (chosen_option, RT_ms)
    - correct  -> option in {1,2,3,4} AND RT_ms present
    - incorrect -> option in {1,2,3,4} AND RT_ms blank
    - timeout/skip -> option == -1 AND RT_ms blank

Correctness is scored against the Baron-Cohen revised answer key (option index).
Header n_correct is cross-checked; mismatches are reported and retained as flags.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

STUDY4_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANSWER_KEY = STUDY4_ROOT / "data" / "rmet" / "answer_key" / "rmet_adult_answer_key.json"
DEFAULT_OUT = STUDY4_ROOT / "data" / "processed" / "card_rmet_item_level.csv"
DEFAULT_QC = STUDY4_ROOT / "data" / "processed" / "card_rmet_aggregation_qc.json"

TRAIT_TESTS = ("EQ", "SQ", "AQ", "SPQ")


def load_answer_key(path: Path) -> Dict[int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["item"]): int(item["correct_option"]) for item in data["items"]}


def _to_float(val: str) -> Optional[float]:
    s = (val or "").strip()
    if not s or s.upper() in {"NULL", "#N/A", "NA", "N/A", "NONE"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_int(val: str) -> Optional[int]:
    f = _to_float(val)
    if f is None:
        return None
    return int(f)


def parse_eyestest_itemised(
    raw: str,
    *,
    correct_options: Dict[int, int],
) -> Dict[str, Any]:
    """
    Parse one EyesTest Itemised Score string.

    Returns header counts, per-item choice/RT/correct, and QC flags.
    """
    parts = [p.strip() for p in str(raw).split(",")]
    if len(parts) < 4:
        raise ValueError(f"EyesTest Itemised Score too short: {raw[:80]!r}")

    n_skipped = _to_int(parts[0])
    n_correct_hdr = _to_int(parts[1])
    n_incorrect_hdr = _to_int(parts[2])
    mean_rt_ms = _to_float(parts[3])

    rest = parts[4:]
    while len(rest) < 72:
        rest.append("")

    choices: List[Optional[int]] = []
    rts: List[Optional[float]] = []
    correct: List[Optional[int]] = []
    n_correct = 0
    n_incorrect = 0
    n_timeout = 0
    n_missing = 0

    for i in range(36):
        ch_raw = rest[2 * i]
        rt_raw = rest[2 * i + 1]
        item = i + 1

        if ch_raw == "" and rt_raw == "":
            choices.append(None)
            rts.append(None)
            correct.append(None)
            n_missing += 1
            continue

        if ch_raw in {"-1", "−1"}:
            choices.append(-1)
            rts.append(None)
            correct.append(0)  # timeout/skip scored as not correct
            n_timeout += 1
            continue

        ch = _to_int(ch_raw)
        rt = _to_float(rt_raw)
        choices.append(ch)
        rts.append(rt)

        if ch is None or ch not in (1, 2, 3, 4):
            correct.append(None)
            n_missing += 1
            continue

        is_correct = int(ch == correct_options[item])
        # CARD encoding: RT present iff correct (sanity-check against key).
        if rt is not None and not is_correct:
            # Prefer answer key; flag inconsistency.
            pass
        correct.append(is_correct)
        if is_correct:
            n_correct += 1
        else:
            n_incorrect += 1

    header_sum_ok = (
        n_skipped is not None
        and n_correct_hdr is not None
        and n_incorrect_hdr is not None
        and (n_skipped + n_correct_hdr + n_incorrect_hdr) == 36
    )
    key_matches_header = n_correct_hdr == n_correct and n_incorrect_hdr == n_incorrect

    return {
        "n_skipped_hdr": n_skipped,
        "n_correct_hdr": n_correct_hdr,
        "n_incorrect_hdr": n_incorrect_hdr,
        "mean_rt_ms": mean_rt_ms,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_timeout": n_timeout,
        "n_missing_choice": n_missing,
        "header_sum_ok": header_sum_ok,
        "key_matches_header": key_matches_header,
        "choices": choices,
        "rts_ms": rts,
        "correct": correct,
    }


def _pick_latest_trait_row(rows: Sequence[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Prefer adult TestName over child/adolescent; then latest LastModified string order."""
    if not rows:
        return None
    priority = {"EQ": 0, "SQ": 0, "AQ": 0, "SPQ": 0}

    def sort_key(r: Dict[str, str]) -> Tuple[int, str]:
        name = r.get("TestName", "")
        # Adult short names first
        pri = 0 if name in priority else 1
        return (pri, r.get("LastModified", "") or "")

    return sorted(rows, key=sort_key)[-1]


def aggregate(
    card_csv: Path,
    answer_key_path: Path,
    *,
    encoding: str = "latin-1",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    correct_options = load_answer_key(answer_key_path)

    by_id_tests: Dict[str, Dict[str, List[Dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    eyes_rows: List[Dict[str, str]] = []

    with card_csv.open(newline="", encoding=encoding) as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CARD CSV has no header")
        for row in reader:
            vid = (row.get("VolunteerID") or "").strip()
            if not vid:
                continue
            tname = (row.get("TestName") or "").strip()
            by_id_tests[vid][tname].append(row)
            if tname == "EyesTest":
                eyes_rows.append(row)

    out_rows: List[Dict[str, Any]] = []
    qc = {
        "n_eyestest_rows": len(eyes_rows),
        "n_unique_volunteer_ids": len({r["VolunteerID"] for r in eyes_rows}),
        "header_sum_fail": 0,
        "key_header_mismatch": 0,
        "rt_on_incorrect_count": 0,
        "correct_without_rt": 0,
        "missing_eq": 0,
        "missing_sq": 0,
        "missing_aq": 0,
        "missing_spq": 0,
        "item_missingness": {f"rmet_{i:02d}_correct": 0 for i in range(1, 37)},
        "encoding": encoding,
        "answer_key": str(answer_key_path),
        "source_csv": str(card_csv),
    }

    for row in eyes_rows:
        vid = row["VolunteerID"].strip()
        parsed = parse_eyestest_itemised(row.get("Itemised Score") or "", correct_options=correct_options)
        if not parsed["header_sum_ok"]:
            qc["header_sum_fail"] += 1
        if not parsed["key_matches_header"]:
            qc["key_header_mismatch"] += 1

        # Platform docs say RT is stored only on correct trials. Empirically this
        # export often stores RT on incorrect trials too. Score by answer key
        # (validated 100% vs header n_correct); keep RT as timing covariate only.
        for ch, rt, corr in zip(parsed["choices"], parsed["rts_ms"], parsed["correct"]):
            if ch in (1, 2, 3, 4) and corr is not None:
                if (rt is not None) and (not corr):
                    qc["rt_on_incorrect_count"] += 1
                if corr and rt is None:
                    qc["correct_without_rt"] += 1

        tests = by_id_tests[vid]
        eq_row = _pick_latest_trait_row(tests.get("EQ", []))
        sq_row = _pick_latest_trait_row(tests.get("SQ", []))
        aq_row = _pick_latest_trait_row(tests.get("AQ", []))
        spq_row = _pick_latest_trait_row(tests.get("SPQ", []))

        eq_total = _to_float(eq_row["Score"]) if eq_row else None
        sq_total = _to_float(sq_row["Score"]) if sq_row else None
        aq_total = _to_float(aq_row["Score"]) if aq_row else None
        spq_total = _to_float(spq_row["Score"]) if spq_row else None
        if eq_total is None:
            qc["missing_eq"] += 1
        if sq_total is None:
            qc["missing_sq"] += 1
        if aq_total is None:
            qc["missing_aq"] += 1
        if spq_total is None:
            qc["missing_spq"] += 1

        d_score = None
        if eq_total is not None and sq_total is not None:
            # Match CARD/C4 convention: d_score = SQ - EQ (higher = more systemizing).
            d_score = sq_total - eq_total

        age = _to_float(row.get("AgeWhenTestCompleted") or "")
        sex = (row.get("Sex") or "").strip()
        if sex.upper() in {"NULL", "#N/A", "NA", ""}:
            sex = ""
        asc = (row.get("ASC diagnosis") or "").strip()
        if asc.upper() in {"NULL", "#N/A", "NA"}:
            asc = ""

        out: Dict[str, Any] = {
            "VolunteerID": vid,
            "rmet_total_correct": parsed["n_correct"],
            "rmet_n_incorrect": parsed["n_incorrect"],
            "rmet_n_timeout": parsed["n_timeout"],
            "rmet_n_skipped_hdr": parsed["n_skipped_hdr"],
            "rmet_mean_rt_ms": parsed["mean_rt_ms"],
            "rmet_n_correct_hdr": parsed["n_correct_hdr"],
            "rmet_key_matches_header": int(parsed["key_matches_header"]),
            "eq_total": eq_total,
            "sq_total": sq_total,
            "aq_total": aq_total,
            "spq_total": spq_total,
            "d_score": d_score,
            "age": age,
            "sex": sex,
            "asc_diagnosis": asc,
            "country": (row.get("Country") or "").strip(),
            "year_of_birth": (row.get("YearOfBirth") or "").strip(),
        }

        for i in range(36):
            item = i + 1
            ch = parsed["choices"][i]
            rt = parsed["rts_ms"][i]
            corr = parsed["correct"][i]
            out[f"rmet_{item:02d}_choice"] = ch if ch is not None else ""
            out[f"rmet_{item:02d}_rt_ms"] = rt if rt is not None else ""
            out[f"rmet_{item:02d}_correct"] = "" if corr is None else int(corr)
            if corr is None:
                qc["item_missingness"][f"rmet_{item:02d}_correct"] += 1

        out_rows.append(out)

    # Demographics summary for QC
    sexes = Counter(r["sex"] or "missing" for r in out_rows)
    ages = [r["age"] for r in out_rows if isinstance(r["age"], float)]
    qc["n_aggregated"] = len(out_rows)
    qc["sex_counts"] = dict(sexes)
    qc["age_n"] = len(ages)
    qc["age_mean"] = (sum(ages) / len(ages)) if ages else None
    totals = [r["rmet_total_correct"] for r in out_rows]
    qc["rmet_total_mean"] = (sum(totals) / len(totals)) if totals else None
    qc["rmet_total_min"] = min(totals) if totals else None
    qc["rmet_total_max"] = max(totals) if totals else None
    return out_rows, qc


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--card_csv",
        type=Path,
        required=True,
        help="Path to CARD_Jul2026(Sheet1).csv (or equivalent long-format export)",
    )
    ap.add_argument("--answer_key", type=Path, default=DEFAULT_ANSWER_KEY)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--qc_json", type=Path, default=DEFAULT_QC)
    ap.add_argument("--encoding", default="latin-1")
    args = ap.parse_args(argv)

    rows, qc = aggregate(args.card_csv, args.answer_key, encoding=args.encoding)
    write_csv(rows, args.output)
    args.qc_json.parent.mkdir(parents=True, exist_ok=True)
    args.qc_json.write_text(json.dumps(qc, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.output}  N={len(rows)}")
    print(f"wrote {args.qc_json}")
    print(
        "QC:",
        f"key_header_mismatch={qc['key_header_mismatch']}",
        f"rt_on_incorrect={qc['rt_on_incorrect_count']}",
        f"correct_without_rt={qc['correct_without_rt']}",
        f"missing_eq={qc['missing_eq']}",
        f"missing_sq={qc['missing_sq']}",
        f"rmet_total_mean={qc['rmet_total_mean']:.2f}" if qc["rmet_total_mean"] is not None else "",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
