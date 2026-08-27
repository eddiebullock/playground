#!/usr/bin/env python3
"""Per-item human response entropy from the EU-Emotion validation workbook (RQ1.1b target).

Reads the O'Reilly item-level validation data, computes Shannon entropy of the human
forced-choice response distribution per stimulus, and joins it to the trial_ids in
eu_emotions_full_manifest.json.

The workbook is 6AFC: target emotion + 4 named control emotions + "None of the above"
(labelled "I don't know" in the score columns). Two entropies are emitted per item:

  human_entropy          over all 6 options as presented to participants (primary)
  human_entropy_forced   over the 5 named emotions, renormalised, i.e. dropping the
                         "none of the above" mass, for comparison against a model that
                         is not offered an abstain option

The per-item option set is also emitted. This matters for RQ1.1b: trial_foils.py samples
foils pseudo-randomly, so a model scored against sampled foils is not answering the same
question as the human sample. Feed human_options back in as `candidate_labels` to make
the comparison like-for-like (resolve_candidate_labels honours pre-set options).

Usage (local Mac, workbook lives in OneDrive and is not synced to HPC):

  python -m scripts.human_entropy \\
    --xlsx "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/eu_emotions_oreilly_item_level.xlsx" \\
    --manifest data/eu_emotions_full_manifest.json \\
    --out data/eu_emotions_human_entropy.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SHEET_NAME = "EU-Emotion validation data"
FIRST_DATA_ROW = 5  # rows 1-4 are the merged multi-level header

# Column indices (0-based) in the validation sheet.
COL_MODALITY = 0
COL_EMOTION = 1
COL_INTENSITY = 2
COL_STIMULUS_ID = 3
COL_N_INTERVIEWS = 5
COL_RAW_FIRST = 8  # target, control 1-4, "I don't know"
COL_OPTION_LABEL_FIRST = 26  # target, control 1-4, "None of the above"
N_OPTIONS_HUMAN = 6

FACE_MODALITY = "Face"
INTENSITY_SUFFIXES = (" low intensity", " high intensity")

# Actor letter + F + item number + optional version. The version is written four ways
# across the two sources: PF19v1, SF9 v2, CF8(1), and bare-space "excitedOF7 1".
# The emotion prefix inside the identity string is unreliable (the workbook contains
# "frutrated" and "Dissapoint" typos), so the emotion comes from COL_EMOTION instead.
_STIM_CODE_RE = re.compile(r"([A-Za-z])F\s*(\d+)\s*(?:v\s*|\(\s*)?(\d+)?\s*\)?", re.IGNORECASE)


def canon(value: Any) -> str:
    """Lowercase and strip everything that varies between the two ID conventions."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def base_emotion_label(label: str) -> str:
    """'Angry Low Intensity' -> 'angry'; intensity lives in its own workbook column."""
    lowered = label.strip().lower()
    for suffix in INTENSITY_SUFFIXES:
        if lowered.endswith(suffix):
            return lowered[: -len(suffix)].strip()
    return lowered


def parse_stimulus_code(text: str) -> Optional[Tuple[str, int, Optional[int]]]:
    """Return (actor_letter, item_number, version) from a stimulus identity or filename."""
    match = _STIM_CODE_RE.search(str(text))
    if match is None:
        return None
    version = int(match.group(3)) if match.group(3) else None
    return match.group(1).lower(), int(match.group(2)), version


def stimulus_key(emotion: str, code: Tuple[str, int, Optional[int]], *, with_version: bool) -> str:
    letter, number, version = code
    key = f"{canon(emotion)}|{letter}|{number}"
    if with_version and version is not None:
        key = f"{key}|v{version}"
    return key


def shannon_entropy(probs: List[float], *, log_base: str = "e") -> float:
    total = sum(p for p in probs if p > 0)
    if total <= 0:
        return 0.0
    log = math.log if log_base == "e" else (lambda x: math.log(x, float(log_base)))
    return -sum((p / total) * log(p / total) for p in probs if p > 0)


def _cell_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def read_human_items(xlsx: Path, *, modality: str = FACE_MODALITY) -> List[Dict[str, Any]]:
    """One record per validated stimulus, with its response distribution and option set."""
    import openpyxl

    workbook = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    sheet = workbook[SHEET_NAME]

    items: List[Dict[str, Any]] = []
    for row in sheet.iter_rows(min_row=FIRST_DATA_ROW, values_only=True):
        if not row[COL_STIMULUS_ID]:
            continue
        if str(row[COL_MODALITY]).strip() != modality:
            continue

        options = [
            str(row[COL_OPTION_LABEL_FIRST + i]).strip()
            for i in range(N_OPTIONS_HUMAN)
            if row[COL_OPTION_LABEL_FIRST + i]
        ]
        proportions = [_cell_float(row[COL_RAW_FIRST + i]) for i in range(N_OPTIONS_HUMAN)]
        if len(options) != N_OPTIONS_HUMAN:
            continue

        code = parse_stimulus_code(row[COL_STIMULUS_ID])
        if code is None:
            continue

        emotion = str(row[COL_EMOTION]).strip()
        total = sum(proportions)
        distribution = {
            option: (value / total if total > 0 else 0.0)
            for option, value in zip(options, proportions)
        }
        # Last option is the abstain choice; drop it for the forced variant.
        named = proportions[: N_OPTIONS_HUMAN - 1]

        items.append(
            {
                "stimulus_id": str(row[COL_STIMULUS_ID]).strip(),
                "emotion": emotion,
                "intensity": str(row[COL_INTENSITY]).strip(),
                "code": code,
                "n_respondents": int(_cell_float(row[COL_N_INTERVIEWS])) or None,
                "human_options": options,
                "human_response_distribution": distribution,
                "human_entropy": shannon_entropy(proportions),
                "human_entropy_forced": shannon_entropy(named),
                "p_target": (proportions[0] / total) if total > 0 else 0.0,
                "p_abstain": (proportions[-1] / total) if total > 0 else 0.0,
                "raw_proportion_sum": total,
            }
        )
    return items


def index_human_items(
    items: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict], Dict[str, Dict], List[str]]:
    """
    Index by emotion+code, with and without the version suffix.

    Collisions in the versioned index mean two distinct stimuli parsed to the same key,
    which would silently assign one item's human distribution to the other trial, so
    they are returned for the caller to surface rather than dropped.
    """
    versioned: Dict[str, Dict[str, Any]] = {}
    unversioned: Dict[str, Dict[str, Any]] = {}
    collisions: List[str] = []
    for item in items:
        key = stimulus_key(item["emotion"], item["code"], with_version=True)
        if key in versioned:
            collisions.append(f"{key}: {versioned[key]['stimulus_id']} vs {item['stimulus_id']}")
        versioned.setdefault(key, item)
        unversioned.setdefault(stimulus_key(item["emotion"], item["code"], with_version=False), item)
    return versioned, unversioned, collisions


def match_trial(
    trial: Dict[str, Any],
    versioned: Dict[str, Dict[str, Any]],
    unversioned: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Match a manifest trial to a human item; returns (item, match_rule)."""
    stem = Path(str(trial["trial_id"])).stem.replace("_cut", "")
    code = parse_stimulus_code(stem)
    if code is None:
        return None, "no_stimulus_code"
    emotion = base_emotion_label(str(trial.get("label") or trial.get("correct_label") or ""))

    exact = versioned.get(stimulus_key(emotion, code, with_version=True))
    if exact is not None:
        return exact, "emotion_code_version"
    loose = unversioned.get(stimulus_key(emotion, code, with_version=False))
    if loose is not None:
        return loose, "emotion_code"
    return None, "unmatched"


def build_lookup(xlsx: Path, manifest: Path) -> Dict[str, Any]:
    trials = json.loads(manifest.read_text(encoding="utf-8"))["trials"]
    items = read_human_items(xlsx)
    versioned, unversioned, collisions = index_human_items(items)

    entries: Dict[str, Any] = {}
    unmatched: List[Dict[str, str]] = []
    rules: Dict[str, int] = {}
    for trial in trials:
        item, rule = match_trial(trial, versioned, unversioned)
        rules[rule] = rules.get(rule, 0) + 1
        if item is None:
            unmatched.append({"trial_id": trial["trial_id"], "label": trial.get("label", ""), "reason": rule})
            continue
        entries[trial["trial_id"]] = {
            # The human study collapses intensity: a "Disgusted Low Intensity" stimulus is
            # offered as "Disgusted". Scoring against the manifest label while showing the
            # human options would make every low-intensity item unanswerable.
            "human_target_label": item["human_options"][0],
            "manifest_label": trial.get("label"),
            "label_differs_from_manifest": item["human_options"][0] != trial.get("label"),
            "human_entropy": item["human_entropy"],
            "human_entropy_forced": item["human_entropy_forced"],
            "human_options": item["human_options"],
            "human_response_distribution": item["human_response_distribution"],
            "p_target": item["p_target"],
            "p_abstain": item["p_abstain"],
            "n_respondents": item["n_respondents"],
            "matched_stimulus_id": item["stimulus_id"],
            "match_rule": rule,
        }

    return {
        "source_workbook": xlsx.name,
        "manifest": manifest.name,
        "modality": FACE_MODALITY,
        "n_options_human": N_OPTIONS_HUMAN,
        "entropy_log_base": "e",
        "entropy_definition": (
            "Shannon entropy of the human forced-choice response distribution; "
            "human_entropy over all 6 presented options, human_entropy_forced over the "
            "5 named emotions renormalised (abstain mass removed)."
        ),
        "n_manifest_trials": len(trials),
        "n_matched": len(entries),
        "n_unmatched": len(unmatched),
        "n_human_items_available": len(items),
        "stimulus_key_collisions": collisions,
        "match_rule_counts": rules,
        "unmatched_trials": unmatched,
        "trials": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xlsx", required=True, type=Path, help="EU-Emotion item-level validation workbook")
    parser.add_argument("--manifest", default=Path("data/eu_emotions_full_manifest.json"), type=Path)
    parser.add_argument("--out", default=Path("data/eu_emotions_human_entropy.json"), type=Path)
    args = parser.parse_args()

    if not args.xlsx.exists():
        raise FileNotFoundError(f"Workbook not found (OneDrive hydrated?): {args.xlsx}")

    lookup = build_lookup(args.xlsx, args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(lookup, indent=2, sort_keys=True), encoding="utf-8")

    print(f"human items available : {lookup['n_human_items_available']} ({FACE_MODALITY} modality)")
    print(f"manifest trials       : {lookup['n_manifest_trials']}")
    print(f"matched               : {lookup['n_matched']} ({lookup['n_matched'] / lookup['n_manifest_trials']:.1%})")
    print(f"match rules           : {lookup['match_rule_counts']}")
    if lookup["stimulus_key_collisions"]:
        print(f"KEY COLLISIONS        : {len(lookup['stimulus_key_collisions'])} (ambiguous join, check these)")
        for row in lookup["stimulus_key_collisions"]:
            print(f"   {row}")
    if lookup["unmatched_trials"]:
        print(f"unmatched             : {lookup['n_unmatched']}")
        for row in lookup["unmatched_trials"]:
            print(f"   {row['label']:<28} {row['trial_id']}  [{row['reason']}]")
    values = [v["human_entropy"] for v in lookup["trials"].values()]
    if values:
        print(f"human entropy (nats)  : mean {sum(values) / len(values):.3f} | min {min(values):.3f} | max {max(values):.3f}")
    print(f"wrote                 : {args.out}")


if __name__ == "__main__":
    main()
