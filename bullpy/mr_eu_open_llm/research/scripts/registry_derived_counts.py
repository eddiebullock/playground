#!/usr/bin/env python3
"""Print counts that the paper cites but no committed output file stores, from committed inputs.

Read-only: writes nothing, prints one JSON object. Each block names its input and rule.

1. V1 pilot breakdown (Methods, "V1 pilot, before the freeze"), from
   `research/reproduction/a4_v1_pilot.csv` (job 40260039, frozen sha256 c735615a). The headline
   113 of 144 uses the frozen rule in `scripts/a4_analyse.py` `detector_gate` and is recomputed
   with that function. The rest are descriptive counts on the same CSV:
   - intensity pairs whose reading at amplitude 1 minus amplitude .25 is at least .5, and how
     many of those also rise monotonically under the V1 rule;
   - AU12 readings at amplitude 1;
   - presence pairs that rise under the V1 rule, and how many of those have a presence score
     below 0 (negative logit) at amplitude 1.
2. Composite-grid frame loss (Methods, "Delivery audit"; DECISIONS 2026-11-04), from
   `research/reproduction/stimulus_features.json` `duration_s`: frames sampled are
   min(16, max(1, floor(duration))) at 1 frame per second (`scripts/frame_sampling.py`), and a
   clip loses frames when that exceeds the 8 cells the composite grid keeps. `duration_s` is
   stored to two decimals, so a clip within .01 s of 9 s (the 8-to-9 frame boundary) could
   round across it; those are counted separately.
3. Literature table (Results, "Literature check"; Figure 1e), from
   `research/reproduction/literature_table.csv`: the number of papers, and per coded column
   (target position, option sets, delivery check, face-dependence test) how many rows carry each
   code, plus how many rows have any cell marked as inferred. The codes are the table's own.

Usage (from bullpy/mr_eu_open_llm):
    PYTHONPATH=.:src python research/scripts/registry_derived_counts.py
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "research" / "reproduction"
GRID_CELLS = 8
FRAME_CAP = 16


def v1_pilot() -> dict:
    from scripts import a4_analyse

    rows = list(csv.DictReader((REPRO / "a4_v1_pilot.csv").open(encoding="utf-8")))
    gate = a4_analyse.detector_gate(rows)
    series = defaultdict(dict)
    kind = {}
    for r in rows:
        key = (r["au"], r["identity"])
        series[key][float(r["amplitude"])] = float(r["intensity"])
        kind[key] = r["readout"]
    rises = {k: v[0.25] <= v[0.5] <= v[1.0] and v[1.0] > v[0.25] for k, v in series.items()}
    intensity = [k for k in series if kind[k] == "intensity"]
    presence = [k for k in series if kind[k] == "presence"]
    big = [k for k in intensity if series[k][1.0] - series[k][0.25] >= 0.5]
    rising_presence = [k for k in presence if rises[k]]
    au12 = [series[k][1.0] for k in intensity if k[0].startswith("AU12")]
    return {
        "images": len(rows),
        "face_found": sum(int(r["face_found"]) for r in rows),
        "pairs": len(series),
        "pairs_rising_detector_gate": round(gate["share_rising"] * len(series)),
        "share_rising_detector_gate": gate["share_rising"],
        "v1_pass": gate["pass"],
        "intensity_pairs": len(intensity),
        "presence_pairs": len(presence),
        "intensity_pairs_gain_at_least_0p5": len(big),
        "of_which_monotonic_rise": sum(rises[k] for k in big),
        "columns_with_gain_at_least_0p5": sorted({k[0] for k in big}),
        "au12_at_amplitude_1_min_max": [round(min(au12), 3), round(max(au12), 3)],
        "presence_pairs_rising": len(rising_presence),
        "of_which_score_below_0_at_amplitude_1": sum(series[k][1.0] < 0 for k in rising_presence),
    }


def frame_loss() -> dict:
    data = json.loads((REPRO / "stimulus_features.json").read_text(encoding="utf-8"))
    trials = data["trials"]
    trials = trials if isinstance(trials, list) else list(trials.values())
    ok = [t for t in trials if t["status"] == "ok"]
    sampled = [min(FRAME_CAP, max(1, math.floor(t["duration_s"]))) for t in ok]
    boundary = GRID_CELLS + 1
    near = sum(abs(t["duration_s"] - boundary) <= 0.01 for t in ok)
    return {"clips": len(ok), "clips_losing_frames_to_grid": sum(n > GRID_CELLS for n in sampled),
            "clips_within_0p01_s_of_9_s": near}


LITERATURE_CODED = ("target_position_code", "option_sets_code", "delivery_check_code", "face_dependence_code")


def literature_table() -> dict:
    with (REPRO / "literature_table.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out: dict = {"papers": len(rows)}
    for col in LITERATURE_CODED:
        out[col] = dict(sorted(Counter(r[col] for r in rows).items()))
    out["rows_with_inferred_cells"] = sum(bool(r["inferred_cells"].strip()) for r in rows)
    return out


def main() -> int:
    print(json.dumps({"v1_pilot": v1_pilot(), "grid_frame_loss": frame_loss(),
                      "literature_table": literature_table()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
