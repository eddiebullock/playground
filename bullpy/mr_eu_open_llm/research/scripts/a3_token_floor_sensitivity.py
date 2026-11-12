"""
A3 on clips with at least 8 tokens across the face: the sensitivity DECISIONS 2026-11-04 declared.

DECISIONS 2026-11-04, "Gate 1 is read as the median; clips under 8 are a sensitivity check": "In A2 and A3,
repeat each confirmatory analysis on clips at 8 or more, and report the share excluded." A2's analysis
does this itself; A3's frozen summary did not. Found missing by the deviations table of 2026-11-12
(research/paper/v2/supplementary_table_decisions.md) and computed here, after the A3 result was read,
with the frozen functions only, as the declared sensitivity. It cannot change the A3 verdicts.

Steps, each of which must pass before the next:
1. The locked A3 cells reproduce the committed summary (`face_input_validity_summary.json`): the frozen
   `summarise_model` and `decide` on all items give the same delta, p and Holm p for every model.
2. The clip-level token count is gate 1's own: for each model's `single_face`, `across` is the median of
   `tokens_across_view` over the delivery file's checked items, a clip's count is across times its face
   width over its face-box width (the delivery step's formula), and the share at 8 or more equals the
   delivery file's `gate_1_share_clips_at_least_8`.
3. Clips under the floor are those A2's frozen `below_token_floor` returns (no geometry counts as under).
   The frozen `summarise_model` and `decide` (Holm over the three models) then run on the remaining items.

Writes research/reproduction/a3_token_floor_sensitivity.json with every input's sha256.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scripts import face_input_validity as a3  # noqa: E402  (frozen)
from scripts import analyse_stimulus_interventions as an  # noqa: E402  (frozen; below_token_floor)
from scripts import run_stimulus_interventions as run  # noqa: E402  (frozen; geometry pin, MIN_TOKENS_ACROSS_FACE)

REPRO = ROOT / "research" / "reproduction"
ROUTE = "single_face"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--geometry", type=Path, default=ROOT / "results/stimulus_interventions/geometry.json")
    ap.add_argument("--cells_dir", type=Path, default=ROOT / "results/face_input_validity")
    ap.add_argument("--lookup", type=Path, default=ROOT / "data/eu_emotions_human_entropy.json")
    ap.add_argument("--output", type=Path, default=REPRO / "a3_token_floor_sensitivity.json")
    args = ap.parse_args(argv)
    if sha256(args.geometry) != run.GEOMETRY_SHA256:
        raise SystemExit("geometry.json does not match A2's pin.")
    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))["items"]
    human = json.loads(args.lookup.read_text(encoding="utf-8"))["trials"]
    committed_path = REPRO / "face_input_validity_summary.json"
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    inputs = {str(p.relative_to(ROOT)) if ROOT in p.parents else str(p): sha256(p)
              for p in [committed_path, args.lookup] + [args.cells_dir / f"cells_{m}.jsonl" for m in a3.MODELS]}
    inputs["results/stimulus_interventions/geometry.json"] = sha256(args.geometry)

    rows = {m: a3.read_cells(args.cells_dir / f"cells_{m}.jsonl")["rows"] for m in a3.MODELS}
    primary = a3.decide({m: a3.summarise_model(rows[m], human) for m in a3.MODELS})
    for m in a3.MODELS:
        for key in ("delta_real_minus_mismatch", "p_delta_actor_sign_flip", "p_delta_holm"):
            if not close(primary[m][key], committed[m][key]):
                raise SystemExit(f"Step 1 failed: {m} {key} {primary[m][key]} against the committed {committed[m][key]}.")

    out = {"description": __doc__.strip().splitlines()[0], "route": ROUTE, "floor": run.MIN_TOKENS_ACROSS_FACE,
           "primary_reproduced": True, "models": {}}
    below = {}
    for m in a3.MODELS:
        delivery_path = REPRO / f"input_delivery_{m}.json"
        inputs[str(delivery_path.relative_to(ROOT))] = sha256(delivery_path)
        delivery = json.loads(delivery_path.read_text(encoding="utf-8"))
        checked = [i[ROUTE] for i in delivery["items"] if "failed" not in i.get(ROUTE, {"failed": 1})]
        across = float(np.median([c["tokens_across_view"] for c in checked]))
        usable = [r for r in geometry.values() if r.get("face_width_px") and r.get("face_box")]
        faces = [across * r["face_width_px"] / (r["face_box"][2] - r["face_box"][0]) for r in usable]
        share = round(float(np.mean(np.asarray(faces) >= run.MIN_TOKENS_ACROSS_FACE)), 4)
        reported = delivery["verdicts"][ROUTE]["gate_1_share_clips_at_least_8"]
        if share != reported:
            raise SystemExit(f"Step 2 failed: {m} share {share} against the delivery file's {reported}.")
        trials = sorted({r["trial_id"] for r in rows[m]})
        below[m] = set(an.below_token_floor({t: {} for t in trials}, across, geometry))
        out["models"][m] = {"tokens_across_view_median": across, "gate_1_share_reproduced": share,
                            "n_items_a3": len(trials), "n_items_below_floor": len(below[m]),
                            "share_items_excluded": round(len(below[m]) / len(trials), 4),
                            "items_below_floor": sorted(below[m])}

    kept = a3.decide({m: a3.summarise_model([r for r in rows[m] if r["trial_id"] not in below[m]]) for m in a3.MODELS})
    for m in a3.MODELS:
        out["models"][m]["sensitivity"] = {k: kept[m][k] for k in (
            "n_items", "accuracy_real", "accuracy_mismatch", "delta_real_minus_mismatch", "delta_ci_actor_bootstrap",
            "p_delta_actor_sign_flip", "p_delta_holm", "vision_dependent")}
        out["models"][m]["primary_vision_dependent"] = committed[m]["vision_dependent"]
    out["inputs_sha256"] = inputs
    out["script_sha256"] = sha256(__file__)
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for m in a3.MODELS:
        s = out["models"][m]["sensitivity"]
        print(m, "excluded", out["models"][m]["n_items_below_floor"], "of", out["models"][m]["n_items_a3"],
              "| delta %.3f CI %s Holm p %.5f pass %s" % (s["delta_real_minus_mismatch"],
              [round(x, 3) for x in s["delta_ci_actor_bootstrap"]], s["p_delta_holm"], s["vision_dependent"]))


if __name__ == "__main__":
    main()
