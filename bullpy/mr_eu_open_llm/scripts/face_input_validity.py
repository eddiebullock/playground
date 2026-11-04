"""
Amendment A3: A1's design on the single_face route. See research/PREREG_A3_DRAFT.md.

Same 243 items, five rotations, real versus mismatch, same partners, same estimand and
decision rule, as its own Holm family over three models. A3 does not replace A1.

Scoring is fixed in advance as exact option probabilities: each option's joint logprob
after the prefill "EMOTION:", renormalised over the six options. An item is correct on a
rotation if the target has the highest probability, which is the limit of A1's modal
answer as draws grow. No sampling, so no seed enters scoring.

This script refuses to run a model until the amendment is frozen. Statistics are imported
from research/scripts/analyse_rotation.py, so the two amendments cannot drift apart. It
never reads rotation results. Frames stay in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "scripts"))
from analyse_rotation import actor_bootstrap_ci, actor_sign_flip_p, holm  # noqa: E402

from paper_v2.option_score import rotate_options, vision_mismatch_partners  # noqa: E402

from scripts import run_stimulus_interventions as run  # noqa: E402
from scripts import stimulus_interventions as si  # noqa: E402

ROTATIONS = tuple(range(5))
MISMATCH_SEED = 42
MODELS = ("qwen3vl", "gemma4", "molmo2")
CONDITIONS = ("real", "mismatch")
INPUT_ROUTE = "single_face"


def partners_for(items: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    return vision_mismatch_partners(
        [i["trial_id"] for i in items],
        [i["label"] for i in items],
        [i["actor"] for i in items],
        seed=MISMATCH_SEED,
    )


def correct_from_scores(options: Sequence[str], logprobs: Sequence[float], target: str) -> Dict[str, Any]:
    """Correctness and target probability from option scores. Ties count as wrong."""
    values = np.asarray(logprobs, dtype=np.float64)
    probabilities = np.exp(values - values.max())
    probabilities = probabilities / probabilities.sum()
    best = np.flatnonzero(values == values.max())
    index = list(options).index(target)
    return {
        "correct": bool(len(best) == 1 and best[0] == index),
        "p_target": float(probabilities[index]),
    }


def item_estimates(rows: Sequence[Dict[str, Any]], condition: str) -> Dict[str, float]:
    """a_i or b_i: mean correct over five rotations. Items missing a rotation are left out."""
    counts: Dict[str, List[float]] = {}
    for row in rows:
        if row["condition"] == condition and row["status"] == "ok":
            counts.setdefault(row["trial_id"], []).append(float(row["correct"]))
    return {k: float(np.mean(v)) for k, v in counts.items() if len(v) == len(ROTATIONS)}


def summarise_model(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    real, mismatch = item_estimates(rows, "real"), item_estimates(rows, "mismatch")
    ids = sorted(set(real) & set(mismatch))
    actor_of = {r["trial_id"]: r["actor"] for r in rows}
    a = np.array([real[k] for k in ids])
    b = np.array([mismatch[k] for k in ids])
    actors = [actor_of[k] for k in ids]
    delta = a - b
    failures = Counter(r.get("reason") for r in rows if r["status"] != "ok")
    return {
        "n_items": len(ids),
        "n_items_expected": len({r["trial_id"] for r in rows}),
        "failure_reasons": {str(k): v for k, v in failures.items()},
        "accuracy_real": float(a.mean()),
        "accuracy_mismatch": float(b.mean()),
        "delta_real_minus_mismatch": float(delta.mean()),
        "delta_ci_actor_bootstrap": actor_bootstrap_ci(delta, actors),
        "p_delta_actor_sign_flip": actor_sign_flip_p(delta, actors),
    }


def decide(summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """A1's rule, with Holm over A3's own three models."""
    models = list(summaries)
    adjusted = holm([summaries[m]["p_delta_actor_sign_flip"] for m in models])
    out = {}
    for model, p in zip(models, adjusted):
        ci_low = summaries[model]["delta_ci_actor_bootstrap"][0]
        out[model] = {**summaries[model], "p_delta_holm": p, "vision_dependent": bool(p < 0.05 and ci_low > 0)}
    return out


def reading(a1_passed: bool, a3_passed: bool) -> str:
    return {
        (True, True): "Face use on both inputs",
        (False, True): "The grid limited A1; use single_face from here",
        (True, False): "Evidence from A1 needs inspection: the face crop may remove useful context",
        (False, False): "No demonstrated face use on either route; no RQ1 to RQ5 claims for that model",
    }[(bool(a1_passed), bool(a3_passed))]


def trial_cells(item: Dict[str, Any], partner: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Ten cells for one item: the item's options always, the face from item or partner."""
    return [
        {
            "trial_id": item["trial_id"],
            "actor": item["actor"],
            "label": item["label"],
            "rotation": rotation,
            "condition": condition,
            "face_item": item["trial_id"] if condition == "real" else partner["trial_id"],
            "options": rotate_options(item["options"], rotation),
        }
        for rotation in ROTATIONS
        for condition in CONDITIONS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--manifest", type=Path, default=Path("data/eu_emotions_full_manifest.json"))
    parser.add_argument("--lookup", type=Path, default=Path("data/eu_emotions_human_entropy.json"))
    parser.add_argument("--data_root", type=Path, default=Path("data/eu_emotions"))
    parser.add_argument("--geometry", type=Path, default=Path("results/stimulus_interventions/geometry.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/face_input_validity"))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--a3_frozen", action="store_true", help="Required. Set only after A3 is frozen.")
    parser.add_argument("--summarise", action="store_true", help="Combine finished models and decide.")
    args = parser.parse_args()
    if not args.a3_frozen:
        raise SystemExit("A3 is not frozen. This script will not run until the amendment is frozen.")
    if args.summarise:
        rows = {
            m: [json.loads(l) for l in (args.output_dir / f"cells_{m}.jsonl").read_text(encoding="utf-8").splitlines()]
            for m in MODELS
        }
        out = decide({m: summarise_model(r) for m, r in rows.items()})
        path = args.output_dir / "face_input_validity_summary.json"
        path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(out, indent=2, sort_keys=True))
        return
    if args.model is None:
        raise SystemExit("--model is required")

    from scripts.prompts import build_4afc_prompt

    geometry = json.loads(args.geometry.read_text(encoding="utf-8"))["items"]
    items = run.load_items(args.manifest, args.lookup, args.data_root)
    by_id = {i["trial_id"]: i for i in items}
    for trial_id, record in geometry.items():
        record["path"] = by_id[trial_id]["path"] if trial_id in by_id else ""
    neutral_by_actor = {i["actor"]: i["trial_id"] for i in items if i["label"] == "Neutral"}
    partners = partners_for(items)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"cells_{args.model}.jsonl"
    done = {
        (r["trial_id"], r["rotation"], r["condition"])
        for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines())
    } if path.exists() else set()
    handle = run.load_handle(args.model, args.dtype)
    media_cache: Dict[str, Any] = {}

    def face(trial_id: str) -> run.Media:
        if trial_id not in media_cache:
            try:
                clip = run.load_clip(by_id[trial_id], geometry, neutral_by_actor)
                media_cache[trial_id] = run.build_media(clip, INPUT_ROUTE, "original")
            except si.EditFailure as exc:
                media_cache[trial_id] = exc
        value = media_cache[trial_id]
        if isinstance(value, si.EditFailure):
            raise value
        return value

    started = time.time()
    with path.open("a", encoding="utf-8") as sink:
        for count, item in enumerate(sorted(items, key=lambda i: i["trial_id"])):
            for cell in trial_cells(item, by_id[partners[item["trial_id"]]]):
                if (cell["trial_id"], cell["rotation"], cell["condition"]) in done:
                    continue
                try:
                    media = face(cell["face_item"])
                    inputs = run.encode_media(handle, media, build_4afc_prompt(cell["options"], condition="video_only"))
                    ids = [list(handle.tokenizer(f" {o}", add_special_tokens=False).input_ids) for o in cell["options"]]
                    scores = run.score_sequences(handle, inputs, handle.mcq_prefill, ids)
                    cell.update(status="ok", joint_logprob=[round(v, 4) for v in scores])
                    cell["sent_frame_indices"] = media.sent_indices
                    cell.update(correct_from_scores(cell["options"], scores, cell["label"]))
                except si.EditFailure as exc:
                    cell.update(status="failed", reason=exc.reason)
                sink.write(json.dumps(cell) + "\n")
            sink.flush()
            print(f"{count + 1}/{len(items)} {round(time.time() - started)} s", flush=True)
    print("Done:", path, hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
