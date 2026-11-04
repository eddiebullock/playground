"""
Amendment A3: A1's design on the single_face route. See research/PREREG_A3_DRAFT.md.

Same 243 items, five rotations, real versus mismatch, same partners, same estimand and
decision rule, as its own Holm family over three models. A3 does not replace A1.

Scoring is fixed in advance as exact option probabilities: each option's joint logprob
after the prefill "EMOTION:", renormalised over the six options. An item is correct on a
rotation if the target has the highest probability, which is the limit of A1's modal
answer as draws grow. No sampling, so no seed enters scoring.

This script refuses to score a model until the amendment is frozen. Statistics are imported
from research/scripts/analyse_rotation.py, so the two amendments cannot drift apart. It
never reads rotation results. Frames stay in memory.

--mechanics_only N runs N items through the whole path without the freeze flag, including
mismatch partners and all ten cells. It writes only the fields in MECHANICS_FIELDS, so no
score can be read from it, and it writes outside results/face_input_validity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest, spearmanr

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
GEOMETRY_SHA256 = "2bc76dd60be233ee6072ba5b2049721899c5febcd953ec49bff1dcc2f2c72383"
MAX_LOST_SHARE = 0.05
MECHANICS_FIELDS = ("status", "reason", "seconds", "forwards", "sent_frame_indices")
OUTCOME_DIR = Path("results/face_input_validity")
MECHANICS_DIR = Path("results/face_input_validity_mechanics")


def read_cells(path: Path) -> Dict[str, Any]:
    """The one rule for partial lines, used by resume and by --summarise alike.

    A row counts only if it is valid JSON ending in a newline. One bad final line is what a
    killed job leaves, so it is dropped and counted. A bad line anywhere else is corruption
    and raises. No file: no rows.
    """
    if not path.exists():
        return {"rows": [], "n_partial_dropped": 0, "good_bytes": 0}
    data = path.read_bytes()
    lines = data.split(b"\n")
    tail = lines.pop()  # empty when the file ends in a newline
    rows = []
    good_bytes = 0
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
            good_bytes += len(line) + 1
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index == len(lines) - 1 and not tail:
                return {"rows": rows, "n_partial_dropped": 1, "good_bytes": good_bytes}
            raise ValueError(f"{path}: bad line {index + 1} is not the final line") from None
    return {"rows": rows, "n_partial_dropped": int(bool(tail)), "good_bytes": good_bytes}


def prepare_resume(path: Path) -> set:
    """Cut a truncated tail, so new rows start on a clean line. Returns the finished cells.

    A cell cut off by a kill is missing, so its item gets that cell rescored. Scoring is
    deterministic, so this gives the row the killed job would have written.
    """
    state = read_cells(path)
    if state["n_partial_dropped"]:
        with path.open("r+b") as handle:
            handle.truncate(state["good_bytes"])
    return {(r["trial_id"], r["rotation"], r["condition"]) for r in state["rows"]}


def mechanics_row(cell: Dict[str, Any]) -> Dict[str, Any]:
    """Whitelist, so a field added to a scored cell cannot reach the mechanics output."""
    return {k: cell[k] for k in MECHANICS_FIELDS if k in cell}


def mechanics_output_dir(path: Path) -> Path:
    resolved, outcomes = path.resolve(), OUTCOME_DIR.resolve()
    if resolved == outcomes or outcomes in resolved.parents:
        raise SystemExit("Mechanics output must stay outside results/face_input_validity.")
    return path


def check_geometry(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != GEOMETRY_SHA256:
        raise SystemExit(f"Geometry hash {digest} is not the pinned one.")


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


def summarise_model(rows: Sequence[Dict[str, Any]], human: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    real, mismatch = item_estimates(rows, "real"), item_estimates(rows, "mismatch")
    ids = sorted(set(real) & set(mismatch))
    actor_of = {r["trial_id"]: r["actor"] for r in rows}
    a = np.array([real[k] for k in ids])
    b = np.array([mismatch[k] for k in ids])
    actors = [actor_of[k] for k in ids]
    delta = a - b
    failures = Counter(r.get("reason") for r in rows if r["status"] != "ok")
    expected = len({r["trial_id"] for r in rows})
    lost = 1.0 - len(ids) / expected if expected else 1.0
    out = {
        "n_items": len(ids),
        "n_items_expected": expected,
        "share_items_lost": lost,
        "over_lost_item_limit": bool(lost > MAX_LOST_SHARE),
        "failure_reasons": {str(k): v for k, v in failures.items()},
        "accuracy_real": float(a.mean()),
        "accuracy_mismatch": float(b.mean()),
        "delta_real_minus_mismatch": float(delta.mean()),
        "delta_ci_actor_bootstrap": actor_bootstrap_ci(delta, actors),
        "p_delta_actor_sign_flip": actor_sign_flip_p(delta, actors),
    }
    if human is not None:
        out["secondary"] = secondary(rows, ids, human)
    return out


def _delta_table(rows: Sequence[Dict[str, Any]], ids: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    real, mismatch = item_estimates(rows, "real"), item_estimates(rows, "mismatch")
    actor_of = {r["trial_id"]: r["actor"] for r in rows}
    return np.array([real[k] - mismatch[k] for k in ids]), [actor_of[k] for k in ids]


def secondary(rows: Sequence[Dict[str, Any]], ids: Sequence[str], human: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The draft's secondary rows, on items that enter the primary estimate."""
    keep = set(ids)
    real = [r for r in rows if r["condition"] == "real" and r["status"] == "ok" and r["trial_id"] in keep]
    by_position: Dict[int, List[float]] = {}
    target_probability: Dict[str, List[float]] = {}
    modal_rows = []
    for row in real + [r for r in rows if r["condition"] == "mismatch" and r["status"] == "ok" and r["trial_id"] in keep]:
        distribution = human[row["trial_id"]]["human_response_distribution"]
        named = {k: v for k, v in distribution.items() if k != "None of the above"}
        modal = max(sorted(named), key=lambda k: named[k])
        modal_rows.append({**row, "correct": correct_from_scores(row["options"], row["joint_logprob"], modal)["correct"]})
    for row in real:
        by_position.setdefault(row["options"].index(row["label"]), []).append(float(row["correct"]))
        target_probability.setdefault(row["trial_id"], []).append(row["p_target"])
    n_correct = sum(int(r["correct"]) for r in real)
    model = np.array([np.mean(target_probability[k]) for k in ids])
    people = np.array([human[k]["p_target"] for k in ids])
    labels = np.array([human[k]["human_target_label"] for k in ids])
    within_model, within_people = model.copy(), people.copy()
    for label in set(labels.tolist()):
        mask = labels == label
        within_model[mask] -= model[mask].mean()
        within_people[mask] -= people[mask].mean()
    strong = [k for k in ids if human[k]["human_target_label"] not in ("Kind", "Unfriendly", "Jealous")]
    delta_strong, actors_strong = _delta_table(rows, strong)
    delta_modal, actors_modal = _delta_table(modal_rows, ids)
    return {
        "accuracy_real_vs_chance_pooled_p": float(binomtest(n_correct, len(real), 1 / 6, alternative="greater").pvalue),
        "accuracy_by_target_position": {str(k): float(np.mean(v)) for k, v in sorted(by_position.items())},
        "mean_target_probability_real": float(model.mean()),
        "rho_ptarget_human": float(spearmanr(model, people)[0]),
        "within_category_r_ptarget_human": float(np.corrcoef(within_model, within_people)[0, 1]),
        "delta_without_kind_unfriendly_jealous": float(delta_strong.mean()),
        "delta_without_those_ci_actor_bootstrap": actor_bootstrap_ci(delta_strong, actors_strong),
        "delta_scored_against_human_modal_label": float(delta_modal.mean()),
        "delta_modal_ci_actor_bootstrap": actor_bootstrap_ci(delta_modal, actors_modal),
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


def score_cell(handle: Any, cell: Dict[str, Any], face: Any) -> Dict[str, Any]:
    """Score one cell in place. A face that cannot be built fails the cell with its reason."""
    from scripts.prompts import build_4afc_prompt

    start, tick = handle.forwards, time.time()
    try:
        media = face(cell["face_item"])
        inputs = run.encode_media(handle, media, build_4afc_prompt(cell["options"], condition="video_only"))
        ids = [list(handle.tokenizer(f" {o}", add_special_tokens=False).input_ids) for o in cell["options"]]
        scores = run.score_sequences(handle, inputs, handle.mcq_prefill, ids)
        cell.update(status="ok", joint_logprob=[round(v, 4) for v in scores], sent_frame_indices=media.sent_indices)
        cell.update(correct_from_scores(cell["options"], scores, cell["label"]))
    except si.EditFailure as exc:
        cell.update(status="failed", reason=exc.reason)
    cell.update(seconds=round(time.time() - tick, 3), forwards=handle.forwards - start)
    return cell


def face_loader(items: Sequence[Dict[str, Any]], geometry: Dict[str, Any]) -> Any:
    by_id = {i["trial_id"]: i for i in items}
    for trial_id, record in geometry.items():
        record["path"] = by_id[trial_id]["path"] if trial_id in by_id else ""
    neutral_by_actor = {i["actor"]: i["trial_id"] for i in items if i["label"] == "Neutral"}
    cache: Dict[str, Any] = {}

    def face(trial_id: str) -> run.Media:
        if trial_id not in cache:
            try:
                clip = run.load_clip(by_id[trial_id], geometry, neutral_by_actor)
                cache[trial_id] = run.build_media(clip, INPUT_ROUTE, "original")
            except si.EditFailure as exc:
                cache[trial_id] = exc
        value = cache[trial_id]
        if isinstance(value, si.EditFailure):
            raise value
        return value

    return face


def summarise(args: argparse.Namespace) -> None:
    human = json.loads(args.lookup.read_text(encoding="utf-8"))["trials"]
    states = {m: read_cells(args.output_dir / f"cells_{m}.jsonl") for m in MODELS}
    out = decide({m: summarise_model(state["rows"], human) for m, state in states.items()})
    for model, state in states.items():
        out[model]["n_partial_lines_dropped"] = state["n_partial_dropped"]
    path = args.output_dir / "face_input_validity_summary.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


def mechanics(args: argparse.Namespace) -> None:
    """All ten cells on N items. Prints and writes no score."""
    output_dir = mechanics_output_dir(args.mechanics_dir)
    check_geometry(args.geometry)
    items = run.load_items(args.manifest, args.lookup, args.data_root)
    by_id = {i["trial_id"]: i for i in items}
    partners = partners_for(items)
    face = face_loader(items, json.loads(args.geometry.read_text(encoding="utf-8"))["items"])
    handle = run.load_handle(args.model, args.dtype)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.time()
    for item in sorted(items, key=lambda i: i["trial_id"])[: args.mechanics_only]:
        for cell in trial_cells(item, by_id[partners[item["trial_id"]]]):
            rows.append(mechanics_row(score_cell(handle, cell, face)))
    path = output_dir / f"mechanics_{args.model}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ok = [r for r in rows if r["status"] == "ok"]
    print(
        json.dumps(
            {
                "model": args.model,
                "n_items": args.mechanics_only,
                "n_cells": len(rows),
                "n_ok": len(ok),
                "reasons": dict(Counter(r.get("reason") for r in rows if r["status"] != "ok")),
                "mean_seconds_per_cell": round(float(np.mean([r["seconds"] for r in ok])), 3) if ok else None,
                "mean_forwards_per_cell": round(float(np.mean([r["forwards"] for r in ok])), 2) if ok else None,
                "seconds": round(time.time() - started, 1),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--manifest", type=Path, default=Path("data/eu_emotions_full_manifest.json"))
    parser.add_argument("--lookup", type=Path, default=Path("data/eu_emotions_human_entropy.json"))
    parser.add_argument("--data_root", type=Path, default=Path("data/eu_emotions"))
    parser.add_argument("--geometry", type=Path, default=Path("results/stimulus_interventions/geometry.json"))
    parser.add_argument("--output_dir", type=Path, default=OUTCOME_DIR)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--a3_frozen", action="store_true", help="Required for scoring. Set only after A3 is frozen.")
    parser.add_argument("--summarise", action="store_true", help="Combine finished models and decide.")
    parser.add_argument("--mechanics_only", type=int, default=None, help="N items, no scores written. No freeze needed.")
    parser.add_argument("--mechanics_dir", type=Path, default=MECHANICS_DIR)
    args = parser.parse_args()
    if args.mechanics_only is not None:
        if args.a3_frozen or args.summarise:
            raise SystemExit("--mechanics_only cannot be combined with --a3_frozen or --summarise.")
        if args.model is None:
            raise SystemExit("--model is required")
        mechanics(args)
        return
    if not args.a3_frozen:
        raise SystemExit("A3 is not frozen. This script will not run until the amendment is frozen.")
    if args.summarise:
        summarise(args)
        return
    if args.model is None:
        raise SystemExit("--model is required")
    check_geometry(args.geometry)
    items = run.load_items(args.manifest, args.lookup, args.data_root)
    by_id = {i["trial_id"]: i for i in items}
    partners = partners_for(items)
    face = face_loader(items, json.loads(args.geometry.read_text(encoding="utf-8"))["items"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"cells_{args.model}.jsonl"
    done = prepare_resume(path)
    handle = run.load_handle(args.model, args.dtype)
    started = time.time()
    with path.open("a", encoding="utf-8") as sink:
        for count, item in enumerate(sorted(items, key=lambda i: i["trial_id"])):
            for cell in trial_cells(item, by_id[partners[item["trial_id"]]]):
                if (cell["trial_id"], cell["rotation"], cell["condition"]) in done:
                    continue
                sink.write(json.dumps(score_cell(handle, cell, face)) + "\n")
            sink.flush()
            print(f"{count + 1}/{len(items)} {round(time.time() - started)} s", flush=True)
    print("Done:", path, hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
