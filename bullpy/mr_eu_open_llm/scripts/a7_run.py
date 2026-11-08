"""
Stage 1 of amendment A7: endpoints, hybrids and sentinels per item, in A2's row format.

Thin GPU entry point, not exercised locally. Every cell of an item, the two native endpoints
included, comes from this one process, unrounded (audit 2026-11-08). Held tensors live in
memory for one item; frames are edited in memory; nothing but rows is written. Resumable
(cells already in the output are skipped) and sharded by item.

Cells per item, single_face, blind readout, A2's prompt, prefill and lexicon:
    original                 y(A, B)            native, captures the state at x
    mask_eyesmouth_neutral   y(A', B')          native, captures the state at x'
    route arm:   hold_route_A_edit  y(A', B)    pixels x, block A held at x'
                 hold_route_B_edit  y(A, B')    pixels x, block B held at x'
    token arm:   hold_token_S_edit      y(S', rest)   pixels x, S held at x'
                 hold_token_rest_edit   y(S, rest')   pixels x', S held at x
                 hold_token_Q_edit      y(Q', rest)   pixels x, Q held at x'
                 hold_token_Qrest_edit  y(Q, rest')   pixels x', Q held at x
    sentinels (positions 0, 10, 20, ... of A2's canonical order of the 200 items):
                 hold_sentinel_route   pixels x', every block held at x
                 hold_sentinel_token   pixels x', every exit token held at x

Arms run only if Stage 0 admits them (scripts/a7_stage0.arms) and a decision to run them is
recorded; the job refuses otherwise.

Usage on the cluster, after the freeze:
    python -m scripts.a7_run --model gemma4 --arms route --stage0 research/reproduction/a7_stage0_gemma4.json \
        --decision_date 2026-11-XX --a7_frozen --shard 0 --n_shards 2
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from scripts import a7_design
from scripts import a7_stage0 as s0
from scripts.a7_analyse import CELLS, EDIT, ORIGINAL, ROUTE, SENTINEL, sentinel_items, stage1_items

D = a7_design.load_design()


def placebo_set(s_tokens: Sequence[int], n_tokens: int, clip_key: str) -> List[int]:
    """Q: a random set outside S, the same size, fixed by a seed keyed on the clip."""
    from scripts.vision_gates import stable_int

    outside = np.array(sorted(set(range(n_tokens)) - set(int(t) for t in s_tokens)))
    if len(outside) < len(s_tokens):
        raise ValueError("S is larger than its complement; the share cap should prevent this.")
    rng = np.random.default_rng(stable_int(int(D["seed"]), "a7_placebo", clip_key))
    return sorted(int(t) for t in rng.choice(outside, size=len(s_tokens), replace=False))


def main() -> None:  # pragma: no cover - needs a GPU, model weights and the clips
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, choices=D["models"])
    parser.add_argument("--arms", required=True, help="Comma-separated: route, token.")
    parser.add_argument("--stage0", type=Path, required=True)
    parser.add_argument("--decision_date", required=True, help="Date of the DECISIONS.md entry that runs these arms.")
    parser.add_argument("--manifest", type=Path, default=Path("data/eu_emotions_full_manifest.json"))
    parser.add_argument("--lookup", type=Path, default=Path("data/eu_emotions_human_entropy.json"))
    parser.add_argument("--data_root", type=Path, default=Path("."))
    parser.add_argument("--geometry", type=Path, default=Path("results/stimulus_interventions/geometry.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/a7"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n_shards", type=int, default=1)
    parser.add_argument("--a7_frozen", action="store_true")
    args = parser.parse_args()
    if not args.a7_frozen:
        raise SystemExit("A7 is not frozen. Stage 1 runs only after the freeze.")
    design = a7_design.load_design()
    a7_design.check_pins(design)
    pins = a7_design.check_code_pins()
    table = s0.load_table(args.stage0)
    admitted = s0.arms(table)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        if arm not in CELLS or not admitted.get(arm):
            raise SystemExit(f"Stage 0 does not admit the {arm} arm on {args.model}.")

    import torch

    from scripts import run_stimulus_interventions as run
    from scripts import stimulus_interventions as si
    from scripts.a7_cuts import Session, block_tensors, relative_change
    from scripts.vision_gates import adapter_for

    run.check_geometry(args.geometry)
    geometry_items = json.loads(args.geometry.read_text(encoding="utf-8"))["items"]
    all_items = run.load_items(args.manifest, args.lookup, args.data_root)
    paths = {i["trial_id"]: i["path"] for i in all_items}
    for trial_id, record in geometry_items.items():
        record["path"] = paths.get(trial_id, "")
    neutral_by_actor = {i["actor"]: i["trial_id"] for i in all_items if i["label"] == "Neutral"}
    ordered = stage1_items(all_items)
    sentinels = set(sentinel_items(ordered))
    items = [i for n, i in enumerate(ordered) if n % args.n_shards == args.shard]

    handle = run.load_handle(args.model, D["dtype"])
    adapter = adapter_for(args.model, handle.model)

    def readout(inputs: Dict[str, Any]) -> np.ndarray:
        return np.array(run.score_sequences(handle, inputs, handle.blind_prefill, handle.word_ids), dtype=np.float64)

    session = Session(adapter, args.model, lambda media, _c: run.encode_media(handle, media, run.BLIND_PROMPT),
                      readout, D["route_blocks"][args.model])
    n_tokens = int(D["image_tokens"][args.model])
    side = int(round(math.sqrt(n_tokens)))
    buffer = (table.get("region_test") or {}).get("b")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_shard{args.shard}" if args.n_shards > 1 else ""
    cells_path = args.output_dir / f"a7_cells_{args.model}{suffix}.jsonl"
    done = run.done_cells(cells_path)

    def row_for(item: Dict[str, Any], condition: str, values: np.ndarray, **extra: Any) -> Dict[str, Any]:
        family = run.family_scores(handle.words, values.tolist())
        return {"trial_id": item["trial_id"], "actor": item["actor"], "label": item["label"],
                "human_modal_label": item["human_modal_label"], "weak_label": item["weak_label"],
                "input_name": ROUTE, "condition": condition, "status": "ok", "a7_design_sha256": a7_design.DESIGN_SHA256,
                "blind": {"word_logprob": values.tolist(), "family_logprob": family,
                          "lexicon_mass": math.exp(run.logsumexp(values.tolist()))}, **extra}

    with cells_path.open("a", encoding="utf-8") as sink:
        for item in items:
            if (item["trial_id"], ROUTE, ORIGINAL) in done:
                continue
            started = time.time()
            rows: List[Dict[str, Any]] = []
            try:
                clip = run.load_clip(item, geometry_items, neutral_by_actor)
                media_x, media_e = run.build_media(clip, ROUTE, ORIGINAL), run.build_media(clip, ROUTE, EDIT)
                x, xe = session.encode(media_x, ORIGINAL), session.encode(media_e, EDIT)
                if int(session.positions(x).numel()) != n_tokens or int(session.positions(xe).numel()) != n_tokens:
                    raise si.EditFailure("token_count")
                y00, state_x = session.native(x)
                y11, state_e = session.native(xe)
                change: Dict[str, Any] = {}
                if "route" in arms:
                    bx, be = block_tensors(args.model, state_x, D["route_blocks"][args.model]), block_tensors(args.model, state_e, D["route_blocks"][args.model])
                    change["route"] = {b: relative_change(bx[b], be[b]) for b in ("A", "B")}
                    rows.append(row_for(item, CELLS["route"]["A_edit"], session.hold(x, ("A",), state_e)))
                    rows.append(row_for(item, CELLS["route"]["B_edit"], session.hold(x, ("B",), state_e)))
                    if item["trial_id"] in sentinels:
                        rows.append(row_for(item, SENTINEL["route"], session.hold(xe, ("A", "B"), state_x)))
                if "token" in arms:
                    alpha = si.region_mask(clip.frames[clip.apex].shape[:2], clip.landmarks.get(clip.apex), "eyesmouth")
                    crop = si.fixed_size(si.crop((alpha * 255).astype(np.uint8)[..., None].repeat(3, -1), clip.geometry.face_box))
                    hit = crop[..., 0] > 0
                    step = int(D["crop_px"]) // side
                    mask = [r * side + c for r in range(side) for c in range(side) if hit[r * step:(r + 1) * step, c * step:(c + 1) * step].any()]
                    s_tokens = s0.dilate(mask, int(buffer), side)
                    rest = sorted(set(range(n_tokens)) - set(s_tokens))
                    if len(s_tokens) > len(rest):
                        # The share cap was checked on calibration clips only, so on an analysis item S
                        # can outgrow its complement and no same-size Q exists. The item leaves the token
                        # arm with a reason, instead of stopping the job (independent audit 2026-11-08).
                        change["token"] = None
                        change["token_failed"] = {"reason": "token_set_larger_than_complement", "n": len(s_tokens)}
                    else:
                        q_tokens = placebo_set(s_tokens, n_tokens, item["trial_id"])
                        tx, te = state_x.routes, state_e.routes
                        idx = lambda ts: torch.as_tensor(ts, dtype=torch.long)
                        change["token"] = {
                            "S": relative_change({k: v[idx(s_tokens).to(v.device)] for k, v in tx.items()}, {k: v[idx(s_tokens).to(v.device)] for k, v in te.items()}),
                            "rest": relative_change({k: v[idx(rest).to(v.device)] for k, v in tx.items()}, {k: v[idx(rest).to(v.device)] for k, v in te.items()}),
                        }
                        rows.append(row_for(item, CELLS["token"]["S_edit"], session.token_hold(x, s_tokens, state_e), token_set={"b": buffer, "n": len(s_tokens)}))
                        rows.append(row_for(item, CELLS["token"]["rest_edit"], session.token_hold(xe, s_tokens, state_x)))
                        rows.append(row_for(item, CELLS["token"]["Q_edit"], session.token_hold(x, q_tokens, state_e)))
                        rows.append(row_for(item, CELLS["token"]["Qrest_edit"], session.token_hold(xe, q_tokens, state_x)))
                        if item["trial_id"] in sentinels:
                            rows.append(row_for(item, SENTINEL["token"], session.token_hold(xe, list(range(n_tokens)), state_x)))
                rows.insert(0, row_for(item, ORIGINAL, y00))
                rows.insert(1, row_for(item, EDIT, y11, block_change=change, edit=media_e.log))
            except si.EditFailure as exc:
                rows = [{"trial_id": item["trial_id"], "actor": item["actor"], "label": item["label"], "input_name": ROUTE,
                         "condition": ORIGINAL, "status": "failed", "reason": exc.reason,
                         "a7_design_sha256": a7_design.DESIGN_SHA256}]
            for row in rows:
                row["seconds_item"] = round(time.time() - started, 2)
                row["a7_code_sha256"] = pins
                row["decision_date"] = args.decision_date
                sink.write(json.dumps(row) + "\n")
            sink.flush()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"Done: {cells_path}")


if __name__ == "__main__":
    main()
