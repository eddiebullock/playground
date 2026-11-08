"""
Recovery for amendment A7, both levels, on CPU. Nothing here reads a clip, a label or a model output.

Level 1 (hooks): CPU toys with two routes and a hidden third route of settable strength,
through the real `vision_intervention`, `LayerLab`, `swap_tokens` and `_cut_fn` (via
scripts/a7_cuts.py) and the real Stage 0 driver and checks (scripts/a7_stage0.py). Each toy
must give the row the draft expects. The Gemma hold is also run on the real `transformers`
Gemma 4 and Qwen3-VL classes built tiny with random weights (scripts/a7_toy.py).

Level 2 (verdicts): synthetic observers over the real 200 Stage 1 items and 17 actors, with
actor and item noise from the design block and TE sizes from `effect_grid`, through
scripts/a7_analyse.py's analyse() unchanged, with the full joint family of 16 columns.

Construction choices, written here because the draft leaves them open:
- The SDs apply to the item-level TE and to each item-level term (draft, Recovery). The four
  cells have three free differences, so the noise is drawn independently for TE, PE_A and
  PE_B (and PE_S, PE_rest, PE_Q); N, phi and I follow from them.
- The observer sits on gemma4. qwen3vl is a null companion (TE 0 and every term 0, with
  noise) in every replicate, so every family has 16 columns with 8 that carry nothing, the
  worst case for power. Route observers put a mirror of their route shares in the token arm
  (phi_Q 0); token observers put an Additive split in the route arm.
- Fills on A2's cells carry the observer's TE with the same noise, so they agree in sign
  when TE does. Gates all pass; the A2 margin is `te_margin_nats`.
- Observers marked "added_by_code" are not in the draft's table. They exist so that every
  named verdict and reason code is reached by some input.

Usage:
    caffeinate -i python -m scripts.a7_recovery --level 1
    caffeinate -i python -m scripts.a7_recovery --level 2 --workers 3
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import a7_analyse as a7  # noqa: E402
from scripts import a7_design  # noqa: E402
from scripts import a7_stage0 as s0  # noqa: E402
from scripts import stimulus_interventions as si  # noqa: E402
from scripts.run_stimulus_interventions import LEXICON, OPTION_TO_FAMILY  # noqa: E402

D = a7.D
R = D["recovery"]
FAMILIES = tuple(sorted(LEXICON))
OUT_DIR = ROOT / "research" / "reproduction"
CHECKPOINT_DIR = ROOT / "results" / "a7_recovery"


# ---------------------------------------------------------------------------
# Level 1
# ---------------------------------------------------------------------------


def _toy_session(model: Any, style: str, readout: Any = None, encode: Any = None):
    from scripts import a7_toy as toy
    from scripts.a7_cuts import Session
    from scripts.vision_gates import adapter_for

    return Session(adapter_for(style, model), style, encode or (lambda clip, _c: toy.toy_inputs(clip)),
                   readout or toy.readout_for(model), D["route_blocks"][style])


def _toy_region(images: Sequence[np.ndarray]) -> s0.RegionSpec:
    from scripts import a7_toy as toy

    scale = toy.IMAGE_PX / float(D["crop_px"])
    centres = [[c[0] * scale, c[1] * scale] for c in D["token_map_centres_px"]]
    index = {id(img): k for k, img in enumerate(images)}
    square = float(D["token_map_square_px"]) * scale
    return s0.RegionSpec(
        paint=lambda clip, centre: toy.toy_inputs(toy.paint_square(clip, centre, square)),
        edit=lambda clip: toy.toy_inputs(toy.blur_region(clip, index[id(clip)])),
        mask_tokens=lambda clip: toy.region_tokens(index[id(clip)]),
        expected=lambda centre: s0.expected_token(centre, toy.SIDE, crop_px=toy.IMAGE_PX),
        centres=centres,
        side=toy.SIDE,
    )


def summarise_table(table: Mapping[str, Any]) -> Dict[str, Any]:
    checks = table.get("checks") or {}
    out = {"status": table.get("status"), "complete_cuts": table.get("complete_cuts"),
           "blocks_exhaust_exit": table.get("blocks_exhaust_exit"), "arms": table.get("arms"),
           "dead_blocks": s0.dead_blocks(table) if table.get("status") == "complete" else None}
    for key in ("C1", "C2", "C3", "C4"):
        if key in checks:
            out[key] = {"worst_abs_nats": checks[key]["worst_abs_nats"], "pass": checks[key]["pass"]}
    if "C5" in checks:
        out["C5"] = {b: {"ratio": v["ratio"], "live": v["live"]} for b, v in checks["C5"]["blocks"].items()}
    region = table.get("region_test")
    if region:
        out["region_test"] = {"pass": region["pass"], "b": region["b"], "reason": region["reason"],
                              "leaks": region.get("leaks"), "token_map_hit_rate": (region.get("token_map") or {}).get("hit_rate"),
                              "edit_mean_rms": region.get("edit_mean_rms"), "edit_floor": region.get("edit_floor"),
                              "admissible_buffers": region.get("admissible_buffers")}
    return out


def level1() -> Dict[str, Any]:
    from scripts import a7_toy as toy

    images = toy.toy_images(int(D["n_calibration_clips"]))
    rows: List[Dict[str, Any]] = []

    def run(name: str, style: str, expected: str, judge, region: bool = False, **kwargs: Any) -> None:
        model = toy.ToyVLM(style, **kwargs).eval()
        started = time.time()
        table = s0.run_stage0(_toy_session(model, style), images, expected_tokens=toy.SIDE * toy.SIDE,
                              region=_toy_region(images) if region else None)
        summary = summarise_table(table)
        rows.append({"toy": name, "style": style, "settings": kwargs, "expected": expected,
                     "pass": bool(judge(table)), "seconds": round(time.time() - started, 1), "result": summary})

    complete = lambda t: t["status"] == "complete" and t["complete_cuts"] == ["exit"] and t["blocks_exhaust_exit"]
    broken = lambda t: (t["status"] == "complete" and not any(t["checks"][k]["pass"] for k in ("C2", "C3", "C4"))
                        and t["complete_cuts"] == [] and t["arms"] == {"route": False, "token": False})
    for style in ("gemma4", "qwen3vl"):
        for leak in R["toy_leak_nats"]:
            if leak <= s0.TOL:
                run(f"third_route_{leak}", style, "C1 to C4 pass" + (" (what the tolerance allows)" if leak else ""),
                    complete, leak_nats=float(leak))
            else:
                run(f"third_route_{leak}", style, "C2 to C4 fail; not testable", broken, leak_nats=float(leak))
    run("per_layer_by_keyword", "gemma4", "C1 to C4 pass with the per-layer input passed by keyword", complete,
        per_layer_positional=False)
    for leak in R["toy_region_leak"]:
        passes = leak < float(D["leak_limit"])
        run(f"region_leak_{leak}", "gemma4", "region test passes" if passes else "region test fails",
            (lambda t: s0.arms(t)["token"] and t["region_test"]["pass"]) if passes else
            (lambda t: (not s0.arms(t)["token"]) and t["region_test"]["pass"] is False),
            region=True, mix=toy.mix_for_leak(float(leak)))
    for style in ("gemma4", "qwen3vl"):
        run("dead_block", style, "C5 fails for B; block_dead", lambda t: complete(t) and s0.dead_blocks(t) == ["B"], dead=True)
    run("non_deterministic", "gemma4", "C1 fails and the run stops",
        lambda t: t["status"] == "stopped_c1" and t["arms"] == {"route": False, "token": False} and "C2" not in t["checks"],
        noise=0.01)
    return {"level": 1, "rows": rows, "pass": all(r["pass"] for r in rows), "failed": [r["toy"] for r in rows if not r["pass"]]}


def gemma_hold() -> Dict[str, Any]:
    """Fixed before freezing, item 4: the Gemma L0 hold with every per-layer input held, on the
    real Gemma 4 module code (tiny, random weights), and the same for Qwen3-VL's routes."""
    import torch

    from scripts import a7_toy as toy
    from scripts.a7_cuts import Session, check_site, SiteRefused
    from scripts.vision_gates import adapter_for

    out: Dict[str, Any] = {"transformers": __import__("transformers").__version__, "models": {}}
    builders = {"gemma4": (toy.tiny_gemma4, toy.tiny_gemma4_inputs), "qwen3vl": (toy.tiny_qwen3vl, toy.tiny_qwen3vl_inputs)}
    for style, (build, inputs_for) in builders.items():
        model = build()
        adapter = adapter_for(style, model)
        clips = list(range(int(D["n_calibration_clips"])))
        session = Session(adapter, style, lambda clip, _c: inputs_for(1000 + clip), toy.readout_for(model, toy.TINY_WORD_IDS),
                          D["route_blocks"][style])
        started = time.time()
        table = s0.run_stage0(session, clips, expected_tokens=16)
        # The single-block holds with pixels from j: each must leave a difference, both together none.
        single: Dict[str, float] = {}
        for blocks in (("A",), ("B",), ("A", "B")):
            worst = 0.0
            for i in clips[:6]:
                xi, xj = inputs_for(1000 + i), inputs_for(1000 + s0.partner(i))
                native, state = session.native(xi)
                held = session.hold(xj, blocks, state)
                worst = max(worst, float(np.abs(held - native).max()))
            single["+".join(blocks)] = worst
        refused = {}
        for site in (["hidden_L1", "hidden_L24", "image_positions_from_L3"] if style == "gemma4" else ["image_positions_from_L3", "hidden_L1"]):
            try:
                check_site(style, site)
                refused[site] = False
            except SiteRefused:
                refused[site] = True
        n_layers = len(adapter.lm_layers)
        kv_shared = [k for k, layer in enumerate(adapter.lm_layers) if getattr(getattr(layer, "self_attn", None), "is_kv_shared_layer", False)]
        ok = (table["status"] == "complete" and table["complete_cuts"] == ["exit"] and table["blocks_exhaust_exit"]
              and s0.dead_blocks(table) == [] and single["A+B"] == 0.0 and single["A"] > s0.TOL and single["B"] > s0.TOL
              and all(refused.values()))
        out["models"][style] = {
            "stage0": summarise_table(table),
            "pixels_from_j_hold_worst_abs_nats": single,
            "lm_layers": n_layers,
            "kv_shared_layers": kv_shared,
            "sites_refused": refused,
            "pass": bool(ok),
            "seconds": round(time.time() - started, 1),
            "dtype": str(next(model.parameters()).dtype),
        }
    out["pass"] = all(m["pass"] for m in out["models"].values())
    out["what_this_shows"] = ("On the real transformers module code at 5.9.0 with random weights, holding Gemma's layer 0 "
                              "hidden state and every per-layer input at image positions makes the output with another "
                              "image's pixels bit-identical to the native output, while either stream alone does not; "
                              "the same holds for Qwen's main and deepstack routes. It does not show this at bfloat16, on "
                              "the real weights, at transformers 5.10.1, or on single_face and the blind prompt: that is Stage 0.")
    return out


# ---------------------------------------------------------------------------
# Level 2
# ---------------------------------------------------------------------------


def real_items() -> List[Dict[str, Any]]:
    from scripts.run_stimulus_interventions import load_items

    keep = ("trial_id", "actor", "label", "human_modal_label", "weak_label")
    return [{k: i[k] for k in keep} for i in load_items(ROOT / "data/eu_emotions_full_manifest.json",
                                                         ROOT / "data/eu_emotions_human_entropy.json", Path("."))]


def observers() -> List[Dict[str, Any]]:
    """name, route shares (PE_A, PE_B in units of TE), token shares ((PE_S, PE_rest), (PE_Q, PE_Qrest)), expected."""
    obs: List[Dict[str, Any]] = []

    def add(name: str, arm: str, expected: Optional[Tuple[str, Optional[str], Optional[str]]], *, route=(0.5, 0.5),
            token=((0.5, 0.5), (0.0, 1.0)), te: bool = True, **extra: Any) -> None:
        obs.append({"name": name, "arm": arm, "expected": expected, "route": route, "token": token, "te": te, **extra})

    add("a_only", "route", ("Carried by A", None, "Additive"), route=(1.0, 0.0), token=((1.0, 0.0), (0.0, 1.0)))
    add("b_only", "route", ("Carried by B", None, "Additive"), route=(0.0, 1.0), token=((0.0, 1.0), (0.0, 1.0)))
    add("additive", "route", ("Both blocks", None, "Additive"), route=(0.5, 0.5))
    add("redundant", "route", ("Both blocks", None, "Redundant"), route=(1.0, 1.0))
    add("synergic", "route", ("Both blocks", None, "Synergic"), route=(0.0, 0.0))
    add("opposed", "route", ("Opposed blocks", None, None), route=(1.5, -0.5))
    add("prior_only", "route", ("No effect to carry", None, None), te=False, prior_only=True, effect_free=True)
    add("edit_does_nothing", "route", ("No effect to carry", None, None), te=False, identity_sd=3.0, effect_free=True)
    add("dead_block", "route", ("Inconclusive", "block_dead", None), route=(1.0, 0.0), dead_c5="B")
    add("dead_block_effect_via_both", "route", ("Inconclusive", "block_dead", None), route=(0.5, 0.5), dead_c5="B")
    add("edit_null_dead_block", "route", ("No effect to carry", None, None), te=False, dead_c5="B", effect_free=True)
    for share in R["false_carry_shares"]:
        # phi_B = share, I = 0: PE_A = 1 - share, PE_B = share.
        add(f"false_carry_{share}", "route", None, route=(1.0 - share, share), false_carry=True)
    add("token_local", "token", ("Carried by A", "carried by cue tokens", None), token=((1.0, 0.0), (0.0, 1.0)))
    add("token_not_local", "token", ("Carried by A", "not_local", None), token=((1.0, 0.0), (1.0, 0.0)))
    add("all_null_family", "route", None, te=False, all_null=True, effect_free=True,
        replicates=int(R["all_null_replicates"]))
    # Added by code: inputs that reach the remaining verdicts and reason codes.
    add("stage0_exit_fails", "route", ("Not testable", "stage0_not_shown", None), stage0_fail=True, added_by_code=True, replicates=50)
    add("sentinel_fails", "route", ("Not testable", "sentinel_failed", None), sentinel_fail=True, added_by_code=True, replicates=50)
    add("no_decision", "route", ("Not run", "no_recorded_decision", None), no_decision=True, added_by_code=True, replicates=50)
    add("gate_b_fails", "route", ("Not tested", "precondition_failed", None), gate_fail=True, added_by_code=True, replicates=50)
    add("items_missing", "route", ("Inconclusive", "items_missing", None), drop_items=30, added_by_code=True, replicates=50)
    add("fill_missing", "route", ("Inconclusive", "te_unresolved", None), route=(1.0, 0.0), no_fills=True, added_by_code=True, replicates=50)
    add("a2_margin_missing", "route", ("Inconclusive", "te_unresolved", None), te=False, no_margin=True, effect_free=True,
        added_by_code=True, replicates=50)
    return obs


class World:
    """The real 200 items in canonical order, their actors, and fixed family scores per item."""

    def __init__(self, items: Sequence[Mapping[str, Any]]) -> None:
        self.items = a7.stage1_items(items)
        self.actors = a7.actors_of(self.items)
        self.actor = np.array([self.actors.index(i["actor"]) for i in self.items])
        self.acted = np.array([FAMILIES.index(OPTION_TO_FAMILY[i["label"]]) for i in self.items])
        self.sentinels = set(a7.sentinel_items(self.items))
        self.n = len(self.items)


def term(rng: np.random.Generator, world: World, mean: float) -> np.ndarray:
    """Item-level values: mean plus an actor part and an item part, in nats."""
    actor = rng.normal(0, R["actor_sd"], len(world.actors))[world.actor]
    return mean + actor + rng.normal(0, R["item_sd"], world.n)


def model_rows(world: World, rng: np.random.Generator, effect: float, spec: Mapping[str, Any], null: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Stage 1 rows and A2 fill rows for one model. null: TE 0 and every term 0, with noise."""
    te_mean = 0.0 if (null or not spec["te"]) else effect
    identity = spec.get("identity_sd", 1.0)
    base = (rng.normal(0, identity, len(world.actors))[world.actor] + rng.normal(0, identity, world.n))
    others = rng.normal(0, 1, (world.n, len(FAMILIES)))
    if spec.get("prior_only") and not null:
        zero = np.zeros(world.n)
        te = pe_a = pe_b = pe_s = pe_r = pe_q = pe_qr = zero
        fills = [zero, zero]
    else:
        ra, rb = (0.0, 0.0) if null else spec["route"]
        (sa, sr), (qa, qr) = ((0.0, 0.0), (0.0, 0.0)) if null else spec["token"]
        te = term(rng, world, te_mean)
        pe_a, pe_b = term(rng, world, ra * te_mean), term(rng, world, rb * te_mean)
        pe_s, pe_r = term(rng, world, sa * te_mean), term(rng, world, sr * te_mean)
        pe_q, pe_qr = term(rng, world, qa * te_mean), term(rng, world, qr * te_mean)
        fills = [term(rng, world, te_mean), term(rng, world, te_mean)]
    cells = {
        a7.ORIGINAL: base, a7.EDIT: base + te,
        "hold_route_A_edit": base + pe_a, "hold_route_B_edit": base + pe_b,
        "hold_token_S_edit": base + pe_s, "hold_token_rest_edit": base + pe_r,
        "hold_token_Q_edit": base + pe_q, "hold_token_Qrest_edit": base + pe_qr,
    }
    drop = set(range(spec.get("drop_items", 0))) if not null else set()
    words = np.round(rng.normal(-5, 1, (world.n, s0.N_WORDS)), 6)
    rows: List[Dict[str, Any]] = []
    for k, item in enumerate(world.items):
        meta = {"trial_id": item["trial_id"], "actor": item["actor"], "label": item["label"],
                "human_modal_label": item["human_modal_label"], "weak_label": item["weak_label"],
                "input_name": a7.ROUTE, "status": "ok"}
        for condition, values in cells.items():
            if k in drop and condition.startswith("hold_"):
                continue
            row = {**meta, "condition": condition, "blind": {"family_logprob": scores(others[k], world.acted[k], values[k])}}
            if condition == a7.ORIGINAL and item["trial_id"] in world.sentinels:
                row["blind"]["word_logprob"] = words[k].tolist()
            if condition == a7.EDIT:
                row["block_change"] = {"route": {"A": 0.1, "B": 0.1}, "token": {"S": 0.1, "rest": 0.1}}
            rows.append(row)
        if item["trial_id"] in world.sentinels:
            shift = 0.01 if (spec.get("sentinel_fail") and not null) else 0.0
            for arm in a7.ARMS:
                rows.append({**meta, "condition": a7.SENTINEL[arm],
                             "blind": {"family_logprob": {}, "word_logprob": (words[k] + shift).tolist()}})
    fill_rows: List[Dict[str, Any]] = []
    if not (spec.get("no_fills") and not null):
        for k, item in enumerate(world.items):
            meta = {"trial_id": item["trial_id"], "actor": item["actor"], "label": item["label"], "input_name": a7.ROUTE, "status": "ok"}
            for condition, values in ((a7.ORIGINAL, base), (a7.FILLS[0], base + fills[0]), (a7.FILLS[1], base + fills[1])):
                fill_rows.append({**meta, "condition": condition, "blind": {"family_logprob": scores(others[k], world.acted[k], values[k])}})
    return rows, fill_rows


def scores(others: np.ndarray, acted: int, y: float) -> Dict[str, float]:
    """Family log scores whose acted-family CLR is exactly y."""
    v = others.copy()
    n = len(v)
    rest = v.sum() - v[acted]
    v[acted] = (y + rest / n) * n / (n - 1)
    return dict(zip(FAMILIES, v.tolist()))


def stage0_for(spec: Mapping[str, Any], null: bool) -> Dict[str, Any]:
    """A synthetic Stage 0 table in the real format: everything passes unless the observer says not."""
    ok = {"worst_abs_nats": 0.0, "pass": True}
    exit_fail = bool(spec.get("stage0_fail")) and not null
    dead = None if null else spec.get("dead_c5")
    blocks = {b: {"ratio": 0.0 if b == dead else 0.5, "live": b != dead} for b in ("A", "B")}
    checks = {"C1": ok, "C2": {"worst_abs_nats": 0.5, "pass": False} if exit_fail else ok, "C3": ok, "C4": ok,
              "C5": {"denominator": 1.0, "blocks": blocks}}
    region = {"pass": True, "b": 0, "reason": None}
    table = {"amendment": "A7", "design_sha256": a7_design.DESIGN_SHA256, "status": "complete", "checks": checks,
             "region_test": region}
    table["arms"] = s0.arms(table)
    return table


def a2_output_for(spec: Mapping[str, Any], null: bool) -> Dict[str, Any]:
    gates = {k: True for k in ("rq0", "g2_g4", "gate_b", "delivery", "sham")}
    if spec.get("gate_fail") and not null:
        gates["gate_b"] = False
    margin = None if (spec.get("no_margin") and not null) else float(R["te_margin_nats"])
    return {a7.ROUTE: {"gates": gates, "margin": margin}}


def decisions_for(spec: Mapping[str, Any], null: bool) -> Optional[Dict[str, Any]]:
    if spec.get("no_decision") and not null:
        return None
    return {arm: {"run": True, "date": "synthetic", "reason": "recovery"} for arm in a7.ARMS}


def replicate(world: World, spec: Mapping[str, Any], effect: float, k: int) -> Dict[str, Any]:
    seq = np.random.SeedSequence([int(R["seed"]), si.stable_seed(spec["name"]), int(round(effect * 1000)), k])
    rng = np.random.default_rng(seq)
    inputs = {}
    for model in a7.MODELS:
        null = model != "gemma4" or bool(spec.get("all_null"))
        rows, fill_rows = model_rows(world, rng, effect, spec, null)
        inputs[model] = a7.ModelInputs(rows, stage0_for(spec, null), a2_output_for(spec, null), fill_rows, decisions_for(spec, null))
    out = a7.analyse(world_items(world), inputs)
    row = out["models"]["gemma4"]["arms"][spec["arm"]]
    columns = out["family"]["columns"]
    p_values = {c: None for c in columns}
    for model, body in out["models"].items():
        if body["te"]:
            p_values[f"{model}/endpoints/TE"] = body["te"]["p_maxt"]
        for arm, arm_row in body["arms"].items():
            for name, summary in (arm_row.get("terms") or {}).items():
                p_values[f"{model}/{arm}/{name}"] = summary["p_maxt"]
    any_sig = any(p is not None and p < float(D["alpha"]) for p in p_values.values())
    return {"verdict": row["verdict"], "reason": row["reason"], "riders": row["riders"], "reading": row["reading"],
            "carrying_class": row.get("carrying_class"), "n_columns": len(columns), "any_column_significant": any_sig}


# One departure, in recovery workers only: A2's signflip_maxt rebuilds all 2^17 flips and loops
# over blocks on every call. fast_signflip_maxt computes the same statistics from one cached
# flip matrix by a matrix product (the sum of squares is unchanged by a sign flip), and falls
# back to the original for NaN, a constant column or more than 17 actors. tests/test_a7.py
# checks that both give the same p-values. The real analysis always uses A2's function.
_ORIGINAL_SIGNFLIP = si.signflip_maxt
_FLIPS: Dict[int, np.ndarray] = {}


def _flips(n: int) -> np.ndarray:
    if n not in _FLIPS:
        import itertools

        _FLIPS[n] = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
    return _FLIPS[n]


def fast_signflip_maxt(actor_effects: np.ndarray, *, alternative: Sequence[int] = (), max_exact_actors: int = 17,
                       draws: int = 100_000, seed: int = si.SEED) -> Dict[str, Any]:
    effects = np.asarray(actor_effects, dtype=np.float64)
    n_actors, n_tests = effects.shape
    if n_actors > max_exact_actors or np.isnan(effects).any() or np.any(np.ptp(effects, axis=0) == 0):
        return _ORIGINAL_SIGNFLIP(effects, alternative=alternative, max_exact_actors=max_exact_actors, draws=draws, seed=seed)
    signs_wanted = np.asarray(alternative if len(alternative) else [0] * n_tests, dtype=np.float64)
    flips = _flips(n_actors)

    def signed(t: np.ndarray) -> np.ndarray:
        return np.where(signs_wanted == 0, np.abs(t), t * signs_wanted)

    observed = signed(si._t_stats(effects))
    means = flips @ effects / n_actors
    sumsq = (effects ** 2).sum(0)
    var = np.maximum((sumsq[None, :] - n_actors * means ** 2) / (n_actors - 1), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = means / (np.sqrt(var) / np.sqrt(n_actors))
    null = signed(np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0))
    order = np.argsort(-observed, kind="stable")
    running = np.full(flips.shape[0], -np.inf)
    adjusted = np.empty(n_tests)
    for test in order[::-1]:
        running = np.maximum(running, null[:, test])
        adjusted[test] = float((running >= observed[test] - 1e-12).mean())
    previous = 0.0
    for test in order:
        previous = max(previous, adjusted[test])
        adjusted[test] = previous
    raw = (null >= observed - 1e-12).mean(0)
    return {"t": observed.tolist(), "p_raw": raw.tolist(), "p_maxt": adjusted.tolist(),
            "n_flips": int(flips.shape[0]), "exact": True}


def install_fast_statistics() -> None:
    """Only in recovery workers, in memory."""
    si.signflip_maxt = fast_signflip_maxt


_ITEMS: Optional[List[Dict[str, Any]]] = None
_WORLD: Optional[World] = None


def world_items(world: World) -> List[Dict[str, Any]]:
    assert _ITEMS is not None
    return _ITEMS


def _init(items: Sequence[Mapping[str, Any]]) -> None:
    global _ITEMS, _WORLD
    install_fast_statistics()
    _ITEMS = [dict(i) for i in items]
    _WORLD = World(items)


def _task(task: Tuple[Dict[str, Any], float, int]) -> Tuple[str, Dict[str, Any]]:
    spec, effect, k = task
    assert _WORLD is not None
    return key_of(spec, effect, k), replicate(_WORLD, spec, effect, k)


def key_of(spec: Mapping[str, Any], effect: float, k: int) -> str:
    return f"{spec['name']}|{effect}|{k}"


def tasks_for(specs: Sequence[Mapping[str, Any]], replicates: int) -> List[Tuple[Dict[str, Any], float, int]]:
    tasks = []
    for spec in specs:
        effects = [float(R["pass_effect"])] if spec.get("effect_free") or spec.get("added_by_code") else [float(e) for e in R["effect_grid"]]
        n = spec.get("replicates", replicates)
        n = min(n, replicates) if spec.get("added_by_code") else n
        if replicates < int(R["replicates"]) and spec.get("all_null"):
            n = replicates
        for effect in effects:
            tasks.extend((dict(spec), effect, k) for k in range(n))
    return tasks


def read_checkpoint(path: Path) -> Dict[str, Dict[str, Any]]:
    """Rows already finished. A torn last line, from a stop mid-write, is dropped."""
    done: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        done[entry["key"]] = entry["row"]
    return done


def run_tasks(tasks: Sequence[Tuple[Dict[str, Any], float, int]], items: Sequence[Mapping[str, Any]], workers: int,
              checkpoint: Path) -> Dict[str, Dict[str, Any]]:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    stamp = {"design_sha256": a7_design.DESIGN_SHA256, "code": a7_design.own_code_sha256()}
    stamp_path = checkpoint.with_suffix(".stamp.json")
    if checkpoint.exists() and stamp_path.exists() and json.loads(stamp_path.read_text()) != stamp:
        raise SystemExit(f"{checkpoint} was written by other code; move it aside before resuming.")
    stamp_path.write_text(json.dumps(stamp, indent=1) + "\n", encoding="utf-8")
    done = read_checkpoint(checkpoint)
    todo = [t for t in tasks if key_of(*t) not in done]
    print(f"{len(tasks)} replicates, {len(done)} already done, {len(todo)} to run", flush=True)
    if todo:
        started, count = time.time(), 0
        with checkpoint.open("a", encoding="utf-8") as handle:
            if workers <= 1:
                _init(items)
                results = map(_task, todo)
                pool = None
            else:
                pool = multiprocessing.Pool(workers, _init, (items,))
                results = pool.imap_unordered(_task, todo, chunksize=4)
            for key, row in results:
                done[key] = row
                handle.write(json.dumps({"key": key, "row": row}) + "\n")
                handle.flush()
                count += 1
                if count % 500 == 0:
                    rate = count / (time.time() - started)
                    print(f"{count}/{len(todo)} done, {rate:.2f}/s, about {(len(todo) - count) / rate / 60:.0f} min left", flush=True)
            if pool is not None:
                pool.close()
                pool.join()
    return done


def outcome_matches(row: Mapping[str, Any], expected: Tuple[str, Optional[str], Optional[str]]) -> bool:
    verdict, second, rider = expected
    if row["verdict"] != verdict:
        return False
    if second is not None:
        if verdict == "Inconclusive" or verdict in (a7.NOT_TESTABLE, a7.NOT_RUN, a7.NOT_TESTED):
            if row["reason"] != second:
                return False
        elif second == "not_local":
            if "not_local" not in row["riders"]:
                return False
        elif row["reading"] != second or "not_local" in row["riders"]:
            return False
    if rider is not None and rider not in row["riders"]:
        return False
    return True


NAMED = tuple(v for v in a7.VERDICTS if v != a7.INCONCLUSIVE)


def summarise(spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    verdicts = Counter(r["verdict"] if r["verdict"] != a7.INCONCLUSIVE else f"Inconclusive/{r['reason']}" for r in rows)
    out: Dict[str, Any] = {"n": n, "verdicts": dict(sorted(verdicts.items())),
                           "riders": dict(Counter(tuple(r["riders"]) and "+".join(r["riders"]) or "none" for r in rows))}
    if spec.get("expected"):
        expected = spec["expected"]
        out["expected"] = list(expected)
        out["expected_rate"] = sum(outcome_matches(r, expected) for r in rows) / n
        out["expected_verdict_rate"] = sum(r["verdict"] == expected[0] for r in rows) / n
        out["other_named_rate"] = sum(r["verdict"] in NAMED and r["verdict"] != expected[0] for r in rows) / n
        out["inconclusive_rate"] = sum(r["verdict"] == a7.INCONCLUSIVE for r in rows) / n
        if expected[0] == a7.INCONCLUSIVE:
            out["inconclusive_rate_other_reason"] = sum(r["verdict"] == a7.INCONCLUSIVE and r["reason"] != expected[1] for r in rows) / n
    if spec.get("false_carry"):
        out["carried_by_a_rate"] = sum(r["verdict"] == "Carried by A" for r in rows) / n
    if spec.get("all_null"):
        out["any_column_significant_rate"] = sum(r["any_column_significant"] for r in rows) / n
        out["n_columns"] = sorted({r["n_columns"] for r in rows})
    return out


def judge(spec: Mapping[str, Any], summary: Mapping[str, Any]) -> Dict[str, Any]:
    """The draft's pass rule at pass_effect. Inconclusive is the expected row for block_dead observers,
    so for them the Inconclusive cap applies to other reason codes only."""
    if spec.get("all_null"):
        rate = summary["any_column_significant_rate"]
        return {"pass": rate <= float(R["false_positive_max"]), "rule": f"any column significant <= {R['false_positive_max']}", "value": rate}
    if spec.get("false_carry"):
        rate = summary["carried_by_a_rate"]
        return {"pass": rate <= float(R["false_positive_max"]), "rule": f"Carried by A <= {R['false_positive_max']}", "value": rate}
    expected = spec["expected"]
    inconclusive = summary.get("inconclusive_rate_other_reason", summary["inconclusive_rate"]) if expected[0] == a7.INCONCLUSIVE else summary["inconclusive_rate"]
    checks = {
        "expected_rate": (summary["expected_rate"], summary["expected_rate"] >= float(R["expected_verdict_min"])),
        "other_named_rate": (summary["other_named_rate"], summary["other_named_rate"] <= float(R["other_named_verdict_max"])),
        "inconclusive_rate": (inconclusive, inconclusive <= float(R["inconclusive_max"])),
    }
    return {"pass": all(ok for _v, ok in checks.values()), "checks": {k: {"value": v, "pass": ok} for k, (v, ok) in checks.items()}}


def level2(workers: int, replicates: int, checkpoint: Path, only: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    code_at_start = a7_design.own_code_sha256()
    items = real_items()
    specs = [o for o in observers() if not only or o["name"] in only]
    tasks = tasks_for(specs, replicates)
    done = run_tasks(tasks, items, workers, checkpoint)
    table: Dict[str, Any] = {}
    reached: Counter = Counter()
    for spec in specs:
        effects = sorted({t[1] for t in tasks if t[0]["name"] == spec["name"]})
        table[spec["name"]] = {"arm": spec["arm"], "added_by_code": bool(spec.get("added_by_code")), "by_effect": {}}
        for effect in effects:
            rows = [done[key_of(t[0], t[1], t[2])] for t in tasks if t[0]["name"] == spec["name"] and t[1] == effect]
            for r in rows:
                reached[r["verdict"] if r["verdict"] != a7.INCONCLUSIVE else f"Inconclusive/{r['reason']}"] += 1
            summary = summarise(spec, rows)
            if effect == float(R["pass_effect"]):
                summary["judged"] = judge(spec, summary)
            table[spec["name"]]["by_effect"][str(effect)] = summary
    wanted = list(NAMED) + [f"Inconclusive/{r}" for r in a7.REASONS]
    never = [v for v in wanted if reached[v] == 0]
    judged = {name: body["by_effect"][str(float(R["pass_effect"]))]["judged"] for name, body in table.items()
              if str(float(R["pass_effect"])) in body["by_effect"]}
    failed = [name for name, j in judged.items() if not j["pass"]]
    return {
        "level": 2,
        "replicates": replicates,
        "pass_effect": R["pass_effect"],
        "observers": table,
        "verdicts_reached": dict(sorted(reached.items())),
        "verdicts_never_reached": never,
        "failed": failed + ([f"never reached: {v}" for v in never]),
        "pass": not failed and not never,
        "code_sha256_at_start": code_at_start,
        "code_changed_during_run": code_at_start != a7_design.own_code_sha256(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--level", type=int, choices=(1, 2), required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--replicates", type=int, default=int(R["replicates"]))
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "level2.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    design = a7_design.load_design()
    a7_design.check_pins(design)
    a7_design.check_code_pins(require=False)
    started = time.time()
    if args.level == 1:
        out = {"level1": level1(), "gemma_hold": gemma_hold()}
        out["pass"] = out["level1"]["pass"] and out["gemma_hold"]["pass"]
        output = args.output or OUT_DIR / "a7_recovery_level1.json"
    else:
        out = level2(args.workers, args.replicates, args.checkpoint, args.only)
        output = args.output or OUT_DIR / "a7_recovery.json"
    out["design_sha256"] = a7_design.DESIGN_SHA256
    out["code_sha256"] = a7_design.own_code_sha256()
    out["seconds"] = round(time.time() - started, 1)
    output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"pass": out["pass"], "failed": out.get("failed") or out.get("level1", {}).get("failed")}, indent=1))


if __name__ == "__main__":
    main()
