"""
Stage 0 of amendment A7: is a site a complete cut on single_face and the blind readout?

Mechanics on the 24 CREMA-D calibration clips only, which are outside every analysis set.
The job refuses any clip id not in research/reproduction/crema_calibration_clips.csv.

The checks are pure functions on arrays of unrounded word logprobs, shape (24, 106):

    C1  native against repeat                         determinism; on failure Stage 0 stops
    C2  exit cut on i against image_free on i         exit is image-free
    C3  exit cut on i against exit cut on j           exit ignores the image
    C4  all blocks held at i, pixels from j, against native on i   the blocks exhaust the site
    C5  per block, mean RMS of the 21 family CLRs of (hold from j on i - native i), over the
        mean RMS of (native j - native i): at least block_live_fraction, else the block is dead

C1 to C4 compare the worst absolute difference over every clip and every word with
completeness_tolerance_nats (A2's B1 rule). One word on one clip over the limit fails the
check. There is no partial credit. Missing, misshapen or non-finite input fails safe.

The region test (token map, cue token sets S_b, leak with a clip bootstrap and a floor on its
denominator) is also here. arms() maps a Stage 0 table to the Stage 1 arms it admits, by code.

The only complete cut this module can report is the exit. A partition into blocks that
passes C4 is reported as "blocks_exhaust_exit", never as a complete cut.

Usage on the cluster, after the freeze (one task per model):
    python -m scripts.a7_stage0 --model gemma4 --a7_frozen --output research/reproduction/a7_stage0_gemma4.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import a7_design  # noqa: E402

D = a7_design.load_design()
TOL = float(D["completeness_tolerance_nats"])
N_WORDS = int(D["n_word_logprobs"])
N_FAMILIES = int(D["n_families"])
N_CLIPS = int(D["n_calibration_clips"])
CLIPS_CSV = ROOT / "research" / "reproduction" / "crema_calibration_clips.csv"
COMPLETE_CHECKS = ("C1", "C2", "C3")


# ---------------------------------------------------------------------------
# Clip guard
# ---------------------------------------------------------------------------


def calibration_ids(path: Path = CLIPS_CSV) -> List[str]:
    """The 24 calibration clip ids in file order, from the pinned CSV only."""
    pinned = D["evidence_sha256"]["research/reproduction/crema_calibration_clips.csv"]
    if a7_design.file_sha256(path) != pinned:
        raise SystemExit(f"{path} is not the pinned calibration list.")
    with path.open(encoding="utf-8") as handle:
        ids = [row["clip_id"] for row in csv.DictReader(handle)]
    if len(ids) != N_CLIPS or len(set(ids)) != N_CLIPS:
        raise SystemExit(f"{path} lists {len(ids)} clips, not {N_CLIPS} distinct ones.")
    return ids


def check_clip_ids(ids: Sequence[str], path: Path = CLIPS_CSV) -> List[str]:
    """Refuses any clip id outside the calibration list. Returns the ids in file order."""
    allowed = calibration_ids(path)
    outside = [i for i in ids if i not in allowed]
    if outside:
        raise SystemExit(f"Clip ids outside crema_calibration_clips.csv refused: {outside[:5]}")
    if sorted(ids) != sorted(allowed):
        raise SystemExit("Stage 0 runs on all 24 calibration clips, no fewer.")
    return list(allowed)


def partner(index: int, n: int = N_CLIPS) -> int:
    """Clip i is paired with clip i + 1, the last with the first."""
    return (index + 1) % n


# ---------------------------------------------------------------------------
# C1 to C4
# ---------------------------------------------------------------------------


def as_cells(values: Any, n_clips: int = N_CLIPS) -> Optional[np.ndarray]:
    """A (clips, 106) float64 array, or None if it is missing, misshapen or not finite."""
    if values is None:
        return None
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.shape != (n_clips, N_WORDS) or not np.all(np.isfinite(array)):
        return None
    return array


def compare(a: Any, b: Any, n_clips: int = N_CLIPS, tolerance: float = TOL) -> Dict[str, Any]:
    """Worst absolute difference over every clip and word, against the tolerance."""
    left, right = as_cells(a, n_clips), as_cells(b, n_clips)
    if left is None or right is None:
        return {"worst_abs_nats": None, "pass": False, "reason": "missing_or_invalid_input"}
    diff = np.abs(left - right)
    clip, word = np.unravel_index(int(np.argmax(diff)), diff.shape)
    worst = float(diff[clip, word])
    n_over = int((diff > tolerance).sum())
    return {
        "worst_abs_nats": worst,
        "worst_clip_index": int(clip),
        "worst_word_index": int(word),
        "n_values_over_tolerance": n_over,
        "pass": bool(worst <= tolerance),
        "reason": None if worst <= tolerance else "over_tolerance",
    }


def rolled(values: Any, n_clips: int = N_CLIPS) -> Optional[np.ndarray]:
    """Row i of the result is the partner clip's row."""
    array = as_cells(values, n_clips)
    return None if array is None else array[[partner(i, n_clips) for i in range(n_clips)]]


def c1(native: Any, repeat: Any, n_clips: int = N_CLIPS) -> Dict[str, Any]:
    return compare(native, repeat, n_clips)


def c2(exit_cut: Any, image_free: Any, n_clips: int = N_CLIPS) -> Dict[str, Any]:
    return compare(exit_cut, image_free, n_clips)


def c3(exit_cut: Any, n_clips: int = N_CLIPS) -> Dict[str, Any]:
    return compare(exit_cut, rolled(exit_cut, n_clips), n_clips)


def c4(all_held: Any, native: Any, n_clips: int = N_CLIPS) -> Dict[str, Any]:
    return compare(all_held, native, n_clips)


# ---------------------------------------------------------------------------
# C5 and the family scale
# ---------------------------------------------------------------------------


def family_clr(word_logprobs: np.ndarray) -> np.ndarray:
    """The 21 family CLRs of each row, by A2's family_scores and clr."""
    from scripts import stimulus_interventions as si
    from scripts.run_stimulus_interventions import family_scores, lexicon_words

    words = lexicon_words()
    if len(words) != N_WORDS:
        raise SystemExit(f"A2's lexicon has {len(words)} words, not {N_WORDS}.")
    out = []
    for row in np.atleast_2d(word_logprobs):
        family = family_scores(words, row.tolist())
        out.append(si.clr([family[k] for k in sorted(family)]))
    array = np.array(out)
    if array.shape[1] != N_FAMILIES:
        raise SystemExit(f"A2's lexicon has {array.shape[1]} families, not {N_FAMILIES}.")
    return array


def mean_rms(a: np.ndarray, b: np.ndarray) -> float:
    """Mean over rows of the RMS over columns of (a - b)."""
    return float(np.mean(np.sqrt(((a - b) ** 2).mean(1))))


def c5_denominator(native: Any, n_clips: int = N_CLIPS) -> Optional[float]:
    """Mean RMS over the 21 family CLRs of (native j - native i), over the clips."""
    array = as_cells(native, n_clips)
    if array is None:
        return None
    clr = family_clr(array)
    return mean_rms(clr[[partner(i, n_clips) for i in range(n_clips)]], clr)


def c5(native: Any, holds: Mapping[str, Any], n_clips: int = N_CLIPS, fraction: float = float(D["block_live_fraction"])) -> Dict[str, Any]:
    """Each block's ratio of means, a point estimate. A block not shown live is dead."""
    base = as_cells(native, n_clips)
    denominator = c5_denominator(native, n_clips)
    out: Dict[str, Any] = {"denominator": denominator, "fraction": fraction, "blocks": {}}
    for block in ("A", "B"):
        hold = as_cells(holds.get(block), n_clips)
        if base is None or hold is None or denominator is None or not denominator > 0:
            out["blocks"][block] = {"ratio": None, "live": False, "reason": "missing_or_invalid_input"}
            continue
        ratio = mean_rms(family_clr(hold), family_clr(base)) / denominator
        out["blocks"][block] = {"ratio": ratio, "live": bool(ratio >= fraction), "reason": None if ratio >= fraction else "below_fraction"}
    return out


# ---------------------------------------------------------------------------
# Region test
# ---------------------------------------------------------------------------


def chebyshev(a: int, b: int, side: int) -> int:
    return max(abs(a // side - b // side), abs(a % side - b % side))


def token_map_check(found: Sequence[Optional[int]], expected: Sequence[int], side: int) -> Dict[str, Any]:
    """Share of trials whose most-changed token is within tolerance of the expected token."""
    tolerance = int(D["token_map_tolerance_tokens"])
    trials = int(D["n_calibration_clips"]) * int(D["token_map_positions"])
    hits = [f is not None and chebyshev(int(f), int(e), side) <= tolerance for f, e in zip(found, expected)]
    complete = len(found) == len(expected) == trials
    rate = float(np.mean(hits)) if hits else 0.0
    return {"n_trials": len(hits), "hit_rate": rate, "complete": complete,
            "pass": bool(complete and rate >= float(D["token_map_min_hit_rate"]))}


def expected_token(centre_px: Sequence[float], side: int, crop_px: float = float(D["crop_px"])) -> int:
    """The token whose footprint holds a pixel, on a side by side grid over the crop."""
    step = crop_px / side
    col, row = int(centre_px[0] // step), int(centre_px[1] // step)
    return row * side + col


def dilate(tokens: Sequence[int], b: int, side: int) -> List[int]:
    """Tokens within Chebyshev distance b on the grid of any given token."""
    chosen = set(int(t) for t in tokens)
    out = set()
    for t in chosen:
        r, c = divmod(t, side)
        for dr in range(-b, b + 1):
            for dc in range(-b, b + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < side and 0 <= cc < side:
                    out.add(rr * side + cc)
    return sorted(out)


def token_sets(mask_tokens: Sequence[Sequence[int]], side: int) -> Dict[int, List[List[int]]]:
    """S_b per clip, for every b in token_buffers."""
    return {int(b): [dilate(m, int(b), side) for m in mask_tokens] for b in D["token_buffers"]}


def admissible(sets: Mapping[int, Sequence[Sequence[int]]], n_tokens: int) -> List[int]:
    """Buffers whose S_b holds at most max_region_token_share of the tokens on every clip."""
    cap = float(D["max_region_token_share"])
    return sorted(b for b, per_clip in sets.items()
                  if per_clip and all(0 < len(s) and len(s) / n_tokens <= cap for s in per_clip))


def leak(v_x: np.ndarray, v_edit: np.ndarray, v_hold: np.ndarray, *, draws: int = int(D["bootstrap_draws"]),
         seed: int = int(D["seed"])) -> Dict[str, float]:
    """Ratio of mean RMS, with the upper limit of a 95% percentile clip bootstrap.

    The upper limit is the 97.5th percentile, the top of a two-sided 95% interval, the more
    conservative reading of "upper 95% limit".
    """
    num = np.sqrt(((v_hold - v_x) ** 2).mean(1))
    den = np.sqrt(((v_edit - v_x) ** 2).mean(1))
    point = float(num.mean() / den.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(draws, len(num)))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = num[idx].mean(1) / den[idx].mean(1)
    level = float(D["ci_decision"])
    upper = float(np.quantile(ratios, (1 + level) / 2))
    return {"point": point, "upper": upper}


def region_test(
    token_map: Optional[Mapping[str, Any]],
    sets: Optional[Mapping[int, Sequence[Sequence[int]]]],
    n_tokens: int,
    native: Any,
    edited: Any,
    holds: Optional[Mapping[int, Any]],
    c5_denom: Optional[float],
    n_clips: int = N_CLIPS,
    edit_failed: bool = False,
) -> Dict[str, Any]:
    """The cue token set: the smallest admissible b whose leak's upper limit is at most leak_limit."""
    out: Dict[str, Any] = {"pass": False, "b": None, "reason": None, "leaks": {}, "token_map": token_map}
    if token_map is None or token_map.get("pass") is not True:
        out["reason"] = "token_map_failed"
        return out
    base, edit = as_cells(native, n_clips), as_cells(edited, n_clips)
    if edit_failed or base is None or edit is None:
        out["reason"] = "edit_failed"
        return out
    v_x, v_edit = family_clr(base), family_clr(edit)
    effect = mean_rms(v_edit, v_x)
    floor = float(D["region_edit_min_fraction"]) * c5_denom if c5_denom is not None else None
    out.update({"edit_mean_rms": effect, "edit_floor": floor})
    if floor is None or not floor > 0 or effect < floor:
        out["reason"] = "edit_too_weak"
        return out
    allowed = admissible(sets or {}, n_tokens)
    out["admissible_buffers"] = allowed
    if not allowed:
        out["reason"] = "no_admissible_buffer"
        return out
    for b in allowed:
        hold = as_cells((holds or {}).get(b), n_clips)
        if hold is None:
            out["leaks"][str(b)] = None
            continue
        stats = leak(v_x, v_edit, family_clr(hold))
        out["leaks"][str(b)] = stats
        if stats["upper"] <= float(D["leak_limit"]):
            out.update({"pass": True, "b": b, "reason": None})
            return out
    out["reason"] = "leak_over_limit"
    return out


# ---------------------------------------------------------------------------
# The table and the arms it admits
# ---------------------------------------------------------------------------


def stage0_table(cells: Mapping[str, Any], n_clips: int = N_CLIPS, region: Optional[Mapping[str, Any]] = None,
                 meta: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Every check, its worst value and its result. Keys of cells:

    native, repeat, image_free, exit_cut, all_held, hold_A, hold_B (each clips by 106).
    """
    checks: Dict[str, Any] = {"C1": c1(cells.get("native"), cells.get("repeat"), n_clips)}
    table: Dict[str, Any] = {"amendment": "A7", "stage": 0, "design_sha256": a7_design.DESIGN_SHA256,
                             "tolerance_nats": TOL, "n_clips": n_clips, **(meta or {})}
    if not checks["C1"]["pass"]:
        table.update({"status": "stopped_c1", "checks": checks, "complete_cuts": [], "exit_complete": False,
                      "blocks_exhaust_exit": False, "region_test": None})
        table["arms"] = arms(table)
        return table
    checks["C2"] = c2(cells.get("exit_cut"), cells.get("image_free"), n_clips)
    checks["C3"] = c3(cells.get("exit_cut"), n_clips)
    checks["C4"] = c4(cells.get("all_held"), cells.get("native"), n_clips)
    checks["C5"] = c5(cells.get("native"), {"A": cells.get("hold_A"), "B": cells.get("hold_B")}, n_clips)
    exit_ok = all(checks[k]["pass"] for k in COMPLETE_CHECKS)
    table.update({
        "status": "complete",
        "checks": checks,
        "exit_complete": exit_ok,
        "complete_cuts": ["exit"] if exit_ok else [],
        "blocks_exhaust_exit": bool(exit_ok and checks["C4"]["pass"]),
        "region_test": dict(region) if region is not None else None,
    })
    table["arms"] = arms(table)
    return table


def _passed(table: Mapping[str, Any], key: str) -> bool:
    """A stored pass counts only if its stored worst difference is at most the tolerance too."""
    check = (table.get("checks") or {}).get(key) or {}
    worst = check.get("worst_abs_nats")
    return check.get("pass") is True and isinstance(worst, (int, float)) and math.isfinite(worst) and worst <= TOL


def exit_complete(table: Optional[Mapping[str, Any]]) -> bool:
    """Recomputed from the checks, never read from a stored flag. Missing means not shown."""
    return bool(table) and table.get("status") == "complete" and all(_passed(table, k) for k in COMPLETE_CHECKS)


def arms(table: Optional[Mapping[str, Any]]) -> Dict[str, bool]:
    """Which Stage 1 arms Stage 0 admits. C5 never decides this."""
    exit_ok = exit_complete(table)
    region = (table or {}).get("region_test") or {}
    return {
        "route": bool(exit_ok and _passed(table, "C4")),
        "token": bool(exit_ok and region.get("pass") is True and region.get("b") is not None),
    }


def dead_blocks(table: Optional[Mapping[str, Any]]) -> List[str]:
    """Blocks not shown live by C5. Missing evidence counts as dead."""
    blocks = (((table or {}).get("checks") or {}).get("C5") or {}).get("blocks") or {}
    return [b for b in ("A", "B") if (blocks.get(b) or {}).get("live") is not True]


def load_table(path: Path) -> Optional[Dict[str, Any]]:
    """A Stage 0 table, or None if the file is missing or not this design's."""
    if not path.exists():
        return None
    table = json.loads(path.read_text(encoding="utf-8"))
    if table.get("design_sha256") != a7_design.DESIGN_SHA256 or table.get("amendment") != "A7":
        return None
    return table


# ---------------------------------------------------------------------------
# The run: the same driver for the cluster and the CPU toys
# ---------------------------------------------------------------------------


@dataclass
class RegionSpec:
    """What the region test needs from the caller.

    paint(clip, centre) and edit(clip) return model inputs; mask_tokens(clip) returns the exit
    tokens whose footprint holds any pixel of the joint mask; expected(centre) the token a
    painted square should change most.
    """

    paint: Callable[[Any, Sequence[float]], Dict[str, Any]]
    edit: Callable[[Any], Dict[str, Any]]
    mask_tokens: Callable[[Any], Sequence[int]]
    expected: Callable[[Sequence[float]], int]
    centres: Sequence[Sequence[float]]
    side: int


class Fault(RuntimeError):
    """A fault in the run, not a result: the whole Stage 0 output for the model is rerun."""


def run_stage0(session: Any, clips: Sequence[Any], *, expected_tokens: int, region: Optional[RegionSpec] = None,
               meta: Optional[Mapping[str, Any]] = None, log: Callable[[str], None] = lambda s: None) -> Dict[str, Any]:
    """The 7 cells per clip (and 4 more for the region test), then the table."""
    n = len(clips)
    inputs, states = [], []
    cells: Dict[str, List[np.ndarray]] = {k: [] for k in ("native", "repeat")}
    for index, clip in enumerate(clips):
        x = session.encode(clip, "original")
        count = int(session.positions(x).numel())
        if count != expected_tokens:
            raise Fault(f"clip {index}: {count} image tokens, not {expected_tokens}")
        native, state = session.native(x)
        cells["native"].append(native)
        cells["repeat"].append(session.plain(x))
        inputs.append(x)
        states.append(state)
        log(f"native {index + 1}/{n}")
    first = c1(cells["native"], cells["repeat"], n)
    if not first["pass"]:
        return stage0_table({k: np.array(v) for k, v in cells.items()}, n, None, meta)
    for key in ("image_free", "exit_cut", "all_held", "hold_A", "hold_B"):
        cells[key] = []
    for index, x in enumerate(inputs):
        j = partner(index, n)
        free, placeholder = session.image_free(x)
        cells["image_free"].append(free)
        cells["exit_cut"].append(session.exit_cut(x, placeholder))
        cells["all_held"].append(session.hold(inputs[j], ("A", "B"), states[index]))
        cells["hold_A"].append(session.hold(x, ("A",), states[j]))
        cells["hold_B"].append(session.hold(x, ("B",), states[j]))
        log(f"cells {index + 1}/{n}")
    arrays = {k: np.array(v) for k, v in cells.items()}
    result = None
    if region is not None:
        result = run_region(session, clips, inputs, states, arrays["native"], region, n, log)
    return stage0_table(arrays, n, result, meta)


def run_region(session: Any, clips: Sequence[Any], inputs: Sequence[Dict[str, Any]], states: Sequence[Any],
               native: np.ndarray, region: RegionSpec, n: int, log: Callable[[str], None]) -> Dict[str, Any]:
    from scripts.a7_cuts import token_change

    found, expected = [], []
    for index, clip in enumerate(clips):
        base = session.exit_routes(inputs[index])
        for centre in region.centres:
            painted = session.exit_routes(region.paint(clip, centre))
            change = token_change(base, painted)
            found.append(int(np.argmax(change)) if change.size else None)
            expected.append(region.expected(centre))
    token_map = token_map_check(found, expected, region.side)
    n_tokens = region.side * region.side
    mask_tokens, edited, failed = [], [], False
    edits = []
    for index, clip in enumerate(clips):
        try:
            mask_tokens.append(list(region.mask_tokens(clip)))
            x_edit = region.edit(clip)
        except Exception as exc:  # an edit that cannot be made: the region test has not passed
            log(f"edit failed on clip {index}: {type(exc).__name__}: {exc}")
            failed = True
            break
        edits.append(x_edit)
        edited.append(session.plain(x_edit))
    sets = token_sets(mask_tokens, region.side) if not failed else {}
    holds: Dict[int, np.ndarray] = {}
    if not failed and token_map["pass"]:
        for b in admissible(sets, n_tokens):
            holds[b] = np.array([session.token_hold(edits[i], sets[b][i], states[i]) for i in range(n)])
            log(f"token hold b={b}")
    return region_test(token_map, sets, n_tokens, native, np.array(edited) if not failed else None, holds,
                       c5_denominator(native, n), n, edit_failed=failed)


# ---------------------------------------------------------------------------
# Cluster entry point: thin, not exercised locally
# ---------------------------------------------------------------------------


def _real_session(model_key: str):  # pragma: no cover - needs a GPU and model weights
    from scripts import run_stimulus_interventions as run
    from scripts.a7_cuts import Session
    from scripts.vision_gates import adapter_for, last_logits

    handle = run.load_handle(model_key, D["dtype"])
    adapter = adapter_for(model_key, handle.model)

    def readout(inputs: Dict[str, Any]) -> np.ndarray:
        # Unrounded: score_blind rounds to 4 decimals, so it is not used here.
        return np.array(run.score_sequences(handle, inputs, handle.blind_prefill, handle.word_ids), dtype=np.float64)

    def encode(image: np.ndarray) -> Dict[str, Any]:
        return run.encode_media(handle, run.Media("image", [image], [], {}), run.BLIND_PROMPT)

    def vision_pass(inputs: Dict[str, Any]) -> Any:
        # The token map needs only the exit routes: one forward, not a full blind readout
        # (independent audit 2026-11-08; the draft costs these as vision-only passes).
        return last_logits(handle.model, inputs)

    return handle, Session(adapter, model_key, lambda clip, _c: encode(clip["original"]), readout, D["route_blocks"][model_key],
                           vision_only=vision_pass), encode


def main() -> None:  # pragma: no cover - needs a GPU and model weights
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, choices=D["models"])
    parser.add_argument("--records", type=Path, required=True, help="Readout ladder records of the calibration clips.")
    parser.add_argument("--region", action="store_true", help="Also run the region test.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--a7_frozen", action="store_true")
    args = parser.parse_args()
    if not args.a7_frozen:
        raise SystemExit("A7 is not frozen. Stage 0 runs only after the freeze.")
    design = a7_design.load_design()
    a7_design.check_pins(design)
    pins = a7_design.check_code_pins()
    from scripts import stimulus_interventions as si
    from scripts.readout_ladder import face_views

    records = json.loads(args.records.read_text(encoding="utf-8"))
    by_id = {r["clip_id"]: r for r in records}
    ids = check_clip_ids(list(by_id))
    handle, session, encode = _real_session(args.model)
    clips = []
    for clip_id in ids:
        view, _shifted = face_views(by_id[clip_id])
        clips.append({"clip_id": clip_id, "original": view, "record": by_id[clip_id]})
    side = int(round(math.sqrt(D["image_tokens"][args.model])))
    region = None
    if args.region:
        def paint(clip: Dict[str, Any], centre: Sequence[float]) -> Dict[str, Any]:
            image = clip["original"].copy()
            half = int(D["token_map_square_px"]) // 2
            cx, cy = int(centre[0]), int(centre[1])
            image[max(0, cy - half):cy + half, max(0, cx - half):cx + half] = 128
            return encode(image)

        def edit_inputs(clip: Dict[str, Any]) -> Dict[str, Any]:
            record = clip["record"]
            frames, _ = si.read_frames(Path(record["path"]), [record["apex"]])
            landmarks = np.asarray(record["apex_landmarks"], dtype=np.float64)
            geometry = si.ClipGeometry()
            geometry.face_box = tuple(record["face_box"])
            geometry.face_width_px = float(record["face_width_px"])
            image, _log = si.apply_condition(D["stage0_edit"], frames[record["apex"]], landmarks, geometry)
            return encode(image)

        def mask_tokens(clip: Dict[str, Any]) -> List[int]:
            record = clip["record"]
            frames, _ = si.read_frames(Path(record["path"]), [record["apex"]])
            frame = frames[record["apex"]]
            landmarks = np.asarray(record["apex_landmarks"], dtype=np.float64)
            alpha = si.region_mask(frame.shape[:2], landmarks, "eyesmouth")
            crop = si.fixed_size(si.crop((alpha * 255).astype(np.uint8)[..., None].repeat(3, -1), tuple(record["face_box"])))
            hit = crop[..., 0] > 0
            step = int(D["crop_px"]) // side
            return [r * side + c for r in range(side) for c in range(side)
                    if hit[r * step:(r + 1) * step, c * step:(c + 1) * step].any()]

        region = RegionSpec(paint, edit_inputs, mask_tokens, lambda c: expected_token(c, side),
                            D["token_map_centres_px"], side)
    table = run_stage0(session, clips, expected_tokens=int(D["image_tokens"][args.model]), region=region,
                       meta={"model": args.model, "clip_ids": ids, "code_sha256": pins}, log=print)
    args.output.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"model": args.model, "complete_cuts": table["complete_cuts"], "arms": table["arms"]}))


if __name__ == "__main__":
    main()
