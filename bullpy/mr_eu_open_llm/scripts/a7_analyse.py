"""
Analysis for draft amendment A7: RQ3 from Stage 0, and the secondary decomposition (Stage 1).

RQ3 rests on Stage 0 only (the ruling on blocks, 2026-11-08). For each model it is "not
testable": beyond the interface if the exit is a complete cut, at every level if not. The
route and token arms are a secondary decomposition of the complete exit cut. Every row with
a block result is labelled "decomposition", never "cut".

Stage 1 estimands, per item, with y the CLR of the acted family by A2's clr_of:

    TE = y(A',B') - y(A,B)      PE_A = y(A',B) - y(A,B)      PE_B = y(A,B') - y(A,B)
    N_A = y(A',B') - y(A,B')    N_B = y(A',B') - y(A',B)     I = TE - PE_A - PE_B
    phi_A = (PE_A + N_A) / 2    phi_B = (PE_B + N_B) / 2     phi_A + phi_B = TE

Inference is A2's: actor means, one joint step-down maxT matrix over both models (17 rows,
at most 8 columns per model, two-sided), actor-bootstrap intervals and equivalence.

Verdict precedence: (1) Not testable (Stage 0 does not admit the arm, or a sentinel failed);
then Not run (no recorded decision to run it); (2) Not tested (a precondition failed);
(3) Inconclusive items_missing; (4) the effect class; (5) with the effect Present,
block_dead; (6) the carrying class. Riders are attached after and never change it.

Every constant comes from the design block through scripts/a7_design.py. main refuses until
A7 is frozen. scripts/a7_recovery.py runs analyse() unchanged on synthetic observers.

Usage, after the freeze and after Stage 1:
    python -m scripts.a7_analyse --a7_frozen --a2_frozen --inputs a7_inputs.json --output results/a7/a7_analysis.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from scripts import a7_design
from scripts import a7_stage0 as s0
from scripts import analyse_stimulus_interventions as an
from scripts import stimulus_interventions as si
from scripts.run_stimulus_interventions import OPTION_TO_FAMILY, select_items
from scripts.stimulus_design import Contrast

D = a7_design.load_design()
MODELS: Tuple[str, ...] = tuple(D["models"])
ROUTE = D["route"]
ALPHA = float(D["alpha"])
ORIGINAL = D["baseline_condition"]
EDIT = D["stage1_edit"]
FILLS: Tuple[str, ...] = tuple(D["other_fills"])
ARMS = ("route", "token")

# A2's row format, conditions named hold_*. A prime (block from the edit) is written "edit".
CELLS = {
    "route": {"A_edit": "hold_route_A_edit", "B_edit": "hold_route_B_edit"},
    "token": {"S_edit": "hold_token_S_edit", "rest_edit": "hold_token_rest_edit",
              "Q_edit": "hold_token_Q_edit", "Qrest_edit": "hold_token_Qrest_edit"},
}
SENTINEL = {"route": "hold_sentinel_route", "token": "hold_sentinel_token"}
BLOCK_NAMES = {"route": ("A", "B"), "token": ("S", "rest")}
COLUMNS = {"route": ("phi_A", "phi_B", "I"), "token": ("phi_S", "phi_rest", "I", "phi_S_minus_phi_Q")}

NOT_TESTABLE, NOT_RUN, NOT_TESTED = "Not testable", "Not run", "Not tested"
NO_EFFECT, INCONCLUSIVE = "No effect to carry", "Inconclusive"
CARRYING = {"A only": "Carried by A", "B only": "Carried by B", "Both": "Both blocks", "Opposed": "Opposed blocks"}
VERDICTS = (NOT_TESTABLE, NOT_RUN, NOT_TESTED, NO_EFFECT, INCONCLUSIVE, *CARRYING.values())
REASONS = ("te_unresolved", "carrying_unresolved", "block_dead", "items_missing")
RIDERS = ("Additive", "Redundant", "Synergic", "Interaction unresolved")
LABEL = "decomposition"
HEADING = "decomposition of the complete exit cut"


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------


def stage1_items(items: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The 200 items: all but the flagged labels and Neutral, in A2's canonical order."""
    kept = [dict(i) for i in items if i["label"] not in D["excluded_labels"]]
    return select_items(kept, None)


def sentinel_items(ordered: Sequence[Mapping[str, Any]]) -> List[str]:
    """Positions 0, 10, 20 and so on of the canonical order."""
    return [i["trial_id"] for k, i in enumerate(ordered) if k % int(D["sentinel_every"]) == 0]


def actors_of(items: Sequence[Mapping[str, Any]]) -> List[str]:
    return sorted({i["actor"] for i in items})


# ---------------------------------------------------------------------------
# Estimands
# ---------------------------------------------------------------------------


def terms(y00: np.ndarray, y11: np.ndarray, y10: np.ndarray, y01: np.ndarray) -> Dict[str, np.ndarray]:
    """y00 = y(A,B), y11 = y(A',B'), y10 = y(A',B), y01 = y(A,B')."""
    te = y11 - y00
    pe_a, pe_b = y10 - y00, y01 - y00
    n_a, n_b = y11 - y01, y11 - y10
    return {"TE": te, "PE_A": pe_a, "PE_B": pe_b, "N_A": n_a, "N_B": n_b, "I": te - pe_a - pe_b,
            "phi_A": (pe_a + n_a) / 2, "phi_B": (pe_b + n_b) / 2}


def family_of(cells: an.Cells, trial: str, target: str) -> Optional[str]:
    item = cells.items[trial]
    return OPTION_TO_FAMILY.get(item["human_modal_label"] if target == "human_mode" else item["label"])


def score(cells: an.Cells, trial: str, condition: str, target: str) -> Any:
    """y: the CLR of the acted family by A2's clr_of; for target "whole", all 21 family CLRs."""
    row = cells.ok[(trial, ROUTE, condition)]
    if target == "whole":
        fams = sorted(row["blind"]["family_logprob"])
        return si.clr([row["blind"]["family_logprob"][f] for f in fams])
    return an.clr_of(row, family_of(cells, trial, target))


def complete_items(cells: an.Cells, trials: Sequence[str], conditions: Sequence[str], target: str = "acted") -> List[str]:
    """Items with every named cell. An item missing any leaves every term, and is counted."""
    return [t for t in trials if t in cells.items and all((t, ROUTE, c) in cells.ok for c in conditions)
            and (target == "whole" or family_of(cells, t, target) is not None)]


def item_table(cells: an.Cells, trials: Sequence[str], arm: Optional[str], target: str = "acted") -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Item-level terms for the endpoints (arm None) or for an arm.

    For target "whole" each term is the RMS over the 21 family CLRs of that term's vector, a
    sensitivity in which phi_A + phi_B = TE no longer holds.
    """
    names = {} if arm is None else CELLS[arm]
    kept = complete_items(cells, trials, (ORIGINAL, EDIT, *names.values()), target)
    y = {k: np.array([score(cells, t, c, target) for t in kept]) for k, c in
         (("00", ORIGINAL), ("11", EDIT), *((k, v) for k, v in names.items()))}
    if not kept:
        y = {k: np.zeros(0) for k in y}

    def fold(values: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if target != "whole":
            return values
        return {k: (np.sqrt((v ** 2).mean(1)) if v.ndim == 2 else v) for k, v in values.items()}

    if arm is None:
        return kept, fold({"TE": y["11"] - y["00"]})
    if arm == "route":
        out = terms(y["00"], y["11"], y["A_edit"], y["B_edit"])
        return kept, fold({"phi_A": out["phi_A"], "phi_B": out["phi_B"], "I": out["I"], "TE": out["TE"],
                           "PE_A": out["PE_A"], "PE_B": out["PE_B"], "N_A": out["N_A"], "N_B": out["N_B"]})
    s = terms(y["00"], y["11"], y["S_edit"], y["rest_edit"])
    q = terms(y["00"], y["11"], y["Q_edit"], y["Qrest_edit"])
    return kept, fold({"phi_S": s["phi_A"], "phi_rest": s["phi_B"], "I": s["I"], "TE": s["TE"],
                       "phi_S_minus_phi_Q": s["phi_A"] - q["phi_A"], "PE_S": s["PE_A"], "PE_rest": s["PE_B"],
                       "N_S": s["N_A"], "N_rest": s["N_B"], "phi_Q": q["phi_A"]})


def actor_means(values: np.ndarray, trial_actors: Sequence[str], actors: Sequence[str]) -> np.ndarray:
    """Mean over each actor's items; NaN for an actor with none. Each actor counts once."""
    groups: Dict[str, List[float]] = defaultdict(list)
    for value, actor in zip(values.tolist(), trial_actors):
        groups[actor].append(value)
    return np.array([float(np.mean(groups[a])) if a in groups else np.nan for a in actors])


# ---------------------------------------------------------------------------
# Inputs that are not Stage 1 cells
# ---------------------------------------------------------------------------


def a2_margin(a2_output: Optional[Mapping[str, Any]]) -> Optional[float]:
    """A2's own TE margin on single_face (10% of mismatch_margin_basis), read, not recomputed."""
    value = ((a2_output or {}).get(ROUTE) or {}).get("margin")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def a2_gates(a2_output: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(((a2_output or {}).get(ROUTE) or {}).get("gates") or {})


def preconditions_met(gates: Mapping[str, Any]) -> bool:
    """RQ0, G2 to G4, Gate B, delivery and sham, each exactly True. Missing has not passed."""
    return an.gates_passed(gates)


def fill_means(a2_cells: Optional[an.Cells], items: Sequence[Mapping[str, Any]], actors: Sequence[str]) -> Dict[str, Optional[float]]:
    """Mean actor effect of each other fill against original on A2's cells, by A2's actor_effects."""
    out: Dict[str, Optional[float]] = {}
    keep = {i["trial_id"] for i in items}
    labels = tuple(sorted({i["label"] for i in items}))
    for fill in FILLS:
        if a2_cells is None:
            out[fill] = None
            continue
        contrast = Contrast(f"A7_{fill}", ROUTE, labels, (fill,), b=(ORIGINAL,))
        exclude = [t for t in a2_cells.items if t not in keep]
        values = an.actor_effects(a2_cells, contrast, actors, exclude=exclude)
        out[fill] = float(np.nanmean(values)) if np.any(~np.isnan(values)) else None
    return out


def sentinel_check(cells: an.Cells, arm: str, sentinels: Sequence[str]) -> Dict[str, Any]:
    """Every sentinel equals the same job's unrounded native cell within the tolerance, on all 106 words."""
    worst, failed = 0.0, []
    for trial in sentinels:
        native = cells.ok.get((trial, ROUTE, ORIGINAL))
        held = cells.ok.get((trial, ROUTE, SENTINEL[arm]))
        a = None if native is None else s0.as_cells([native["blind"].get("word_logprob")], 1)
        b = None if held is None else s0.as_cells([held["blind"].get("word_logprob")], 1)
        if a is None or b is None:
            failed.append({"trial_id": trial, "reason": "missing"})
            continue
        gap = float(np.abs(a - b).max())
        worst = max(worst, gap)
        if gap > s0.TOL:
            failed.append({"trial_id": trial, "reason": "over_tolerance", "worst_abs_nats": gap})
    return {"n": len(sentinels), "worst_abs_nats": worst, "failed": failed, "pass": bool(sentinels) and not failed}


def block_changes_live(cells: an.Cells, arm: str, trials: Sequence[str]) -> bool:
    """Each block's relative L2 change under the edit is above zero on every item. Missing is not shown."""
    for trial in trials:
        row = cells.ok.get((trial, ROUTE, EDIT))
        change = ((row or {}).get("block_change") or {}).get(arm) or {}
        for block in BLOCK_NAMES[arm]:
            value = change.get(block)
            if value is None or not (float(value) > 0):
                return False
    return True


def arm_ran(cells: Optional[an.Cells], arm: str) -> bool:
    if cells is None:
        return False
    wanted = set(CELLS[arm].values()) | {SENTINEL[arm]}
    return any(r.get("condition") in wanted for r in cells.rows)


# ---------------------------------------------------------------------------
# Statistics on one column
# ---------------------------------------------------------------------------


@dataclass
class Column:
    values: np.ndarray
    p: float

    @property
    def mean(self) -> float:
        present = self.values[~np.isnan(self.values)]
        return float(present.mean()) if present.size else float("nan")

    def ci(self, level: float) -> Optional[Tuple[float, float]]:
        return an.actor_interval(self.values, level, draws=int(D["bootstrap_draws"]), seed=int(D["seed"]))

    def passes(self) -> bool:
        """Step-down p below alpha and a 95% interval that excludes 0."""
        interval = self.ci(float(D["ci_decision"]))
        return bool(self.p < ALPHA and interval is not None and (interval[0] > 0 or interval[1] < 0))

    def inside(self, margin: Optional[float]) -> bool:
        """The 90% interval inside the margin, by A2's equivalence_pass."""
        if margin is None or not math.isfinite(margin) or margin <= 0:
            return False
        return an.equivalence_pass(self.values, margin, draws=int(D["bootstrap_draws"]), seed=int(D["seed"]))

    def summary(self) -> Dict[str, Any]:
        return {"mean": self.mean, "p_maxt": self.p, "ci95": self.ci(float(D["ci_decision"])),
                "ci90": self.ci(float(D["ci_equivalence"])), "n_actors": int((~np.isnan(self.values)).sum())}


def same_sign(a: float, b: float) -> bool:
    return bool(np.sign(a) == np.sign(b) and a != 0)


def effect_class(te: Column, margin: Optional[float], fills: Mapping[str, Optional[float]]) -> str:
    """Present, Absent or Unresolved. A missing fill cannot be Present; a missing margin cannot be Absent."""
    fills_agree = all(fills.get(f) is not None and same_sign(float(fills[f]), te.mean) for f in FILLS)
    if te.passes() and fills_agree:
        return "Present"
    if te.p >= ALPHA and te.inside(margin):
        return "Absent"
    return "Unresolved"


def carrying_class(a: Column, b: Column, te_mean: float, share_margin: float) -> str:
    """Rows in table order; the first that applies is the class."""
    pass_a, pass_b = a.passes(), b.passes()
    dir_a, dir_b = same_sign(a.mean, te_mean), same_sign(b.mean, te_mean)
    if pass_a and dir_a and b.inside(share_margin):
        return "A only"
    if pass_b and dir_b and a.inside(share_margin):
        return "B only"
    if pass_a and dir_a and pass_b and dir_b:
        return "Both"
    if (pass_a and dir_a and pass_b and not dir_b) or (pass_b and dir_b and pass_a and not dir_a):
        return "Opposed"
    return "Unresolved"


def interaction_rider(i: Column, te_mean: float, share_margin: float) -> str:
    if i.inside(share_margin):
        return "Additive"
    if i.passes() and not same_sign(i.mean, te_mean):
        return "Redundant"
    if i.passes() and same_sign(i.mean, te_mean):
        return "Synergic"
    return "Interaction unresolved"


# ---------------------------------------------------------------------------
# One analysis over both models
# ---------------------------------------------------------------------------


@dataclass
class ModelInputs:
    """Everything one model contributes. A missing piece is None and fails safe."""

    stage1_rows: Optional[Sequence[Dict[str, Any]]]
    stage0: Optional[Mapping[str, Any]]
    a2_output: Optional[Mapping[str, Any]]
    a2_rows: Optional[Sequence[Dict[str, Any]]]
    decisions: Optional[Mapping[str, Any]]


def rq3_row(model: str, table: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """RQ3 itself, from Stage 0 alone. It is never testable beyond the interface here."""
    exit_ok = s0.exit_complete(table)
    return {
        "model": model,
        "rq3": "not testable",
        "scope": "beyond the interface" if exit_ok else "at every level",
        "complete_cuts": ["exit"] if exit_ok else [],
        "complete_cut_contrast": "zero by construction (validity, not a finding)" if exit_ok else None,
        "stage0_status": None if table is None else table.get("status"),
        "a5_w_caveat": not exit_ok,
    }


def analyse(items: Sequence[Mapping[str, Any]], inputs: Mapping[str, ModelInputs], *, exclude: Sequence[str] = (),
            target: str = "acted") -> Dict[str, Any]:
    """The whole A7 result: RQ3 per model, and every arm's verdict with its terms.

    exclude and target exist for the sensitivities only (target: acted, human_mode or whole).
    Sentinel positions are always those of the full 200-item order.
    """
    full = stage1_items(items)
    sentinels = sentinel_items(full)
    dropped = set(exclude)
    ordered = [i for i in full if i["trial_id"] not in dropped]
    trials = [i["trial_id"] for i in ordered]
    actors = actors_of(full)
    trial_actor = {i["trial_id"]: i["actor"] for i in ordered}
    min_items = int(math.ceil(float(D["min_item_share"]) * len(ordered)))

    prepared: Dict[str, Dict[str, Any]] = {}
    columns: List[Tuple[str, str, str, np.ndarray]] = []
    for model in MODELS:
        given = inputs.get(model) or ModelInputs(None, None, None, None, None)
        cells = an.Cells(given.stage1_rows) if given.stage1_rows is not None else None
        admitted = s0.arms(given.stage0)
        gates = a2_gates(given.a2_output)
        state: Dict[str, Any] = {"cells": cells, "given": given, "admitted": admitted, "gates": gates, "arms": {}}
        model_columns: List[Tuple[str, str, str, np.ndarray]] = []
        for arm in ARMS:
            decision = ((given.decisions or {}).get(arm)) or {}
            ran = arm_ran(cells, arm)
            sentinel = sentinel_check(cells, arm, sentinels) if ran else None
            info: Dict[str, Any] = {"decision": decision, "ran": ran, "sentinel": sentinel}
            if not admitted[arm]:
                info["stop"] = (NOT_TESTABLE, "stage0_not_shown")
            elif ran and not sentinel["pass"]:
                info["stop"] = (NOT_TESTABLE, "sentinel_failed")
            elif decision.get("run") is not True:
                info["stop"] = (NOT_RUN, "no_recorded_decision" if not decision else "decided_not_to_run")
            elif not preconditions_met(gates):
                info["stop"] = (NOT_TESTED, "precondition_failed")
            else:
                info["stop"] = None
                kept, values = item_table(cells, trials, arm, target) if cells is not None else ([], {})
                info["kept"], info["values"] = kept, values
                info["n_complete"] = len(kept)
                if len(kept) >= 2:
                    for name in COLUMNS[arm]:
                        model_columns.append((model, arm, name, actor_means(values[name], [trial_actor[t] for t in kept], actors)))
            state["arms"][arm] = info
        if model_columns and cells is not None:
            # The TE column exists only beside at least one arm's columns, so the family is never larger than it needs.
            kept, values = item_table(cells, trials, None, target)
            state["te_kept"] = kept
            columns.append((model, "endpoints", "TE", actor_means(values["TE"], [trial_actor[t] for t in kept], actors)))
            columns.extend(model_columns)
        prepared[model] = state

    p_of: Dict[Tuple[str, str, str], float] = {}
    family = None
    if columns:
        matrix = np.column_stack([c[3] for c in columns])
        family = si.signflip_maxt(matrix, alternative=[0] * len(columns))
        for index, (model, arm, name, _v) in enumerate(columns):
            p_of[(model, arm, name)] = float(family["p_maxt"][index])
    col_of = {(m, a, n): Column(v, p_of[(m, a, n)]) for m, a, n, v in columns}

    out: Dict[str, Any] = {
        "amendment": "A7",
        "design_sha256": a7_design.DESIGN_SHA256,
        "target": target,
        "n_excluded": len(full) - len(ordered),
        "rq3": {m: rq3_row(m, prepared[m]["given"].stage0) for m in MODELS},
        "family": {"columns": [f"{m}/{a}/{n}" for m, a, n, _ in columns],
                   "n_flips": None if family is None else family["n_flips"],
                   "exact": None if family is None else family["exact"]},
        "models": {},
    }
    for model in MODELS:
        state = prepared[model]
        te = col_of.get((model, "endpoints", "TE"))
        margin = a2_margin(state["given"].a2_output)
        a2_cells = an.Cells(state["given"].a2_rows) if state["given"].a2_rows is not None else None
        fills = fill_means(a2_cells, ordered, actors) if te is not None else {f: None for f in FILLS}
        effect = effect_class(te, margin, fills) if te is not None else None
        share_margin = float(D["share_margin_fraction"]) * abs(te.mean) if te is not None else None
        model_out: Dict[str, Any] = {
            "stage0_arms_admitted": state["admitted"],
            "dead_blocks_stage0": s0.dead_blocks(state["given"].stage0),
            "preconditions": {k: state["gates"].get(k) for k in an.GATE_KEYS},
            "te": None if te is None else {**te.summary(), "n_items": len(state.get("te_kept", []))},
            "te_margin": margin,
            "fills": fills,
            "effect_class": effect,
            "share_margin": share_margin,
            "arms": {},
        }
        for arm in ARMS:
            model_out["arms"][arm] = arm_verdict(model, arm, state, te, effect, share_margin, col_of, min_items, len(ordered),
                                                 [trial_actor[t] for t in state["arms"][arm].get("kept", [])], actors)
        out["models"][model] = model_out
    check_rows(out)
    return out


def arm_verdict(model: str, arm: str, state: Mapping[str, Any], te: Optional[Column], effect: Optional[str],
                share_margin: Optional[float], col_of: Mapping[Tuple[str, str, str], Column], min_items: int,
                n_items: int, kept_actors: Sequence[str] = (), actors: Sequence[str] = ()) -> Dict[str, Any]:
    info = state["arms"][arm]
    row: Dict[str, Any] = {"model": model, "arm": arm, "kind": "block", "label": LABEL, "heading": HEADING,
                           "verdict": None, "reason": None, "riders": [], "reading": None,
                           "decision": info.get("decision") or None, "sentinel": info.get("sentinel")}
    if info.get("stop") is not None:
        row["verdict"], row["reason"] = info["stop"]
        return row
    row["n_items_complete"] = info.get("n_complete", 0)
    row["n_items"] = n_items
    if info.get("n_complete", 0) < min_items or te is None or any((model, arm, n) not in col_of for n in COLUMNS[arm]):
        row["verdict"], row["reason"] = INCONCLUSIVE, "items_missing"
        cells = state["cells"]
        row["missing_items"] = sorted(set(t for t in (cells.items if cells is not None else {})) - set(info.get("kept", [])))[:50]
        return row
    cols = {n: col_of[(model, arm, n)] for n in COLUMNS[arm]}
    row["terms"] = {n: c.summary() for n, c in cols.items()}
    values = info["values"]
    row["riders_descriptive"] = rider_summary(values, kept_actors, actors, state["cells"], arm, info["kept"])
    if effect == "Absent":
        row["verdict"] = NO_EFFECT
        return row
    if effect != "Present":
        row["verdict"], row["reason"] = INCONCLUSIVE, "te_unresolved"
        return row
    a_name, b_name = COLUMNS[arm][0], COLUMNS[arm][1]
    dead = (arm == "route" and bool(s0.dead_blocks(state["given"].stage0))) or not block_changes_live(state["cells"], arm, info["kept"])
    row["riders"].append(interaction_rider(cols["I"], te.mean, share_margin))
    carrying = carrying_class(cols[a_name], cols[b_name], te.mean, share_margin)
    row["carrying_class"] = carrying
    if dead:
        row["verdict"], row["reason"] = INCONCLUSIVE, "block_dead"
        return row
    if carrying == "Unresolved":
        row["verdict"], row["reason"] = INCONCLUSIVE, "carrying_unresolved"
        return row
    row["verdict"] = CARRYING[carrying]
    if arm == "token":
        diff = cols["phi_S_minus_phi_Q"]
        local = diff.passes() and same_sign(diff.mean, te.mean)
        if not local:
            row["riders"].append("not_local")
        if carrying == "A only":
            row["reading"] = "carried by cue tokens" if local else "no spatial specificity over a same-size random set"
    return row


def rider_summary(values: Mapping[str, np.ndarray], kept_actors: Sequence[str], actors: Sequence[str],
                  cells: Optional[an.Cells], arm: str, kept: Sequence[str]) -> Dict[str, Any]:
    """PE and N terms with actor-bootstrap 95% intervals, and each block's relative change under the edit."""
    out: Dict[str, Any] = {}
    for name, v in values.items():
        if name in COLUMNS[arm]:
            continue
        means = actor_means(v, kept_actors, actors)
        out[name] = {"mean": float(np.nanmean(means)) if np.any(~np.isnan(means)) else None,
                     "ci95": an.actor_interval(means, float(D["ci_decision"]), draws=int(D["bootstrap_draws"]), seed=int(D["seed"]))}
    changes: Dict[str, List[float]] = {b: [] for b in BLOCK_NAMES[arm]}
    for trial in kept:
        row = cells.ok.get((trial, ROUTE, EDIT)) if cells is not None else None
        change = ((row or {}).get("block_change") or {}).get(arm) or {}
        for block in BLOCK_NAMES[arm]:
            if change.get(block) is not None:
                changes[block].append(float(change[block]))
    out["block_relative_change"] = {b: ({"min": min(v), "median": float(np.median(v)), "n": len(v)} if v else None)
                                    for b, v in changes.items()}
    return out


def check_rows(out: Mapping[str, Any]) -> None:
    """An output row with no verdict or reason code is a code fault and stops the analysis."""
    for model, body in out["models"].items():
        for arm, row in body["arms"].items():
            if row["verdict"] not in VERDICTS:
                raise SystemExit(f"{model}/{arm}: no verdict.")
            if row["verdict"] == INCONCLUSIVE and row["reason"] not in REASONS:
                raise SystemExit(f"{model}/{arm}: Inconclusive with no reason code.")
            if row["kind"] != "block" or row["label"] != LABEL:
                raise SystemExit(f"{model}/{arm}: a block result must be labelled {LABEL!r}.")


def sensitivities(items: Sequence[Mapping[str, Any]], inputs: Mapping[str, ModelInputs],
                  token_floor: Optional[Mapping[str, Sequence[str]]] = None) -> Dict[str, Any]:
    """Reported beside the primary; none changes a verdict. No new forward."""
    from scripts.stimulus_design import SMOKE_ITEMS

    full = stage1_items(items)
    flagged = sorted({r["trial_id"] for given in inputs.values() for r in (given.stage1_rows or []) if an.align_flagged(r)})
    mode_wrong = [i["trial_id"] for i in full if i["human_modal_label"] != i["label"]]
    canonical = {m: ModelInputs(an.canonical_rows(g.stage1_rows) if g.stage1_rows is not None else None, g.stage0,
                                g.a2_output, g.a2_rows, g.decisions) for m, g in inputs.items()}
    out = {
        "without_smoke_items": analyse(items, inputs, exclude=SMOKE_ITEMS),
        "without_align_flagged": analyse(items, inputs, exclude=flagged),
        "items_whose_human_mode_is_the_target": analyse(items, inputs, exclude=mode_wrong),
        "outcome_against_human_modal_label": analyse(items, inputs, target="human_mode"),
        "canonical_word_only": analyse(items, canonical),
        "whole_judgement_rms": analyse(items, inputs, target="whole"),
    }
    if token_floor is not None:
        small = sorted({t for v in token_floor.values() for t in v})
        out["clips_at_8_tokens_or_more"] = {**analyse(items, inputs, exclude=small), "share_excluded": len(small) / len(full)}
    return out


# ---------------------------------------------------------------------------
# Entry point: refuses until frozen
# ---------------------------------------------------------------------------


def _jsonl(paths: Sequence[str]) -> List[Dict[str, Any]]:
    rows = []
    for path in paths:
        rows.extend(json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())
    return rows


def main() -> None:  # pragma: no cover - runs on real outputs only after the freeze
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--inputs", type=Path, required=True,
                        help="JSON: per model, stage0 path, stage1 cell paths, A2 analysis output path, A2 cell paths, decisions.")
    parser.add_argument("--manifest", type=Path, default=a7_design.ROOT / "data/eu_emotions_full_manifest.json")
    parser.add_argument("--lookup", type=Path, default=a7_design.ROOT / "data/eu_emotions_human_entropy.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--a7_frozen", action="store_true")
    parser.add_argument("--a2_frozen", action="store_true")
    args = parser.parse_args()
    if not (args.a7_frozen and args.a2_frozen):
        raise SystemExit("A7 is not frozen. No A2 cell is read for A7 before that.")
    design = a7_design.load_design()
    a7_design.check_pins(design)
    pins = a7_design.check_code_pins()
    from scripts.run_stimulus_interventions import load_items

    items = [{k: i[k] for k in ("trial_id", "actor", "label", "human_modal_label", "weak_label")}
             for i in load_items(args.manifest, args.lookup, Path("."))]
    spec = json.loads(args.inputs.read_text(encoding="utf-8"))
    inputs = {}
    for model in MODELS:
        given = spec.get(model) or {}
        stage0 = s0.load_table(Path(given["stage0"])) if given.get("stage0") else None
        a2_output = json.loads(Path(given["a2_output"]).read_text(encoding="utf-8")) if given.get("a2_output") else None
        inputs[model] = ModelInputs(
            _jsonl(given["stage1_cells"]) if given.get("stage1_cells") else None,
            stage0,
            a2_output,
            _jsonl(given["a2_cells"]) if given.get("a2_cells") else None,
            given.get("decisions"),
        )
    out = analyse(items, inputs)
    out["sensitivities"] = sensitivities(items, inputs, spec.get("below_token_floor"))
    out["code_sha256"] = pins
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
