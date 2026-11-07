"""
Rule 7 for A5: synthetic observers with known truth through the real analysis.

Each observer writes `original` cells and baseline rows in A2's format, over the real 243
items, 17 actors, labels and Gate B partners, with made-up scores. No model output, clip
or result is read. Noise has a shared actor part, an item part and correlated families,
and each actor has a portrayal gain, because A2's recovery had independent noise and large
effects and so showed mechanics, not power.

Every replicate is a full family: the observer's cell plus `companion_cells` cells with no
truth on either test, so the step-down runs over 8 columns, the most it can in A5, against
columns that give it nothing. That is the worst case for power.

Pass rule and sizes are in the design block. They were written before any rate was seen. A
40-replicate pilot then ran for mechanics and runtime, the block's recovery settings did not
change after it, and the draft says what it printed. A failure is reported, not tuned away.

Two departures, here only: workers swap A1's two statistics loops for cached versions of the
same numbers, which tests/test_a5.py checks, and companion cells use `companion_shuffles`,
except in `all_null_family`, which uses the real analysis's count.

The false-Even, boundary, inside-Even and all-null observers and the wider calibration were
added after run 1 failed (research/reproduction/a5_recovery_run1.json). The draft says why.

Usage:
    python -m scripts.a5_recovery --workers 3 --output research/reproduction/a5_recovery.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import a5_analyse as a5  # noqa: E402
from scripts import a5_design  # noqa: E402
from scripts import analyse_stimulus_interventions as an  # noqa: E402

D = a5.D
R = D["recovery"]
F = len(a5.FAMILIES)
ROUTE = "single_face"
GATES_PASSED = {"rq0": True, "g2_g4": True, "delivery": True}
EVIDENCE_VERDICTS = ("Evidence-led", "Default-led with evidence", "Even")
NAMED = tuple(dict.fromkeys(a5.VERDICTS.values()))


class World:
    """The real items, and the fixed vectors every replicate of every observer shares."""

    def __init__(self, items: Sequence[Mapping[str, Any]]) -> None:
        self.items = [dict(i) for i in items]
        self.actors = sorted({i["actor"] for i in items})
        self.actor = np.array([self.actors.index(i["actor"]) for i in items])
        self.acted = np.array([a5.FAMILIES.index(a5.OPTION_TO_FAMILY[i["label"]]) for i in items])
        self.n = len(items)
        self.partners = a5.partners_for({i["trial_id"]: i for i in items})
        rng = np.random.default_rng(R["seed"])
        prior = rng.normal(0, 1, F)
        self.prior = (prior - prior.mean()) / prior.std()
        self.loadings = rng.normal(0, 1, (F, R["family_factors"]))
        self.loadings /= np.linalg.norm(self.loadings, axis=1, keepdims=True)
        self.wrong = (np.arange(F) + rng.integers(1, F)) % F  # one fixed wrong family per family
        self.spill = np.array([rng.choice(np.delete(np.arange(F), f), R["spill_neighbours"], replace=False) for f in range(F)])

    def family_noise(self, rng: np.random.Generator, rows: int) -> np.ndarray:
        """Unit-variance noise per family, with a shared share across correlated families."""
        share = R["family_factor_share"]
        common = rng.normal(0, 1, (rows, self.loadings.shape[1])) @ self.loadings.T
        return np.sqrt(share) * common + np.sqrt(1 - share) * rng.normal(0, 1, (rows, F))

    def scores(self, spec: Mapping[str, Any], rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Log family scores for one replicate, and the image-free and grey log scores."""
        n_actors = len(self.actors)
        s = spec["prior"] * self.prior[None, :] + spec["item_sd"] * self.family_noise(rng, self.n)
        s = s + (spec["actor_sd"] * self.family_noise(rng, n_actors))[self.actor]
        if spec.get("identity"):
            s = s + (spec["identity"] * self.family_noise(rng, n_actors))[self.actor]
        if spec["signal"]:
            gain = np.exp(rng.normal(0, spec["gain"], n_actors))[self.actor] * spec["signal"]
            rows = np.arange(self.n)
            if spec["mode"] == "wrong":
                s[rows, self.wrong[self.acted]] += gain
            else:
                s[rows, self.acted] += gain
            if spec["mode"] == "spread":
                for k in range(self.spill.shape[1]):
                    s[rows, self.spill[self.acted, k]] += R["spill_share"] * gain
        free = (spec["prior"] if spec["w"] == "prior" else 0.0) * self.prior
        w = free + R["baseline_noise"] * rng.normal(0, 1, F)
        g = free + R["baseline_noise"] * rng.normal(0, 1, F)
        return s, w, g

    def rows(self, s: np.ndarray) -> List[Dict[str, Any]]:
        out = []
        for item, values in zip(self.items, s.tolist()):
            out.append({**item, "input_name": ROUTE, "condition": "original", "status": "ok",
                        "blind": {"family_logprob": dict(zip(a5.FAMILIES, values)), "lexicon_mass": 0.5}})
        return out

    def report(self, w: np.ndarray, g: np.ndarray) -> Dict[str, Any]:
        return {"baselines": [
            {"input_name": ROUTE, "condition": c, "status": "ok", "family_logprob": dict(zip(a5.FAMILIES, v.tolist())),
             "lexicon_mass": 0.5} for c, v in (("image_free", w), ("grey", g))]}

    def judgement(self, s: np.ndarray) -> a5.Judgement:
        labels = tuple(sorted({i["label"] for i in self.items}))
        return a5.Judgement(tuple(i["trial_id"] for i in self.items), a5.softmax_of_log(s), self.actor,
                            np.array([labels.index(i["label"]) for i in self.items]), self.acted,
                            tuple(self.actors), labels, np.full(self.n, 0.5))


# ---------------------------------------------------------------------------
# Hitting a stated Phi
# ---------------------------------------------------------------------------


def base_spec(**overrides: Any) -> Dict[str, Any]:
    return {"prior": 0.0, "signal": 0.0, "mode": "acted", "identity": 0.0, "actor_sd": R["actor_sd"], "item_sd": R["item_sd"],
            "gain": R["portrayal_gain_log_sd"], "w": "prior", **overrides}


def shares(world: World, spec: Mapping[str, Any]) -> Tuple[float, float]:
    """Mean Phi and E/T over a fixed set of draws."""
    totals = np.zeros(3)
    for k in range(R["calibration_draws"]):
        s, _, _ = world.scores(spec, np.random.default_rng([R["seed"], 7, k]))
        parts = a5.decompose(world.judgement(s))
        rows = {x: np.nanmean(a5.actor_means(parts[x], world.actor, len(world.actors))) for x in ("T", "D", "E")}
        totals += [rows["D"], rows["E"], rows["T"]]
    return float(totals[0] / totals[2]), float(totals[1] / totals[2])


def prior_for_phi(world: World, target: float, spec: Mapping[str, Any]) -> Dict[str, Any]:
    """The prior scale that puts Phi at target, by bisection. Phi falls as the prior grows."""
    low, high = 0.0, 16.0
    if shares(world, {**spec, "prior": low})[0] <= target:
        return {**spec, "prior": low}  # Phi cannot reach the target: reported by its measured Phi
    for _ in range(24):
        middle = (low + high) / 2
        if shares(world, {**spec, "prior": middle})[0] > target:
            low = middle
        else:
            high = middle
    return {**spec, "prior": (low + high) / 2}


def boundary_spec(world: World, phi: float) -> Dict[str, Any]:
    """Phi at phi with E near the margin, by bisection on the signal."""
    low, high = 0.05, 8.0
    spec = base_spec()
    for _ in range(12):
        middle = (low + high) / 2
        spec = prior_for_phi(world, phi, base_spec(signal=middle))
        if shares(world, spec)[1] < R["boundary_e_share"]:
            low = middle
        else:
            high = middle
    return spec


def expected_for(phi: float) -> str:
    return "Evidence-led" if phi > 0.5 else "Default-led with evidence" if phi < 0.5 else "Even"


def observers(world: World) -> List[Dict[str, Any]]:
    """Every observer. `judged` means the pass rule applies; others report power."""
    out: List[Dict[str, Any]] = []

    def add(name: str, spec: Mapping[str, Any], expected: Optional[str], judged: bool, **meta: Any) -> None:
        out.append({"name": name, "spec": dict(spec), "expected": expected, "judged": judged, **meta})

    scale, small, free = R["prior_only_scale"], R["prior_only_noise"], R["label_free_prior_scale"]
    quiet = {"actor_sd": small, "item_sd": small}
    add("prior_only_word", base_spec(prior=scale, **quiet), "No face evidence", True, label_free=True, share="word_prior")
    add("prior_only_face_default", base_spec(prior=scale, w="flat", **quiet), "No face evidence", True, label_free=True, share="image_or_face")
    add("evidence_only", base_spec(signal=R["pass_signal"], w="flat"), "Evidence-led", True)
    add("identity", base_spec(prior=free, identity=R["identity_shift"]), "Face-dependent, no expression", True, label_free=True)
    add("noise", base_spec(prior=free, item_sd=R["noise_item_sd"]), "Face-dependent, no expression", True, label_free=True)
    null = prior_for_phi(world, 0.5, base_spec())
    add("null_balance_no_evidence", null, None, False, label_free=True, true_balance_zero=True)
    add("all_null_family", null, None, False, label_free=True, true_balance_zero=True, share_of_replicates=R["all_null_replicates"] / R["replicates"],
        companion_shuffles=R["all_null_shuffles"])
    for phi in R["phi_grid"]:
        for signal in R["signal_grid"]:
            judged = signal == R["pass_signal"]
            add(f"mixed_phi{phi}_signal{signal}", prior_for_phi(world, phi, base_spec(signal=signal)), expected_for(phi), judged,
                phi=phi, signal=signal, true_balance_zero=phi == 0.5, grid=True)
    for signal in R["unmapped_signals"]:
        judged = signal == R["pass_signal"]
        add(f"unmapped_face_signal{signal}", prior_for_phi(world, 0.8, base_spec(signal=signal, mode="wrong")), "Face-led, unmapped", judged, signal=signal)
        add(f"unmapped_default_signal{signal}", prior_for_phi(world, 0.2, base_spec(signal=signal, mode="wrong")), "Default-led, unmapped", judged, signal=signal)
    add("spread", base_spec(signal=R["pass_signal"], mode="spread", w="flat"), "Evidence-led", True)
    for phi in R["boundary_phi"]:
        add(f"boundary_phi{phi}", boundary_spec(world, phi), None, False, phi=phi)
    for phi in R["false_even_phi"]:
        for signal in R["false_even_signals"]:
            add(f"false_even_phi{phi}_signal{signal}", prior_for_phi(world, phi, base_spec(signal=signal)), None, False, phi=phi)
    for phi in R["inside_even_phi"]:
        add(f"inside_even_phi{phi}", prior_for_phi(world, phi, base_spec(signal=R["pass_signal"])), None, False, phi=phi)
    for gain in R["gain_sweep"]:
        for phi in R["gain_sweep_phi"]:
            spec = prior_for_phi(world, phi, base_spec(signal=R["pass_signal"], gain=gain))
            add(f"gain{gain}_phi{phi}", spec, expected_for(phi), False, phi=phi, gain=gain)
    return out


# ---------------------------------------------------------------------------
# One replicate, through the real pipeline
# ---------------------------------------------------------------------------


def replicate(world: World, observer: Mapping[str, Any], companion: Mapping[str, Any], index: int, number: int,
              shuffles: int = D["label_shuffles"]) -> Dict[str, Any]:
    rng = np.random.default_rng([R["seed"], index, number])
    s, w, g = world.scores(observer["spec"], rng)
    cell = a5.build_cell("synthetic", ROUTE, an.Cells(world.rows(s)), world.actors, world.partners,
                         a5.baselines([world.report(w, g)], ROUTE), dict(GATES_PASSED), 0.0, shuffles)
    others = [a5.estimate(world.judgement(world.scores(companion, rng)[0]), shuffles=min(shuffles, observer.get("companion_shuffles", R["companion_shuffles"])))
              for _ in range(R["companion_cells"])]
    tests = a5.family_tests([cell.est, *others])
    row = a5.finish_cell(cell, tests[0])
    alpha = D["alpha"]
    return {
        "verdict": row["verdict"],
        "reasons": row["reasons"],
        "share": (row["word_share"] or {}).get("reading"),
        "conflict": "survival_conflict" in row["flags"],
        "b3_failed": "b3_failed" in row["flags"],
        "b4": row["gate_b"]["B4"],
        "b5": row["gate_b"]["B5"],
        "e_sig": tests[0]["p_E"] < alpha,
        "e_raw_sig": tests[0]["p_E_raw"] < alpha,
        "e_passes": row["evidence"] in ("Shown", "Unmapped"),
        "b_sig": tests[0]["p_B"] < alpha,
        "b_raw_sig": tests[0]["p_B_raw"] < alpha,
        "any_b_sig": any(t["p_B"] < alpha for t in tests),
        "any_e_sig": any(t["p_E"] < alpha for t in tests),
        "evidence": row["evidence"],
        "balance": row["balance"],
        "phi": row["Phi"],
        "T": row["T"],
        "b_over_t": row["B"]["mean"] / row["T"],
        "e_over_t": row["E_star"]["mean"] / row["T"],
        "b_half_over_t": (row["B"]["ci90"][1] - row["B"]["ci90"][0]) / 2 / row["T"],
        "e_half_over_t": (row["E_star"]["ci90"][1] - row["E_star"]["ci90"][0]) / 2 / row["T"],
    }


def rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([bool(r[key]) for r in rows]))


def summarise(observer: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = Counter(r["verdict"] for r in rows)
    n = len(rows)
    verdicts = {k: v / n for k, v in sorted(counts.items())}
    reasons = Counter(code for r in rows for code in r["reasons"])
    out: Dict[str, Any] = {
        "expected": observer["expected"], "judged": observer["judged"], "n": n,
        "prior_scale": round(observer["spec"]["prior"], 4), "signal": round(observer["spec"]["signal"], 4),
        "verdicts": verdicts, "reason_rates": {k: v / n for k, v in sorted(reasons.items())},
        "measured": {k: round(float(np.mean([r[k] for r in rows])), 4)
                     for k in ("phi", "T", "b_over_t", "e_over_t", "b_half_over_t", "e_half_over_t")},
        "rates": {k: rate(rows, k) for k in ("b4", "b5", "e_sig", "e_raw_sig", "e_passes", "b_sig", "b_raw_sig", "any_b_sig", "any_e_sig", "conflict", "b3_failed")},
        "evidence_classes": {k: v / n for k, v in sorted(Counter(str(r["evidence"]) for r in rows).items())},
        "balance_classes": {k: v / n for k, v in sorted(Counter(str(r["balance"]) for r in rows).items())},
        "word_share": {k: v / n for k, v in sorted(Counter(str(r["share"]) for r in rows).items())},
    }
    if observer["expected"]:
        hit = verdicts.get(observer["expected"], 0.0)
        wrong = sum(v for k, v in verdicts.items() if k not in (observer["expected"], a5.INCONCLUSIVE))
        out["power"] = hit
        out["other_named"] = wrong
        out["inconclusive"] = verdicts.get(a5.INCONCLUSIVE, 0.0)
        out["pass"] = bool(hit >= R["expected_verdict_min"] and wrong <= R["other_named_verdict_max"]
                           and out["inconclusive"] <= R["inconclusive_max"])
    return out


def false_positive_checks(table: Mapping[str, Mapping[str, Any]], spec_of: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Error rates on observers where the truth is known to be null, against a fixed limit."""
    limit = R["false_positive_max"]
    low_even, high_even = (1 - D["balance_margin_fraction"]) / 2, (1 + D["balance_margin_fraction"]) / 2
    out: Dict[str, Any] = {}
    for name, row in table.items():
        meta = spec_of[name]
        if meta.get("label_free"):
            out[f"{name}|E_star_passes"] = {"rate": row["rates"]["e_passes"], "limit": limit}
            out[f"{name}|B4_passes"] = {"rate": row["rates"]["b4"], "limit": limit}
            shown = sum(row["verdicts"].get(v, 0.0) for v in (*EVIDENCE_VERDICTS, "Face-led, unmapped", "Default-led, unmapped"))
            out[f"{name}|any_evidence_verdict"] = {"rate": shown, "limit": limit}
        if meta.get("true_balance_zero"):
            out[f"{name}|B_significant"] = {"rate": row["rates"]["b_sig"], "limit": limit}
        if "phi" in meta and not low_even <= meta["phi"] <= high_even:
            out[f"{name}|false_even"] = {"rate": row["verdicts"].get("Even", 0.0), "limit": limit}
    null = table.get("all_null_family")
    if null:
        out["all_null_family|any_B_column"] = {"rate": null["rates"]["any_b_sig"], "limit": limit}
        out["all_null_family|any_E_star_column"] = {"rate": null["rates"]["any_e_sig"], "limit": limit}
    for value in out.values():
        value["pass"] = bool(value["rate"] <= value["limit"])
    return out


def build_report(table: Mapping[str, Mapping[str, Any]], meta: Mapping[str, Mapping[str, Any]], replicates: int) -> Dict[str, Any]:
    reached = {v: max((row["verdicts"].get(v, 0.0) for row in table.values()), default=0.0) for v in NAMED}
    judged = {k: v["pass"] for k, v in table.items() if v["judged"]}
    false_positive = false_positive_checks(table, meta)
    word = {k: table[k]["word_share"].get(meta[k]["share"], 0.0) for k in table if meta[k].get("share")}
    power: Dict[str, Dict[str, Any]] = {}
    for key, row in table.items():
        if meta[key].get("grid"):
            power.setdefault(str(meta[key]["phi"]), {})[str(meta[key]["signal"])] = row["power"]
    checks = {
        **{f"observer|{k}": v for k, v in judged.items()},
        **{f"reached|{k}": v > 0 for k, v in reached.items()},
        **{f"false_positive|{k}": v["pass"] for k, v in false_positive.items()},
        **{f"word_share|{k}": v >= R["expected_verdict_min"] for k, v in word.items()},
    }
    return {
        "design_sha256": a5_design.block_hash(),
        "replicates": replicates,
        "settings": R,
        "uses_model_output": False,
        "observers": table,
        "best_rate_of_each_named_verdict": reached,
        "power_by_phi_and_signal": power,
        "word_share_expected_reading_rate": word,
        "false_positives": false_positive,
        "checks": checks,
        "failed": sorted(k for k, v in checks.items() if not v),
        "pass": all(checks.values()),
    }


# ---------------------------------------------------------------------------
# Speed: A1's two loops, the same numbers from cached draws
# ---------------------------------------------------------------------------

_DRAWS: Dict[Tuple[str, int, int, int], np.ndarray] = {}


def _cached(kind: str, levels: int, n: int, seed: int) -> np.ndarray:
    key = (kind, levels, n, seed)
    if key not in _DRAWS:
        rng = np.random.default_rng(seed)
        if kind == "flip":
            _DRAWS[key] = rng.choice([-1.0, 1.0], size=(n, levels))
        else:
            _DRAWS[key] = np.stack([rng.integers(0, levels, levels) for _ in range(n)])
    return _DRAWS[key]


def _totals(delta: Any, actors: Any) -> Tuple[np.ndarray, np.ndarray]:
    delta = np.asarray(delta, float)
    _, inverse = np.unique(np.asarray(actors), return_inverse=True)
    return np.bincount(inverse, weights=delta), np.bincount(inverse).astype(np.float64)


def fast_sign_flip_p(delta: Any, actors: Any, *, n_perm: int = 100000, seed: int = 42) -> float:
    """Equal to analyse_rotation.actor_sign_flip_p, which draws the same signs every call."""
    totals, _ = _totals(delta, actors)
    signs = _cached("flip", totals.size, n_perm, seed)
    return (int(((signs @ totals) >= totals.sum()).sum()) + 1) / (n_perm + 1)


def fast_bootstrap_ci(delta: Any, actors: Any, *, n_boot: int = 10000, seed: int = 42) -> List[float]:
    """Equal to analyse_rotation.actor_bootstrap_ci, up to rounding in the last digits."""
    totals, counts = _totals(delta, actors)
    draw = _cached("boot", totals.size, n_boot, seed)
    means = totals[draw].sum(1) / counts[draw].sum(1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def install_fast_statistics() -> None:
    """Only in recovery workers, in memory. tests/test_a5.py checks equality with the originals."""
    an.actor_sign_flip_p = fast_sign_flip_p
    an.actor_bootstrap_ci = fast_bootstrap_ci


_WORLD: Optional[World] = None
_COMPANION: Optional[Dict[str, Any]] = None


def _init(items: Sequence[Mapping[str, Any]], companion: Dict[str, Any]) -> None:
    global _WORLD, _COMPANION
    install_fast_statistics()
    _WORLD, _COMPANION = World(items), companion


def _task(task: Tuple[Mapping[str, Any], int, int]) -> Tuple[str, Dict[str, Any]]:
    observer, index, number = task
    assert _WORLD is not None and _COMPANION is not None
    return observer["name"], replicate(_WORLD, observer, _COMPANION, index, number)


def real_items(manifest: Path, lookup: Path) -> List[Dict[str, Any]]:
    from scripts.run_stimulus_interventions import load_items

    keep = ("trial_id", "actor", "label", "human_modal_label", "weak_label")
    return [{k: i[k] for k in keep} for i in load_items(manifest, lookup, Path("."))]


def run_all(items: Sequence[Mapping[str, Any]], replicates: int, workers: int, only: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    world = World(items)
    chosen = observers(world)
    companion = next(o["spec"] for o in chosen if o["name"] == "null_balance_no_evidence")
    if only:
        chosen = [o for o in chosen if o["name"] in only]
    tasks = [(o, i, k) for i, o in enumerate(chosen) for k in range(int(replicates * o.get("share_of_replicates", 1)))]
    with multiprocessing.Pool(workers, _init, (items, companion)) as pool:
        results = pool.map(_task, tasks, chunksize=max(1, min(25, replicates // 4)))
    meta = {o["name"]: o for o in chosen}
    table = {o["name"]: summarise(o, [r for name, r in results if name == o["name"]]) for o in chosen}
    return build_report(table, meta, replicates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, default=Path("data/eu_emotions_full_manifest.json"))
    parser.add_argument("--lookup", type=Path, default=Path("data/eu_emotions_human_entropy.json"))
    parser.add_argument("--replicates", type=int, default=R["replicates"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = run_all(real_items(args.manifest, args.lookup), args.replicates, args.workers, args.only)
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"pass": out["pass"], "failed": out["failed"]}, indent=1))


if __name__ == "__main__":
    main()
