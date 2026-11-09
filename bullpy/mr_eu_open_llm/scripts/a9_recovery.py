"""
Draft amendment A9: recovery, A4's rule 7 on Liu Experiment II's real design.

Everything is A4's machinery, imported from scripts/a4_recovery.py and scripts/a4_analyse.py:
the same eight synthetic observers built from the human log-odds matrix (here Liu's), the same
realism (stimulus logit noise SD 1, lognormal AU gain error log SD 0.5), strengths, estimator,
decision rule and pass rule (a4_design.RECOVERY). Only the replicate wrapper is written here,
because a4_recovery.one_replicate seeds from A4's dataset tuple, and design B is added.

Runs, all fixed in the design block before they ran:
  A          each configuration once, on its manifest face, with its gender, as A4.
  B          each configuration on all four Western faces, two of each gender; responses averaged
             per configuration and fitted with gender 0; truth averaged over the four faces.
             Stimulus noise independent across faces.
  B_rho<r>   B with stimulus noise correlated across the four faces of a configuration at r:
             noise = sqrt(r) * shared + sqrt(1 - r) * own, per configuration and option.

No model, render or model output is read.

Usage:
    python -m scripts.a9_recovery --manifest results/a9/manifest.csv \
        --out research/reproduction/a9_recovery.json --workers 6
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import a4_analyse as an  # noqa: E402
from scripts import a4_recovery as rec  # noqa: E402
from scripts import a9_design as a9  # noqa: E402

OBSERVERS = rec.OBSERVERS
STATISTICS = rec.STATISTICS


def run_parts(run: str) -> Tuple[str, float]:
    """'A' -> ('A', 0); 'B' -> ('B', 0); 'B_rho0.25' -> ('B', 0.25)."""
    if run in ("A", "B"):
        return run, 0.0
    design, rho = run.split("_rho")
    return design, float(rho)


def respond_b(target, base, X, rng, noise, rho):
    """Design B: four faces, noise correlated across faces at rho, responses averaged."""
    shape = (len(X), len(base))
    shared = rng.normal(0, 1, shape)
    out = []
    for g in a9.DESIGN_B_GENDERS:
        eps = noise * (np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.normal(0, 1, shape))
        eta = base + X @ target.T + 0.2 * g * np.linspace(-1, 1, len(base)) + eps
        p = np.exp(eta - eta.max(axis=1, keepdims=True))
        out.append(p / p.sum(axis=1, keepdims=True))
    return np.mean(out, axis=0)


def one_replicate(task: Tuple) -> Dict[str, Any]:
    """a4_recovery.one_replicate, with A9's seed, design B and correlated noise."""
    (name, dataset, strength, replicate, X_au, gender, W, base, human, lam, draws, cue_n, signal_n, splits,
     run) = task
    design, rho = run_parts(run)
    rng = np.random.default_rng([a9.SEED, a9.DATASETS.index(dataset), OBSERVERS.index(name),
                                 int(strength * 100), replicate, a9.DESIGNS.index(design), int(round(rho * 100))])
    target = strength * rec.observer_weights(name, W, rng)
    gain = np.exp(rng.normal(0, a9.RECOVERY["gain_log_sd"], (1, X_au.shape[1])))
    noise = a9.RECOVERY["stimulus_logit_sd"]
    if design == "A":
        P = rec.respond(target, base, X_au * gain, gender, rng, noise)
        fit_gender = gender
        truth_tau = rec.true_tau(target * gain, base, X_au, gender)
    else:
        P = respond_b(target, base, X_au * gain, rng, noise, rho)
        fit_gender = np.zeros(len(X_au))
        truth_tau = np.mean([rec.true_tau(target * gain, base, X_au, np.full(len(X_au), g))
                             for g in sorted(set(a9.DESIGN_B_GENDERS))], axis=0)
    result = an.estimate(P, X_au, fit_gender, lam, human, rng=rng, cue_n=cue_n, signal_n=signal_n, draws=draws,
                         splits=splits)
    truth = {"raw": an.cosine(truth_tau, human),
             "residual": an.cosine(an.remove_common(truth_tau), an.remove_common(human))}
    out: Dict[str, Any] = {"observer": name, "dataset": dataset, "strength": strength, "run": run,
                           "p_signal": result["p_signal"],
                           "natural_r": rec.natural_correlation(W, target, base, rng) if name == "pair_swapped" else None}
    for statistic in STATISTICS:
        block = result[statistic]
        p_two = min(1.0, 2 * block["p_cue"])  # Holm's worst case across the two eligible models
        out[statistic] = {"cfa": block["cfa"], "truth": truth[statistic], "p_cue": block["p_cue"], "ci": block["ci"],
                          "r_s": block["r_s"], "decision": an.decide({"all": True}, result["p_signal"], p_two,
                                                                     block["cfa"], block["ci"], block["r_s"])}
    return out


def build_tasks(manifest: Sequence[Mapping[str, Any]], store: Mapping[str, Any], run: str, replicates: int,
                draws: int, cue_n: int, signal_n: int, splits: int) -> List[Tuple]:
    tasks = []
    for dataset in a9.DATASETS:
        rows = [r for r in manifest if r["set"] == "main" and r["dataset"] == dataset]
        _ids, X_au, gender = an.design(rows)
        W, base = rec.human_weights(store, dataset)
        human, lam = an.human_matrix(store, dataset)
        for strength in a9.RECOVERY["strengths"]:
            for name in OBSERVERS:
                for k in range(replicates):
                    tasks.append((name, dataset, strength, k, X_au, gender, W, base, human, lam, draws, cue_n,
                                  signal_n, splits, run))
    return tasks


def tabulate(rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Dict[str, bool]], Dict[str, bool]]:
    tables: Dict[str, Dict[str, Any]] = {s: {} for s in STATISTICS}
    for statistic in STATISTICS:
        for dataset in a9.DATASETS:
            for strength in a9.RECOVERY["strengths"]:
                key = f"{dataset}|{strength}"
                tables[statistic][key] = {o: rec.rates([r for r in rows if r["dataset"] == dataset
                                                        and r["strength"] == strength and r["observer"] == o],
                                                       statistic) for o in OBSERVERS}
    checks = {s: rec.verdict(tables[s]) for s in STATISTICS}
    passed = {s: all(checks[s].values()) for s in STATISTICS}
    return tables, checks, passed


def choose(passed_by_design: Mapping[str, Mapping[str, bool]]) -> Dict[str, Any]:
    """Design A if its confirmatory statistic passes; else B if B's passes; else none.
    The statistic is never switched."""
    for design in a9.DESIGNS:
        if design in passed_by_design and passed_by_design[design].get(a9.STATISTIC):
            return {"design": design, "statistic": a9.STATISTIC}
    return {"design": None, "statistic": None}


def r_star(passed_by_run: Mapping[str, Mapping[str, bool]], design: str = "B") -> Any:
    """Largest tested face-noise correlation at which the design passes every check on the
    confirmatory statistic, counting up from 0 and stopping at the first failure."""
    best = None
    for rho in (0.0, *a9.CORRELATED_RHOS):
        run = design if rho == 0 else f"{design}_rho{rho}"
        if run not in passed_by_run or not passed_by_run[run].get(a9.STATISTIC):
            break
        best = rho
    return best


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--human", type=Path, default=ROOT / "research/reproduction/a9_human_target.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", choices=a9.RECOVERY_RUNS, default=list(a9.RECOVERY_RUNS))
    parser.add_argument("--replicates", type=int, default=a9.RECOVERY["replicates"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    a9.check_pins()
    a9.check_manifest(args.manifest)
    a9.check_code_pins(require=False)
    started = time.time()
    store = json.loads(args.human.read_text(encoding="utf-8"))
    manifest = pd.read_csv(args.manifest).to_dict("records")
    out: Dict[str, Any] = {"amendment": "A9", "design_sha256": a9.design_hash(), "code_sha256": a9.own_code_sha256(),
                           "replicates": args.replicates, "manifest_sha256": a9.MANIFEST_SHA256,
                           "settings": {k: v for k, v in a9.RECOVERY.items() if k not in ("pairs", "pair_error_withdrawn")},
                           "withdrawn_cells": list(a9.RECOVERY_WITHDRAWN), "runs": {}}
    passed_by_run = {}
    for run in args.runs:
        tasks = build_tasks(manifest, store, run, args.replicates, a9.RECOVERY["bootstrap_draws"],
                            a9.RECOVERY["cue_permutations"], a9.RECOVERY["signal_permutations"],
                            a9.RECOVERY["split_half_splits"])
        with multiprocessing.Pool(args.workers) as pool:
            rows = pool.map(one_replicate, tasks, chunksize=4)
        tables, checks, passed = tabulate(rows)
        passed_by_run[run] = passed
        out["runs"][run] = {"tables": tables, "checks": checks, "pass": passed,
                            "n_checks": {s: len(checks[s]) for s in STATISTICS},
                            "n_passed": {s: int(sum(checks[s].values())) for s in STATISTICS},
                            "failed": {s: [k for k, v in checks[s].items() if not v] for s in STATISTICS}}
        print(json.dumps({"run": run, "pass": passed, "failed": out["runs"][run]["failed"],
                          "elapsed": round(time.time() - started)}), flush=True)
    out["choice"] = choose({k: v for k, v in passed_by_run.items() if k in a9.DESIGNS})
    out["r_star"] = r_star(passed_by_run)
    out["choice_matches_block"] = out["choice"]["design"] == a9.DESIGN_SELECTED
    out["runtime_seconds"] = round(time.time() - started)
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"choice": out["choice"], "r_star": out["r_star"], "matches_block": out["choice_matches_block"]}))


if __name__ == "__main__":
    main()
