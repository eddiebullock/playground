"""
Rule 7 for A4: synthetic observers with known cue weights through the real analysis.

Each observer answers the real stimulus design (the frozen 1,200 configurations per dataset)
from weights derived from the human log-odds matrix. No model, render or result is read.
Realism: every option gets stimulus-level logit noise, and each AU gets a lognormal gain
error, standing in for an unknown amplitude mapping. Strengths are multiples of the human
effect. The pass rule in a4_design.RECOVERY was fixed before this ran.

Usage:
    python -m scripts.a4_recovery --manifest results/a4/manifest.csv \
        --out research/reproduction/a4_recovery.json --workers 6
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import a4_analyse as an  # noqa: E402
from scripts import a4_design as d  # noqa: E402

OBSERVERS = ("human_like", "half_shared", "different_cues", "prior_only", "generic", "pair_swapped")
DATASETS = ("basic_EA", "conversational_EA")


def human_weights(store: Mapping[str, Any], dataset: str) -> Tuple[np.ndarray, np.ndarray]:
    """Log-odds weights on 33 AUs (options by AUs), and log base rates."""
    block = store["datasets"][dataset]
    beta = np.array(block["beta_log_odds_per_unit_amplitude"])[:, :33]
    counts = block["structure"]["label_counts"]
    labels = block["labels"]
    base = np.array([counts[k] for k in labels], dtype=float)
    return beta - beta.mean(axis=0), np.log(base / base.sum())


def permute_families(W: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Move each family's weights to another family, keeping each side's own columns."""
    groups = [list(m) for m in an.FAMILY_MEMBERS]
    out = np.zeros_like(W)
    for j, members in enumerate(groups):
        source = groups[order[j]]
        out[:, members] = W[:, source].sum(axis=1, keepdims=True) / len(members)
    return out


def swap_pairs(W: np.ndarray) -> np.ndarray:
    order = np.arange(len(an.FAMILIES))
    for a, b in d.RECOVERY["pairs"]:
        i, j = an.FAMILIES.index(a), an.FAMILIES.index(b)
        order[i], order[j] = j, i
    return permute_families(W, order)


def observer_weights(name: str, W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = permute_families(W, rng.permutation(W.shape[1] and len(an.FAMILIES)))
    if name == "human_like":
        return W
    if name == "half_shared":
        return 0.5 * W + np.sqrt(0.75) * shuffled
    if name == "different_cues":
        return shuffled
    if name == "prior_only":
        return np.zeros_like(W)
    if name == "pair_swapped":
        return swap_pairs(W)
    if name == "generic":
        out = np.zeros_like(W)
        out[-1] = np.abs(W).mean()
        return out - out.mean(axis=0)
    raise ValueError(name)


def respond(W: np.ndarray, base: np.ndarray, X_au: np.ndarray, gender: np.ndarray, rng: np.random.Generator,
            noise: float) -> np.ndarray:
    eta = base + X_au @ W.T + 0.2 * gender[:, None] * np.linspace(-1, 1, len(base))
    eta = eta + rng.normal(0, noise, eta.shape)
    p = np.exp(eta - eta.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def true_cfa(W: np.ndarray, base: np.ndarray, X_au: np.ndarray, gender: np.ndarray, reference: np.ndarray) -> float:
    """CFA of the observer's noiseless do effects, by direct G-computation."""
    tau = np.zeros((len(base), len(an.FAMILY_MEMBERS)))
    for j, members in enumerate(an.FAMILY_MEMBERS):
        on, off = X_au.copy(), X_au.copy()
        on[:, members], off[:, members] = 1.0, 0.0
        zero = np.random.default_rng(0)
        tau[:, j] = (respond(W, base, on, gender, zero, 0.0) - respond(W, base, off, gender, zero, 0.0)).mean(axis=0)
    return an.cosine(tau, reference)


def natural_correlation(W_a: np.ndarray, W_b: np.ndarray, base: np.ndarray, rng: np.random.Generator) -> float:
    """Output r between two observers on natural-like stimuli, with paired cues at pair_rho.

    Pair members share a latent, as Proposition 2 assumes. No pair information goes into CFA.
    """
    n, rho = 2000, d.RECOVERY["pair_rho"]
    fam = np.zeros((n, len(an.FAMILIES)))
    for a, b in d.RECOVERY["pairs"]:
        latent = rng.normal(size=n)
        for name in (a, b):
            fam[:, an.FAMILIES.index(name)] = np.sqrt(rho) * latent + np.sqrt(1 - rho) * rng.normal(size=n)
    X = np.zeros((n, 33))
    for j, members in enumerate(an.FAMILY_MEMBERS):
        X[:, members] = 0.25 * fam[:, [j]]
    zero = np.random.default_rng(0)
    gender = np.zeros(n)
    return an.hae.pearson(respond(W_a, base, X, gender, zero, 0.0), respond(W_b, base, X, gender, zero, 0.0))


def one_replicate(task: Tuple) -> Dict[str, Any]:
    name, dataset, strength, replicate, X_au, gender, W, base, human, lam, draws, cue_n, signal_n = task
    rng = np.random.default_rng([d.SEED, DATASETS.index(dataset), OBSERVERS.index(name), int(strength * 100), replicate])
    target = strength * observer_weights(name, W, rng)
    gain = np.exp(rng.normal(0, d.RECOVERY["gain_log_sd"], (1, X_au.shape[1])))
    P = respond(target, base, X_au * gain, gender, rng, d.RECOVERY["stimulus_logit_sd"])
    result = an.estimate(P, X_au, gender, lam, human, rng=rng, cue_n=cue_n, signal_n=signal_n, draws=draws)
    truth = true_cfa(target * gain, base, X_au, gender, human)
    gates = {"all": True}
    p_two = min(1.0, 2 * result["p_cue"])  # Holm's worst case across the two eligible models
    return {"observer": name, "dataset": dataset, "strength": strength, "cfa": result["cfa"], "truth": truth,
            "p_cue": result["p_cue"], "p_signal": result["p_signal"], "ci": result["ci"],
            "decision": an.decide(gates, result["p_signal"], p_two, result["cfa"], result["ci"]),
            "natural_r": natural_correlation(W, target, base, rng) if name == "pair_swapped" else None}


def rates(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    decisions = [r["decision"] for r in rows]
    errors = [abs(r["cfa"] - r["truth"]) for r in rows]
    naturals = [r["natural_r"] for r in rows if r["natural_r"] is not None]
    return {"n": len(rows), "cfa_mean": round(float(np.mean([r["cfa"] for r in rows])), 3),
            "truth_mean": round(float(np.mean([r["truth"] for r in rows])), 3),
            "aligned": float(np.mean([x == "Aligned" for x in decisions])),
            "not_aligned": float(np.mean([x == "Not aligned" for x in decisions])),
            "inconclusive": float(np.mean([x == "Inconclusive" for x in decisions])),
            "alignment_p_rate": float(np.mean([r["p_cue"] < d.ALPHA / 2 for r in rows])),
            "signal_rate": float(np.mean([r["p_signal"] < d.ALPHA for r in rows])),
            "share_cfa_error_within_limit": float(np.mean([e <= d.RECOVERY["pass"]["pair_cfa_error_max"] for e in errors])),
            "natural_output_r": round(float(np.mean(naturals)), 3) if naturals else None}


def verdict(table: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> Dict[str, bool]:
    rule = d.RECOVERY["pass"]
    out: Dict[str, bool] = {}
    for key, cells in table.items():
        strength = key.split("|")[1]
        if strength != "0.25":
            out[f"{key}|human_like"] = cells["human_like"]["aligned"] >= rule["human_like_aligned_min"]
        out[f"{key}|false_aligned"] = max(cells[o]["aligned"] for o in ("different_cues", "prior_only", "generic")) <= rule["false_aligned_max"]
        out[f"{key}|prior_signal"] = cells["prior_only"]["signal_rate"] <= rule["prior_only_signal_max"]
        out[f"{key}|pair_error"] = cells["pair_swapped"]["share_cfa_error_within_limit"] >= rule["pair_within_error_share_min"]
        if strength == "1.0":
            out[f"{key}|half_alignment"] = cells["half_shared"]["alignment_p_rate"] >= rule["half_shared_alignment_min"]
            out[f"{key}|half_aligned"] = cells["half_shared"]["aligned"] >= rule["half_shared_aligned_min_at_1x"]
            out[f"{key}|not_aligned_power"] = cells["different_cues"]["not_aligned"] >= rule["not_aligned_different_cues_min"]
    return out


def build_tasks(manifest: Sequence[Mapping[str, Any]], store, replicates, draws, cue_n, signal_n) -> List[Tuple]:
    tasks = []
    for dataset in DATASETS:
        rows = [r for r in manifest if r["set"] == "main" and r["dataset"] == dataset]
        _ids, X_au, gender = an.design(rows)
        W, base = human_weights(store, dataset)
        human, lam = an.human_matrix(store, dataset)
        for strength in d.RECOVERY["strengths"]:
            for name in OBSERVERS:
                for k in range(replicates):
                    tasks.append((name, dataset, strength, k, X_au, gender, W, base, human, lam, draws, cue_n, signal_n))
    return tasks


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--human", type=Path, default=Path("research/reproduction/human_au_effects.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=d.RECOVERY["replicates"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    store = json.loads(args.human.read_text(encoding="utf-8"))
    manifest = pd.read_csv(args.manifest).to_dict("records")
    tasks = build_tasks(manifest, store, args.replicates, d.RECOVERY["bootstrap_draws"],
                        d.RECOVERY["cue_permutations"], d.RECOVERY["signal_permutations"])
    with multiprocessing.Pool(args.workers) as pool:
        rows = pool.map(one_replicate, tasks, chunksize=4)
    table: Dict[str, Dict[str, Any]] = {}
    for dataset in DATASETS:
        for strength in d.RECOVERY["strengths"]:
            key = f"{dataset}|{strength}"
            table[key] = {o: rates([r for r in rows if r["dataset"] == dataset and r["strength"] == strength
                                    and r["observer"] == o]) for o in OBSERVERS}
    checks = verdict(table)
    out = {"design_sha256": d.design_hash(), "settings": {k: v for k, v in d.RECOVERY.items() if k != "pairs"},
           "replicates": args.replicates, "table": table, "checks": checks, "pass": all(checks.values())}
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"pass": out["pass"], "failed": [k for k, v in checks.items() if not v]}))


if __name__ == "__main__":
    main()
