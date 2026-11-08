"""
Checks behind numbers stated in the A5 draft, on synthetic observers and CREMA-D votes.

1. cause_of_even: why B's interval was too wide for Even in recovery run 1. At Phi .5, the
   spread of actor rows of B / T by signal and portrayal gain, the share of it fixed by an
   actor's label set, and the share explained by the actor's log gain.
2. phi_above_one: whether leaving the actor out of q_def, or partial crossing, puts Phi
   above 1 on an observer with no prior.
3. h3_human_lambda: Lambda from raw CREMA-D vote shares, and what a pseudocount would do.

No model output is read. Writes research/reproduction/a5_checks.json.

Usage:
    python -m scripts.a5_checks --output research/reproduction/a5_checks.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from scripts import a5_analyse as a5
from scripts import a5_design
from scripts import a5_recovery as rec

SEED = 77
CAUSE_DRAWS = 300
CAUSE_CALIBRATION = 60
PHI_DRAWS = 200
PSEUDOCOUNT = 0.5


def cause_of_even(world: rec.World) -> Dict[str, Any]:
    n = len(world.actors)
    rec.R["calibration_draws"] = CAUSE_CALIBRATION
    out: Dict[str, Any] = {}
    for signal in (0.25, 0.5, 1.0, 2.0):
        for gain in (0.0, 0.1, 0.2, 0.3, 0.5):
            spec = rec.prior_for_phi(world, 0.5, rec.base_spec(signal=signal, gain=gain))
            rows, gains = [], []
            for k in range(CAUSE_DRAWS):
                rng = np.random.default_rng([SEED, k])
                s = spec["prior"] * world.prior[None, :] + spec["item_sd"] * world.family_noise(rng, world.n)
                s = s + (spec["actor_sd"] * world.family_noise(rng, n))[world.actor]
                log_gain = rng.normal(0, gain, n)
                s[np.arange(world.n), world.acted] += np.exp(log_gain)[world.actor] * signal
                parts = a5.decompose(world.judgement(s))
                T = np.nanmean(a5.actor_means(parts["T"], world.actor, n))
                rows.append(a5.actor_means(parts["D"] - parts["P"], world.actor, n) / T)
                gains.append(log_gain)
            rows_a, gains_a = np.array(rows), np.array(gains)
            spread = float(rows_a.std(1, ddof=1).mean())
            r2 = 0.0
            if gain > 0:
                r2 = float(np.corrcoef((gains_a - gains_a.mean(0)).ravel(), (rows_a - rows_a.mean(0)).ravel())[0, 1] ** 2)
            out[f"signal{signal}|gain{gain}"] = {
                "actor_sd_of_B_over_T": round(spread, 3),
                "expected_half_width_90": round(1.645 * spread / np.sqrt(n), 3),
                "sd_fixed_by_label_set": round(float(rows_a.mean(0).std(ddof=1)), 3),
                "share_explained_by_log_gain": round(r2, 3),
            }
    return out


def phi_above_one(world: rec.World, items) -> Dict[str, Any]:
    n = len(world.actors)
    spec = rec.base_spec(signal=2.0, w="flat")
    perm = np.random.default_rng(99).permutation(len(items))
    crossed = rec.World([{**i, "label": items[perm[k]]["label"]} for k, i in enumerate(items)])
    crossed.prior, crossed.loadings = world.prior, world.loadings
    out: Dict[str, Any] = {}
    for name, w in (("real_labels", world), ("labels_random_across_clips", crossed)):
        for leave_out in (True, False):
            values = []
            for k in range(PHI_DRAWS):
                s, _, _ = w.scores(spec, np.random.default_rng([11, k]))
                parts = a5.decompose(w.judgement(s), leave_out=leave_out)
                T, P, Dv = (np.nanmean(a5.actor_means(parts[x], w.actor, n)) for x in ("T", "P", "D"))
                values.append((P, Dv / T))
            values_a = np.array(values)
            out[f"{name}|{'leave_actor_out' if leave_out else 'all_actors'}"] = {
                "P": round(float(values_a[:, 0].mean()), 4), "Phi": round(float(values_a[:, 1].mean()), 4),
                "share_of_draws_with_Phi_above_1": float((values_a[:, 1] > 1).mean())}
    return out


def h3_human_lambda(path: Path) -> Dict[str, Any]:
    rows = list(csv.DictReader(path.open()))
    actors = sorted({r["actor_id"] for r in rows})
    options = "ADFHNS"
    votes = np.array([[float(r["votes_" + c]) for c in options] for r in rows])
    actor = np.array([actors.index(r["actor_id"]) for r in rows])
    label = np.array([options.index(r["emotion"]) for r in rows])
    out: Dict[str, Any] = {"n_clips": len(rows), "n_actors": len(actors)}
    raw = votes / votes.sum(1, keepdims=True)
    used = a5.templates(raw, actor, label, len(actors), 6)[label, actor]
    out["templates_zero_where_votes"] = int(((used == 0) & (raw > 0)).sum())
    for name, q in (("raw", raw), (f"pseudocount_{PSEUDOCOUNT}", (votes + PSEUDOCOUNT) / (votes + PSEUDOCOUNT).sum(1, keepdims=True))):
        parts = a5.lambda_share(q, actor, label, len(actors), 6)
        e, p = float(parts["E"].mean()), float(parts["P"].mean())
        out[name] = {"E": round(e, 4), "P": round(p, 4), "Lambda": round(e / (e + p), 4)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, default=a5.ROOT / "data/eu_emotions_full_manifest.json")
    parser.add_argument("--lookup", type=Path, default=a5.ROOT / "data/eu_emotions_human_entropy.json")
    parser.add_argument("--votes", type=Path, default=a5.ROOT / "research/reproduction/crema_d_selection.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    items = rec.real_items(args.manifest, args.lookup)
    world = rec.World(items)
    out: Mapping[str, Any] = {
        "design_sha256": a5_design.block_hash(),
        "uses_model_output": False,
        "phi_above_one": phi_above_one(world, items),
        "h3_human_lambda": h3_human_lambda(args.votes),
        "cause_of_even": cause_of_even(world),
    }
    args.output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
