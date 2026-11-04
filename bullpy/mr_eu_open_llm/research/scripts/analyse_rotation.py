#!/usr/bin/env python
"""Prereg amendment A1: analyse the rotated 6AFC, real versus vision-mismatch."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from paper_v2.schema import actor_id_from_eu_filename  # noqa: E402

MODELS = ("qwen3vl", "gemma4", "molmo2")
ROTATIONS = range(5)
EVAL = "results/baseline/eu_emotions/{m}/eval_v2_eu_emotions_{m}_video_only_seed42_rot{r}_{c}.json"


def actor_sign_flip_p(delta, actors, *, n_perm=100000, seed=42):
    """One-sided p for mean(delta) > 0, flipping signs of whole actors."""
    delta, actors = np.asarray(delta, float), np.asarray(actors)
    levels = sorted(set(actors.tolist()))
    totals = np.array([delta[actors == a].sum() for a in levels])
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(levels)))
    hits = int(((signs * totals).sum(1) >= totals.sum()).sum())
    return (hits + 1) / (n_perm + 1)


def actor_bootstrap_ci(delta, actors, *, n_boot=10000, seed=42):
    delta, actors = np.asarray(delta, float), np.asarray(actors)
    levels = sorted(set(actors.tolist()))
    groups = [delta[actors == a] for a in levels]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        draw = rng.integers(0, len(levels), len(levels))
        means[b] = np.concatenate([groups[i] for i in draw]).mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[idx]))
        adjusted[idx] = float(running)
    return adjusted


def load(model, condition):
    runs = []
    for r in ROTATIONS:
        obj = json.loads((ROOT / EVAL.format(m=model, r=r, c=condition)).read_text(encoding="utf-8"))
        runs.append({t["trial_id"]: t["stage2"] for t in (obj.get("trials") or obj["results"])})
    return runs


def summarise(model, human):
    real, mismatch = load(model, "real"), load(model, "mismatch")
    ids = sorted(real[0])
    def correct(runs):
        return np.array([np.mean([bool(run[k]["correct"]) for run in runs]) for k in ids])
    def p_target(runs):
        return np.array(
            [np.mean([run[k]["response_distribution"].get(human[k]["human_target_label"], 0) for run in runs]) for k in ids]
        )
    a, b = correct(real), correct(mismatch)
    actors = [actor_id_from_eu_filename(k) for k in ids]
    delta = a - b
    n_correct = int(round(a.sum() * len(real)))
    by_position = collections.defaultdict(list)
    for run in real:
        for k in ids:
            s = run[k]
            target = human[k]["human_target_label"]
            by_position[s["options"].index(target)].append(bool(s["correct"]))
    labels = np.array([human[k]["human_target_label"] for k in ids])
    pt, ht = p_target(real), np.array([human[k]["p_target"] for k in ids])
    within_pt, within_ht = pt.copy(), ht.copy()
    for label in set(labels.tolist()):
        mask = labels == label
        within_pt[mask] -= pt[mask].mean()
        within_ht[mask] -= ht[mask].mean()
    return {
        "n_items": len(ids),
        "accuracy_real": float(a.mean()),
        "accuracy_mismatch": float(b.mean()),
        "delta_real_minus_mismatch": float(delta.mean()),
        "delta_ci_actor_bootstrap": actor_bootstrap_ci(delta, actors),
        "p_delta_actor_sign_flip": actor_sign_flip_p(delta, actors),
        "secondary_real_vs_chance_pooled_p": float(
            binomtest(n_correct, len(ids) * len(real), 1 / 6, alternative="greater").pvalue
        ),
        "secondary_accuracy_by_target_position": {str(k): float(np.mean(v)) for k, v in sorted(by_position.items())},
        "secondary_rho_ptarget_human": float(spearmanr(pt, ht)[0]),
        "secondary_within_category_r_ptarget_human": float(np.corrcoef(within_pt, within_ht)[0, 1]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=ROOT / "research/reproduction/rotation_summary.json")
    args = ap.parse_args()
    human = json.loads((ROOT / "data/eu_emotions_human_entropy.json").read_text())["trials"]
    out = {m: summarise(m, human) for m in MODELS}
    adjusted = holm([out[m]["p_delta_actor_sign_flip"] for m in MODELS])
    for m, p in zip(MODELS, adjusted):
        out[m]["p_delta_holm"] = p
        out[m]["vision_dependent"] = bool(p < 0.05 and out[m]["delta_ci_actor_bootstrap"][0] > 0)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
