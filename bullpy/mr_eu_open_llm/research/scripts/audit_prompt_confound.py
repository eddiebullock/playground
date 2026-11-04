#!/usr/bin/env python
"""Audit the EU-Emotion 6AFC prompt for text confounds.

Reads eval JSON, human norms and cached activations; needs no GPU. Writes
research/reproduction/prompt_confound_audit.json. Exploratory, post hoc.

Checks: (1) where the target sits in the prompt; (2) what category alone predicts
of human entropy; (3) whether activations decode category, which only text could
supply at near-perfect accuracy; (4) behaviour and probes within a category, where
the prompt is identical, so any signal has to come through vision.
"""

from __future__ import annotations

import collections
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from paper_v2.schema import actor_id_from_eu_filename  # noqa: E402

MODELS = ("qwen3vl", "gemma4", "molmo2")
ALPHAS = np.logspace(-2, 6, 9)
SEED = 42
N_PERM_BEHAVIOUR = 10000
N_PERM_PROBE = 2000


def demean_within(values, groups):
    out = np.asarray(values, float).copy()
    groups = np.asarray(groups)
    for g in set(groups.tolist()):
        out[groups == g] -= out[groups == g].mean()
    return out


def within_category_perm(a, b, groups, n_perm, rng):
    """Pearson r on category-demeaned values, permuting inside each category."""
    groups = np.asarray(groups)
    obs = float(np.corrcoef(a, b)[0, 1])
    index = {g: np.flatnonzero(groups == g) for g in set(groups.tolist())}
    hits = 0
    for _ in range(n_perm):
        shuffled = a.copy()
        for idx in index.values():
            shuffled[idx] = a[rng.permutation(idx)]
        hits += abs(np.corrcoef(shuffled, b)[0, 1]) >= abs(obs)
    return {"r": obs, "p_perm": (hits + 1) / (n_perm + 1), "n_perm": n_perm}


def one_hot(values):
    levels = sorted(set(values))
    out = np.zeros((len(values), len(levels)))
    for i, v in enumerate(values):
        out[i, levels.index(v)] = 1
    return out


def grouped_rho(X, y, groups):
    pred = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        pred[te] = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X[tr], y[tr]).predict(X[te])
    return float(spearmanr(pred, y)[0])


def eval_trials(model):
    path = ROOT / f"results/baseline/eu_emotions/{model}/eval_v2_eu_emotions_{model}_video_only_seed42.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    return {t["trial_id"]: t["stage2"] for t in (obj.get("trials") or obj["results"])}


def behaviour(model, human, rng):
    stage2 = eval_trials(model)
    ids = sorted(stage2)
    target = [human[k]["human_target_label"] for k in ids]
    positions = collections.Counter()
    for k in ids:
        options, pred = stage2[k]["options"], stage2[k]["prediction"]
        positions[str(options.index(pred)) if pred in options else "unparsed"] += 1
    p_target = demean_within([stage2[k]["response_distribution"].get(t, 0) for k, t in zip(ids, target)], target)
    h_target = demean_within([human[k]["p_target"] for k in ids], target)
    p_abstain = demean_within([stage2[k]["response_distribution"].get("None of the above", 0) for k in ids], target)
    h_abstain = demean_within([human[k]["p_abstain"] for k in ids], target)
    return {
        "prompt_target_position_counts": collections.Counter(
            str(stage2[k]["options"].index(t)) for k, t in zip(ids, target)
        ),
        "modal_choice_position_counts": positions,
        "within_category_p_target": within_category_perm(p_target, h_target, target, N_PERM_BEHAVIOUR, rng),
        "within_category_abstain": within_category_perm(p_abstain, h_abstain, target, N_PERM_BEHAVIOUR, rng),
    }


def activations(model, human, rng):
    rows = []
    for path in sorted(glob.glob(str(ROOT / f"results/activations/baseline_{model}_6afc/{model}/layer*_eu_emotions_seed42.npy"))):
        layer = int(re.search(r"layer(\d+)_", path).group(1))
        ids = json.loads(Path(path.replace("_eu_emotions_seed42.npy", "_trial_ids.json")).read_text())
        X = np.load(path).astype(np.float64)
        X = X.reshape(len(X), -1)
        cat = np.array([human[k]["human_target_label"] for k in ids])
        actors = np.array([actor_id_from_eu_filename(k) for k in ids])
        y = np.array([human[k]["human_entropy"] for k in ids])
        hits, pred = 0, np.zeros(len(y))
        for tr, te in LeaveOneGroupOut().split(X, y, actors):
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
            Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
            hits += int((RidgeClassifierCV(alphas=ALPHAS).fit(Xtr, cat[tr]).predict(Xte) == cat[te]).sum())
            # Category means come from the training fold only.
            Xw, yw, means = Xtr.copy(), y[tr].astype(float), {}
            for c in set(cat[tr].tolist()):
                mask = cat[tr] == c
                means[c] = (Xtr[mask].mean(0), y[tr][mask].mean())
                Xw[mask] -= means[c][0]
                yw[mask] -= means[c][1]
            fit = RidgeCV(alphas=ALPHAS).fit(Xw, yw)
            for row, i in enumerate(te):
                centre = means.get(cat[i], (Xtr.mean(0), 0.0))[0]
                pred[i] = fit.predict((Xte[row] - centre)[None])[0]
        rows.append(
            {
                "layer": layer,
                "category_decode_accuracy_grouped": hits / len(y),
                "n_categories": len(set(cat.tolist())),
                "within_category_probe": within_category_perm(
                    demean_within(pred, cat), demean_within(y, cat), cat, N_PERM_PROBE, rng
                ),
            }
        )
    return rows


def main():
    rng = np.random.default_rng(SEED)
    human = json.loads((ROOT / "data/eu_emotions_human_entropy.json").read_text())["trials"]
    ids = list(human)
    actors = [actor_id_from_eu_filename(k) for k in ids]
    target = [human[k]["human_target_label"] for k in ids]
    options = [tuple(sorted(human[k]["human_options"])) for k in ids]
    y = np.array([human[k]["human_entropy"] for k in ids])
    out = {
        "note": "Exploratory, post hoc. Not preregistered.",
        "n_items": len(ids),
        "n_target_labels": len(set(target)),
        "n_unique_option_sets": len(set(options)),
        "option_sets_per_target_max": max(
            len({o for o, t in zip(options, target) if t == label}) for label in set(target)
        ),
        "lookup_target_position_counts": collections.Counter(
            str(human[k]["human_options"].index(human[k]["human_target_label"])) for k in ids
        ),
        "category_only_grouped_rho_t1": grouped_rho(one_hot(target), y, actors),
        "behaviour": {m: behaviour(m, human, rng) for m in MODELS},
        "activations": {m: activations(m, human, rng) for m in MODELS},
    }
    dest = ROOT / "research/reproduction/prompt_confound_audit.json"
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in out if k not in ("behaviour", "activations")}, indent=1))
    for m in MODELS:
        print(m, out["behaviour"][m]["modal_choice_position_counts"], out["behaviour"][m]["within_category_p_target"], out["behaviour"][m]["within_category_abstain"])
        for row in out["activations"][m]:
            print(" ", row["layer"], round(row["category_decode_accuracy_grouped"], 3), row["within_category_probe"])


if __name__ == "__main__":
    main()
