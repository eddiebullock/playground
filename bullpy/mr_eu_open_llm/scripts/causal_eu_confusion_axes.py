#!/usr/bin/env python3
"""Build intervention axes for EU confusability mech-interp (study3 v2).

Offline CPU: from saved activations + human_confusion_meta.json, build unit axes:
  - confusability: mean(high 1-p_target) - mean(low)  [generic; entangled with entropy]
  - entropy: mean(high human_entropy) - mean(low)     [generic difficulty]
  - pair_<a>_<b>: mean(pair-confused trials) - mean(other trials)
  - random: seeded unit vector (control)

Also reports entanglement (pair vs generic axes), difficulty-matched non-confused pairs
for RSA, and reuse geometry. Axes feed activation patching (ablation) and optional
exploratory steering — patching is the primary v2 causal method.

Usage:
  python -m scripts.causal_eu_confusion_axes --model qwen3vl --layer 4
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import LOCAL_DATA_DIR, LOCAL_RESULTS_DIR, SEED


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def mean_diff_axis(X: np.ndarray, high_idx: List[int], low_idx: List[int]) -> np.ndarray:
    if not high_idx or not low_idx:
        raise ValueError("empty high or low index list for axis")
    return _unit(X[high_idx].mean(axis=0) - X[low_idx].mean(axis=0))


def project(X: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return X @ axis


def tercile_indices(values: Sequence[float], *, high_tail: bool) -> Tuple[List[int], List[int]]:
    arr = np.asarray(values, float)
    n = len(arr)
    order = np.argsort(arr)
    third = max(1, n // 3)
    low_idx = order[:third].tolist()
    high_idx = order[-third:].tolist()
    if high_tail:
        return high_idx, low_idx
    return low_idx, high_idx


def pair_key(label_a: str, label_b: str) -> str:
    a, b = sorted([label_a.strip(), label_b.strip()], key=str.casefold)
    slug = re.sub(r"[^a-z0-9]+", "_", f"{a}_{b}".lower()).strip("_")
    return f"pair_{slug}"


def pair_trial_indices(rows: Sequence[Dict[str, Any]], label_a: str, label_b: str) -> List[int]:
    want = {label_a, label_b}
    out: List[int] = []
    for i, row in enumerate(rows):
        labels = {row["human_target_label"], row["top_foil_label"]}
        if labels == want:
            out.append(i)
    return out


def label_pair_tuple(label_a: str, label_b: str) -> Tuple[str, str]:
    return tuple(sorted([label_a.strip(), label_b.strip()], key=str.casefold))


def group_indices_by_label_pair(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], List[int]]:
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        key = label_pair_tuple(row["human_target_label"], row["top_foil_label"])
        groups[key].append(i)
    return dict(groups)


def axis_entanglement(pair_axis: np.ndarray, generic_axis: np.ndarray) -> Dict[str, float]:
    """Subspace overlap between pair and generic (entropy) axes."""
    pa = _unit(np.asarray(pair_axis, float))
    ga = _unit(np.asarray(generic_axis, float))
    cos = float(np.dot(pa, ga))
    cos_abs = float(abs(cos))
    spec = float(np.sqrt(max(0.0, 1.0 - cos_abs**2)))
    return {
        "cos_pair_vs_generic": cos,
        "cos_abs_pair_vs_generic": cos_abs,
        "specificity_ratio_vs_generic": spec,
    }


def select_difficulty_matched_pairs(
    rows: Sequence[Dict[str, Any]],
    label_a: str,
    label_b: str,
    confused_keys: Sequence[Tuple[str, str]],
    *,
    min_matches: int = 3,
    tol_sd: float = 0.5,
) -> Dict[str, Any]:
    """Non-confused label-pairs matched on mean item entropy to a confused pair."""
    groups = group_indices_by_label_pair(rows)
    confused_set = set(confused_keys)
    confused_idx = pair_trial_indices(rows, label_a, label_b)
    if not confused_idx:
        return {
            "label_a": label_a,
            "label_b": label_b,
            "n_confused_items": 0,
            "matched_pairs": [],
            "matching_tolerance_sd": tol_sd,
            "widened": False,
        }
    ent_vals = [float(r["human_entropy"]) for r in rows]
    pool_sd = float(np.std(ent_vals))
    if pool_sd < 1e-12:
        pool_sd = 1.0
    mean_h = float(np.mean([ent_vals[i] for i in confused_idx]))

    def _matches(tolerance: float) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for key, idxs in sorted(groups.items()):
            if key in confused_set:
                continue
            pair_mean = float(np.mean([ent_vals[i] for i in idxs]))
            if abs(pair_mean - mean_h) <= tolerance * pool_sd:
                out.append(
                    {
                        "label_a": key[0],
                        "label_b": key[1],
                        "n_items": len(idxs),
                        "mean_human_entropy": pair_mean,
                    }
                )
        return out

    matched = _matches(tol_sd)
    widened = False
    if len(matched) < min_matches:
        tol_sd = 1.0
        matched = _matches(tol_sd)
        widened = True
    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_confused_items": len(confused_idx),
        "mean_confused_entropy": mean_h,
        "matched_pairs": matched,
        "matching_tolerance_sd": tol_sd,
        "widened": widened,
    }


def trial_indices_for_label_pairs(
    rows: Sequence[Dict[str, Any]],
    pair_keys: Sequence[Tuple[str, str]],
) -> List[int]:
    groups = group_indices_by_label_pair(rows)
    out: List[int] = []
    for key in pair_keys:
        out.extend(groups.get(key, []))
    return sorted(set(out))


def load_activations(
    act_dir: Path,
    layer: int,
    trial_ids: Sequence[str],
) -> np.ndarray:
    path = act_dir / f"layer{layer}_eu_emotions_seed42.npy"
    if not path.exists():
        cands = sorted(act_dir.glob(f"layer{layer}_eu_emotions_seed*.npy"))
        if not cands:
            raise FileNotFoundError(path)
        path = cands[0]
    X = np.load(path)
    sidecar = act_dir / f"layer{layer}_trial_ids.json"
    if sidecar.exists():
        act_ids = json.loads(sidecar.read_text(encoding="utf-8"))
        index = {tid: i for i, tid in enumerate(act_ids)}
        return np.stack([X[index[tid]] for tid in trial_ids], axis=0)
    if X.shape[0] != len(trial_ids):
        raise ValueError(f"activation rows {X.shape[0]} != trials {len(trial_ids)}")
    return X


def analyze_model_layer(
    model: str,
    layer: int,
    X: np.ndarray,
    rows: Sequence[Dict[str, Any]],
    pair_specs: Sequence[Tuple[str, str]],
    *,
    seed: int = SEED,
) -> Dict[str, Any]:
    ent_vals = [float(r["human_entropy"]) for r in rows]
    conf_vals = [float(r["confusability_1_minus_p_target"]) for r in rows]

    high_e, low_e = tercile_indices(ent_vals, high_tail=True)
    high_c, low_c = tercile_indices(conf_vals, high_tail=True)

    axis_conf = mean_diff_axis(X, high_c, low_c)
    axis_ent = mean_diff_axis(X, high_e, low_e)
    align = float(np.dot(axis_conf, axis_ent))

    proj_c = project(X, axis_conf)
    proj_e = project(X, axis_ent)
    reuse_conf_on_entropy = float(proj_c[high_e].mean() - proj_c[low_e].mean())
    reuse_entropy_on_conf = float(proj_e[high_c].mean() - proj_e[low_c].mean())
    own_conf = float(proj_c[high_c].mean() - proj_c[low_c].mean())
    own_ent = float(proj_e[high_e].mean() - proj_e[low_e].mean())

    rng = np.random.default_rng(seed)
    rand_gaps = []
    for _ in range(200):
        ax = _unit(rng.normal(size=X.shape[1]))
        pr = project(X, ax)
        rand_gaps.append(float(pr[high_c].mean() - pr[low_c].mean()))
    rand_gaps = np.asarray(rand_gaps)

    pair_rows: List[Dict[str, Any]] = []
    for la, lb in pair_specs:
        pidx = pair_trial_indices(rows, la, lb)
        rest = [i for i in range(len(rows)) if i not in pidx]
        if len(pidx) < 2 or len(rest) < 2:
            pair_rows.append(
                {
                    "pair": pair_key(la, lb),
                    "label_a": la,
                    "label_b": lb,
                    "n_pair_trials": len(pidx),
                    "status": "insufficient_trials",
                }
            )
            continue
        ax_p = mean_diff_axis(X, pidx, rest)
        pr_p = project(X, ax_p)
        entang = axis_entanglement(ax_p, axis_ent)
        entang_conf = axis_entanglement(ax_p, axis_conf)
        pair_rows.append(
            {
                "pair": pair_key(la, lb),
                "label_a": la,
                "label_b": lb,
                "n_pair_trials": len(pidx),
                "own_effect_pair_vs_rest": float(pr_p[pidx].mean() - pr_p[rest].mean()),
                "reuse_on_entropy_high": float(pr_p[high_e].mean() - pr_p[low_e].mean()),
                "entanglement_vs_entropy": entang,
                "entanglement_vs_confusability": entang_conf,
            }
        )

    confused_keys = [label_pair_tuple(la, lb) for la, lb in pair_specs]
    difficulty_matched = [
        select_difficulty_matched_pairs(rows, la, lb, confused_keys)
        for la, lb in pair_specs
    ]

    return {
        "model": model,
        "layer": layer,
        "n_trials": int(X.shape[0]),
        "axis_alignment_conf_vs_entropy": align,
        "own_effect_confusability": own_conf,
        "own_effect_entropy": own_ent,
        "reuse_conf_axis_on_entropy_tertiles": reuse_conf_on_entropy,
        "reuse_entropy_axis_on_confusability_tertiles": reuse_entropy_on_conf,
        "random_axis_conf_gap_mean": float(rand_gaps.mean()),
        "random_axis_conf_gap_p": float(
            (np.sum(np.abs(rand_gaps) >= abs(own_conf)) + 1) / (len(rand_gaps) + 1)
        ),
        "pair_axes": pair_rows,
        "difficulty_matched_control_pairs": difficulty_matched,
        "steer_patch_status": "axes_ready",
    }


def save_axes(
    outdir: Path,
    model: str,
    layer: int,
    X: np.ndarray,
    rows: Sequence[Dict[str, Any]],
    pair_specs: Sequence[Tuple[str, str]],
    *,
    seed: int,
) -> Dict[str, np.ndarray]:
    ent_vals = [float(r["human_entropy"]) for r in rows]
    conf_vals = [float(r["confusability_1_minus_p_target"]) for r in rows]
    high_e, low_e = tercile_indices(ent_vals, high_tail=True)
    high_c, low_c = tercile_indices(conf_vals, high_tail=True)

    axes: Dict[str, np.ndarray] = {
        "confusability": mean_diff_axis(X, high_c, low_c),
        "entropy": mean_diff_axis(X, high_e, low_e),
    }
    rng = np.random.default_rng(seed)
    axes["random"] = _unit(rng.normal(size=X.shape[1]).astype(np.float32))

    for la, lb in pair_specs:
        pidx = pair_trial_indices(rows, la, lb)
        rest = [i for i in range(len(rows)) if i not in pidx]
        if len(pidx) >= 2 and len(rest) >= 2:
            axes[pair_key(la, lb)] = mean_diff_axis(X, pidx, rest)

    outdir.mkdir(parents=True, exist_ok=True)
    for kind, vec in axes.items():
        np.save(outdir / f"axis_{kind}_{model}_layer{layer}.npy", vec)
    return axes


def default_pair_specs(pairs_json: Path, *, top_k: int = 3) -> List[Tuple[str, str]]:
    if not pairs_json.exists():
        return [("Worried", "Disappointed"), ("Interested", "Kind")]
    obj = json.loads(pairs_json.read_text(encoding="utf-8"))
    specs: List[Tuple[str, str]] = []
    for row in obj.get("label_pair_confusion", [])[:top_k]:
        specs.append((row["label_a"], row["label_b"]))
    return specs


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument(
        "--activations_dir",
        type=Path,
        default=None,
        help="default: results/activations/baseline_{model}_6afc/{model}",
    )
    ap.add_argument("--human_meta", type=Path, default=LOCAL_DATA_DIR / "human_confusion_meta.json")
    ap.add_argument("--pairs_json", type=Path, default=LOCAL_DATA_DIR / "human_confused_pairs.json")
    ap.add_argument("--outdir", type=Path, default=LOCAL_RESULTS_DIR / "mech")
    ap.add_argument("--top_pairs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    meta = json.loads(args.human_meta.read_text(encoding="utf-8"))
    rows = list(meta["per_item"])
    trial_ids = list(meta["trial_ids"])

    act_dir = args.activations_dir or (
        LOCAL_RESULTS_DIR / "activations" / f"baseline_{args.model}_6afc" / args.model
    )
    X = load_activations(act_dir, args.layer, trial_ids)
    pair_specs = default_pair_specs(args.pairs_json, top_k=args.top_pairs)

    result = analyze_model_layer(
        args.model, args.layer, X, rows, pair_specs, seed=args.seed
    )
    save_axes(args.outdir, args.model, args.layer, X, rows, pair_specs, seed=args.seed)

    csv_path = args.outdir / f"{args.model}_eu_causal_axis_geometry.csv"
    pd.DataFrame([result]).to_csv(csv_path, index=False)
    summary_path = args.outdir / f"{args.model}_eu_causal_axes_layer{args.layer}.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"wrote axes under {args.outdir}/axis_*_{args.model}_layer{args.layer}.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
