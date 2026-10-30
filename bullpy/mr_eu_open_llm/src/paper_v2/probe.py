"""Phase 2 confirmatory probes: nested-CV Ridge, baselines, Branch A/B gate.

Pilot 3-layer last-token caches are labelled exploratory in the preregistration
and do not decide the branch. The same protocol is applied so numbers are ready
and the official gate can be re-run on the all-layer extract without retuning.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed
from paper_v2.behaviour import EVAL_PATHS, summarise_eval
from paper_v2.datasets.eu_emotion import load_eu_emotion_items
from paper_v2.metrics import (
    RIDGE_ALPHAS,
    bootstrap_delta_rho_ci,
    bootstrap_spearman_ci,
    choose_ridge_alpha,
    grouped_ridge_preds_fixed_alpha,
    holm_bonferroni,
    nested_ridge_preds,
    one_hot,
    permutation_p,
    residualise,
    spearman_from_preds,
)

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
_LAYER_RE = re.compile(r"layer(\d+)_eu_emotions_seed\d+\.npy$")

ACT_DIRS = {
    "qwen3vl": ROOT / "results/activations/baseline_qwen3vl_6afc/qwen3vl",
    "gemma4": ROOT / "results/activations/baseline_gemma4_6afc/gemma4",
    "molmo2": ROOT / "results/activations/baseline_molmo2_6afc/molmo2",
}


def layer_index_from_path(path: Path) -> int:
    m = _LAYER_RE.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse layer index from {path.name}")
    return int(m.group(1))


def list_activation_layers(act_dir: Path) -> List[Tuple[int, Path]]:
    files = sorted(act_dir.glob("layer*_eu_emotions_seed*.npy"), key=layer_index_from_path)
    if not files:
        raise FileNotFoundError(f"No activation .npy in {act_dir}")
    return [(layer_index_from_path(p), p) for p in files]


def load_trial_ids(act_dir: Path, layer_path: Path) -> List[str]:
    layer_idx = layer_index_from_path(layer_path)
    sidecar = act_dir / f"layer{layer_idx}_trial_ids.json"
    if sidecar.is_file():
        return json.loads(sidecar.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"missing trial-id sidecar for {layer_path}")


def align_X(act_path: Path, act_dir: Path, trial_ids: Sequence[str]) -> np.ndarray:
    X = np.load(act_path)
    ids = load_trial_ids(act_dir, act_path)
    index = {tid: i for i, tid in enumerate(ids)}
    rows = [X[index[tid]] for tid in trial_ids]
    return np.stack(rows, axis=0).astype(np.float64, copy=False)


def _stage2(trial: Dict[str, Any]) -> Dict[str, Any]:
    return trial.get("stage2") or {}


def load_model_outputs(eval_path: Path, trial_ids: Sequence[str]) -> Dict[str, np.ndarray]:
    obj = json.loads(eval_path.read_text(encoding="utf-8"))
    by_id = {t["trial_id"]: t for t in obj["trials"]}
    entropy = []
    p_target = []
    n_frames = []
    last_frame = []
    for tid in trial_ids:
        t = by_id[tid]
        s2 = _stage2(t)
        entropy.append(float(s2.get("forced_choice_entropy", t.get("forced_choice_entropy"))))
        dist = s2.get("response_distribution") or {}
        label = t.get("label") or s2.get("scoring_label")
        p_target.append(float(dist.get(label, 0.0)))
        n_frames.append(float(t.get("n_frames_used") or 0.0))
        frames = t.get("frame_indices") or []
        last_frame.append(float(max(frames) if frames else 0.0))
    return {
        "model_entropy": np.asarray(entropy, dtype=float),
        "model_p_target": np.asarray(p_target, dtype=float),
        "n_frames": np.asarray(n_frames, dtype=float),
        "last_frame": np.asarray(last_frame, dtype=float),
    }


def stimulus_features(
    trial_ids: Sequence[str],
    actors: Sequence[str],
    model_out: Dict[str, np.ndarray],
    extra_path: Optional[Path] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Hand-built stimulus matrix: actor one-hot + duration proxies (+ optional video stats)."""
    parts = [one_hot(actors)]
    names = [f"actor_{a}" for a in sorted(set(actors))]
    parts.append(np.column_stack([model_out["n_frames"], model_out["last_frame"]]))
    names += ["n_frames_used", "last_frame_index"]
    if extra_path is not None and extra_path.is_file():
        extra = json.loads(extra_path.read_text(encoding="utf-8"))
        rows = extra.get("trials") or extra
        lum, mot, dur = [], [], []
        for tid in trial_ids:
            rec = rows[tid] if isinstance(rows, dict) else None
            lum.append(float((rec or {}).get("mean_luminance", np.nan)))
            mot.append(float((rec or {}).get("motion_energy", np.nan)))
            dur.append(float((rec or {}).get("duration_s", np.nan)))
        block = np.column_stack([lum, mot, dur])
        if np.isfinite(block).any():
            col_ok = np.isfinite(block).any(axis=0)
            block = np.nan_to_num(block, nan=0.0)
            parts.append(block[:, col_ok])
            all_names = ["mean_luminance", "motion_energy", "duration_s"]
            names += [n for n, ok in zip(all_names, col_ok) if ok]
    X = np.column_stack(parts)
    return X, names


def _fit_preds(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "loo": nested_ridge_preds(X, y, groups=None),
        "grouped": nested_ridge_preds(X, y, groups=groups),
    }


def _rho_block(y: np.ndarray, preds: np.ndarray, *, n_boot: int, seed: int) -> Dict[str, Any]:
    rho = spearman_from_preds(y, preds)
    ci = bootstrap_spearman_ci(y, preds, n_boot=n_boot, seed=seed)
    return {"rho": rho, **{k: v for k, v in ci.items() if k != "rho"}}


def _one_perm_rho(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alpha: float,
    seed_i: int,
) -> float:
    rng = np.random.default_rng(seed_i)
    y_p = rng.permutation(y)
    pred = grouped_ridge_preds_fixed_alpha(X, y_p, groups, alpha)
    return spearman_from_preds(y_p, pred)


def _perm_rhos(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    alpha: float,
    n_perm: int,
    seed: int,
    n_jobs: int,
) -> np.ndarray:
    return np.asarray(
        Parallel(n_jobs=n_jobs)(
            delayed(_one_perm_rho)(X, y, groups, alpha, seed + i + 1) for i in range(n_perm)
        ),
        dtype=float,
    )


def probe_layer(
    X: np.ndarray,
    y_t1: np.ndarray,
    y_t2: np.ndarray,
    y_t3: np.ndarray,
    groups: np.ndarray,
    categories: Sequence[str],
    model_out: Dict[str, np.ndarray],
    X_stim: np.ndarray,
    *,
    n_perm: int,
    n_boot: int,
    seed: int,
    n_jobs: int,
    run_perm: bool = True,
) -> Dict[str, Any]:
    cat_z = one_hot(categories)
    y_t1_cat = residualise(y_t1, cat_z)
    y_t1_conf = residualise(
        y_t1, np.column_stack([model_out["model_entropy"], model_out["model_p_target"]])
    )

    t1 = _fit_preds(X, y_t1, groups)
    t2 = _fit_preds(X, y_t2, groups)
    t3 = _fit_preds(X, y_t3, groups)
    t1_cat = _fit_preds(X, y_t1_cat, groups)
    t1_conf = _fit_preds(X, y_t1_conf, groups)
    stim = _fit_preds(X_stim, y_t1, groups)

    block: Dict[str, Any] = {
        "t1_loo": _rho_block(y_t1, t1["loo"], n_boot=n_boot, seed=seed),
        "t1_grouped": _rho_block(y_t1, t1["grouped"], n_boot=n_boot, seed=seed + 1),
        "t2_loo": _rho_block(y_t2, t2["loo"], n_boot=n_boot, seed=seed + 2),
        "t2_grouped": _rho_block(y_t2, t2["grouped"], n_boot=n_boot, seed=seed + 3),
        "t3_loo": _rho_block(y_t3, t3["loo"], n_boot=n_boot, seed=seed + 4),
        "t3_grouped": _rho_block(y_t3, t3["grouped"], n_boot=n_boot, seed=seed + 5),
        "t1_category_residual_grouped": _rho_block(
            y_t1_cat, t1_cat["grouped"], n_boot=n_boot, seed=seed + 6
        ),
        "t1_confidence_residual_grouped": _rho_block(
            y_t1_conf, t1_conf["grouped"], n_boot=n_boot, seed=seed + 7
        ),
        "stimulus_t1_grouped": _rho_block(y_t1, stim["grouped"], n_boot=n_boot, seed=seed + 8),
        "stimulus_t1_loo": _rho_block(y_t1, stim["loo"], n_boot=n_boot, seed=seed + 9),
        "delta_t1_minus_stimulus_grouped": bootstrap_delta_rho_ci(
            y_t1, t1["grouped"], stim["grouped"], n_boot=n_boot, seed=seed + 10
        ),
    }

    if run_perm and n_perm > 0:
        alpha = choose_ridge_alpha(X, y_t1)
        null = _perm_rhos(
            X, y_t1, groups, alpha=alpha, n_perm=n_perm, seed=seed, n_jobs=n_jobs
        )
        obs = block["t1_grouped"]["rho"]
        block["t1_grouped_perm"] = {
            "p_perm": permutation_p(obs, null, alternative="two-sided"),
            "n_perm": int(n_perm),
            "alpha_frozen": float(alpha),
            "null_mean": float(np.nanmean(null)) if np.isfinite(null).any() else float("nan"),
            "null_std": float(np.nanstd(null)) if np.isfinite(null).any() else float("nan"),
        }
    else:
        block["t1_grouped_perm"] = {"p_perm": float("nan"), "n_perm": 0}

    return block


def apply_holm(layers: List[Dict[str, Any]]) -> None:
    pvals = [row.get("t1_grouped_perm", {}).get("p_perm", float("nan")) for row in layers]
    adj = holm_bonferroni(pvals)
    for row, p_h in zip(layers, adj):
        row["t1_grouped_perm"]["p_perm_holm"] = float(p_h)


def layer_passes_gate(row: Dict[str, Any], *, delta_min: float = 0.10) -> Dict[str, Any]:
    perm = row["t1_grouped_perm"]
    delta = row["delta_t1_minus_stimulus_grouped"]
    cat = row["t1_category_residual_grouped"]
    conf = row["t1_confidence_residual_grouped"]
    checks = {
        "perm_holm_lt_0p01": bool(
            np.isfinite(perm.get("p_perm_holm", np.nan)) and perm["p_perm_holm"] < 0.01
        ),
        "delta_rho_ge_0p10": bool(
            np.isfinite(delta["delta_rho"]) and delta["delta_rho"] >= delta_min
        ),
        "delta_ci_excludes_0": bool(delta["excludes_zero"] and delta["delta_rho"] > 0),
        "category_residual_ci_excludes_0": bool(cat["excludes_zero"]),
        "confidence_residual_ci_excludes_0": bool(conf["excludes_zero"]),
    }
    checks["all"] = all(checks.values())
    return checks


def run_model(
    model: str,
    *,
    n_perm: int,
    n_boot: int,
    seed: int,
    n_jobs: int,
    extra_stim: Optional[Path] = None,
    act_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    items = load_eu_emotion_items()
    acc = summarise_eval(EVAL_PATHS[model])
    act_dir = Path(act_dir) if act_dir is not None else ACT_DIRS[model]
    layers = list_activation_layers(act_dir)
    trial_ids = load_trial_ids(act_dir, layers[0][1])
    item_by_id = {it["item_id"]: it for it in items}
    y_t1 = np.array([float(item_by_id[t]["metadata"]["human_entropy"]) for t in trial_ids])
    y_t3 = np.array([float(item_by_id[t]["metadata"]["p_target"]) for t in trial_ids])
    y_t2 = 1.0 - y_t3
    actors = [str(item_by_id[t]["metadata"]["actor_id"]) for t in trial_ids]
    groups = np.array(actors)
    categories = [str(item_by_id[t]["category"]) for t in trial_ids]
    model_out = load_model_outputs(EVAL_PATHS[model], trial_ids)
    X_stim, stim_names = stimulus_features(trial_ids, actors, model_out, extra_stim)

    layer_rows: List[Dict[str, Any]] = []
    for layer_idx, path in layers:
        X = align_X(path, act_dir, trial_ids)
        print(f"  {model} layer {layer_idx} X={X.shape} n_perm={n_perm}", flush=True)
        row = probe_layer(
            X,
            y_t1,
            y_t2,
            y_t3,
            groups,
            categories,
            model_out,
            X_stim,
            n_perm=n_perm,
            n_boot=n_boot,
            seed=seed + int(layer_idx),
            n_jobs=n_jobs,
        )
        row["layer"] = int(layer_idx)
        row["token_position"] = "last_token"
        row["n_trials"] = int(X.shape[0])
        row["n_features"] = int(X.shape[1])
        layer_rows.append(row)

    apply_holm(layer_rows)
    for row in layer_rows:
        row["gate"] = layer_passes_gate(row)

    passing = [r["layer"] for r in layer_rows if r["gate"]["all"]]
    return {
        "model": model,
        "above_chance": bool(acc["p_binom_gt_chance"] < 0.05),
        "accuracy": acc,
        "n_trials": len(trial_ids),
        "n_actors": len(set(actors)),
        "stimulus_feature_names": stim_names,
        "alphas": list(RIDGE_ALPHAS),
        "n_perm": int(n_perm),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "activations_dir": str(act_dir),
        "layers": layer_rows,
        "passing_layers": passing,
        "any_layer_passes": bool(passing),
    }


def decide_branch(model_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Official rule, applied only to above-chance models.

    The caller must still honour the prereg note that a 3-layer last-token
    cache is exploratory and does not lock the branch.
    """
    eligible = []
    for block in model_results:
        if not block.get("above_chance"):
            continue
        if block.get("any_layer_passes"):
            eligible.append({"model": block["model"], "layers": block["passing_layers"]})
    branch = "A" if eligible else "B"
    return {
        "branch": branch,
        "eligible_models": eligible,
        "rule": (
            "Branch A if at least one above-chance model has a layer with "
            "T1 grouped-CV permutation p_holm < 0.01, Δρ vs stimulus >= 0.10 "
            "with CI excluding 0, and category- and confidence-residual CIs "
            "excluding 0."
        ),
    }


def run_gate(
    *,
    models: Sequence[str] = ("qwen3vl", "gemma4", "molmo2"),
    n_perm: int = 1000,
    n_boot: int = 2000,
    seed: int = SEED,
    n_jobs: int = -1,
    extra_stim: Optional[Path] = None,
    out_path: Optional[Path] = None,
    layer_set: str = "pilot_3layer_last_token",
    act_dirs: Optional[Dict[str, Path]] = None,
) -> Dict[str, Any]:
    model_results = [
        run_model(
            m,
            n_perm=n_perm,
            n_boot=n_boot,
            seed=seed,
            n_jobs=n_jobs,
            extra_stim=extra_stim,
            act_dir=(act_dirs or {}).get(m),
        )
        for m in models
    ]
    decision = decide_branch(model_results)
    exploratory = layer_set == "pilot_3layer_last_token"
    payload: Dict[str, Any] = {
        "prereg": "prereg-v1",
        "layer_set": layer_set,
        "token_position": "last_token",
        "exploratory": exploratory,
        "branch_locked": (not exploratory),
        "would_be_branch": decision["branch"],
        "branch": None if exploratory else decision["branch"],
        "decision": decision,
        "note": (
            "Pilot 3-layer last-token probes are exploratory and do not decide "
            "the branch (PREREGISTRATION.md). Re-run on the all-layer extract "
            "with the same code to lock Branch A vs B."
            if exploratory
            else "Official Phase 2 gate on the stated layer set."
        ),
        "models": {block["model"]: block for block in model_results},
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["qwen3vl", "gemma4", "molmo2"])
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--layer_set",
        default="pilot_3layer_last_token",
        help="Label for the activation cache. Official gate requires all_layers.",
    )
    ap.add_argument(
        "--extra_stim",
        type=Path,
        default=ROOT / "research/reproduction/stimulus_features.json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "research/reproduction/phase2_gate.json",
    )
    ap.add_argument(
        "--act_dir",
        type=Path,
        default=None,
        help="Activation directory for a single --models entry. Default: 3-layer 6AFC cache.",
    )
    args = ap.parse_args()
    extra = args.extra_stim if args.extra_stim.is_file() else None
    act_dirs = None
    if args.act_dir is not None:
        if len(args.models) != 1:
            ap.error("--act_dir requires exactly one --models entry")
        act_dirs = {args.models[0]: args.act_dir}
    payload = run_gate(
        models=args.models,
        n_perm=args.n_perm,
        n_boot=args.n_boot,
        seed=args.seed,
        n_jobs=args.n_jobs,
        extra_stim=extra,
        out_path=args.out,
        layer_set=args.layer_set,
        act_dirs=act_dirs,
    )
    print(f"wrote {args.out}")
    print(f"exploratory={payload['exploratory']} would_be_branch={payload['would_be_branch']}")
    for name, block in payload["models"].items():
        flag = "ABOVE" if block["above_chance"] else "AT/BELOW"
        layers = ", ".join(
            f"L{r['layer']} ρg={r['t1_grouped']['rho']:.3f} pHolm={r['t1_grouped_perm'].get('p_perm_holm')}"
            for r in block["layers"]
        )
        print(f"{name} {flag} pass={block['passing_layers']} {layers}")


if __name__ == "__main__":
    main()
