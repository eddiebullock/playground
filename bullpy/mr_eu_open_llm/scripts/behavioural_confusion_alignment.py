#!/usr/bin/env python3
"""Motivating RQ (b): behavioural model vs human confusion structure (study3).

Distinct from activation RSA (`confusability_probe_rsa.py`). Here both sides are
*response distributions*:

  - Human: per-item 6AFC distribution from EU validation workbook
  - Model: per-item empirical distribution from forced-choice samples in eval JSON

Comparisons:
  1. Item×item RDM RSA — JS distance between soft distributions, Spearman on
     upper triangle (same geometry as human_confusion_rdm.npy, but model soft).
  2. Per-item JS(model, human) — mean / median distributional distance.
  3. Soft foil mass — Spearman(model p on human top foil, human top_foil_mass).
  4. Label-pair aggregate — correlate human aggregate foil mass with model soft
     mass on the same (target, top-foil) pairs.

Usage:
  python -m scripts.behavioural_confusion_alignment
  python -m scripts.behavioural_confusion_alignment --n_perm 2000
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

from config import LOCAL_DATA_DIR, LOCAL_RESULTS_DIR, SEED
from scripts.build_human_confusion import build_rdm, js_divergence
from scripts.confusability_probe_rsa import perm_rsa_pvalue
from scripts.rsa import rsa_spearman


def _normalize(probs: Sequence[float], *, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    p = np.maximum(p, 0.0)
    s = p.sum()
    if s <= eps:
        return np.full_like(p, 1.0 / max(len(p), 1))
    return p / s


def load_eval_soft(path: Path) -> Dict[str, Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, Any]] = {}
    for trial in obj.get("trials") or []:
        tid = str(trial.get("trial_id") or trial.get("stimulus_relpath") or "")
        s2 = trial.get("stage2") or {}
        dist = s2.get("response_distribution") or {}
        options = list(s2.get("options") or [])
        if not tid or not options or not dist:
            continue
        out[tid] = {
            "options": options,
            "response_distribution": {str(k): float(v) for k, v in dist.items()},
            "forced_choice_entropy": s2.get("forced_choice_entropy"),
            "prediction": s2.get("prediction"),
            "correct": s2.get("correct"),
            "n_samples": s2.get("fc_n_samples"),
        }
    return out


def vec_in_option_order(
    dist: Dict[str, float],
    options: Sequence[str],
) -> np.ndarray:
    return _normalize([float(dist.get(o, 0.0)) for o in options])


def discover_evals(root: Path) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    base = root / "baseline" / "eu_emotions"
    for model in ("qwen3vl", "gemma4", "molmo2"):
        path = base / model / f"eval_v2_eu_emotions_{model}_video_only_seed42.json"
        if path.exists():
            found[model] = path
    return found


def analyse_model(
    model: str,
    eval_path: Path,
    *,
    human_lookup: Dict[str, Any],
    human_meta: Dict[str, Any],
    human_rdm: np.ndarray,
    trial_ids: List[str],
    n_perm: int,
    seed: int,
) -> Dict[str, Any]:
    soft = load_eval_soft(eval_path)
    human_vecs: List[np.ndarray] = []
    model_vecs: List[np.ndarray] = []
    per_item: List[Dict[str, Any]] = []
    missing: List[str] = []

    # Label-pair soft mass (same pairs as human meta aggregation)
    model_pair_mass: Dict[Tuple[str, str], float] = defaultdict(float)
    model_pair_n: Dict[Tuple[str, str], int] = defaultdict(int)

    for tid in trial_ids:
        h = human_lookup.get(tid)
        m = soft.get(tid)
        if h is None or m is None:
            missing.append(tid)
            continue
        options = list(h["human_options"])
        h_vec = vec_in_option_order(h["human_response_distribution"], options)
        # Prefer human option order so RDM geometry matches human_confusion_rdm
        m_vec = vec_in_option_order(m["response_distribution"], options)
        human_vecs.append(h_vec)
        model_vecs.append(m_vec)

        js_mh = js_divergence(h_vec, m_vec)
        target = str(h["human_target_label"])
        # top foil among named options excluding target / abstain
        named = [o for o in options[:-1] if o != target]
        foil = max(named, key=lambda o: float(h["human_response_distribution"].get(o, 0.0))) if named else ""
        human_foil_mass = float(h["human_response_distribution"].get(foil, 0.0)) if foil else 0.0
        model_foil_mass = float(m["response_distribution"].get(foil, 0.0)) if foil else 0.0
        model_p_target = float(m["response_distribution"].get(target, 0.0))

        if foil:
            key = tuple(sorted((target, foil)))
            model_pair_mass[key] += model_foil_mass
            model_pair_n[key] += 1

        per_item.append(
            {
                "trial_id": tid,
                "js_model_human": js_mh,
                "human_entropy": float(h["human_entropy"]),
                "model_forced_choice_entropy": m.get("forced_choice_entropy"),
                "human_top_foil": foil,
                "human_top_foil_mass": human_foil_mass,
                "model_top_foil_mass": model_foil_mass,
                "model_p_target": model_p_target,
                "human_p_target": float(h.get("p_target", h_vec[0])),
                "model_prediction": m.get("prediction"),
                "correct": m.get("correct"),
            }
        )

    if len(model_vecs) < 10:
        raise RuntimeError(f"{model}: only {len(model_vecs)} aligned trials")

    model_rdm = build_rdm(model_vecs)
    # Rebuild human RDM on the same aligned subset (should match file if complete)
    human_rdm_aligned = build_rdm(human_vecs)
    if len(model_vecs) == human_rdm.shape[0] and not missing:
        human_compare = human_rdm
    else:
        human_compare = human_rdm_aligned

    rsa = perm_rsa_pvalue(model_rdm, human_compare, n_perm=n_perm, seed=seed)
    rsa_file = (
        {"rho": float(rsa_spearman(model_rdm, human_rdm)), "note": "vs saved human_confusion_rdm.npy"}
        if len(model_vecs) == human_rdm.shape[0]
        else None
    )

    js_vals = np.asarray([r["js_model_human"] for r in per_item], dtype=float)
    foil_h = np.asarray([r["human_top_foil_mass"] for r in per_item], dtype=float)
    foil_m = np.asarray([r["model_top_foil_mass"] for r in per_item], dtype=float)
    foil_rho, foil_p = spearmanr(foil_h, foil_m)

    # Human label-pair aggregates from meta
    human_pairs = {
        tuple(sorted((row["label_a"], row["label_b"]))): float(row["aggregate_foil_mass"])
        for row in human_meta.get("label_pair_confusion") or []
    }
    shared_keys = sorted(set(human_pairs) & set(model_pair_mass))
    if len(shared_keys) >= 5:
        h_mass = np.asarray([human_pairs[k] for k in shared_keys], float)
        m_mass = np.asarray([model_pair_mass[k] for k in shared_keys], float)
        pair_rho, pair_p = spearmanr(h_mass, m_mass)
        top_pairs = []
        for k in shared_keys:
            top_pairs.append(
                {
                    "label_a": k[0],
                    "label_b": k[1],
                    "human_aggregate_foil_mass": human_pairs[k],
                    "model_aggregate_soft_foil_mass": model_pair_mass[k],
                    "n_items_model": model_pair_n[k],
                }
            )
        top_pairs.sort(key=lambda r: r["human_aggregate_foil_mass"], reverse=True)
    else:
        pair_rho, pair_p, top_pairs = float("nan"), float("nan"), []

    return {
        "model": model,
        "eval": str(eval_path),
        "n_trials_aligned": len(per_item),
        "n_missing": len(missing),
        "behavioural_rdm_rsa": {
            "rho": rsa["rho"],
            "p_perm": rsa["p_perm"],
            "n_perm": int(rsa["n_perm"]),
            "distance": "jensen_shannon_nats",
            "note": (
                "Spearman RSA between item×item JS RDMs of model soft 6AFC "
                "distributions vs human response distributions (Motivating RQ b)."
            ),
        },
        "behavioural_rdm_rsa_vs_saved_human_file": rsa_file,
        "per_item_js_model_vs_human": {
            "mean": float(js_vals.mean()),
            "median": float(np.median(js_vals)),
            "std": float(js_vals.std()),
        },
        "soft_foil_mass_alignment": {
            "spearman_rho": float(foil_rho) if foil_rho == foil_rho else None,
            "p_value": float(foil_p) if foil_p == foil_p else None,
            "n": len(per_item),
            "note": "Does model put more mass on the human top foil when humans do?",
        },
        "label_pair_aggregate_alignment": {
            "spearman_rho": float(pair_rho) if pair_rho == pair_rho else None,
            "p_value": float(pair_p) if pair_p == pair_p else None,
            "n_pairs": len(shared_keys),
            "top_human_pairs": top_pairs[:10],
        },
        "per_item": per_item,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results_root", type=Path, default=LOCAL_RESULTS_DIR)
    ap.add_argument("--human", type=Path, default=LOCAL_DATA_DIR / "eu_emotions_human_entropy.json")
    ap.add_argument("--human_meta", type=Path, default=LOCAL_DATA_DIR / "human_confusion_meta.json")
    ap.add_argument("--human_rdm", type=Path, default=LOCAL_DATA_DIR / "human_confusion_rdm.npy")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    human_obj = json.loads(args.human.read_text(encoding="utf-8"))
    human_lookup = human_obj["trials"]
    human_meta = json.loads(args.human_meta.read_text(encoding="utf-8"))
    trial_ids = list(human_meta["trial_ids"])
    human_rdm = np.load(args.human_rdm)

    evals = discover_evals(args.results_root)
    if not evals:
        raise SystemExit(f"No eval JSONs under {args.results_root / 'baseline' / 'eu_emotions'}")

    models_out: List[Dict[str, Any]] = []
    for model, path in evals.items():
        print(f"Analysing {model} ({path.name})...")
        row = analyse_model(
            model,
            path,
            human_lookup=human_lookup,
            human_meta=human_meta,
            human_rdm=human_rdm,
            trial_ids=trial_ids,
            n_perm=args.n_perm,
            seed=args.seed,
        )
        # drop bulky per_item from console summary path; keep in JSON
        models_out.append(row)
        rsa = row["behavioural_rdm_rsa"]
        print(
            f"  RDM RSA ρ={rsa['rho']:.4f} p_perm={rsa['p_perm']:.4f} | "
            f"mean JS(model,human)={row['per_item_js_model_vs_human']['mean']:.4f} | "
            f"foil-mass ρ={row['soft_foil_mass_alignment']['spearman_rho']}"
        )

    out_path = args.out or (args.results_root / "stats" / "rq_motivating_b_behavioural_confusion.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol_note": (
            "Motivating RQ (b): behavioural correlation of model pairwise confusion "
            "structure with the human EU-Emotion confusion matrix. Not activation RSA."
        ),
        "n_perm": args.n_perm,
        "seed": args.seed,
        "models": [],
        "per_item_by_model": {m["model"]: m["per_item"] for m in models_out},
    }
    for m in models_out:
        slim = {k: v for k, v in m.items() if k != "per_item"}
        slim["n_per_item_rows"] = len(m["per_item"])
        summary["models"].append(slim)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Compact CSV of headline metrics
    csv_path = out_path.with_suffix(".csv")
    lines = [
        "model,rdm_rsa_rho,rdm_rsa_p_perm,mean_js_model_human,foil_mass_rho,foil_mass_p,label_pair_rho,label_pair_p,n_trials"
    ]
    for m in models_out:
        lines.append(
            ",".join(
                [
                    m["model"],
                    f"{m['behavioural_rdm_rsa']['rho']:.6f}",
                    f"{m['behavioural_rdm_rsa']['p_perm']:.6f}",
                    f"{m['per_item_js_model_vs_human']['mean']:.6f}",
                    f"{m['soft_foil_mass_alignment']['spearman_rho']}",
                    f"{m['soft_foil_mass_alignment']['p_value']}",
                    f"{m['label_pair_aggregate_alignment']['spearman_rho']}",
                    f"{m['label_pair_aggregate_alignment']['p_value']}",
                    str(m["n_trials_aligned"]),
                ]
            )
        )
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
