"""Behaviour summary: accuracy, binomial vs chance, entropy alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from paper_v2.datasets.eu_emotion import load_eu_emotion_items
from paper_v2.metrics import binomial_gt_chance, bootstrap_spearman_ci, spearman_rho

ROOT = Path(__file__).resolve().parents[2]
CHANCE_6AFC = 1.0 / 6.0
EVAL_PATHS = {
    "qwen3vl": ROOT
    / "results/baseline/eu_emotions/qwen3vl/eval_v2_eu_emotions_qwen3vl_video_only_seed42.json",
    "gemma4": ROOT
    / "results/baseline/eu_emotions/gemma4/eval_v2_eu_emotions_gemma4_video_only_seed42.json",
    "molmo2": ROOT
    / "results/baseline/eu_emotions/molmo2/eval_v2_eu_emotions_molmo2_video_only_seed42.json",
}


def summarise_eval(path: Path, *, p0: float = CHANCE_6AFC) -> Dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    n = int(d["n_scored"])
    k = int(d["n_correct"])
    p = binomial_gt_chance(k, n, p0)
    return {
        "model": d["model"],
        "source": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "n_scored": n,
        "n_correct": k,
        "accuracy": k / n,
        "chance": p0,
        "p_binom_gt_chance": p,
        "above_chance_alpha_05": bool(p < 0.05),
        "wilson_ci_95": d.get("accuracy_wilson_ci_95"),
        "seed": d.get("seed"),
        "n_options": d.get("n_options"),
        "protocol_version": d.get("protocol_version"),
    }


def entropy_alignment(eval_path: Path, items: Optional[List[dict]] = None) -> Dict[str, Any]:
    d = json.loads(eval_path.read_text(encoding="utf-8"))
    if items is None:
        items = load_eu_emotion_items()
    human_h = {it["item_id"]: float(it["metadata"]["human_entropy"]) for it in items}
    xs, ys = [], []
    missing = 0
    for trial in d["trials"]:
        tid = trial["trial_id"]
        if tid not in human_h:
            missing += 1
            continue
        fc = trial.get("forced_choice_entropy")
        if fc is None:
            stage2 = trial.get("stage2") or {}
            fc = stage2.get("forced_choice_entropy")
        if fc is None:
            missing += 1
            continue
        xs.append(float(fc))
        ys.append(human_h[tid])
    rho, p = spearman_rho(xs, ys)
    ci = bootstrap_spearman_ci(ys, xs, n_boot=2000, seed=42)
    return {
        "model": d["model"],
        "n": len(xs),
        "n_missing": missing,
        "spearman_rho": rho,
        "p": p,
        "bootstrap_ci_95": {
            "ci_lo": ci["ci_lo"],
            "ci_hi": ci["ci_hi"],
            "excludes_zero": ci["excludes_zero"],
            "n_boot": ci["n_boot"],
        },
    }


def report_all(out_path: Optional[Path] = None) -> Dict[str, Any]:
    items = load_eu_emotion_items()
    actors = sorted({(it["metadata"] or {}).get("actor_id") for it in items} - {None})
    payload: Dict[str, Any] = {
        "n_items": len(items),
        "n_actors": len(actors),
        "human_modality": (items[0].get("metadata") or {}).get("modality") if items else None,
        "models": {},
    }
    for key, path in EVAL_PATHS.items():
        payload["models"][key] = {
            "accuracy": summarise_eval(path),
            "entropy_alignment": entropy_alignment(path, items),
        }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    out = ROOT / "research" / "reproduction" / "behaviour_summary.json"
    payload = report_all(out)
    print(f"wrote {out.relative_to(ROOT)}")
    for model, block in payload["models"].items():
        acc = block["accuracy"]
        ent = block["entropy_alignment"]
        flag = "ABOVE" if acc["above_chance_alpha_05"] else "AT/BELOW"
        print(
            f"{model}: {acc['n_correct']}/{acc['n_scored']} "
            f"acc={acc['accuracy']:.4f} p={acc['p_binom_gt_chance']:.4g} {flag} "
            f"entropy_rho={ent['spearman_rho']:.4f} p={ent['p']:.3g}"
        )


if __name__ == "__main__":
    main()
