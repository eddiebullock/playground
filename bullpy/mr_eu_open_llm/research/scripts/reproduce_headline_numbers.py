#!/usr/bin/env python3
"""Reproduce study3 headline numbers from cached JSON/npy (no model forward).

Writes research/reproduction/phase0_numbers.json. Every printed figure is also
stored in that file so the paper registry can point at a commit + output path.

Usage (conda env mr_eu_open_llm):

  python research/scripts/reproduce_headline_numbers.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "research" / "reproduction"
OUT_PATH = OUT_DIR / "phase0_numbers.json"
CHANCE = 1.0 / 6.0
N_BRIEF = 243
ACTOR_RE = re.compile(r"([A-Za-z])F\s*(\d+)", re.IGNORECASE)
MODELS = ("qwen3vl", "gemma4", "molmo2")
BRIEF_ACC = {"qwen3vl": 0.379, "gemma4": 0.313, "molmo2": 0.198}
BRIEF_ENTROPY_RHO = {"qwen3vl": -0.001, "gemma4": 0.117, "molmo2": -0.034}


def load_json(rel: str):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def accuracy_block() -> dict:
    out = {}
    for model in MODELS:
        rel = (
            f"results/baseline/eu_emotions/{model}/"
            f"eval_v2_eu_emotions_{model}_video_only_seed42.json"
        )
        d = load_json(rel)
        n, k = int(d["n_scored"]), int(d["n_correct"])
        acc = k / n
        bt = binomtest(k, n, CHANCE, alternative="greater")
        out[model] = {
            "source": rel,
            "n_scored": n,
            "n_correct": k,
            "accuracy": acc,
            "accuracy_pct_1dp": round(100 * acc, 1),
            "stored_p_binom_gt_chance": d["p_binom_gt_chance"],
            "recomputed_p_binom_gt_chance": float(bt.pvalue),
            "p_match_stored": abs(float(d["p_binom_gt_chance"]) - float(bt.pvalue)) < 1e-15
            or np.isclose(d["p_binom_gt_chance"], bt.pvalue),
            "wilson_ci_95": d["accuracy_wilson_ci_95"],
            "above_chance_alpha_05": bool(bt.pvalue < 0.05),
            "matches_brief_1dp": abs(acc - BRIEF_ACC[model]) < 5e-4,
            "seed": d["seed"],
            "n_options": d["n_options"],
            "chance_level": d["chance_level"],
            "protocol_version": d["protocol_version"],
            "condition": d["condition"],
            "stage2_prompt_mode_field": d.get("stage2_prompt_mode"),
            "mean_forced_choice_entropy": d.get("mean_forced_choice_entropy"),
            "created_at_utc": d.get("created_at_utc"),
            "slurm_job_id": (d.get("run_metadata") or {}).get("slurm_job_id"),
        }
    return out


def entropy_block() -> dict:
    raw = load_json("results/stats/rq1_1b_entropy_alignment.json")
    out = {}
    for block in raw:
        model = block["model"]
        stored = block["rq1_1b_forced_choice_vs_human"]["spearman_rho"]
        xs = [t["forced_choice_entropy"] for t in block["trials"]]
        ys = [t["human_entropy"] for t in block["trials"]]
        rho, p = spearmanr(xs, ys)
        out[model] = {
            "source": "results/stats/rq1_1b_entropy_alignment.json",
            "n": len(block["trials"]),
            "stored_spearman_rho": stored,
            "recomputed_spearman_rho": float(rho),
            "recomputed_p": float(p),
            "matches_brief_3dp": stored is not None
            and abs(float(stored) - BRIEF_ENTROPY_RHO[model]) < 1e-3,
            "significant_alpha_05": bool(p < 0.05),
        }
    return out


def calibration_block() -> dict:
    raw = load_json("results/stats/human_calibration.json")
    out = {}
    for block in raw:
        t = block["terciles"]
        p = float(block["high_vs_low_consensus_accuracy_p"])
        out[block["model"]] = {
            "source": "results/stats/human_calibration.json",
            "high_consensus_acc": t["high_consensus"]["accuracy"],
            "mid_acc": t["mid"]["accuracy"],
            "low_consensus_acc": t["high_disagreement"]["accuracy"],
            "n_per_tercile": t["high_consensus"]["n"],
            "high_vs_low_p": p,
            "p_lt_1e-4": bool(p < 1e-4),
            "p_lt_0.05": bool(p < 0.05),
        }
    return out


def probe_rsa_block() -> dict:
    out = {}
    for model in MODELS:
        rel = f"results/mech/{model}_confusability_probe_rsa.json"
        d = load_json(rel)
        rows = []
        for row in d["layers"]:
            rows.append(
                {
                    "layer": row["layer"],
                    "n_trials": row["n_trials"],
                    "probe_confusability_rho": row["probe_confusability_rho"],
                    "probe_human_entropy_rho": row["probe_human_entropy_rho"],
                    "rsa_human_confusion_rho": row["rsa_human_confusion_rho"],
                    "rsa_human_confusion_p_perm": row["rsa_human_confusion_p_perm"],
                    "has_difficulty_matched": "rsa_difficulty_matched" in row,
                }
            )
        out[model] = {
            "source": rel,
            "n_trials": d["n_trials"],
            "layers": rows,
            "probe_conf_range": [
                min(r["probe_confusability_rho"] for r in rows),
                max(r["probe_confusability_rho"] for r in rows),
            ],
            "probe_entropy_range": [
                min(r["probe_human_entropy_rho"] for r in rows),
                max(r["probe_human_entropy_rho"] for r in rows),
            ],
            "rsa_range": [
                min(r["rsa_human_confusion_rho"] for r in rows),
                max(r["rsa_human_confusion_rho"] for r in rows),
            ],
        }
    return out


def axis_block() -> dict:
    names = [
        "confusability",
        "entropy",
        "random",
        "pair_bored_unfriendly",
        "pair_interested_kind",
        "pair_disappointed_worried",
    ]
    axes = {}
    for name in names:
        rel = f"results/mech/axis_{name}_qwen3vl_layer4.npy"
        vec = np.load(ROOT / rel)
        axes[name] = vec
    ent = axes["entropy"]
    conf = axes["confusability"]
    cosines = {}
    for name, vec in axes.items():
        cosines[name] = float(
            np.dot(vec, ent) / (np.linalg.norm(vec) * np.linalg.norm(ent))
        )
    axj = load_json("results/mech/qwen3vl_eu_causal_axes_layer4.json")
    return {
        "source_json": "results/mech/qwen3vl_eu_causal_axes_layer4.json",
        "source_npy_glob": "results/mech/axis_*_qwen3vl_layer4.npy",
        "hidden_dim": int(ent.shape[0]),
        "json_axis_alignment_conf_vs_entropy": axj["axis_alignment_conf_vs_entropy"],
        "npy_cos_conf_vs_entropy": cosines["confusability"],
        "npy_cos_pair_vs_entropy": {
            k: cosines[k]
            for k in (
                "pair_bored_unfriendly",
                "pair_interested_kind",
                "pair_disappointed_worried",
                "random",
            )
        },
        "npy_abs_cos_pair_vs_entropy": {
            k: abs(cosines[k])
            for k in (
                "pair_bored_unfriendly",
                "pair_interested_kind",
                "pair_disappointed_worried",
            )
        },
        "pair_own_vs_rest": [
            {
                "pair": p["pair"],
                "n_pair_trials": p["n_pair_trials"],
                "own_effect_pair_vs_rest": p["own_effect_pair_vs_rest"],
            }
            for p in axj["pair_axes"]
        ],
        "json_has_entanglement_vs_entropy_field": any(
            "entanglement_vs_entropy" in p for p in axj["pair_axes"]
        ),
        "json_has_difficulty_matched_control_pairs": "difficulty_matched_control_pairs"
        in axj,
    }


def steer_block() -> dict:
    rel_csv = "results/mech/steer_summary_qwen3vl_layer4.csv"
    rel_json = "results/mech/steer_summary_qwen3vl_layer4.json"
    df = pd.read_csv(ROOT / rel_csv)
    meta = load_json(rel_json)
    return {
        "source_csv": rel_csv,
        "source_json": rel_json,
        "n_trials": meta["n_trials"],
        "n_samples": meta["n_samples"],
        "layer": meta["layer"],
        "hook_module": meta["hook_module"],
        "mean_delta_js_by_axis_alpha": df[
            ["axis", "alpha", "mean_delta_js_human", "n_trials"]
        ].to_dict(orient="records"),
    }


def human_meta_block() -> dict:
    he = load_json("data/eu_emotions_human_entropy.json")
    actors = []
    for tid in he["trials"]:
        match = ACTOR_RE.search(tid)
        actors.append(match.group(1).upper() if match else "?")
    counts = Counter(actors)
    return {
        "source": "data/eu_emotions_human_entropy.json",
        "modality": he["modality"],
        "n_matched": he["n_matched"],
        "n_human_items_available": he["n_human_items_available"],
        "source_workbook": he["source_workbook"],
        "n_actors_from_filename": len(counts),
        "actor_counts": dict(sorted(counts.items())),
        "grouped_cv_by_actor_possible": "?" not in counts and len(counts) >= 2,
    }


def activation_cache_block() -> dict:
    mapping = {
        "qwen3vl": ("baseline_qwen3vl_6afc/qwen3vl", [4, 13, 26]),
        "gemma4": ("baseline_gemma4_6afc/gemma4", [5, 15, 31]),
        "molmo2": ("baseline_molmo2_6afc/molmo2", [4, 12, 23]),
    }
    out = {}
    for model, (folder, _layers) in mapping.items():
        d = ROOT / "results" / "activations" / folder
        meta = json.loads((d / "extract_meta.json").read_text(encoding="utf-8"))
        npys = list(d.glob("*.npy"))
        out[model] = {
            "extract_meta": str(d / "extract_meta.json"),
            "n_trials": meta["n_trials"],
            "pooling": meta["pooling"],
            "saved_layers": meta["saved_layers"],
            "max_frames": meta["max_frames"],
            "n_npy_files_local": len(npys),
            "activations_present_locally": len(npys) > 0,
        }
    return out


def missing_artifacts() -> dict:
    ablate = list((ROOT / "results").rglob("*ablat*"))
    causal_ablate = [
        p
        for p in ablate
        if "eu_ablation" in p.name or "ablate_summary" in p.name or "ablate_trials" in p.name
    ]
    return {
        "difficulty_matched_rsa_in_probe_json": False,
        "causal_ablation_result_files": [str(p.relative_to(ROOT)) for p in causal_ablate],
        "frame_policy_ablation_dir_exists": (ROOT / "results" / "ablation").is_dir(),
        "note": (
            "results/ablation/ is the frame-policy / modality ablation, not the "
            "axis mean-projection causal test. No qwen3vl_eu_ablation_layer4.json "
            "is present locally."
        ),
    }


def main() -> None:
    payload = {
        "n_brief": N_BRIEF,
        "chance": CHANCE,
        "accuracy": accuracy_block(),
        "entropy_alignment": entropy_block(),
        "human_calibration": calibration_block(),
        "probe_rsa": probe_rsa_block(),
        "axes": axis_block(),
        "steer": steer_block(),
        "human_meta": human_meta_block(),
        "activation_cache": activation_cache_block(),
        "missing": missing_artifacts(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")
    acc = payload["accuracy"]
    print("accuracy:", {m: acc[m]["accuracy_pct_1dp"] for m in MODELS})
    print(
        "molmo_above_chance:",
        acc["molmo2"]["above_chance_alpha_05"],
        "p=",
        acc["molmo2"]["recomputed_p_binom_gt_chance"],
    )
    print("entropy_rho:", {m: payload["entropy_alignment"][m]["recomputed_spearman_rho"] for m in MODELS})
    print("cos(conf, entropy):", payload["axes"]["npy_cos_conf_vs_entropy"])
    print("pair vs entropy:", payload["axes"]["npy_cos_pair_vs_entropy"])
    print("6afc npy present:", {m: payload["activation_cache"][m]["activations_present_locally"] for m in MODELS})


if __name__ == "__main__":
    main()
