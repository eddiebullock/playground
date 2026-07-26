"""
Aggregate Study 3 cross-model comparison (behavior, probes, RSA, patching, tertiles, SAE).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import LOCAL_RESULTS_DIR, STUDY_MODELS


def _modality_for(model: str) -> str:
    return "multimodal" if model == "gemma4" else "video_only"


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _acc_from_eval(path: Path) -> Optional[float]:
    obj = _read_json(path)
    if not obj:
        return None
    return float(obj.get("accuracy", float("nan")))


def _probe_peak(probe_dir: Path, model: str) -> Optional[Dict[str, Any]]:
    summary = _read_json(probe_dir / model / "probes_summary.json")
    if not summary:
        return None
    tert = summary.get("entropy_tertiles") or {}
    return {
        "peak_layer": summary.get("peak_layer"),
        "peak_accuracy": summary.get("peak_accuracy"),
        "low_ambiguity_accuracy": tert.get("peak_layer_low_ambiguity_accuracy"),
        "high_ambiguity_accuracy": tert.get("peak_layer_high_ambiguity_accuracy"),
        "n_low": tert.get("n_low_ambiguity"),
        "n_high": tert.get("n_high_ambiguity"),
    }


def _patching_summary(path: Path) -> Optional[Dict[str, Any]]:
    obj = _read_json(path)
    if not obj:
        return None
    nonempty = sum(
        1
        for t in obj.get("trials", [])
        if (t.get("raw_before") or "").strip() or (t.get("raw_after") or "").strip()
    )
    n = int(obj.get("n_trials_requested") or 0)
    return {
        "n_trials": n,
        "fix_rate": obj.get("fix_rate"),
        "change_rate": obj.get("prediction_change_rate"),
        "acc_before": obj.get("accuracy_before"),
        "acc_after": obj.get("accuracy_after"),
        "peak_layer": obj.get("peak_layer"),
        "n_trials_with_nonempty_output": nonempty,
        "patching_outputs_valid": nonempty > 0 and n > 5,
    }


def summarize_model(model: str, root: Path) -> Dict[str, Any]:
    modality = _modality_for(model)
    missing: List[str] = []

    base_eval = root / f"baseline/eu_emotions/{model}/eval_v2_eu_emotions_{model}_{modality}_fps1_cap16_two_stage_seed42.json"
    ft_eval = root / f"finetune/eu_post_ft/eval_v2_eu_emotions_{model}_{modality}_finetuned_seed42.json"
    base_probe_dir = root / f"probes/baseline_{model}_4afc"
    ft_probe_dir = root / f"probes/finetuned_{model}_4afc"
    rsa_path = root / f"rsa/baseline_vs_finetuned_{model}_4afc.json"
    patch_path = root / f"patching/patching_results_{model}_v2_4afc.json"
    path_patch = root / f"patching/path_patching_{model}_4afc.json"
    sae_path = root / f"sae/{model}_peak_nmf.json"

    for p in (base_probe_dir / model, ft_probe_dir / model, rsa_path, patch_path):
        if not p.exists():
            missing.append(str(p))

    b_acc = _acc_from_eval(base_eval)
    f_acc = _acc_from_eval(ft_eval)
    delta = None
    if b_acc is not None and f_acc is not None:
        delta = round((f_acc - b_acc) * 100, 1)

    if delta is not None and delta <= -15:
        b0_pattern = "collapse"
    elif delta is not None and abs(delta) < 1.0:
        b0_pattern = "null_transfer"
    else:
        b0_pattern = "gain_or_preserved"

    base_probe = _probe_peak(base_probe_dir, model)
    ft_probe = _probe_peak(ft_probe_dir, model)
    probes = None
    if base_probe and ft_probe:
        probes = {
            "peak_layer": base_probe["peak_layer"],
            "baseline_peak_acc": base_probe["peak_accuracy"],
            "finetuned_peak_acc": ft_probe["peak_accuracy"],
            "baseline_low_ambiguity": base_probe.get("low_ambiguity_accuracy"),
            "baseline_high_ambiguity": base_probe.get("high_ambiguity_accuracy"),
            "finetuned_low_ambiguity": ft_probe.get("low_ambiguity_accuracy"),
            "finetuned_high_ambiguity": ft_probe.get("high_ambiguity_accuracy"),
        }

    rsa_obj = _read_json(rsa_path)
    rsa = None
    if rsa_obj:
        rsa = {"mean_rho": rsa_obj.get("mean_rho"), "layers": rsa_obj.get("layers")}

    return {
        "model": model,
        "modality": modality,
        "baseline_4afc_acc": b_acc,
        "finetuned_4afc_acc": f_acc,
        "delta_pp": delta,
        "b0_pattern": b0_pattern,
        "probes": probes,
        "rsa": rsa,
        "patching": _patching_summary(patch_path),
        "path_patching": _read_json(path_patch),
        "sae": _read_json(sae_path),
        "missing": missing,
    }


def _fmt_pct(x: Optional[float]) -> str:
    if x is None or x != x:
        return "—"
    return f"{100 * x:.1f}%"


def build_markdown(models: List[Dict[str, Any]]) -> str:
    lines = [
        "# Study 3 comparison (auto-generated)",
        "",
        "Train: **MindReading** LoRA. Eval: **EU-Emotions** 118 trials, 4AFC stage-2.",
        "",
        "| Model | Modality | B0 (base→FT) | Δ pp | Peak probe | Low/H high tertile (base) | RSA ρ | Patch fix |",
        "|-------|----------|--------------|------|------------|---------------------------|-------|-----------|",
    ]
    for m in models:
        b0 = f"{_fmt_pct(m.get('baseline_4afc_acc'))}→{_fmt_pct(m.get('finetuned_4afc_acc'))}"
        probes = m.get("probes") or {}
        if probes:
            peak = (
                f"L{probes['peak_layer']} "
                f"{_fmt_pct(probes.get('baseline_peak_acc'))}→{_fmt_pct(probes.get('finetuned_peak_acc'))}"
            )
            tert = (
                f"{_fmt_pct(probes.get('baseline_low_ambiguity'))}/"
                f"{_fmt_pct(probes.get('baseline_high_ambiguity'))}"
            )
        else:
            peak = "—"
            tert = "—"
        rsa = m.get("rsa") or {}
        rho = rsa.get("mean_rho")
        rho_s = f"{rho:.2f}" if isinstance(rho, (int, float)) else "—"
        patch = m.get("patching") or {}
        fix = patch.get("fix_rate")
        if fix is None:
            fix_s = "—"
        elif not patch.get("patching_outputs_valid", True):
            fix_s = f"{100 * fix:.1f}% (invalid outputs)"
        else:
            fix_s = f"{100 * fix:.1f}%"
        lines.append(
            f"| {m['model']} | {m['modality']} | {b0} | {m.get('delta_pp', '—')} | {peak} | {tert} | {rho_s} | {fix_s} |"
        )
    lines.extend(["", "Regenerate: `python -m scripts.study3_summarize`", "Fast pull: `./sync.sh pull-study3`"])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize Study 3 cross-model results.")
    ap.add_argument("--results-root", type=Path, default=LOCAL_RESULTS_DIR)
    ap.add_argument("--output-json", type=Path, default=LOCAL_RESULTS_DIR / "stats" / "study3_comparison.json")
    ap.add_argument("--output-md", type=Path, default=LOCAL_RESULTS_DIR / "stats" / "study3_comparison.md")
    args = ap.parse_args()

    models = [summarize_model(m, args.results_root) for m in STUDY_MODELS]
    payload = {"models": models, "table_markdown": build_markdown(models)}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(build_markdown(models), encoding="utf-8")
    print(f"Wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
