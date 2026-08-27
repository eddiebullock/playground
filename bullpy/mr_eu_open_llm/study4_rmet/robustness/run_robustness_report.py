#!/usr/bin/env python3
"""
Orchestrate the study4_rmet robustness layer and write robustness_report.md + CSV.

Does not modify scripts/alignment_analyses.py. Passes through existing Spearman
permutation p-values from results/alignment/a1_summary.json.

Preferred interpreter (has pingouin/statsmodels):
  /Users/eb2007/playground/bullpy/mr_ts_play/venv/bin/python \\
    study4_rmet/robustness/run_robustness_report.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_PREFERRED_PY = Path("/Users/eb2007/playground/bullpy/mr_ts_play/venv/bin/python")


def _ensure_deps_or_reexec() -> None:
    """If pingouin missing, re-exec under mr_ts_play venv when available."""
    try:
        import pingouin  # noqa: F401
        return
    except ImportError:
        pass
    # Note: venv/bin/python may symlink to conda python; compare prefixes, not resolve().
    preferred_prefix = str(_PREFERRED_PY.parent.parent)
    already_in_preferred = Path(sys.prefix).resolve() == Path(preferred_prefix).resolve()
    if _PREFERRED_PY.is_file() and not already_in_preferred:
        os.execv(str(_PREFERRED_PY), [str(_PREFERRED_PY), str(Path(__file__).resolve()), *sys.argv[1:]])
    raise SystemExit(
        "Missing dependency `pingouin` in this Python.\n"
        f"  current: {sys.executable} (prefix={sys.prefix})\n"
        "Install it, or re-run with:\n"
        f"  {_PREFERRED_PY} study4_rmet/robustness/run_robustness_report.py"
    )


_ensure_deps_or_reexec()

import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from data_io import (  # noqa: E402
    DEFAULT_A1,
    load_a1_perm_summary,
    load_all_model_item_tables,
    load_human_item_sensitivity,
    paired_item_vectors,
)
from disattenuation import human_sensitivity_reliability, run_disattenuation_for_model  # noqa: E402
from equivalence_bayes import bootstrap_corr_ci, equivalence_battery  # noqa: E402
from meta_analysis import random_effects_meta_fisher_z  # noqa: E402
from power_analysis import achieved_power, format_power_sentence, min_detectable_r, power_summary_table  # noqa: E402
from trial_level_model import fit_human_vs_model, fit_omnibus  # noqa: E402

OUTDIR = ROOT / "results" / "robustness"


def _perm_p(a1: Dict[str, Any], model: str, metric: str = "sample_accuracy") -> float:
    try:
        return float(a1["per_model"][model][metric]["p_perm"])
    except Exception:
        try:
            return float(a1["per_model"][model]["det_correct"]["p_perm"])
        except Exception:
            return float("nan")


def _perm_rho(a1: Dict[str, Any], model: str, metric: str = "sample_accuracy") -> float:
    try:
        return float(a1["per_model"][model][metric]["rho"])
    except Exception:
        try:
            return float(a1["per_model"][model]["det_correct"]["rho"])
        except Exception:
            return float("nan")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    ap.add_argument("--metric", default="sample_accuracy")
    ap.add_argument("--n_boot", type=int, default=10_000)
    ap.add_argument("--skip_trial", action="store_true")
    ap.add_argument("--skip_disattenuation", action="store_true")
    ap.add_argument("--max_human_trials", type=int, default=80_000)
    args = ap.parse_args(argv)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    human = load_human_item_sensitivity()
    models = load_all_model_item_tables()
    a1 = load_a1_perm_summary(DEFAULT_A1)

    power_md = ["## Power analysis (Fisher-z, Pearson)", ""]
    power_frames = []
    for n in (36,):
        power_md.append(format_power_sentence(n))
        power_md.append("")
        tbl = power_summary_table(n)
        power_frames.append(tbl)
        try:
            power_md.append(tbl.to_markdown(index=False))
        except Exception:
            power_md.append(tbl.to_string(index=False))
        power_md.append("")
    if power_frames:
        pd.concat(power_frames).to_csv(outdir / "power_table.csv", index=False)

    human_rel = None
    if not args.skip_disattenuation:
        print("Computing human sensitivity reliability (bootstrap subjects)…", flush=True)
        human_rel = human_sensitivity_reliability(n_boot=40, seed=42)

    item_rows = []
    item_md = [
        "## Item-level analysis (n≈36, existing approach + rigor layer)",
        "",
        "Existing pipeline uses **Spearman ρ + item-label permutation** "
        "(`scripts/alignment_analyses.py::perm_spearman`). "
        "Below, Pearson r is used for TOST / BF / power / meta; "
        "Spearman perm p is passed through unchanged.",
        "",
        "### TOST note",
        "Default equivalence bound eps=0.30 ≈ Cohen's medium |r|; analyst choice "
        "(limitation). Sensitivity also at eps=0.20 and 0.40.",
        "",
    ]

    rs_meta, ns_meta = [], []
    disatt_blocks = []
    for model, mdf in sorted(models.items()):
        x, y, n = paired_item_vectors(human, mdf, metric=args.metric)
        if n < 5:
            x, y, n = paired_item_vectors(human, mdf, metric="det_correct")
        r_p, p_p = pearsonr(x, y)
        r_s, _ = spearmanr(x, y)
        bat = equivalence_battery(x, y, seed=42)
        boot = bootstrap_corr_ci(x, y, n_boot=args.n_boot, seed=42)
        bat["bootstrap"] = boot
        mdr = min_detectable_r(n)
        ap30 = achieved_power(n, 0.3)
        perm_p = _perm_p(a1, model, args.metric)
        perm_rho = _perm_rho(a1, model, args.metric)
        tost30 = bat["tost"]["eps_0.30"]
        bayes = bat["bayes"]

        row = {
            "model": model,
            "n_items": n,
            "pearson_r": float(r_p),
            "pearson_p": float(p_p),
            "spearman_r_recomputed": float(r_s),
            "spearman_rho_perm_pipeline": perm_rho,
            "spearman_p_perm_pipeline": perm_p,
            "boot_ci_low": boot["ci_low"],
            "boot_ci_high": boot["ci_high"],
            "tost_p_eps0.20": bat["tost"]["eps_0.20"]["p_tost"],
            "tost_equiv_eps0.20": bat["tost"]["eps_0.20"]["equivalent"],
            "tost_p_eps0.30": tost30["p_tost"],
            "tost_equiv_eps0.30": tost30["equivalent"],
            "tost_p_eps0.40": bat["tost"]["eps_0.40"]["p_tost"],
            "tost_equiv_eps0.40": bat["tost"]["eps_0.40"]["equivalent"],
            "BF10": bayes["BF10"],
            "BF01": bayes["BF01"],
            "BF_label": bayes["label"],
            "min_detectable_r": mdr,
            "power_at_r0.3": ap30,
        }
        item_rows.append(row)
        rs_meta.append(float(r_p))
        ns_meta.append(n)

        item_md += [
            f"### {model}",
            f"- Pearson r={r_p:.3f} (95% boot CI [{boot['ci_low']:.3f}, {boot['ci_high']:.3f}]), n={n}",
            f"- Existing Spearman ρ (perm pipeline)={perm_rho:.3f}, p_perm={perm_p:.4f}",
            f"- TOST eps=0.30: p_tost={tost30['p_tost']:.4f}, equivalent={bool(tost30['equivalent'])} "
            f"(eps0.20 equiv={bool(bat['tost']['eps_0.20']['equivalent'])}; "
            f"eps0.40 equiv={bool(bat['tost']['eps_0.40']['equivalent'])})",
            f"- BF10={bayes.get('BF10')}, BF01={bayes.get('BF01')} → {bayes.get('label')}",
            f"- {format_power_sentence(n)}",
            "",
        ]

        if not args.skip_disattenuation:
            print(f"Disattenuation: {model}…", flush=True)
            dres = run_disattenuation_for_model(
                model, metric=args.metric, human_rel_cache=human_rel, seed=42
            )
            disatt_blocks.append(dres)
            if dres.get("status") == "ok":
                d = dres["disattenuation"]
                item_md.append(
                    f"- Disattenuated r={d['r_disattenuated']:.3f} "
                    f"[{d['ci_low']:.3f}, {d['ci_high']:.3f}] "
                    f"(rel_human={d['rel_x']:.3f}, rel_model={d['rel_y']:.3f}); {dres.get('note', '')}"
                )
                row["r_disattenuated"] = d["r_disattenuated"]
                row["r_disatt_ci_low"] = d["ci_low"]
                row["r_disatt_ci_high"] = d["ci_high"]
            else:
                item_md.append(f"- Disattenuation: {dres.get('message', 'stub')}")
            item_md.append("")

    meta = random_effects_meta_fisher_z(rs_meta, ns_meta)
    item_md += [
        "### Random-effects meta-analysis (Pearson r, Fisher-z DL)",
        f"- k={int(meta['k'])} models; pooled r={meta['r_pooled']:.3f} "
        f"95% CI [{meta['ci_low']:.3f}, {meta['ci_high']:.3f}]",
        f"- τ²={meta['tau2']:.4f}; Q={meta['Q']:.3f} (df={int(meta['df'])}, p={meta['p_Q']:.3f}); "
        f"I²={meta['I2']:.1f}%",
        "",
        "Replaces informal concatenated pooled ρ with a random-effects estimate "
        "(still item-level; distinct from the trial-level omnibus).",
        "",
    ]

    trial_md = [
        "## Trial-level analysis (mixed-effects re-analysis of H1)",
        "",
        "Backend: `statsmodels.MixedLM` Gaussian approximation on binary outcomes "
        "(pymer4/lme4 and bambi/PyMC not in this venv). "
        "Maximal RE `(1 + eq_sensitivity_z | item)`; falls back to `(1|item)` on failure.",
        "",
        "Data used: human trials from `card_rmet_item_level.csv`; model trials from "
        "k=10 `samples.predictions` per item (no log-probs available for commercial APIs).",
        "",
    ]
    trial_summaries: List[Dict[str, Any]] = []
    if not args.skip_trial:
        model_names = sorted(models.keys())
        for model in model_names:
            print(f"Trial-level LMM: human vs {model}…", flush=True)
            res = fit_human_vs_model(model, max_human_trials=args.max_human_trials, seed=42)
            trial_summaries.append({k: v for k, v in res.items() if k != "blups"})
            if res.get("status") != "ok":
                trial_md.append(f"### {model}\n- STUB: {res.get('message')}\n")
                continue
            trial_md += [
                f"### human vs {model}",
                f"- N_human={res['n_human_trials']}, N_model={res['n_model_trials']}, "
                f"n_items={res['n_items']}; RE={res['re_formula_used']}",
                f"- {res['power_contrast_note']}",
            ]
            for t in res["interaction_terms"]:
                trial_md.append(
                    f"- {t['term']}: coef={t['coef']:.4f}, SE={t['se']:.4f}, p={t['p']:.4g}"
                )
            if res.get("blups") is not None:
                blup_path = outdir / f"blups_human_vs_{model}.csv"
                res["blups"].to_csv(blup_path, index=False)
                trial_md.append(
                    f"- BLUPs → `{blup_path.name}` (vs naive `trait_sensitivity_coef`)."
                )
            trial_md.append("")

        print("Trial-level omnibus…", flush=True)
        omn = fit_omnibus(model_names, max_human_trials=min(args.max_human_trials, 60_000), seed=42)
        trial_summaries.append({k: v for k, v in omn.items() if k != "blups"})
        if omn.get("status") == "ok":
            lrt = omn["lrt_interaction"]
            trial_md += [
                "### Omnibus (human + all models)",
                f"- N_total={omn['n_total_trials']} (human={omn['n_human_trials']}, "
                f"model={omn['n_model_trials']})",
                f"- LRT for eq_sensitivity × agent_type: LR={lrt['LR']:.3f}, "
                f"df={lrt['df']}, p={lrt['p']:.4g}",
                f"- {omn['power_contrast_note']}",
                "",
            ]
            for t in omn["interaction_terms"][:12]:
                trial_md.append(
                    f"- {t['term']}: coef={t['coef']:.4f}, SE={t['se']:.4f}, p={t['p']:.4g}"
                )
            trial_md.append("")
    else:
        trial_md.append("_Skipped (`--skip_trial`)._")

    header = [
        "# study4_rmet robustness report",
        "",
        "Generated by `study4_rmet/robustness/run_robustness_report.py`.",
        "Original Spearman+permutation A1 results remain in `results/alignment/`.",
        "",
    ]
    report = "\n".join(header + power_md + item_md + trial_md)
    report_path = outdir / "robustness_report.md"
    report_path.write_text(report, encoding="utf-8")

    summary_df = pd.DataFrame(item_rows)
    summary_df["meta_r_pooled"] = meta["r_pooled"]
    summary_df["meta_ci_low"] = meta["ci_low"]
    summary_df["meta_ci_high"] = meta["ci_high"]
    summary_df["meta_I2"] = meta["I2"]
    csv_path = outdir / "robustness_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    payload = {
        "meta_analysis": meta,
        "item_level": item_rows,
        "human_reliability": human_rel,
        "disattenuation": disatt_blocks,
        "trial_level": trial_summaries,
        "note_existing_test": "Spearman + permutation in alignment_analyses.py",
    }
    (outdir / "robustness_results.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )

    print(report)
    print(f"\nwrote {report_path}\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
