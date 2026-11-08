#!/usr/bin/env python3
"""Rebuild the merged all-layer Phase 2 gate and its per-layer CSV from the per-model dumps.

Commit f6dd3cf (2026-11-02) committed `phase2_gate_all_layers.json` and
`phase2_all_layers_summary.csv` after merging the three per-model outputs of
`python -m paper_v2.probe --layer_set all_layers` by hand; no script was committed. This
script reproduces that merge from the committed per-model files only, with no recomputation:

- `models` is the three per-model `models` blocks, unchanged, in the order Qwen, Gemma, Molmo.
- `n_layers`, the branch and the Holm floor note are derived from those blocks.
- `max_frames` (4) and the CSD3 job ids are run metadata that the per-model files do not hold.
  They are written here as constants, copied from the f6dd3cf merged file and the extract and
  probe job logs named in RESULTS_REGISTRY.md.
- The CSV has one row per model and layer. `holm_only_fail` is True when the layer fails only
  `perm_holm_lt_0p01` and passes the other four gate checks.

Usage (from bullpy/mr_eu_open_llm):

    python research/scripts/merge_phase2_all_layers.py            # write both files
    python research/scripts/merge_phase2_all_layers.py --check    # compare with committed files
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "research" / "reproduction"
MODELS = ("qwen3vl", "gemma4", "molmo2")
OUT_JSON = REPRO / "phase2_gate_all_layers.json"
OUT_CSV = REPRO / "phase2_all_layers_summary.csv"

# Run metadata not stored in the per-model files (extract with max_frames=4; CSD3 job ids).
MAX_FRAMES = 4
JOBS = {
    "extract": {"qwen3vl": 39709580, "gemma4": 39709581, "molmo2": 39709582},
    "probe": {"qwen3vl": 39712190, "gemma4": 39712191, "molmo2": 39712192},
}
LABELS = {"qwen3vl": "Qwen", "gemma4": "Gemma", "molmo2": "Molmo"}
CSV_FIELDS = ["model", "layer", "above_chance", "t1_grouped_rho", "t1_loo_rho", "p_perm",
              "p_perm_holm", "stim_rho", "delta_rho", "cat_resid_rho", "conf_resid_rho",
              "gate_all", "holm_only_fail"]
OTHER_CHECKS = ("delta_rho_ge_0p10", "delta_ci_excludes_0", "category_residual_ci_excludes_0",
                "confidence_residual_ci_excludes_0")


def load_per_model(directory: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    for model in MODELS:
        payload = json.loads((directory / f"phase2_gate_all_layers_{model}.json").read_text(encoding="utf-8"))
        out[model] = payload
    return out


def merged_payload(per_model: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    first = per_model[MODELS[0]]
    for model, payload in per_model.items():
        for key in ("prereg", "layer_set", "token_position", "exploratory", "branch_locked"):
            if payload[key] != first[key]:
                raise SystemExit(f"{model}: {key} differs from {MODELS[0]}")
    models = {m: per_model[m]["models"][m] for m in MODELS}
    eligible = sorted({e for m in MODELS for e in per_model[m]["decision"]["eligible_models"]})
    branch = "A" if eligible else "B"
    n_layers = {m: len(models[m]["layers"]) for m in MODELS}
    floors = {m: n_layers[m] / (models[m]["n_perm"] + 1) for m in MODELS}
    n_perm = {models[m]["n_perm"] for m in MODELS}
    if len(n_perm) != 1:
        raise SystemExit("n_perm differs between models")
    n_perm_value = n_perm.pop()
    floor_text = ", ".join(f"{LABELS[m]} {floors[m]:.3f}" for m in MODELS)
    all_above = all(f > 0.01 for f in floors.values())
    holm_note = (f"n_perm={n_perm_value} implies minimum permutation p = 1/{n_perm_value + 1}. "
                 f"Holm floor is n_layers/{n_perm_value + 1} ({floor_text}), "
                 f"{'all > 0.01' if all_above else 'not all > 0.01'}. "
                 + ("No layer can pass perm_holm_lt_0p01 under this n_perm. " if all_above else "")
                 + "Rule not retuned.")
    return {
        "prereg": first["prereg"],
        "layer_set": first["layer_set"],
        "token_position": first["token_position"],
        "max_frames": MAX_FRAMES,
        "exploratory": first["exploratory"],
        "branch_locked": first["branch_locked"],
        "branch": branch,
        "would_be_branch": branch,
        "decision": {"branch": branch, "eligible_models": eligible, "rule": first["decision"]["rule"]},
        "jobs": JOBS,
        "n_layers": n_layers,
        "holm_floor_note": holm_note,
        "note": f"Official Phase 2 gate on all last-token layers. Branch {branch} locked 2026-11-02.",
        "models": models,
    }


def csv_rows(models: Dict[str, Dict[str, Any]]) -> List[List[Any]]:
    rows = []
    for model in MODELS:
        block = models[model]
        for layer in block["layers"]:
            gate = layer["gate"]
            holm_only = (not gate["all"]) and (not gate["perm_holm_lt_0p01"]) and all(gate[c] for c in OTHER_CHECKS)
            rows.append([model, layer["layer"], block["above_chance"],
                         layer["t1_grouped"]["rho"], layer["t1_loo"]["rho"],
                         layer["t1_grouped_perm"]["p_perm"], layer["t1_grouped_perm"]["p_perm_holm"],
                         layer["stimulus_t1_grouped"]["rho"],
                         layer["delta_t1_minus_stimulus_grouped"]["delta_rho"],
                         layer["t1_category_residual_grouped"]["rho"],
                         layer["t1_confidence_residual_grouped"]["rho"],
                         gate["all"], holm_only])
    return rows


def render(per_model: Dict[str, Dict[str, Any]]) -> Dict[Path, str]:
    payload = merged_payload(per_model)
    json_text = json.dumps(payload, indent=2) + "\n"
    buffer = io.StringIO()
    writer = csv.writer(buffer)  # default dialect: CRLF line ends, as committed in f6dd3cf
    writer.writerow(CSV_FIELDS)
    writer.writerows(csv_rows(payload["models"]))
    return {OUT_JSON: json_text, OUT_CSV: buffer.getvalue()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="compare with the committed files, write nothing")
    parser.add_argument("--in-dir", type=Path, default=REPRO, help="folder holding the per-model JSONs")
    args = parser.parse_args(argv)
    outputs = render(load_per_model(args.in_dir))
    if args.check:
        ok = True
        for path, text in outputs.items():
            same = path.read_bytes() == text.encode("utf-8")
            print(f"{path.name}: {'byte-identical' if same else 'DIFFERS'}")
            ok &= same
        return 0 if ok else 1
    for path, text in outputs.items():
        path.write_bytes(text.encode("utf-8"))
        print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
