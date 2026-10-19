"""RQ1 summary from saved extracts. Not a steering result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from esvlm.analysis.plots import plot_layer_series
from esvlm.analysis.separability import unique_fraction
from esvlm.cli.analyze_extract import analyze_root, load_layer
from esvlm.utils.logging import echo_run_args, setup_logging

CONFIRMATORY = ("qwen3vl_8b", "qwen3vl_2b", "gemma4_e4b", "qwen3_text")
CONTROL_AUROC_MIN = 0.65


def _held(rec: dict, name: str) -> float | None:
    block = rec.get(name)
    if isinstance(block, dict) and "held_out_auroc" in block:
        return float(block["held_out_auroc"])
    return None


def report_model(extract_root: Path, out: Path, *, reuse_separability: bool = True) -> dict:
    extract_root = Path(extract_root)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(extract_root.glob("*__layer_*.npz"))
    if not files:
        raise SystemExit(f"no layer npz under {extract_root}")
    model = files[0].name.split("__", 1)[0]
    sep_path = out / f"separability_{model}.json"
    if reuse_separability and sep_path.exists():
        sep = json.loads(sep_path.read_text(encoding="utf-8"))
    else:
        sep = analyze_root(extract_root, out)

    by_layer = {int(row["layer"]): row for row in sep["layers"]}
    rows: list[dict] = []
    unique_e: list[float] = []
    unique_s: list[float] = []
    e_vs_s: list[float] = []
    layers: list[int] = []
    for path in files:
        layer = int(path.stem.rsplit("_", 1)[-1])
        vecs = load_layer(path)
        rec = {
            "layer": layer,
            "cos_E_S": float(by_layer.get(layer, {}).get("cos_E_S", float("nan"))),
            "E_vs_control_held": _held(by_layer.get(layer, {}), "E"),
            "S_vs_control_held": _held(by_layer.get(layer, {}), "S"),
            "E_vs_S_held": _held(by_layer.get(layer, {}), "E_minus_S"),
        }
        if "E" in vecs and "S" in vecs:
            rec["unique_E_frac"] = unique_fraction(vecs["E"], vecs["S"])
            rec["unique_S_frac"] = unique_fraction(vecs["S"], vecs["E"])
            unique_e.append(rec["unique_E_frac"])
            unique_s.append(rec["unique_S_frac"])
        if rec["E_vs_S_held"] is not None:
            e_vs_s.append(rec["E_vs_S_held"])
        layers.append(layer)
        rows.append(rec)

    eligible = [
        r
        for r in rows
        if (r.get("E_vs_control_held") or 0) >= CONTROL_AUROC_MIN
        and (r.get("S_vs_control_held") or 0) >= CONTROL_AUROC_MIN
        and r.get("E_vs_S_held") is not None
    ]
    if eligible:
        pick = max(eligible, key=lambda r: float(r["E_vs_S_held"]))
        reason = (
            f"max E-vs-S held-out AUROC among layers with both control "
            f"AUROCs >= {CONTROL_AUROC_MIN}"
        )
    elif e_vs_s:
        pick = max(rows, key=lambda r: float(r.get("E_vs_S_held") or -1.0))
        reason = "no layer met the control AUROC floor; max E-vs-S held-out AUROC"
    else:
        pick = None
        reason = "E-vs-S AUROC missing"

    summary = {
        "model": model,
        "n_layers": len(rows),
        "cos_E_S_mean": sep.get("cos_E_S_mean"),
        "E_vs_control_held_mean": sep.get("E_held_out_auroc_mean"),
        "S_vs_control_held_mean": sep.get("S_held_out_auroc_mean"),
        "E_vs_S_held_mean": float(np.mean(e_vs_s)) if e_vs_s else None,
        "E_vs_S_held_min": float(np.min(e_vs_s)) if e_vs_s else None,
        "E_vs_S_held_max": float(np.max(e_vs_s)) if e_vs_s else None,
        "unique_E_frac_mean": float(np.mean(unique_e)) if unique_e else None,
        "unique_S_frac_mean": float(np.mean(unique_s)) if unique_s else None,
        "candidate_layer": None if pick is None else pick["layer"],
        "candidate_E_vs_S_held": None if pick is None else pick.get("E_vs_S_held"),
        "candidate_reason": reason,
        "control_auroc_min": CONTROL_AUROC_MIN,
        "note": "path check; not a confirmatory steering result",
        "layers": rows,
    }
    (out / f"rq1_{model}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if layers and e_vs_s:
        plot_layer_series(
            layers,
            {
                "E vs S AUROC": [float(r.get("E_vs_S_held") or float("nan")) for r in rows],
                "unique E frac": [float(r.get("unique_E_frac") or float("nan")) for r in rows],
            },
            out / f"rq1_{model}_es.png",
            ylabel="AUROC / fraction",
            title=f"{model} E vs S (held-out AUROC) and unique-E fraction",
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--out", type=Path, default=Path("results/aggregate"))
    parser.add_argument(
        "--models",
        default=",".join(CONFIRMATORY),
        help="comma-separated model keys",
    )
    parser.add_argument(
        "--recompute-separability",
        action="store_true",
        help="ignore existing separability JSON and re-run the 200-split null",
    )
    args = parser.parse_args(argv)
    echo_run_args(vars(args), setup_logging())
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    panel: list[dict] = []
    for key in models:
        root = args.outputs / f"extract_{key}_0"
        if not root.is_dir():
            raise SystemExit(f"missing extract dir {root}")
        summary = report_model(
            root,
            args.out,
            reuse_separability=not args.recompute_separability,
        )
        brief = {
            k: summary[k]
            for k in (
                "model",
                "n_layers",
                "cos_E_S_mean",
                "E_vs_control_held_mean",
                "S_vs_control_held_mean",
                "E_vs_S_held_mean",
                "unique_E_frac_mean",
                "candidate_layer",
                "candidate_E_vs_S_held",
            )
        }
        print(json.dumps(brief, indent=2))
        panel.append(brief)
    (args.out / "rq1_panel.json").write_text(json.dumps({"models": panel}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
