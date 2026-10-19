"""RQ1 cosine readout from saved directions. Not a steering result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from esvlm.analysis.plots import plot_alpha_accuracy, plot_separability
from esvlm.analysis.separability import separability_matrix, summarize_direction
from esvlm.utils.logging import echo_run_args, setup_logging


def load_layer(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files if k != "meta_json"}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        (a / np.linalg.norm(a)) @ (b / np.linalg.norm(b))
    )


def analyze_root(root: Path, out: Path) -> dict:
    files = sorted(root.glob("*__layer_*.npz"))
    if not files:
        raise SystemExit(f"no layer npz under {root}")
    names: list[str] | None = None
    rows: list[dict] = []
    es: list[float] = []
    e_held: list[float] = []
    s_held: list[float] = []
    layers: list[int] = []
    mid_mat = None
    mid_names: list[str] = []
    for path in files:
        vecs = load_layer(path)
        keys = sorted(k for k in vecs if not k.endswith("_pos") and not k.endswith("_neg"))
        if names is None:
            names = keys
        layer = int(path.stem.rsplit("_", 1)[-1])
        layers.append(layer)
        rec = {"layer": layer}
        for a in keys:
            for b in keys:
                rec[f"cos_{a}_{b}"] = cosine(vecs[a], vecs[b])
        if "E" in vecs and "S" in vecs:
            es.append(rec["cos_E_S"])
        if "E_pos" in vecs and "E_neg" in vecs and "E" in vecs:
            rec["E"] = summarize_direction(vecs["E_pos"], vecs["E_neg"], vecs["E"])
            e_held.append(float(rec["E"]["held_out_auroc"]))
        if "S_pos" in vecs and "S_neg" in vecs and "S" in vecs:
            rec["S"] = summarize_direction(vecs["S_pos"], vecs["S_neg"], vecs["S"])
            s_held.append(float(rec["S"]["held_out_auroc"]))
        if "E_minus_S_pos" in vecs and "E_minus_S_neg" in vecs and "E_minus_S" in vecs:
            rec["E_minus_S"] = summarize_direction(
                vecs["E_minus_S_pos"], vecs["E_minus_S_neg"], vecs["E_minus_S"]
            )
        rows.append(rec)
        if path == files[len(files) // 2]:
            mid_names = keys
            mid_mat = separability_matrix({k: vecs[k] for k in keys})

    out.mkdir(parents=True, exist_ok=True)
    model = files[0].name.split("__", 1)[0]
    summary = {
        "model": model,
        "n_layers": len(layers),
        "contrasts": names,
        "cos_E_S_mean": float(np.mean(es)) if es else None,
        "cos_E_S_min": float(np.min(es)) if es else None,
        "cos_E_S_max": float(np.max(es)) if es else None,
        "E_held_out_auroc_mean": float(np.mean(e_held)) if e_held else None,
        "S_held_out_auroc_mean": float(np.mean(s_held)) if s_held else None,
        "layers": rows,
    }
    (out / f"separability_{model}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if mid_mat is not None and mid_names:
        plot_separability(mid_mat, mid_names, out / f"separability_{model}_mid.png")
    if es:
        plot_alpha_accuracy(
            layers,
            {"cos(E,S)": es},
            out / f"separability_{model}_es.png",
            title=f"{model} cos(E, S) by layer",
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/aggregate"))
    args = parser.parse_args(argv)
    echo_run_args(vars(args), setup_logging())
    summary = analyze_root(args.root, args.out)
    keys = (
        "model",
        "n_layers",
        "cos_E_S_mean",
        "cos_E_S_min",
        "cos_E_S_max",
        "E_held_out_auroc_mean",
        "S_held_out_auroc_mean",
    )
    print(json.dumps({k: summary[k] for k in keys}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
