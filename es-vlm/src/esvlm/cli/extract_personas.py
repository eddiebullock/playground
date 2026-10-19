"""Project 500 mundane personas onto saved E/S directions. Not a steering sweep."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from esvlm.analysis.plots import plot_layer_series
from esvlm.analysis.separability import residualize
from esvlm.cli.analyze_extract import load_layer
from esvlm.utils.envcheck import check_job_env
from esvlm.utils.logging import echo_run_args, setup_logging
from esvlm.utils.seeding import seed_everything
from esvlm.vectors.extract import text_residuals
from esvlm.vectors.personas import persona_matrix, persona_texts


def _layer_files(directions: Path, model: str) -> list[Path]:
    files = sorted(directions.glob(f"{model}__layer_*.npz"))
    if files:
        return files
    return sorted(directions.glob("*__layer_*.npz"))


def main(argv: list[str] | None = None) -> int:
    check_job_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(os.environ["ESVLM_MODELS_DIR"]) if os.environ.get("ESVLM_MODELS_DIR") else None,
    )
    args = parser.parse_args(argv)
    log = setup_logging()
    echo_run_args(vars(args), log)
    seed_everything(args.seed)

    files = _layer_files(args.directions, args.model)
    if not files:
        raise SystemExit(f"no direction npz under {args.directions}")
    wanted = None if args.layers in {"", "all"} else {int(x) for x in args.layers.split(",") if x}
    layer_paths: list[tuple[int, Path]] = []
    for path in files:
        layer = int(path.stem.rsplit("_", 1)[-1])
        if wanted is not None and layer not in wanted:
            continue
        layer_paths.append((layer, path))
    if not layer_paths:
        raise SystemExit("no matching direction layers")

    texts = persona_texts(args.n, seed=args.seed)
    from esvlm.models.loader import load_model, residual_modules
    from esvlm.utils.device import torch_device

    t0 = time.perf_counter()
    loaded = load_model(args.model, models_dir=args.models_dir, offline=args.offline)
    modules = residual_modules(loaded)
    layer_ids = [layer for layer, _ in layer_paths]
    for layer in layer_ids:
        if layer < 0 or layer >= len(modules):
            raise SystemExit(f"layer {layer} out of range 0..{len(modules) - 1}")
    hooked = [modules[layer] for layer in layer_ids]
    device = torch_device(loaded.device)
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    acts = text_residuals(loaded.model, loaded.processor, hooked, texts, device)
    if acts.shape[0] != len(texts):
        raise SystemExit(f"persona residual count {acts.shape[0]} != {len(texts)}")

    e_scores = []
    s_scores = []
    eu_scores = []
    su_scores = []
    rows = []
    for offset, (layer, path) in enumerate(layer_paths):
        vecs = load_layer(path)
        if "E" not in vecs or "S" not in vecs:
            raise SystemExit(f"{path} missing E/S directions")
        raw = persona_matrix(acts[:, offset, :], vecs["E"], vecs["S"])
        e_u = residualize(vecs["E"], vecs["S"])
        s_u = residualize(vecs["S"], vecs["E"])
        uniq = persona_matrix(acts[:, offset, :], e_u, s_u)
        e_scores.append(raw.e)
        s_scores.append(raw.s)
        eu_scores.append(uniq.e)
        su_scores.append(uniq.s)
        rows.append(
            {
                "layer": layer,
                "r_E_S": raw.correlation,
                "r_E_unique_S_unique": uniq.correlation,
                "n": int(len(raw.e)),
            }
        )
        log.info("persona layer=%s r=%.3f r_unique=%.3f", layer, raw.correlation, uniq.correlation)

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / "persona_scores.npz",
        e=np.stack(e_scores, axis=1),
        s=np.stack(s_scores, axis=1),
        e_unique=np.stack(eu_scores, axis=1),
        s_unique=np.stack(su_scores, axis=1),
        layers=np.asarray(layer_ids, dtype=np.int32),
    )
    (args.out / "persona_texts.json").write_text(json.dumps(texts, indent=2), encoding="utf-8")
    rs = [float(r["r_E_S"]) for r in rows]
    ru = [float(r["r_E_unique_S_unique"]) for r in rows]
    summary = {
        "model": loaded.spec.key,
        "revision": loaded.spec.revision,
        "n": args.n,
        "seed": args.seed,
        "layers": layer_ids,
        "r_E_S_mean": float(np.mean(rs)) if rs else None,
        "r_E_S_min": float(np.min(rs)) if rs else None,
        "r_E_S_max": float(np.max(rs)) if rs else None,
        "r_unique_mean": float(np.mean(ru)) if ru else None,
        "load_seconds": round(load_s, 3),
        "extract_seconds": round(time.perf_counter() - t1, 3),
        "directions": str(args.directions),
        "out": str(args.out),
        "note": "path check; mundane personas, not questionnaire items",
        "by_layer": rows,
    }
    (args.out / "persona_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_layer_series(
        layer_ids,
        {"r(E, S)": rs, "r(E_perp, S_perp)": ru},
        args.out / "persona_r_by_layer.png",
        ylabel="Pearson r",
        title=f"{loaded.spec.key} persona E/S correlation",
    )
    log.info("persona_summary %s", json.dumps({k: summary[k] for k in summary if k != "by_layer"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
