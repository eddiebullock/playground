"""Extract E/S and control residual directions. Not a steering sweep."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from esvlm.utils.envcheck import check_job_env
from esvlm.utils.logging import echo_run_args, setup_logging
from esvlm.utils.seeding import seed_everything
from esvlm.steering.hooks import unit_vector
from esvlm.vectors.contrasts import default_contrasts
from esvlm.vectors.extract import directions_for_layers, text_residuals


def parse_layers(spec: str, n_layers: int) -> list[int]:
    if spec in {"", "all"}:
        return list(range(n_layers))
    out = [int(x) for x in spec.split(",") if x != ""]
    for i in out:
        if i < 0 or i >= n_layers:
            raise SystemExit(f"layer {i} out of range 0..{n_layers - 1}")
    return out


def main(argv: list[str] | None = None) -> int:
    check_job_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--max-texts", type=int, default=0, help="0 = all contrast texts")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(os.environ["ESVLM_MODELS_DIR"]) if os.environ.get("ESVLM_MODELS_DIR") else None,
    )
    args = parser.parse_args(argv)
    log = setup_logging()
    echo_run_args(vars(args), log)
    seed_everything(args.seed)

    from esvlm.models.loader import load_model, residual_modules
    from esvlm.utils.device import torch_device

    t0 = time.perf_counter()
    loaded = load_model(args.model, models_dir=args.models_dir, offline=args.offline)
    modules = residual_modules(loaded)
    layer_ids = parse_layers(args.layers, len(modules))
    hooked = [modules[i] for i in layer_ids]
    device = torch_device(loaded.device)
    load_s = time.perf_counter() - t0

    contrasts = default_contrasts()
    cap = args.max_texts if args.max_texts > 0 else None
    vectors: dict[str, np.ndarray] = {}
    residuals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    t1 = time.perf_counter()
    for name, pair in contrasts.items():
        pos_texts = pair.positive[:cap]
        neg_texts = pair.negative[:cap]
        pos = text_residuals(loaded.model, loaded.processor, hooked, pos_texts, device)
        neg = text_residuals(loaded.model, loaded.processor, hooked, neg_texts, device)
        vectors[name] = directions_for_layers(pos, neg)
        if name in {"E", "S", "E_minus_S"}:
            residuals[name] = (pos, neg)
        log.info("extracted %s shape=%s", name, list(vectors[name].shape))

    rng = np.random.default_rng(args.seed)
    dim = next(iter(vectors.values())).shape[-1]
    vectors["random"] = np.stack(
        [unit_vector(rng.normal(size=dim)) for _ in layer_ids],
        axis=0,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    for offset, layer_i in enumerate(layer_ids):
        path = args.out / f"{loaded.spec.key}__layer_{layer_i:02d}.npz"
        payload = {name: vec[offset] for name, vec in vectors.items()}
        for name, (pos, neg) in residuals.items():
            payload[f"{name}_pos"] = pos[:, offset, :]
            payload[f"{name}_neg"] = neg[:, offset, :]
        payload["meta_json"] = np.asarray(
            json.dumps(
                {
                    "model": loaded.spec.key,
                    "revision": loaded.spec.revision,
                    "layer": layer_i,
                    "n_layers": len(modules),
                    "seed": args.seed,
                    "contrasts": sorted(k for k in vectors if k != "random"),
                },
                sort_keys=True,
            )
        )
        np.savez_compressed(path, **payload)

    summary = {
        "model": loaded.spec.key,
        "revision": loaded.spec.revision,
        "n_layers_model": len(modules),
        "layers": layer_ids,
        "contrasts": sorted(vectors),
        "dim": dim,
        "load_seconds": round(load_s, 3),
        "extract_seconds": round(time.perf_counter() - t1, 3),
        "out": str(args.out),
    }
    (args.out / "extract_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("extract_summary %s", json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
