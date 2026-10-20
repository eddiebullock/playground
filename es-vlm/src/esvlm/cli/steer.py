"""Layer x alpha steering sweep. Resumable per cell parquet. OSF is not a gate."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

import numpy as np

from esvlm.analysis.separability import residualize
from esvlm.cli.analyze_extract import load_layer
from esvlm.cli.score import DEFAULT_TEMPLATE, load_task_items, slice_items
from esvlm.scoring.logprob import score_item
from esvlm.steering.hooks import SteeringHook
from esvlm.steering.sweep import expand_grid, shard_path, write_rows
from esvlm.tasks.media import load_media
from esvlm.utils.envcheck import check_job_env
from esvlm.utils.io import finished_item_ids
from esvlm.utils.logging import echo_run_args, setup_logging
from esvlm.utils.seeding import seed_everything

FLUSH_EVERY = 8
_STOP = False


def _on_usr1(_signum: int, _frame: object) -> None:
    global _STOP
    _STOP = True


def _parse_ints(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x != ""]


def _parse_floats(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",") if x != ""]


def _parse_csv(spec: str) -> list[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def load_item_id_filter(path: Path, *, key: str = "hold_social_ids") -> set[str]:
    """JSON list of ids, or an extract summary with ``key``."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        ids = payload
    elif isinstance(payload, dict):
        ids = payload.get(key)
        if ids is None:
            raise SystemExit(f"{path} has no {key}")
    else:
        raise SystemExit(f"{path} is not a JSON list or object")
    return {str(x) for x in ids}


def resolve_direction(vecs: dict[str, np.ndarray], name: str) -> np.ndarray:
    if name in vecs:
        return vecs[name]
    if name == "E_unique":
        return residualize(vecs["E"], vecs["S"])
    if name == "S_unique":
        return residualize(vecs["S"], vecs["E"])
    raise SystemExit(f"direction {name} missing; have {sorted(vecs)}")


def cell_accuracy(path: Path) -> float | None:
    if not path.exists():
        return None
    import pyarrow.parquet as pq

    flags = pq.read_table(path, columns=["correct"]).column("correct").to_pylist()
    if not flags:
        return None
    return round(sum(bool(x) for x in flags) / len(flags), 4)


def main(argv: list[str] | None = None) -> int:
    check_job_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--alphas", required=True)
    parser.add_argument("--vectors", default="E,S,E_minus_S")
    parser.add_argument("--modes", default="add")
    parser.add_argument("--position", default="last_token")
    parser.add_argument("--directions", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--trial_offset", type=int, default=0)
    parser.add_argument("--trial_limit", type=int, default=0)
    parser.add_argument("--item-ids", type=Path)
    parser.add_argument("--item-ids-key", default="hold_social_ids")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--template", default="")
    parser.add_argument(
        "--allow-unpreregistered",
        action="store_true",
        help="ignored; OSF is not required",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(os.environ["ESVLM_MODELS_DIR"]) if os.environ.get("ESVLM_MODELS_DIR") else None,
    )
    args = parser.parse_args(argv)
    args.template = args.template or DEFAULT_TEMPLATE.get(args.task, "mechanism")
    args.directions = args.directions or Path(f"outputs/extract_{args.model}_0")
    args.out = args.out or Path(f"outputs/steer_{args.model}_{args.task}_0")
    log = setup_logging()
    echo_run_args(vars(args), log)
    seed_everything(args.seed)

    layers = _parse_ints(args.layers)
    alphas = _parse_floats(args.alphas)
    vectors = _parse_csv(args.vectors)
    modes = _parse_csv(args.modes)
    cells = expand_grid(
        layers, alphas, modes=modes, vectors=vectors, position=args.position
    )
    items, manifest = load_task_items(args.task, args.manifest, template_id=args.template)
    items = slice_items(items, args.trial_offset, args.trial_limit)
    if args.item_ids is not None:
        keep = load_item_id_filter(args.item_ids, key=args.item_ids_key)
        items = [it for it in items if it.item_id in keep]
    if not items:
        raise SystemExit(f"no items for task {args.task}")
    n_unique = len({it.item_id for it in items})
    if n_unique != len(items):
        raise SystemExit(
            f"duplicate item_id in {args.task} slice: {len(items)} items, {n_unique} unique"
        )
    item_ids = [it.item_id for it in items]

    from esvlm.models.loader import load_model, residual_modules
    from esvlm.utils.device import torch_device

    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _on_usr1)

    t0 = time.perf_counter()
    loaded = load_model(args.model, models_dir=args.models_dir, offline=args.offline)
    modules = residual_modules(loaded)
    device = torch_device(loaded.device)
    load_s = time.perf_counter() - t0
    for layer in layers:
        if layer < 0 or layer >= len(modules):
            raise SystemExit(f"layer {layer} out of range 0..{len(modules) - 1}")

    shard = os.environ.get("SHARD", "0")
    args.out.mkdir(parents=True, exist_ok=True)
    log.info(
        "steer_start %s",
        json.dumps(
            {
                "model": loaded.spec.key,
                "task": args.task,
                "n_items": len(items),
                "n_cells": len(cells),
                "directions": str(args.directions),
                "load_seconds": round(load_s, 3),
            }
        ),
    )

    cell_rows: list[dict] = []
    scored = 0
    t1 = time.perf_counter()
    for cell in cells:
        if _STOP:
            break
        npz = args.directions / f"{loaded.spec.key}__layer_{cell.layer:02d}.npz"
        if not npz.is_file():
            raise SystemExit(f"missing direction file {npz}")
        vecs = load_layer(npz)
        direction = resolve_direction(vecs, cell.vector)
        parquet = shard_path(args.out, loaded.spec.key, args.task, cell, shard)
        done = finished_item_ids(parquet)
        pending = [it for it in items if it.item_id not in done]
        log.info(
            "steer_cell %s",
            json.dumps(
                {
                    "vector": cell.vector,
                    "layer": cell.layer,
                    "mode": cell.mode,
                    "alpha": cell.alpha,
                    "n_pending": len(pending),
                    "parquet": str(parquet),
                }
            ),
        )
        if not pending:
            cell_rows.append(
                {
                    "vector": cell.vector,
                    "layer": cell.layer,
                    "mode": cell.mode,
                    "alpha": cell.alpha,
                    "n_rows": len(done),
                    "accuracy": cell_accuracy(parquet),
                    "skipped": True,
                }
            )
            continue
        hook = SteeringHook(
            modules[cell.layer],
            direction,
            alpha=cell.alpha,
            mode=cell.mode,  # type: ignore[arg-type]
            position=cell.position,  # type: ignore[arg-type]
        )
        hook.attach()
        try:
            buf = []
            for item in pending:
                if _STOP:
                    break
                media = load_media(item, frames=args.frames)
                row = score_item(
                    loaded.model,
                    loaded.processor,
                    item,
                    device=device,
                    images=media or None,
                    model_key=loaded.spec.key,
                    revision=loaded.spec.revision,
                    template_id=args.template,
                    condition="steer",
                    vector=cell.vector,
                    layer=cell.layer,
                    alpha=cell.alpha,
                    mode=cell.mode,
                    frames=args.frames if media else 0,
                    seed=args.seed,
                )
                buf.append(row)
                scored += 1
                if len(buf) >= FLUSH_EVERY:
                    write_rows(parquet, buf)
                    buf = []
            if buf:
                write_rows(parquet, buf)
        finally:
            hook.remove()
        cell_rows.append(
            {
                "vector": cell.vector,
                "layer": cell.layer,
                "mode": cell.mode,
                "alpha": cell.alpha,
                "n_rows": len(finished_item_ids(parquet)),
                "accuracy": cell_accuracy(parquet),
                "skipped": False,
            }
        )

    summary = {
        "model": loaded.spec.key,
        "revision": loaded.spec.revision,
        "task": args.task,
        "manifest": str(manifest) if manifest else None,
        "n_items": len(items),
        "item_ids_head": item_ids[:8],
        "n_cells": len(cells),
        "n_scored_this_run": scored,
        "load_seconds": round(load_s, 3),
        "score_seconds": round(time.perf_counter() - t1, 3),
        "stopped": _STOP,
        "directions": str(args.directions),
        "out": str(args.out),
        "cells": cell_rows,
    }
    (args.out / "steer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("steer_summary %s", json.dumps({k: summary[k] for k in summary if k != "cells"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
