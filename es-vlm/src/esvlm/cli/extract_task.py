"""Task-conditioned residual directions. Not narrative E/S prose."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from esvlm.cli.score import DEFAULT_TEMPLATE, load_task_items, slice_items
from esvlm.scoring.logprob import _encode
from esvlm.tasks.media import load_media
from esvlm.utils.envcheck import check_job_env
from esvlm.utils.logging import echo_run_args, setup_logging
from esvlm.utils.seeding import seed_everything
from esvlm.vectors.extract import ResidualCatcher, difference_of_means, directions_for_layers
from esvlm.vectors.validate import held_out_auroc


def item_residuals(model, processor, modules, item, images, device) -> np.ndarray:
    """Last-token residual at every hooked layer for the question only. [L, D]."""

    import torch

    inputs = _encode(processor, item.prompt_text, images)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    catcher = ResidualCatcher(modules)
    with torch.no_grad(), catcher:
        model(**inputs)
    return catcher.stacked()


def stack_task(
    model,
    processor,
    modules,
    items,
    device,
    *,
    frames: int,
    log,
) -> tuple[np.ndarray, list[str]]:
    acts = []
    ids = []
    for item in items:
        media = load_media(item, frames=frames)
        acts.append(item_residuals(model, processor, modules, item, media or None, device))
        ids.append(item.item_id)
        if len(ids) % 32 == 0:
            log.info("task_residual n=%s last=%s", len(ids), item.item_id)
    return np.stack(acts, axis=0), ids


def main(argv: list[str] | None = None) -> int:
    check_job_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--tasks",
        default="rmet,clevrer",
        help="comma-separated; first is treated as social, second as mechanical",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trial_limit_social", type=int, default=0)
    parser.add_argument("--trial_limit_mechanical", type=int, default=256)
    parser.add_argument("--frames", type=int, default=8)
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

    names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if len(names) != 2:
        raise SystemExit("need exactly two tasks: social,mechanical")
    social_task, mech_task = names
    social_items, _ = load_task_items(
        social_task, None, template_id=DEFAULT_TEMPLATE.get(social_task, "")
    )
    mech_items, _ = load_task_items(
        mech_task, None, template_id=DEFAULT_TEMPLATE.get(mech_task, "")
    )
    social_items = slice_items(social_items, 0, args.trial_limit_social)
    mech_items = slice_items(mech_items, 0, args.trial_limit_mechanical)
    if len(social_items) < 8 or len(mech_items) < 8:
        raise SystemExit("need at least 8 items per task")

    from esvlm.models.loader import load_model, residual_modules
    from esvlm.utils.device import torch_device

    t0 = time.perf_counter()
    loaded = load_model(args.model, models_dir=args.models_dir, offline=args.offline)
    modules = residual_modules(loaded)
    device = torch_device(loaded.device)
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    social, social_ids = stack_task(
        loaded.model, loaded.processor, modules, social_items, device, frames=args.frames, log=log
    )
    mech, mech_ids = stack_task(
        loaded.model, loaded.processor, modules, mech_items, device, frames=args.frames, log=log
    )
    vecs = directions_for_layers(social, mech)
    held = [
        held_out_auroc(social[:, i, :], mech[:, i, :], seed=args.seed) for i in range(vecs.shape[0])
    ]
    best = int(np.argmax(np.asarray(held)))
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / f"{loaded.spec.key}__task_residuals.npz",
        social=social,
        mechanical=mech,
        social_minus_mechanical=vecs,
        social_ids=np.asarray(social_ids),
        mechanical_ids=np.asarray(mech_ids),
    )
    for i in range(vecs.shape[0]):
        np.savez_compressed(
            args.out / f"{loaded.spec.key}__layer_{i:02d}.npz",
            social_minus_mechanical=vecs[i],
            E=vecs[i],
            S=difference_of_means(mech[:, i, :], social[:, i, :]),
        )
    summary = {
        "model": loaded.spec.key,
        "revision": loaded.spec.revision,
        "construct": "task_conditioned",
        "social_task": social_task,
        "mechanical_task": mech_task,
        "n_social": len(social_ids),
        "n_mechanical": len(mech_ids),
        "n_layers": int(vecs.shape[0]),
        "dim": int(vecs.shape[-1]),
        "held_out_auroc_by_layer": [float(x) for x in held],
        "held_out_auroc_mean": float(np.mean(held)),
        "candidate_layer": best,
        "candidate_held_out_auroc": float(held[best]),
        "load_seconds": round(load_s, 3),
        "extract_seconds": round(time.perf_counter() - t1, 3),
        "out": str(args.out),
        "note": "question-only last-token residual; not narrative E/S",
    }
    (args.out / "task_extract_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log.info("task_extract_summary %s", json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
