"""Unsteered baseline scoring. Not a confirmatory steering sweep."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

from esvlm.scoring.logprob import score_item
from esvlm.tasks.items import load_eu_emotions_items, load_mindreading_items, load_rmet_items
from esvlm.tasks.media import load_media
from esvlm.tasks.sdomain import load_sdomain_items
from esvlm.types import MCQItem
from esvlm.utils.envcheck import check_job_env
from esvlm.utils.io import append_score_rows, finished_item_ids, parquet_shard_path
from esvlm.utils.logging import echo_run_args, setup_logging
from esvlm.utils.seeding import seed_everything

PUBLIC_TASKS = frozenset({"physbench", "clevrer", "piqa"})
E_TASKS = frozenset({"rmet", "eu_emotions", "mindreading"})
DEFAULT_TEMPLATE = {
    "physbench": "mechanism",
    "clevrer": "mechanism",
    "piqa": "mechanism",
    "rmet": "classic",
    "eu_emotions": "direct",
    "mindreading": "direct",
}
FLUSH_EVERY = 8
_STOP = False


def _on_usr1(_signum: int, _frame: object) -> None:
    global _STOP
    _STOP = True


def _env_path(name: str) -> Path:
    raw = os.environ.get(name, "")
    if not raw:
        raise SystemExit(f"{name} is unset")
    return Path(raw)


def load_task_items(
    task: str, manifest: Path | None, *, template_id: str = ""
) -> tuple[list[MCQItem], Path | None]:
    tid = template_id or DEFAULT_TEMPLATE.get(task, "mechanism")
    if task == "synthetic":
        return [], None
    if task in PUBLIC_TASKS:
        key = task.upper()
        man = manifest or _env_path(f"ESVLM_{key}_MANIFEST")
        media_root = _env_path(f"ESVLM_{key}_DIR")
        return load_sdomain_items(man, media_root, task=task, template_id=tid), man
    if task == "rmet":
        man = manifest or _env_path("ESVLM_RMET_MANIFEST")
        return (
            load_rmet_items(man, _env_path("ESVLM_RMET_ANSWER_KEY"), _env_path("ESVLM_RMET_DIR"), template_id=tid),
            man,
        )
    if task == "eu_emotions":
        man = manifest or _env_path("ESVLM_EU_EMOTIONS_MANIFEST")
        entropy = os.environ.get("ESVLM_EU_EMOTIONS_ENTROPY")
        return (
            load_eu_emotions_items(
                man,
                _env_path("ESVLM_EU_EMOTIONS_DIR"),
                template_id=tid,
                entropy_path=Path(entropy) if entropy else None,
            ),
            man,
        )
    if task == "mindreading":
        man = manifest or _env_path("ESVLM_MINDREADING_MANIFEST")
        return load_mindreading_items(man, _env_path("ESVLM_MINDREADING_DIR"), template_id=tid), man
    raise SystemExit(f"unknown task {task}")


def slice_items(items: list[MCQItem], offset: int, limit: int) -> list[MCQItem]:
    if offset < 0:
        raise ValueError("trial_offset must be >= 0")
    end = None if limit <= 0 else offset + limit
    return items[offset:end]


def main(argv: list[str] | None = None) -> int:
    check_job_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trial_offset", type=int, default=0)
    parser.add_argument("--trial_limit", type=int, default=0)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--template", default="")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(os.environ["ESVLM_MODELS_DIR"]) if os.environ.get("ESVLM_MODELS_DIR") else None,
    )
    args = parser.parse_args(argv)
    args.template = args.template or DEFAULT_TEMPLATE.get(args.task, "mechanism")
    log = setup_logging()
    echo_run_args(vars(args), log)
    seed_everything(args.seed)

    if args.task == "synthetic":
        from esvlm.cli.smoke import main as smoke

        smoke_args = [
            "--model",
            args.model,
            "--out",
            str(args.out),
            "--seed",
            str(args.seed),
        ]
        if args.offline:
            smoke_args.append("--offline")
        if args.models_dir is not None:
            smoke_args.extend(["--models-dir", str(args.models_dir)])
        return smoke(smoke_args)

    items, manifest = load_task_items(args.task, args.manifest, template_id=args.template)
    items = slice_items(items, args.trial_offset, args.trial_limit)
    if not items:
        raise SystemExit(f"no items for task {args.task}")
    n_unique = len({it.item_id for it in items})
    if n_unique != len(items):
        raise SystemExit(
            f"duplicate item_id in {args.task} slice: {len(items)} items, {n_unique} unique"
        )

    from esvlm.models.loader import load_model
    from esvlm.utils.device import torch_device

    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _on_usr1)

    t0 = time.perf_counter()
    loaded = load_model(args.model, models_dir=args.models_dir, offline=args.offline)
    load_s = time.perf_counter() - t0
    device = torch_device(loaded.device)
    shard = os.environ.get("SHARD", "0")
    parquet = parquet_shard_path(args.out, args.task, loaded.spec.key, f"baseline_{shard}")
    done = finished_item_ids(parquet)
    pending = [it for it in items if it.item_id not in done]
    log.info(
        "baseline_start %s",
        json.dumps(
            {
                "model": loaded.spec.key,
                "task": args.task,
                "manifest": str(manifest) if manifest else None,
                "n_slice": len(items),
                "n_pending": len(pending),
                "load_seconds": round(load_s, 3),
            }
        ),
    )

    rows = []
    scored = 0
    t1 = time.perf_counter()
    for item in pending:
        if _STOP:
            log.info("usr1_stop after %s items", scored)
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
            condition="baseline",
            vector="none",
            layer=None,
            alpha=0.0,
            mode="none",
            frames=args.frames if media else 0,
            seed=args.seed,
        )
        rows.append(row)
        scored += 1
        if len(rows) >= FLUSH_EVERY:
            append_score_rows(parquet, rows)
            rows = []
    if rows:
        append_score_rows(parquet, rows)
    score_s = time.perf_counter() - t1
    correct = 0
    total = 0
    if parquet.exists():
        import pyarrow.parquet as pq

        table = pq.read_table(parquet, columns=["correct", "item_id"])
        keep = {it.item_id for it in items}
        flags = [
            bool(c) for c, iid in zip(table.column("correct").to_pylist(), table.column("item_id").to_pylist()) if iid in keep
        ]
        total = len(flags)
        correct = sum(flags)
    summary = {
        "model": loaded.spec.key,
        "revision": loaded.spec.revision,
        "task": args.task,
        "n_slice": len(items),
        "n_scored_this_run": scored,
        "n_rows": total,
        "accuracy": round(correct / total, 4) if total else None,
        "load_seconds": round(load_s, 3),
        "score_seconds": round(score_s, 3),
        "parquet": str(parquet),
        "stopped": _STOP,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("baseline_summary %s", json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
