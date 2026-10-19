"""Sharded parquet for rows; one npz per model x layer x run.

RDS has an inode quota of about 1.05M files. Never write one file per item.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from esvlm import __version__
from esvlm.types import ScoreRow


def write_layer_npz(
    path: Path,
    *,
    activations: np.ndarray,
    item_ids: list[str],
    meta: Mapping[str, Any],
) -> None:
    """One file per model x layer x run."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "activations": np.asarray(activations),
        "item_ids": np.asarray(item_ids),
        "meta_json": np.asarray(json.dumps(dict(meta), sort_keys=True)),
    }
    np.savez_compressed(path, **payload)


def read_layer_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta_json"]))
        return {
            "activations": z["activations"],
            "item_ids": z["item_ids"].tolist(),
            "meta": meta,
        }


def parquet_shard_path(root: Path, task: str, model: str, shard: str) -> Path:
    return root / f"{task}__{model}__{shard}.parquet"


def finished_item_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    table = pq.read_table(path, columns=["item_id"])
    return set(table.column("item_id").to_pylist())


def append_score_rows(path: Path, rows: Iterable[ScoreRow]) -> None:
    """Append rows, creating the file if needed. Keyed by item for resume."""

    records = [row_to_record(r) for r in rows]
    if not records:
        return
    frame = pd.DataFrame.from_records(records)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pq.read_table(path)
        table = pa.concat_tables([existing, table], promote_options="default")
    pq.write_table(table, path)


def row_to_record(row: ScoreRow) -> dict[str, Any]:
    return {
        "model": row.model,
        "revision": row.revision,
        "package_version": __version__,
        "task": row.task,
        "item_id": row.item_id,
        "template_id": row.template_id,
        "condition": row.condition,
        "vector": row.vector,
        "layer": row.layer,
        "alpha": row.alpha,
        "mode": row.mode,
        "frames": row.frames,
        "option_logprobs": json.dumps(list(row.option_logprobs)),
        "pred": row.pred,
        "correct": bool(row.correct),
        "seed": row.seed,
        "tokens": row.tokens,
        "meta": json.dumps(row.meta, sort_keys=True),
    }
