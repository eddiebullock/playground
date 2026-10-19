"""Resumable layer x alpha sweep. Chunk to 30 min-2 h for backfill."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from esvlm.types import ScoreRow
from esvlm.utils.io import append_score_rows, finished_item_ids, parquet_shard_path


@dataclass(frozen=True)
class SweepCell:
    layer: int
    alpha: float
    mode: str
    vector: str
    position: str


def expand_grid(
    layers: Sequence[int],
    alphas: Sequence[float],
    *,
    modes: Sequence[str] = ("add",),
    vectors: Sequence[str] = ("E", "S"),
    position: str = "last_token",
) -> list[SweepCell]:
    """Ablate ignores alpha (one cell per layer x vector). Add uses every alpha."""

    cells: list[SweepCell] = []
    for layer in layers:
        for mode in modes:
            use_alphas: Sequence[float] = (0.0,) if mode == "ablate" else alphas
            for alpha in use_alphas:
                for vector in vectors:
                    cells.append(
                        SweepCell(
                            layer=int(layer),
                            alpha=float(alpha),
                            mode=str(mode),
                            vector=str(vector),
                            position=position,
                        )
                    )
    return cells


def shard_path(root: Path, model: str, task: str, cell: SweepCell, shard: str) -> Path:
    name = f"{cell.vector}_L{cell.layer}_{cell.mode}_a{cell.alpha}_{cell.position}"
    return parquet_shard_path(root, task, model, f"{name}_{shard}")


def pending_items(path: Path, item_ids: Iterable[str]) -> list[str]:
    done = finished_item_ids(path)
    return [i for i in item_ids if i not in done]


def write_rows(path: Path, rows: Iterable[ScoreRow]) -> None:
    append_score_rows(path, rows)
