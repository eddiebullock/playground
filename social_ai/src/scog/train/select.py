"""Pick the exploratory LoRA run with the lowest dev negative log-likelihood."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def choose(records: list[dict]) -> dict:
    """Lowest answer-token NLL. Ties: fewer epochs, then lower rank, then lower learning rate."""
    if not records:
        raise RuntimeError("hyperparameter selection received zero records")
    required = {"run_id", "epoch", "rank", "lr", "nll", "n_examples"}
    for record in records:
        missing = required - set(record)
        if missing:
            raise RuntimeError(f"selection record is missing {sorted(missing)}")
        if int(record["n_examples"]) < 10:
            raise RuntimeError("selection record has fewer than 10 dev examples")

    def key(record: dict) -> tuple:
        return (float(record["nll"]), int(record["epoch"]), int(record["rank"]), float(record["lr"]))

    winner = min(records, key=key)
    return {
        "run_id": winner["run_id"],
        "epoch": int(winner["epoch"]),
        "rank": int(winner["rank"]),
        "lr": float(winner["lr"]),
        "nll": float(winner["nll"]),
    }


def load_grid_records(runs_dir: Path, expected: list[dict]) -> list[dict] | None:
    """Return 12 records when every grid cell has epoch 1 and epoch 2. Otherwise None."""
    records = []
    for cell in expected:
        run_id = cell["run_id"]
        for epoch in (1, 2):
            path = runs_dir / run_id / f"dev_nll_epoch{epoch}.json"
            if not path.is_file():
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["run_id"] = run_id
            payload["epoch"] = epoch
            payload["rank"] = int(cell["rank"])
            payload["lr"] = float(cell["lr"])
            records.append(payload)
    if len(records) != len(expected) * 2:
        raise RuntimeError("grid record count does not match the preregistered grid")
    return records


def main() -> int:
    runs_dir = Path(os.environ["A2_RUNS"])
    grid_path = Path(os.environ.get("A2_GRID", Path(__file__).resolve().parents[3] / "configs" / "a2_grid.json"))
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    records = load_grid_records(runs_dir, grid["grid"])
    if records is None:
        print("grid_incomplete")
        return 2
    winner = choose(records)
    out = runs_dir / "winner.json"
    out.write_text(json.dumps(winner, indent=2) + "\n", encoding="utf-8")
    print(f"winner\t{winner['run_id']}\tepoch\t{winner['epoch']}\trank\t{winner['rank']}\tlr\t{winner['lr']}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"select failed: {exc}", file=sys.stderr)
        raise
