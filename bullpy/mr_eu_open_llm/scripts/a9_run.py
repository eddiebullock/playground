"""
Draft amendment A9: score Liu renders on a model. See research/PREREG_A9_DRAFT.md.

Runner glue only. Scoring is A4's (`a4_run.score`: exact option probabilities after "EMOTION:",
one image, A1's prompt, basic options, rotation by `a4_design.rotated_options`), and renders are
read from tar files by `a4_run.RenderStore`, which uses the same tar sharding as
`a4_stimuli.render_tar`. Nothing in A4 is edited.

Refuses to run unless `--a9_frozen` is given and the draft pins A9's code (check_code_pins with
require=True), the A4 files and A9 inputs match the block, and the manifest is the pinned one, so
that no model sees a Liu render before the freeze. Resumable; shards are by stimulus id. Every
cell carries the A9 design hash.

Usage (cluster, after the freeze):
    python -m scripts.a9_run --model gemma4 --shard 0 --n_shards 12 --a9_frozen
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import a9_design as a9  # noqa: E402


def planned_cells(manifest: Sequence[Mapping[str, Any]], shard: int, n_shards: int) -> Iterator[Dict[str, Any]]:
    """Every cell for one shard, once per image whatever sets it belongs to: all rotations on
    the basic task, plus an image-free cell for the first BASELINE_STIMULI main rows (the
    manifest is in rank order). No blind readout in A9."""
    images: List[str] = []
    seen = set()
    baseline: List[str] = []
    for r in manifest:
        if r["dataset"] != a9.DATASET:
            raise SystemExit(f"Row from dataset {r['dataset']} in an A9 manifest.")
        if r["stim_id"] not in seen:
            seen.add(r["stim_id"])
            images.append(r["stim_id"])
        if r["set"] == "main" and len(baseline) < a9.BASELINE_STIMULI:
            baseline.append(r["stim_id"])
    base = set(baseline)
    for stim in sorted(images):
        if a9.shard_of(stim, n_shards) != shard:
            continue
        for rotation in range(a9.ROTATIONS):
            yield {"stim_id": stim, "task": a9.TASK, "rotation": rotation, "readout": "mcq"}
        if stim in base:
            yield {"stim_id": stim, "task": a9.TASK, "rotation": 0, "readout": "image_free"}


def cell_key(cell: Mapping[str, Any]) -> tuple:
    return (cell["stim_id"], cell["task"], cell["rotation"], cell["readout"])


def preflight(manifest_path: Path, frozen: bool) -> str:
    """Everything that must hold before a model may see a render. Returns the design hash."""
    if not frozen:
        raise SystemExit("A9 is not frozen. No model may see a Liu render until it is.")
    a9.check_code_pins(require=True)
    a9.check_pins()
    a9.check_manifest(manifest_path)
    return a9.design_hash()


def main() -> None:
    import pandas as pd

    from scripts import a4_run
    from scripts import run_stimulus_interventions as run
    from scripts.face_input_validity import read_cells

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, choices=(*a9.MODELS, *a9.DESCRIPTIVE_MODELS))
    parser.add_argument("--manifest", type=Path, default=Path("results/a9/manifest.csv"))
    parser.add_argument("--renders", type=Path, default=Path("results/a9/renders"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/a9"))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--n_shards", type=int, required=True)
    parser.add_argument("--max_cells", type=int, help="Timing smoke, written beside the real cells.")
    parser.add_argument("--a9_frozen", action="store_true", help="Required. Set only after A9 is frozen.")
    args = parser.parse_args()
    if not 0 <= args.shard < args.n_shards:
        raise SystemExit("--shard must lie in 0 to n_shards - 1.")
    digest = preflight(args.manifest, args.a9_frozen)
    manifest = pd.read_csv(args.manifest).to_dict("records")
    suffix = f"smoke_{args.shard}" if args.max_cells else str(args.shard)
    path = args.output_dir / f"cells_{args.model}_{suffix}.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state = read_cells(path)
    if state["n_partial_dropped"]:
        with path.open("r+b") as handle_file:
            handle_file.truncate(state["good_bytes"])
    done = {cell_key(r) for r in state["rows"]}
    handle = run.load_handle(args.model, args.dtype)
    store = a4_run.RenderStore(args.renders)
    written = 0
    with path.open("a", encoding="utf-8") as sink:
        for cell in planned_cells(manifest, args.shard, args.n_shards):
            if cell_key(cell) in done:
                continue
            image = store.image(cell["stim_id"])
            cell["design_sha256"] = digest
            sink.write(json.dumps(a4_run.score(handle, cell, image)) + "\n")
            sink.flush()
            written += 1
            if args.max_cells and written >= args.max_cells:
                break
    print("Done:", path, written, digest)


if __name__ == "__main__":
    main()
