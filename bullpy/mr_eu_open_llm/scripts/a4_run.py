"""
Amendment A4: score rendered AU faces on a model. See research/PREREG_A4_DRAFT.md.

Scoring is A3's: exact option probabilities after "EMOTION:", on one image. The blind readout
is A2's frozen prompt. This script refuses to run until the amendment is frozen, so that no
model sees a render before then. It is resumable, and shards are by stimulus id.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import a4_design as d  # noqa: E402

BASELINE_STIMULI = 50


class RenderStore:
    """Reads renders out of their tar files. One PNG per render would hit the file quota."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.open_tars: Dict[int, tarfile.TarFile] = {}

    def image(self, stim_id: str) -> np.ndarray:
        from PIL import Image

        index = d.shard_of(stim_id, d.RENDER_TARS)
        if index not in self.open_tars:
            self.open_tars[index] = tarfile.open(self.folder / f"renders_{index:02d}.tar", "r:")
        member = self.open_tars[index].extractfile(f"{stim_id}.png")
        return np.asarray(Image.open(io.BytesIO(member.read())).convert("RGB"))


def planned_cells(manifest: Sequence[Mapping[str, Any]], shard: int, n_shards: int) -> Iterator[Dict[str, Any]]:
    """Every cell for one shard, each once, whichever sets a stimulus belongs to."""
    tasks: Dict[str, set] = {}
    blind: set = set()
    baseline: Dict[str, List[str]] = {t: [] for t in d.OPTIONS}
    for r in manifest:
        if r["set"] in ("one_au", "neutral"):
            tasks.setdefault(r["stim_id"], set()).update(d.OPTIONS)
        elif r["set"] == "blind":
            blind.add(r["stim_id"])
        else:
            task = d.TASK_OF[r["dataset"]]
            tasks.setdefault(r["stim_id"], set()).add(task)
            if r["set"] == "main" and len(baseline[task]) < BASELINE_STIMULI:
                baseline[task].append(r["stim_id"])
            if r["set"] == "replicate" and r["dataset"] == d.BLIND_DATASET:
                blind.add(r["stim_id"])
    blind_baseline = set(sorted(blind)[:BASELINE_STIMULI])
    for stim in sorted(set(tasks) | blind):
        if d.shard_of(stim, n_shards) != shard:
            continue
        for task in sorted(tasks.get(stim, ())):
            for rotation in d.rotations(task):
                yield {"stim_id": stim, "task": task, "rotation": rotation, "readout": "mcq"}
            if stim in baseline[task]:
                yield {"stim_id": stim, "task": task, "rotation": 0, "readout": "image_free"}
        if stim in blind:
            yield {"stim_id": stim, "task": "basic", "rotation": 0, "readout": "blind"}
            if stim in blind_baseline:
                yield {"stim_id": stim, "task": "basic", "rotation": 0, "readout": "blind_image_free"}


def cell_key(cell: Mapping[str, Any]) -> tuple:
    return (cell["stim_id"], cell["task"], cell["rotation"], cell["readout"])


def score(handle: Any, cell: Dict[str, Any], image: np.ndarray) -> Dict[str, Any]:
    from scripts import run_stimulus_interventions as run
    from scripts.prompts import build_4afc_prompt
    from scripts.vision_gates import strip_pixels

    start, tick = handle.forwards, time.time()
    media = run.Media("image", [image], [0], {})
    if cell["readout"] in ("blind", "blind_image_free"):
        inputs = run.encode_media(handle, media, run.BLIND_PROMPT)
        if cell["readout"] == "blind_image_free":
            inputs = strip_pixels(inputs, run.PIXEL_KEYS)
        values = run.score_sequences(handle, inputs, handle.blind_prefill, handle.word_ids)
        again = run.score_sequences(handle, inputs, handle.blind_prefill, handle.word_ids)
        cell.update(family_logprob={k: round(v, 4) for k, v in run.family_scores(handle.words, values).items()},
                    lexicon_mass=round(math.exp(run.logsumexp(values)), 5),
                    repeat_change=float(max(abs(a - b) for a, b in zip(values, again))))
    else:
        options = d.rotated_options(cell["task"], cell["rotation"])
        inputs = run.encode_media(handle, media, build_4afc_prompt(options, condition="video_only"))
        if cell["readout"] == "image_free":
            inputs = strip_pixels(inputs, run.PIXEL_KEYS)
        ids = [list(handle.tokenizer(f" {o}", add_special_tokens=False).input_ids) for o in options]
        scores = run.score_sequences(handle, inputs, handle.mcq_prefill, ids)
        cell.update(options=options, joint_logprob=[round(v, 4) for v in scores])
    cell.update(status="ok", seconds=round(time.time() - tick, 3), forwards=handle.forwards - start)
    return cell


def main() -> None:
    import pandas as pd

    from scripts import run_stimulus_interventions as run
    from scripts.face_input_validity import prepare_resume, read_cells  # noqa: F401

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, choices=(*d.MODELS, *d.DESCRIPTIVE_MODELS))
    parser.add_argument("--manifest", type=Path, default=Path("results/a4/manifest.csv"))
    parser.add_argument("--renders", type=Path, default=Path("results/a4/renders"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/a4"))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--n_shards", type=int, required=True)
    parser.add_argument("--max_cells", type=int, help="Timing smoke, written beside the real cells.")
    parser.add_argument("--a4_frozen", action="store_true", help="Required. Set only after A4 is frozen.")
    args = parser.parse_args()
    if not args.a4_frozen:
        raise SystemExit("A4 is not frozen. No model may see a render until it is.")
    if not 0 <= args.shard < args.n_shards:
        raise SystemExit("--shard must lie in 0 to n_shards - 1.")
    digest = d.design_hash()
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
    store = RenderStore(args.renders)
    written = 0
    with path.open("a", encoding="utf-8") as sink:
        for cell in planned_cells(manifest, args.shard, args.n_shards):
            if cell_key(cell) in done:
                continue
            image = store.image(cell["stim_id"])
            cell["design_sha256"] = digest
            sink.write(json.dumps(score(handle, cell, image)) + "\n")
            sink.flush()
            written += 1
            if args.max_cells and written >= args.max_cells:
                break
    print("Done:", path, written, digest)


if __name__ == "__main__":
    main()
