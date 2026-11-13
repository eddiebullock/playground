#!/usr/bin/env python3
"""SYNTHETIC rehearsal cells for A4 or A9, in the runners' exact row format (written 2026-11-13).

Used to rehearse the Phase 3 chain (pre-run check, frozen analysis, registry tables, Table 1, Figure 5)
before any real cell exists. Nothing here is a model output: option log probabilities come from a synthetic
observer that maps each stimulus's AU configuration to options with random weights (so the AUs carry signal
but not a human-like pattern), through the rotated option order the runner uses; blind rows carry family
scores, `lexicon_mass` and `repeat_change`, as `scripts/a4_run.score` writes them.

The output folder is required and must lie outside the project's `results/` tree, so that rehearsal cells can
never sit where `post_phase3.py pull` puts real ones. The manifest is rebuilt from public inputs:
    python research/scripts/a4_stimuli.py manifest --zip data/external/snoek_au/au_emotion_data.zip --out <dir>/manifest.csv
    python -I research/scripts/a9_manifest.py manifest --out <dir>/manifest.csv

Usage (project root):
    python research/scripts/rehearse_cells.py a4 --manifest <dir>/manifest.csv --out <dir>/cells
    python research/scripts/a4_prerun_check.py --cells_dir <dir>/cells --manifest <dir>/manifest.csv --clean_dir <dir>/clean --record <dir>/prerun.json
    python -m scripts.a4_analyse --cells_dir <dir>/cells --manifest <dir>/manifest.csv --gates research/reproduction/a4_gates.json \
        --detector research/reproduction/a4_v1_pilot.csv --output <dir>/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "research" / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

N_SHARDS = 12
SEEDS = {"a4": {"gemma4": 1, "qwen3vl": 2}, "a9": {"gemma4": 3, "qwen3vl": 4}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("amendment", choices=("a4", "a9"))
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    out = args.out.resolve()
    if out == ROOT / "results" or (ROOT / "results") in out.parents:
        raise SystemExit("Rehearsal cells may not be written under results/: real cells are pulled there.")
    import human_au_effects as hae
    from scripts import a4_design as d
    from scripts.analyse_stimulus_interventions import LEXICON

    if args.amendment == "a4":
        from scripts import a4_run as run
        design = d.design_hash()
    else:
        from scripts import a9_design as a9
        from scripts import a9_run as run
        design = a9.design_hash()
    manifest = pd.read_csv(args.manifest).to_dict("records")
    au_of = {r["stim_id"]: np.array([float(json.loads(r["au_json"]).get(c, 0.0)) for c in hae.AU_COLS]) for r in manifest}
    families = sorted(LEXICON)
    out.mkdir(parents=True, exist_ok=True)
    for model, seed in SEEDS[args.amendment].items():
        rng = np.random.default_rng(seed)
        weights = {t: rng.normal(0, 1.5, size=(len(hae.AU_COLS), len(d.OPTIONS[t]))) for t in d.OPTIONS}
        for k in range(N_SHARDS):
            rows = []
            for cell in run.planned_cells(manifest, k, N_SHARDS):
                row = {**cell, "status": "ok", "design_sha256": design, "seconds": 1.0, "forwards": 1, "synthetic": True}
                if cell["readout"] in ("blind", "blind_image_free"):
                    row["family_logprob"] = {f: round(float(rng.normal(-5, 1)), 4) for f in families}
                    row["lexicon_mass"] = round(float(rng.uniform(0.2, 0.9)), 5)
                    row["repeat_change"] = 0.0
                else:
                    canon = list(d.OPTIONS[cell["task"]])
                    options = d.rotated_options(cell["task"], cell["rotation"])
                    x = au_of.get(cell["stim_id"], np.zeros(len(hae.AU_COLS)))
                    logits = x @ weights[cell["task"]] + rng.normal(0, 0.5, size=len(canon))
                    if cell["readout"] == "image_free":
                        logits = rng.normal(0, 0.5, size=len(canon))
                    logp = logits - np.log(np.exp(logits).sum())
                    row["options"] = options
                    row["joint_logprob"] = [round(float(logp[canon.index(o)]), 4) for o in options]
                rows.append(row)
            (out / f"cells_{model}_{k}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        print(model, "written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
