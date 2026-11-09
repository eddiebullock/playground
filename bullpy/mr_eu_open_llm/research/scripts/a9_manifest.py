#!/usr/bin/env python3
"""Draft amendment A9: the render manifest for Liu et al. 2022 Experiment II. No model is involved.

Rows have A4's manifest columns (research/scripts/a4_stimuli.COLUMNS), and `au_json` is a
{column: amplitude} map over A4's 33 columns with amplitudes in 0 to 1, which is what
render_au_faces.render_face consumes; ids come from a4_stimuli.stim_id, so a repeated image has one
id. Rendering reuses a4_stimuli.render_tar, into tar files, never loose images and never into git.

Configurations come from Liu's peak amplitudes (`p`) through a9_design.liu_to_columns. Liu does
not release face gender per stimulus (humans saw 600 female and 600 male random identities), so
gender is assigned here by seeded hash, 600 and 600, using nothing but the stimulus index.

Usage:
    python -I research/scripts/a9_manifest.py manifest --out results/a9/manifest.csv
    python -I research/scripts/a9_manifest.py render --manifest results/a9/manifest.csv \
        --out-dir results/a9/renders --tar 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "scripts"))
sys.path.insert(0, str(ROOT))

import a4_stimuli as stim  # noqa: E402

from scripts import a9_design as a9  # noqa: E402

COLUMNS = stim.COLUMNS


def hash_key(*parts: object) -> str:
    return hashlib.sha256("|".join([str(a9.SEED), *map(str, parts)]).encode()).hexdigest()


def cfg_name(index: int) -> str:
    return f"liu{index + 1:04d}"


def ranked(n: int = a9.N_STIMULI) -> List[int]:
    """Stimulus order by seeded hash of the index. It picks the replicate and V2 subsets."""
    return sorted(range(n), key=lambda i: hash_key("rank", a9.DATASET, cfg_name(i)))


def genders(n: int = a9.N_STIMULI) -> Dict[int, str]:
    """Half female, half male, by seeded hash, as humans saw 600 and 600."""
    order = sorted(range(n), key=lambda i: hash_key("gender", a9.DATASET, cfg_name(i)))
    return {i: ("F" if k < n // 2 else "M") for k, i in enumerate(order)}


def face_for(index: int, gender: str):
    faces = a9.IDENTITIES[gender]
    return faces[int(hash_key("face", a9.DATASET, cfg_name(index)), 16) % len(faces)]


def peak_frame_amplitudes(pseq: np.ndarray, index: int) -> np.ndarray:
    """Amplitudes of the 42 AUs at the frame with the largest summed activation."""
    frames = pseq[:, :, index]
    return frames[:, int(np.argmax(frames.sum(axis=0)))]


def build(p: np.ndarray, pseq: np.ndarray, design: str = "A") -> List[dict]:
    if design not in a9.DESIGNS:
        raise ValueError(design)
    sex = genders(p.shape[1])
    rank = {i: k for k, i in enumerate(ranked(p.shape[1]))}
    rows: List[dict] = []
    for i in ranked(p.shape[1]):
        cfg = cfg_name(i)
        gender = sex[i]
        identity, seed = face_for(i, gender)
        au = a9.liu_to_columns(dict(zip(a9.LIU_LABELS, p[:, i])))
        rows.append(stim.row("main", a9.DATASET, cfg, identity, seed, gender, 1.0, au))
        other = [f for f in a9.IDENTITIES[gender] if f != (identity, seed)][0]
        if rank[i] < a9.REPLICATE_CONFIGS:
            rows.append(stim.row("replicate", a9.DATASET, cfg, other[0], other[1], gender, 1.0, au))
        frame_au = a9.liu_to_columns(dict(zip(a9.LIU_LABELS, peak_frame_amplitudes(pseq, i))))
        rows.append(stim.row("frame_max", a9.DATASET, cfg, identity, seed, gender, 1.0, frame_au))
        if design == "B":
            for g in ("F", "M"):
                for face in a9.IDENTITIES[g]:
                    if face != (identity, seed):
                        rows.append(stim.row("all_identities", a9.DATASET, cfg, face[0], face[1], g, 1.0, au))
    return rows


def summary(rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    sets: Dict[str, int] = {}
    for r in rows:
        sets[str(r["set"])] = sets.get(str(r["set"]), 0) + 1
    main = [r for r in rows if r["set"] == "main"]
    return {"rows": len(rows), "unique_images": len({r["stim_id"] for r in rows}), "rows_by_set": sets,
            "main_gender": {g: sum(r["gender"] == g for r in main) for g in ("F", "M")},
            "main_identity": {f"{a}_{b}": sum((r["identity"], r["seed"]) == (a, b) for r in main)
                              for faces in a9.IDENTITIES.values() for a, b in faces},
            "main_empty_configurations": sum(r["au_json"] == "{}" for r in main)}


def main(argv=None) -> int:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--design", choices=a9.DESIGNS, default=a9.DESIGN_SELECTED)
    r = sub.add_parser("render")
    r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--tar", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "render":
        frame = pd.read_csv(args.manifest)
        print(stim.render_tar(frame, args.out_dir, args.tar))
        return 0
    from a9_human_target import load_liu

    data = load_liu()
    rows = build(data["p"], data["pseq"], args.design)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(args.out, index=False)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(json.dumps({"manifest": str(args.out), "sha256": digest, "design": args.design, **summary(rows)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
