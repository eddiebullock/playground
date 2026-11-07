#!/usr/bin/env python3
"""Amendment A4: the stimulus manifest, and rendering from it. No model is involved.

The manifest is built from the public Snoek files by seeded hash, using nothing but the
configuration strings and face gender. `manifest` writes it; `render --tar i` renders one tar
of PNGs. Images go under results/ in tar files, never as loose files and never into git.

Usage:
    python research/scripts/a4_stimuli.py manifest --zip data/external/snoek_au/au_emotion_data.zip \
        --out results/a4/manifest.csv
    python research/scripts/a4_stimuli.py render --manifest results/a4/manifest.csv \
        --out-dir results/a4/renders --tar 0
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "scripts"))
from scripts import a4_design as d  # noqa: E402

COLUMNS = ["stim_id", "set", "dataset", "cfg", "identity", "seed", "gender", "scale", "au_json"]


def hash_key(*parts: object) -> str:
    return hashlib.sha256("|".join([str(d.SEED), *map(str, parts)]).encode()).hexdigest()


def stim_id(identity: str, seed: int, au: Mapping[str, float]) -> str:
    """Same image, same id, so that a repeated configuration is rendered once."""
    return hash_key("image", identity, seed, json.dumps(dict(sorted(au.items()))))[:20]


def scaled(au: Mapping[str, float], scale: float) -> Dict[str, float]:
    """Scale amplitudes. Rendered amplitude stays within 0 to 1, as the renderer requires."""
    return {k: round(min(1.0, float(v) * scale), 4) for k, v in au.items()}


def choose_face(dataset: str, cfg: str, genders: Sequence[str]):
    group = dataset.split("_")[1]
    gender = sorted(genders, key=lambda g: hash_key("gender", dataset, cfg, g))[0]
    faces = d.IDENTITIES[group][gender]
    return gender, faces[int(hash_key("face", dataset, cfg), 16) % len(faces)]


def main_rows(frames: Mapping[str, pd.DataFrame], au_cols: Sequence[str]) -> List[dict]:
    rows = []
    for dataset in d.DATASETS:
        frame = frames[dataset]
        table = frame.drop_duplicates("cfg").set_index("cfg")
        genders = frame.groupby("cfg")["face_gender"].unique()
        ranked = sorted(table.index, key=lambda c: hash_key("rank", dataset, c))
        for rank, cfg in enumerate(ranked[: d.CONFIGS_PER_DATASET]):
            gender, (identity, seed) = choose_face(dataset, cfg, list(genders[cfg]))
            au = {c: float(table.at[cfg, c]) for c in au_cols if table.at[cfg, c] > 0}
            rows.append(row("main", dataset, cfg, identity, seed, gender, 1.0, au))
            other = [f for f in d.IDENTITIES[dataset.split("_")[1]][gender] if f != (identity, seed)]
            if rank < d.REPLICATE_CONFIGS:
                rows.append(row("replicate", dataset, cfg, other[0][0], other[0][1], gender, 1.0, au))
            if dataset in d.SCALE_DATASETS and rank < d.SCALE_CONFIGS:
                for scale in d.SCALES:
                    rows.append(row(f"scale_{scale}", dataset, cfg, identity, seed, gender, scale, au))
            if dataset == d.BLIND_DATASET and rank < d.BLIND_CONFIGS:
                rows.append(row("blind", dataset, cfg, identity, seed, gender, 1.0, au))
    return rows


def row(kind, dataset, cfg, identity, seed, gender, scale, au) -> dict:
    au = scaled(au, scale)
    return {"stim_id": stim_id(identity, seed, au), "set": kind, "dataset": dataset, "cfg": cfg,
            "identity": identity, "seed": seed, "gender": gender, "scale": scale,
            "au_json": json.dumps(dict(sorted(au.items())))}


def one_au_rows(au_cols: Sequence[str]) -> List[dict]:
    rows = []
    for identity, seed in d.ONE_AU_IDENTITIES:
        gender = "M" if "male" in identity and "female" not in identity else "F"
        rows.append(row("neutral", "one_au", "neutral", identity, seed, gender, 1.0, {}))
        for column in au_cols:
            for amplitude in d.ONE_AU_AMPLITUDES:
                rows.append(row("one_au", "one_au", column, identity, seed, gender, 1.0, {column: amplitude}))
    return rows


def build(frames: Mapping[str, pd.DataFrame], au_cols: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(main_rows(frames, au_cols) + one_au_rows(au_cols), columns=COLUMNS)


def render_tar(manifest: pd.DataFrame, out_dir: Path, index: int) -> int:
    """Render one tar of PNGs, built aside and moved into place, so a killed job leaves none.

    One tar per file-quota slot: images are never written as loose files. A finished tar is
    skipped, and a render is deterministic, so a rerun gives the same pixels.
    """
    import render_au_faces as raf
    from PIL import Image

    path = out_dir / f"renders_{index:02d}.tar"
    if path.exists():
        return 0
    unique = manifest.drop_duplicates("stim_id").sort_values("stim_id")
    chosen = [r for r in unique.itertuples() if d.shard_of(r.stim_id, d.RENDER_TARS) == index]
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    with tarfile.open(partial, "w:") as tar:
        for r in chosen:
            buffer = io.BytesIO()
            Image.fromarray(raf.render_face(json.loads(r.au_json), r.identity, int(r.seed)), "RGB").save(
                buffer, format="PNG", compress_level=6)
            info = tarfile.TarInfo(f"{r.stim_id}.png")
            info.size = buffer.tell()
            buffer.seek(0)
            tar.addfile(info, buffer)
    os.replace(partial, path)
    return len(chosen)


def build_assets(src: Path, out: Path) -> Dict[str, object]:
    """Archive the render assets under a src/ prefix, which the Slurm scripts expect.

    Jobs unpack this to node-local storage, because it holds thousands of small files.
    """
    def keep(info: tarfile.TarInfo):
        return None if Path(info.name).name.startswith("._") or "/.git" in info.name else info

    with tarfile.open(out, "w:gz") as tar:
        tar.add(src, arcname="src", filter=keep)
    with tarfile.open(out, "r:gz") as tar:
        count = sum(1 for m in tar if m.isfile())
    return {"files": count, "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("manifest")
    a.add_argument("--zip", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("render")
    b.add_argument("--manifest", type=Path, required=True)
    b.add_argument("--out-dir", type=Path, required=True)
    b.add_argument("--tar", type=int, required=True, help="0 to RENDER_TARS - 1")
    c = sub.add_parser("assets", help="Build the one archive the cluster jobs unpack.")
    c.add_argument("--src", type=Path, required=True, help="Folder holding FACSHuman and the rest.")
    c.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "assets":
        print(json.dumps(build_assets(args.src, args.out)))
    elif args.command == "manifest":
        import human_au_effects as hae

        loaded, _ = hae.load_zip(args.zip)
        table = build({k: v[0] for k, v in loaded.items()}, hae.AU_COLS)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(json.dumps({"rows": len(table), "unique_images": int(table.stim_id.nunique()),
                          "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}))
    else:
        if not 0 <= args.tar < d.RENDER_TARS:
            raise SystemExit("--tar is out of range.")
        print(render_tar(pd.read_csv(args.manifest), args.out_dir, args.tar))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
