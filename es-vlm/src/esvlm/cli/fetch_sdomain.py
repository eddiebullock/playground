"""Download public S-domain sets into $ESVLM_PUBLIC_DIR and write manifests."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlopen

from esvlm.tasks.sdomain import (
    clevrer_to_trials,
    merge_physbench_labels,
    physbench_to_trial,
    piqa_to_trial,
    write_manifest,
)
from esvlm.utils.logging import echo_run_args, setup_logging

PHYS_REPOS = ("USC-GVL/PhysBench", "USC-PSI-Lab/PhysBench")
PHYS_LABEL_REPOS = ("lmms-lab-eval/PhysBench",)
CLEVRER_Q_URLS = (
    "https://data.csail.mit.edu/clevrer/questions/validation.json",
    "http://data.csail.mit.edu/clevrer/questions/validation.json",
)
CLEVRER_VID_URLS = (
    "https://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip",
    "http://data.csail.mit.edu/clevrer/videos/validation/video_validation.zip",
)
PIQA_ZIP_URLS = (
    "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip",
)
PIQA_VALID_URLS = (
    "https://yonatanbisk.com/piqa/data/valid.jsonl",
    "https://yonatanbisk.com/piqa/data/dev.jsonl",
)
PIQA_LABEL_URLS = (
    "https://yonatanbisk.com/piqa/data/valid-labels.lst",
    "https://yonatanbisk.com/piqa/data/dev-labels.lst",
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    wget = _which("wget")
    if wget:
        subprocess.check_call([wget, "-c", "-O", str(dest), url])
        return
    with urlopen(url) as src, dest.open("wb") as out:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def _download_first(urls: tuple[str, ...], dest: Path) -> None:
    errors: list[str] = []
    for url in urls:
        try:
            _download(url, dest)
            if dest.exists() and dest.stat().st_size > 0:
                return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url}: {exc}")
    raise RuntimeError("download failed:\n" + "\n".join(errors))


def _load_json_or_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl" or (not text.lstrip().startswith("[") and not text.lstrip().startswith("{")):
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    for key in ("data", "items", "videos", "questions"):
        if isinstance(data.get(key), list):
            return data[key]
    return [data]


def _load_table(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    return _load_json_or_jsonl(path)


def _which(name: str) -> str | None:
    from shutil import which

    return which(name)


def _unzip(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        return
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)


def _load_physbench_labels(raw: Path, log) -> list[dict]:
    label_root = raw / "labels"
    label_root.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    last_err: Exception | None = None
    for repo in PHYS_LABEL_REPOS:
        log.info("physbench labels %s -> %s", repo, label_root)
        try:
            snapshot_download(repo, repo_type="dataset", local_dir=str(label_root))
            last_err = None
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            log.warning("physbench labels %s failed: %s", repo, exc)
    if last_err is not None:
        raise last_err
    labeled: list[dict] = []
    for path in (
        *sorted(label_root.rglob("*.parquet")),
        *sorted(label_root.rglob("*.jsonl")),
        *sorted(label_root.rglob("*.json")),
    ):
        if path.name in {".gitattributes"} or "cache" in path.parts:
            continue
        try:
            rows = _load_table(path)
        except Exception:  # noqa: BLE001
            continue
        if rows and any(r.get("answer") not in (None, "") for r in rows[:20]):
            labeled.extend(rows)
            log.info("physbench labels from %s n=%s", path, len(rows))
    if not labeled:
        raise FileNotFoundError(f"no labeled PhysBench rows under {label_root}")
    return labeled


def fetch_physbench(root: Path, log) -> Path:
    raw = root / "physbench" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    questions = next((p for p in raw.rglob("test.json") if "labels" not in p.parts), None)
    if questions is None or not questions.is_file():
        last_err: Exception | None = None
        for repo in PHYS_REPOS:
            log.info("physbench snapshot %s -> %s", repo, raw)
            try:
                snapshot_download(repo, repo_type="dataset", local_dir=str(raw))
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                log.warning("physbench %s failed: %s", repo, exc)
        if last_err is not None:
            raise last_err
        questions = next(raw.rglob("test.json"), raw / "test.json")
    for name in ("image.zip", "video.zip"):
        zpath = next((p for p in raw.rglob(name) if "labels" not in p.parts), raw / name)
        dest = raw / name.replace(".zip", "")
        if zpath.is_file() and not (dest.exists() and any(dest.iterdir())):
            _unzip(zpath, dest)
    if not questions.is_file():
        raise FileNotFoundError(f"physbench test.json missing under {raw}")
    rows = _load_table(questions)
    if not any(r.get("answer") for r in rows[:50]):
        rows = merge_physbench_labels(rows, _load_physbench_labels(raw, log))
    trials = []
    for idx, row in enumerate(rows):
        row = {**row, "idx": row.get("idx", idx)}
        trial = physbench_to_trial(row, media_root=raw)
        if trial is not None:
            trials.append(trial)
    if not trials:
        raise RuntimeError(f"physbench kept 0 / {len(rows)} items after label merge")
    dest = root / "physbench" / "manifest.json"
    write_manifest(dest, "physbench", trials)
    log.info("physbench kept %s / %s text-option folk-physics items", len(trials), len(rows))
    return dest


def fetch_clevrer(root: Path, log) -> Path:
    raw = root / "clevrer" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    qpath = raw / "validation.json"
    vzip = raw / "video_validation.zip"
    log.info("clevrer questions")
    _download_first(CLEVRER_Q_URLS, qpath)
    log.info("clevrer videos")
    _download_first(CLEVRER_VID_URLS, vzip)
    vdir = raw / "videos"
    if vzip.is_file() and not any(vdir.rglob("*.mp4")):
        _unzip(vzip, vdir)
    rows = _load_table(qpath)
    trials: list[dict] = []
    for row in rows:
        trials.extend(clevrer_to_trials(row))
    index = {p.name: p for p in vdir.rglob("*.mp4")} if vdir.exists() else {}
    for trial in trials:
        found = index.get(Path(trial["media"]).name)
        if found is not None:
            trial["media"] = str(found.relative_to(raw))
    dest = root / "clevrer" / "manifest.json"
    write_manifest(dest, "clevrer", trials)
    log.info("clevrer kept %s MCQ items", len(trials))
    return dest


def _first_existing(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        found = next(root.rglob(name), None)
        if found is not None and found.is_file():
            return found
    return None


def _read_labels(path: Path | None, n: int) -> list[int]:
    if path is None or not path.is_file():
        return []
    out: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == "":
            continue
        out.append(int(line.strip()))
    if out and len(out) != n:
        raise ValueError(f"label count {len(out)} != item count {n} in {path}")
    return out


def fetch_piqa(root: Path, log) -> Path:
    raw = root / "piqa" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    zpath = raw / "physicaliqa-train-dev.zip"
    log.info("piqa zip %s", PIQA_ZIP_URLS[0])
    try:
        _download_first(PIQA_ZIP_URLS, zpath)
        if zpath.is_file() and not _first_existing(raw, ("dev.jsonl", "valid.jsonl")):
            _unzip(zpath, raw)
    except Exception as exc:  # noqa: BLE001
        log.warning("piqa zip failed: %s", exc)
        _download_first(PIQA_VALID_URLS, raw / "valid.jsonl")
        _download_first(PIQA_LABEL_URLS, raw / "valid-labels.lst")
    qpath = _first_existing(raw, ("dev.jsonl", "valid.jsonl"))
    lpath = _first_existing(raw, ("dev-labels.lst", "valid-labels.lst"))
    if qpath is None:
        raise FileNotFoundError(f"piqa jsonl missing under {raw}")
    rows = _load_table(qpath)
    labels = _read_labels(lpath, len(rows))
    trials: list[dict] = []
    for idx, row in enumerate(rows):
        if "label" not in row and idx < len(labels):
            row = {**row, "label": labels[idx]}
        trial = piqa_to_trial(row, split="valid", idx=idx)
        if trial is not None:
            trials.append(trial)
    if not trials:
        raise RuntimeError(f"piqa kept 0 items from {qpath}")
    dest = root / "piqa" / "manifest.json"
    write_manifest(dest, "piqa", trials)
    log.info("piqa kept %s items from %s", len(trials), qpath)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch public S-domain datasets")
    parser.add_argument("--dataset", default=os.environ.get("DATASET", "all"))
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ["ESVLM_PUBLIC_DIR"]) if os.environ.get("ESVLM_PUBLIC_DIR") else None,
    )
    args = parser.parse_args(argv)
    if args.root is None:
        raise SystemExit("ESVLM_PUBLIC_DIR is unset")
    log = setup_logging()
    echo_run_args(vars(args), log)
    wanted = args.dataset.lower()
    if wanted in {"physbench", "all"}:
        fetch_physbench(args.root, log)
    if wanted in {"clevrer", "all"}:
        fetch_clevrer(args.root, log)
    if wanted in {"piqa", "all"}:
        fetch_piqa(args.root, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
