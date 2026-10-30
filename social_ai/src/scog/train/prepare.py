"""Build A2 JSONL on the cluster and freeze the MMLU id list.

Prints counts only. Refuses to run unless prereg/FROZEN lists A2.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scog.data.channel4 import clean_channel4, load_channel4
from scog.data.splits import (
    SPLIT_SEED,
    assert_split_allowed,
    assign_splits,
    hash_ids,
    read_lock,
    stratum_labels,
)
from scog.train.examples import (
    build_examples,
    dev_id_set,
    held_out_mask,
    shuffled_id_set,
)

MMLU_N = 200
MMLU_SEED = 20261029


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def exploratory_hash_ok(ids: list[str]) -> str:
    digest = hash_ids(ids)
    expected = read_lock((_project_root() / "configs" / "splits.lock").read_text(encoding="utf-8"))
    if digest != expected["channel4.exploratory"]:
        raise RuntimeError("exploratory split hash does not match configs/splits.lock")
    return digest


def quota_headroom(min_gb: float = 8, min_files: int = 20000) -> None:
    """Stop if the project volume cannot take another write of this size."""
    text = subprocess.check_output(["quota", "-s"], text=True)
    found = False
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            used_gb = float(parts[1])
            quota_gb = float(parts[2])
            files = int(parts[5])
            file_quota = int(parts[6])
        except ValueError:
            continue
        if file_quota != 1048576 or quota_gb < 100:
            continue
        found = True
        free_gb = quota_gb - used_gb
        free_files = file_quota - files
        print(f"quota_free_gb\t{free_gb:.1f}\tquota_free_files\t{free_files}")
        if free_gb < min_gb or free_files < min_files:
            raise RuntimeError(
                f"project storage headroom is {free_gb:.1f} GB and {free_files} files"
            )
    if not found:
        raise RuntimeError("project quota line not found")


def assign_channel4(frame: pd.DataFrame) -> pd.DataFrame:
    ids = frame["userid"].map(lambda value: str(int(value)))
    strata = stratum_labels(frame["primary_group"], frame["sex"], frame["age"])
    return assign_splits(ids, strata, seed=SPLIT_SEED, with_a2=True)


def _ids_for(assigned: pd.DataFrame, split: str) -> list[str]:
    return assigned.loc[assigned["split"].eq(split), "id"].astype(str).tolist()


def plan_splits(frame: pd.DataFrame, *, check_hash: bool = True) -> dict:
    """Id lists for the preregistered pools. Confirmatory is included; the caller must be frozen."""
    assert_split_allowed("confirmatory", study="A2")
    assigned = assign_channel4(frame)
    exploratory = _ids_for(assigned, "exploratory")
    if check_hash:
        exploratory_hash_ok(exploratory)
    held = set(frame.loc[held_out_mask(frame), "userid"].map(lambda value: str(int(value))))
    explore_pool = [item for item in exploratory if item not in held]
    dev = dev_id_set(explore_pool)
    explore_train = [item for item in explore_pool if item not in dev]
    confirmatory = [item for item in _ids_for(assigned, "confirmatory") if item not in held]
    if check_hash and (len(explore_train) < 10 or len(dev) < 10 or len(confirmatory) < 10):
        raise RuntimeError("a training pool is below 10 people")
    return {
        "explore_train": explore_train,
        "dev": sorted(dev),
        "confirmatory": confirmatory,
        "n_held_out_exploratory": sum(1 for item in exploratory if item in held),
        "exploratory_hash": hash_ids(exploratory),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write an empty file {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    temporary.replace(path)


def sample_mmlu(pairs: list[tuple[str, int]], n: int = MMLU_N, seed: int = MMLU_SEED) -> list[dict]:
    if len(pairs) < n:
        raise RuntimeError(f"MMLU pool has {len(pairs)} questions")
    draw = np.random.default_rng(seed).permutation(len(pairs))[:n]
    chosen = [pairs[int(index)] for index in draw]
    return [{"subject": subject, "index": int(index)} for subject, index in chosen]


def hash_mmlu(rows: list[dict]) -> str:
    payload = "".join(f"{row['subject']}\t{row['index']}\n" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def mmlu_subject_names(configs: list[str], splits_for) -> list[str]:
    """Keep subject configs that publish a test split. Skip the aggregate and train-only sets."""
    chosen = []
    for name in configs:
        if name == "all":
            continue
        if "test" not in splits_for(name):
            continue
        chosen.append(name)
    if not chosen:
        raise RuntimeError("cais/mmlu returned no subject with a test split")
    return chosen


def load_mmlu_pairs() -> list[tuple[str, int]]:
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

    names = mmlu_subject_names(
        get_dataset_config_names("cais/mmlu"),
        lambda subject: get_dataset_split_names("cais/mmlu", subject),
    )
    pairs = []
    for subject in names:
        dataset = load_dataset("cais/mmlu", subject, split="test")
        if len(dataset) == 0:
            raise RuntimeError(f"MMLU subject {subject} has zero test rows")
        pairs.extend((subject, index) for index in range(len(dataset)))
    if len(pairs) < MMLU_N:
        raise RuntimeError("MMLU test pool is smaller than 200")
    return pairs


def build_pools(frame: pd.DataFrame, *, check_hash: bool = True) -> dict[str, list[dict] | dict]:
    pools = plan_splits(frame, check_hash=check_hash)
    pieces = {}
    skipped = 0
    for name, key in (
        ("exploratory_train", "explore_train"),
        ("exploratory_dev", "dev"),
        ("confirmatory_train", "confirmatory"),
    ):
        id_list = pools[key]
        shuffled = shuffled_id_set(id_list) if name != "exploratory_dev" else set()
        rows, counts = build_examples(frame, id_list, shuffled=shuffled)
        pieces[name] = rows
        skipped += counts["n_skipped_incomplete"]
    manifest = {
        "n_exploratory_train": len(pieces["exploratory_train"]),
        "n_exploratory_dev": len(pieces["exploratory_dev"]),
        "n_confirmatory_train": len(pieces["confirmatory_train"]),
        "n_skipped_incomplete": skipped,
        "n_held_out_exploratory": pools["n_held_out_exploratory"],
        "exploratory_hash": pools["exploratory_hash"],
    }
    return {"rows": pieces, "manifest": manifest}


def ensure_model(dest: Path, model_id: str) -> None:
    if (dest / "config.json").is_file():
        print("model_present")
        return
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(model_id, local_dir=str(dest))
    if not (dest / "config.json").is_file():
        raise RuntimeError("model download did not write config.json")
    print("model_downloaded")


def main() -> None:
    csv_path = os.environ.get("C4_CSV", "")
    out_root = os.environ.get("A2_ROOT", "")
    model_id = os.environ.get("A2_MODEL_ID", "Qwen/Qwen3-8B-Base")
    model_dir = os.environ.get("A2_MODEL", "")
    if not csv_path or not out_root or not model_dir:
        raise RuntimeError("C4_CSV, A2_ROOT and A2_MODEL must be set")
    quota_headroom(30, 80000)
    ensure_model(Path(model_dir), model_id)
    destination = Path(out_root) / "data"
    manifest_path = destination / "manifest.json"
    if manifest_path.is_file():
        print("data_manifest_exists")
    else:
        frame, _audit = clean_channel4(load_channel4(csv_path))
        if len(frame) == 0:
            raise RuntimeError("Channel 4 cleaning returned zero rows")
        built = build_pools(frame)
        for name, rows in built["rows"].items():
            write_jsonl(destination / f"{name}.jsonl", rows)
        write_jsonl_manifest(manifest_path, built["manifest"])
        for key, value in built["manifest"].items():
            print(f"{key}\t{value}")
    mmlu_path = Path(out_root) / "mmlu_ids.json"
    if not mmlu_path.is_file():
        chosen = sample_mmlu(load_mmlu_pairs())
        digest = hash_mmlu(chosen)
        payload = {"sha256": digest, "n": len(chosen), "rows": chosen}
        mmlu_path.write_text(json.dumps(payload), encoding="utf-8")
        print(f"mmlu_sha256\t{digest}\tmmlu_n\t{len(chosen)}")
    else:
        existing = json.loads(mmlu_path.read_text(encoding="utf-8"))
        print(f"mmlu_sha256\t{existing['sha256']}\tmmlu_n\t{existing['n']}")


def write_jsonl_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"prepare failed: {exc}", file=sys.stderr)
        raise
