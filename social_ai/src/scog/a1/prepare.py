"""Build the exploratory prompt manifest on the cluster.

Prints counts only. Writes the manifest under the run directory. Refuses
the confirmatory split, and refuses a Channel 4 split whose hash does not
match configs/splits.lock.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from scog.a1.prompts import user_message
from scog.a1.sample import draw_group
from scog.data.channel4 import clean_channel4, load_channel4
from scog.data.splits import assert_split_allowed, assign_splits, hash_ids, read_lock, stratum_labels

SEED = 20261029


def _split_frame(cleaned: pd.DataFrame, lock_path: Path) -> pd.DataFrame:
    splits = assign_splits(
        cleaned["userid"].astype("int64").astype(str),
        stratum_labels(cleaned["primary_group"], cleaned["sex"], cleaned["age"]),
        with_a2=True,
    )
    lock = read_lock(lock_path.read_text(encoding="utf-8"))
    got = hash_ids(splits.loc[splits["split"].eq("exploratory"), "id"].tolist())
    expected = lock["channel4.exploratory"]
    if got != expected:
        raise RuntimeError(
            "channel4 exploratory hash does not match the lock "
            f"({got} != {expected})"
        )
    return splits


def build_manifest(
    cleaned: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    n_per_group: int,
    n_condition1: int,
    n_condition4: int,
    seed: int = SEED,
) -> tuple[list[dict], dict]:
    assert_split_allowed("exploratory")
    explore_ids = set(splits.loc[splits["split"].eq("exploratory"), "id"].astype(str))
    work = cleaned.copy()
    work["userid"] = work["userid"].astype("int64").astype(str)
    work = work.loc[work["userid"].isin(explore_ids)].copy()
    if len(work) == 0:
        raise RuntimeError("exploratory split: zero rows")
    records: list[dict] = []
    summary = {"n_exploratory_rows": int(len(work))}
    for group in ("autistic", "no_diagnosis"):
        drawn = draw_group(work, group, n_per_group, seed)
        summary[f"n_drawn_{group}"] = int(len(drawn))
        for persona in drawn.to_dict(orient="records"):
            for condition in (2, 3):
                records.append(
                    {
                        "persona_id": persona["persona_id"],
                        "group": group,
                        "condition": condition,
                        "seed": seed,
                        "prompt": user_message(persona, condition),
                    }
                )
    for i in range(n_condition1):
        records.append(
            {
                "persona_id": f"self-{i}",
                "group": "none",
                "condition": 1,
                "seed": seed,
                "prompt": user_message(None, 1),
            }
        )
    for group in ("autistic", "no_diagnosis"):
        persona = {"group": group}
        for i in range(n_condition4):
            records.append(
                {
                    "persona_id": f"label-{group}-{i}",
                    "group": group,
                    "condition": 4,
                    "seed": seed,
                    "prompt": user_message(persona, 4),
                }
            )
    summary["n_condition1"] = n_condition1
    summary["n_condition4_per_label"] = n_condition4
    summary["n_records"] = len(records)
    return records, summary


def main() -> None:
    data = Path(os.environ["SCOG_C4_PATH"])
    lock = Path(os.environ["SCOG_SPLIT_LOCK"])
    out = Path(os.environ["SCOG_MANIFEST"])
    n_per_group = int(os.environ.get("SCOG_N_PER_GROUP", "200"))
    n_condition1 = int(os.environ.get("SCOG_N_CONDITION1", "20"))
    n_condition4 = int(os.environ.get("SCOG_N_CONDITION4", "20"))
    cleaned, audit = clean_channel4(load_channel4(data))
    if audit["n_after"] == 0:
        raise RuntimeError("Channel 4 cleaning left zero rows")
    splits = _split_frame(cleaned, lock)
    records, summary = build_manifest(
        cleaned,
        splits,
        n_per_group=n_per_group,
        n_condition1=n_condition1,
        n_condition4=n_condition4,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
