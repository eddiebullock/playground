"""CPU preflight for papers 1 and 2. No GPU. No MMToM video zip."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

from esvlm.cli.score import load_task_items
from esvlm.utils.logging import echo_run_args, setup_logging

EXPECTED_NEXTQA_PAIRS = 461
EXPECTED_MMTOM = 600


def _word_stats(texts: list[str]) -> dict[str, float | int]:
    counts = [len(t.split()) for t in texts]
    if not counts:
        return {"n": 0}
    return {
        "n": len(counts),
        "mean": round(float(mean(counts)), 2),
        "median": float(median(counts)),
        "min": int(min(counts)),
        "max": int(max(counts)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load public Paper 1/2 items on CPU")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/aggregate/public_item_lock.json"),
    )
    args = parser.parse_args(argv)
    log = setup_logging()
    echo_run_args(vars(args), log)

    social, nextqa_man = load_task_items("nextqa_social", None)
    physical, _ = load_task_items("nextqa_physical", None)
    mmtom, mmtom_man = load_task_items("mmtom", None)
    piqa, piqa_man = load_task_items("piqa", None)

    if len(social) != len(physical):
        raise SystemExit(f"nextqa side length mismatch {len(social)} vs {len(physical)}")
    identical_q = sum(
        1
        for a, b in zip(social, physical)
        if a.prompt_text.split("\n\n", 1)[0].strip().lower()
        == b.prompt_text.split("\n\n", 1)[0].strip().lower()
    )
    payload = {
        "nextqa_pairs": len(social),
        "nextqa_videos": len({it.meta.get("video") for it in social}),
        "nextqa_manifest": str(nextqa_man) if nextqa_man else None,
        "physical_types": dict(Counter(str(it.meta.get("question_type")) for it in physical)),
        "cw_words": _word_stats([it.prompt_text.split("\n\n", 1)[0] for it in social]),
        "phys_words": _word_stats([it.prompt_text.split("\n\n", 1)[0] for it in physical]),
        "identical_question_text": identical_q,
        "n_nextqa_missing_media": sum(1 for it in social if it.media_type == "none"),
        "mmtom_n": len(mmtom),
        "mmtom_manifest": str(mmtom_man) if mmtom_man else None,
        "mmtom_question_types": dict(
            sorted(Counter(str(it.meta.get("question_type")) for it in mmtom).items())
        ),
        "mmtom_2way": sum(1 for it in mmtom if len(it.options) == 2),
        "piqa_n": len(piqa),
        "piqa_manifest": str(piqa_man) if piqa_man else None,
        "expected_nextqa_pairs": EXPECTED_NEXTQA_PAIRS,
        "expected_mmtom": EXPECTED_MMTOM,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log.info("public_item_lock %s", json.dumps(payload))
    errors = []
    if len(social) != EXPECTED_NEXTQA_PAIRS:
        errors.append(f"nextqa pairs {len(social)} != {EXPECTED_NEXTQA_PAIRS}")
    if len(mmtom) != EXPECTED_MMTOM:
        errors.append(f"mmtom {len(mmtom)} != {EXPECTED_MMTOM}")
    if errors:
        raise SystemExit("; ".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
