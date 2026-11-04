"""Option-logprob scoring interface and option-order robustness helpers.

Does not modify ``scripts/evaluate.py``. Option-order evals are produced by
writing a shuffled copy of the human option lookup (evaluate.py reads
``human_options`` in listed order) and pointing ``--human_options`` at it.

GPU scoring is a caller-supplied first-token logprob function so this module
stays off the model stack until a CSD3 job is free.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from paper_v2.metrics import shannon_entropy

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
N_ORDERS = 3


def _stable_seed(seed: int, trial_id: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}|{trial_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def shuffle_options(options: Sequence[str], *, seed: int, trial_id: str) -> List[str]:
    """Deterministic permutation of option *order*; labels themselves are unchanged."""
    rng = np.random.default_rng(_stable_seed(seed, trial_id))
    out = list(options)
    rng.shuffle(out)
    return out


def write_shuffled_human_lookup(
    src: Path,
    dest: Path,
    *,
    seed: int,
) -> Dict[str, int]:
    """Copy human entropy JSON with per-trial shuffled ``human_options``."""
    obj = json.loads(src.read_text(encoding="utf-8"))
    n = 0
    for trial_id, rec in obj["trials"].items():
        rec["human_options"] = shuffle_options(rec["human_options"], seed=seed, trial_id=trial_id)
        rec["option_order_seed"] = int(seed)
        n += 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    return {"n_trials": n, "seed": int(seed), "path": str(dest)}


ABSTAIN_LABEL = "None of the above"


def rotate_options(options: Sequence[str], rotation: int) -> List[str]:
    """Cyclically rotate the named labels; any abstain option stays last.

    ``human_options`` lists the target first on every EU-Emotion item. Running every
    rotation of the named labels puts each target in each named position exactly once,
    so position cannot be confounded with correctness across a full rotation set.
    """
    named = [o for o in options if o != ABSTAIN_LABEL]
    tail = [o for o in options if o == ABSTAIN_LABEL]
    r = int(rotation) % len(named)
    return named[r:] + named[:r] + tail


def write_rotated_human_lookup(src: Path, dest: Path, *, rotation: int) -> Dict[str, int]:
    """Copy human entropy JSON with each trial's named options rotated."""
    obj = json.loads(src.read_text(encoding="utf-8"))
    for rec in obj["trials"].values():
        rec["human_options"] = rotate_options(rec["human_options"], rotation)
        rec["option_rotation"] = int(rotation)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    return {"n_trials": len(obj["trials"]), "rotation": int(rotation), "path": str(dest)}


def vision_mismatch_partners(
    trial_ids: Sequence[str],
    labels: Sequence[str],
    actors: Sequence[Optional[str]],
    *,
    seed: int,
) -> Dict[str, str]:
    """Map each trial to a partner whose video will replace its own.

    The partner has a different target label and, where one exists, a different actor.
    The map is one-to-one where possible: every video is reused as evenly as it can be,
    so the mismatch condition keeps the marginal distribution of faces. Deterministic.
    """
    ids = list(trial_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("trial_ids must be unique")
    label_of = dict(zip(ids, labels))
    actor_of = dict(zip(ids, actors))
    use_count: Dict[str, int] = {t: 0 for t in ids}
    out: Dict[str, str] = {}
    order = sorted(ids, key=lambda t: _stable_seed(seed, t))
    for tid in order:
        pool = [c for c in ids if c != tid and label_of[c] != label_of[tid]]
        if not pool:
            raise ValueError(f"No mismatch partner for {tid!r}")
        best = [c for c in pool if actor_of[c] != actor_of[tid]] or pool
        lowest = min(use_count[c] for c in best)
        tied = sorted(
            (c for c in best if use_count[c] == lowest),
            key=lambda c: _stable_seed(seed, f"{tid}|{c}"),
        )
        out[tid] = tied[0]
        use_count[tied[0]] += 1
    return out


def option_logprob_scores(
    score_fn: Callable[[str], float],
    options: Sequence[str],
) -> Dict[str, float]:
    """Score each option via ``score_fn(option_label) -> logprob``.

    ``score_fn`` is provided by the GPU runner (first-token or sequence logprob
    of the option string given the frozen prompt prefix).
    """
    logs = {opt: float(score_fn(opt)) for opt in options}
    m = max(logs.values())
    exps = {k: float(np.exp(v - m)) for k, v in logs.items()}
    z = sum(exps.values())
    probs = {k: v / z for k, v in exps.items()}
    return {
        "logprobs": logs,
        "probs": probs,
        "argmax": max(probs, key=probs.get),
        "entropy": shannon_entropy(list(probs.values())),
    }


def plan_option_order_jobs(
    *,
    seeds: Sequence[int] = (42, 43, 44),
    models: Sequence[str] = ("qwen3vl", "gemma4", "molmo2"),
) -> List[Dict[str, object]]:
    """≥3 option orders per model. Submit only when GPU queue has room."""
    src = ROOT / "data" / "eu_emotions_human_entropy.json"
    jobs = []
    for seed in seeds:
        dest = ROOT / "data" / f"eu_emotions_human_entropy_order{seed}.json"
        meta = write_shuffled_human_lookup(src, dest, seed=int(seed))
        for model in models:
            jobs.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "human_options": meta["path"],
                    "out": str(
                        ROOT
                        / "results/baseline/eu_emotions"
                        / model
                        / f"eval_v2_eu_emotions_{model}_video_only_seed42_order{seed}.json"
                    ),
                }
            )
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write_lookups", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument(
        "--write_rotation_lookups",
        action="store_true",
        help="Write data/eu_emotions_human_entropy_rot{r}.json for r = 0..4.",
    )
    args = ap.parse_args()
    if args.write_rotation_lookups:
        src = ROOT / "data" / "eu_emotions_human_entropy.json"
        for r in range(5):
            dest = ROOT / "data" / f"eu_emotions_human_entropy_rot{r}.json"
            print(write_rotated_human_lookup(src, dest, rotation=r))
    elif args.write_lookups:
        jobs = plan_option_order_jobs(seeds=args.seeds)
        summary = ROOT / "research/reproduction/option_order_plan.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps({"n_jobs": len(jobs), "jobs": jobs}, indent=2) + "\n")
        print(f"wrote {len(jobs)} planned evals -> {summary}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
