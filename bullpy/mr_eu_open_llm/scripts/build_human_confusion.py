#!/usr/bin/env python3
"""Build human confusion geometry for study3 EU mech-interp (243 trials, 6AFC).

Inputs (JSON only — no OneDrive access at runtime):
  - data/eu_emotions_full_manifest.json  trial order + labels
  - data/eu_emotions_human_entropy.json  per-item human 6-way distributions

The human distributions were built on Mac from the O'Reilly validation workbook
(see scripts/human_entropy.py); stimulus *videos* live under OneDrive EU_Emotions
on Mac or ~/rds/hpc-work/study3/data/eu_emotions on HPC and are not read here.

Outputs:
  - human_confusion_rdm.npy           (n_trials x n_trials) JS distance between response DMs
  - human_confusion_meta.json         trial_ids, per-item confusability scalars
  - human_confused_pairs.json         aggregated label-pair confusion + nearest-neighbour items

Usage (Mac or HPC, after human_entropy.json exists):

  python -m scripts.build_human_confusion \\
    --manifest data/eu_emotions_full_manifest.json \\
    --human data/eu_emotions_human_entropy.json \\
    --out-dir data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from scripts.human_entropy import shannon_entropy


def _normalize(probs: Sequence[float], *, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    p = np.maximum(p, 0.0)
    s = p.sum()
    if s <= eps:
        return np.full_like(p, 1.0 / len(p))
    return p / s


def js_divergence(p: Sequence[float], q: Sequence[float], *, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence between two discrete distributions (nats)."""
    p_arr = _normalize(p, eps=eps)
    q_arr = _normalize(q, eps=eps)
    m = 0.5 * (p_arr + q_arr)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > eps
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * (_kl(p_arr, m) + _kl(q_arr, m))


def distribution_vector(
    entry: Dict[str, Any],
    *,
    use_forced: bool = False,
) -> Tuple[List[str], np.ndarray]:
    """Return (option_labels, probability_vector) for one human item."""
    options = list(entry["human_options"])
    dist = entry["human_response_distribution"]
    if use_forced:
        named = options[:-1]
        probs = [float(dist.get(o, 0.0)) for o in named]
        total = sum(probs)
        if total <= 0:
            probs = [1.0 / len(named)] * len(named)
        else:
            probs = [p / total for p in probs]
        return named, np.asarray(probs, dtype=np.float64)

    probs = [float(dist.get(o, 0.0)) for o in options]
    return options, _normalize(probs)


def build_rdm(vectors: Sequence[np.ndarray]) -> np.ndarray:
    n = len(vectors)
    rdm = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = js_divergence(vectors[i], vectors[j])
            rdm[i, j] = d
            rdm[j, i] = d
    return rdm


def confusability_scalars(entry: Dict[str, Any]) -> Dict[str, float]:
    options, probs = distribution_vector(entry, use_forced=False)
    target = options[0]
    p_target = float(entry.get("p_target", probs[0]))
    # Top foil among named emotions (exclude abstain).
    named_probs = [(o, float(entry["human_response_distribution"].get(o, 0.0))) for o in options[:-1]]
    named_probs = [(o, p) for o, p in named_probs if o != target]
    top_foil, top_foil_mass = max(named_probs, key=lambda x: x[1]) if named_probs else ("", 0.0)
    return {
        "human_entropy": float(entry["human_entropy"]),
        "human_entropy_forced": float(entry["human_entropy_forced"]),
        "p_target": p_target,
        "confusability_1_minus_p_target": float(1.0 - p_target),
        "top_foil_label": top_foil,
        "top_foil_mass": float(top_foil_mass),
        "manifest_label": str(entry.get("manifest_label", "")),
        "human_target_label": str(entry.get("human_target_label", target)),
    }


def aggregate_label_pair_confusion(
    trial_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Sum target->foil mass across items; used for cross-pair reuse candidate lists."""
    pair_mass: Dict[Tuple[str, str], float] = {}
    pair_items: Dict[Tuple[str, str], List[str]] = {}
    for row in trial_rows:
        target = row["human_target_label"]
        foil = row["top_foil_label"]
        if not target or not foil or target == foil:
            continue
        key = tuple(sorted((target, foil)))
        mass = float(row["top_foil_mass"])
        pair_mass[key] = pair_mass.get(key, 0.0) + mass
        pair_items.setdefault(key, []).append(row["trial_id"])
    ranked = sorted(pair_mass.items(), key=lambda kv: kv[1], reverse=True)
    out: List[Dict[str, Any]] = []
    for (a, b), total_mass in ranked:
        out.append(
            {
                "label_a": a,
                "label_b": b,
                "aggregate_foil_mass": total_mass,
                "n_items": len(pair_items[(a, b)]),
                "example_trial_ids": pair_items[(a, b)][:5],
            }
        )
    return out


def nearest_item_pairs(
    trial_ids: Sequence[str],
    rdm: np.ndarray,
    *,
    top_k: int = 30,
) -> List[Dict[str, Any]]:
    n = len(trial_ids)
    pairs: List[Tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((float(rdm[i, j]), i, j))
    pairs.sort(key=lambda x: x[0])
    out: List[Dict[str, Any]] = []
    for dist, i, j in pairs[:top_k]:
        out.append(
            {
                "trial_id_a": trial_ids[i],
                "trial_id_b": trial_ids[j],
                "js_distance": dist,
            }
        )
    return out


def build_confusion_artifacts(
    manifest: Path,
    human: Path,
    *,
    use_forced: bool = False,
) -> Dict[str, Any]:
    manifest_obj = json.loads(manifest.read_text(encoding="utf-8"))
    human_obj = json.loads(human.read_text(encoding="utf-8"))
    lookup = human_obj.get("trials", {})

    trial_ids: List[str] = []
    vectors: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    for trial in manifest_obj["trials"]:
        tid = str(trial["trial_id"])
        entry = lookup.get(tid)
        if entry is None:
            missing.append(tid)
            continue
        _opts, vec = distribution_vector(entry, use_forced=use_forced)
        trial_ids.append(tid)
        vectors.append(vec)
        scalars = confusability_scalars(entry)
        rows.append({"trial_id": tid, **scalars})

    if missing:
        raise ValueError(f"{len(missing)} manifest trials missing from human lookup (first: {missing[0]})")

    rdm = build_rdm(vectors)
    label_pairs = aggregate_label_pair_confusion(rows)
    item_pairs = nearest_item_pairs(trial_ids, rdm)

    return {
        "manifest": manifest.name,
        "human_lookup": human.name,
        "n_trials": len(trial_ids),
        "distribution_variant": "human_entropy_forced" if use_forced else "human_entropy_6way",
        "distance_metric": "jensen_shannon_nats",
        "trial_ids": trial_ids,
        "per_item": rows,
        "label_pair_confusion": label_pairs,
        "nearest_item_pairs": item_pairs,
        "rdm_shape": list(rdm.shape),
        "rdm_upper_tri_mean": float(np.mean(rdm[np.triu_indices(len(trial_ids), k=1)])),
        "notes": [
            "Human distributions from O'Reilly workbook via eu_emotions_human_entropy.json.",
            "Stimulus videos are not read by this script.",
            "Use label_pair_confusion for cross-pair reuse candidate pairs.",
            "Use nearest_item_pairs for item-level RSA alignment checks.",
        ],
        "_rdm": rdm,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("data/eu_emotions_full_manifest.json"))
    ap.add_argument("--human", type=Path, default=Path("data/eu_emotions_human_entropy.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument(
        "--forced",
        action="store_true",
        help="Also write 5-way (no abstain) RDM as human_confusion_rdm_forced.npy",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    primary = build_confusion_artifacts(args.manifest, args.human, use_forced=False)

    rdm_path = args.out_dir / "human_confusion_rdm.npy"
    np.save(rdm_path, primary["_rdm"])

    meta = {k: v for k, v in primary.items() if k != "_rdm"}
    meta_path = args.out_dir / "human_confusion_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    pairs_path = args.out_dir / "human_confused_pairs.json"
    pairs_payload = {
        "n_trials": meta["n_trials"],
        "label_pair_confusion": meta["label_pair_confusion"],
        "nearest_item_pairs": meta["nearest_item_pairs"],
    }
    pairs_path.write_text(json.dumps(pairs_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.forced:
        forced = build_confusion_artifacts(args.manifest, args.human, use_forced=True)
        np.save(args.out_dir / "human_confusion_rdm_forced.npy", forced["_rdm"])

    print(f"n_trials     : {meta['n_trials']}")
    print(f"rdm          : {rdm_path} shape={meta['rdm_shape']}")
    print(f"meta         : {meta_path}")
    print(f"pairs        : {pairs_path}")
    if meta["label_pair_confusion"]:
        top = meta["label_pair_confusion"][0]
        print(
            f"top label pair: {top['label_a']} <-> {top['label_b']} "
            f"(aggregate foil mass={top['aggregate_foil_mass']:.3f}, n_items={top['n_items']})"
        )


if __name__ == "__main__":
    main()
