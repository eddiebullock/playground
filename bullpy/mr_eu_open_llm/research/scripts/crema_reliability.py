#!/usr/bin/env python3
"""CREMA-D face-only human reliability and split-half noise ceilings.

Reads the raw responses, applies upstream's 10 s filter, confirms the result equals
tabulatedVotes.csv, then splits raters within each clip. Reliability is reported pooled
and within intended emotion, because pooled values include between-emotion variance.

    python -I research/scripts/crema_reliability.py --data-dir data/external/crema_d
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "research" / "reproduction"
LABELS = list("ADFHNS")
FROM_FILENAME = {"ANG": "A", "DIS": "D", "FEA": "F", "HAP": "H", "NEU": "N", "SAD": "S"}
SEED = 42
N_SPLITS = 2000
N_RATER_SPLITS = 200
FACE = 2


def filtered_responses(data_dir: Path) -> pd.DataFrame:
    """finishedResponses minus trials whose first click exceeded 10 s (upstream rule)."""
    resp = pd.read_csv(
        data_dir / "finishedResponses.csv", index_col=0, dtype={"ttr": str, "respLevel": str, "ans": str},
        low_memory=False,
    )
    emo = pd.read_csv(data_dir / "finishedEmoResponses.csv", index_col=0, dtype={"ttr": str}, low_memory=False)
    slow = emo[pd.to_numeric(emo.ttr, errors="coerce") > 10000]
    slow_keys = set(slow.sessionNums * 1000 + slow.queryType * 100 + slow.questNum)
    keys = resp.sessionNums * 1000 + resp.queryType * 100 + resp.questNum
    return resp[~keys.isin(slow_keys)]


def split_half_r(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def within_group_r(a: np.ndarray, b: np.ndarray, groups: np.ndarray) -> float:
    a, b = a.astype(float).copy(), b.astype(float).copy()
    for g in np.unique(groups):
        idx = groups == g
        a[idx] -= a[idx].mean()
        b[idx] -= b[idx].mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def spearman_brown(r: float) -> float:
    return 2 * r / (1 + r)


def entropy(p: np.ndarray) -> np.ndarray:
    logp = np.log(p, where=p > 0, out=np.zeros_like(p))
    return 0.0 - (p * logp).sum(axis=1)  # 0.0 - x, so unanimous clips give 0.0 not -0.0


def jsd_bits(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (entropy((p + q) / 2) - (entropy(p) + entropy(q)) / 2) / np.log(2)


def krippendorff_alpha(counts: np.ndarray) -> float:
    """Nominal alpha for a clips-by-categories count matrix with unequal raters per clip."""
    counts = counts[counts.sum(1) >= 2].astype(float)
    n_u = counts.sum(1, keepdims=True)
    coincidence = ((counts[:, :, None] * counts[:, None, :]) / (n_u[:, :, None] - 1)).sum(0)
    coincidence -= np.diag((counts / (n_u - 1)).sum(0))
    n_c, total = coincidence.sum(0), coincidence.sum()
    observed = total - np.trace(coincidence)
    expected = (total**2 - (n_c**2).sum()) / (total - 1)
    return float(1 - observed / expected)


def pairwise_agreement(counts: np.ndarray) -> np.ndarray:
    n = counts.sum(1).astype(float)
    return (counts * (counts - 1)).sum(1) / (n * (n - 1))


def leave_one_rater_modal(counts: np.ndarray) -> np.ndarray:
    """Chance one rater matches the mode of the others; ties share credit."""
    out = np.zeros(len(counts))
    for i, row in enumerate(counts):
        score = 0.0
        for k in np.nonzero(row)[0]:
            rest = row.copy()
            rest[k] -= 1
            modal = np.flatnonzero(rest == rest.max())
            score += row[k] * (k in modal) / len(modal)
        out[i] = score / row.sum()
    return out


def stack_votes(face: pd.DataFrame, clips: Sequence[str]) -> Dict[str, np.ndarray]:
    groups = {c: g for c, g in face.groupby("clipName")}
    width = max(len(g) for g in groups.values())
    votes = np.full((len(clips), width), -1, dtype=np.int16)
    raters = np.full((len(clips), width), -1, dtype=np.int32)
    rater_code = {r: i for i, r in enumerate(sorted(face.localid.unique()))}
    label_code = {e: i for i, e in enumerate(LABELS)}
    for i, clip in enumerate(clips):
        g = groups[clip]
        votes[i, : len(g)] = [label_code[v] for v in g.respEmo]
        raters[i, : len(g)] = [rater_code[r] for r in g.localid]
    return {"votes": votes, "raters": raters}


def counts_for(votes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.stack([((votes == k) & mask).sum(1) for k in range(len(LABELS))], axis=1)


def half_mask(votes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Half of each clip's raters; odd clips give the extra rater to a random side."""
    n = (votes >= 0).sum(1)
    noise = rng.random(votes.shape)
    noise[votes < 0] = 2.0
    ranks = noise.argsort(1).argsort(1)
    n_a = n // 2 + (rng.random(len(n)) < 0.5) * (n % 2)
    return ranks < n_a[:, None]


def summarise(rs: List[float]) -> Dict[str, float]:
    r = float(np.mean(rs))
    sb = spearman_brown(r)
    return {
        "half_r": round(r, 4),
        "half_r_sd": round(float(np.std(rs)), 4),
        "spearman_brown": round(sb, 4),
        "ceiling_sqrt_sb": round(float(np.sqrt(max(sb, 0))), 4),
    }


def scalar_reliability(
    ca: np.ndarray, cb: np.ndarray, tgt: np.ndarray, emo: np.ndarray, acc: Dict[str, List[float]]
) -> None:
    keep = (ca.sum(1) > 0) & (cb.sum(1) > 0)
    ca, cb, tgt, emo = ca[keep].astype(float), cb[keep].astype(float), tgt[keep], emo[keep]
    pa, pb = ca / ca.sum(1, keepdims=True), cb / cb.sum(1, keepdims=True)
    rows = np.arange(len(ca))
    for name, a, b in (
        ("p_target", pa[rows, tgt], pb[rows, tgt]),
        ("entropy", entropy(pa), entropy(pb)),
    ):
        acc[f"{name}_pooled"].append(split_half_r(a, b))
        acc[f"{name}_within_emotion"].append(within_group_r(a, b, emo))
    acc["all_six_proportions_pooled"].append(split_half_r(pa.ravel(), pb.ravel()))
    flat = np.repeat(emo, 6) * 6 + np.tile(np.arange(6), len(emo))
    acc["all_six_proportions_within_emotion_by_option"].append(within_group_r(pa.ravel(), pb.ravel(), flat))
    ma = np.stack([ca[emo == e].sum(0) for e in range(6)])
    mb = np.stack([cb[emo == e].sum(0) for e in range(6)])
    ma, mb = ma / ma.sum(1, keepdims=True), mb / mb.sum(1, keepdims=True)
    off = ~np.eye(6, dtype=bool)
    acc["confusion_6x6_off_diagonal"].append(split_half_r(ma[off], mb[off]))


def analyse(
    votes: np.ndarray,
    raters: np.ndarray,
    tgt: np.ndarray,
    emo: np.ndarray,
    subset: np.ndarray,
    *,
    n_splits: int = N_SPLITS,
    n_rater_splits: int = N_RATER_SPLITS,
    seed: int = SEED,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    full: Dict[str, List[float]] = {}
    sub: Dict[str, List[float]] = {}
    for key in (
        "p_target_pooled", "p_target_within_emotion", "entropy_pooled", "entropy_within_emotion",
        "all_six_proportions_pooled", "all_six_proportions_within_emotion_by_option",
        "confusion_6x6_off_diagonal",
    ):  # fmt: skip
        full[key], sub[key] = [], []
    cosine = np.zeros(len(votes))
    jsd = np.zeros(len(votes))
    valid = votes >= 0
    for _ in range(n_splits):
        mask = half_mask(votes, rng)
        ca, cb = counts_for(votes, mask), counts_for(votes, valid & ~mask)
        scalar_reliability(ca, cb, tgt, emo, full)
        if subset.any():
            scalar_reliability(ca[subset], cb[subset], tgt[subset], emo[subset], sub)
        pa, pb = ca / ca.sum(1, keepdims=True), cb / cb.sum(1, keepdims=True)
        cosine += (pa * pb).sum(1) / np.sqrt((pa * pa).sum(1) * (pb * pb).sum(1))
        jsd += jsd_bits(pa, pb)

    disjoint: Dict[str, List[float]] = {k: [] for k in full}
    n_raters = int(raters.max()) + 1
    for _ in range(n_rater_splits):
        side = rng.random(n_raters) < 0.5
        mask = valid & side[np.clip(raters, 0, None)]
        scalar_reliability(counts_for(votes, mask), counts_for(votes, valid & ~mask), tgt, emo, disjoint)

    return {
        "all_clips": {k: summarise(v) for k, v in full.items()},
        "selected_clips": {k: summarise(v) for k, v in sub.items() if v},
        "all_clips_rater_disjoint_halves": {k: summarise(v) for k, v in disjoint.items()},
        "per_clip_cosine": cosine / n_splits,
        "per_clip_jsd_bits": jsd / n_splits,
    }


def describe(x: np.ndarray) -> Dict[str, float]:
    q = np.percentile(x, [10, 50, 90])
    return {
        "mean": round(float(x.mean()), 4),
        "p10": round(float(q[0]), 4),
        "median": round(float(q[1]), 4),
        "p90": round(float(q[2]), 4),
    }


def run(data_dir: Path, selection: Path, out_json: Path, out_csv: Path) -> Mapping[str, object]:
    responses = filtered_responses(data_dir)
    face = responses[responses.queryType == FACE]
    table = pd.read_csv(data_dir / "tabulatedVotes.csv", index_col=0)
    table = table[table.index.astype(str).str[0] == str(FACE)]
    clips = table.fileName.tolist()

    recount = face.groupby(["clipName", "respEmo"]).size().unstack(fill_value=0).reindex(columns=LABELS, fill_value=0)
    recount = recount.reindex(clips).fillna(0).astype(int)
    if not (recount.values == table[LABELS].values).all():
        raise ValueError("recount from raw responses differs from tabulatedVotes.csv")

    arrays = stack_votes(face, clips)
    votes, raters = arrays["votes"], arrays["raters"]
    emo = np.array([LABELS.index(FROM_FILENAME[c.split("_")[2]]) for c in clips])
    selected = set(pd.read_csv(selection).clip_id) if selection.exists() else set()
    subset = np.array([c in selected for c in clips])
    counts = recount.values
    result = analyse(votes, raters, emo, emo, subset)

    props = counts / counts.sum(1, keepdims=True)
    rows = np.arange(len(clips))
    per_clip = pd.DataFrame(
        {
            "clip_id": clips,
            "n_face": counts.sum(1),
            "p_target": props[rows, emo].round(4),
            "entropy_nats": entropy(props.astype(float)).round(4),
            "pairwise_agreement": pairwise_agreement(counts).round(4),
            "leave_one_rater_modal": leave_one_rater_modal(counts).round(4),
            "split_half_cosine": result.pop("per_clip_cosine").round(4),
            "split_half_jsd_bits": result.pop("per_clip_jsd_bits").round(4),
        }
    )
    per_clip.to_csv(out_csv, index=False)

    raw_face = pd.read_csv(data_dir / "finishedResponses.csv", usecols=["localid", "queryType", "respEmo", "dispEmo"])
    rater_acc = raw_face.assign(hit=raw_face.respEmo == raw_face.dispEmo).groupby(["queryType", "localid"]).hit.mean()
    columns = ["n_face", "p_target", "entropy_nats", "pairwise_agreement", "leave_one_rater_modal",
               "split_half_cosine", "split_half_jsd_bits"]  # fmt: skip
    payload = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "n_rater_disjoint_splits": N_RATER_SPLITS,
        "n_clips": len(clips),
        "n_selected": int(subset.sum()),
        "face_votes_after_filter": int(counts.sum()),
        "face_raters": int(face.localid.nunique()),
        "mean_rater_accuracy_unfiltered_by_query_type": {
            str(k): round(float(v), 4) for k, v in rater_acc.groupby("queryType").mean().items()
        },
        "krippendorff_alpha_nominal": {
            "all_clips": round(krippendorff_alpha(counts), 4),
            "selected_clips": round(krippendorff_alpha(counts[subset]), 4) if subset.any() else None,
        },
        "per_clip_all": {c: describe(per_clip[c].values) for c in columns},
        "per_clip_selected": {c: describe(per_clip[c].values[subset]) for c in columns} if subset.any() else None,
        **result,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "external" / "crema_d")
    parser.add_argument("--selection", type=Path, default=OUT_DIR / "crema_d_selection.csv")
    parser.add_argument("--out-json", type=Path, default=OUT_DIR / "crema_d_reliability.json")
    parser.add_argument("--out-csv", type=Path, default=OUT_DIR / "crema_d_clip_reliability.csv")
    args = parser.parse_args(list(argv) or None)
    print(json.dumps(run(args.data_dir, args.selection, args.out_json, args.out_csv), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
