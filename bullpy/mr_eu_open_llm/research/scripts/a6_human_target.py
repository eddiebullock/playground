#!/usr/bin/env python3
"""The human target of draft amendment A6, from public CREMA-D face-only votes. No model enters.

Writes research/reproduction/a6_human_target.json, whose sha256 is human_target_sha256 in the
design block. So that file carries no design_sha256. It holds the vote-share table, pair
excess X with actor-bootstrap intervals, the pair rule's result, its stability, how far each
target confusion is clips that read as the other label, and a split-half rank ceiling.

Needs the gitignored raw tables in data/external/crema_d, and stops without them, so that the
hashed file never depends on what happens to be on disk.

    PYTHONPATH=.:src python research/scripts/a6_human_target.py

--cue_frames runs the human cue check after the AU step on the cluster: within Anger and within
Disgust clips, the human partner share on detector AU9 and AU10 at the apex. It is reported, gates
nothing, and writes its own file, so the human target hash does not depend on it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from scripts import a6_analyse as an  # noqa: E402
from scripts import a6_design as d  # noqa: E402
from scripts import a6_run as ar  # noqa: E402
from scripts.analyse_stimulus_interventions import actor_interval  # noqa: E402

D = d.DESIGN
OPTIONS = an.OPTIONS
SEED = int(D["seed"])
DRAWS = int(D["bootstrap_draws"])
NONE = "No Emotion"
N_SPLITS = int(D["human_split_halves"])
CUE_COLUMNS = {"AU09": ("lf_AU09", "of_AU09"), "AU10": ("lfocc_AU10", "ge_AU10")}


def shares(votes: np.ndarray, pseudo: float) -> np.ndarray:
    counts = votes.astype(np.float64) + pseudo
    with np.errstate(invalid="ignore", divide="ignore"):
        return counts / counts.sum(1, keepdims=True)


def holder(ids: Sequence[str], actors: Sequence[str], labels: Sequence[str]) -> an.Cells:
    return an.Cells([{"trial_id": t, "actor": a, "label": l} for t, a, l in zip(ids, actors, labels)], [])


def pair_rows(cells: an.Cells, p: np.ndarray) -> List[Dict[str, Any]]:
    rows = []
    for x, y in combinations(range(len(OPTIONS)), 2):
        forward, back = an.directed_excess(cells, p, x, y), an.directed_excess(cells, p, y, x)
        values = (forward + back) / 2.0
        ci = actor_interval(values, D["ci_decision"], draws=DRAWS, seed=SEED)
        rows.append({
            "pair": [OPTIONS[x], OPTIONS[y]],
            "X": float(np.nanmean(values)),
            "ci95": [ci[0], ci[1]],
            "actor_sd": float(np.nanstd(values, ddof=1)),
            "directed": {f"{OPTIONS[x]}>{OPTIONS[y]}": float(np.nanmean(forward)), f"{OPTIONS[y]}>{OPTIONS[x]}": float(np.nanmean(back))},
        })
    return sorted(rows, key=lambda r: -r["X"])


def pair_rule(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The rule in the draft, on rows ranked by X. No model output enters."""
    emotions = [r["pair"] for r in rows if NONE not in r["pair"]]
    with_none = [r["pair"] for r in rows if NONE in r["pair"]]
    return {"target_pair": emotions[0], "shared_family": emotions[:3] + with_none[:1]}


def directed_table(cells: an.Cells, p: np.ndarray) -> Dict[str, Dict[str, Any]]:
    out = {}
    for x in range(len(OPTIONS)):
        for y in range(len(OPTIONS)):
            if x != y:
                values = an.directed_excess(cells, p, x, y)
                out[f"{OPTIONS[x]}>{OPTIONS[y]}"] = {"X": float(np.nanmean(values)), "actors_above_zero": int(np.nansum(values > 0))}
    return out


def vote_table(votes: np.ndarray, emo: np.ndarray) -> Dict[str, List[float]]:
    p = votes / votes.sum(1, keepdims=True)
    return {OPTIONS[e]: p[emo == e].mean(0).tolist() for e in range(len(OPTIONS))}


def read_as_other(votes: np.ndarray, emo: np.ndarray, a: int, b: int) -> Dict[str, Any]:
    """Is the confusion on many clips, or on a few clips that simply read as the other label?"""
    clips = votes[emo == a]
    partner = clips[:, b].astype(np.float64)
    top = clips.max(1)
    ordered = np.sort(partner)[::-1]
    n = len(ordered)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(np.sort(partner)) / (n * partner.sum())) if partner.sum() else None
    return {
        "n_clips": int(n),
        "share_partner_ties_or_tops_mode": float((clips[:, b] == top).mean()),
        "share_partner_strict_mode": float(((clips[:, b] == top) & ((clips == top[:, None]).sum(1) == 1)).mean()),
        "share_any_partner_vote": float((partner > 0).mean()),
        "share_of_partner_votes_in_top_decile_of_clips": float(ordered[: max(1, n // 10)].sum() / partner.sum()),
        "gini_of_partner_votes": gini,
    }


def split_half(data_dir: Path, ids: Sequence[str], cells: an.Cells, pseudo: float) -> Dict[str, Any]:
    """Ceiling for the 30-cell rank correlation: raters split within clip, 200 times."""
    import crema_reliability as cr

    face = cr.filtered_responses(data_dir)
    face = face[(face.queryType == cr.FACE) & face.clipName.isin(ids)]
    votes = cr.stack_votes(face, list(ids))["votes"]
    rng = np.random.default_rng(SEED)
    rhos = []
    for _ in range(N_SPLITS):
        mask = cr.half_mask(votes, rng)
        first = directed_table(cells, shares(cr.counts_for(votes, mask), pseudo))
        second = directed_table(cells, shares(cr.counts_for(votes, ~mask & (votes >= 0)), pseudo))
        keys = sorted(first)
        rhos.append(float(spearmanr([first[k]["X"] for k in keys], [second[k]["X"] for k in keys])[0]))
    half = float(np.mean(rhos))
    brown = cr.spearman_brown(half)
    return {"n_splits": N_SPLITS, "half_rank_r": half, "half_rank_r_sd": float(np.std(rhos)), "spearman_brown": brown,
            "ceiling_sqrt_sb": float(np.sqrt(max(brown, 0.0)))}


def build(data_dir: Path) -> Dict[str, Any]:
    import pandas as pd

    frame = pd.read_csv(d.SELECTION)
    columns = [f"votes_{c}" for c in "ADFHNS"]
    votes = frame[columns].to_numpy(dtype=np.float64)
    labels = [ar.CODE_TO_OPTION[c] for c in frame.emotion]
    actors = [str(a) for a in frame.actor_id]
    ids = frame.clip_id.tolist()
    cells = holder(ids, actors, labels)
    emo = cells.emo
    # Cells sorts clips by id, so the votes are put in that order.
    order = np.argsort(ids)
    votes = votes[order]
    pseudo = float(D["pseudo_count_human"])
    p = shares(votes, pseudo)
    rows = pair_rows(cells, p)
    rule = pair_rule(rows)
    # The standardised share takes logs, so a pseudo-count is needed; stability is checked at
    # a tenth and at twice the pinned one instead of at none.
    low_pseudo = pair_rows(cells, shares(votes, 0.1))
    high_pseudo = pair_rows(cells, shares(votes, 1.0))

    table = pd.read_csv(data_dir / "tabulatedVotes.csv", index_col=0)
    table = table[table.index.astype(str).str[0] == "2"].sort_values("fileName")
    if len(table) != 7442:
        raise SystemExit(f"Expected 7,442 face-only rows, found {len(table)}.")
    every = holder(table.fileName.tolist(), [c.split("_")[0] for c in table.fileName],
                   [ar.CODE_TO_OPTION[{"ANG": "A", "DIS": "D", "FEA": "F", "HAP": "H", "NEU": "N", "SAD": "S"}[c.split("_")[2]]] for c in table.fileName])
    all_rows = pair_rows(every, shares(table[list("ADFHNS")].to_numpy(dtype=np.float64), pseudo))

    modal = votes.max(1) == votes[np.arange(len(votes)), emo]
    a, b = (an.INDEX[x] for x in rule["target_pair"])
    return {
        "selection_sha256": d.file_digest(d.SELECTION),
        "note": "Options in columns: " + ", ".join(OPTIONS) + ". Rows of vote_share are intended emotions.",
        "n_clips": len(ids),
        "n_actors": len(set(actors)),
        "pseudo_count": pseudo,
        "bootstrap_draws": DRAWS,
        "seed": SEED,
        "vote_share": vote_table(votes, emo),
        "pairs": rows,
        "directed": directed_table(cells, p),
        "rule": rule,
        "stability": {
            "pseudo_count_0.1": {"rule": pair_rule(low_pseudo), "top_rows": low_pseudo[:5]},
            "pseudo_count_1.0": {"rule": pair_rule(high_pseudo), "top_rows": high_pseudo[:5]},
            "all_7442_clips": {"rule": pair_rule(all_rows), "top_rows": all_rows[:5]},
        },
        "read_as_other": {f"{OPTIONS[x]}>{OPTIONS[y]}": read_as_other(votes, emo, x, y) for x, y in ((a, b), (b, a))},
        "clips_mode_is_intended": {"n": int(modal.sum()), "note": "A tie for top counts as the mode.",
                                   "n_tie_with_intended": int((modal & ((votes == votes.max(1, keepdims=True)).sum(1) > 1)).sum())},
        "split_half_rank_ceiling": split_half(data_dir, ids, cells, pseudo),
    }


def cue_check(frames: Sequence[Path], geometry: Path, output: Path) -> None:
    """Partner share on AU9 and AU10 at the apex, within actor and emotion, per detector pair."""
    import pandas as pd

    records = json.loads(geometry.read_text(encoding="utf-8"))["items"]
    frames_table = pd.concat([pd.read_csv(p) for p in frames], ignore_index=True)
    apex = {t: r.get("apex") for t, r in records.items()}
    first = {t: (r.get("base_indices") or [0])[0] for t, r in records.items()}
    table = frames_table[frames_table.frame_index == frames_table.trial_id.map(apex)].groupby("trial_id").first()
    opening = frames_table[frames_table.frame_index == frames_table.trial_id.map(first)].groupby("trial_id").first()
    items = ar.load_items(d.SELECTION, Path("."))
    cells = holder([i["trial_id"] for i in items], [i["actor"] for i in items], [i["label"] for i in items])
    selection = {i["trial_id"]: i for i in items}
    votes = np.array([[selection[t]["human_votes"][o] for o in OPTIONS] for t in cells.ids], dtype=np.float64)
    p = shares(votes, float(D["pseudo_count_human"]))
    x, y = D["target_pair"]
    out: Dict[str, Any] = {"apex_rows": int(len(table)), "gates_nothing": True, "cells": {}, "first_frame": {}}
    # The first fill is only a fill if the first frame is not already expressive. This check
    # blocks the freeze, not a verdict: the mean first-frame unit over the mean apex unit, on
    # clips of the pair, must stay at or under first_frame_au_ratio_max for every detector.
    limit = float(D["first_frame_au_ratio_max"])
    pair_ids = [i["trial_id"] for i in items if i["label"] in D["target_pair"]]
    blocked = False
    for column in [c for unit in CUE_COLUMNS.values() for c in unit]:
        if column not in table.columns or column not in opening.columns:
            out["first_frame"][column] = {"missing_detector": True}
            continue
        at_apex = float(table.reindex(pair_ids)[column].mean())
        at_first = float(opening.reindex(pair_ids)[column].mean())
        ratio = at_first / at_apex if at_apex else None
        over = ratio is None or ratio > limit
        blocked = blocked or over
        out["first_frame"][column] = {"mean_apex": at_apex, "mean_first": at_first, "ratio": ratio, "over_limit": over}
    out["first_frame_blocks_freeze"] = blocked
    for acted, partner in ((x, y), (y, x)):
        mask = cells.emo == an.INDEX[acted]
        share = an.error_share(p, cells.emo, an.INDEX[partner])[mask]
        actors = cells.actor[mask]
        for au_nine in CUE_COLUMNS["AU09"]:
            for au_ten in CUE_COLUMNS["AU10"]:
                if au_nine not in table.columns or au_ten not in table.columns:
                    out["cells"][f"{acted}>{partner}|{au_nine}|{au_ten}"] = {"missing_detector": True}
                    continue
                values = table.reindex(np.array(cells.ids)[mask])[[au_nine, au_ten]].to_numpy(dtype=np.float64)
                keep = ~np.isnan(values).any(1)
                design = np.column_stack([values[keep], np.eye(actors.max() + 1)[actors[keep]]])
                beta = np.linalg.lstsq(design, share[keep], rcond=None)[0]
                out["cells"][f"{acted}>{partner}|{au_nine}|{au_ten}"] = {"n_clips": int(keep.sum()), "slope_au9": float(beta[0]), "slope_au10": float(beta[1])}
    output.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data_dir", type=Path, default=ROOT / "data" / "external" / "crema_d")
    parser.add_argument("--output", type=Path, default=d.HUMAN_TARGET)
    parser.add_argument("--cue_frames", type=Path, nargs="*", default=None)
    parser.add_argument("--geometry", type=Path, default=d.GEOMETRY)
    parser.add_argument("--cue_output", type=Path, default=ROOT / "research" / "reproduction" / "a6_human_cue_check.json")
    args = parser.parse_args()
    if args.cue_frames:
        cue_check(args.cue_frames, args.geometry, args.cue_output)
        return
    out = build(args.data_dir)
    text = json.dumps(out, indent=1, sort_keys=True) + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(hashlib.sha256(text.encode("utf-8")).hexdigest())


if __name__ == "__main__":
    main()
