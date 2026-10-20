"""Task-conditioned residual directions. Not narrative E/S prose.

same_image_rmet: mental-state vs appearance on the same faces, plus a
text-only control so a layer-0 AUROC of 1.0 is not mistaken for a
social mechanism. video_eu_clevrer: modality-matched video pair.
cross_modal_rmet_clevrer: faces vs videos; treat as a confound check.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from esvlm.analysis.separability import residualize, unique_fraction
from esvlm.cli.score import DEFAULT_TEMPLATE, load_task_items, slice_items
from esvlm.scoring.logprob import _encode
from esvlm.tasks.items import drop_media, rmet_appearance_items
from esvlm.tasks.media import load_media
from esvlm.utils.envcheck import check_job_env
from esvlm.utils.logging import echo_run_args, setup_logging
from esvlm.utils.seeding import seed_everything
from esvlm.vectors.extract import ResidualCatcher, difference_of_means, directions_for_layers
from esvlm.vectors.validate import projection_auroc


def item_residuals(model, processor, modules, item, images, device) -> np.ndarray:
    """Last-token residual at every hooked layer for the question only. [L, D]."""

    import torch

    inputs = _encode(processor, item.prompt_text, images)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    catcher = ResidualCatcher(modules)
    with torch.no_grad(), catcher:
        model(**inputs)
    return catcher.stacked()


def stack_task(model, processor, modules, items, device, *, frames: int, log) -> tuple[np.ndarray, list[str]]:
    acts = []
    ids = []
    for item in items:
        media = load_media(item, frames=frames) if item.media_type != "none" else []
        acts.append(item_residuals(model, processor, modules, item, media or None, device))
        ids.append(item.item_id)
        if len(ids) % 32 == 0:
            log.info("task_residual n=%s last=%s", len(ids), item.item_id)
    return np.stack(acts, axis=0), ids


def train_hold_idx(n: int, *, hold_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_hold = max(4, int(n * hold_frac))
    n_hold = min(n_hold, n - 4)
    return np.sort(perm[n_hold:]), np.sort(perm[:n_hold])


def paired_unit_dirs(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """pos/neg [N, L, D] aligned. Unit mean(pos-neg) per layer."""

    from esvlm.steering.hooks import unit_vector

    delta = pos - neg
    return np.stack([unit_vector(delta[:, i, :].mean(axis=0)) for i in range(delta.shape[1])], axis=0)


def leftover_dirs(vecs: np.ndarray, remove: np.ndarray) -> tuple[np.ndarray, list[bool]]:
    """Residualize each layer direction against ``remove``. Zeros if collinear."""

    out = []
    collinear = []
    for i in range(len(vecs)):
        try:
            out.append(residualize(vecs[i], remove[i]))
            collinear.append(False)
        except ValueError:
            out.append(np.zeros_like(vecs[i]))
            collinear.append(True)
    return np.stack(out, axis=0), collinear


def safe_projection_auroc(pos: np.ndarray, neg: np.ndarray, direction: np.ndarray) -> float:
    if float(np.linalg.norm(direction)) < 1e-12:
        return float("nan")
    return projection_auroc(pos, neg, direction)


def pick_candidate_layer(scores: np.ndarray, *, min_layer: int = 4) -> int:
    """Lock the steer layer from hold-out AUROC before any steer is run."""

    if scores.size == 0:
        raise ValueError("no layer scores")
    eligible = [i for i in range(len(scores)) if i >= min_layer and np.isfinite(scores[i])]
    if not eligible:
        eligible = [i for i in range(len(scores)) if np.isfinite(scores[i])]
    if not eligible:
        return 0
    return int(max(eligible, key=lambda i: scores[i]))


def is_modality_confound(layer0: float, *, threshold: float = 0.95) -> bool:
    return bool(np.isfinite(layer0) and layer0 >= threshold)


def decide_gates(
    *,
    raw_layer0: float,
    pick_scores: np.ndarray,
    unique_vs_text: list[float] | None = None,
    min_unique: float = 1e-3,
    auroc_min: float = 0.60,
) -> dict[str, object]:
    """Lock steer_ok. A last-layer AUROC spike on a ~0 leftover is not a mechanism."""

    raw = np.asarray(pick_scores, dtype=np.float64)
    scores = raw.copy()
    visual_mean = float(np.nanmean(raw)) if raw.size else float("nan")
    wording_only = bool(
        unique_vs_text is not None
        and np.isfinite(raw_layer0)
        and raw_layer0 >= 0.95
        and np.isfinite(visual_mean)
        and visual_mean < auroc_min
    )
    if unique_vs_text is not None:
        uniq = np.asarray(unique_vs_text, dtype=np.float64)
        if uniq.shape == scores.shape:
            scores[uniq < min_unique] = np.nan
        confound_score = float(raw[0]) if raw.size else float("nan")
    else:
        confound_score = float(raw[0]) if raw.size else raw_layer0
    confound = is_modality_confound(confound_score)
    best = pick_candidate_layer(scores)
    cand = float(scores[best]) if scores.size and np.isfinite(scores[best]) else float("nan")
    degenerate = bool(not np.isfinite(cand) or cand < auroc_min)
    return {
        "pick_scores": scores,
        "confound_score": confound_score,
        "modality_confound": confound,
        "wording_only": wording_only,
        "candidate_layer": best,
        "candidate_held_out_auroc": cand,
        "degenerate": degenerate,
        "steer_ok": bool(not confound and not degenerate and not wording_only),
    }


def load_design_items(design: str, args) -> tuple[list, list, str, str, bool]:
    if design == "same_image_rmet":
        social, _ = load_task_items("rmet", None, template_id="classic")
        social = slice_items(social, 0, args.trial_limit_social)
        return social, rmet_appearance_items(social), "rmet_classic", "rmet_appearance", True
    if design == "video_eu_clevrer":
        social, _ = load_task_items("eu_emotions", None, template_id=DEFAULT_TEMPLATE["eu_emotions"])
        mech, _ = load_task_items("clevrer", None, template_id=DEFAULT_TEMPLATE["clevrer"])
        return (
            slice_items(social, 0, args.trial_limit_social),
            slice_items(mech, 0, args.trial_limit_mechanical),
            "eu_emotions",
            "clevrer",
            False,
        )
    if design == "cross_modal_rmet_clevrer":
        social, _ = load_task_items("rmet", None, template_id="classic")
        mech, _ = load_task_items("clevrer", None, template_id=DEFAULT_TEMPLATE["clevrer"])
        return (
            slice_items(social, 0, args.trial_limit_social),
            slice_items(mech, 0, args.trial_limit_mechanical),
            "rmet",
            "clevrer",
            False,
        )
    raise SystemExit(f"unknown design {design}")


def main(argv: list[str] | None = None) -> int:
    check_job_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--design",
        default="same_image_rmet",
        choices=("same_image_rmet", "video_eu_clevrer", "cross_modal_rmet_clevrer"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trial_limit_social", type=int, default=0)
    parser.add_argument("--trial_limit_mechanical", type=int, default=256)
    parser.add_argument("--hold_frac", type=float, default=0.33)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(os.environ["ESVLM_MODELS_DIR"]) if os.environ.get("ESVLM_MODELS_DIR") else None,
    )
    args = parser.parse_args(argv)
    log = setup_logging()
    echo_run_args(vars(args), log)
    seed_everything(args.seed)

    social_items, other_items, social_name, other_name, paired = load_design_items(args.design, args)
    if len(social_items) < 8 or len(other_items) < 8:
        raise SystemExit("need at least 8 items per side")

    from esvlm.models.loader import load_model, residual_modules
    from esvlm.utils.device import torch_device

    t0 = time.perf_counter()
    loaded = load_model(args.model, models_dir=args.models_dir, offline=args.offline)
    modules = residual_modules(loaded)
    device = torch_device(loaded.device)
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    social, social_ids = stack_task(
        loaded.model, loaded.processor, modules, social_items, device, frames=args.frames, log=log
    )
    other, other_ids = stack_task(
        loaded.model, loaded.processor, modules, other_items, device, frames=args.frames, log=log
    )
    train_i, hold_i = train_hold_idx(len(social_ids), hold_frac=args.hold_frac, seed=args.seed)
    if paired and len(other_ids) != len(social_ids):
        raise SystemExit("paired design length mismatch")
    if not paired:
        other_train, other_hold = train_hold_idx(len(other_ids), hold_frac=args.hold_frac, seed=args.seed + 1)
    else:
        other_train, other_hold = train_i, hold_i

    if paired:
        vecs = paired_unit_dirs(social[train_i], other[other_train])
    else:
        vecs = directions_for_layers(social[train_i], other[other_train])

    held = [
        safe_projection_auroc(social[hold_i, i, :], other[other_hold, i, :], vecs[i])
        for i in range(vecs.shape[0])
    ]

    text_held: list[float] | None = None
    visual_held: list[float] | None = None
    visual_dirs = None
    text_collinear: list[bool] | None = None
    unique_vs_text: list[float] | None = None
    if args.design == "same_image_rmet":
        social_txt, _ = stack_task(
            loaded.model,
            loaded.processor,
            modules,
            drop_media(social_items),
            device,
            frames=args.frames,
            log=log,
        )
        other_txt, _ = stack_task(
            loaded.model,
            loaded.processor,
            modules,
            drop_media(other_items),
            device,
            frames=args.frames,
            log=log,
        )
        text_dirs = paired_unit_dirs(social_txt[train_i], other_txt[other_train])
        visual_dirs, text_collinear = leftover_dirs(vecs, text_dirs)
        text_held = [
            safe_projection_auroc(social_txt[hold_i, i, :], other_txt[other_hold, i, :], text_dirs[i])
            for i in range(vecs.shape[0])
        ]
        visual_held = [
            safe_projection_auroc(social[hold_i, i, :], other[other_hold, i, :], visual_dirs[i])
            for i in range(vecs.shape[0])
        ]
        unique_vs_text = [unique_fraction(vecs[i], text_dirs[i]) for i in range(vecs.shape[0])]

    layer0 = float(held[0]) if held else float("nan")
    pick_scores = np.asarray(visual_held if visual_held is not None else held, dtype=np.float64)
    gates = decide_gates(
        raw_layer0=layer0,
        pick_scores=pick_scores,
        unique_vs_text=unique_vs_text,
    )
    pick_scores = np.asarray(gates["pick_scores"], dtype=np.float64)
    confound_score = float(gates["confound_score"])
    confound = bool(gates["modality_confound"])
    best = int(gates["candidate_layer"])
    degenerate = bool(gates["degenerate"])
    wording_only = bool(gates["wording_only"])
    steer_ok = bool(gates["steer_ok"])

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "social": social,
        "other": other,
        "social_minus_other": vecs,
        "social_ids": np.asarray(social_ids),
        "other_ids": np.asarray(other_ids),
        "train_idx": train_i,
        "hold_idx": hold_i,
        "other_hold_idx": other_hold,
    }
    if visual_dirs is not None:
        payload["visual_minus_text"] = visual_dirs
    np.savez_compressed(args.out / f"{loaded.spec.key}__task_residuals.npz", **payload)

    for i in range(vecs.shape[0]):
        layer_payload = {
            "social_minus_mechanical": vecs[i],
            "E": visual_dirs[i] if visual_dirs is not None else vecs[i],
            "S": difference_of_means(other[other_train, i, :], social[train_i, i, :]),
        }
        if visual_dirs is not None:
            layer_payload["E_visual"] = visual_dirs[i]
            layer_payload["E_raw"] = vecs[i]
        np.savez_compressed(args.out / f"{loaded.spec.key}__layer_{i:02d}.npz", **layer_payload)

    hold_social_ids = [social_ids[int(i)] for i in hold_i]
    hold_other_ids = [other_ids[int(i)] for i in other_hold]
    train_social_ids = [social_ids[int(i)] for i in train_i]
    candidate_score = float(pick_scores[best]) if pick_scores.size else float("nan")
    summary = {
        "model": loaded.spec.key,
        "revision": loaded.spec.revision,
        "construct": "task_conditioned",
        "design": args.design,
        "social_task": social_name,
        "mechanical_task": other_name,
        "paired": paired,
        "n_social": len(social_ids),
        "n_other": len(other_ids),
        "n_train": int(len(train_i)),
        "n_hold": int(len(hold_i)),
        "n_layers": int(vecs.shape[0]),
        "dim": int(vecs.shape[-1]),
        "train_social_ids": train_social_ids,
        "hold_social_ids": hold_social_ids,
        "hold_other_ids": hold_other_ids,
        "held_out_auroc_by_layer": [float(x) for x in held],
        "held_out_auroc_mean": float(np.nanmean(held)),
        "held_out_auroc_layer0": layer0,
        "confound_score": confound_score,
        "wording_only": wording_only,
        "text_only_auroc_by_layer": [float(x) for x in text_held] if text_held is not None else None,
        "visual_residual_auroc_by_layer": [float(x) for x in visual_held] if visual_held is not None else None,
        "unique_vs_text_by_layer": [float(x) for x in unique_vs_text] if unique_vs_text is not None else None,
        "text_collinear_layers": [i for i, flag in enumerate(text_collinear or []) if flag],
        "modality_confound": confound,
        "degenerate": degenerate,
        "candidate_layer": best,
        "candidate_held_out_auroc": candidate_score,
        "candidate_score_source": "visual_residual" if visual_held is not None else "raw_hold",
        "steer_ok": steer_ok,
        "load_seconds": round(load_s, 3),
        "extract_seconds": round(time.perf_counter() - t1, 3),
        "out": str(args.out),
        "note": "train-only directions; projection AUROC on hold; not narrative E/S",
    }
    (args.out / "task_extract_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("task_extract_summary %s", json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
