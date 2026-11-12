#!/usr/bin/env python3
"""Numerical check of Proposition 5 for residual CEA with estimated projectors (G9, Supplementary Note 1 gap 1).

Proposition 5 (noise ceiling) is exact, as a ratio of expectations, for raw CEA, and for residual
CEA only when each matrix's projector is fixed. A4 estimates each projector from the noisy matrix
itself (`remove_common` in `scripts/a4_analyse.py`), so no identity links the split-half residual
reliability to the attenuation of residual CEA. This script measures the gap by simulation. It is a
theory check on synthetic matrices: it reads no model output, changes no frozen file, threshold or
rule, and its result is reported as a numerical check, not a proof.

Model of the estimates. Each system's effect matrix (options by cues, zero column sums, the
orientation `remove_common` uses) is estimated as truth plus zero-mean Gaussian noise projected to
zero column sums. A half-sample estimate has twice the noise variance. The model-side reliability is
estimated as A4 does (`split_half`): the mean cosine between two independent half-sample matrices,
raw and after `remove_common`, stepped up by Spearman-Brown. Ridge shrinkage and the refit on each
half are not modelled, which is the main limit of this check.

Truth. tau_H = a_H u_H w' + N_H, with u_H a unit zero-sum profile, w positive cue saliences, and N_H
in the space orthogonal to 1 and u_H with zero row sums, so remove_common(tau_H) = N_H exactly. The
common-profile share of tau_H is set to .30 (A4 human values .26 to .35, a4_human_ceiling.json).
tau_S is built the same way around its own profile u_S, either equal to u_H or at cosine .7 with it,
with its residual part at a set cosine with N_H projected into its own residual space.

Reported per cell: the true residual and raw CEA; the mean observed value; the mean split-half
reliability; and the bias, against the truth, of (a) observed / sqrt(r_s), the form A4's Not aligned
rule uses (model reliability only), and (b) observed / sqrt(r_s * R_H), the full correction with the
human reliability known. The same quantities with each true projector held fixed show Proposition 5
holding, which checks the simulation itself.

Usage (from bullpy/mr_eu_open_llm):
    PYTHONPATH=.:src python research/scripts/residual_cea_attenuation.py \
        --out research/reproduction/residual_cea_attenuation.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# A4's shapes: options by cue families (`(K - 2)(p - 1)` is 120 for basic and 72 for
# conversational in Supplementary Note 1, Corollary S3, so p = 25 with K = 7 and K = 5).
SHAPES = {"basic": (7, 25), "conversational": (5, 25)}
HUMAN_COMMON_SHARE = 0.30
MODEL_COMMON_SHARES = (0.1, 0.3, 0.6)
TARGET_RESIDUAL_COSINES = (0.0, 0.3, 0.6, 0.9)
MODEL_RELIABILITIES = (0.5, 0.7, 0.85, 0.95)  # raw, full sample
HUMAN_RELIABILITY = 0.95  # raw; A4's residual human ceiling squared is .92 to .97
PROFILE_COSINES = (1.0, 0.7)  # cosine between u_S and u_H
SPLITS = 20  # scripts/a4_design.py SPLIT_HALF_SPLITS
SEED = 20261112


def remove_common_batch(tau: np.ndarray) -> np.ndarray:
    """`scripts/a4_analyse.remove_common` over a leading batch axis (tested equal to it)."""
    mean = tau.mean(axis=-1)  # (..., K)
    size = np.linalg.norm(mean, axis=-1, keepdims=True)
    unit = np.divide(mean, size, out=np.zeros_like(mean), where=size > 0)
    return tau - unit[..., :, None] * np.einsum("...k,...kj->...j", unit, tau)[..., None, :]


def project_fixed(tau: np.ndarray, unit: np.ndarray) -> np.ndarray:
    """Remove the component along a fixed, known unit profile from every cue column."""
    return tau - np.outer(unit, unit @ tau)


def cosine_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a2, b2 = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    denom = np.linalg.norm(a2, axis=1) * np.linalg.norm(b2, axis=1)
    return np.einsum("ij,ij->i", a2, b2) / denom


def spearman_brown(r: np.ndarray) -> np.ndarray:
    return 2 * r / (1 + r)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def zero_sum_unit(K: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(K)
    return _unit(v - v.mean())


def residual_space_projector(K: int, unit: np.ndarray) -> np.ndarray:
    """Projector onto the options subspace orthogonal to 1 and to `unit`."""
    C = np.eye(K) - np.ones((K, K)) / K
    return C - np.outer(unit, unit)


def into_residual_space(M: np.ndarray, unit: np.ndarray) -> np.ndarray:
    K, p = M.shape
    L = residual_space_projector(K, unit)
    R = np.eye(p) - np.ones((p, p)) / p
    return L @ M @ R


def build_matrix(unit: np.ndarray, residual: np.ndarray, share: float, rng: np.random.Generator) -> np.ndarray:
    """a * unit w' + residual, with the common part carrying `share` of the squared norm."""
    p = residual.shape[1]
    w = np.abs(rng.normal(1.0, 0.5, size=p)) + 0.05
    common = np.outer(unit, w)
    res_sq, com_sq = float(np.sum(residual ** 2)), float(np.sum(common ** 2))
    a = np.sqrt(share / (1 - share) * res_sq / com_sq)
    return a * common + residual


def truth(K: int, p: int, model_share: float, target: float, profile_cos: float,
          rng: np.random.Generator) -> Dict[str, np.ndarray]:
    u_h = zero_sum_unit(K, rng)
    n_h = into_residual_space(rng.standard_normal((K, p)), u_h)
    n_h /= np.linalg.norm(n_h)
    tau_h = build_matrix(u_h, n_h, HUMAN_COMMON_SHARE, rng)
    if profile_cos >= 1.0:
        u_s = u_h.copy()
    else:
        other = zero_sum_unit(K, rng)
        other = _unit(other - (other @ u_h) * u_h)
        u_s = _unit(profile_cos * u_h + np.sqrt(1 - profile_cos ** 2) * other)
    tilde = into_residual_space(n_h, u_s)
    tilde /= np.linalg.norm(tilde)
    v = into_residual_space(rng.standard_normal((K, p)), u_s)
    v -= np.sum(v * tilde) * tilde
    v /= np.linalg.norm(v)
    n_s = target * tilde + np.sqrt(1 - target ** 2) * v
    tau_s = build_matrix(u_s, n_s, model_share, rng)
    return {"tau_s": tau_s, "tau_h": tau_h, "u_s": u_s, "u_h": u_h}


def noise(shape, sigma: float, reps: int, rng: np.random.Generator) -> np.ndarray:
    K = shape[0]
    e = rng.standard_normal((reps, *shape)) * sigma
    return e - e.mean(axis=1, keepdims=True)  # zero column sums


def sigma_for(tau: np.ndarray, reliability: float) -> float:
    K, p = tau.shape
    return float(np.sqrt(np.sum(tau ** 2) * (1 / reliability - 1) / ((K - 1) * p)))


def run_cell(K: int, p: int, model_share: float, target: float, rel_s: float, profile_cos: float,
             reps: int, rng: np.random.Generator) -> Dict[str, float]:
    t = truth(K, p, model_share, target, profile_cos, rng)
    tau_s, tau_h = t["tau_s"], t["tau_h"]
    sig_s, sig_h = sigma_for(tau_s, rel_s), sigma_for(tau_h, HUMAN_RELIABILITY)
    hat_s = tau_s + noise(tau_s.shape, sig_s, reps, rng)
    hat_h = tau_h + noise(tau_h.shape, sig_h, reps, rng)
    out: Dict[str, float] = {}
    true_res = float(cosine_batch(remove_common_batch(tau_s[None]), remove_common_batch(tau_h[None]))[0])
    true_raw = float(cosine_batch(tau_s[None], tau_h[None])[0])
    # Split-half model reliability as A4 computes it: two half-sample matrices per split.
    halves = (tau_s + noise(tau_s.shape, sig_s * np.sqrt(2), 2 * SPLITS * reps, rng)).reshape(
        reps, SPLITS, 2, *tau_s.shape)
    first, second = halves[:, :, 0].reshape(-1, K, p), halves[:, :, 1].reshape(-1, K, p)
    r_raw = cosine_batch(first, second).reshape(reps, SPLITS).mean(axis=1)
    r_res = cosine_batch(remove_common_batch(first), remove_common_batch(second)).reshape(reps, SPLITS).mean(axis=1)
    rs_raw, rs_res = spearman_brown(r_raw), spearman_brown(r_res)
    # Human residual reliability with the human projector estimated, from the same construction.
    h_first = tau_h + noise(tau_h.shape, sig_h * np.sqrt(2), reps, rng)
    h_second = tau_h + noise(tau_h.shape, sig_h * np.sqrt(2), reps, rng)
    rh_res = float(np.mean(spearman_brown(cosine_batch(remove_common_batch(h_first), remove_common_batch(h_second)))))
    rh_raw = float(np.mean(spearman_brown(cosine_batch(h_first, h_second))))

    obs_res = cosine_batch(remove_common_batch(hat_s), remove_common_batch(hat_h))
    obs_raw = cosine_batch(hat_s, hat_h)
    # Fixed, known projectors: Proposition 5 holds here as a ratio of expectations.
    fixed_s = np.stack([project_fixed(m, t["u_s"]) for m in hat_s])
    fixed_h = np.stack([project_fixed(m, t["u_h"]) for m in hat_h])
    obs_fixed = cosine_batch(fixed_s, fixed_h)
    pos = rs_res > 0
    dec_res = obs_res[pos] / np.sqrt(rs_res[pos])
    full_res = obs_res[pos] / np.sqrt(rs_res[pos] * rh_res)
    dec_raw = obs_raw / np.sqrt(rs_raw)
    full_raw = obs_raw / np.sqrt(rs_raw * rh_raw)
    out.update({
        "true_residual_cea": round(true_res, 4), "true_raw_cea": round(true_raw, 4),
        "mean_observed_residual": round(float(obs_res.mean()), 4),
        "mean_observed_residual_fixed_projector": round(float(obs_fixed.mean()), 4),
        "mean_r_s_residual": round(float(rs_res.mean()), 4), "mean_r_s_raw": round(float(rs_raw.mean()), 4),
        "r_h_residual": round(rh_res, 4), "r_h_raw": round(rh_raw, 4),
        "share_r_s_residual_not_positive": round(float(np.mean(~pos)), 4),
        "bias_decision_form_residual": round(float(dec_res.mean() - true_res), 4),
        "bias_full_correction_residual": round(float(full_res.mean() - true_res), 4),
        "q05_q95_decision_form_residual": [round(float(np.quantile(dec_res, q)), 4) for q in (0.05, 0.95)],
        "bias_decision_form_raw": round(float(dec_raw.mean() - true_raw), 4),
        "bias_full_correction_raw": round(float(full_raw.mean() - true_raw), 4),
    })
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    cells = []
    for dataset, (K, p) in SHAPES.items():
        for profile_cos in PROFILE_COSINES:
            for share in MODEL_COMMON_SHARES:
                for target in TARGET_RESIDUAL_COSINES:
                    for rel in MODEL_RELIABILITIES:
                        cell = {"dataset": dataset, "K": K, "p": p, "profile_cosine": profile_cos,
                                "model_common_share": share, "target_residual_cosine": target,
                                "model_raw_reliability": rel}
                        cell.update(run_cell(K, p, share, target, rel, profile_cos, args.reps, rng))
                        cells.append(cell)
    dec = np.array([c["bias_decision_form_residual"] for c in cells])
    full = np.array([c["bias_full_correction_residual"] for c in cells])
    at_bound = [c for c in cells if c["target_residual_cosine"] == 0.3]
    gap = np.array([c["mean_observed_residual"] - c["mean_observed_residual_fixed_projector"] for c in cells])
    summary = {
        "cells": len(cells), "reps_per_cell": args.reps, "seed": args.seed,
        "bias_decision_form_residual_range": [round(float(dec.min()), 4), round(float(dec.max()), 4)],
        "bias_full_correction_residual_range": [round(float(full.min()), 4), round(float(full.max()), 4)],
        "bias_decision_form_residual_at_target_0p3_range": [
            round(min(c["bias_decision_form_residual"] for c in at_bound), 4),
            round(max(c["bias_decision_form_residual"] for c in at_bound), 4)],
        "true_residual_cea_at_target_0p3_range": [
            round(min(c["true_residual_cea"] for c in at_bound), 4),
            round(max(c["true_residual_cea"] for c in at_bound), 4)],
        "estimated_minus_fixed_projector_mean_observed_range": [round(float(gap.min()), 4), round(float(gap.max()), 4)],
        "max_share_r_s_residual_not_positive": max(c["share_r_s_residual_not_positive"] for c in cells),
    }
    script = Path(__file__).read_bytes()
    result = {"what": "Proposition 5 for residual CEA with estimated projectors: simulation, not a proof",
              "script_sha256": hashlib.sha256(script).hexdigest(), "design": {
                  "shapes": SHAPES, "human_common_share": HUMAN_COMMON_SHARE,
                  "model_common_shares": MODEL_COMMON_SHARES, "target_residual_cosines": TARGET_RESIDUAL_COSINES,
                  "model_raw_reliabilities": MODEL_RELIABILITIES, "human_raw_reliability": HUMAN_RELIABILITY,
                  "profile_cosines": PROFILE_COSINES, "splits": SPLITS},
              "summary": summary, "cells": cells}
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
