#!/usr/bin/env python3
"""Numerical companion to research/theory/causal_feature_alignment.md.

Checks the constructions in the note on synthetic systems, and measures how far
natural cue covariance limits what behaviour can say about cue effects, on the EU-Emotion
cue table against the randomised Snoek design. Reads no model output.

Usage:
    python3 research/scripts/cfa_identifiability.py \
        --au-csv research/reproduction/eu_au_measures.csv \
        --snoek-zip data/external/snoek_au/au_emotion_data.zip \
        --out research/reproduction/cfa_identifiability.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parents[2]


def _module(name):
    path = ROOT / "research/scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def cosine(a, b):
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def theorem_1_system(f, target):
    """Interventional responses that give effect matrix `target` for natural responses `f`.

    f has one row per item and one column per option. target has one row per cue and
    sums to zero within a row. Returns responses of shape (cues, options), the same for
    every item, and raises if they leave the simplex.
    """
    fbar = f.mean(axis=0)
    g = fbar[None, :] + target
    if g.min() < -1e-12 or abs(g.sum(axis=1) - 1).max() > 1e-12:
        raise ValueError("Target matrix is outside the feasible set")
    return g


def effect_matrix(f, g):
    """Mean over items of intervened minus natural response, by cue and option."""
    return g - f.mean(axis=0)[None, :]


def matrix_with_correlation(reference, r, rng):
    """A matrix of the same shape with row sums zero and cosine r with `reference`."""
    u = rng.normal(size=reference.shape)
    u -= u.mean(axis=1, keepdims=True)
    unit = reference / np.linalg.norm(reference)
    u -= (u * unit).sum() * unit
    u /= np.linalg.norm(u)
    return r * unit + np.sqrt(1 - r**2) * u


def two_cue_case(rho):
    """Exact values for two standardised cues with correlation rho, weights (1,0) and (0,1)."""
    return {"rho": rho, "output_correlation": rho, "mean_squared_difference": 2 * (1 - rho),
            "effect_cosine": 0.0}


def spectrum(table):
    """Eigenvalues of the correlation matrix, and the bound factor of proposition 3."""
    values = table.to_numpy(dtype=float)
    keep = values.std(axis=0) > 0
    corr = np.corrcoef(values[:, keep], rowvar=False)
    eig = np.linalg.eigvalsh(corr)
    off = corr[np.triu_indices_from(corr, 1)]
    smallest = float(eig[0])
    return {"n_rows": int(values.shape[0]), "n_cues": int(keep.sum()),
            "lambda_min": round(smallest, 5), "lambda_max": round(float(eig[-1]), 4),
            "condition_number": round(float(eig[-1] / eig[0]), 2),
            "invisibility_factor_1_over_lambda_min": round(1 / smallest, 2),
            "mean_abs_off_diagonal_r": round(float(np.abs(off).mean()), 4),
            "max_abs_off_diagonal_r": round(float(np.abs(off).max()), 4)}


def eu_spectra(au_csv):
    scc = _module("same_clip_cue_analysis")
    out = {}
    for detector in ("libreface", "openface3"):
        table = scc.load_cues(au_csv, detector=detector, frame_set="all")
        out[detector] = spectrum(table.dropna())
    return out


def snoek_spectra(path):
    hae = _module("human_au_effects")
    frames, _ = hae.load_zip(path)
    return {name: spectrum(frame[list(hae.AU_COLS)].drop_duplicates())
            for name, (frame, _label) in frames.items()}


def checks(seed=42):
    rng = np.random.default_rng(seed)
    n, p, k = 200, 8, 6
    f = rng.dirichlet(np.ones(k), size=n)
    human = rng.normal(size=(p, k))
    human -= human.mean(axis=1, keepdims=True)
    human *= 0.1 / np.abs(human).max()
    fbar = f.mean(axis=0)
    room = float((fbar.min()) / np.abs(human).max())
    rows = []
    for r in (-1.0, -0.5, 0.0, 0.5, 1.0):
        target = matrix_with_correlation(human, r, rng)
        target *= 0.5 * room * np.abs(human).max() / np.abs(target).max()
        got = effect_matrix(f, theorem_1_system(f, target))
        rows.append({"r_requested": r, "r_achieved": round(cosine(got, human), 8),
                     "max_entry_error": float(np.abs(got - target).max())})
    sigma = np.array([[1, .9, .2], [.9, 1, .1], [.2, .1, 1]])
    eig, vec = np.linalg.eigh(sigma)
    delta = np.outer([1, -1], vec[:, 0])
    ratio = np.linalg.norm(delta)**2 / np.trace(delta @ sigma @ delta.T)
    return {"theorem_1": rows,
            "proposition_3_tight_case": {"ratio": round(float(ratio), 6),
                                         "one_over_lambda_min": round(float(1 / eig[0]), 6)},
            "proposition_2": [two_cue_case(x) for x in (0.5, 0.9, 0.99)]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--au-csv", type=Path, required=True)
    parser.add_argument("--snoek-zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = {"checks": checks(), "eu_emotion_natural_clips": eu_spectra(args.au_csv),
              "snoek_randomised_design": snoek_spectra(args.snoek_zip)}
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
