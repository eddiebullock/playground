"""
Draft amendment A9: analysis of model responses to Liu renders. See research/PREREG_A9_DRAFT.md.

Every estimating, testing and deciding step is A4's, imported from scripts/a4_analyse.py
unchanged: stimulus_responses, design, estimate (fit_tau, residual and raw CFA, cue-shuffle
null, stimulus bootstrap, split-half R_S, signal test), sensitivities, replicate_check, image_rms,
detector_gate, validity_gate, behaviour_metrics, apply_rule (Holm across the eligible models,
then decide). What is written here replaces a4_analyse.analyse_dataset, which looks up
a4_design.TASK_OF[dataset] and so cannot take A9's dataset name, and adds design B's averaging
over faces, the presence and frame_max sensitivities, and the face-shared residual correlation,
which is descriptive: by the lead decision of 2026-11-09 it attaches no label to any decision. A9 has its own Holm family and never touches an A4 output.

Usage (after the freeze and after all shards finish):
    python -m scripts.a9_analyse --gates gates.json --output results/a9/summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

from scripts import a4_analyse as an  # noqa: E402
from scripts import a9_design as a9  # noqa: E402


def configuration_responses(responses: Mapping[str, np.ndarray], manifest: Sequence[Mapping[str, Any]],
                            design: str) -> Tuple[List[Mapping[str, Any]], np.ndarray, np.ndarray, float, List[np.ndarray]]:
    """Main rows kept, the response matrix, the gender vector for the fit, the lost share, and
    under design B the per-face response matrices (main face first).

    A: one image per configuration, its rendered gender. B: the mean over the configuration's
    four faces, gender 0; a configuration is kept only if all four faces have a complete set."""
    main = [r for r in manifest if r["set"] == "main" and r["dataset"] == a9.DATASET]
    if design == "A":
        kept = [r for r in main if r["stim_id"] in responses]
        _ids, _X, gender = an.design(kept)
        P = np.vstack([responses[r["stim_id"]] for r in kept]) if kept else np.empty((0, len(a9.OPTIONS)))
        return kept, P, gender, 1 - len(kept) / len(main), []
    faces: Dict[str, List[str]] = {}
    for r in manifest:
        if r["set"] in ("main", "all_identities") and r["dataset"] == a9.DATASET:
            faces.setdefault(r["cfg"], []).append(r["stim_id"])
    kept, per_face = [], []
    for r in main:
        ids = faces[r["cfg"]]
        if len(ids) == len(a9.DESIGN_B_GENDERS) and all(s in responses for s in ids):
            kept.append(r)
            per_face.append(np.vstack([responses[s] for s in ids]))
    stack = np.stack(per_face, axis=1) if per_face else np.empty((len(a9.DESIGN_B_GENDERS), 0, len(a9.OPTIONS)))
    P = stack.mean(axis=0) if per_face else np.empty((0, len(a9.OPTIONS)))
    return kept, P, np.zeros(len(kept)), 1 - len(kept) / len(main), list(stack)


def face_shared_correlation(per_face: Sequence[np.ndarray], X_au: np.ndarray) -> Optional[float]:
    """rho_hat: mean correlation, across pairs of faces and over options, of each face's residual
    after an OLS fit of its responses on the AU design. Descriptive. Residuals include any
    non-additive configuration effect, which is shared across faces, so this leans high."""
    if len(per_face) < 2 or len(X_au) < X_au.shape[1] + 2:
        return None
    Z = np.column_stack([np.ones(len(X_au)), X_au])
    H = Z @ np.linalg.pinv(Z)
    resid = [P - H @ P for P in per_face]
    values = []
    for i in range(len(resid)):
        for j in range(i + 1, len(resid)):
            for k in range(resid[i].shape[1]):
                a, b = resid[i][:, k], resid[j][:, k]
                if a.std() > 0 and b.std() > 0:
                    values.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(values)) if values else None


def presence_cfa(P: np.ndarray, X_au: np.ndarray, gender: np.ndarray, lam: float, target: np.ndarray) -> Dict[str, float]:
    tau = an.fit_tau(P, (X_au > 0).astype(float), gender, lam)
    return {"raw": an.cosine(tau, target), "residual": an.cosine(an.remove_common(tau), an.remove_common(target))}


def frame_max_cfa(responses, manifest, kept, human, lam) -> Dict[str, Any]:
    """frame_max renders, fitted on the human peak design of the same configurations."""
    frame = {r["cfg"]: r for r in manifest if r["set"] == "frame_max" and r["dataset"] == a9.DATASET}
    rows = [r for r in kept if r["cfg"] in frame and frame[r["cfg"]]["stim_id"] in responses]
    if not rows:
        return {}
    _ids, X_au, gender = an.design(rows)
    P = np.vstack([responses[frame[r["cfg"]]["stim_id"]] for r in rows])
    tau = an.fit_tau(P, X_au, gender, lam)
    return {"n": len(rows), "raw": an.cosine(tau, human),
            "residual": an.cosine(an.remove_common(tau), an.remove_common(human))}


def analyse_model(rows, manifest, store, ceiling, human_table, design, seed=a9.SEED, **kw) -> Dict[str, Any]:
    """One model on liu_WC. Holm and gates are applied by the caller, as in A4."""
    responses = an.stimulus_responses(rows, a9.TASK)
    kept, P, gender, lost, per_face = configuration_responses(responses, manifest, design)
    _ids, X_au, _g = an.design(kept)
    human, lam = an.human_matrix(store, a9.DATASET)
    if lam != a9.HUMAN_RIDGE:
        raise SystemExit(f"Human ridge {lam} is not the block's {a9.HUMAN_RIDGE}.")
    rng = np.random.default_rng(seed)
    out = an.estimate(P, X_au, gender, lam, human, rng=rng, **kw)
    tau = out.pop("tau")
    block = store["datasets"][a9.DATASET]
    out.update(design=design, n_configurations=len(kept), share_lost=round(lost, 4), lost_ok=lost <= a9.MAX_LOST_SHARE,
               sensitivities=an.sensitivities(tau, human), tau=np.round(tau, 4).tolist())
    out["sensitivities"]["presence"] = presence_cfa(P, X_au, gender, block["presence_sensitivity"]["ridge_lambda_selected"],
                                                    np.array(block["presence_sensitivity"]["do_family_25"]))
    out["sensitivities"]["frame_max"] = frame_max_cfa(responses, manifest, kept, human, lam)
    for name, key in (("raw", "ceiling"), ("residual", "ceiling_residual")):
        b = out[name]
        b["over_ceiling"] = b["cfa"] / (float(ceiling[a9.DATASET][key]) * np.sqrt(max(b["r_s"], 1e-6)))
        b["lower_bound_only"] = not b["r_s"] >= a9.R_S_LOWER_BOUND
    wanted = [r for r in manifest if r["set"] == "main" and r["dataset"] == a9.DATASET]
    first, second, groups = an.replicate_pairs(responses, wanted, manifest, a9.DATASET)
    out["replicate"] = an.replicate_check(first, second, groups, rng) if len(groups) > 1 else {}
    out["P_main_mean"] = P.mean(axis=0).tolist() if len(P) else []
    out["face_shared_residual_correlation"] = face_shared_correlation(per_face, X_au)
    if human_table is not None and len(kept):
        out["behaviour"] = an.behaviour_metrics(P, np.vstack([human_table[r["cfg"]] for r in kept]))
    return out, P


def human_configuration_table() -> Optional[Dict[str, np.ndarray]]:
    """Mean human response per configuration, in A4's label order. Needs the Liu files."""
    try:
        import a9_human_target as ht

        data = ht.load_liu()
    except (SystemExit, FileNotFoundError, OSError):
        return None
    Y = ht.response_labels(data["codes"])
    labels = list(a9.HUMAN_LABELS)
    table = {}
    for i in range(Y.shape[0]):
        counts = np.array([(Y[i] == lab).sum() for lab in labels], dtype=float)
        table[f"liu{i + 1:04d}"] = counts / counts.sum()
    return table


def recovery_choice(path: Path) -> Tuple[str, str, Any]:
    """Returns (statistic, design, r_star); r_star is reported, never used to label a decision."""
    # Statistic and design come from recovery, on this design hash, and must match the block.
    block = json.loads(path.read_text(encoding="utf-8"))
    if block.get("design_sha256") != a9.design_hash():
        raise SystemExit("A9 recovery was not run on this design.")
    choice = block.get("choice", {})
    if choice.get("statistic") != a9.STATISTIC or choice.get("design") != a9.DESIGN_SELECTED:
        raise SystemExit(f"Recovery choice {choice} does not give the block's design and statistic; no A9 analysis.")
    return choice["statistic"], choice["design"], block.get("r_star")


def main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, default=Path("results/a9/manifest.csv"))
    parser.add_argument("--cells_dir", type=Path, default=Path("results/a9"))
    parser.add_argument("--human", type=Path, default=Path("research/reproduction/a9_human_target.json"))
    parser.add_argument("--ceiling", type=Path, default=Path("research/reproduction/a9_human_ceiling.json"))
    parser.add_argument("--recovery", type=Path, default=Path("research/reproduction/a9_recovery.json"))
    parser.add_argument("--gates", type=Path, required=True,
                        help="JSON, per model: a1_passed, a3_passed, g2_g4. Missing means not passed.")
    parser.add_argument("--detector", type=Path, default=Path("research/reproduction/a4_v1_pilot.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/a9/summary.json"))
    args = parser.parse_args()
    a9.check_code_pins(require=True)
    a9.check_pins()
    a9.check_manifest(args.manifest)
    digest = a9.design_hash()
    statistic, design, r_star = recovery_choice(args.recovery)
    store = json.loads(args.human.read_text(encoding="utf-8"))
    ceiling = json.loads(args.ceiling.read_text(encoding="utf-8"))["datasets"]
    manifest = pd.read_csv(args.manifest).to_dict("records")
    recorded = json.loads(args.gates.read_text(encoding="utf-8"))
    detector = an.detector_gate(pd.read_csv(args.detector, keep_default_na=False).to_dict("records"))
    human_table = human_configuration_table()
    models = [m for m in (*a9.MODELS, *a9.DESCRIPTIVE_MODELS) if an.shard_files(args.cells_dir, m)]
    cells = {m: an.read_shards(an.shard_files(args.cells_dir, m), digest) for m in models}
    results: Dict[str, Dict[str, Any]] = {}
    gates: Dict[str, Dict[str, bool]] = {}
    for model in models:
        entry, P = analyse_model(cells[model], manifest, store, ceiling, human_table, design)
        given = recorded.get(model, {})
        gates[model] = {"rq0": an.rq0_gate(given), "g2_g4": given.get("g2_g4") is True,
                        **an.validity_gate(P, entry["replicate"], an.image_rms(cells[model], a9.TASK), detector)}
        results[model] = entry
    an.apply_rule(results, gates, statistic)
    out = {"amendment": "A9", "design_sha256": digest, "code_sha256": a9.own_code_sha256(), "statistic": statistic,
           "design": design, "recovery_r_star_descriptive": r_star,
           "withdrawn_under_correlated_face_noise": list(a9.DESIGN["recovery"]["withdrawn_under_correlated_face_noise"]), "detector": detector, "human_reference": json.loads(
               args.ceiling.read_text(encoding="utf-8"))["human_reference_cfa"],
           "note": "A9's own Holm family. Never changes any A4 verdict.", "models": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    print(args.output, hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
