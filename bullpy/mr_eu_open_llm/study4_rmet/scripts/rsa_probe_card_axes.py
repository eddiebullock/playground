"""
Phase 3 — RSA vs CARD feature targets + linear probes for diagnosticity vs entropy.

Uses existing activation .npy under results/activations/{model}/full/.
Does not modify parent probing.py; reimplements a small Ridge probe locally.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

STUDY4_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from alignment_analyses import compute_rdm_from_vectors, perm_rsa, upper_tri  # noqa: E402

DEFAULT_STRUCT = STUDY4_ROOT / "results" / "card_structure"
DEFAULT_OUT = STUDY4_ROOT / "results" / "mech"
_LAYER_RE = re.compile(r"layer(\d+)_rmet_seed\d+\.npy$")


def list_layers(act_dir: Path) -> List[Tuple[int, Path]]:
    files = sorted(act_dir.glob("layer*_rmet_seed*.npy"))
    out = []
    for p in files:
        m = _LAYER_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return out


def spearman_rsa(a: np.ndarray, b: np.ndarray) -> float:
    rho, _ = spearmanr(upper_tri(a), upper_tri(b))
    return float(rho)


def loo_ridge_r(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Leave-one-out Ridge: predicted vs true Spearman rho (n=36)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mask = np.isfinite(y)
    X, y = X[mask], y[mask]
    n = len(y)
    if n < 8:
        return {"rho": float("nan"), "n": float(n)}
    preds = np.zeros(n)
    loo = LeaveOneOut()
    for train, test in loo.split(X):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X[train])
        Xv = scaler.transform(X[test])
        model = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0))
        model.fit(Xt, y[train])
        preds[test] = model.predict(Xv)
    rho, _ = spearmanr(y, preds)
    return {"rho": float(rho), "n": float(n), "rmse": float(np.sqrt(np.mean((y - preds) ** 2)))}


def run_model(
    model: str,
    act_dir: Path,
    struct: pd.DataFrame,
    feat_rdm: np.ndarray,
    entropy_rdm: np.ndarray,
    diag_rdm: np.ndarray,
    *,
    n_perm: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    layers = list_layers(act_dir)
    if not layers:
        return {"status": "missing_activations", "model": model}

    # Align items 1..36 with activation rows via trial_ids if present
    y_ent = struct.sort_values("item")["human_entropy"].to_numpy(float)
    y_diag = struct.sort_values("item")["trait_diagnosticity_eq_slope"].to_numpy(float)

    layer_rows = []
    for layer_idx, path in layers:
        X = np.load(path)
        if X.shape[0] != 36:
            # try reorder by trial ids
            tid_path = act_dir / f"layer{layer_idx}_trial_ids.json"
            if tid_path.exists():
                tids = json.loads(tid_path.read_text(encoding="utf-8"))
                # expect rmet_XX
                order = []
                for tid in tids:
                    m = re.search(r"(\d+)", str(tid))
                    order.append(int(m.group(1)) if m else -1)
                # map to item 1..36 order
                idx = [order.index(i) for i in range(1, 37)]
                X = X[idx]
            else:
                return {"status": "shape_mismatch", "model": model, "shape": list(X.shape)}

        act_rdm = compute_rdm_from_vectors(X)
        rsa_feat = perm_rsa(feat_rdm, act_rdm, n_perm=n_perm, seed=seed)
        rsa_ent = perm_rsa(entropy_rdm, act_rdm, n_perm=n_perm, seed=seed)
        rsa_diag = perm_rsa(diag_rdm, act_rdm, n_perm=n_perm, seed=seed)
        probe_ent = loo_ridge_r(X, y_ent)
        probe_diag = loo_ridge_r(X, y_diag)
        layer_rows.append(
            {
                "model": model,
                "layer": layer_idx,
                "rsa_card_feature_rho": rsa_feat["rho"],
                "rsa_card_feature_p": rsa_feat["p_perm"],
                "rsa_entropy_rho": rsa_ent["rho"],
                "rsa_entropy_p": rsa_ent["p_perm"],
                "rsa_diagnosticity_rho": rsa_diag["rho"],
                "rsa_diagnosticity_p": rsa_diag["p_perm"],
                "probe_entropy_rho": probe_ent["rho"],
                "probe_diagnosticity_rho": probe_diag["rho"],
            }
        )
        np.save(
            STUDY4_ROOT / "results" / "mech" / f"act_rdm_{model}_layer{layer_idx}.npy",
            act_rdm,
        )

    df = pd.DataFrame(layer_rows)
    # M1 headline: peak |probe| for diagnosticity vs entropy
    peak_diag = df.iloc[df["probe_diagnosticity_rho"].abs().idxmax()].to_dict()
    peak_ent = df.iloc[df["probe_entropy_rho"].abs().idxmax()].to_dict()
    return {
        "status": "ok",
        "model": model,
        "n_layers": len(df),
        "layers": layer_rows,
        "M1_peak_diagnosticity": peak_diag,
        "M1_peak_entropy": peak_ent,
        "note_multiplicity": (
            f"{len(df)} layers tested; treat peak as exploratory without correction; "
            "report full layer table."
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--structure_dir", type=Path, default=DEFAULT_STRUCT)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n_perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--models",
        default="qwen3vl,gemma4,molmo2",
        help="Open-weight models with activations",
    )
    args = ap.parse_args(argv)

    struct = pd.read_csv(args.structure_dir / "item_card_structure.csv")
    feat_rdm = np.load(args.structure_dir / "human_card_feature_rdm.npy")
    entropy_rdm = np.load(args.structure_dir / "human_entropy_rdm.npy")
    diag_rdm = np.load(args.structure_dir / "human_diagnosticity_rdm.npy")

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {"models": {}}
    all_layers = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        act_dir = STUDY4_ROOT / "results" / "activations" / model / "full"
        print(f"mech rsa/probe: {model} ({act_dir})", flush=True)
        res = run_model(
            model,
            act_dir,
            struct,
            feat_rdm,
            entropy_rdm,
            diag_rdm,
            n_perm=args.n_perm,
            seed=args.seed,
        )
        summary["models"][model] = {k: v for k, v in res.items() if k != "layers"}
        if res.get("status") == "ok":
            all_layers.extend(res["layers"])
            pd.DataFrame(res["layers"]).to_csv(
                args.outdir / f"{model}_rsa_probe_layers.csv", index=False
            )

    if all_layers:
        pd.DataFrame(all_layers).to_csv(args.outdir / "all_rsa_probe_layers.csv", index=False)
    (args.outdir / "mech_rsa_probe_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
