"""
Steps 4–6 — behavioural / RSA / confusion alignment (study4 only).

Step 4 (A1): correlate human per-item trait-sensitivity with model accuracy/entropy
             (n=36); permutation p-values.
Step 5 (A2): build 36×36 RDMs (human trait-sensitivity; model activations if provided)
             and RSA Spearman + permutation. Activation extraction is a separate HPC job;
             this script can also build a behavioural model RDM from per-item accuracy.
Step 6 (A3): compare low-EQ human incorrect-choice distributions vs model samples
             (KL / chi-square) per item and aggregated.

Does not modify study3 code. Uses cosine-distance RDM + Spearman upper-triangle
(same metric family as study3 scripts/rsa.py, reimplemented here to avoid coupling).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, entropy as scipy_entropy, spearmanr

STUDY4_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HUMAN = STUDY4_ROOT / "results" / "human" / "item_trait_sensitivity.csv"
DEFAULT_CARD = STUDY4_ROOT / "data" / "processed" / "card_rmet_item_level.csv"
DEFAULT_KEY = STUDY4_ROOT / "data" / "rmet" / "answer_key" / "rmet_adult_answer_key.json"
DEFAULT_OUT = STUDY4_ROOT / "results" / "alignment"


def compute_rdm_from_vectors(x: np.ndarray) -> np.ndarray:
    """Cosine distance RDM from row vectors."""
    x = x.astype(np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, 1e-12)
    sim = x @ x.T
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float32)


def rdm_from_scalar(values: np.ndarray) -> np.ndarray:
    """Pairwise absolute difference RDM from a 36-vector (e.g. trait-sensitivity)."""
    v = values.astype(np.float64).reshape(-1, 1)
    d = np.abs(v - v.T)
    np.fill_diagonal(d, 0.0)
    return d.astype(np.float32)


def upper_tri(rdm: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(rdm.shape[0], k=1)
    return rdm[iu]


def spearman_rsa(a: np.ndarray, b: np.ndarray) -> float:
    rho, _ = spearmanr(upper_tri(a), upper_tri(b))
    return float(rho)


def perm_spearman(x: np.ndarray, y: np.ndarray, n_perm: int = 5000, seed: int = 42) -> Dict[str, float]:
    """Permutation test shuffling y (item labels)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 5:
        return {"rho": float("nan"), "p_perm": float("nan"), "n": float(len(x))}
    rho_obs, _ = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        yp = rng.permutation(y)
        null[i], _ = spearmanr(x, yp)
    # two-sided
    p = (np.sum(np.abs(null) >= abs(rho_obs)) + 1) / (n_perm + 1)
    return {"rho": float(rho_obs), "p_perm": float(p), "n": float(len(x))}


def perm_rsa(rdm_a: np.ndarray, rdm_b: np.ndarray, n_perm: int = 5000, seed: int = 42) -> Dict[str, float]:
    rho_obs = spearman_rsa(rdm_a, rdm_b)
    rng = np.random.default_rng(seed)
    n = rdm_a.shape[0]
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm = rng.permutation(n)
        null[i] = spearman_rsa(rdm_a, rdm_b[np.ix_(perm, perm)])
    p = (np.sum(np.abs(null) >= abs(rho_obs)) + 1) / (n_perm + 1)
    return {"rho": float(rho_obs), "p_perm": float(p), "n_items": float(n)}


def load_model_item_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for t in data["trials"]:
        rows.append(
            {
                "item": t["item"],
                "det_correct": int(t["deterministic"]["correct"]),
                "sample_accuracy": t["samples"]["accuracy"],
                "sample_entropy": t["samples"]["entropy"],
                "det_prediction": t["deterministic"]["prediction"],
                "sample_predictions": t["samples"]["predictions"],
                "options": t["options"],
                "correct_label": t["correct_label"],
            }
        )
    return pd.DataFrame(rows)


def step4_behavioural(
    human: pd.DataFrame,
    model_tables: Dict[str, pd.DataFrame],
    *,
    n_perm: int = 5000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    summary: Dict[str, Any] = {"per_model": {}, "pooled": {}}
    pooled_h = []
    pooled_a = []
    for model, mdf in model_tables.items():
        merged = human.merge(mdf, on="item", how="inner")
        for metric in ("det_correct", "sample_accuracy", "sample_entropy"):
            if metric not in merged.columns:
                continue
            # For det_correct treat as 0/1 accuracy
            y = pd.to_numeric(merged[metric], errors="coerce").to_numpy()
            # Entropy: higher = less confident; correlate with trait-sensitivity as-is
            x = merged["trait_sensitivity_coef"].to_numpy()
            # Protocol: trait-sensitivity vs model accuracy — for entropy, also report
            res = perm_spearman(x, y, n_perm=n_perm, seed=seed)
            rows.append({"model": model, "metric": metric, **res})
            summary["per_model"].setdefault(model, {})[metric] = res
            if metric in ("det_correct", "sample_accuracy"):
                pooled_h.append(x)
                pooled_a.append(y)
        # Headline: trait-sensitivity vs deterministic accuracy
    if pooled_h:
        ph = np.concatenate(pooled_h)
        pa = np.concatenate(pooled_a)
        summary["pooled"]["trait_sens_vs_accuracy"] = perm_spearman(ph, pa, n_perm=n_perm, seed=seed)
    return pd.DataFrame(rows), summary


def step5_rsa(
    human: pd.DataFrame,
    model_tables: Dict[str, pd.DataFrame],
    activation_rdms: Optional[Dict[str, np.ndarray]] = None,
    *,
    n_perm: int = 5000,
    seed: int = 42,
    outdir: Path,
) -> Dict[str, Any]:
    # Human difficulty RDM from trait-sensitivity coefficients
    human_sorted = human.sort_values("item")
    hvec = human_sorted["trait_sensitivity_coef"].to_numpy(dtype=np.float64)
    human_rdm = rdm_from_scalar(hvec)
    np.save(outdir / "human_trait_sensitivity_rdm.npy", human_rdm)

    out: Dict[str, Any] = {"human_rdm": str(outdir / "human_trait_sensitivity_rdm.npy"), "models": {}}
    for model, mdf in model_tables.items():
        m = mdf.sort_values("item")
        # Behavioural proxy RDM from sample accuracy (until activations available)
        avec = pd.to_numeric(m.get("sample_accuracy", m["det_correct"]), errors="coerce").to_numpy()
        model_rdm = rdm_from_scalar(avec)
        np.save(outdir / f"model_{model}_accuracy_rdm.npy", model_rdm)
        rsa = perm_rsa(human_rdm, model_rdm, n_perm=n_perm, seed=seed)
        entry = {"accuracy_rdm_rsa": rsa, "activation_rsa": None}
        if activation_rdms and model in activation_rdms:
            act_rdm = activation_rdms[model]
            np.save(outdir / f"model_{model}_activation_rdm.npy", act_rdm)
            entry["activation_rsa"] = perm_rsa(human_rdm, act_rdm, n_perm=n_perm, seed=seed)
        out["models"][model] = entry
    return out


def _choice_dist_from_predictions(preds: Sequence[Optional[str]], options: Sequence[str]) -> np.ndarray:
    c = Counter(p for p in preds if p is not None)
    arr = np.array([c.get(o, 0) for o in options], dtype=np.float64)
    if arr.sum() == 0:
        return arr
    return arr / arr.sum()


def step6_confusion(
    card_csv: Path,
    key_path: Path,
    model_jsons: Dict[str, Path],
    *,
    eq_tertile: str = "low",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    key = json.loads(key_path.read_text(encoding="utf-8"))
    items = {int(it["item"]): it for it in key["items"]}
    df = pd.read_csv(card_csv)
    eq = pd.to_numeric(df["eq_total"], errors="coerce")
    valid = df.loc[eq.notna()].copy()
    valid["eq_tertile"] = pd.qcut(eq.loc[eq.notna()], 3, labels=["low", "mid", "high"], duplicates="drop")
    low = valid[valid["eq_tertile"] == eq_tertile]

    rows = []
    agg = {}
    for model, path in model_jsons.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        kls = []
        chis = []
        for t in data["trials"]:
            item = int(t["item"])
            options = list(t["options"])
            correct = t["correct_label"]
            # Human incorrect choices among low-EQ
            human_counts = Counter()
            col = f"rmet_{item:02d}_choice"
            corr_col = f"rmet_{item:02d}_correct"
            for _, r in low.iterrows():
                if pd.isna(r.get(corr_col)) or int(r[corr_col]) == 1:
                    continue
                ch = r.get(col)
                try:
                    ch_i = int(float(ch))
                except Exception:
                    continue
                if 1 <= ch_i <= 4:
                    human_counts[options[ch_i - 1]] += 1
            # Model incorrect from samples
            model_counts = Counter()
            for p in t["samples"]["predictions"]:
                if p is None:
                    continue
                if str(p).lower() == str(correct).lower():
                    continue
                model_counts[p] += 1

            h = np.array([human_counts.get(o, 0) for o in options], dtype=np.float64)
            m = np.array([model_counts.get(o, 0) for o in options], dtype=np.float64)
            # Drop correct option index for confusion comparison
            corr_idx = options.index(correct) if correct in options else None
            if corr_idx is not None:
                mask = np.ones(len(options), dtype=bool)
                mask[corr_idx] = False
                h2, m2 = h[mask], m[mask]
                opt2 = [o for i, o in enumerate(options) if i != corr_idx]
            else:
                h2, m2, opt2 = h, m, options

            kl = float("nan")
            chi_p = float("nan")
            if h2.sum() > 0 and m2.sum() > 0:
                hp = h2 / h2.sum()
                mp = m2 / m2.sum()
                # Add smoothing
                hp = (hp + 1e-6) / (hp + 1e-6).sum()
                mp = (mp + 1e-6) / (mp + 1e-6).sum()
                kl = float(scipy_entropy(hp, mp))
                table = np.vstack([h2, m2])
                if table.sum() > 0 and (table > 0).any():
                    try:
                        _, chi_p, _, _ = chi2_contingency(table + 0.5)
                        chi_p = float(chi_p)
                    except Exception:
                        chi_p = float("nan")
                kls.append(kl)
                chis.append(chi_p)

            rows.append(
                {
                    "model": model,
                    "item": item,
                    "kl_lowEQ_vs_model": kl,
                    "chi2_p": chi_p,
                    "human_incorrect_n": int(h2.sum()) if h2 is not None else 0,
                    "model_incorrect_n": int(m2.sum()) if m2 is not None else 0,
                }
            )
        agg[model] = {
            "mean_kl": float(np.nanmean(kls)) if kls else None,
            "median_kl": float(np.nanmedian(kls)) if kls else None,
            "n_items_scored": len(kls),
        }
    return pd.DataFrame(rows), agg


def maybe_plot_step4(
    human: pd.DataFrame,
    model_tables: Dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    models = list(model_tables.keys())
    fig, axes = plt.subplots(1, max(1, len(models)), figsize=(4 * max(1, len(models)), 3.5), squeeze=False)
    for ax, model in zip(axes[0], models):
        m = human.merge(model_tables[model], on="item")
        y = pd.to_numeric(m.get("sample_accuracy", m["det_correct"]), errors="coerce")
        ax.scatter(m["trait_sensitivity_coef"], y, s=28, c="#222")
        ax.set_xlabel("Human EQ trait-sensitivity")
        ax.set_ylabel("Model accuracy")
        ax.set_title(model)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def discover_model_evals(results_model_dir: Path) -> Dict[str, Path]:
    """
    Prefer full evals; fall back to largest rmet_eval_*.json per model folder.
    Covers open-weight (qwen3vl/…) and commercial API (gpt5/claude_opus/gemini_flash).
    """
    found: Dict[str, Path] = {}
    if not results_model_dir.is_dir():
        return found
    for model_dir in sorted(p for p in results_model_dir.iterdir() if p.is_dir()):
        full = sorted(model_dir.glob("rmet_eval_*_full_seed*.json"))
        if full:
            found[model_dir.name] = full[-1]
            continue
        any_json = sorted(model_dir.glob("rmet_eval_*.json"))
        if any_json:
            # Prefer most items by file size as a cheap proxy
            found[model_dir.name] = max(any_json, key=lambda p: p.stat().st_size)
    return found


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--human_csv", type=Path, default=DEFAULT_HUMAN)
    ap.add_argument("--card_csv", type=Path, default=DEFAULT_CARD)
    ap.add_argument("--answer_key", type=Path, default=DEFAULT_KEY)
    ap.add_argument(
        "--model_eval",
        action="append",
        default=[],
        help="model_key=/path/to/rmet_eval_*.json (repeatable). "
        "If omitted, auto-discovers study4_rmet/results/model/*/rmet_eval_*_full_*.json",
    )
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n_perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if not args.human_csv.exists():
        raise SystemExit(f"Missing human Step-2 table: {args.human_csv} (run human_item_difficulty.py first)")

    human = pd.read_csv(args.human_csv)
    model_tables: Dict[str, pd.DataFrame] = {}
    model_jsons: Dict[str, Path] = {}
    specs = list(args.model_eval)
    if not specs:
        auto = discover_model_evals(STUDY4_ROOT / "results" / "model")
        specs = [f"{k}={v}" for k, v in auto.items()]
        if specs:
            print(f"Auto-discovered {len(specs)} model evals under results/model/")
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"Bad --model_eval {spec}; expected model=/path/to/json")
        mk, path_s = spec.split("=", 1)
        path = Path(path_s)
        model_tables[mk] = load_model_item_table(path)
        if path.suffix.lower() == ".json":
            model_jsons[mk] = path

    args.outdir.mkdir(parents=True, exist_ok=True)

    if model_tables:
        s4_table, s4_sum = step4_behavioural(human, model_tables, n_perm=args.n_perm, seed=args.seed)
        s4_table.to_csv(args.outdir / "a1_behavioural_alignment.csv", index=False)
        (args.outdir / "a1_summary.json").write_text(json.dumps(s4_sum, indent=2) + "\n")
        maybe_plot_step4(human, model_tables, args.outdir / "figures" / "a1_trait_vs_model_acc.png")

        s5 = step5_rsa(human, model_tables, n_perm=args.n_perm, seed=args.seed, outdir=args.outdir)
        (args.outdir / "a2_rsa_summary.json").write_text(json.dumps(s5, indent=2) + "\n")

        if model_jsons:
            s6_table, s6_agg = step6_confusion(args.card_csv, args.answer_key, model_jsons)
            s6_table.to_csv(args.outdir / "a3_confusion_alignment.csv", index=False)
            (args.outdir / "a3_summary.json").write_text(json.dumps(s6_agg, indent=2) + "\n")
        print(json.dumps({"a1": s4_sum, "a2": s5}, indent=2))
    else:
        # Still write human RDM for later
        human_sorted = human.sort_values("item")
        hvec = human_sorted["trait_sensitivity_coef"].to_numpy(dtype=np.float64)
        human_rdm = rdm_from_scalar(hvec)
        np.save(args.outdir / "human_trait_sensitivity_rdm.npy", human_rdm)
        print(f"No model evals yet; wrote human RDM -> {args.outdir / 'human_trait_sensitivity_rdm.npy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
