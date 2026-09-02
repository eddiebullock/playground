#!/usr/bin/env python3
"""Paper figures for Study 3: mental-state ambiguity / confusability in VLMs.

Outputs under results/figures/study3/ (PNG + PDF):
  fig1_behaviour.*       accuracy, entropy alignment, consensus calibration
  fig2_geometry.*        probe + RSA by layer (3 models)
  fig3_steer.*           Qwen L4 exploratory steer Delta-JS (pilot)
  fig4_axis_geometry.*   activation-space own-effect / reuse summary
  fig5_ablation_dissociation.*  v2 primary causal: ablation double-dissociation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "study3"

MODELS = [
    ("qwen3vl", "Qwen3-VL", "#1f4e79"),
    ("gemma4", "Gemma 4", "#2a6f4e"),
    ("molmo2", "Molmo2", "#8b4513"),
]
CHANCE = 1.0 / 6.0


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


def _load_accuracy() -> Dict[str, float]:
    out = {}
    for key, _, _ in MODELS:
        p = ROOT / "results" / "baseline" / "eu_emotions" / key / f"eval_v2_eu_emotions_{key}_video_only_seed42.json"
        d = json.loads(p.read_text())
        out[key] = float(d["accuracy"])
    return out


def _load_entropy_alignment() -> Tuple[Dict[str, float], Dict[str, List[Dict[str, Any]]]]:
    rows = json.loads((ROOT / "results" / "stats" / "rq1_1b_entropy_alignment.json").read_text())
    rho = {}
    trials = {}
    for r in rows:
        rho[r["model"]] = float(r["rq1_1b_forced_choice_vs_human"]["spearman_rho"])
        trials[r["model"]] = list(r["trials"])
    return rho, trials


def _load_calibration() -> Dict[str, Dict[str, float]]:
    rows = json.loads((ROOT / "results" / "stats" / "human_calibration.json").read_text())
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        t = r["terciles"]
        out[r["model"]] = {
            "high_consensus": float(t["high_consensus"]["accuracy"]),
            "mid": float(t["mid"]["accuracy"]),
            "high_disagreement": float(t["high_disagreement"]["accuracy"]),
        }
    return out


def fig1_behaviour() -> None:
    acc = _load_accuracy()
    rho, trials = _load_entropy_alignment()
    cal = _load_calibration()

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), constrained_layout=True)

    # A: accuracy
    ax = axes[0]
    keys = [m[0] for m in MODELS]
    labels = [m[1] for m in MODELS]
    colors = [m[2] for m in MODELS]
    vals = [acc[k] * 100 for k in keys]
    bars = ax.bar(labels, vals, color=colors, width=0.65, edgecolor="none")
    ax.axhline(CHANCE * 100, color="#666666", ls="--", lw=1, label=f"chance ({CHANCE*100:.1f}%)")
    ax.set_ylabel("6AFC accuracy (%)")
    ax.set_ylim(0, 55)
    ax.set_title("A. Recognition accuracy")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)

    # B: entropy scatter for Qwen (primary) with rho annotation strip for all
    ax = axes[1]
    q = trials["qwen3vl"]
    hx = [float(t["human_entropy"]) for t in q if t.get("forced_choice_entropy") is not None]
    my = [float(t["forced_choice_entropy"]) for t in q if t.get("forced_choice_entropy") is not None]
    ax.scatter(hx, my, s=14, alpha=0.45, c=MODELS[0][2], edgecolors="none")
    # identity-ish guide
    lim = max(max(hx), max(my), 1.0)
    ax.plot([0, lim], [0, lim], color="#bbbbbb", lw=0.8, ls=":")
    ax.set_xlabel("Human response entropy")
    ax.set_ylabel("Model 6AFC entropy")
    ax.set_title("B. Entropy alignment (Qwen3-VL)")
    rho_txt = "\n".join(f"{lab}: ρ = {rho[k]:+.2f}" for k, lab, _ in MODELS)
    ax.text(
        0.98,
        0.04,
        rho_txt,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#dddddd", alpha=0.9),
    )

    # C: accuracy by human consensus tercile
    ax = axes[2]
    terciles = ["high_consensus", "mid", "high_disagreement"]
    terc_labels = ["High\nconsensus", "Mid", "High\ndisagreement"]
    x = np.arange(len(terciles))
    width = 0.24
    for i, (key, lab, col) in enumerate(MODELS):
        ys = [cal[key][t] * 100 for t in terciles]
        ax.bar(x + (i - 1) * width, ys, width=width, color=col, label=lab, edgecolor="none")
    ax.axhline(CHANCE * 100, color="#666666", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(terc_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 70)
    ax.set_title("C. Accuracy by human consensus")
    ax.legend(loc="upper right", fontsize=8)

    _save(fig, "fig1_behaviour")


def fig2_geometry() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)

    for ax, metric, title, ylabel, ylim in [
        (
            axes[0],
            "probe_confusability_rho",
            "A. Linear probe: human confusability",
            "Spearman ρ (LOO Ridge)",
            (0, 0.75),
        ),
        (
            axes[1],
            "rsa_human_confusion_rho",
            "B. RSA vs human confusion RDM",
            "Spearman ρ",
            (-0.15, 0.15),
        ),
    ]:
        for key, lab, col in MODELS:
            df = pd.read_csv(ROOT / "results" / "mech" / f"{key}_confusability_probe_rsa.csv")
            # x as relative depth rank 0,1,2 for comparability across models
            xs = np.arange(len(df))
            ys = df[metric].to_numpy(float)
            ax.plot(xs, ys, "-o", color=col, label=lab, lw=1.8, ms=7)
            if metric.startswith("rsa"):
                for x, y, p in zip(xs, ys, df["rsa_human_confusion_p_perm"]):
                    if float(p) < 0.05:
                        ax.scatter([x], [y], s=90, facecolors="none", edgecolors=col, lw=1.4, zorder=5)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Early", "Mid", "Late"])
        ax.set_xlabel("Layer depth")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.axhline(0.0, color="#999999", lw=0.8)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)

    axes[1].text(
        0.02,
        0.02,
        "open circle: p < 0.05 (perm)",
        transform=axes[1].transAxes,
        fontsize=7,
        color="#555555",
        va="bottom",
    )
    _save(fig, "fig2_geometry")


def fig3_steer() -> None:
    df = pd.read_csv(ROOT / "results" / "mech" / "steer_summary_qwen3vl_layer4.csv")
    # keep α = -1 and +1; order axes
    order = [
        "confusability",
        "entropy",
        "random",
        "pair_bored_unfriendly",
        "pair_interested_kind",
        "pair_disappointed_worried",
    ]
    short = {
        "confusability": "Confusability",
        "entropy": "Entropy",
        "random": "Random",
        "pair_bored_unfriendly": "Bored–Unfriendly",
        "pair_interested_kind": "Interested–Kind",
        "pair_disappointed_worried": "Disappointed–Worried",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True, gridspec_kw={"width_ratios": [1.35, 1]})

    # A: global mean delta JS
    ax = axes[0]
    x = np.arange(len(order))
    w = 0.38
    neg = [float(df[(df.axis == a) & (df.alpha == -1.0)]["mean_delta_js_human"].iloc[0]) for a in order]
    pos = [float(df[(df.axis == a) & (df.alpha == 1.0)]["mean_delta_js_human"].iloc[0]) for a in order]
    ax.bar(x - w / 2, neg, width=w, color="#3d5a80", label="α = −1", edgecolor="none")
    ax.bar(x + w / 2, pos, width=w, color="#ee6c4d", label="α = +1", edgecolor="none")
    ax.axhline(0.0, color="#333333", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([short[a] for a in order], rotation=25, ha="right")
    ax.set_ylabel("Mean ΔJS(model, human)\n(negative = closer to human)")
    ax.set_title("A. Qwen3-VL L4 steer (36 trials, last-token)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(-0.02, 0.06)

    # B: own-pair vs other-pairs for pair axes at α=-1 (best directional shot)
    ax = axes[1]
    pair_axes = [
        ("pair_bored_unfriendly", "own_pair_bored_unfriendly_mean_delta_js"),
        ("pair_interested_kind", "own_pair_interested_kind_mean_delta_js"),
        ("pair_disappointed_worried", "own_pair_disappointed_worried_mean_delta_js"),
    ]
    x = np.arange(len(pair_axes))
    own_vals, oth_vals = [], []
    for axis, own_col in pair_axes:
        row = df[(df.axis == axis) & (df.alpha == -1.0)].iloc[0]
        own_vals.append(float(row[own_col]))
        oth_vals.append(float(row["reuse_other_pairs_mean_delta_js"]))
    ax.bar(x - w / 2, own_vals, width=w, color="#1f4e79", label="Own pair trials", edgecolor="none")
    ax.bar(x + w / 2, oth_vals, width=w, color="#9aa5b1", label="Other confused pairs", edgecolor="none")
    ax.axhline(0.0, color="#333333", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([short[a] for a, _ in pair_axes], rotation=20, ha="right")
    ax.set_ylabel("Mean ΔJS (α = −1)")
    ax.set_title("B. Pair-axis reuse test")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-0.06, 0.08)

    _save(fig, "fig3_steer")


def fig4_axis_geometry() -> None:
    d = json.loads((ROOT / "results" / "mech" / "qwen3vl_eu_causal_axes_layer4.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)

    # A: generic reuse / alignment summary
    ax = axes[0]
    labels = [
        "Conf ↔ Entropy\nalignment",
        "Conf axis on\nentropy tertiles",
        "Entropy axis on\nconfusability tertiles",
        "Own effect\n(confusability)",
        "Own effect\n(entropy)",
    ]
    vals = [
        d["axis_alignment_conf_vs_entropy"],
        d["reuse_conf_axis_on_entropy_tertiles"],
        d["reuse_entropy_axis_on_confusability_tertiles"],
        d["own_effect_confusability"],
        d["own_effect_entropy"],
    ]
    colors = ["#1f4e79", "#3d5a80", "#3d5a80", "#2a6f4e", "#2a6f4e"]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, edgecolor="none", height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Effect size (activation space)")
    ax.set_xlim(0, 1.05)
    ax.set_title("A. Generic ambiguity axis (Qwen L4)")
    for yi, v in zip(y, vals):
        ax.text(v + 0.02, yi, f"{v:.2f}", va="center", fontsize=8)

    # B: pair-axis own vs rest
    ax = axes[1]
    pairs = d["pair_axes"]
    names = [f"{p['label_a']}–{p['label_b']}\n(n={p['n_pair_trials']})" for p in pairs]
    own = [float(p["own_effect_pair_vs_rest"]) for p in pairs]
    reuse = [float(p["reuse_on_entropy_high"]) for p in pairs]
    x = np.arange(len(pairs))
    w = 0.35
    ax.bar(x - w / 2, own, width=w, color="#1f4e79", label="Own pair vs rest", edgecolor="none")
    ax.bar(x + w / 2, reuse, width=w, color="#c9a227", label="Reuse on high-entropy", edgecolor="none")
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Activation-space effect")
    ax.set_title("B. Pair axes: specificity vs reuse")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-0.2, 1.0)

    _save(fig, "fig4_axis_geometry")


def fig5_ablation_dissociation() -> None:
    summary_path = ROOT / "results" / "mech" / "ablate_summary_qwen3vl_layer4.csv"
    if not summary_path.exists():
        print(f"skip fig5: missing {summary_path}")
        return

    df = pd.read_csv(summary_path)
    pair_axes = [
        c for c in df["axis"].tolist()
        if str(c).startswith("pair_") and c in df["axis"].values
    ]
    generic = [a for a in ("entropy", "random") if a in set(df["axis"])]

    short = {
        "entropy": "Entropy",
        "random": "Random",
        "pair_bored_unfriendly": "Bored–Unfriendly",
        "pair_interested_kind": "Interested–Kind",
        "pair_disappointed_worried": "Disappointed–Worried",
    }
    order = [a for a in generic + pair_axes if a in set(df["axis"])]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)

    ax = axes[0]
    x = np.arange(len(order))
    w = 0.35
    acc = [float(df[df.axis == a]["mean_delta_accuracy"].iloc[0]) for a in order]
    ax.bar(x, acc, width=0.65, color="#1f4e79", edgecolor="none")
    ax.axhline(0.0, color="#333333", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([short.get(a, a) for a in order], rotation=20, ha="right")
    ax.set_ylabel("Mean Δaccuracy (ablated − baseline)")
    ax.set_title("A. Global ablation effect (Qwen L4)")

    ax = axes[1]
    pair_only = [a for a in order if a.startswith("pair_")]
    x = np.arange(len(pair_only))
    own_acc, oth_acc = [], []
    for axis in pair_only:
        row = df[df.axis == axis].iloc[0]
        pk = axis
        own_col = f"own_{pk}_mean_delta_accuracy"
        own_acc.append(float(row[own_col]) if own_col in row and pd.notna(row[own_col]) else 0.0)
        oth_acc.append(
            float(row["reuse_other_pairs_mean_delta_accuracy"])
            if pd.notna(row.get("reuse_other_pairs_mean_delta_accuracy"))
            else 0.0
        )
    ax.bar(x - w / 2, own_acc, width=w, color="#1f4e79", label="Own pair trials", edgecolor="none")
    ax.bar(x + w / 2, oth_acc, width=w, color="#9aa5b1", label="Other confused pairs", edgecolor="none")
    if "random" in set(df["axis"]):
        rand_acc = float(df[df.axis == "random"]["mean_delta_accuracy"].iloc[0])
        ax.axhline(rand_acc, color="#c9a227", ls="--", lw=1, label=f"Random axis ({rand_acc:+.3f})")
    ax.axhline(0.0, color="#333333", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([short.get(a, a) for a in pair_only], rotation=20, ha="right")
    ax.set_ylabel("Mean Δaccuracy")
    ax.set_title("B. Pair-specific vs generic (ablation)")
    ax.legend(loc="best", fontsize=8)

    _save(fig, "fig5_ablation_dissociation")


def fig_entropy_all_models() -> None:
    """Supplementary: entropy scatter for all three models."""
    rho, trials = _load_entropy_alignment()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), constrained_layout=True, sharex=True, sharey=True)
    for ax, (key, lab, col) in zip(axes, MODELS):
        rows = trials[key]
        hx = [float(t["human_entropy"]) for t in rows if t.get("forced_choice_entropy") is not None]
        my = [float(t["forced_choice_entropy"]) for t in rows if t.get("forced_choice_entropy") is not None]
        ax.scatter(hx, my, s=12, alpha=0.4, c=col, edgecolors="none")
        lim = 2.0
        ax.plot([0, lim], [0, lim], color="#bbbbbb", lw=0.8, ls=":")
        ax.set_title(f"{lab}\nρ = {rho[key]:+.3f}")
        ax.set_xlabel("Human entropy")
    axes[0].set_ylabel("Model 6AFC entropy")
    fig.suptitle("Per-item entropy alignment (full EU, 243 trials)", fontsize=11, y=1.02)
    _save(fig, "figS1_entropy_scatter_all")


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is not None:
        OUT = args.out
    _setup_style()
    fig1_behaviour()
    fig2_geometry()
    fig3_steer()
    fig4_axis_geometry()
    fig5_ablation_dissociation()
    fig_entropy_all_models()
    print(f"\nAll figures in {OUT}")


if __name__ == "__main__":
    main()
