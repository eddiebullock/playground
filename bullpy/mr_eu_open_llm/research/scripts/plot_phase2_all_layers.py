#!/usr/bin/env python3
"""Plot grouped T1 rho vs layer from the official all-layer Phase 2 dump.

Usage (conda env mr_eu_open_llm):

  python research/scripts/plot_phase2_all_layers.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "research" / "reproduction" / "phase2_all_layers_summary.csv"
OUT_PATH = ROOT / "research" / "figures" / "phase2_grouped_t1_rho.png"

MODELS = ("qwen3vl", "gemma4", "molmo2")
LABELS = {"qwen3vl": "Qwen3-VL-8B", "gemma4": "Gemma 4 E4B", "molmo2": "Molmo2-O-7B"}
COLOURS = {"qwen3vl": "#1f77b4", "gemma4": "#ff7f0e", "molmo2": "#2ca02c"}
STIM_RHO = 0.13784107290892722


def load_rows():
    with CSV_PATH.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    rows = load_rows()
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.axhline(STIM_RHO, color="#555555", ls="--", lw=1.0, label="stimulus baseline")
    for model in MODELS:
        sub = [r for r in rows if r["model"] == model]
        xs = [int(r["layer"]) for r in sub]
        ys = [float(r["t1_grouped_rho"]) for r in sub]
        holm_only = [r["holm_only_fail"] == "True" for r in sub]
        ax.plot(xs, ys, color=COLOURS[model], lw=1.4, label=LABELS[model])
        ax.scatter(
            [x for x, h in zip(xs, holm_only) if h],
            [y for y, h in zip(ys, holm_only) if h],
            s=22,
            color=COLOURS[model],
            zorder=3,
        )
        ax.scatter(
            [x for x, h in zip(xs, holm_only) if not h],
            [y for y, h in zip(ys, holm_only) if not h],
            s=16,
            facecolors="none",
            edgecolors=COLOURS[model],
            zorder=3,
        )
    ax.set_xlabel("layer (last token)")
    ax.set_ylabel("grouped T1 Spearman rho")
    ax.set_title("Official Phase 2: human-entropy probe (leave-one-actor-out)")
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([0], [0], marker="o", color="0.3", ls="none", markersize=5, label="fails only Holm"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="0.3",
            ls="none",
            markersize=5,
            markerfacecolor="none",
            label="fails another check",
        ),
    ]
    ax.legend(handles=handles, labels=labels + ["fails only Holm", "fails another check"], frameon=False, loc="lower right")
    ax.set_ylim(-0.05, 0.62)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
