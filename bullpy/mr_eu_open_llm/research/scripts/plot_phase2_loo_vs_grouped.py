#!/usr/bin/env python3
"""Scatter LOO vs actor-grouped T1 rho (actor leakage).

Usage (conda env mr_eu_open_llm):

  python research/scripts/plot_phase2_loo_vs_grouped.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "research" / "reproduction" / "phase2_all_layers_summary.csv"
OUT_PATH = ROOT / "research" / "figures" / "phase2_loo_vs_grouped.png"

MODELS = ("qwen3vl", "gemma4", "molmo2")
LABELS = {"qwen3vl": "Qwen3-VL-8B", "gemma4": "Gemma 4 E4B", "molmo2": "Molmo2-O-7B"}
COLOURS = {"qwen3vl": "#1f77b4", "gemma4": "#ff7f0e", "molmo2": "#2ca02c"}


def main() -> None:
    with CSV_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 0.65], [0, 0.65], color="#888888", ls="--", lw=1.0, label="LOO = grouped")
    for model in MODELS:
        sub = [r for r in rows if r["model"] == model]
        ax.scatter(
            [float(r["t1_grouped_rho"]) for r in sub],
            [float(r["t1_loo_rho"]) for r in sub],
            s=18,
            color=COLOURS[model],
            label=LABELS[model],
            alpha=0.85,
        )
    ax.set_xlabel("grouped T1 rho (leave-one-actor-out)")
    ax.set_ylabel("LOO T1 rho")
    ax.set_title("Actor leakage: LOO exceeds grouped CV")
    ax.set_xlim(0.05, 0.60)
    ax.set_ylim(0.20, 0.70)
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
