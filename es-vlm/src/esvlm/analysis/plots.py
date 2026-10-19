"""Small matplotlib helpers. Figures in results/aggregate/ may be committed."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_alpha_accuracy(
    alphas: Sequence[float],
    series: dict[str, Sequence[float]],
    path: Path,
    *,
    title: str = "Accuracy vs alpha",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, ys in series.items():
        ax.plot(list(alphas), list(ys), marker="o", label=name)
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_layer_series(
    layers: Sequence[int],
    series: dict[str, Sequence[float]],
    path: Path,
    *,
    ylabel: str,
    title: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, ys in series.items():
        ax.plot(list(layers), list(ys), marker="o", label=name)
    ax.set_xlabel("layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_separability(matrix: np.ndarray, labels: Sequence[str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(list(labels), rotation=45, ha="right")
    ax.set_yticklabels(list(labels))
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
