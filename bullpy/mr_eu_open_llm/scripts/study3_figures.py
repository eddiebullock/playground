"""Study 3 killer figure: behavior, probes, RSA, patching across models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from config import LOCAL_RESULTS_DIR


def _load_comparison(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj.get("models") or []


def plot_study3_figure(models: List[Dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    names = [m["model"] for m in models]
    x = np.arange(len(names))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel A: behavioral delta
    ax = axes[0, 0]
    deltas = [m.get("delta_pp") or 0 for m in models]
    colors = ["#c44e52" if d < -10 else "#55a868" if d > 1 else "#8172b3" for d in deltas]
    ax.bar(x, deltas, color=colors)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Δ 4AFC accuracy (pp)")
    ax.set_title("A. Post-FT EU transfer")

    # Panel B: peak probe accuracy
    ax = axes[0, 1]
    base_probe = []
    ft_probe = []
    for m in models:
        p = m.get("probes") or {}
        base_probe.append(100 * (p.get("baseline_peak_acc") or 0))
        ft_probe.append(100 * (p.get("finetuned_peak_acc") or 0))
    w = 0.35
    ax.bar(x - w / 2, base_probe, w, label="baseline", color="#4c72b0")
    ax.bar(x + w / 2, ft_probe, w, label="finetuned", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Peak-layer probe CV acc (%)")
    ax.set_title("B. Linear probe at peak layer")
    ax.legend()

    # Panel C: RSA mean rho
    ax = axes[1, 0]
    rhos = []
    for m in models:
        r = (m.get("rsa") or {}).get("mean_rho")
        rhos.append(r if r is not None else 0)
    ax.bar(x, rhos, color="#937860")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Mean Spearman ρ (base vs FT)")
    ax.set_title("C. Representational geometry")

    # Panel D: patching fix rate
    ax = axes[1, 1]
    fixes = []
    for m in models:
        p = m.get("patching") or {}
        fixes.append(100 * (p.get("fix_rate") or 0))
    ax.bar(x, fixes, color="#8c8c8c")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Patch fix rate (%)")
    ax.set_title("D. Causal patching at peak layer")

    fig.suptitle("Study 3: MR LoRA boundary conditions across VLMs", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Study 3 four-panel summary figure.")
    ap.add_argument(
        "--input",
        type=Path,
        default=LOCAL_RESULTS_DIR / "stats" / "study3_comparison.json",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=LOCAL_RESULTS_DIR / "stats" / "figures" / "study3_killer_figure.png",
    )
    args = ap.parse_args()
    models = _load_comparison(args.input)
    if not models:
        raise SystemExit(f"No models in {args.input}")
    plot_study3_figure(models, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
