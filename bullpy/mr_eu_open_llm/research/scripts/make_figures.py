"""Build paper v2 Figures 1 and 3 from committed outputs.

Every plotted number is read from a committed file in research/reproduction/
or from a file named in research/RESULTS_REGISTRY.md. Nothing is typed in.
No stimulus, frame or face image is drawn; schematics are boxes and text.

Usage:
    python research/scripts/make_figures.py --fig 1
    python research/scripts/make_figures.py --fig 3
    python research/scripts/make_figures.py --all [--out DIR]

Writes figN.pdf and figN.png to research/paper/v2/figures/ (or --out), and
updates SOURCES.md there with the input file, its sha256 and the script line
that read it, per panel.
"""

import argparse
import ast
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPRO = ROOT / "research" / "reproduction"
DEFAULT_OUT = ROOT / "research" / "paper" / "v2" / "figures"
SCRIPT_REL = "research/scripts/make_figures.py"

# Inputs. All committed; the last two are named in RESULTS_REGISTRY.md.
AUDIT = REPRO / "prompt_confound_audit.json"
DELIVERY = {m: REPRO / f"input_delivery_{m}.json" for m in ("gemma4", "qwen3vl", "molmo2")}
CFA = REPRO / "cfa_identifiability.json"
NORMS = ROOT / "data" / "eu_emotions_human_entropy.json"
WEAK_SRC = ROOT / "scripts" / "run_stimulus_interventions.py"

# Palette: reference categorical slots 1-3 (validated, light mode), plus inks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID, NEUTRAL, SURFACE = "#e4e3df", "#c9c8c2", "#ffffff"

MODEL_NAMES = {"gemma4": "Gemma 4", "qwen3vl": "Qwen3-VL", "molmo2": "Molmo2"}
MODEL_STYLE = {"gemma4": (BLUE, "o"), "qwen3vl": (ORANGE, "s"), "molmo2": (AQUA, "D")}
ROUTES = ("single_face", "video_face", "single_full", "legacy_full")

_SOURCES = defaultdict(list)  # figure -> [(panel, path, line)]


def _record(fig, panel, path, depth=2):
    line = sys._getframe(depth).f_lineno
    _SOURCES[fig].append((panel, Path(path), line))


def load_json(path, fig, panel):
    """Read a committed JSON file and record which panel used it and where."""
    _record(fig, panel, path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_constant(path, name, fig, panel):
    """Read a module-level constant from source without importing the module."""
    _record(fig, panel, path)
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {path}")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 7,
        "axes.labelcolor": INK2,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "text.color": INK,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    })


def panel_letter(ax, letter, x=-0.02, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right", color=INK)


def hgrid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- Figure 1


def fig1_target_position(ax):
    audit = load_json(AUDIT, 1, "a, target position")
    prompt_counts = [b["prompt_target_position_counts"] for b in audit["behaviour"].values()]
    counts = audit["lookup_target_position_counts"]
    if any(c != counts for c in prompt_counts):
        raise ValueError("prompt and lookup target positions differ")
    n_items = audit["n_items"]
    n_opts = load_json(NORMS, 1, "a, number of options")["n_options_human"]
    pos = list(range(n_opts))
    vals = [counts.get(str(p), 0) for p in pos]
    ax.bar([p + 1 for p in pos], vals, width=0.8, color=BLUE, edgecolor=SURFACE, linewidth=1)
    for p, v in zip(pos, vals):
        if v:
            ax.text(p + 1, v + n_items * 0.02, f"{v} of {n_items}", ha="center", va="bottom",
                    fontsize=6.5, color=INK)
    ax.set_xticks([p + 1 for p in pos])
    ax.set_xlim(0.4, n_opts + 0.6)
    ax.set_ylim(0, n_items * 1.15)
    ax.set_xlabel("Position of the acted label in the option list")
    ax.set_ylabel("Items")
    hgrid(ax)
    ax.set_title("Target position in the v1 prompt")


def fig1_option_sets(ax):
    audit = load_json(AUDIT, 1, "b, option-set counts")
    norms = load_json(NORMS, 1, "b, option sets per label")["trials"]
    sets = defaultdict(set)
    for t in norms.values():
        sets[t["human_target_label"]].update(o for o in t["human_options"] if o != "None of the above")
    labels = sorted(sets)
    offered_all = labels + sorted(set().union(*sets.values()) - set(labels))
    idx = {lab: i for i, lab in enumerate(offered_all)}
    for r, target in enumerate(labels):
        for offered in sets[target]:
            c = idx[offered]
            colour = BLUE if offered == target else NEUTRAL
            ax.add_patch(plt.Rectangle((c - 0.42, r - 0.42), 0.84, 0.84, color=colour, linewidth=0))
    ax.set_xlim(-0.6, len(offered_all) - 0.4)
    ax.set_ylim(len(labels) - 0.4, -0.6)
    ax.set_aspect("equal")
    ax.set_xticks(range(len(offered_all)))
    ax.set_xticklabels(offered_all, rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Label offered")
    ax.set_ylabel("Acted label of the item")
    ax.set_title(f"Option sets: {audit['n_unique_option_sets']} sets for "
                 f"{audit['n_target_labels']} labels")
    h1 = plt.Rectangle((0, 0), 1, 1, color=BLUE)
    h2 = plt.Rectangle((0, 0), 1, 1, color=NEUTRAL)
    ax.legend([h1, h2], ["Acted label", "Other label offered"], loc="upper center",
              bbox_to_anchor=(0.5, -0.31), ncol=2, handlelength=1, handleheight=1)


def fig1_delivery(ax):
    rows = {}
    threshold = None
    for m, path in DELIVERY.items():
        d = load_json(path, 1, f"c, delivery {MODEL_NAMES[m]}")
        rows[m] = d["verdicts"]
        rule_n = float(re.search(r"at least (\d+(?:\.\d+)?)", d["gate_1_rule"]).group(1))
        if threshold is not None and rule_n != threshold:
            raise ValueError("gate 1 rule differs between models")
        threshold = rule_n
    offsets = {"gemma4": -0.22, "qwen3vl": 0.0, "molmo2": 0.22}
    for m, verdicts in rows.items():
        colour, marker = MODEL_STYLE[m]
        for i, route in enumerate(ROUTES):
            v = verdicts[route]
            x = v["gate_1_median_tokens_across_face"]
            ax.plot([x], [i + offsets[m]], marker=marker, markersize=5.5, linestyle="none",
                    markeredgecolor=colour, markeredgewidth=1.2,
                    markerfacecolor=colour if v["pass"] else SURFACE)
            ax.text(x + 0.35, i + offsets[m], f"{x:g}", va="center", fontsize=5.5, color=INK2)
    ax.axvline(threshold, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.text(threshold - 0.2, -0.75, f"gate: {threshold:g}", ha="right", va="bottom", fontsize=6,
            color=INK2)
    ax.set_yticks(range(len(ROUTES)))
    ax.set_yticklabels(ROUTES, fontfamily="monospace", fontsize=6)
    ax.set_ylim(len(ROUTES) - 0.5, -0.9)
    ax.set_xlim(0, None)
    ax.set_xlabel("Median image tokens across the face")
    hgrid(ax, "x")
    ax.set_title("Delivery: tokens across the face per route")
    handles = [plt.Line2D([], [], marker=MODEL_STYLE[m][1], color=MODEL_STYLE[m][0],
                          linestyle="none", markersize=5) for m in rows]
    handles += [plt.Line2D([], [], marker="o", color=INK2, linestyle="none", markersize=5),
                plt.Line2D([], [], marker="o", color=INK2, markerfacecolor=SURFACE,
                           linestyle="none", markersize=5)]
    names = [MODEL_NAMES[m] for m in rows] + ["Passes all four gates", "Fails a gate"]
    ax.legend(handles, names, loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=3,
              handletextpad=0.3, columnspacing=1.0)


def fig1_weak_labels(ax):
    norms = load_json(NORMS, 1, "d, human p_target per label")
    weak = set(load_constant(WEAK_SRC, "WEAK_LABELS", 1, "d, weak-label list"))
    chance = 1.0 / norms["n_options_human"]
    by_label = defaultdict(list)
    for t in norms["trials"].values():
        by_label[t["human_target_label"]].append(t["p_target"])
    means = sorted(((sum(v) / len(v), lab, len(v)) for lab, v in by_label.items()), reverse=True)
    for i, (m, lab, n) in enumerate(means):
        colour = ORANGE if lab in weak else NEUTRAL
        ax.barh(i, m, height=0.72, color=colour, edgecolor=SURFACE, linewidth=0.8)
        if lab in weak:
            ax.text(m + 0.012, i, f"{m:.3f} (n = {n})", va="center", fontsize=5.8, color=INK)
    ax.axvline(chance, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.text(chance + 0.01, -1.0, f"chance 1/{norms['n_options_human']}", fontsize=6,
            color=INK2, va="bottom")
    ax.set_yticks(range(len(means)))
    ax.set_yticklabels([lab for _, lab, _ in means], fontsize=5.8)
    ax.set_ylim(len(means) - 0.4, -1.4)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Face-only humans choosing the acted label (item mean)")
    hgrid(ax, "x")
    ax.set_title("Weak labels in the human norms")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=NEUTRAL)],
              ["Flagged weak label", "Other label"], loc="lower right", handlelength=1,
              handleheight=1)


def make_fig1():
    fig = plt.figure(figsize=(7.08, 6.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.05], height_ratios=[1, 1.25],
                          left=0.14, right=0.97, top=0.95, bottom=0.11, wspace=0.42, hspace=0.5)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    axd = fig.add_subplot(gs[1, 1])
    fig1_target_position(axa)
    fig1_option_sets(axb)
    fig1_delivery(axc)
    fig1_weak_labels(axd)
    for ax, letter in zip((axa, axb, axc, axd), "abcd"):
        panel_letter(ax, letter, x=-0.12 if letter in "ac" else -0.3)
    return fig


# ---------------------------------------------------------------- Figure 3


def _box(ax, xy, w, h, text, colour=INK2, fill=SURFACE, size=6.5):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.01,rounding_size=0.025",
                                linewidth=0.9, edgecolor=colour, facecolor=fill))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=size,
            color=INK)


def _arrow(ax, a, b, colour, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=8, linewidth=1.1,
                                 color=colour, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def fig3_schematic(ax):
    """Proposition 2 as a diagram. No numbers: theory only."""
    _SOURCES[3].append(("a, schematic of proposition 2", None, sys._getframe(0).f_lineno))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(ax, (0.02, 0.60), 0.21, 0.17, "cue $z_1$")
    _box(ax, (0.02, 0.20), 0.21, 0.17, "cue $z_2$")
    ax.annotate("", xy=(0.125, 0.385), xytext=(0.125, 0.585),
                arrowprops=dict(arrowstyle="<->", color=MUTED, linewidth=0.9,
                                linestyle=(0, (2, 2))))
    ax.text(0.145, 0.485, r"corr $\rho$ on" + "\nnatural clips", fontsize=5.8, color=INK2,
            va="center")
    _box(ax, (0.50, 0.60), 0.48, 0.17, "observer S\n" + r"$f_S = a + b\,z_1$", colour=BLUE,
         size=6.2)
    _box(ax, (0.50, 0.20), 0.48, 0.17, "observer T\n" + r"$f_T = a + b\,z_2$", colour=ORANGE,
         size=6.2)
    _arrow(ax, (0.23, 0.685), (0.49, 0.685), BLUE)
    _arrow(ax, (0.23, 0.285), (0.49, 0.285), ORANGE)
    ax.text(0.5, 0.95, r"Behaviour on natural clips: corr($f_S$, $f_T$) = $\rho$",
            ha="center", va="center", fontsize=6, color=INK)
    ax.text(0.5, 0.05, "Cue effects: S on $z_1$ only, T on $z_2$ only, so CFA(S, T) = 0",
            ha="center", va="center", fontsize=6, color=INK)
    ax.set_title("Two observers, one behaviour, different cues")


def fig3_prop2(ax):
    cfa = load_json(CFA, 3, "b, proposition 2")
    rows = cfa["checks"]["proposition_2"]
    xs = list(range(len(rows)))
    beh = [r["output_correlation"] for r in rows]
    cos = [r["effect_cosine"] for r in rows]
    ax.plot(xs, beh, color=BLUE, marker="o", markersize=5, linewidth=2,
            markeredgecolor=SURFACE, markeredgewidth=1)
    ax.plot(xs, cos, color=ORANGE, marker="s", markersize=5, linewidth=2,
            markeredgecolor=SURFACE, markeredgewidth=1)
    ax.text(xs[1] + 0.1, beh[1] - 0.1, "Behaviour:\ncorr($f_S$, $f_T$)", ha="left", va="top",
            fontsize=6, color=INK)
    ax.text(xs[1], cos[1] + 0.06, "Cue effects: CFA(S, T)", ha="center", va="bottom",
            fontsize=6, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['rho']:g}" for r in rows])
    ax.set_xlim(-0.3, len(rows) - 0.7)
    ax.set_ylim(-0.08, 1.08)
    ax.set_xlabel(r"Cue correlation $\rho$")
    ax.set_ylabel("Correlation or cosine")
    hgrid(ax)
    ax.set_title("Proposition 2, computed")


def fig3_theorem1(ax):
    cfa = load_json(CFA, 3, "c, theorem 1")
    rows = cfa["checks"]["theorem_1"]
    req = [r["r_requested"] for r in rows]
    got = [r["r_achieved"] for r in rows]
    err = max(r["max_entry_error"] for r in rows)
    ax.plot([-1, 1], [-1, 1], color=GRID, linewidth=1, zorder=0)
    ax.plot(req, got, linestyle="none", marker="o", markersize=5.5, color=BLUE,
            markeredgecolor=SURFACE, markeredgewidth=1)
    ax.text(1.1, -1.05, "Outputs on natural\nitems unchanged:\n"
            f"largest change {err:.1e}", fontsize=5.8, color=INK, ha="right", va="bottom")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel("CFA requested")
    ax.set_ylabel("CFA of the constructed system")
    hgrid(ax, "both")
    ax.set_title("Theorem 1, computed")


def _spectra_rows(cfa):
    rows = []
    for key, name in (("libreface", "LibreFace"), ("openface3", "OpenFace 3.0")):
        r = cfa["eu_emotion_natural_clips"][key]
        rows.append((f"EU-Emotion, {name}, {r['n_cues']} cues", [("natural", r)]))
    for key, r in cfa["snoek_randomised_design"].items():
        task, group = key.split("_")
        rows.append((f"Snoek, {task} {group}",
                     [("as_coded", r["as_coded"]), ("left_right_merged", r["left_right_merged"])]))
    return rows


SPECTRA_STYLE = {
    "natural": (BLUE, "o", "Natural clips, detector cues"),
    "as_coded": (ORANGE, "s", "Randomised design, AUs as coded"),
    "left_right_merged": (AQUA, "D", "Randomised design, left and right merged"),
}


def _spectra_legend_names(cfa):
    snoek = list(cfa["snoek_randomised_design"].values())
    names = {}
    for kind, (_, _, base) in SPECTRA_STYLE.items():
        if kind == "natural":
            names[kind] = base
        else:
            counts = sorted({r[kind]["n_cues"] for r in snoek})
            names[kind] = f"{base}, {' or '.join(map(str, counts))} cues"
    return names


def fig3_spectra(ax, field, label, panel):
    cfa = load_json(CFA, 3, panel)
    rows = _spectra_rows(cfa)
    for i, (_, entries) in enumerate(rows):
        low = min(r[field] for _, r in entries)
        for kind, r in entries:
            colour, marker, _ = SPECTRA_STYLE[kind]
            v = r[field]
            ax.plot([v], [i], marker=marker, markersize=5.5, linestyle="none", color=colour,
                    markeredgecolor=SURFACE, markeredgewidth=0.8)
            left = len(entries) > 1 and v == low
            ax.text(v / 1.15 if left else v * 1.15, i, f"{v:.1f}" if v >= 10 else f"{v:.3g}", va="center",
                    ha="right" if left else "left", fontsize=5.8, color=INK2)
    ax.axvline(1.0, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_xscale("log")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([name for name, _ in rows])
    ax.set_ylim(len(rows) - 0.5, -0.9)
    ax.text(1.0 * 1.06, -0.62, "identity: 1", ha="left", va="center", fontsize=5.8,
            color=INK2)
    ax.set_xlabel(label)
    hgrid(ax, "x")
    return _spectra_legend_names(cfa)


def make_fig3():
    fig = plt.figure(figsize=(7.08, 5.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1, 1], height_ratios=[1, 1.05],
                          left=0.06, right=0.98, top=0.94, bottom=0.15, wspace=0.5, hspace=0.6)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[0, 2])
    gs2 = gs[1, :].subgridspec(1, 2, wspace=0.08)
    axd = fig.add_subplot(gs2[0, 0])
    axe = fig.add_subplot(gs2[0, 1], sharey=axd)
    fig3_schematic(axa)
    fig3_prop2(axb)
    fig3_theorem1(axc)
    names = fig3_spectra(axd, "lambda_min", r"Smallest eigenvalue $\lambda_{min}$ of the cue correlation",
                         "d, lambda_min per cue table")
    axd.set_xlim(0.03, 2.5)
    axd.set_title("Cue spectra: smallest eigenvalue")
    fig3_spectra(axe, "condition_number", "Condition number", "e, condition number per cue table")
    axe.set_xlim(0.7, 250)
    axe.tick_params(labelleft=False)
    axe.set_title("Cue spectra: condition number")
    # Bottom row needs room for its row labels.
    pos_d, pos_e = axd.get_position(), axe.get_position()
    shift = 0.235 - pos_d.x0
    axd.set_position([pos_d.x0 + shift, pos_d.y0, pos_d.width - shift / 2, pos_d.height])
    axe.set_position([pos_e.x0 + shift / 2, pos_e.y0, pos_e.width - shift / 2, pos_e.height])
    handles = [plt.Line2D([], [], marker=m, color=c, linestyle="none", markersize=5)
               for c, m, _ in SPECTRA_STYLE.values()]
    fig.legend(handles, [names[k] for k in SPECTRA_STYLE], loc="lower center",
               ncol=3, bbox_to_anchor=(0.55, 0.005), handletextpad=0.3, columnspacing=1.4)
    for ax, letter in zip((axa, axb, axc, axd, axe), "abcde"):
        panel_letter(ax, letter, x=-0.04 if letter == "a" else -0.12)
    return fig


# ---------------------------------------------------------------- output


FIGURES = {1: make_fig1, 3: make_fig3}


def write_sources(out_dir, built):
    path = out_dir / "SOURCES.md"
    sections = {}
    if path.exists():
        current = None
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"## Figure (\d+)", line)
            if m:
                current = int(m.group(1))
                sections[current] = []
            elif current is not None:
                sections[current].append(line)
    for n in built:
        lines = ["", "| Panel | Input file | sha256 | Script line |", "|---|---|---|---|"]
        for panel, src, line in _SOURCES[n]:
            if src is None:
                lines.append(f"| {panel} | none: diagram of "
                             "`research/theory/causal_feature_alignment.md`, no plotted numbers "
                             f"| - | `{SCRIPT_REL}:{line}` |")
                continue
            rel = src.resolve().relative_to(ROOT).as_posix()
            lines.append(f"| {panel} | `{rel}` | `{sha256(src)}` | `{SCRIPT_REL}:{line}` |")
        lines.append("")
        sections[n] = lines
    text = ["# Figure sources", "",
            f"Written by `{SCRIPT_REL}`. Each row is one read of a committed input file. "
            "Rebuild with `python research/scripts/make_figures.py --all`.", ""]
    for n in sorted(sections):
        text.append(f"## Figure {n}")
        text.extend(sections[n])
    path.write_text("\n".join(text).rstrip() + "\n", encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fig", type=int, choices=sorted(FIGURES))
    group.add_argument("--all", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    style()
    args.out.mkdir(parents=True, exist_ok=True)
    built = sorted(FIGURES) if args.all else [args.fig]
    written = []
    for n in built:
        _SOURCES[n].clear()
        fig = FIGURES[n]()
        for ext in ("pdf", "png"):
            p = args.out / f"fig{n}.{ext}"
            fig.savefig(p, metadata={"CreationDate": None} if ext == "pdf" else None)
            written.append(p)
        plt.close(fig)
    written.append(write_sources(args.out, built))
    for p in written:
        print(p)
    return written


if __name__ == "__main__":
    main()
