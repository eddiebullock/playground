"""Build paper v2 figures from committed outputs, and test the unlocked ones on synthetic data.

Figures 1 and 3 read committed files only: every plotted number comes from a committed
file in research/reproduction/ or one named in research/RESULTS_REGISTRY.md. Nothing is
typed in. No stimulus, frame or face image is drawn; schematics are boxes and text.

Figures 2, 4, 5 and 6 are written against the output schemas of the frozen analysis
scripts (analyse_rotation.py with face_input_validity.py, same_clip_family_analysis.py,
a4_analyse.py, analyse_stimulus_interventions.py). Their results are not yet locked, so:
  --synthetic  builds each from synthetic observers passed through the frozen code's own
               summary functions, titles it SYNTHETIC, and writes only to a temporary
               directory (never research/paper/v2/figures/).
  --real       reads the locked output files in LOCKED and refuses unless every one exists.

Usage:
    python research/scripts/make_figures.py --fig 1
    python research/scripts/make_figures.py --all [--out DIR]          # Figures 1 and 3
    python research/scripts/make_figures.py --fig 4 --synthetic [--out TMPDIR]
    python research/scripts/make_figures.py --fig 4 --real

Writes figN.pdf and figN.png to research/paper/v2/figures/ (or --out), and updates
SOURCES.md there with the input file, its sha256 and the script line that read it, per
panel.
"""

import argparse
import ast
import contextlib
import csv
import hashlib
import io
import json
import re
import sys
import tempfile
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
LIT_TABLE = REPRO / "literature_table.csv"
MANIFEST = ROOT / "data" / "eu_emotions_full_manifest.json"
HUMAN_AU = REPRO / "human_au_effects.json"
A4_CEILING = REPRO / "a4_human_ceiling.json"
ROUTE_RECOVERY = {1.0: REPRO / "same_clip_family_route_pattern_within_category.json",
                  0.75: REPRO / "same_clip_family_route_pattern_075x_within_category.json"}

# Locked outputs for the figures that are not yet built from results. --real refuses
# unless every file listed for the figure exists. Paths follow each frozen script's
# own output path; A2's script has no default, so its committed copy is named here.
LOCKED = {
    2: [REPRO / "rotation_summary.json", REPRO / "face_input_validity_summary.json"],
    4: [REPRO / f"same_clip_family_model_{m}.json" for m in ("gemma4", "qwen3vl", "molmo2")],
    5: [ROOT / "results" / "a4" / "summary.json"],
    6: [REPRO / f"stimulus_interventions_{m}.json" for m in ("gemma4", "qwen3vl")],  # Molmo is out of A2 (PREREG_A2 Models); trim further if RQ0 removes a model
}
UNLOCKED = tuple(sorted(LOCKED))
SYNTHETIC_TAG = "SYNTHETIC"

# Palette: reference categorical slots 1-3 (validated, light mode), plus inks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID, NEUTRAL, SURFACE = "#e4e3df", "#c9c8c2", "#ffffff"
# Tints for table cells: sequential blue step 250, its step 100, and orange mixed half
# with the surface. Cell text always names the state, so colour never carries it alone.
BLUE_250, BLUE_100, ORANGE_TINT, MID_GREY = "#86b6ef", "#cde2fb", "#f5b39a", "#f0efec"

MODEL_NAMES = {"gemma4": "Gemma 4", "qwen3vl": "Qwen3-VL", "molmo2": "Molmo2"}
MODEL_STYLE = {"gemma4": (BLUE, "o"), "qwen3vl": (ORANGE, "s"), "molmo2": (AQUA, "D")}
ROUTES = ("single_face", "video_face", "single_full", "legacy_full")

_SOURCES = defaultdict(list)  # figure -> [(panel, path or None, line, note or None)]


def _record(fig, panel, path, depth=2):
    line = sys._getframe(depth).f_lineno
    _SOURCES[fig].append((panel, Path(path), line, None))


def _record_note(fig, panel, note, depth=1):
    """A panel with no input file: a diagram, or a synthetic test input."""
    _SOURCES[fig].append((panel, None, sys._getframe(depth).f_lineno, note))


def load_json(path, fig, panel):
    """Read a committed JSON file and record which panel used it and where."""
    _record(fig, panel, path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_csv(path, fig, panel):
    """Read a committed CSV file and record which panel used it and where."""
    _record(fig, panel, path)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


LIT_COLUMNS = (
    ("target_position", "Target position\ncontrolled"),
    ("option_sets", "One option set\nacross items"),
    ("delivery_check", "Delivery to the\nmodel checked"),
    ("face_dependence", "Face dependence\ntested"),
)
LIT_CODES = {  # code -> (fill, cell text, legend text)
    "yes": (BLUE_250, "yes", "Yes"),
    "partial": (BLUE_100, "partly", "Partly"),
    "no": (ORANGE_TINT, "no", "No"),
    "na": (SURFACE, "does not arise", "Does not arise (no option list)"),
    "not_stated": (SURFACE, "not stated", "Not stated"),
}


def fig1_literature(ax):
    """The eight-paper table as a grid of coded cells. Text names every state."""
    rows = load_csv(LIT_TABLE, 1, "e, eight-paper table")
    plt.rcParams["hatch.color"] = GRID
    for r, row in enumerate(rows):
        inferred = {c.strip() for c in row["inferred_cells"].split(";") if c.strip()}
        for c, (key, _) in enumerate(LIT_COLUMNS):
            code = row[f"{key}_code"]
            if code not in LIT_CODES:
                raise ValueError(f"unknown code {code!r} in {LIT_TABLE.name}, ref {row['ref']}")
            fill, text, _ = LIT_CODES[code]
            ax.add_patch(plt.Rectangle((c + 0.04, r + 0.06), 0.92, 0.88, facecolor=fill,
                                       edgecolor=GRID if code in ("na", "not_stated") else SURFACE,
                                       linewidth=0.8,
                                       hatch="////" if code == "not_stated" else None))
            mark = " (inferred)" if key in inferred else ""
            ax.text(c + 0.5, r + 0.5, text + mark, ha="center", va="center", fontsize=5.8,
                    color=INK2 if code in ("na", "not_stated") else INK,
                    bbox=dict(boxstyle="square,pad=0.15", facecolor=SURFACE, edgecolor="none")
                    if code == "not_stated" else None)
    ax.set_xlim(0, len(LIT_COLUMNS))
    ax.set_ylim(len(rows), 0)
    ax.xaxis.tick_top()
    ax.set_xticks([c + 0.5 for c in range(len(LIT_COLUMNS))])
    ax.set_xticklabels([label for _, label in LIT_COLUMNS], fontsize=6.2)
    ax.set_yticks([r + 0.5 for r in range(len(rows))])
    ax.set_yticklabels([f"[{row['ref']}] {row['paper']}" for row in rows], fontsize=6.2)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Eight published VLM emotion designs, as the repo records them", pad=26)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=LIT_CODES[k][0],
                             edgecolor=GRID if k in ("na", "not_stated") else LIT_CODES[k][0],
                             hatch="////" if k == "not_stated" else None) for k in LIT_CODES]
    ax.legend(handles, [LIT_CODES[k][2] for k in LIT_CODES], loc="center left",
              bbox_to_anchor=(1.03, 0.5), handlelength=1.2, handleheight=1.2, fontsize=6)


def make_fig1():
    fig = plt.figure(figsize=(7.08, 9.3))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.05], height_ratios=[1, 1.25, 0.95],
                          left=0.14, right=0.97, top=0.965, bottom=0.04, wspace=0.42, hspace=0.62)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    axd = fig.add_subplot(gs[1, 1])
    axe = fig.add_subplot(gs[2, :])
    fig1_target_position(axa)
    fig1_option_sets(axb)
    fig1_delivery(axc)
    fig1_weak_labels(axd)
    fig1_literature(axe)
    pos = axe.get_position()
    axe.set_position([0.24, pos.y0, 0.46, pos.height])
    for ax, letter in zip((axa, axb, axc, axd), "abcd"):
        panel_letter(ax, letter, x=-0.12 if letter in "ac" else -0.3)
    panel_letter(axe, "e", x=-0.36, y=1.13)
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
    _record_note(3, "a, schematic of proposition 2",
                 "none: diagram of `research/theory/causal_feature_alignment.md`, no plotted numbers")
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


# ---------------------------------------------------------------- shared, Figures 2, 4, 5, 6


def require(block, keys, where):
    """Fail loudly if a frozen output lacks a field this figure reads."""
    missing = [k for k in keys if k not in block]
    if missing:
        raise KeyError(f"{where}: missing {', '.join(missing)}")
    return block


def _tag(title, synthetic):
    return f"{SYNTHETIC_TAG}: {title}" if synthetic else title


def _synthetic_banner(fig, synthetic):
    if synthetic:
        text = (f"{SYNTHETIC_TAG} TEST RENDER: synthetic observers through the frozen "
                "summary code. Not a result.")
        if fig.get_figwidth() < 6:
            text = text.replace(" through", "\nthrough")
        fig.suptitle(text, x=0.02, ha="left", fontsize=8, fontweight="bold", color=INK)


def _model_name(key):
    return MODEL_NAMES.get(key, key.replace("_", " "))


def _model_style(key, i=0):
    fallback = ((BLUE, "o"), (ORANGE, "s"), (AQUA, "D"))
    return MODEL_STYLE.get(key, fallback[i % 3])


def _frozen_imports():
    for path in (ROOT, ROOT / "src", ROOT / "research" / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _norm_trials():
    return json.loads(NORMS.read_text(encoding="utf-8"))["trials"]


# ---------------------------------------------------------------- Figure 2
#
# Reads two outputs with one shape per model (A3 imports A1's statistics):
#   rotation_summary.json (analyse_rotation.py): accuracy_real, accuracy_mismatch,
#     delta_real_minus_mismatch, delta_ci_actor_bootstrap, p_delta_holm, vision_dependent,
#     secondary_accuracy_by_target_position {"0".."5": accuracy}
#   face_input_validity_summary.json (face_input_validity.py): the same first six fields,
#     with accuracy_by_target_position inside "secondary".

FIG2_KEYS = ("accuracy_real", "accuracy_mismatch", "delta_real_minus_mismatch",
             "delta_ci_actor_bootstrap", "p_delta_holm", "vision_dependent")
FIG2_INPUTS = (("a1", "A1, grid"), ("a3", "A3, single face"))


def _fig2_rows(a1, a3):
    rows = []
    for m in [m for m in MODEL_NAMES if m in a1 or m in a3]:
        for key, label in FIG2_INPUTS:
            src = a1 if key == "a1" else a3
            if m not in src:
                continue
            entry = require(src[m], FIG2_KEYS, f"{key} {m}")
            if key == "a1":
                pos = require(entry, ["secondary_accuracy_by_target_position"], f"a1 {m}")[
                    "secondary_accuracy_by_target_position"]
            else:
                pos = require(require(entry, ["secondary"], f"a3 {m}")["secondary"],
                              ["accuracy_by_target_position"], f"a3 {m}")["accuracy_by_target_position"]
            rows.append((m, key, f"{_model_name(m)}, {label}", entry, pos))
    return rows


def fig2_delta(ax, rows, synthetic):
    for i, (m, _, _, e, _) in enumerate(rows):
        colour, marker = _model_style(m)
        lo, hi = e["delta_ci_actor_bootstrap"]
        x = e["delta_real_minus_mismatch"]
        ax.plot([lo, hi], [i, i], color=colour, linewidth=1.6, solid_capstyle="round")
        ax.plot([x], [i], marker=marker, markersize=6, linestyle="none", markeredgecolor=colour,
                markeredgewidth=1.2, markerfacecolor=colour if e["vision_dependent"] else SURFACE)
        ax.text(hi + 0.012, i, f"{x:.3f}", va="center", fontsize=5.8, color=INK2)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[2] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_xlabel("Accuracy real minus mismatch (actor bootstrap 95% CI)")
    hgrid(ax, "x")
    ax.set_title(_tag("RQ0: real minus mismatch", synthetic))


def fig2_accuracy(ax, rows, synthetic):
    for i, (m, _, _, e, _) in enumerate(rows):
        colour, marker = _model_style(m)
        a, b = e["accuracy_real"], e["accuracy_mismatch"]
        ax.plot([b, a], [i, i], color=NEUTRAL, linewidth=1.2, zorder=1)
        ax.plot([b], [i], marker=marker, markersize=5.5, linestyle="none", color=NEUTRAL,
                markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=2)
        ax.plot([a], [i], marker=marker, markersize=5.5, linestyle="none", color=colour,
                markeredgecolor=SURFACE, markeredgewidth=0.8, zorder=3)
    ax.axvline(1 / 6, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.text(1 / 6 + 0.008, -0.55, "chance 1/6", fontsize=5.8, color=INK2, va="center")
    ax.set_yticks(range(len(rows)))
    ax.tick_params(labelleft=False)
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_xlim(0, max(1 / 6, *(r[3]["accuracy_real"] for r in rows)) * 1.2)
    ax.set_xlabel("Accuracy")
    hgrid(ax, "x")
    ax.set_title(_tag("Accuracy against chance", synthetic))
    ax.legend([plt.Line2D([], [], marker="o", color=INK2, linestyle="none", markersize=5),
               plt.Line2D([], [], marker="o", color=NEUTRAL, linestyle="none", markersize=5)],
              ["Real face (model colour)", "Mismatched face"], loc="lower right")


def fig2_position(ax, rows, key, synthetic):
    picked = [r for r in rows if r[1] == key]
    n = max((len(r[4]) for r in picked), default=6)
    for m, _, _, _, pos in picked:
        colour, marker = _model_style(m)
        xs = sorted(int(k) for k in pos)
        ax.plot([x + 1 for x in xs], [pos[str(x)] for x in xs], color=colour, linewidth=2,
                marker=marker, markersize=5, markeredgecolor=SURFACE, markeredgewidth=0.8,
                label=_model_name(m))
    ax.axhline(1 / 6, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.text(n + 0.1, 1 / 6, "chance\n1/6", fontsize=5.6, color=INK2, va="center", ha="left")
    ax.set_xticks(range(1, n + 1))
    ax.set_ylim(0, None)
    ax.set_xlabel("Position of the acted label")
    ax.set_ylabel("Accuracy, real face")
    hgrid(ax)
    ax.set_title(_tag(f"By target position, {dict(FIG2_INPUTS)[key]}", synthetic))


def fig2_from(a1, a3, synthetic=False):
    """Figure 2 from the A1 and A3 summaries (dicts keyed by model)."""
    rows = _fig2_rows(a1, a3)
    fig = plt.figure(figsize=(7.08, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], left=0.17, right=0.95,
                          top=0.9 if synthetic else 0.94, bottom=0.15, wspace=0.12, hspace=0.55)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    gs2 = gs[1, :].subgridspec(1, 2, wspace=0.3)
    axc, axd = fig.add_subplot(gs2[0, 0]), fig.add_subplot(gs2[0, 1])
    fig2_delta(axa, rows, synthetic)
    fig2_accuracy(axb, rows, synthetic)
    fig2_position(axc, rows, "a1", synthetic)
    fig2_position(axd, rows, "a3", synthetic)
    handles = [plt.Line2D([], [], marker=_model_style(m)[1], color=_model_style(m)[0],
                          linestyle="none", markersize=5) for m in dict.fromkeys(r[0] for r in rows)]
    handles += [plt.Line2D([], [], marker="o", color=INK2, linestyle="none", markersize=5),
                plt.Line2D([], [], marker="o", color=INK2, markerfacecolor=SURFACE,
                           linestyle="none", markersize=5)]
    names = [_model_name(m) for m in dict.fromkeys(r[0] for r in rows)]
    names += ["Vision dependent by the rule", "Not by the rule"]
    fig.legend(handles, names, loc="lower center", ncol=len(names), bbox_to_anchor=(0.55, 0.0),
               handletextpad=0.3, columnspacing=1.2)
    for ax, letter in zip((axa, axb, axc, axd), "abcd"):
        panel_letter(ax, letter, x=-0.04 if letter == "b" else -0.12)
    _synthetic_banner(fig, synthetic)
    return fig


def make_fig2_real():
    a1 = load_json(LOCKED[2][0], 2, "a to c, A1 rotation summary")
    a3 = load_json(LOCKED[2][1], 2, "a, b, d, A3 face input validity summary")
    return fig2_from(a1, a3)


def synthetic_fig2(seed=20261108):
    """A1 and A3 summaries from synthetic answers, through the frozen summary functions.

    A1: analyse_rotation.main, with its file loader replaced by synthetic runs over the real
    item ids and option lists. A3: face_input_validity.summarise_model and decide on
    synthetic cells built by its own trial_cells. No result file is read.
    """
    import numpy as np

    _frozen_imports()
    import analyse_rotation as ar
    from paper_v2.option_score import rotate_options
    from scripts import face_input_validity as fiv
    from scripts.run_stimulus_interventions import load_items

    human = _norm_trials()
    ids = sorted(k for k, t in human.items() if len(t["human_options"]) == 6)
    rng = np.random.default_rng(seed)
    # Synthetic observers: (real accuracy, mismatch accuracy, first-slot bias).
    params = {"qwen3vl": (0.42, 0.18, 0.10), "gemma4": (0.30, 0.20, 0.04),
              "molmo2": (0.14, 0.12, 0.12)}

    def fake_load(model, condition):
        p_real, p_mis, bias = params[model]
        runs = []
        for r in ar.ROTATIONS:
            run = {}
            for k in ids:
                options = rotate_options(human[k]["human_options"], r)
                target = human[k]["human_target_label"]
                p = (p_real if condition == "real" else p_mis) + (bias if options.index(target) == 0 else 0)
                p = min(max(p + 0.15 * (human[k]["p_target"] - 0.6), 0.01), 0.99)
                run[k] = {"correct": bool(rng.random() < p), "options": options,
                          "response_distribution": {target: p}}
            runs.append(run)
        return runs

    original, argv = ar.load, sys.argv
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "rotation_summary.json"
        try:
            ar.load, sys.argv = fake_load, ["analyse_rotation.py", "--output", str(out)]
            with contextlib.redirect_stdout(io.StringIO()):
                ar.main()
        finally:
            ar.load, sys.argv = original, argv
        a1 = json.loads(out.read_text(encoding="utf-8"))

    items = load_items(MANIFEST, NORMS, Path("."))
    by_id = {i["trial_id"]: i for i in items}
    partners = fiv.partners_for(items)
    strength = {"qwen3vl": 1.6, "gemma4": 1.1, "molmo2": 0.3}
    summaries = {}
    for m in fiv.MODELS:
        rows = []
        for item in items:
            for cell in fiv.trial_cells(item, by_id[partners[item["trial_id"]]]):
                lp = rng.normal(0, 1, len(cell["options"]))
                if cell["condition"] == "real":
                    lp[cell["options"].index(cell["label"])] += strength[m]
                lp[0] += 0.3
                cell.update(fiv.correct_from_scores(cell["options"], lp, cell["label"]))
                cell.update(status="ok", joint_logprob=lp.tolist())
                rows.append(cell)
        summaries[m] = fiv.summarise_model(rows, human)
    a3 = fiv.decide(summaries)
    for m in a3:
        a3[m]["n_partial_lines_dropped"] = 0
    return json.loads(json.dumps(a1)), json.loads(json.dumps(a3))


def make_fig2_synthetic():
    _record_note(2, "a to d, A1 summary", "SYNTHETIC: `synthetic_fig2`, through "
                 "`analyse_rotation.main` with a synthetic loader")
    _record_note(2, "a, b, d, A3 summary", "SYNTHETIC: `synthetic_fig2`, through "
                 "`face_input_validity.summarise_model` and `decide`")
    return fig2_from(*synthetic_fig2(), synthetic=True)


# ---------------------------------------------------------------- Figure 4
#
# Reads same_clip_family_model_{model}.json (same_clip_family_analysis.model_route_summary):
#   rq1_status, coverage {share_dropped, over_loss_limit}, and per input "composite"
#   (confirmatory), "first", "second": {lam, score, p, null_sd, beta}.
# Reference bands come from the committed route-pattern recovery (synthetic observers):
#   pattern[factor] -> human_like / prior_only {median_score, q05, q95}.

FIG4_INPUTS = (("composite", "Composite (confirmatory)"), ("first", "LibreFace only"),
               ("second", "OpenFace 3.0 only"))


def _fig4_references(recovery):
    refs = []
    for factor in sorted(recovery, reverse=True):
        block = recovery[factor]
        entry = next(p for p in block["pattern"] if float(p["factor"]) == factor)
        refs.append((f"Human-like observer, {factor:g}x effect", require(entry["human_like"],
                     ["median_score", "q05", "q95"], f"recovery {factor}")))
    top = recovery[max(recovery)]
    entry = next(p for p in top["pattern"] if float(p["factor"]) == max(recovery))
    refs.append(("Prior-only observer", entry["prior_only"]))
    return refs


def fig4_from(models, recovery, synthetic=False):
    """Figure 4 from model_route_summary outputs keyed by model, plus recovery bands."""
    refs = _fig4_references(recovery)
    for m, block in models.items():
        require(block, ["rq1_status", "coverage", *(k for k, _ in FIG4_INPUTS)], f"fig4 {m}")
        for k, _ in FIG4_INPUTS:
            require(block[k], ["score", "p", "null_sd"], f"fig4 {m} {k}")
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    fig.subplots_adjust(left=0.38, right=0.96, top=0.8 if synthetic else 0.86, bottom=0.27)
    labels = []
    y = 0
    for name, r in refs:
        ax.plot([r["q05"], r["q95"]], [y, y], color=NEUTRAL, linewidth=4, solid_capstyle="butt")
        ax.plot([r["median_score"]], [y], marker="|", markersize=8, color=INK2)
        labels.append(name)
        y += 1
    split = y - 0.5
    y += 0.3
    offsets = {"composite": 0.0, "first": -0.22, "second": 0.22}
    small = {"first": "^", "second": "v"}
    order = [m for m in MODEL_NAMES if m in models] + [m for m in models if m not in MODEL_NAMES]
    ys = []
    for i, m in enumerate(order):
        block = models[m]
        colour, marker = _model_style(m, i)
        c = block["composite"]
        band = 1.96 * c["null_sd"]
        ax.plot([-band, band], [y, y], color=GRID, linewidth=6, solid_capstyle="butt", zorder=0)
        for k, _ in FIG4_INPUTS[1:]:
            ax.plot([block[k]["score"]], [y + offsets[k]], marker=small[k], markersize=4,
                    linestyle="none", color=MUTED)
        ax.plot([c["score"]], [y], marker=marker, markersize=7, linestyle="none",
                markeredgecolor=colour, markeredgewidth=1.3,
                markerfacecolor=colour if c["p"] < 0.05 else SURFACE)
        tag = "" if block["rq1_status"].startswith("confirmatory") else ", descriptive"
        right = max(band, *(block[k]["score"] for k, _ in FIG4_INPUTS))
        ax.text(right + 0.05, y, f"{c['score']:.2f} (p = {c['p']:.3g}{tag})",
                va="center", fontsize=5.8, color=INK2)
        labels.append(_model_name(m))
        ys.append(y)
        y += 1
    ax.axhline(split + 0.15, color=GRID, linewidth=0.6)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_yticks(list(range(len(refs))) + ys)
    ax.set_yticklabels(labels)
    ax.set_ylim(y - 0.4, -0.7)
    ax.set_xlim(-0.35, 1.0)
    ax.set_xlabel("Family pattern score with the human matrix (Pearson over cells)")
    hgrid(ax, "x")
    ax.set_title(_tag("RQ1: natural-clip pattern score per model", synthetic))
    handles = [plt.Line2D([], [], marker="o", color=INK2, linestyle="none", markersize=5),
               plt.Line2D([], [], marker="o", color=INK2, markerfacecolor=SURFACE,
                          linestyle="none", markersize=5),
               plt.Line2D([], [], marker="^", color=MUTED, linestyle="none", markersize=4),
               plt.Line2D([], [], marker="v", color=MUTED, linestyle="none", markersize=4),
               plt.Line2D([], [], color=GRID, linewidth=5),
               plt.Line2D([], [], color=NEUTRAL, linewidth=4)]
    ax.legend(handles, ["Composite, p < .05", "Composite, p >= .05", "LibreFace only",
                        "OpenFace 3.0 only", "Null, +/- 1.96 SD", "Recovery, 5th to 95th pct"],
              loc="upper center", bbox_to_anchor=(0.32, -0.2), ncol=3, fontsize=5.8,
              handlelength=1.8, handletextpad=0.8, columnspacing=1.0)
    _synthetic_banner(fig, synthetic)
    return fig


def _load_route_recovery(fig):
    return {f: load_json(p, fig, f"reference band, route recovery {f:g}x")
            for f, p in ROUTE_RECOVERY.items()}


def make_fig4_real():
    models = {}
    for path in LOCKED[4]:
        m = path.stem.rsplit("_", 1)[1]
        models[m] = load_json(path, 4, f"pattern score, {MODEL_NAMES[m]}")
    return fig4_from(models, _load_route_recovery(4))


def synthetic_fig4(seed=20261108):
    """model_route_summary outputs for synthetic observers.

    Matrices are random; scores and p values come from the frozen pattern_test, coverage
    from clip_coverage, and the file layout from model_route_summary. No result is read.
    """
    import numpy as np

    _frozen_imports()
    import same_clip_family_analysis as scf

    human = _norm_trials()
    ids = sorted(human)
    categories = {k: human[k]["human_target_label"] for k in ids}
    rng = np.random.default_rng(seed)
    reference = rng.normal(0, 1, (21, 4))
    out = {}
    for m, strength, lost in (("gemma4", 0.55, 0), ("qwen3vl", 0.25, 6), ("molmo2", 0.0, 20)):
        result = {}
        for key in ("composite", "first", "second"):
            B = strength * reference + rng.normal(0, 1, reference.shape)
            null = rng.normal(0, 1, (199, *reference.shape))
            result[key] = {"lam": 1.0, "B": B, "pattern": scf.pattern_test(B, reference, null)}
        kept = ids[lost:]
        attrs = {"coverage": scf.clip_coverage(ids, kept, categories), "unparsed_share": 0.0,
                 "rotations": 5}
        out[m] = json.loads(json.dumps(scf.model_route_summary(result, attrs)))
    return out


def make_fig4_synthetic():
    _record_note(4, "pattern scores", "SYNTHETIC: `synthetic_fig4`, through "
                 "`same_clip_family_analysis.pattern_test` and `model_route_summary`")
    return fig4_from(synthetic_fig4(), _load_route_recovery(4), synthetic=True)


# ---------------------------------------------------------------- Figure 5
#
# Reads the A4 summary (a4_analyse.main): human_reference {basic, basic_residual,
#   conversational, conversational_residual}, statistic, and datasets[dataset][model]:
#   raw and residual {cfa, ci, p_cue, r_s, over_ceiling}, p_holm, decision,
#   behaviour {output_r, modal_agreement, modal_kappa}.

FIG5_DATASETS = ("basic_WC", "basic_EA", "conversational_WC", "conversational_EA")
SESOI_CFA = 0.30  # scripts/a4_design.py SESOI_CFA; checked against it in the tests


def _fig5_rows(summary):
    require(summary, ["human_reference", "statistic", "datasets"], "A4 summary")
    rows = []
    for ds in FIG5_DATASETS:
        block = summary["datasets"].get(ds, {})
        order = [m for m in MODEL_NAMES if m in block] + [m for m in block if m not in MODEL_NAMES]
        for m in order:
            e = require(block[m], ["raw", "residual", "decision", "behaviour"], f"A4 {ds} {m}")
            for k in ("raw", "residual"):
                require(e[k], ["cfa", "ci"], f"A4 {ds} {m} {k}")
            require(e["behaviour"], ["output_r"], f"A4 {ds} {m} behaviour")
            rows.append((ds, m, e))
    return rows


def _fig5_axis(ax, rows, value, title, ref=None, labels=True):
    for i, (ds, m, e) in enumerate(rows):
        colour, marker = _model_style(m)
        v, ci = value(e)
        filled = e["decision"] != "Descriptive"
        if ci is not None:
            ax.plot(ci, [i, i], color=colour, linewidth=1.5, solid_capstyle="round")
        ax.plot([v], [i], marker=marker, markersize=5.5, linestyle="none", markeredgecolor=colour,
                markeredgewidth=1.2, markerfacecolor=colour if filled else SURFACE)
    if ref is not None:
        groups = defaultdict(list)
        for i, (ds, _, _) in enumerate(rows):
            groups[ds].append(i)
        for ds, idx in groups.items():
            x = ref(ds)
            ax.plot([x, x], [min(idx) - 0.4, max(idx) + 0.4], color=INK, linewidth=1.4)
    ax.axvline(0, color=INK2, linewidth=0.6)
    ax.set_yticks(range(len(rows)))
    if labels:
        ax.set_yticklabels([f"{ds.replace('_', ' ')}, {_model_name(m)}" for ds, m, _ in rows])
    else:
        ax.tick_params(labelleft=False)
    ax.set_ylim(len(rows) - 0.5, -1.1)
    ax.set_xlim(-0.5, 1.05)
    ax.set_xticks([-0.5, 0, 0.5, 1])
    hgrid(ax, "x")
    ax.set_title(title)
    previous = None
    for i, (ds, _, _) in enumerate(rows):
        if previous is not None and ds != previous:
            ax.axhline(i - 0.5, color=GRID, linewidth=0.6)
        previous = ds


def fig5_from(summary, synthetic=False):
    """Figure 5 from the A4 summary JSON."""
    rows = _fig5_rows(summary)
    href = summary["human_reference"]
    fig = plt.figure(figsize=(7.08, 3.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1], left=0.19, right=0.98,
                          top=0.82 if synthetic else 0.88, bottom=0.25, wspace=0.12)
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))
    _fig5_axis(axa, rows, lambda e: (e["residual"]["cfa"], e["residual"]["ci"]),
               _tag("Residual CFA (confirmatory)", synthetic),
               ref=lambda ds: href[f"{ds.split('_')[0]}_residual"])
    axa.axvline(SESOI_CFA, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)))
    axa.text(SESOI_CFA + 0.02, -0.85, f"SESOI {SESOI_CFA:.2f}", fontsize=5.8, color=INK2,
             va="center")
    axa.set_xlim(-0.5, 1.62)
    axa.spines["bottom"].set_bounds(-0.5, 1)
    for i, (_, _, e) in enumerate(rows):
        axa.text(1.1, i, e["decision"], ha="left", va="center", fontsize=5.6, color=INK)
    axa.set_xlabel("CFA with the human do-effects (95% CI)")
    _fig5_axis(axb, rows, lambda e: (e["raw"]["cfa"], e["raw"]["ci"]), _tag("Raw CFA", synthetic),
               ref=lambda ds: href[ds.split("_")[0]], labels=False)
    axb.set_xlabel("CFA, raw (95% CI)")
    _fig5_axis(axc, rows, lambda e: (e["behaviour"]["output_r"], None),
               _tag("Behaviour", synthetic), labels=False)
    axc.set_xlabel("Output correlation with humans")
    handles = [plt.Line2D([], [], marker=_model_style(m)[1], color=_model_style(m)[0],
                          linestyle="none", markersize=5) for m in dict.fromkeys(r[1] for r in rows)]
    handles += [plt.Line2D([], [], marker="o", color=INK2, markerfacecolor=SURFACE,
                           linestyle="none", markersize=5),
                plt.Line2D([], [], color=INK, linewidth=1.4)]
    names = [_model_name(m) for m in dict.fromkeys(r[1] for r in rows)]
    names += ["Descriptive model (open)", "Human WC to EA reference"]
    fig.legend(handles, names, loc="lower center", ncol=len(names), bbox_to_anchor=(0.58, 0.0),
               handlelength=1.8, handletextpad=0.5, columnspacing=1.2)
    for ax, letter in zip((axa, axb, axc), "abc"):
        panel_letter(ax, letter, x=-0.03)
    _synthetic_banner(fig, synthetic)
    return fig


def make_fig5_real():
    return fig5_from(load_json(LOCKED[5][0], 5, "a to c, A4 summary"))


def synthetic_fig5(seed=20261108):
    """An A4 summary for synthetic observers.

    CFA values are drawn at random; the decision per model and dataset comes from the
    frozen apply_rule, the headline from headline, and the human reference from
    human_reference on the committed human effects. No model output is read.
    """
    import numpy as np

    _frozen_imports()
    from scripts import a4_analyse as a4
    from scripts import a4_design as d

    rng = np.random.default_rng(seed)
    store = json.loads(HUMAN_AU.read_text(encoding="utf-8"))
    centre = {"gemma4": 0.55, "qwen3vl": 0.2, "molmo2": 0.05}
    out = {"design_sha256": SYNTHETIC_TAG, "statistic": "residual",
           "human_reference": a4.human_reference(store), "detector": {}, "datasets": {}, "models": {}}
    for ds in d.DATASETS:
        results = {}
        for m in (*d.MODELS, *d.DESCRIPTIVE_MODELS):
            entry = {"p_signal": 0.001, "share_lost": 0.0, "lost_ok": True}
            for k, shift in (("raw", 0.15), ("residual", 0.0)):
                v = float(np.clip(centre[m] + shift + rng.normal(0, 0.08), -1, 1))
                half = 0.12 + 0.05 * rng.random()
                entry[k] = {"cfa": v, "p_cue": 0.001 if v > 0.3 else 0.4, "null_q95": 0.25,
                            "ci": [v - half, v + half], "r_s": 0.9, "over_ceiling": v / 0.97,
                            "lower_bound_only": False}
            entry["behaviour"] = {"output_r": float(np.clip(0.6 + rng.normal(0, 0.1), -1, 1)),
                                  "modal_agreement": 0.5, "modal_kappa": 0.4}
            results[m] = entry
        gates = {m: {"rq0": True, "g2_g4": m in d.MODELS} for m in results}
        a4.apply_rule(results, gates, "residual")
        out["datasets"][ds] = results
    for m in (*d.MODELS, *d.DESCRIPTIVE_MODELS):
        out["models"][m] = {"headline": a4.headline({k: out["datasets"][k][m]["decision"]
                                                     for k in d.DATASETS})}
    return json.loads(json.dumps(out, default=float))


def make_fig5_synthetic():
    _record_note(5, "a to c, A4 summary", "SYNTHETIC: `synthetic_fig5`, decisions through "
                 "`a4_analyse.apply_rule`, reference through `a4_analyse.human_reference`")
    return fig5_from(synthetic_fig5(), synthetic=True)


# ---------------------------------------------------------------- Figure 6
#
# Reads one A2 output per model (analyse_stimulus_interventions.main): per route
# ("single_face", "video_face"), results.primary_blind_acted[contrast id] =
#   {mean_change, ci95, direction, verdict, p_maxt}. Verdicts: matches_prediction,
#   opposes_prediction, null, inconclusive.

FIG6_ROUTES = ("single_face", "video_face")
VERDICT_STYLE = {  # verdict -> (colour, filled, legend)
    "matches_prediction": (BLUE, True, "Matches prediction"),
    "opposes_prediction": (ORANGE, True, "Opposes prediction"),
    "null": (INK2, True, "Null (all six conditions)"),
    "inconclusive": (MUTED, False, "Inconclusive"),
}


def _fig6_rows(outputs):
    ids = []
    for out in outputs.values():
        for route in FIG6_ROUTES:
            block = require(require(require(out, [route], "A2 output")[route], ["results"],
                                    route)["results"], ["primary_blind_acted"], route)
            for cid in block["primary_blind_acted"]:
                if (route, cid) not in ids:
                    ids.append((route, cid))
    return ids


def fig6_panel(ax, out, ids, title, labels=True):
    for i, (route, cid) in enumerate(ids):
        row = out[route]["results"]["primary_blind_acted"].get(cid)
        if row is None:
            continue
        require(row, ["mean_change", "ci95", "direction", "verdict"], f"A2 {route} {cid}")
        sign = row["direction"]
        colour, filled, _ = VERDICT_STYLE[row["verdict"]]
        if row["ci95"] is not None:
            lo, hi = sorted(v * sign for v in row["ci95"])
            ax.plot([lo, hi], [i, i], color=colour, linewidth=1.4, solid_capstyle="round")
        ax.plot([row["mean_change"] * sign], [i], marker="o", markersize=5, linestyle="none",
                markeredgecolor=colour, markeredgewidth=1.2,
                markerfacecolor=colour if filled else SURFACE)
    ax.axvline(0, color=INK2, linewidth=0.8)
    routes = [r for r, _ in ids]
    split = next((i for i in range(1, len(routes)) if routes[i] != routes[i - 1]), None)
    if split is not None:
        ax.axhline(split - 0.5, color=GRID, linewidth=0.8)
    ax.set_yticks(range(len(ids)))
    if labels:
        ax.set_yticklabels([f"{cid}, {route}" for route, cid in ids], fontsize=6)
    else:
        ax.tick_params(labelleft=False)
    ax.set_ylim(len(ids) - 0.5, -0.6)
    hgrid(ax, "x")
    ax.set_title(title, fontsize=7)


def fig6_from(outputs, synthetic=False):
    """Figure 6 from A2 outputs keyed by model (or synthetic observer)."""
    ids = _fig6_rows(outputs)
    n = len(outputs)
    fig, axes = plt.subplots(1, n, figsize=(7.08, 0.2 * len(ids) + 1.9), sharex=True, squeeze=False)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.89 if synthetic else 0.93, bottom=0.16,
                        wspace=0.1)
    for j, (m, out) in enumerate(outputs.items()):
        fig6_panel(axes[0, j], out, ids, _tag(_model_name(m), synthetic), labels=j == 0)
    fig.supxlabel("Change in the predicted direction (blind readout, actor 95% CI)", fontsize=7,
                  y=0.06, color=INK2)
    handles = [plt.Line2D([], [], marker="o", color=c, markerfacecolor=c if f else SURFACE,
                          linestyle="none", markersize=5) for c, f, _ in VERDICT_STYLE.values()]
    fig.legend(handles, [v[2] for v in VERDICT_STYLE.values()], loc="lower center", ncol=4,
               bbox_to_anchor=(0.56, 0.0), handletextpad=0.3, columnspacing=1.2)
    _synthetic_banner(fig, synthetic)
    return fig


def make_fig6_real():
    outputs = {}
    for path in LOCKED[6]:
        m = path.stem.rsplit("_", 1)[1]
        outputs[m] = load_json(path, 6, f"A2 output, {MODEL_NAMES[m]}")
    return fig6_from(outputs)


def synthetic_fig6(seed=3):
    """A2 outputs for three synthetic observers through the frozen analyse_route.

    Cells come from recovery_stimulus_interventions.synthetic_rows over the real items,
    with a synthetic MCQ block added so the MCQ bridge can run. All gates are set True,
    so the verdicts exercise the decision table; no null can be reached without delivery
    and region data, which synthetic cells do not have.
    """
    _frozen_imports()
    from scripts import analyse_stimulus_interventions as an
    from scripts import recovery_stimulus_interventions as rec
    from scripts.run_stimulus_interventions import load_items

    items = load_items(MANIFEST, NORMS, Path("."))
    options = {i["trial_id"]: i["options"] for i in items}
    keep = ("trial_id", "actor", "label", "human_modal_label", "weak_label")
    items = [{k: i[k] for k in keep} for i in items]
    gates = {k: True for k in an.GATE_KEYS}
    out = {}
    for name in ("human_like", "mouth_only", "prior_only"):
        rows = rec.synthetic_rows(name, items, seed)
        for r in rows:
            fam = r["blind"]["family_logprob"]
            opts = options[r["trial_id"]]
            r["mcq"] = [{"options": opts,
                         "joint_logprob": [fam.get(rec.OPTION_TO_FAMILY.get(o), -5.0) for o in opts]}]
        cells = an.Cells(rows)
        actors = sorted({r["actor"] for r in cells.items.values()})
        entry = {"model": name, "design_sha256": SYNTHETIC_TAG, "n_items": len(cells.items)}
        for route in FIG6_ROUTES:
            with contextlib.redirect_stderr(io.StringIO()):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    results = an.analyse_route(cells, route, actors, gates, 0.1, flip_args=rec.FLIPS)
            entry[route] = {"gates": gates, "margin": 0.1, "results": results}
        out[f"synthetic_{name}"] = json.loads(json.dumps(entry, default=float))
    return out


def make_fig6_synthetic():
    _record_note(6, "all panels", "SYNTHETIC: `synthetic_fig6`, through "
                 "`analyse_stimulus_interventions.analyse_route` on recovery observers")
    return fig6_from(synthetic_fig6(), synthetic=True)


# ---------------------------------------------------------------- output


FIGURES = {1: make_fig1, 3: make_fig3}  # built from committed outputs
REAL = {2: make_fig2_real, 4: make_fig4_real, 5: make_fig5_real, 6: make_fig6_real}
SYNTHETIC = {2: make_fig2_synthetic, 4: make_fig4_synthetic, 5: make_fig5_synthetic,
             6: make_fig6_synthetic}


def check_locked(n):
    """Refuse a real-data build unless every locked output for the figure exists."""
    missing = [p for p in LOCKED[n] if not p.is_file()]
    if missing:
        names = ", ".join(p.relative_to(ROOT).as_posix() if ROOT in p.parents else str(p)
                          for p in missing)
        raise SystemExit(f"Figure {n}: locked output missing ({names}). "
                         "No real-data figure is built before its result is locked.")


def check_synthetic_out(out):
    """Synthetic renders never go into the paper's figure folder."""
    out = Path(out).resolve()
    figures = DEFAULT_OUT.resolve()
    if out == figures or figures in out.parents:
        raise SystemExit(f"Synthetic figures may not be written to {DEFAULT_OUT}. "
                         "Pass a temporary --out, or omit it.")
    return out


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
        for panel, src, line, note in _SOURCES[n]:
            if src is None:
                lines.append(f"| {panel} | {note} | - | `{SCRIPT_REL}:{line}` |")
                continue
            rel = src.resolve().relative_to(ROOT).as_posix()
            lines.append(f"| {panel} | `{rel}` | `{sha256(src)}` | `{SCRIPT_REL}:{line}` |")
        lines.append("")
        sections[n] = lines
    text = ["# Figure sources", "",
            f"Written by `{SCRIPT_REL}`. Each row is one read of a committed input file. "
            "Rebuild with `python research/scripts/make_figures.py --all`.", "",
            "Figures 2, 4, 5 and 6 get a section only when built with `--real`, which refuses "
            "until every locked output in `LOCKED` exists. Synthetic test renders "
            "(`--synthetic`) are written to a temporary folder, never here.", ""]
    for n in sorted(sections):
        text.append(f"## Figure {n}")
        text.extend(sections[n])
    path.write_text("\n".join(text).rstrip() + "\n", encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fig", type=int, choices=sorted({*FIGURES, *UNLOCKED}))
    group.add_argument("--all", action="store_true", help="Figures 1 and 3, from committed outputs.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--real", action="store_true",
                      help="Figures 2, 4, 5, 6 from locked outputs; refuses if any is missing.")
    mode.add_argument("--synthetic", action="store_true",
                      help="Figures 2, 4, 5, 6 from synthetic observers, to a temporary folder.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.all or args.fig in FIGURES:
        if args.real or args.synthetic:
            parser.error("--real and --synthetic apply to Figures 2, 4, 5 and 6 only")
        builders = FIGURES if args.all else {args.fig: FIGURES[args.fig]}
        out = args.out or DEFAULT_OUT
    elif args.real:
        check_locked(args.fig)
        builders, out = {args.fig: REAL[args.fig]}, args.out or DEFAULT_OUT
    elif args.synthetic:
        out = check_synthetic_out(args.out) if args.out else Path(tempfile.mkdtemp(prefix="synthetic_figs_"))
        builders = {args.fig: SYNTHETIC[args.fig]}
    else:
        parser.error(f"Figure {args.fig} is not locked: pass --synthetic, or --real once its output is locked")
    style()
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for n, build in sorted(builders.items()):
        _SOURCES[n].clear()
        fig = build()
        for ext in ("pdf", "png"):
            p = out / f"fig{n}.{ext}"
            fig.savefig(p, metadata={"CreationDate": None} if ext == "pdf" else None)
            written.append(p)
        plt.close(fig)
    written.append(write_sources(out, sorted(builders)))
    for p in written:
        print(p)
    return written


if __name__ == "__main__":
    main()
