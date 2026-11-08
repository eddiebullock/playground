#!/usr/bin/env python3
"""Rerun every CPU-only, public-input step and compare with the committed outputs (goal G12).

The script builds a clean copy of the committed tree (`git archive HEAD`) in a temporary
directory, adds the public external tables, runs each step there, and compares what the step
writes with the file committed at HEAD. The working tree is never written to. Untracked files,
cluster caches and licence-restricted data are never copied in, so a step either runs on
public inputs alone or is skipped with the reason.

Modes:
  --quick   steps that take under about 5 minutes each on 2 CPU workers (about 6 minutes in all)
  default   quick steps plus the long CPU recoveries (A4, A5, A6, same-clip recoveries);
            these take hours and are meant for a large CPU node

Public inputs that are not in git (gitignored, downloaded once):
  data/external/crema_d/   five CREMA-D rating tables, public on GitHub (ODbL / DbCL)
  data/external/snoek_au/  au_emotion_data.zip, Snoek et al. (CC BY 4.0)
They are taken from --external (default: data/external in this checkout's main working
tree, if present) or downloaded with --fetch. Either way each file's sha256 is checked.

Comparison: byte-identical files pass as "exact". Otherwise JSON and CSV files are compared
field by field with a stated tolerance (floats rel 1e-9 / abs 1e-12 by default; Monte Carlo
permutation p-values abs 0.01 where a step says so), skipping fields that record a path,
a wall-clock time or a timestamp, and PNGs are compared by pixels. Each rule is set per step
in STEPS below.

Usage (conda env mr_eu_open_llm):
    python scripts/reproduce.py --quick
    python scripts/reproduce.py --quick --fetch --keep
    python scripts/reproduce.py --only behaviour crema_select
    python scripts/reproduce.py --list
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parents[1]

# Public external inputs: path under data/external, URL, sha256.
CREMA_COMMIT = "1658cd34"
CREMA_BASE = f"https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/{CREMA_COMMIT}/"
EXTERNAL = {
    "crema_d/finishedResponses.csv": (
        CREMA_BASE + "finishedResponses.csv",
        "5eab9668f6adc2666b0179ae80553a50a5172dbc9812e582cd0ffa7bc0dd237e"),
    "crema_d/finishedEmoResponses.csv": (
        CREMA_BASE + "finishedEmoResponses.csv",
        "e6d44cbbf1dca05746c789f87ecdbf975e01cb142bba88266f23189a47bb4667"),
    "crema_d/VideoDemographics.csv": (
        CREMA_BASE + "VideoDemographics.csv",
        "e33ddc8b60bedf98541735801127cbb67098b5957271adbdd5dea8a44294eec3"),
    "crema_d/tabulatedVotes.csv": (
        CREMA_BASE + "processedResults/tabulatedVotes.csv",
        "774d9d759cb5caf2542758a39eca8c0f1d2777f9987ff5d5adf8066d192f377b"),
    "crema_d/summaryTable.csv": (
        CREMA_BASE + "processedResults/summaryTable.csv",
        "fb0e382700d32c0d9adc0c6815292215a36bcf912f2657048862c91e3bd2b317"),
    "snoek_au/au_emotion_data.zip": (
        "https://ndownloader.figshare.com/files/37888164",  # DOI 10.21942/uva.21261885.v1
        "6542ef31530545135f9572609a194d34c80566078b82fea0490266e6f015d706"),
}
CREMA = tuple(k for k in EXTERNAL if k.startswith("crema_d/"))
SNOEK = ("snoek_au/au_emotion_data.zip",)

# Fields that record where or when a file was made, never a result.
PROVENANCE_KEYS = ("runtime_seconds", "seconds", "locked_at_utc")

EVAL = "results/baseline/eu_emotions/{m}/eval_v2_eu_emotions_{m}_video_only_seed42.json"


@dataclass
class Out:
    """One output file and how to compare it with the committed copy."""

    path: str
    kind: str = "auto"  # auto, json, csv, png, sha256
    ignore: Tuple[str, ...] = ()  # dotted key paths or bare key names skipped anywhere
    only: Tuple[str, ...] = ()  # compare only these top-level keys
    mc_keys: Tuple[str, ...] = ()  # keys holding Monte Carlo p-values, compared at mc_tol
    mc_tol: float = 0.01
    unordered: Tuple[str, ...] = ()  # keys holding lists whose order carries no meaning
    rel: float = 1e-9
    abs_: float = 1e-12
    sha256: Optional[str] = None  # for kind sha256: pinned value instead of a committed file
    note: str = ""


@dataclass
class Step:
    name: str
    cmd: List[str]
    outputs: List[Out]
    needs: Tuple[str, ...] = ()
    quick: bool = True
    minutes: float = 0.1  # wall time on 2 CPU workers, Apple M-series, measured 2026-11-08
    pre: Optional[str] = None  # name of a setup action in PRE
    note: str = ""


def _py(*args: str) -> List[str]:
    return [sys.executable, *args]


W = "{workers}"  # replaced by --workers

STEPS: List[Step] = [
    # ---------------------------------------------------------------- behaviour, Phase 0 to 5
    Step("behaviour", _py("-m", "paper_v2.behaviour"),
         [Out("research/reproduction/behaviour_summary.json")]),
    Step("phase0_numbers", _py("research/scripts/reproduce_headline_numbers.py"),
         [Out("research/reproduction/phase0_numbers.json",
              ignore=("activation_cache", "missing", "has_difficulty_matched",
                      "json_has_difficulty_matched_control_pairs", "json_has_entanglement_vs_entropy_field"),
              note="activation_cache and missing describe the machine that ran it; the has_* flags record which "
                   "fields the input JSONs had on 2026-10-30, before commit a438754 added them. Every number is compared")],
         pre="activation_meta",
         note="The extract_meta.json files it reads are staged from research/reproduction/activation_meta"),
    Step("human_calibration",
         _py("-m", "scripts.human_calibration", "--eval", *[EVAL.format(m=m) for m in ("gemma4", "molmo2", "qwen3vl")],
             "--out", "results/stats/human_calibration.json"),
         [Out("results/stats/human_calibration.json")]),
    Step("phase3_pilot_ablation",
         _py("-m", "paper_v2.causal", "--trials", "results/mech/pilot_n36/ablate_trials_qwen3vl_layer4.csv",
             "--out", "research/reproduction/phase3_pilot_ablation.json"),
         [Out("research/reproduction/phase3_pilot_ablation.json", ignore=("source",),
              note="source is an absolute path in the committed file")]),
    Step("prompt_confound_audit", _py("research/scripts/audit_prompt_confound.py"),
         [Out("research/reproduction/prompt_confound_audit.json", ignore=("activations",), mc_keys=("p_perm",),
              note="activations needs cluster caches; p_perm are 10,000-draw permutation p-values")]),
    Step("merge_phase2_all_layers", _py("research/scripts/merge_phase2_all_layers.py"),
         [Out("research/reproduction/phase2_gate_all_layers.json"),
          Out("research/reproduction/phase2_all_layers_summary.csv")],
         note="Rebuilds the hand merge of commit f6dd3cf from the three committed per-model JSONs"),
    Step("plot_phase2_grouped", _py("research/scripts/plot_phase2_all_layers.py"),
         [Out("research/figures/phase2_grouped_t1_rho.png")],
         note="Plots the committed CSV from the cluster probe run"),
    Step("plot_phase2_loo", _py("research/scripts/plot_phase2_loo_vs_grouped.py"),
         [Out("research/figures/phase2_loo_vs_grouped.png")]),
    # ---------------------------------------------------------------- CREMA-D, public tables
    Step("crema_select", _py("-I", "research/scripts/crema_select.py", "--data-dir", "data/external/crema_d"),
         [Out("research/reproduction/crema_d_selection.csv")], needs=CREMA),
    Step("crema_reliability", _py("-I", "research/scripts/crema_reliability.py", "--data-dir", "data/external/crema_d"),
         [Out("research/reproduction/crema_d_reliability.json"),
          Out("research/reproduction/crema_d_clip_reliability.csv")], needs=CREMA, minutes=0.5),
    Step("crema_calibration_clips", _py("-m", "scripts.readout_ladder", "--step", "select"),
         [Out("research/reproduction/crema_calibration_clips.csv")],
         note="ids only, from the committed selection; the videos themselves are not needed"),
    Step("a6_human_target", _py("research/scripts/a6_human_target.py"),
         [Out("research/reproduction/a6_human_target.json")], needs=CREMA, minutes=0.1),
    # ---------------------------------------------------------------- Snoek AU data, public
    Step("human_au_effects",
         _py("-I", "research/scripts/human_au_effects.py", "--zip", "data/external/snoek_au/au_emotion_data.zip",
             "--out-json", "research/reproduction/human_au_effects.json",
             "--out-csv", "research/reproduction/human_au_effects.csv"),
         [Out("research/reproduction/human_au_effects.json", unordered=("pair",),
              note="largest_au_correlation.pair is an unordered pair; AU02L/AU02R order follows a floating-point tie"),
          Out("research/reproduction/human_au_effects.csv")],
         needs=SNOEK, minutes=1.0),
    Step("a4_manifest",
         _py("research/scripts/a4_stimuli.py", "manifest", "--zip", "data/external/snoek_au/au_emotion_data.zip",
             "--out", "results/a4/manifest.csv"),
         [Out("results/a4/manifest.csv", kind="sha256",
              sha256="2ef1161e895224d499b360144690a07c5440bb018105e9c2ffc762f68badd2d1",
              note="not committed; checked against the manifest sha256 pinned in PREREG_A4_DRAFT.md")],
         needs=SNOEK),
    Step("a4_human_ceiling", _py("-m", "scripts.a4_analyse", "--write_ceiling"),
         [Out("research/reproduction/a4_human_ceiling.json")], needs=SNOEK, minutes=0.3),
    Step("human_response_predictor", _py("-m", "scripts.human_response_predictor"),
         [Out("research/reproduction/human_response_predictor.json")], needs=SNOEK, minutes=0.3),
    Step("cfa_identifiability",
         _py("research/scripts/cfa_identifiability.py", "--au-csv", "research/reproduction/eu_au_measures.csv",
             "--snoek-zip", "data/external/snoek_au/au_emotion_data.zip",
             "--out", "research/reproduction/cfa_identifiability.json"),
         [Out("research/reproduction/cfa_identifiability.json")], needs=SNOEK,
         note="eu_au_measures.csv is a committed derived table; making it needs the EU-Emotion videos"),
    # ---------------------------------------------------------------- A2 recovery (no model)
    Step("a2_recovery", _py("-m", "scripts.recovery_stimulus_interventions", "--replicates", "100"),
         [Out("research/reproduction/recovery_stimulus_interventions.json")], minutes=0.7,
         note="the committed file used 100 replicates; the script default is 30"),
    # ---------------------------------------------------------------- same-clip, from committed AU table
    Step("same_clip_cue_human",
         _py("research/scripts/same_clip_cue_analysis.py", "--au-csv", "research/reproduction/eu_au_measures.csv",
             "--workers", W),
         [Out("research/reproduction/same_clip_cue.json"), Out("research/reproduction/same_clip_cue_matrix.csv"),
          Out("research/reproduction/same_clip_cue_detector_verdicts.csv"),
          Out("research/reproduction/same_clip_cue_human_beta_wide.csv")], minutes=2.7),
    # The attempt-2 lock hashes the committed attempt-1 outputs, so it runs before they are rewritten.
    Step("same_clip_family_a2_criteria",
         _py("research/scripts/same_clip_family_analysis.py", "--write-attempt-2-criteria"),
         [Out("research/reproduction/same_clip_family_a2_criteria.json", ignore=("script_sha256_at_lock",),
              note="lock record: script hash at lock differs by design")]),
    Step("same_clip_family_criteria",
         _py("research/scripts/same_clip_family_analysis.py", "--write-criteria",
             "--criteria-out", "research/reproduction/same_clip_family_criteria.json"),
         [Out("research/reproduction/same_clip_family_criteria.json", ignore=("script_sha256_at_lock",),
              note="lock record: script hash at lock differs by design")]),
    # ---------------------------------------------------------------- long recoveries, full mode
    Step("same_clip_cue_recovery",
         _py("research/scripts/same_clip_cue_analysis.py", "--recovery", "--au-csv",
             "research/reproduction/eu_au_measures.csv", "--workers", W),
         [Out("research/reproduction/same_clip_cue_recovery.json")], quick=False, minutes=None),
    Step("same_clip_family_human",
         _py("research/scripts/same_clip_family_analysis.py", "--au-csv", "research/reproduction/eu_au_measures.csv",
             "--graphau-csv", "research/reproduction/eu_au_measures_graphau.csv", "--workers", W),
         [Out("research/reproduction/same_clip_family.json"), Out("research/reproduction/same_clip_family_matrix.csv"),
          Out("research/reproduction/same_clip_family_verdicts.csv")], quick=False, minutes=None,
         note="cluster job: 2 h on 8 CPUs"),
    Step("same_clip_family_recovery",
         _py("research/scripts/same_clip_family_analysis.py", "--recovery", "--cue-set", "families",
             "--au-csv", "research/reproduction/eu_au_measures.csv",
             "--graphau-csv", "research/reproduction/eu_au_measures_graphau.csv", "--workers", W),
         [Out("research/reproduction/same_clip_family_recovery.json", ignore=("script_sha256",))],
         quick=False, minutes=None, note="cluster job: hours on 40 CPUs"),
    Step("same_clip_family_recovery_singles",
         _py("research/scripts/same_clip_family_analysis.py", "--recovery", "--cue-set", "families_singles",
             "--au-csv", "research/reproduction/eu_au_measures.csv",
             "--graphau-csv", "research/reproduction/eu_au_measures_graphau.csv", "--workers", W),
         [Out("research/reproduction/same_clip_family_recovery_families_singles.json", ignore=("script_sha256",))],
         quick=False, minutes=None, note="cluster job: hours on 40 CPUs"),
    Step("same_clip_route_pattern",
         _py("research/scripts/same_clip_family_analysis.py", "--recovery", "--route-pattern", "--cue-set", "families",
             "--frame-set", "model_1fps", "--scheme", "within_category", "--seed-offset", "200000",
             "--recovery-out", "research/reproduction/same_clip_family_route_pattern_within_category.json",
             "--au-csv", "research/reproduction/eu_au_measures.csv",
             "--graphau-csv", "research/reproduction/eu_au_measures_graphau.csv", "--workers", W),
         [Out("research/reproduction/same_clip_family_route_pattern_within_category.json", ignore=("script_sha256",))],
         quick=False, minutes=None, note="cluster job: 1 h on 40 CPUs"),
    Step("same_clip_route_pattern_075",
         _py("research/scripts/same_clip_family_analysis.py", "--recovery", "--route-pattern", "--cue-set", "families",
             "--frame-set", "model_1fps", "--scheme", "within_category", "--seed-offset", "300000",
             "--pattern-factors", "0.75", "--fixed-scale", "2.0",
             "--recovery-out", "research/reproduction/same_clip_family_route_pattern_075x_within_category.json",
             "--au-csv", "research/reproduction/eu_au_measures.csv",
             "--graphau-csv", "research/reproduction/eu_au_measures_graphau.csv", "--workers", W),
         [Out("research/reproduction/same_clip_family_route_pattern_075x_within_category.json", ignore=("script_sha256",))],
         quick=False, minutes=None, note="cluster job: 1 h on 40 CPUs"),
    Step("same_clip_family_a2_human",
         _py("research/scripts/same_clip_family_analysis.py", "--attempt-2",
             "--au-csv", "research/reproduction/eu_au_measures.csv",
             "--graphau-csv", "research/reproduction/eu_au_measures_graphau.csv", "--workers", W),
         [Out("research/reproduction/same_clip_family_a2.json"), Out("research/reproduction/same_clip_family_a2_matrix.csv"),
          Out("research/reproduction/same_clip_family_a2_verdicts.csv")], quick=False, minutes=None),
    Step("same_clip_family_a2_recovery",
         _py("research/scripts/same_clip_family_analysis.py", "--recovery", "--attempt-2", "--cue-set", "families",
             "--au-csv", "research/reproduction/eu_au_measures.csv",
             "--graphau-csv", "research/reproduction/eu_au_measures_graphau.csv", "--workers", W),
         [Out("research/reproduction/same_clip_family_a2_recovery.json", ignore=("script_sha256",))],
         quick=False, minutes=None, note="cluster job: hours on 40 CPUs"),
    Step("same_clip_family_a2_recovery_singles",
         _py("research/scripts/same_clip_family_analysis.py", "--recovery", "--attempt-2", "--cue-set", "families_singles",
             "--au-csv", "research/reproduction/eu_au_measures.csv",
             "--graphau-csv", "research/reproduction/eu_au_measures_graphau.csv", "--workers", W),
         [Out("research/reproduction/same_clip_family_a2_recovery_families_singles.json", ignore=("script_sha256",))],
         quick=False, minutes=None, note="cluster job: hours on 40 CPUs"),
    Step("same_clip_family_a2_recovery_model_frames",
         _py("research/scripts/same_clip_family_analysis.py", "--recovery", "--attempt-2", "--cue-set", "families",
             "--frame-set", "model_1fps", "--only", "sparse_0.5_frozen_lambda", "prior_only_frozen_lambda",
             "actor_prior_frozen_lambda", "--au-csv", "research/reproduction/eu_au_measures.csv",
             "--graphau-csv", "research/reproduction/eu_au_measures_graphau.csv", "--workers", W),
         [Out("research/reproduction/same_clip_family_a2_recovery_model_frames.json", ignore=("script_sha256",))],
         quick=False, minutes=None, note="cluster job"),
    Step("same_clip_route_pattern_within_actor",
         _py("research/scripts/same_clip_family_analysis.py", "--recovery", "--route-pattern", "--cue-set", "families",
             "--frame-set", "model_1fps", "--scheme", "within_actor", "--seed-offset", "200000",
             "--recovery-out", "research/reproduction/same_clip_family_route_pattern_within_actor.json",
             "--au-csv", "research/reproduction/eu_au_measures.csv",
             "--graphau-csv", "research/reproduction/eu_au_measures_graphau.csv", "--workers", W),
         [Out("research/reproduction/same_clip_family_route_pattern_within_actor.json", ignore=("script_sha256",))],
         quick=False, minutes=None, note="cluster job: 1 h on 40 CPUs"),
    Step("same_clip_route_pattern_075_within_actor",
         _py("research/scripts/same_clip_family_analysis.py", "--recovery", "--route-pattern", "--cue-set", "families",
             "--frame-set", "model_1fps", "--scheme", "within_actor", "--seed-offset", "300000",
             "--pattern-factors", "0.75", "--fixed-scale", "2.0",
             "--recovery-out", "research/reproduction/same_clip_family_route_pattern_075x_within_actor.json",
             "--au-csv", "research/reproduction/eu_au_measures.csv",
             "--graphau-csv", "research/reproduction/eu_au_measures_graphau.csv", "--workers", W),
         [Out("research/reproduction/same_clip_family_route_pattern_075x_within_actor.json", ignore=("script_sha256",))],
         quick=False, minutes=None, note="cluster job: 1 h on 40 CPUs"),
    Step("a4_recovery",
         _py("-m", "scripts.a4_recovery", "--manifest", "results/a4/manifest.csv",
             "--out", "research/reproduction/a4_recovery.json", "--workers", W),
         [Out("research/reproduction/a4_recovery.json")], needs=SNOEK, quick=False, minutes=None,
         note="needs the a4_manifest step first"),
    Step("a6_recovery", _py("-m", "scripts.a6_recovery", "--workers", W),
         [Out("research/reproduction/a6_recovery.json")], needs=CREMA, quick=False, minutes=180,
         note="10,832 s on 3 workers"),
    Step("a6_recovery_both_directions",
         _py("-m", "scripts.a6_recovery", "--variant", "both_directions", "--workers", W,
             "--output", "research/reproduction/a6_recovery_both_directions_paired.json"),
         [Out("research/reproduction/a6_recovery_both_directions_paired.json")], needs=CREMA, quick=False, minutes=17),
    Step("a6_recovery_first_draft",
         _py("-m", "scripts.a6_recovery", "--variant", "first_draft", "--workers", W,
             "--output", "research/reproduction/a6_recovery_draft_as_written.json"),
         [Out("research/reproduction/a6_recovery_draft_as_written.json")], needs=CREMA, quick=False, minutes=1),
    Step("a6_recovery_happy_equivalence",
         _py("-m", "scripts.a6_recovery", "--variant", "happy_equivalence", "--workers", W,
             "--output", "research/reproduction/a6_recovery_happy_equivalence.json"),
         [Out("research/reproduction/a6_recovery_happy_equivalence.json")], needs=CREMA, quick=False, minutes=2),
    Step("a5_recovery", _py("-m", "scripts.a5_recovery", "--workers", W),
         [Out("research/reproduction/a5_recovery.json")], quick=False, minutes=None,
         note="A5 is a draft under active change; a mismatch here may only mean the draft moved on"),
]


# ------------------------------------------------------------------------------- setup actions

def pre_activation_meta(work: Path) -> None:
    """Stage the committed copies of extract_meta.json where reproduce_headline_numbers.py reads them."""
    for model in ("qwen3vl", "gemma4", "molmo2"):
        dest = work / "results" / "activations" / f"baseline_{model}_6afc" / model / "extract_meta.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(work / "research" / "reproduction" / "activation_meta" / f"{model}_extract_meta.json", dest)


PRE = {"activation_meta": pre_activation_meta}


# ------------------------------------------------------------------------------- git and data

def git(*args: str, cwd: Path = HERE, binary: bool = False):
    out = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)
    return out.stdout if binary else out.stdout.decode().strip()


def repo_prefix() -> Tuple[Path, str]:
    top = Path(git("rev-parse", "--show-toplevel"))
    prefix = git("rev-parse", "--show-prefix").rstrip("/")
    return top, prefix


def committed(rev: str, prefix: str, rel: str) -> Optional[bytes]:
    spec = f"{rev}:{prefix + '/' if prefix else ''}{rel}"
    try:
        return git("show", spec, binary=True)
    except subprocess.CalledProcessError:
        return None


def build_clean_copy(rev: str, prefix: str, dest: Path) -> None:
    """Extract the committed project tree at rev into dest (tracked files only)."""
    top = Path(git("rev-parse", "--show-toplevel"))
    data = git("archive", "--format=tar", rev, prefix or ".", cwd=top, binary=True)
    strip = len(Path(prefix).parts) if prefix else 0
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if len(parts) <= strip:
                continue
            member.name = str(Path(*parts[strip:]))
            tar.extract(member, dest, filter="data") if sys.version_info >= (3, 12) else tar.extract(member, dest)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_external() -> Optional[Path]:
    """data/external in the main working tree of this repository, which may differ from HERE in a worktree."""
    candidates = [HERE / "data" / "external"]
    try:
        common = Path(git("rev-parse", "--path-format=absolute", "--git-common-dir"))
        top, prefix = repo_prefix()
        main_top = common.parent
        candidates.append(main_top / prefix / "data" / "external")
    except subprocess.CalledProcessError:
        pass
    for c in candidates:
        if c.is_dir():
            return c
    return None


def stage_external(work: Path, source: Optional[Path], fetch: bool) -> Dict[str, str]:
    """Copy or download each public external file into work/data/external, checking sha256."""
    status: Dict[str, str] = {}
    for rel, (url, want) in EXTERNAL.items():
        dest = work / "data" / "external" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = source / rel if source else None
        if src is not None and src.is_file():
            shutil.copyfile(src, dest)
            how = f"copied from {src}"
        elif fetch:
            with urllib.request.urlopen(url, timeout=120) as r, dest.open("wb") as f:
                shutil.copyfileobj(r, f)
            how = f"downloaded from {url}"
        else:
            status[rel] = "missing"
            continue
        got = sha256_file(dest)
        if got != want:
            dest.unlink()
            status[rel] = f"sha256 mismatch ({got[:12]} != {want[:12]}), {how}"
        else:
            status[rel] = "ok"
    return status


# ------------------------------------------------------------------------------- comparison

def _num(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _close(a: float, b: float, rel: float, abs_: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def compare_values(ref: Any, new: Any, out: Out, path: str, diffs: List[str], stats: Dict[str, float]) -> None:
    if len(diffs) >= 20:
        return
    key = path.rsplit(".", 1)[-1].split("[")[0]
    if key in out.ignore or path.lstrip(".") in out.ignore or key in PROVENANCE_KEYS:
        return
    if isinstance(ref, dict) and isinstance(new, dict):
        for k in sorted(set(ref) | set(new)):
            sub = f"{path}.{k}"
            if k in out.ignore or sub.lstrip(".") in out.ignore or k in PROVENANCE_KEYS:
                continue
            if k not in ref or k not in new:
                diffs.append(f"{sub}: present only in {'committed' if k in ref else 'rerun'}")
                continue
            compare_values(ref[k], new[k], out, sub, diffs, stats)
        return
    if isinstance(ref, list) and isinstance(new, list):
        if key in out.unordered:
            ref, new = sorted(ref, key=repr), sorted(new, key=repr)
        if len(ref) != len(new):
            diffs.append(f"{path}: length {len(ref)} != {len(new)}")
            return
        for i, (a, b) in enumerate(zip(ref, new)):
            compare_values(a, b, out, f"{path}[{i}]", diffs, stats)
        return
    a, b = _num(ref), _num(new)
    if a is not None and b is not None:
        mc = key in out.mc_keys
        rel, abs_ = (0.0, out.mc_tol) if mc else (out.rel, out.abs_)
        gap = abs(a - b) if not (math.isnan(a) or math.isnan(b)) else 0.0
        stat = "max_mc_gap" if mc else "max_abs_gap"
        stats[stat] = max(stats.get(stat, 0.0), gap)
        if not _close(a, b, rel, abs_):
            diffs.append(f"{path}: {ref!r} != {new!r}")
        return
    if ref != new:
        diffs.append(f"{path}: {str(ref)[:80]!r} != {str(new)[:80]!r}")


def parse_csv(data: bytes) -> List[List[Any]]:
    rows = list(csv.reader(io.StringIO(data.decode("utf-8"))))
    parsed = []
    for row in rows:
        cells: List[Any] = []
        for c in row:
            try:
                cells.append(float(c))
            except ValueError:
                cells.append(c)
        parsed.append(cells)
    return parsed


def compare_png(ref: bytes, new: bytes) -> Tuple[bool, str]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return False, "PIL or numpy missing, cannot compare pixels"
    a = np.asarray(Image.open(io.BytesIO(ref)).convert("RGB"), dtype=float)
    b = np.asarray(Image.open(io.BytesIO(new)).convert("RGB"), dtype=float)
    # Font rasterising and tight bounding boxes move by a pixel or two between matplotlib and
    # FreeType builds, so the figure is compared on the shared area, with a size limit.
    if any(abs(x - y) > 0.01 * max(x, y) for x, y in zip(a.shape[:2], b.shape[:2])):
        return False, f"image size {a.shape[:2]} != {b.shape[:2]}"
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    d = np.abs(a[:h, :w] - b[:h, :w])
    changed = float((d.max(axis=2) > 32).mean())
    mean = float(d.mean() / 255)
    ok = changed <= 0.05 and mean <= 0.02
    size = "" if a.shape == b.shape else f"size {a.shape[:2]} vs {b.shape[:2]}, "
    return ok, (f"{size}pixels changed by >32/255 {changed:.2%}, mean abs diff {mean:.4f} "
                f"(tolerance: size within 1%, at most 5% of pixels, mean 0.02)")


def compare(out: Out, ref: Optional[bytes], new_path: Path) -> Tuple[str, str]:
    """Return (verdict, detail). Verdicts: exact, within_tolerance, differs, missing_output, no_reference."""
    if not new_path.is_file():
        return "missing_output", "the step did not write this file"
    new = new_path.read_bytes()
    if out.kind == "sha256":
        got = hashlib.sha256(new).hexdigest()
        return ("exact", f"sha256 {got[:12]} matches the pin") if got == out.sha256 else \
            ("differs", f"sha256 {got[:12]} != pinned {out.sha256[:12]}")
    if ref is None:
        return "no_reference", "no committed copy at this revision"
    if ref == new:
        return "exact", "byte-identical"
    kind = out.kind if out.kind != "auto" else new_path.suffix.lstrip(".")
    diffs: List[str] = []
    stats: Dict[str, float] = {}
    if kind == "json":
        a, b = json.loads(ref), json.loads(new)
        if out.only:
            a = {k: a.get(k) for k in out.only}
            b = {k: b.get(k) for k in out.only}
        compare_values(a, b, out, "", diffs, stats)
    elif kind == "csv":
        compare_values(parse_csv(ref), parse_csv(new), out, "", diffs, stats)
    elif kind == "png":
        ok, detail = compare_png(ref, new)
        return ("within_tolerance" if ok else "differs"), detail
    else:
        return "differs", "bytes differ and no structured comparison for this type"
    gaps = ", ".join(f"{k} {v:.3g}" for k, v in sorted(stats.items()))
    skipped = ", ".join(out.ignore)
    detail = (f"skipped: {skipped}; " if skipped else "") + (f"{gaps}" if gaps else "")
    if diffs:
        return "differs", "; ".join(diffs[:5]) + (f" (+{len(diffs) - 5} more)" if len(diffs) > 5 else "")
    return "within_tolerance", detail or "equal after parsing"


# ------------------------------------------------------------------------------- driver

def run_step(step: Step, work: Path, workers: int, env: Dict[str, str], log_dir: Path) -> Tuple[int, float]:
    if step.pre:
        PRE[step.pre](work)
    cmd = [c.replace(W, str(workers)) for c in step.cmd]
    started = time.time()
    with (log_dir / f"{step.name}.log").open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        rc = subprocess.run(cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    return rc, time.time() - started


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--quick", action="store_true", help="only steps under about 5 minutes")
    ap.add_argument("--only", nargs="+", help="run these steps by name")
    ap.add_argument("--list", action="store_true", help="list steps and exit")
    ap.add_argument("--rev", default="HEAD", help="revision to check out and compare against")
    ap.add_argument("--workers", type=int, default=2, help="CPU workers for steps that take them, and BLAS threads")
    ap.add_argument("--external", type=Path, default=None, help="folder holding crema_d/ and snoek_au/")
    ap.add_argument("--fetch", action="store_true", help="download missing public inputs from their public URLs")
    ap.add_argument("--workdir", type=Path, default=None, help="parent of the temporary directory")
    ap.add_argument("--keep", action="store_true", help="keep the temporary directory")
    ap.add_argument("--report", type=Path, default=None, help="also write the JSON report here")
    args = ap.parse_args(argv)

    steps = [s for s in STEPS if (not args.quick or s.quick)]
    if args.only:
        unknown = set(args.only) - {s.name for s in STEPS}
        if unknown:
            ap.error(f"unknown steps: {sorted(unknown)}")
        steps = [s for s in STEPS if s.name in args.only]
    if args.list:
        for s in STEPS:
            mode = "quick" if s.quick else "full"
            mins = "?" if s.minutes is None else f"{s.minutes:g}"
            print(f"{s.name:36s} {mode:5s} ~{mins:>4s} min  needs={','.join(sorted({n.split('/')[0] for n in s.needs})) or '-'}")
        return 0

    top, prefix = repo_prefix()
    rev = git("rev-parse", args.rev)
    tmp = Path(tempfile.mkdtemp(prefix="reproduce_", dir=args.workdir))
    work, log_dir = tmp / "tree", tmp / "logs"
    work.mkdir()
    log_dir.mkdir()
    print(f"revision {rev[:12]}, clean copy in {work}")
    build_clean_copy(rev, prefix, work)
    source = args.external or default_external()
    ext = stage_external(work, source, args.fetch)
    for rel, st in ext.items():
        if st != "ok":
            print(f"  external {rel}: {st}")

    threads = str(args.workers)
    env = dict(os.environ, OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
               VECLIB_MAXIMUM_THREADS=threads, NUMEXPR_NUM_THREADS=threads, PYTHONHASHSEED="0", MPLBACKEND="Agg",
               PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join([str(work), str(work / "src")]))

    rows: List[Dict[str, Any]] = []
    for step in steps:
        missing = [n for n in step.needs if ext.get(n) != "ok"]
        if missing:
            for out in step.outputs:
                rows.append({"step": step.name, "output": out.path, "verdict": "skipped",
                             "detail": f"missing public input: {', '.join(missing)} (use --external or --fetch)"})
            print(f"[skip] {step.name}: missing {', '.join(missing)}")
            continue
        print(f"[run ] {step.name} ...", flush=True)
        rc, secs = run_step(step, work, args.workers, env, log_dir)
        for out in step.outputs:
            if rc != 0:
                verdict, detail = "error", f"exit code {rc}, see {log_dir / (step.name + '.log')}"
            else:
                verdict, detail = compare(out, committed(rev, prefix, out.path), work / out.path)
            rows.append({"step": step.name, "output": out.path, "verdict": verdict, "detail": detail,
                         "seconds": round(secs, 1), "note": out.note})
            print(f"       {verdict:16s} {out.path}  [{secs:.0f} s] {detail}")

    good = {"exact", "within_tolerance"}
    failed = [r for r in rows if r["verdict"] not in good | {"skipped"}]
    report = {"revision": rev, "mode": "quick" if args.quick else ("only" if args.only else "full"),
              "workers": args.workers, "python": sys.version.split()[0], "external": ext, "rows": rows,
              "n_reproduced": sum(r["verdict"] in good for r in rows), "n_failed": len(failed),
              "n_skipped": sum(r["verdict"] == "skipped" for r in rows)}
    (tmp / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    if args.report:
        args.report.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\n{report['n_reproduced']} reproduced, {report['n_failed']} failed, {report['n_skipped']} skipped"
          f" of {len(rows)} outputs. Report: {tmp / 'report.json'}")
    if not args.keep:
        shutil.rmtree(tmp / "tree", ignore_errors=True)
        print(f"removed {work} (logs and report kept in {tmp}; use --keep to keep the tree)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
