#!/usr/bin/env python3
"""Rerun every CPU-only, public-input step and compare with the committed outputs (goal G12).

The script builds a clean copy of the committed tree (`git archive HEAD`) in a temporary
directory, adds the public external tables, runs each step there, and compares what the step
writes with the file committed at HEAD. The working tree is never written to. Untracked files,
cluster caches and licence-restricted data are never copied in, so a step either runs on
public inputs alone or is skipped with the reason.

Every output the paper uses falls in one of four classes, each printed with its own verdict:
  rebuilt        the step reruns from public inputs and committed files, and the result is
                 compared with the committed copy (verdict exact or within_tolerance)
  hash_verified  the committed file is checked against the sha256 pinned below and NOT rerun:
                 long CPU recoveries in --quick mode (rerun in full mode), outputs made on the
                 cluster from videos, weights or licence-restricted data, snapshots of earlier
                 runs, and hand-curated inputs. The reason is printed per file.
  pending        outputs of analyses not yet run (A1 rotation, A1 same-clip model cells, A2,
                 A4 summary, A5, A6, A7, Figures 2, 4, 5, 6, Table 1). They are listed by path
                 only: this script never runs, opens or stats them. See PENDING.
  skipped        a public input is missing (use --external or --fetch)
A5 and A6 are drafts under active change, so a mismatch in one of their files is reported as
draft_changed (a warning) rather than a failure.

Modes:
  --quick   rebuilds every step under about 5 minutes on 2 CPU workers (about 7 minutes in
            all) and hash-verifies the long recoveries instead of rerunning them
  default   quick steps plus the long CPU recoveries (A4, A5, A6, same-clip recoveries, A2
            single-column skew check); these take hours and are meant for a large CPU node

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
    kind: str = "auto"  # auto, json, csv, png, pdf, text, sha256, expected
    ignore: Tuple[str, ...] = ()  # dotted key paths or bare key names skipped anywhere
    only: Tuple[str, ...] = ()  # compare only these top-level keys
    mc_keys: Tuple[str, ...] = ()  # keys holding Monte Carlo p-values, compared at mc_tol
    mc_tol: float = 0.01
    unordered: Tuple[str, ...] = ()  # keys holding lists whose order carries no meaning
    rel: float = 1e-9
    abs_: float = 1e-12
    sha256: Optional[str] = None  # for kind sha256: pinned value instead of a committed file
    ref: Optional[str] = None  # committed file to compare with, when it is not `path` itself
    expected: Optional[Dict[str, Any]] = None  # for kind expected: values pinned from the registry
    draft: bool = False  # output of an unfrozen draft (A5, A6): a mismatch is a warning
    note: str = ""


@dataclass
class Step:
    name: str
    cmd: Optional[List[str]]  # None: verify the committed outputs by pinned sha256, run nothing
    outputs: List[Out]
    needs: Tuple[str, ...] = ()
    quick: bool = True
    minutes: float = 0.1  # wall time on 2 CPU workers, Apple M-series, measured 2026-11-08
    pre: Optional[str] = None  # name of a setup action in PRE
    stdout: Optional[str] = None  # write the command's stdout to this path in the tree
    stdout_key: Optional[str] = None  # wrap stdout lines as {key: [lines]} instead of writing it raw
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
    Step("a4_gates", _py("research/scripts/a4_gates.py"),
         [Out("research/reproduction/a4_gates.json")], minutes=0.1,
         note="A4's --gates file, built from the locked A1, A3 and vision-gate files by A2's frozen outside_gates "
              "(PREREG_A4 deviation of 2026-11-12, ruling 5)"),
    Step("snoek_config_overlap", _py("-I", "research/scripts/snoek_config_overlap.py"),
         [Out("research/reproduction/snoek_config_overlap.json")], needs=SNOEK, minutes=0.3,
         note="Snoek AU configurations shared between the basic and conversational experiments (draft review "
              "2026-11-12, finding 7)"),
    Step("a9_label_corroboration", _py("-I", "research/scripts/a9_label_corroboration.py"),
         [Out("research/reproduction/a9_label_corroboration.json", ignore=("inputs_sha256",),
              note="input hashes skipped: human_au_effects.json is rebuilt earlier in the same run with an unordered "
                   "pair in tie order, so its bytes can differ; both inputs are checked elsewhere, values in full")],
         minutes=0.1,
         note="A9 label-order corroboration from the two committed human-target files (draft review 2026-11-12, "
              "finding 12)"),
    Step("residual_cea_attenuation",
         _py("research/scripts/residual_cea_attenuation.py", "--out", "research/reproduction/residual_cea_attenuation.json"),
         [Out("research/reproduction/residual_cea_attenuation.json")], minutes=2.7,
         note="G9 numerical check of Proposition 5 for residual CEA with estimated projectors (Supp. Note 1, section 4); "
              "synthetic matrices only"),
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
    # ---------------------------------------------------------------- counts the registry cites, no output file
    Step("registry_derived_counts", _py("research/scripts/registry_derived_counts.py"),
         [Out("results/reproduce/registry_derived_counts.json", kind="expected",
              expected={"v1_pilot": {"images": 432, "face_found": 432, "pairs": 144, "pairs_rising_detector_gate": 113,
                                     "share_rising_detector_gate": 113 / 144, "v1_pass": True, "intensity_pairs": 96,
                                     "presence_pairs": 48, "intensity_pairs_gain_at_least_0p5": 22,
                                     "of_which_monotonic_rise": 20,
                                     "columns_with_gain_at_least_0p5": ["AU01", "AU05", "AU25", "AU26"],
                                     "au12_at_amplitude_1_min_max": [0.014, 0.031], "presence_pairs_rising": 43,
                                     "of_which_score_below_0_at_amplitude_1": 35},
                        "grid_frame_loss": {"clips": 243, "clips_losing_frames_to_grid": 68,
                                            "clips_within_0p01_s_of_9_s": 3},
                        "literature_table": {"papers": 8,
                                             "target_position_code": {"na": 3, "no": 2, "not_stated": 1,
                                                                      "partial": 1, "yes": 1},
                                             "option_sets_code": {"na": 2, "not_stated": 2, "partial": 1, "yes": 3},
                                             "delivery_check_code": {"not_stated": 8},
                                             "face_dependence_code": {"no": 5, "partial": 3},
                                             "rows_with_inferred_cells": 1}},
              note="the script prints and stores nothing; values pinned from RESULTS_REGISTRY.md "
                   "(A4 V1 pilot; Input delivery, 68 of 243; Literature table)")],
         stdout="results/reproduce/registry_derived_counts.json",
         note="from the committed a4_v1_pilot.csv, stimulus_features.json and literature_table.csv"),
    # ---------------------------------------------------------------- paper figures, last: they read the files above
    Step("table1", _py("research/scripts/make_table1.py"), [Out("research/paper/v2/table1.md")], minutes=0.1,
         note="Table 1 from locked outputs only; A2, A4 and A9 cells read pending until their run notes exist"),
    Step("paper_figures_1_3", _py("research/scripts/make_figures.py", "--all"),
         [Out("research/paper/v2/figures/fig1.png"), Out("research/paper/v2/figures/fig3.png"),
          Out("research/paper/v2/figures/fig1.pdf",
              note="matplotlib draws some patches in set order, so the PDF is compared as the same drawing "
                   "operators in any order; the PNG is the pixel check"),
          Out("research/paper/v2/figures/fig3.pdf"),
          Out("research/paper/v2/figures/SOURCES.md", note="per-panel input sha256 and script line")],
         minutes=0.1, pre="committed_figure_inputs",
         note="Figures 1 and 3 from the committed copies of their inputs, so SOURCES.md records the committed "
              "sha256 (prompt_confound_audit.json is rebuilt above only within tolerance: its Monte Carlo p-values, "
              "which no panel plots, differ in the last digits); Figures 5 and 6 are pending (PENDING)"),
    Step("paper_figure_2", _py("research/scripts/make_figures.py", "--fig", "2", "--real"),
         [Out("research/paper/v2/figures/fig2.png"), Out("research/paper/v2/figures/fig2.pdf")],
         minutes=0.1, note="Figure 2 from the locked A1 and A3 summaries, which are hash-verified records"),
    Step("paper_figure_4", _py("research/scripts/make_figures.py", "--fig", "4", "--real"),
         [Out("research/paper/v2/figures/fig4.png"), Out("research/paper/v2/figures/fig4.pdf")],
         minutes=0.1, note="Figure 4 from the locked natural-clip files (hash-verified) and the route recovery files"),
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
    Step("a2_gain_skew_single_column", _py("research/scripts/a2_gain_skew_single_column.py"),
         [Out("results/reproduce/a2_gain_skew_single_column.json", kind="json",
              ref="research/reproduction/a2_gain_skew_sim.json", only=("single_column_check",),
              note="seeded (rng 7); compared line for line with single_column_check in the committed file")],
         quick=False, minutes=50, stdout="results/reproduce/a2_gain_skew_single_column.json",
         stdout_key="single_column_check",
         note="the matrix-level part of the A2 skew check, one process; the 1000-replicate maxT part "
              "(a2_gain_skew_sim.py, hours on many workers, no quick mode) is hash-verified only"),
    Step("a4_recovery",
         _py("-m", "scripts.a4_recovery", "--manifest", "results/a4/manifest.csv",
             "--out", "research/reproduction/a4_recovery.json", "--workers", W),
         [Out("research/reproduction/a4_recovery.json")], needs=SNOEK, quick=False, minutes=None,
         note="needs the a4_manifest step first"),
    Step("a6_recovery", _py("-m", "scripts.a6_recovery", "--workers", W),
         [Out("research/reproduction/a6_recovery.json", draft=True)], needs=CREMA, quick=False, minutes=180,
         note="10,832 s on 3 workers"),
    Step("a6_recovery_both_directions",
         _py("-m", "scripts.a6_recovery", "--variant", "both_directions", "--workers", W,
             "--output", "research/reproduction/a6_recovery_both_directions_paired.json"),
         [Out("research/reproduction/a6_recovery_both_directions_paired.json", draft=True)], needs=CREMA, quick=False, minutes=17),
    Step("a6_recovery_first_draft",
         _py("-m", "scripts.a6_recovery", "--variant", "first_draft", "--workers", W,
             "--output", "research/reproduction/a6_recovery_draft_as_written.json"),
         [Out("research/reproduction/a6_recovery_draft_as_written.json", draft=True)], needs=CREMA, quick=False, minutes=1),
    Step("a6_recovery_happy_equivalence",
         _py("-m", "scripts.a6_recovery", "--variant", "happy_equivalence", "--workers", W,
             "--output", "research/reproduction/a6_recovery_happy_equivalence.json"),
         [Out("research/reproduction/a6_recovery_happy_equivalence.json", draft=True)], needs=CREMA, quick=False, minutes=2),
    Step("a5_recovery", _py("-m", "scripts.a5_recovery", "--workers", W),
         [Out("research/reproduction/a5_recovery.json", draft=True)], quick=False, minutes=None,
         note="A5 is a draft under active change; a mismatch here may only mean the draft moved on"),
]


# ------------------------------------------------------------------------------- hash-verified outputs
# sha256 of each committed output that is not rebuilt here (or, for long recoveries, not rebuilt in
# --quick mode). Pinned from the committed tree at d6e190a (= origin/main ee92354 for these files),
# 2026-11-08. A relock that changes one of these files must update its pin in the same commit.
PINS: Dict[str, str] = {
    "research/reproduction/a1_nota_shares.json":
        "eca6970474a439dc7c7415674c15ee44babc4c9d18dc5b999d87e4a2854a37ce",
    "research/reproduction/a1_frame_strategies.json":
        "164adee7b0f9777b94345ad977507d05fd3f465130bf32c803059eea6b5b5829",
    "research/reproduction/a3_token_floor_sensitivity.json":
        "672d2b641883946c620427ab5b8a2d48b29a361acd941b14ac34dc649b89da2d",
    "research/reproduction/a7_recovery.json":
        "bf5abf634973c825dbfadefb2449e253e7c81422cd17e526773e446a1e493295",
    "research/reproduction/a7_recovery_level1.json":
        "ac0dff06b30b5c1ecd366f3ba8b17c73b9c3e32ed67c1b440fe10ec53ca97a91",
    "research/reproduction/a2_gain_skew_sim.json":
        "e1706b804cb6197f13a48fae33888f1a1ca8e56ed09cfea49396b3b4522179ce",
    "research/reproduction/a3_nota_shares.json":
        "7b81f709438d3d8ce1ee7929545ec800e8c797dac3719fe7625473d213cfd199",
    "research/reproduction/a4_recovery.json":
        "1ebfc8ea00da4261e3a05e6de09b3fbe7bd259ce0a5d840fe4c7ab03894bedca",
    "research/reproduction/a4_recovery_first_run.json":
        "a0fe98dc3a77d3b445cb075796c627335d1de87d26442fe7a2728692d4ba7df9",
    "research/reproduction/a4_recovery_run3.json":
        "69efb5de0d7690b6a1f2879ad5045b2c6369893b95e0e115326d65dc1a171ff9",
    "research/reproduction/a4_recovery_second_run.json":
        "64d7d1c91af62c760b21ecb0109abb81dc22f96278ba1948fab307af507b15bf",
    "research/reproduction/a4_v1_pilot.csv":
        "c735615a034c281909c96894c700d224638e281f55f995a0959eeef8a6fb3994",
    "research/reproduction/a5_checks.json":
        "2c500a885db9d3a356ecb3fcd6c7586647d42ae35d10e9b179609ff9fb1de23a",
    "research/reproduction/a5_family_diagnostic.json":
        "67ac4c90a82811cb3ebc8576204693877308f9ad1785fe67282b4070431183cb",
    "research/reproduction/a5_recovery.json":
        "baa960ca1dd524f696f2a0a075ec8a746fe63df1e243cc4c62f8dbd4825ebe20",
    "research/reproduction/a5_recovery_run1.json":
        "92ee63d07cdc4d8a4820bfa5709fcbda84f473211e8672a3fc4d0c1ec5fdb587",
    "research/reproduction/a5_recovery_run2.json":
        "3460cf006fa97961fecec310b806a85100dcb2a056d72d535a4e68f2c20c7456",
    "research/reproduction/a5_recovery_run3.json":
        "0733a4af590b1338782b4bbb86438d238814e807b078573d9ddb688975dd63e9",
    "research/reproduction/a5_recovery_run4.json":
        "fb8bab4ae9f894b294fcb6aca46ba2a43df1e12414e7be85f4c35c02ff152cb9",
    "research/reproduction/a5_recovery_run4_as_written.json":
        "3cd87b83f4959964d79baabe4e1009548f89075b8601ccd94c5a490c36aa9aba",
    "research/reproduction/a5_recovery_run5.json":
        "8719d90fc305b1a117b1c0280670e52633e6e973e5e56dde68adda432cecbf51",
    "research/reproduction/a5_run3_calibration_check.json":
        "7c0345949715e3877d4fe4ca2b39562ab0cce4035045cc482e3199e429b2e2c1",
    "research/reproduction/a5_run3_dependence_loo.json":
        "91e9f011e9e5955599e7011de5e06218cb0c2f1230ef78f7d04e72ff714383c2",
    "research/reproduction/a5_run3_dependence_oracle.json":
        "3c6df0df26965914c8089bb5efb8c3b40f2a0adca1be9981009b31c3b0930256",
    "research/reproduction/a5_run3_recalibrated_diagnostic.json":
        "43489fd4b310a7626ae73f16f4953be60d9d15f312196b1c8a200c3178a69f96",
    "research/reproduction/a5_run4_skew_diagnostic.json":
        "47689ef3dcfb777a7cdd73d04d5ec66e0748f6bf69a95c4e5004a237de77d489",
    "research/reproduction/a5_run5_gain_guard_diagnostic.json":
        "b2b6af7b0da52bb82c1ab5acfb65f5bbe1dafc82dd29ef2f22605e5ccdc0c5b6",
    "research/reproduction/a6_recovery.json":
        "9d8be7ba8934de6ad9f82934efaf9fc7382c6628b9ce13d6005b720aaa7593a7",
    "research/reproduction/a6_recovery_both_directions_paired.json":
        "9d245b72ff17502afbeada91ce52f6adb58d5a1d0ccef4cf08200bbe033ed8ea",
    "research/reproduction/a6_recovery_draft_as_written.json":
        "6783f6e4a97076456e156719f41847aa1ebcdfe16e0fcbfd4eaa7d5818b771bc",
    "research/reproduction/a6_recovery_happy_equivalence.json":
        "4222717539665b48e7d9b936f226cc0be6c55d10f92f21be08da92211bb8d450",
    "research/reproduction/a6_recovery_run1.json":
        "b04aa9825d2704425a7889d0570bdd66cf8aba6dbde8eeecbc31e847f680e250",
    "research/reproduction/a6_recovery_run2.json":
        "44e54de49612e96e7f5ad260b1d30e64cc5f5a652a63bdc24b3c344626bd8e34",
    "research/reproduction/a6_recovery_run4.json":
        "198d3293ec1583720fc474101c0d9ac032d45aa9f3024bc5705d20b53c4f1f20",
    "research/reproduction/activation_meta/gemma4_extract_meta.json":
        "3e1bf219ae0877bd5cd52fdecf4d169dad4caac1e551e20cc7a424afb55e97a9",
    "research/reproduction/activation_meta/molmo2_extract_meta.json":
        "02b78707a42211e7666cb0218ec76952f95ec08208fd7ed3f314816653afe8ee",
    "research/reproduction/activation_meta/qwen3vl_extract_meta.json":
        "9cf23106c7cfdf33e684d3184fd282ba6a86132055b4f79a928d45a4f47b82eb",
    "research/reproduction/cache_parity_qwen3vl_bfloat16.json":
        "a1026494edf1d6dce695a18dc5c9b21b1f688b1ca8c7cde4e5256b4b9f85369b",
    "research/reproduction/cache_parity_qwen3vl_float32.json":
        "8c2815a858d841207e281c772cda624644345445d71b11f1cfa12f9948d79a1d",
    "research/reproduction/crema_d_video_check.csv":
        "5d6f56c16931eca608a7680e6922f61f320e2d4e1f11a3b47457e40d322c06aa",
    "research/reproduction/eu_au_extraction_graphau_summary.json":
        "11028f130cb4d465346163c9d7daa81c4ad7a0811dcaa3167bd2783c587c4eb8",
    "research/reproduction/eu_au_measures.csv":
        "a148351851e4487ca83d0e5719dad25a99acd7a8d3ce8e00f8a77e4fa47db655",
    "research/reproduction/eu_au_measures_graphau.csv":
        "44ecc175f3125c8abd52601b8dcc0fd013edee90d77c89f62a888fb0e2d68cfb",
    "research/reproduction/face_input_validity_summary.json":
        "c1c64e6975b32356a3314268c054f0dee483ddf74b11cd66d03b2faf972e64a4",
    "research/reproduction/input_delivery_gemma4.json":
        "94f55b9fd96767e0c19c0bbeb34eebd5a47dbbbf5b14058d53d6f15a8b40a8ca",
    "research/reproduction/input_delivery_molmo2.json":
        "200ede896d13ccbbea15f7a5e3a561bac9dc7cc35c7ecc3286f4382dd963ec9b",
    "research/reproduction/input_delivery_qwen3vl.json":
        "802f3231db29f2b27d67e8a6cfd70c78a9ea72c9e3fb4b1cadf7f8c966f1f801",
    "research/reproduction/literature_table.csv":
        "48a7005d3a5ea695a4646f3e42e4117a91929c3ee7b786826e2a7cb2995820db",
    "research/reproduction/phase2_gate.json":
        "e5319dd49d354e431e73343ce1dee3d2e0ee45b69f3eff423bb9aa33fd60c489",
    "research/reproduction/phase2_gate_all_layers_gemma4.json":
        "7bf8a8530909a7ab2f659703e75daf668e37f39cc62d4ce8543e5265c38e30cd",
    "research/reproduction/phase2_gate_all_layers_molmo2.json":
        "14a77971705279855eb2e2d9f494a1dfa8a349dd43e9f693a09b8f889868923e",
    "research/reproduction/phase2_gate_all_layers_qwen3vl.json":
        "fbeef879f7aea142d92a8cb70c61d5ac19990161f050738396bd3b22ee6fe66d",
    "research/reproduction/phase5_entanglement.json":
        "ffdae4b61d446dbdc0ebb437ce58fd925c7346829c90d7f88c993aa45580525f",
    "research/reproduction/readout_ladder_gemma4.json":
        "1829eec4dffcc6291a29fc741591acbbf7c5b77b5083511d348b51224a9bf48d",
    "research/reproduction/readout_ladder_molmo2.json":
        "3cfc2cc134337a29fccb163c325dcdf31955e214043cf13c25a2917487f9553f",
    "research/reproduction/readout_ladder_qwen3vl.json":
        "1bc3fa325d4ade19794ccf6435085efdb028b206ccfabb57b5adc8ccc1c3e83e",
    "research/reproduction/same_clip_cue_recovery.json":
        "5af3b8543becd4a801fc7867683c0167cd09c9417833d96f35f3f8f72a6c11e2",
    "research/reproduction/same_clip_family.json":
        "6068767495e2d82052f0b74b89c994b6e19fd5c708355906d1ad5abcc730c29c",
    "research/reproduction/same_clip_family_a2.json":
        "10c87c63602872f9949588546d71edfeec54f253ab5fee5361bf9253ceb4d352",
    "research/reproduction/same_clip_family_a2_matrix.csv":
        "3e72fcd9a713424cd173ebefca47c6ac013581797214890bc625d9ebdc6690cb",
    "research/reproduction/same_clip_family_a2_recovery.json":
        "7bbf84a1c33f342e6dc636599745aa4ed95459a7098e828bf8a13988ad77966e",
    "research/reproduction/same_clip_family_a2_recovery_families_singles.json":
        "722e1f8668478627209c8693d2c58e6ce51758ab5b5096aa03665d47ae31188e",
    "research/reproduction/same_clip_family_a2_recovery_model_frames.json":
        "8e906d557a0445060574a41a6048a08170441ce6c24f37714df86051abf24ff0",
    "research/reproduction/same_clip_family_a2_verdicts.csv":
        "94c0d738331ff78b64bb21607f9fad7b75f5f6109ddcc780d788e0c062c40163",
    "research/reproduction/same_clip_family_matrix.csv":
        "bf281a18e737a0357c444ef612cdcb538571f4f22e1d94743adc3c8f84e46329",
    "research/reproduction/same_clip_family_recovery.json":
        "922979b3d5e4da1da1b27d2896f9dd50f5515842d32905989bd642eff5978f71",
    "research/reproduction/same_clip_family_recovery_families_singles.json":
        "195900675c9380158d6d3147b203cc6e220a83e86aeb521d5f67b82079404801",
    "research/reproduction/same_clip_family_route_pattern_075x_within_actor.json":
        "da4225be268ce2d23277aa970bbf8410f99cb89e5108c7976a3de8ef32883067",
    "research/reproduction/same_clip_family_route_pattern_075x_within_category.json":
        "3cec1cf52258446df5ad7d917a048af0866fc61c43753add164362b168b56762",
    "research/reproduction/same_clip_family_route_pattern_within_actor.json":
        "e69d03d2cc7f2b92cd2679b0fc1be11549304ffa36cc1554012e9b36f4436d81",
    "research/reproduction/same_clip_family_route_pattern_within_category.json":
        "2bed8ba2dea187d48bd968c2048cb9ec49945ed0e22bb68e0a9946d1209cba8a",
    "research/reproduction/same_clip_family_verdicts.csv":
        "79a320b7f4d736ea5e8b58caf7a6e3c24fffcfbe6942530631dee358485d813c",
    "research/reproduction/stimulus_features.json":
        "4b803635cbef566406070c285856752fe0f628346b451b17b80078e4007ebf4e",
    "research/reproduction/vision_gates_gemma4.json":
        "b9e5743f72c9a9271aa79d2fa0d2f446030793b4e0959a2acec7bbaa7057d764",
    "research/reproduction/vision_gates_molmo2.json":
        "ee9935ec0c5a069238a4a62085e189ce53995373e383006bed7f1a4cbdb27e45",
    "research/reproduction/vision_gates_qwen3vl.json":
        "2f38aafb41c93505b792bd355c72d2a2b30e841a0e29be067a1d7c5ba01ede8d",
    "results/baseline/eu_emotions/gemma4/eval_v2_eu_emotions_gemma4_video_only_seed42.json":
        "0e9afc244078dcf874f8c27d933ef38cf9a04af7e38626bd7c5cffde4f9957d8",
    "results/baseline/eu_emotions/molmo2/eval_v2_eu_emotions_molmo2_video_only_seed42.json":
        "fca89f5d0b577b018df834ce95d481393a6daba28a1e18c1148f51cbe6c65a6f",
    "results/baseline/eu_emotions/qwen3vl/eval_v2_eu_emotions_qwen3vl_video_only_seed42.json":
        "135b6dde4e50b18370d40356be896432ae4798df8253ff38b147e68a4dd85f46",
    "results/mech/axis_confusability_qwen3vl_layer4.npy":
        "1f956f2dc609a8a4096d65fc34b5f96359267868dc367e5c22589e302f806176",
    "results/mech/axis_entropy_qwen3vl_layer4.npy":
        "263497c87fa3defd97ee20de60640172f79cf93883a4ad6b5e15c11e2ec65da7",
    "results/mech/axis_pair_bored_unfriendly_qwen3vl_layer4.npy":
        "a29d8470c32ab5ae1cf13301f16490563005c33afbaf145246aa9b1d132ac633",
    "results/mech/axis_pair_disappointed_worried_qwen3vl_layer4.npy":
        "67d746858e1fe4435f7efec4957fb5fe6aa6eadabc2407152fec9b10e1135d6c",
    "results/mech/axis_pair_interested_kind_qwen3vl_layer4.npy":
        "ffcb87ea2e7cd2a1631dde40d009cae3158d6ebd3d12c2af8c3860d7d979a696",
    "results/mech/axis_random_qwen3vl_layer4.npy":
        "d5755524b6132f921924737176a563fd437294cd02bfe8644110b4e68700c026",
    "results/mech/gemma4_confusability_probe_rsa.json":
        "922c3f629156385bd7f07ebc628da8d6c96596c6b5a489a09b95ef42ac64adf1",
    "results/mech/molmo2_confusability_probe_rsa.json":
        "ba1df96ee877659acf46554ff039f90a7a8106efaec870aed4f1e743805c8ad6",
    "results/mech/pilot_n36/ablate_summary_qwen3vl_layer4.csv":
        "667e176fe06d5e4c17a0cfbb52a18c3bd6977e9a9adeff24fcc7ee62d11709a4",
    "results/mech/qwen3vl_confusability_probe_rsa.json":
        "22f7b0e6c1e1b2d3e846f3de1651c05a697818ae02af5a45179141e63543c5ba",
    "results/mech/qwen3vl_eu_ablation_layer4.json":
        "7b9c9cc28f015136393287e3b31f6fa08bdf3c02b06b0db4e46cb4e3bf6db958",
    "results/mech/qwen3vl_eu_causal_axes_layer4.json":
        "4315aa3dde0de1b977dbf39f28a1992b8cc9b1b0208849eb4dd5a3b72bdf0165",
    "results/mech/steer_summary_qwen3vl_layer4.json":
        "f39bcd09b1da62f2d818fdf223cfd56c1df03dad8aa7843f92e04080e8d98b1a",
    # Added 2026-11-12: outputs committed 2026-11-09 to 2026-11-12 (A1 lock, A6 and A7 cluster checks,
    # A9, benchmark audit, EU-Emotion checks), pinned from the tree that commits them.
    "research/reproduction/rotation_summary.json":
        "81500bdd229f88c5aa4038736d4ccc0eb2e04080613dc95380cc8c3dd067072a",
    "research/reproduction/rotation_errored_trials.json":
        "ec876ebf090b9cd528887bebdee9b8afc9f6bcc2a6c46d05f39177c3b9a336cf",
    "research/reproduction/a1_pull_manifest.json":
        "c769701261f3ff008a3f2d5c4407bdea9599de12e44afda0a26789de29bf5664",
    "research/reproduction/same_clip_family_model_gemma4.json":
        "0f9c0a2658cb15c4ac5e0ec03d87fa1f4a521a71ba5e758d54993e96ac37519d",
    "research/reproduction/same_clip_family_model_qwen3vl.json":
        "bc0d67951cad79abb28cae1697f96946ec87e6f83eb6096d9cd772ad5cedb810",
    "research/reproduction/same_clip_family_model_molmo2.json":
        "9ad3d59dd71b8c2a4f897ac8f05d1c9ed805f9e566c75ceb3651d2eafaf66e3f",
    "research/reproduction/a6_human_cue_check.json":
        "f42a51ef02455afbb26a4ba530eb231afa4db72cc5e8b9813ab1cb5ae889aeec",
    "research/reproduction/a7_stage0_gemma4.json":
        "fff021f80af84864592300357228adb2875eaadc52b711964c723ed62b0b12b4",
    "research/reproduction/a7_stage0_qwen3vl.json":
        "3762c7bdb48c6a0b153d13a8f2d076a6b82a8d4a948bcec70be7d08e0da8a215",
    "research/reproduction/phase2_hash_check.json":
        "1f4949e4869eaa1e7f4c0804166774d2cd28b96f027572c689a39f655af1523e",
    "research/reproduction/eu_duplicate_clip_check.json":
        "a9f61e39b605375c61b58b2fea01001ec42efc0fb8e1b6dd8d279c5d654ed823",
    "research/reproduction/eu_missing_items_check.json":
        "b6a18f2479f01d080f7c8e2b44075753fc3ed318c97489fde553d1ed1c746b8f",
    "research/reproduction/benchmark_leak_audit.json":
        "86a93812fb236a17dbc3b788cabeef11288084801c8e67d43697f44f20e3cb49",
    "research/reproduction/a9_human_target.json":
        "3f93ccbe2506ad9e01d79c8f594da6eb0bcd33790d410fda2d3945aeaa526ab0",
    "research/reproduction/a9_human_ceiling.json":
        "a32e2dbe2e41b00b19594be1261d4aebe78dfb6170e3ab0837d3f612e62de669",
    "research/reproduction/a9_chen_overlap.json":
        "793fa184215f566ef4899d8d3222a55401c03622c93b8950775a6b092f374a44",
    "research/reproduction/a9_recovery.json":
        "0ca7bd249fc0c0f705a5d7676e6eb30db88cb019d5a30c71195d67d20b752d68",
    "research/reproduction/a9_recovery_run1_draft.json":
        "625675e81eeda31f54f67e7fe2d1cf9b1330089f1b2edbf050f4ba326f680afe",
    "research/reproduction/a9_recovery_run2.json":
        "79229b70c4a7d21cfcf9bd957669a899c9b503082932e77cddf94d4aced56e84",
    "research/reproduction/a9_recovery_run3.json":
        "01958263f95c6a2fb614d7e66093eb6ed61e11ede33415d14bc8fab8d076dd85",
}

R = "research/reproduction/"

# Made on the cluster from the EU-Emotion or CREMA-D videos, model weights, activation caches or
# AU detector weights, none of which is public; the committed file is the record.
CLUSTER_RECORDS = [R + f for f in (
    "vision_gates_gemma4.json", "vision_gates_qwen3vl.json", "vision_gates_molmo2.json",
    "input_delivery_gemma4.json", "input_delivery_qwen3vl.json", "input_delivery_molmo2.json",
    "readout_ladder_gemma4.json", "readout_ladder_qwen3vl.json", "readout_ladder_molmo2.json",
    "face_input_validity_summary.json", "a3_nota_shares.json", "stimulus_features.json",
    "phase2_gate.json", "phase2_gate_all_layers_gemma4.json", "phase2_gate_all_layers_qwen3vl.json",
    "phase2_gate_all_layers_molmo2.json", "phase5_entanglement.json",
    "cache_parity_qwen3vl_bfloat16.json", "cache_parity_qwen3vl_float32.json",
    "crema_d_video_check.csv", "a4_v1_pilot.csv",
    "eu_au_measures.csv", "eu_au_measures_graphau.csv", "eu_au_extraction_graphau_summary.json",
    "activation_meta/gemma4_extract_meta.json", "activation_meta/qwen3vl_extract_meta.json",
    "activation_meta/molmo2_extract_meta.json")] + [
    EVAL.format(m=m) for m in ("gemma4", "qwen3vl", "molmo2")] + ["results/mech/" + f for f in (
        "gemma4_confusability_probe_rsa.json", "qwen3vl_confusability_probe_rsa.json",
        "molmo2_confusability_probe_rsa.json", "qwen3vl_eu_causal_axes_layer4.json", "qwen3vl_eu_ablation_layer4.json",
        "steer_summary_qwen3vl_layer4.json", "pilot_n36/ablate_summary_qwen3vl_layer4.csv",
        "axis_confusability_qwen3vl_layer4.npy", "axis_entropy_qwen3vl_layer4.npy", "axis_random_qwen3vl_layer4.npy",
        "axis_pair_bored_unfriendly_qwen3vl_layer4.npy", "axis_pair_disappointed_worried_qwen3vl_layer4.npy",
        "axis_pair_interested_kind_qwen3vl_layer4.npy")]

# Snapshots of earlier recovery and diagnostic runs, made by earlier revisions of the code; the
# current code writes the latest run (rebuilt or hash-verified above), not these.
SNAPSHOTS = [R + f for f in (
    "a4_recovery_first_run.json", "a4_recovery_second_run.json", "a4_recovery_run3.json",
    "a5_recovery_run1.json", "a5_recovery_run2.json", "a5_recovery_run3.json", "a5_recovery_run4.json",
    "a5_recovery_run4_as_written.json", "a5_recovery_run5.json", "a5_checks.json", "a5_family_diagnostic.json",
    "a5_run3_calibration_check.json", "a5_run3_dependence_loo.json", "a5_run3_dependence_oracle.json",
    "a5_run3_recalibrated_diagnostic.json", "a5_run4_skew_diagnostic.json", "a5_run5_gain_guard_diagnostic.json",
    "a6_recovery_run1.json", "a6_recovery_run2.json", "a6_recovery_run4.json",
    "a9_recovery_run1_draft.json", "a9_recovery_run2.json", "a9_recovery_run3.json")]

# A1, locked 2026-11-09: made by research/scripts/post_a1.sh from the 30 rotation files, which are model
# outputs on EU-Emotion clips and stay on the cluster with the clips.
A1_RECORDS = [R + f for f in (
    "rotation_summary.json", "rotation_errored_trials.json", "a1_pull_manifest.json",
    "same_clip_family_model_gemma4.json", "same_clip_family_model_qwen3vl.json", "same_clip_family_model_molmo2.json")]

# Later cluster checks made from videos, AU frames, model forwards or the cluster's file tree.
CLUSTER_CHECKS = [R + f for f in (
    "a6_human_cue_check.json", "a7_stage0_gemma4.json", "a7_stage0_qwen3vl.json", "phase2_hash_check.json",
    "eu_duplicate_clip_check.json", "eu_missing_items_check.json")]

# Made from public data downloaded outside the repository: Liu et al. 2022 Experiment II (CC BY 4.0,
# Mendeley Data 10.17632/dr853shk56.2), Chen et al. 2024 (OSF 3m95w), and benchmark annotation text.
# Rebuild commands are in each script's docstring.
EXTERNAL_PUBLIC = [R + f for f in (
    "a9_human_target.json", "a9_human_ceiling.json", "a9_chen_overlap.json", "benchmark_leak_audit.json")]


# Descriptive outputs added 2026-11-12 after the independent draft review, made from locked outputs that
# stay off the public repo (the A1 rotation files, the A3 cells, the pinned A2 geometry).
REVIEW_RECORDS = [R + f for f in ("a1_nota_shares.json", "a1_frame_strategies.json", "a3_token_floor_sensitivity.json")]


def _hash_outs(paths: Sequence[str], note: str, drafts: Sequence[str] = ()) -> List[Out]:
    return [Out(p, kind="sha256", sha256=PINS[p], draft=p in drafts, note=note) for p in paths]


HASH_STEPS: List[Step] = [
    Step("hash_long_recoveries", None, minutes=0.0, outputs=
         _hash_outs([o.path for s in STEPS if not s.quick for o in s.outputs if o.path in PINS],
                    "long CPU recovery: verified by hash against the committed output, not rerun in --quick "
                    "(full mode reruns it)",
                    drafts=[o.path for s in STEPS for o in s.outputs if o.draft])),
    Step("hash_cpu_no_quick_mode", None, minutes=0.0, outputs=
         _hash_outs([R + "a2_gain_skew_sim.json"],
                    "A2 skew simulation: 1000 replicates per cell with exact 2^17 flips, hours on many workers "
                    "and no quick mode; verified by hash, not rerun (full mode reruns its single-column part)")),
    Step("hash_a7_recovery", None, minutes=0.0, outputs=
         _hash_outs([R + "a7_recovery.json", R + "a7_recovery_level1.json"],
                    "A7 recovery on the frozen code (tag a7-freeze); Level 2 takes about an hour, so verified "
                    "by hash; rerun with python -m scripts.a7_recovery --level 1|2")),
    Step("hash_a9_recovery", None, minutes=0.0, outputs=
         _hash_outs([R + "a9_recovery.json"],
                    "A9 recovery run 4 on the frozen code (tag a9-freeze); long, so verified by hash; "
                    "rerun with python -m scripts.a9_recovery")),
    Step("hash_a1_records", None, minutes=0.0, outputs=
         _hash_outs(A1_RECORDS, "A1 locked 2026-11-09 by post_a1.sh from rotation cells kept on the cluster; "
                                "verified by hash, cannot be rerun from public inputs")),
    Step("hash_cluster_checks", None, minutes=0.0, outputs=
         _hash_outs(CLUSTER_CHECKS, "cluster checks on videos, AU frames, model forwards or the cluster tree; "
                                    "verified by hash, cannot be rerun from public inputs")),
    Step("hash_external_public", None, minutes=0.0, outputs=
         _hash_outs(EXTERNAL_PUBLIC, "made from public data downloaded outside the repo (Liu 2022, Chen 2024, "
                                     "benchmark annotations); verified by hash, rebuild per the script docstrings")),
    Step("hash_cluster_records", None, minutes=0.0, outputs=
         _hash_outs(CLUSTER_RECORDS, "made on the cluster from non-public inputs (videos, weights, caches); "
                                     "verified by hash, cannot be rerun from public inputs")),
    Step("hash_review_records", None, minutes=0.0, outputs=
         _hash_outs(REVIEW_RECORDS, "descriptive outputs of 2026-11-12 from locked cluster outputs (A1 rotation "
                                    "files, A3 cells, pinned geometry); verified by hash, rebuild per the script "
                                    "docstrings where those files are present")),
    Step("hash_snapshots", None, minutes=0.0, outputs=
         _hash_outs(SNAPSHOTS, "snapshot of an earlier run by earlier code; verified by hash, not rerun")),
    Step("hash_curated_inputs", None, minutes=0.0, outputs=
         _hash_outs([R + "literature_table.csv"], "hand-curated eight-paper table (an input to Figure 1e, "
                                                  "no script makes it); verified by hash")),
]
STEPS = STEPS + HASH_STEPS

# Committed outputs deliberately not yet covered (neither rebuilt nor pinned), with the reason.
# Printed as gaps on every run; tests/test_reproduce.py accepts only what is listed here.
UNCOVERED: Dict[str, str] = {}

# Outputs of analyses that have not run. Listed by path only: nothing here is run, opened or
# stat-ed, so a pending result cannot be read early. Each moves into STEPS when it is locked.
PENDING: List[Tuple[str, Tuple[str, ...], str]] = [
    ("A2 stimulus interventions (Figure 6)",
     tuple(R + f"stimulus_interventions_{m}.json" for m in ("gemma4", "qwen3vl")),
     "scripts/analyse_stimulus_interventions.py; needs cluster cells"),
    ("A4 summary (Figure 5)", ("results/a4/summary.json",), "python -m scripts.a4_analyse; needs cluster cells"),
    ("A5 analysis", ("results/a5/a5_analysis.json",), "python -m scripts.a5_analyse (frozen, tag a5-freeze); needs A2 cells"),
    ("A7 Stage 1 route arm", (R + "a7_analysis.json",),
     "python -m scripts.a7_analyse (frozen, tag a7-freeze); runs only where Gate B and the sham pass"),
    ("A9 summary", ("results/a9/summary.json",), "python -m scripts.a9_analyse (frozen, tag a9-freeze); needs cluster cells"),
    ("Figures 5, 6",
     tuple(f"research/paper/v2/figures/fig{n}.{e}" for n in (5, 6) for e in ("pdf", "png")),
     "research/scripts/make_figures.py --fig N --real, which refuses until its LOCKED inputs exist"),
]


# ------------------------------------------------------------------------------- setup actions

def pre_activation_meta(work: Path) -> None:
    """Stage the committed copies of extract_meta.json where reproduce_headline_numbers.py reads them."""
    for model in ("qwen3vl", "gemma4", "molmo2"):
        dest = work / "results" / "activations" / f"baseline_{model}_6afc" / model / "extract_meta.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(work / "research" / "reproduction" / "activation_meta" / f"{model}_extract_meta.json", dest)


# Inputs Figures 1 and 3 read (make_figures.py, SOURCES.md). Snapshotted from the clean tree before any
# step runs, and put back before the figure step, so a within-tolerance rebuild cannot change SOURCES.md.
FIGURE_INPUTS = ("research/reproduction/prompt_confound_audit.json", "research/reproduction/cfa_identifiability.json",
                 "research/reproduction/literature_table.csv", "data/eu_emotions_human_entropy.json",
                 "scripts/run_stimulus_interventions.py",
                 *[f"research/reproduction/input_delivery_{m}.json" for m in ("gemma4", "qwen3vl", "molmo2")])
SNAPSHOT_DIR = "committed_inputs"


def snapshot_figure_inputs(work: Path) -> None:
    for rel in FIGURE_INPUTS:
        dest = work.parent / SNAPSHOT_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(work / rel, dest)


def pre_committed_figure_inputs(work: Path) -> None:
    for rel in FIGURE_INPUTS:
        shutil.copyfile(work.parent / SNAPSHOT_DIR / rel, work / rel)


PRE = {"activation_meta": pre_activation_meta, "committed_figure_inputs": pre_committed_figure_inputs}


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


def pdf_operators(data: bytes) -> "Counter[bytes]":
    """Multiset of the lines of every (Flate-decoded) stream in a PDF: the drawing operators
    and embedded font data, with their order discarded."""
    import re
    import zlib
    from collections import Counter
    ops: Counter = Counter()
    for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
        body = m.group(1)
        try:
            body = zlib.decompress(body)
        except zlib.error:
            pass
        ops.update(body.split(b"\n"))
    return ops


def compare_pdf(ref: bytes, new: bytes) -> Tuple[bool, str]:
    import re
    boxes = [re.findall(rb"/MediaBox\s*\[[^\]]*\]", x) for x in (ref, new)]
    if boxes[0] != boxes[1]:
        return False, f"page boxes differ: {boxes[0][:2]} != {boxes[1][:2]}"
    a, b = pdf_operators(ref), pdf_operators(new)
    if a == b:
        return True, f"same {sum(a.values())} stream lines in another order (object order and ids differ)"
    only_ref, only_new = sum((a - b).values()), sum((b - a).values())
    return False, f"{only_ref} stream lines only in the committed PDF, {only_new} only in the rerun"


def prune_like(template: Any, value: Any) -> Any:
    """value restricted to the keys present in template, recursively (for kind expected)."""
    if isinstance(template, dict) and isinstance(value, dict):
        return {k: prune_like(template[k], value[k]) for k in template if k in value}
    return value


def compare(out: Out, ref: Optional[bytes], new_path: Path) -> Tuple[str, str]:
    """Return (verdict, detail). Verdicts: exact, within_tolerance, differs, missing_output, no_reference."""
    if not new_path.is_file():
        return "missing_output", "the step did not write this file"
    new = new_path.read_bytes()
    if out.kind == "sha256":
        got = hashlib.sha256(new).hexdigest()
        return ("exact", f"sha256 {got[:12]} matches the pin") if got == out.sha256 else \
            ("differs", f"sha256 {got[:12]} != pinned {out.sha256[:12]}")
    if out.kind == "expected":
        try:
            got_json = json.loads(new)
        except ValueError as e:
            return "differs", f"output is not JSON: {e}"
        diffs: List[str] = []
        compare_values(out.expected, prune_like(out.expected, got_json), out, "", diffs, {})
        if diffs:
            return "differs", "; ".join(diffs[:5])
        return "within_tolerance", "every pinned value equal (values from RESULTS_REGISTRY.md)"
    if ref is None:
        return "no_reference", "no committed copy at this revision"
    if ref == new:
        return "exact", "byte-identical"
    kind = out.kind if out.kind != "auto" else new_path.suffix.lstrip(".")
    if kind == "pdf":
        ok, detail = compare_pdf(ref, new)
        return ("within_tolerance" if ok else "differs"), detail
    if kind in ("md", "txt", "text"):
        ref_lines, new_lines = ref.decode("utf-8").splitlines(), new.decode("utf-8").splitlines()
        for i, (x, y) in enumerate(zip(ref_lines, new_lines)):
            if x != y:
                return "differs", f"line {i + 1}: {x[:70]!r} != {y[:70]!r}"
        return "differs", f"{len(ref_lines)} lines committed, {len(new_lines)} rerun"
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
        if step.stdout is None:
            rc = subprocess.run(cmd, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        else:
            res = subprocess.run(cmd, cwd=work, env=env, stdout=subprocess.PIPE, stderr=log)
            rc = res.returncode
            text = res.stdout.decode("utf-8")
            log.write(text)
            dest = work / step.stdout
            dest.parent.mkdir(parents=True, exist_ok=True)
            if step.stdout_key:
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                text = json.dumps({step.stdout_key: lines}, indent=1) + "\n"
            dest.write_text(text, encoding="utf-8")
    return rc, time.time() - started


def verify_by_hash(step: Step, rev: str, prefix: str) -> List[Dict[str, Any]]:
    """Check each committed output of a verify-only step against its pinned sha256. Nothing runs,
    and the committed bytes are read from git, so a rerun elsewhere in the tree cannot mask a change."""
    rows = []
    for out in step.outputs:
        data = committed(rev, prefix, out.path)
        if data is None:
            verdict, detail = "missing_output", "no committed copy at this revision"
        else:
            got = hashlib.sha256(data).hexdigest()
            if got == out.sha256:
                verdict, detail = "hash_verified", f"sha256 {got[:12]} matches the pin; not rerun"
            else:
                verdict, detail = "differs", f"sha256 {got[:12]} != pinned {out.sha256[:12]}"
        rows.append({"step": step.name, "output": out.path, "verdict": verdict, "detail": detail, "note": out.note})
    return rows


def finish_verdict(out: Out, verdict: str) -> str:
    return "draft_changed" if out.draft and verdict == "differs" else verdict


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="only steps under about 5 minutes; long recoveries are verified by hash instead")
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
            mode = "hash" if s.cmd is None else ("quick" if s.quick else "full")
            mins = "?" if s.minutes is None else f"{s.minutes:g}"
            print(f"{s.name:36s} {mode:5s} ~{mins:>4s} min  needs={','.join(sorted({n.split('/')[0] for n in s.needs})) or '-'}"
                  f"  outputs={len(s.outputs)}")
        for name, paths, producer in PENDING:
            print(f"{'[pending] ' + name:46s} {len(paths)} paths, not run or read; {producer}")
        return 0

    top, prefix = repo_prefix()
    rev = git("rev-parse", args.rev)
    tmp = Path(tempfile.mkdtemp(prefix="reproduce_", dir=args.workdir))
    work, log_dir = tmp / "tree", tmp / "logs"
    work.mkdir()
    log_dir.mkdir()
    print(f"revision {rev[:12]}, clean copy in {work}")
    build_clean_copy(rev, prefix, work)
    snapshot_figure_inputs(work)
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
        if step.cmd is None:
            new_rows = verify_by_hash(step, rev, prefix)
            for row, out in zip(new_rows, step.outputs):
                row["verdict"] = finish_verdict(out, row["verdict"])
            rows.extend(new_rows)
            bad = [r for r in new_rows if r["verdict"] != "hash_verified"]
            print(f"[hash] {step.name}: {len(new_rows) - len(bad)} of {len(new_rows)} committed outputs match their "
                  f"pinned sha256 (verified by hash, not rerun). {step.outputs[0].note if step.outputs else ''}")
            for r in bad:
                print(f"       {r['verdict']:16s} {r['output']}  {r['detail']}")
            continue
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
                verdict, detail = compare(out, committed(rev, prefix, out.ref or out.path), work / out.path)
                verdict = finish_verdict(out, verdict)
            rows.append({"step": step.name, "output": out.path, "verdict": verdict, "detail": detail,
                         "seconds": round(secs, 1), "note": out.note})
            print(f"       {verdict:16s} {out.path}  [{secs:.0f} s] {detail}")

    if not args.only:
        print("\nCommitted but not yet covered (gaps, not failures):")
        for path, why in UNCOVERED.items():
            print(f"[gap ] {path}: {why}")
            rows.append({"step": "uncovered", "output": path, "verdict": "uncovered", "detail": why})
        print("\nPending (not yet produced; listed by path only, never run or read):")
        for name, paths, producer in PENDING:
            print(f"[pend] {name}: {', '.join(paths) or '-'}  ({producer})")
            rows.append({"step": "pending", "output": name, "paths": list(paths), "verdict": "pending",
                         "detail": "not yet produced; not run or read", "note": producer})

    good = {"exact", "within_tolerance"}
    ok = good | {"hash_verified", "skipped", "pending", "draft_changed", "uncovered"}
    failed = [r for r in rows if r["verdict"] not in ok]
    report = {"revision": rev, "mode": "quick" if args.quick else ("only" if args.only else "full"),
              "workers": args.workers, "python": sys.version.split()[0], "external": ext, "rows": rows,
              "n_reproduced": sum(r["verdict"] in good for r in rows),
              "n_hash_verified": sum(r["verdict"] == "hash_verified" for r in rows),
              "n_draft_changed": sum(r["verdict"] == "draft_changed" for r in rows),
              "n_failed": len(failed), "n_skipped": sum(r["verdict"] == "skipped" for r in rows),
              "n_pending": sum(r["verdict"] == "pending" for r in rows),
              "n_uncovered": sum(r["verdict"] == "uncovered" for r in rows)}
    (tmp / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    if args.report:
        args.report.write_text(json.dumps(report, indent=1) + "\n")
    print(f"\n{report['n_reproduced']} rebuilt and matched, {report['n_hash_verified']} verified by hash (not rerun), "
          f"{report['n_draft_changed']} draft changed, {report['n_failed']} failed, {report['n_skipped']} skipped, "
          f"{report['n_pending']} pending, {report['n_uncovered']} uncovered. Report: {tmp / 'report.json'}")
    if not args.keep:
        shutil.rmtree(tmp / "tree", ignore_errors=True)
        print(f"removed {work} (logs and report kept in {tmp}; use --keep to keep the tree)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
