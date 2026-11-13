# REPRODUCE

The one documented path from public inputs to every table and figure in the paper (goal G12 in `research/GOALS.md`, roadmap step 4.5). One command runs it:

```bash
cd bullpy/mr_eu_open_llm
python scripts/reproduce.py --quick --fetch      # about 7 minutes on 2 idle CPU workers
```

It extracts the committed tree (`git archive HEAD`) into a temporary directory, adds the public CREMA-D and Snoek tables (sha256-checked), and then treats every output the paper uses in one of four ways. The working tree is never written to, and the script exits non-zero if any output fails.

| Class | What happens | Verdict printed |
|---|---|---|
| Rebuilt | The step reruns from public inputs and committed files; the result is compared with the committed copy (bytes, or fields at a stated tolerance, or pixels for PNG, or drawing operators for PDF). | `exact`, `within_tolerance` |
| Hash-verified | The committed file is checked against a sha256 pinned in `scripts/reproduce.py` and is **not rerun**. The reason is printed per group. | `hash_verified` |
| Pending | Output of an analysis that has not run. Listed by path only; the script never runs, opens or stats it. | `pending` |
| Skipped | A public input is missing (pass `--external DIR` or `--fetch`). | `skipped` |

A5 and A6 are unfrozen drafts, so a mismatch in their current recovery files is printed as `draft_changed`, a warning, not a failure. `python scripts/reproduce.py --list` prints every step, its class and the pending list. Without `--quick` (full mode) the long CPU recoveries rerun as well; that takes hours and is meant for a large CPU node.

**Updated:** 2026-11-08, against `origin/main` `af84cf2` (same project files as `p1-pipeline` `80b389e`).

## What the paper uses, and how each is reproduced

| Paper item | Output(s) | Class (`--quick`) | Step |
|---|---|---|---|
| Figure 1 (validity audit), panels a to e | `research/paper/v2/figures/fig1.{png,pdf}`, `SOURCES.md` | rebuilt | `paper_figures_1_3` (`research/scripts/make_figures.py --all`) |
| Figure 1 inputs | `prompt_confound_audit.json` (rebuilt), `data/eu_emotions_human_entropy.json` (in repo), `input_delivery_*.json` (cluster), `literature_table.csv` (hand-curated) | rebuilt / hash-verified | `prompt_confound_audit`, `hash_cluster_records`, `hash_curated_inputs` |
| Figure 3 (identifiability), panels b to e | `research/paper/v2/figures/fig3.{png,pdf}` from `cfa_identifiability.json` | rebuilt | `cfa_identifiability`, then `paper_figures_1_3` |
| Phase 2 figures (supplement) | `research/figures/phase2_grouped_t1_rho.png`, `phase2_loo_vs_grouped.png` | rebuilt | `plot_phase2_grouped`, `plot_phase2_loo` |
| Methods tables: cue spectra, A4 human target and ceiling, human response predictor, CREMA-D reliability, A6 human target | `cfa_identifiability.json`, `human_au_effects.{json,csv}`, `a4_human_ceiling.json`, `human_response_predictor.json`, `crema_d_*`, `a6_human_target.json` | rebuilt | quick steps of the same names |
| Behaviour, Phase 0 numbers, calibration, prompt audit, pilot ablation, Phase 2 merge | `behaviour_summary.json`, `phase0_numbers.json`, `results/stats/human_calibration.json`, `prompt_confound_audit.json`, `phase3_pilot_ablation.json`, `phase2_gate_all_layers.json`, `phase2_all_layers_summary.csv` | rebuilt | quick steps |
| A2 recovery; same-clip human cells and criteria | `recovery_stimulus_interventions.json`, `same_clip_cue*.{json,csv}`, `same_clip_family*_criteria.json` | rebuilt | `a2_recovery`, `same_clip_cue_human`, `same_clip_family*_criteria` |
| Counts with no output file (V1 pilot breakdown; 68 of 243 clips lose grid frames) | stdout of `research/scripts/registry_derived_counts.py` | rebuilt, against values pinned from `RESULTS_REGISTRY.md` | `registry_derived_counts` |
| Long recoveries: A4 (run 4), A5, A6 (four files), same-clip family and route-pattern recoveries (22 files) | `a4_recovery.json`, `a5_recovery.json`, `a6_recovery*.json`, `same_clip_family*.json`, `same_clip_cue_recovery.json` | hash-verified in `--quick`; rerun and compared in full mode | `hash_long_recoveries`; full-mode steps |
| A2 skew simulation (DECISIONS 2026-11-08) | `a2_gain_skew_sim.json` | hash-verified (1000 replicates per cell with exact 2^17 flips, no quick mode); its seeded single-column part reruns in full mode (about 50 minutes, one process) | `hash_cpu_no_quick_mode`; `a2_gain_skew_single_column` |
| A3 table and its "None of the above" shares; vision gates; input delivery; readout choice; stimulus features; Phase 2 and 5 dumps; AU tables; V1 pilot CSV; eval JSONs; mechanistic pilot files (43 files) | see Appendix B, cluster table | hash-verified: made on the cluster from videos, weights, caches or licence-restricted renders, so they cannot be rerun from public inputs | `hash_cluster_records` |
| Earlier recovery runs cited in Methods and the supplement (A4 runs 1 to 3, A5 runs 1 to 5 and diagnostics, A6 runs 1, 2, 4) | 20 snapshot files | hash-verified: made by earlier code; to rerun one, check out its commit (Appendix B) | `hash_snapshots` |

## Committed but not yet covered

| Output | Why |
|---|---|
| `research/reproduction/a7_recovery.json`, `a7_recovery_level1.json` | A7 draft recovery, committed in `80b389e` while this was being written and still owned by the A7 work. Listed in `UNCOVERED` in `scripts/reproduce.py` and printed as `[gap ]` on every run; add a step or a pin when A7 freezes. |

## Pending (not yet produced; never run or read by the script)

| Item | Path(s) the frozen code will write | Made by | Feeds |
|---|---|---|---|
| A1 rotation summary | `research/reproduction/rotation_summary.json`, `rotation_errored_trials.json`, `a1_pull_manifest.json` | `research/scripts/post_a1.sh`, `analyse_rotation_run.py` (cluster cells) | Figure 2, Table 1 |
| A1 same-clip model cells | `research/reproduction/same_clip_family_model_{gemma4,qwen3vl,molmo2}.json` | `same_clip_family_analysis.py --model M`, via `post_a1.sh` | Figure 4 |
| A2 stimulus interventions | `research/reproduction/stimulus_interventions_{gemma4,qwen3vl}.json` | `scripts/analyse_stimulus_interventions.py` | Figure 6 |
| A4 summary | `results/a4/summary.json` | `python -m scripts.a4_analyse` | Figure 5 |
| A5, A6, A7 analyses | `results/a5/a5_analysis.json`; `results/a6/a6_stage1.json`, `a6_stage2_{model}.json`; `results/a7/a7_analysis.json` | `scripts/a5_analyse.py`, `a6_analyse.py`, `a7_analyse.py` (draft preregistrations) | Results, supplement |
| Figures 2, 4, 5, 6 | `research/paper/v2/figures/fig{2,4,5,6}.{pdf,png}` | `make_figures.py --fig N --real`, which refuses until its `LOCKED` inputs exist | paper |
| Table 1 | not yet written | from the A1, A2 and A4 outputs | paper |

When one of these is locked: add a step to `STEPS` in `scripts/reproduce.py` (rebuilt if CPU-only on public inputs, otherwise a pinned sha256 in `PINS` and a hash group), remove it from `PENDING`, rerun `--quick`, and update this file. `tests/test_reproduce.py` fails if a committed output under `research/reproduction/` or the figure folders is in neither class, or if a pending path appears in any step.

## Environment

- Conda env `mr_eu_open_llm`. Pinned versions of everything the code imports: `research/reproduction/environment_pinned.yml` (from `conda env export --no-builds`, filtered). The CPU steps need only Python 3.10, numpy, scipy, pandas, scikit-learn, joblib, matplotlib, pillow and statsmodels.
- `git` on the path, since the harness checks out the committed tree with `git archive`.
- Model revisions for the GPU steps: the table at the end of `research/REPRODUCTION.md`.
- Cluster video jobs pin cv2 4.11 and must pass `research/scripts/decoder_preflight.py` (amendment A1.1).

```bash
conda env create -f research/reproduction/environment_pinned.yml   # or use mr_eu_open_llm
conda activate mr_eu_open_llm_repro
cd bullpy/mr_eu_open_llm
python scripts/reproduce.py --list            # every step, mode, minutes, external inputs
python scripts/reproduce.py --quick --fetch   # download public inputs, rerun, compare
python scripts/reproduce.py --fetch           # full mode: adds the long recoveries (hours)
```

`--external DIR` takes the public tables from a folder (layout `crema_d/`, `snoek_au/`) instead of downloading them. Without either flag the harness looks in `data/external` of the main working tree. `--workers N` sets CPU workers and BLAS threads (default 2). `--keep` keeps the temporary tree. The harness exits non-zero if any output fails to reproduce. It never writes to the working tree.


## Appendix A. Inputs and their status

| Input | Where | Status | Used by |
|---|---|---|---|
| EU-Emotion item-level norms (O'Reilly et al.), reduced to `data/eu_emotions_human_entropy.json` by `scripts/human_entropy.py` | in repo | public | behaviour, A1 to A3, same-clip, prompt audit |
| EU-Emotion manifest, definitions, synonyms (`data/eu_emotions_full_manifest.json` and others) | in repo | public (file names and labels only) | most EU-Emotion steps |
| EU-Emotion videos | CSD3 `data/eu_emotions` | not in the public repo (collaborator dataset, kept out for size); the team controls EU-Emotion (confirmed by the user 2026-11-08) | every model run, AU extraction, stimulus features |
| CREMA-D rating tables (`finishedResponses.csv`, `finishedEmoResponses.csv`, `VideoDemographics.csv`, `processedResults/tabulatedVotes.csv`, `processedResults/summaryTable.csv`) | GitHub `CheyneyComputerScience/CREMA-D` at `1658cd34`, ODbL / DbCL; gitignored locally in `data/external/crema_d` | public; sha256 pinned in `scripts/reproduce.py` and checked | CREMA-D selection, reliability, A6 human target |
| CREMA-D videos | CSD3 | licence-restricted (access form) | readout ladder, A6 model runs, video check |
| Snoek AU and emotion data, `au_emotion_data.zip` | DOI 10.21942/uva.21261885.v1, figshare file 37888164, CC BY 4.0; gitignored locally in `data/external/snoek_au` | public; sha256 pinned and checked | human AU effects, A4 manifest and ceiling, predictor, CEA |
| FACSHuman targets and MakeHuman meshes | CSD3 | licence-restricted (LGPL targets, no image is published) | A4 renders, V1 pilot |
| RMET items | not used by any output listed here | licence-restricted | Phase 6 only |
| Model weights (Gemma 4 E4B, Qwen3-VL 8B, Molmo2 O-7B) | Hugging Face, pinned commits | public weights, need a GPU | all model runs |
| AU detector weights (LibreFace, OpenFace 3, ME-GraphAU) | CSD3 venv `au_tools` | research or non-commercial licences | AU extraction |
| Activation caches (`results/activations/`) and rotation, A3 and A4 cells | CSD3 | cluster-only derived data (from videos not in the repo) | Phase 2, Phase 5, A3 summary, prompt audit activations |
| Derived AU tables `research/reproduction/eu_au_measures*.csv` | in repo | public derived numbers (per-clip AU aggregates); remaking them needs the videos | CEA, same-clip analyses |

## Appendix B. Output provenance (audit of 2026-11-08)

Mode: **quick** = run by `--quick`, **full** = run only in full mode, **cluster** = cannot run from public inputs. Minutes are wall time on 2 CPU workers on the test Mac, or the logged cluster time. Result is the clean-checkout test of 2026-11-08 (see the last section).

### Rerun by `scripts/reproduce.py --quick`

| Output | Command (from `bullpy/mr_eu_open_llm`, `PYTHONPATH=.:src`) | Inputs | Min | GPU / data not in the repo | Result |
|---|---|---|---|---|---|
| `research/reproduction/behaviour_summary.json` | `python -m paper_v2.behaviour` | three committed eval JSONs, norms | <0.1 | no / no | exact |
| `research/reproduction/phase0_numbers.json` | `python research/scripts/reproduce_headline_numbers.py` (harness stages `activation_meta/*.json` as `results/activations/baseline_*_6afc/*/extract_meta.json`) | eval JSONs, `results/stats/*`, `results/mech/*` | <0.1 | no / no | every number equal; `activation_cache`, `missing` and eleven `has_*` flags skipped (machine state on 2026-10-30, before `a438754` added fields to the mech JSONs) |
| `results/stats/human_calibration.json` | `python -m scripts.human_calibration --eval <gemma4> <molmo2> <qwen3vl eval JSON> --out results/stats/human_calibration.json` | eval JSONs, norms | <0.1 | no / no | within 1e-9 (last-digit float) |
| `research/reproduction/phase3_pilot_ablation.json` | `python -m paper_v2.causal --trials results/mech/pilot_n36/ablate_trials_qwen3vl_layer4.csv --out ...` | committed trial CSV (from a cluster ablation) | <0.1 | no / no | equal; `source` path skipped |
| `research/reproduction/prompt_confound_audit.json` | `python research/scripts/audit_prompt_confound.py` | eval JSONs, norms; `activations` block needs cluster caches | 0.1 | activations part: cluster | equal except `activations` (cluster) and permutation p-values within 0.004 (tolerance 0.01; 10,000 draws) |
| `research/figures/phase2_grouped_t1_rho.png` | `python research/scripts/plot_phase2_all_layers.py` | `phase2_all_layers_summary.csv` | <0.1 | no / no | same plot; 2 px wider, font rasterising (tolerance stated in the script) |
| `research/figures/phase2_loo_vs_grouped.png` | `python research/scripts/plot_phase2_loo_vs_grouped.py` | same CSV | <0.1 | no / no | same plot within tolerance |
| `research/reproduction/crema_d_selection.csv` | `python -I research/scripts/crema_select.py --data-dir data/external/crema_d` | CREMA-D tables | <0.1 | no / no | exact |
| `research/reproduction/crema_d_reliability.json`, `crema_d_clip_reliability.csv` | `python -I research/scripts/crema_reliability.py --data-dir data/external/crema_d` | CREMA-D tables, selection | 0.5 | no / no | exact |
| `research/reproduction/crema_calibration_clips.csv` | `python -m scripts.readout_ladder --step select` | selection CSV (ids only) | <0.1 | no / no | exact |
| `research/reproduction/a6_human_target.json` | `python research/scripts/a6_human_target.py` | CREMA-D tables, selection | 0.1 | no / no | exact |
| `research/reproduction/human_au_effects.json`, `.csv` | `python -I research/scripts/human_au_effects.py --zip data/external/snoek_au/au_emotion_data.zip --out-json ... --out-csv ...` | Snoek zip | 1 to 2 | no / no | CSV exact; JSON equal, with `largest_au_correlation.pair` compared as an unordered pair (AU02L and AU02R swap on a floating-point tie) |
| `results/a4/manifest.csv` (not committed) | `python research/scripts/a4_stimuli.py manifest --zip ... --out results/a4/manifest.csv` | Snoek zip | <0.1 | no / no | sha256 equals the pin in `PREREG_A4_DRAFT.md` |
| `research/reproduction/a4_human_ceiling.json` | `python -m scripts.a4_analyse --write_ceiling` | Snoek zip, `human_au_effects.json` | 0.3 | no / no | exact |
| `research/reproduction/human_response_predictor.json` | `python -m scripts.human_response_predictor` | Snoek zip, `human_au_effects.json` | 0.3 | no / no | exact |
| `research/reproduction/cfa_identifiability.json` | `python research/scripts/cfa_identifiability.py --au-csv research/reproduction/eu_au_measures.csv --snoek-zip ... --out ...` | Snoek zip, committed AU table | <0.1 | no / no | exact |
| `research/reproduction/recovery_stimulus_interventions.json` | `python -m scripts.recovery_stimulus_interventions --replicates 100` | manifest, norms (no model) | 0.7 | no / no | exact (the committed run used 100 replicates; the script default is 30) |
| `research/reproduction/same_clip_cue.json`, `_matrix.csv`, `_detector_verdicts.csv`, `_human_beta_wide.csv` | `python research/scripts/same_clip_cue_analysis.py --au-csv research/reproduction/eu_au_measures.csv --workers 2` | norms, committed AU table | 2.7 | no / no | exact (all four) |
| `research/reproduction/same_clip_family_a2_criteria.json` | `python research/scripts/same_clip_family_analysis.py --write-attempt-2-criteria` | committed attempt-1 outputs | <0.1 | no / no | equal; `script_sha256_at_lock` and `locked_at_utc` differ by design |
| `research/reproduction/same_clip_family_criteria.json` | `python research/scripts/same_clip_family_analysis.py --write-criteria --criteria-out ...` | none | <0.1 | no / no | equal, `criteria_sha256` 4b0a8c3c; lock fields differ by design |

### Full mode only (CPU, long)

| Output | Command | Inputs | Time | Note |
|---|---|---|---|---|
| `same_clip_cue_recovery.json` | `same_clip_cue_analysis.py --recovery --au-csv research/reproduction/eu_au_measures.csv --workers N` | AU table, norms | long | 100 sims x 300 bootstraps |
| `same_clip_family.json`, `_matrix.csv`, `_verdicts.csv` | `same_clip_family_analysis.py --au-csv ... --graphau-csv research/reproduction/eu_au_measures_graphau.csv --workers N` | AU tables, norms | 2 h on 8 CPUs | `slurm_jobs/study3_family_recovery.sh STEP=human` |
| `same_clip_family_recovery.json`, `_families_singles.json` | `... --recovery --cue-set families` / `families_singles` | same | hours on 40 CPUs | STEP=recovery |
| `same_clip_family_a2.json`, `_a2_matrix.csv`, `_a2_verdicts.csv` | `... --attempt-2` | same | as above | disclosed second attempt |
| `same_clip_family_a2_recovery*.json` (three files) | `... --recovery --attempt-2 --cue-set ...`, model frames with `--frame-set model_1fps --only ...` | same | hours | see the job script header |
| `same_clip_family_route_pattern_{within_category,within_actor}.json` | `... --recovery --route-pattern --cue-set families --frame-set model_1fps --scheme S --seed-offset 200000 --recovery-out ...` | same | 1 h on 40 CPUs | STEP=route_pattern |
| `same_clip_family_route_pattern_075x_{within_category,within_actor}.json` | as above with `--seed-offset 300000 --pattern-factors 0.75 --fixed-scale 2.0` | same | 1 h on 40 CPUs | STEP=route_pattern_075 |
| `a4_recovery.json` | `python -m scripts.a4_recovery --manifest results/a4/manifest.csv --out research/reproduction/a4_recovery.json --workers N` | Snoek zip (via manifest), `human_au_effects.json` | hours | frozen, sha256 pinned in A4 |
| `a6_recovery.json` | `python -m scripts.a6_recovery --workers N` | `a6_human_target.json`, selection | 3 h on 3 workers (logged 10,832 s) | draft amendment |
| `a6_recovery_both_directions_paired.json`, `_draft_as_written.json`, `_happy_equivalence.json` | `python -m scripts.a6_recovery --variant {both_directions,first_draft,happy_equivalence} --output ...` | same | 17, 1 and 2 min | audit-trail variants |
| `a5_recovery.json` | `python -m scripts.a5_recovery --workers N` | design block in `PREREG_A5_DRAFT.md` | long | A5 is a draft being edited; a mismatch may only mean the draft moved on |

The `script_sha256` field in same-clip recovery outputs records the script at run time and is skipped; the code has changed since several of these ran (for example `cb5ad541` for attempt 2, `8b691d77` and `dff939e1` for the route runs), so full mode will show whether those edits changed results.

### Historical snapshots (code superseded; rerun by checking out the commit)

These are kept as the audit trail of earlier runs. They are not expected to match current code. To rerun one, `git worktree add <dir> <commit>` and run the same command there.

| Output | Made at commit | Note |
|---|---|---|
| `a4_recovery_first_run.json` | `17dc0cf` | design hash b84bd471 |
| `a4_recovery_second_run.json` | `6176b61` | design hash b0cbf90e |
| `a4_recovery_run3.json` | `568b6f5` (freeze) | design hash fb9d6ffb |
| `a5_recovery_run1.json` to `a5_recovery_run4*.json`, `a5_run3_*`, `a5_run4_skew_diagnostic.json`, `a5_family_diagnostic.json`, `a5_checks.json` | `565d665`, `b0f16d9`, `9644da1` | A5 drafts; owned by the A5 worker |
| `a6_recovery_run1.json`, `a6_recovery_run2.json` | `0d213c5`, `b056a7e` | earlier A6 drafts |

### Cluster only (GPU, data not in the public repo, or cluster caches)

| Output | Script and job | Inputs | GPU | Data not in the repo |
|---|---|---|---|---|
| `results/baseline/eu_emotions/*/eval_v2_*_video_only_seed42.json` | `scripts/evaluate.py` via `slurm_jobs/baseline_eval.sh`; jobs 33520827, 32447858, 32447860 | EU-Emotion videos, weights | yes | EU-Emotion videos |
| `vision_gates_{gemma4,molmo2,qwen3vl}.json` | `python -m scripts.vision_gates`, `slurm_jobs/study3_vision_gates.sh` | videos, weights | yes | yes |
| `input_delivery_*.json`, `cache_parity_qwen3vl_*.json` | `python -m scripts.run_stimulus_interventions --step delivery` / `cache_probe`, `slurm_jobs/study3_stimulus_interventions.sh` | videos, weights | yes | yes |
| `readout_ladder_*.json` | `python -m scripts.readout_ladder --step ladder --model M`, `slurm_jobs/study3_readout_ladder.sh` | CREMA-D calibration videos, weights | yes | CREMA-D videos |
| `crema_d_video_check.csv` | `python -I research/scripts/crema_verify_videos.py --video-dir DIR --ffprobe PATH --out ...` | CREMA-D videos | no | CREMA-D videos |
| `face_input_validity_summary.json` | `python -m scripts.face_input_validity --summarise` after per-model scoring, `slurm_jobs/study3_face_input_validity.sh`, job 40023523 | A3 cells (from videos) | yes | yes |
| `stimulus_features.json` | `python -m paper_v2.stimulus`, `slurm_jobs/study3_stimulus_features.sh`, job 39709870 | EU-Emotion videos | no | yes |
| `eu_au_measures.csv`, `eu_au_measures_graphau.csv`, `eu_au_extraction_graphau_summary.json` | `research/scripts/extract_au_measures.py` steps, `slurm_jobs/study3_au_extract*.sh` | videos, detector weights | optional | yes |
| `activation_meta/*_extract_meta.json` | `python -m scripts.extract_activations`, `slurm_jobs/study3_activation_extract.sh` | videos, weights | yes | yes |
| `phase2_gate.json`, `phase2_gate_all_layers_{model}.json` | `python -m paper_v2.probe --models M --act_dir ... --layer_set ... --out ...`, `slurm_jobs/study3_phase2_probe.sh`, jobs 39709580 to 39712192 | activation caches | no (CPU on caches) | derived from videos |
| `phase2_gate_all_layers.json`, `phase2_all_layers_summary.csv` | merged by hand from the three per-model files in commit `f6dd3cf`, with no script then. `python research/scripts/merge_phase2_all_layers.py` (added 2026-11-08) rebuilds both from the committed per-model JSONs; `--check` reported both byte-identical on 2026-11-08. It is now the quick step `merge_phase2_all_layers` | per-model gate JSONs | no | no |
| `phase5_entanglement.json` | `python research/scripts/run_phase5_entanglement.py` (`paper_v2.entanglement`, n_sim 100) | activation caches | no | derived from videos |
| `a4_v1_pilot.csv` | `research/scripts/a4_detector.py`, `slurm_jobs/study3_a4_detector.sh`, job 40260039 | A4 renders (FACSHuman) | no | FACSHuman |
| `results/mech/*` (probe RSA, axes, steering, ablation) | `scripts/confusability_probe_rsa.py`, `causal_eu_confusion_axes.py`, `steer_eu_confusion_axes.py`, `ablate_eu_confusion_axes.py` | activation caches, weights | yes | derived from videos |
| `results/stats/rq1_1b_entropy_alignment.json` | `scripts/forced_choice_entropy.py` (historical); its numbers are recomputed in `behaviour_summary.json` | eval JSONs | no | no |


## Clean-checkout test

Run on 2026-11-08 in a temporary `git worktree` of `origin/main` `af84cf2`, with only `scripts/reproduce.py`, `tests/test_reproduce.py` and this file copied in (not yet committed), conda env `mr_eu_open_llm`, Python 3.10. The worktree had no `data/external`; the script took the public tables from the main working tree and all six matched their pinned sha256. The worktree was removed afterwards.

```
python scripts/reproduce.py --quick --workers 1
```

**Result: exit code 0. 33 outputs rebuilt and matched (22 byte-identical, 11 within the stated tolerance), 87 verified by hash, 0 failed, 0 skipped, 0 draft changed, 9 pending items listed, 2 uncovered A7 files listed.** 545 s with 1 worker on a machine shared with other CPU jobs (load about 8 on 8 cores); expect about 7 minutes on 2 idle workers. `tests/test_reproduce.py`: 18 passed there.

Not byte-identical, and why:

| Output | Why |
|---|---|
| `phase0_numbers.json` | machine-state blocks and `has_*` flags skipped; every number equal |
| `human_calibration.json` | largest gap 1.7e-21 (float last digit) |
| `phase3_pilot_ablation.json` | `source` is an absolute path |
| `prompt_confound_audit.json` | `activations` needs cluster caches; permutation p-values differ by up to 0.0037 (tolerance 0.01); every r equal to 1e-15 |
| two Phase 2 PNGs | font rasterising; within the pixel tolerance |
| `human_au_effects.json` | AU02L/AU02R order in one unordered pair |
| two same-clip criteria files | lock records: script hash and lock time differ by design |
| `registry_derived_counts` (stdout) | no committed file; every value equal to the one pinned from the registry |
| `fig1.pdf` | matplotlib writes some patches in hash-seed order; the same 4,720 drawing-operator lines in another order. `fig1.png`, `fig3.png`, `fig3.pdf` and `SOURCES.md` are byte-identical |

Full mode was not run (hours of CPU; at most 2 workers were free). Its A2 single-column step was run on its own: see below.

**A2 single-column check (full-mode step `a2_gain_skew_single_column`), run once on its own on 2026-11-08:** `python research/scripts/a2_gain_skew_single_column.py`, one process, about 70 minutes on the loaded machine. All six output lines equal `single_column_check` in the committed `a2_gain_skew_sim.json` (compared with the script's own capture and compare rules: `within_tolerance`, equal after parsing). The 1000-replicate maxT part of that file stays hash-verified.

## Clean-checkout test, 2026-11-12

`python scripts/reproduce.py --quick --rev HEAD --external data/external --workers 2` at revision `6abfa91`, conda env `mr_eu_open_llm`, Python 3.10.20: 37 outputs rebuilt and matched (byte-identical or within the stated tolerance), 109 verified by pinned sha256, 7 pending, 0 uncovered, 1 failed. The failure was `research/paper/v2/figures/SOURCES.md`, one blank line off after a rebuild in a different order; `make_figures.py` now writes it the same way in any order (commit `9ee4603`). Figures 2 and 4 rebuilt byte-identically. A first run the same day used a stale local `origin/main` and is not a result.

## Clean-checkout test, 2026-11-13

`python scripts/reproduce.py --quick --external data/external --workers 2` at revision `6fffab1`, conda env `mr_eu_open_llm`: 40 outputs rebuilt and matched (byte-identical or within the stated tolerance), 112 verified by pinned sha256, 7 pending (the GPU results and Figures 5 and 6), 0 uncovered, 1 failed. The new outputs of 2026-11-12 are covered: `snoek_config_overlap.json` and `a9_label_corroboration.json` are rebuilt, and `a1_nota_shares.json`, `a1_frame_strategies.json` and `a3_token_floor_sensitivity.json` are verified by hash (they are made from locked cluster outputs). The failure was `a9_label_corroboration.json`, whose recorded hash of `human_au_effects.json` changed because that file is rebuilt earlier in the same run with an unordered pair in tie order; its values matched. The step now skips that hash record (both inputs are verified elsewhere), and a rerun of `human_au_effects`, `snoek_config_overlap` and `a9_label_corroboration` at the fixed revision matched with a largest value gap of 0. `tests/`: 1,064 passed, 26 skipped, 0 failed, after `tests/conftest.py` fixed a test-order fault (A7's recovery tests had replaced A2's sign-flip function in-process).
