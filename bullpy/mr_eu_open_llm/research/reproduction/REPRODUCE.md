# REPRODUCE

How every committed output in `research/reproduction/`, `research/figures/` and the numbers in `research/RESULTS_REGISTRY.md` were made, from which inputs, and which of them anyone can rerun from public inputs. Goal G12 in `research/GOALS.md`.

**Written:** 2026-11-08, at commit `34a24cc` on `p1-pipeline`. The paper's planned Figures 1 to 6 and Table 1 (`research/paper/v2/outline.md`) wait on A1, A2 and A4 model results and do not exist yet. Each will get a row here, made by a script, when it does.

## In short

1. The CPU-only steps whose inputs are public rerun with one command, `python scripts/reproduce.py --quick`, in about 6 minutes on 2 CPU workers. It checks out the committed tree into a temporary directory, adds the public CREMA-D and Snoek tables, reruns each step and compares the result with the committed file.
2. Model-facing steps (evaluations, activations, vision gates, readout ladder, A3) need a GPU, the model weights and the EU-Emotion videos. They ran on CSD3 and cannot be rerun from public inputs. Their committed outputs are the record, and each names its job, script and model revision.
3. Long CPU recoveries (A4, A5, A6, the same-clip recoveries) are in the full mode of the same script. They take from about 20 minutes to several hours on a 40-core node.

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

## Inputs and their status

| Input | Where | Status | Used by |
|---|---|---|---|
| EU-Emotion item-level norms (O'Reilly et al.), reduced to `data/eu_emotions_human_entropy.json` by `scripts/human_entropy.py` | in repo | public | behaviour, A1 to A3, same-clip, prompt audit |
| EU-Emotion manifest, definitions, synonyms (`data/eu_emotions_full_manifest.json` and others) | in repo | public (file names and labels only) | most EU-Emotion steps |
| EU-Emotion videos | CSD3 `data/eu_emotions` | not in the public repo (collaborator dataset, kept out for size); the team controls EU-Emotion (confirmed by the user 2026-11-08) | every model run, AU extraction, stimulus features |
| CREMA-D rating tables (`finishedResponses.csv`, `finishedEmoResponses.csv`, `VideoDemographics.csv`, `processedResults/tabulatedVotes.csv`, `processedResults/summaryTable.csv`) | GitHub `CheyneyComputerScience/CREMA-D` at `1658cd34`, ODbL / DbCL; gitignored locally in `data/external/crema_d` | public; sha256 pinned in `scripts/reproduce.py` and checked | CREMA-D selection, reliability, A6 human target |
| CREMA-D videos | CSD3 | licence-restricted (access form) | readout ladder, A6 model runs, video check |
| Snoek AU and emotion data, `au_emotion_data.zip` | DOI 10.21942/uva.21261885.v1, figshare file 37888164, CC BY 4.0; gitignored locally in `data/external/snoek_au` | public; sha256 pinned and checked | human AU effects, A4 manifest and ceiling, predictor, CFA |
| FACSHuman targets and MakeHuman meshes | CSD3 | licence-restricted (LGPL targets, no image is published) | A4 renders, V1 pilot |
| RMET items | not used by any output listed here | licence-restricted | Phase 6 only |
| Model weights (Gemma 4 E4B, Qwen3-VL 8B, Molmo2 O-7B) | Hugging Face, pinned commits | public weights, need a GPU | all model runs |
| AU detector weights (LibreFace, OpenFace 3, ME-GraphAU) | CSD3 venv `au_tools` | research or non-commercial licences | AU extraction |
| Activation caches (`results/activations/`) and rotation, A3 and A4 cells | CSD3 | cluster-only derived data (from videos not in the repo) | Phase 2, Phase 5, A3 summary, prompt audit activations |
| Derived AU tables `research/reproduction/eu_au_measures*.csv` | in repo | public derived numbers (per-clip AU aggregates); remaking them needs the videos | CFA, same-clip analyses |

## Output table

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
| `phase2_gate_all_layers.json`, `phase2_all_layers_summary.csv` | merged by hand from the three per-model files in commit `f6dd3cf`, with no script then. `python research/scripts/merge_phase2_all_layers.py` (added 2026-11-08) rebuilds both from the committed per-model JSONs; `--check` reported both byte-identical on 2026-11-08. It is CPU-only on committed inputs but not yet a step in `scripts/reproduce.py` | per-model gate JSONs | no | no |
| `phase5_entanglement.json` | `python research/scripts/run_phase5_entanglement.py` (`paper_v2.entanglement`, n_sim 100) | activation caches | no | derived from videos |
| `a4_v1_pilot.csv` | `research/scripts/a4_detector.py`, `slurm_jobs/study3_a4_detector.sh`, job 40260039 | A4 renders (FACSHuman) | no | FACSHuman |
| `results/mech/*` (probe RSA, axes, steering, ablation) | `scripts/confusability_probe_rsa.py`, `causal_eu_confusion_axes.py`, `steer_eu_confusion_axes.py`, `ablate_eu_confusion_axes.py` | activation caches, weights | yes | derived from videos |
| `results/stats/rq1_1b_entropy_alignment.json` | `scripts/forced_choice_entropy.py` (historical); its numbers are recomputed in `behaviour_summary.json` | eval JSONs | no | no |

## Registry coverage

Every row of `RESULTS_REGISTRY.md` traces to an output above. The gaps this audit found were closed on 2026-11-08 in the registry section "Registry gaps closed (2026-11-08)": the A4 human target and ceiling, A4 recovery and the V1 pilot, the CFA cue spectra, A2 recovery, vision gates, input delivery, the readout choice, the human response predictor, CREMA-D reliability and selection, and the A6 human target. The consensus-tercile rows are now marked verified. Two new read-only scripts back numbers that no output file stored: `research/scripts/merge_phase2_all_layers.py` (the hand merge of `f6dd3cf`, rebuilt byte-identical) and `research/scripts/registry_derived_counts.py` (the V1 pilot breakdown and the 68 of 243 grid frame loss).

Still untraced:

- A3's descriptive "None of the above" shares (Gemma .206/.305, Qwen .534/.688, Molmo .863/.886). No script or output file holds them; they were computed from the cluster cells outside the frozen summary. They are not cited in `research/paper/v2/`, but `PREREG_A4_DRAFT.md` repeats .534 and .863. Molmo's accuracy by target position is traced: it is in `face_input_validity_summary.json`, `secondary.accuracy_by_target_position`.

## Clean-checkout test

Run on 2026-11-08 in a temporary `git worktree` of HEAD `34a24cc` (with `scripts/reproduce.py` copied in, as it was not yet committed), conda env `mr_eu_open_llm`, 2 CPU workers, public inputs copied from `data/external` and checked against their sha256:

```
python scripts/reproduce.py --quick --workers 2
```

**Result: 25 of 25 outputs reproduced, 0 failed, 0 skipped, exit code 0, about 6.5 minutes.** 17 byte-identical (one of them `results/a4/manifest.csv`, checked against its pinned sha256), 8 within the stated tolerance:

| Output | Why not byte-identical |
|---|---|
| `phase0_numbers.json` | machine-state blocks and `has_*` flags skipped; every number equal |
| `human_calibration.json` | largest gap 1.7e-21 (float last digit) |
| `phase3_pilot_ablation.json` | `source` is an absolute path |
| `prompt_confound_audit.json` | `activations` needs cluster caches (empty in a clean checkout); permutation p-values differ by up to 0.0037 (tolerance 0.01); every r equal to 1e-15 |
| two Phase 2 PNGs | font rasterising; same plot (checked by eye too), within the pixel tolerance |
| `human_au_effects.json` | AU02L/AU02R order in one unordered pair |
| two same-clip criteria files | lock records: script hash and lock time differ by design; criteria and `criteria_sha256` equal |

Full mode was not run in this test (hours of CPU; at most 2 workers were available because other simulations were running).
