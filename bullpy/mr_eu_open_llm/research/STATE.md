# STATE

Single source of truth for the paper-v2 agent. Update after every meaningful step.

**Updated:** 2026-10-30 14:10 Europe/London

## Current phase

Phase 1 complete enough to run confirmatory analyses. Preregistration frozen (`prereg-v1`). Phase 2 protocol is running on the cached 3-layer last-token activations (exploratory; does not lock Branch A/B). GPU jobs for all-layer extract and full-N ablation are queued on CSD3.

## Task in progress

Local CPU: Phase 2 nested-CV probes + Phase 5 entanglement null. HPC: monitor GPU jobs; CPU stimulus-feature extract.

## Last completed step

- Six CSD3 GPU jobs submitted via ControlMaster (all PD / Priority).
- Phase 2 library: nested Ridge, actor-grouped CV, Holm, bootstrap, four baselines, option-order interface, entanglement null.
- Tests 24/24 (conda `mr_eu_open_llm`, no pytest binary; functions invoked directly).
- n=36 ablation and difficulty-matched RSA verified from pulled files.

## Next 5 actions

1. Finish local Phase 2 run; log would-be branch as exploratory in `DECISIONS.md`.
2. Pull stimulus features when the CPU job finishes; re-score the stimulus baseline if luminance/motion add anything.
3. When all-layer extracts complete, re-run the same gate with `layer_set=all_layers` and lock Branch A/B.
4. Full-N ablation (`ablate_full_n243`) after job 39709579.
5. Option-order GPU evals only after the current ampere queue drains.

## Running jobs

SSH: `ssh -o ControlPath=/Users/eb2007/.ssh/cm-%r@%h:%p -o ControlMaster=no -o BatchMode=yes eb2007@login.hpc.cam.ac.uk`

Do not cancel A1/A2/CAS jobs.

| name | pid / slurm | log | expected finish |
|---|---|---|---|
| s3_ablate_full Qwen n=243 | 39709579 PD (Priority) ampere | `~/rds/hpc-work/study3/logs/` | after A1/A2/CAS; 12h wall |
| s3_act_qwen_all f4 | 39709580 PD (Priority) | same | 8h wall |
| s3_act_gemma_all f4 | 39709581 PD (Priority) | same | 8h wall |
| s3_act_molmo_all f4 | 39709582 PD (Priority) | same | 8h wall |
| s3_act_qwen_all16 f16 | 39709583 PD (Priority) | same | 12h wall |
| s3_ablate_gemma L5 n=243 | 39709584 PD (Priority) | same | 12h wall |
| cas_r2 (unrelated) | 39707124 R gpu-q-8 | — | leave running |
| s3_stim_feat (CPU) | 39709870 R cpu-q-162 | `logs/study3_stim_*` | ~minutes |
| local Phase 2 probes | conda `mr_eu_open_llm` | `research/logs/phase2_gate.log` | after Phase 5 |

## Branch / git

- Default branch: `main` (local checkout stale/dirty; do not use).
- Integration branch: `paper-v2`.
- Task branch: `p1-pipeline` (HEAD `a438754` plus this session).
- Tags: `v2-phase0`, `prereg-v1`.
- Repo visibility: **public**.

## Blockers (non-idle)

- Ampere queue is busy (Priority). Study3 GPU jobs wait; other user jobs are left alone.
- Official Branch A/B lock waits for all-layer activations.
- CREMA-D access form and MRMET option-count verification wait for Phase 6.
- Extra GPU option-order jobs are written but not submitted, so they do not delay extracts.
