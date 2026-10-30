# STATE

Single source of truth for the paper-v2 agent. Update after every meaningful step.

**Updated:** 2026-10-30 14:47 Europe/London

## Current phase

Phase 2 protocol has been applied to the pilot 3-layer last-token cache. Outcome is **exploratory would-be Branch A**; the branch is **not locked**. GPU all-layer extracts and full-N ablation are still queued on CSD3. Phase 3 waits for those.

## Task in progress

Monitor CSD3 GPU jobs 39709579–39709584. Re-run the same gate with `layer_set=all_layers` when extracts land.

## Last completed step

- Phase 2 confirmatory protocol on 3-layer last-token activations (`research/reproduction/phase2_gate.json`). Qwen L26 and Gemma L15 pass the four-check rule; Molmo also matches the statistics but is not above chance so it cannot support H1. Branch not locked.
- Phase 5: ρ(1−p_target, H_human) = 0.954; probe cosines sit below the Gaussian null.
- H2: model-output entropy vs human entropy bootstrap CIs all include 0.
- Stimulus features 243/243 (job 39709870).
- Six GPU jobs still PD (Priority). Option-order GPU jobs not submitted.

## Next 5 actions

1. When all-layer extracts complete, re-run `paper_v2.probe --layer_set all_layers` and lock Branch A/B.
2. Full-N ablation (`ablate_full_n243`) after 39709579; Gemma L5 after 39709584.
3. Phase 3 patching heatmap + label-subspace positive control (GPU, Qwen first).
4. Option-order robustness evals once ampere has room.
5. Draft methods from the frozen prereg; do not rewrite hypotheses.

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

Local CPU jobs finished. Stimulus CPU job 39709870 completed (243/243).

## Branch / git

- Default branch: `main` (local checkout stale/dirty; do not use).
- Integration branch: `paper-v2`.
- Task branch: `p1-pipeline` (includes this session).
- Tags: `v2-phase0`, `prereg-v1`.
- Repo visibility: **public**.

## Blockers (non-idle)

- Ampere queue is busy (Priority). Study3 GPU jobs wait; other user jobs are left alone.
- Official Branch A/B lock waits for all-layer activations.
- CREMA-D access form and MRMET option-count verification wait for Phase 6.
- Extra GPU option-order jobs are written but not submitted, so they do not delay extracts.
