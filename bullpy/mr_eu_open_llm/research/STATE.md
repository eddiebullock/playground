# STATE

Single source of truth for the paper-v2 agent. Update after every meaningful step.

**Updated:** 2026-11-02 11:20 Europe/London

## Current phase

**Branch B locked** on the official all-layer last-token Phase 2 gate. Phase 4 (readout/steering) is off. Phase 3 causal map still runs (both branches). Full-N ablation and Qwen max_frames=16 extract remain queued on ampere.

## Task in progress

Keep the pending GPU jobs; Branch B methods/results draft is on disk; do not retune Holm / n_perm.

## Last completed step

- All-layer extracts completed 2026-11-02 morning: Qwen 36/36, Gemma 42/42, Molmo 32/32 last-token layers, n = 243, 0 errors (jobs 39709580–82).
- Official Phase 2 CPU probes completed (39712190–92). **Branch B.** Peak grouped T1 ρ 0.47 / 0.49 / 0.52; Holm floors 0.036 / 0.042 / 0.032, all > 0.01.
- Combined dump: `research/reproduction/phase2_gate_all_layers.json`. Results draft `research/paper/results_v1.md`; figure `research/figures/phase2_grouped_t1_rho.png`.

## Next 5 actions

1. Full-N Qwen ablation 39709579 (est. start 2026-11-03 00:25 Europe/London).
2. Gemma L5 full-N ablation 39709584 (est. 00:25).
3. Qwen all-layer max_frames=16 extract 39709583 (est. 00:25).
4. Phase 3 patching heatmap + label-subspace positive control once ablation/extracts land.
5. Branch B paper draft (no Phase 4). Option-order GPU still waits.

## Running jobs

SSH: `ssh -o ControlPath=/Users/eb2007/.ssh/cm-%r@%h:%p -o ControlMaster=no -o BatchMode=yes eb2007@login.hpc.cam.ac.uk`

Do not cancel A1/A2/CAS jobs.

| name | pid / slurm | log | expected finish |
|---|---|---|---|
| s3_ablate_full Qwen n=243 | 39709579 PD (Priority) ampere | `~/rds/hpc-work/study3/logs/` | est. 2026-11-03 00:25; 12h wall |
| s3_act_qwen_all16 | 39709583 PD (Priority) | same | est. 2026-11-03 00:25; 12h wall |
| s3_ablate_gemma L5 n=243 | 39709584 PD (Priority) | same | est. 2026-11-03 00:25; 12h wall |

Finished this morning: extracts 39709580–82; Phase 2 probes 39712190–92. Stimulus 39709870 already done.

## Branch / git

- Default branch: `main`.
- Integration branch: `paper-v2`.
- Task branch: `p1-pipeline`.
- Tags: `v2-phase0`, `prereg-v1`.
- Repo visibility: **public**.

## Blockers (non-idle)

- Official H1 gate failed Holm (perm-resolution floor). Branch B locked; do not retune.
- Ampere still Priority-bound (CAS `cas_r2` 39827858 estimated 2026-11-02 11:51). Study3 GPU jobs share a 2026-11-03 00:25 start estimate.
- CREMA-D access form and MRMET option-count verification wait for Phase 6.
- Phase 4 cancelled under Branch B.
