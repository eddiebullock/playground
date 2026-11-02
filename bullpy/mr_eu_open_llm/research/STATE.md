# STATE

Single source of truth for the paper-v2 agent. Update after every meaningful step.

**Updated:** 2026-11-02 13:30 Europe/London

## Current phase

**Branch B locked** on the official all-layer last-token Phase 2 gate. Phase 4 (readout/steering) is off. Phase 3 causal map still runs (both branches). Full-N ablation and Qwen max_frames=16 extract remain queued on ampere.

## Task in progress

Keep pending GPU jobs (do not cancel A1/A2/CAS). All-layer RSA on icelake. Intro/discussion drafts on disk. Do not retune Holm / n_perm.

## Last completed step

- All-layer extracts completed 2026-11-02 morning: Qwen 36/36, Gemma 42/42, Molmo 32/32 last-token layers, n = 243, 0 errors (jobs 39709580–82).
- Official Phase 2 CPU probes completed (39712190–92). **Branch B.** Peak grouped T1 ρ 0.47 / 0.49 / 0.52; Holm floors 0.036 / 0.042 / 0.032, all > 0.01.
- Combined dump: `research/reproduction/phase2_gate_all_layers.json`. Results draft `research/paper/results_v1.md`; figure `research/figures/phase2_grouped_t1_rho.png`.
- Intro/discussion drafts; LOO-vs-grouped figure; `paper_v2.causal` recomputes the n=36 5/180 flips. All-layer RSA submitted icelake 39834753–55.

## Next 5 actions

1. Full-N Qwen ablation 39709579 (est. start 2026-11-03 05:15 Europe/London).
2. Qwen all-layer max_frames=16 extract 39709583 (est. 05:40).
3. Gemma L5 full-N ablation 39709584 (est. 05:45).
4. Pull all-layer RSA when icelake jobs finish; summarise full-N ablation with `paper_v2.causal`.
5. Option-order GPU still waits. Phase 3 heatmap after ablation.

## Running jobs

SSH: `ssh -o ControlPath=/Users/eb2007/.ssh/cm-%r@%h:%p -o ControlMaster=no -o BatchMode=yes eb2007@login.hpc.cam.ac.uk`

Do not cancel A1/A2/CAS jobs.

| name | pid / slurm | log | expected finish |
|---|---|---|---|
| s3_ablate_full Qwen n=243 | 39709579 PD (Priority) ampere | `~/rds/hpc-work/study3/logs/` | est. 2026-11-03 05:15; 12h wall |
| s3_act_qwen_all16 | 39709583 PD (Priority) | same | est. 2026-11-03 05:40; 12h wall |
| s3_ablate_gemma L5 n=243 | 39709584 PD (Priority) | same | est. 2026-11-03 05:45; 12h wall |
| A2 running (do not cancel) | 39824252, 39824251 R ampere | — | started 13:12–13:20 |
| s3_rsa_qwen_all | 39834753 PD icelake | same | 8h wall; all-layer RSA |
| s3_rsa_gemma_all | 39834754 PD icelake | same | 8h wall |
| s3_rsa_molmo_all | 39834755 PD icelake | same | 8h wall |

Finished this morning: extracts 39709580–82; Phase 2 probes 39712190–92. Stimulus 39709870 already done.

## Branch / git

- Default branch: `main`.
- Integration branch: `paper-v2`.
- Task branch: `p1-pipeline`.
- Tags: `v2-phase0`, `prereg-v1`.
- Repo visibility: **public**.

## Blockers (non-idle)

- Official H1 gate failed Holm (perm-resolution floor). Branch B locked; do not retune.
- Ampere Priority-bound behind A2 (running) then CAS. Study3 GPU start estimates 2026-11-03 05:15–05:45.
- CREMA-D access form and MRMET option-count verification wait for Phase 6.
- Phase 4 cancelled under Branch B.
