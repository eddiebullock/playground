# STATE

Single source of truth for the paper-v2 agent. Update after every meaningful step.

**Updated:** 2026-11-06 Europe/London

## Current phase

**Rebuilding on a clean task.** Prompt confound found 2026-11-04 (DECISIONS). Branch B stays as a record, but accuracy and last-token probes are confounded. Amendment A1 is the active confirmatory step. Literature-led redesign under way in `research_notes/`.

## Next actions

1. A1 rotation evals: Gemma and Molmo complete, 20 of 30 files on the cluster at 2026-11-06 08:59, not pulled, nothing read. Qwen arrays 39982137 and 39982138 still pending on priority (Slurm estimate 7 Nov about 15:40 to 17:35, it moves hourly), about 8 h per task, up to 78 GPU-hours, 2,859 available. When all 30 exist, pull them, then run `research/scripts/analyse_rotation.py`, then `model_route` per model. Do not analyse early: Holm covers all three models.
2. A3 done 2026-11-05: all three models pass on `single_face`, Molmo at accuracy .072 (RESULTS_REGISTRY). Pass tests vision dependence, not competence. Full reading needs A1.
3. Shared readout chosen; smoke_v2 and A2 and A3 costs are done. A2 stays unfrozen until RQ0 and its two design points are reviewed.
4. Same-clip analysis: RQ1 pattern score on model frames is confirmatory, subject to the 5 percent clip rule (DECISIONS). The claim is "aligned with the human pattern". Cells are exploratory.
5. Vision gates: Gemma and Qwen pass G2 to G4, Qwen at bfloat16 only because float32 ran out of memory (DECISIONS 2026-11-05). Molmo fails G2 at bfloat16 and G4, so it is not eligible for A2 as drafted.
6. Scoring path: full forwards for all models unless DECISIONS says otherwise. Never patch Gemma image positions at L24 or later.

Merge with true merge commits, not squash: a squash on PR 50 caused a conflict on the next PR. Cluster quota: 1100.1 GB at 2026-11-06 08:59, over the 1099.5 GB soft limit since 2026-11-05 13:02, hard limit 1209.5. Grace ends about 2026-11-12 13:00; after that writes fail until usage is below the soft limit, so free space before then. Storage audit done, deletion awaits a decision.

Do not cancel A1/A2/CAS jobs. `cas_*` jobs belong to the separate CASCADE project (`~/rds/hpc-work/cascade/code`), and appear to be resubmitted by a local loop there (`build/chain_loop.sh`), inferred from timing. Ablation 39709579, 39709583, 39709584 were cancelled 2026-11-02 and are not requeued.

SSH: `ssh -o ControlPath=/Users/eb2007/.ssh/cm-%r@%h:%p -o ControlMaster=no -o BatchMode=yes eb2007@login.hpc.cam.ac.uk`

## Decoder

All video jobs must pin cv2 4.11 and pass `research/scripts/decoder_preflight.py`. See amendment A1.1.

## Access confirmed

EU-Emotion: full access and licence, per user 2026-11-04. Full RMET access. CREMA-D form done. Nothing from these goes in the public repo.

## Branch / git

- Default branch: `main`.
- Integration branch: `paper-v2`.
- Task branch: `p1-pipeline`.
- Tags: `v2-phase0`, `prereg-v1`.
- Repo visibility: **public**.

## Blockers (non-idle)

- Official H1 gate failed Holm (perm-resolution floor). Branch B locked; do not retune.
- CREMA-D access form and MRMET option-count verification wait for Phase 6.
- Phase 4 cancelled under Branch B.
