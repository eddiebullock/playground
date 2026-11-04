# STATE

Single source of truth for the paper-v2 agent. Update after every meaningful step.

**Updated:** 2026-11-04 Europe/London

## Current phase

**Rebuilding on a clean task.** Prompt confound found 2026-11-04 (DECISIONS). Branch B stays as a record, but accuracy and last-token probes are confounded. Amendment A1 is the active confirmatory step. Literature-led redesign under way in `research_notes/`.

## Next actions

1. A1 rotation evals: arrays 39982133 to 39982138, 9 of 30 files on disk at 2026-11-04 21:36. Nothing read yet. Then `research/scripts/analyse_rotation.py` once all 30 are pulled, then `model_route` per model.
2. A3 frozen and running: Gemma 40000202 and Qwen 40000203 complete, Molmo 40000204 queued. Summarise once, after all three: `A3_FROZEN=1 SUMMARISE=1 sbatch --time=00:10:00 slurm_jobs/study3_face_input_validity.sh`. Full reading needs A1.
3. Shared readout chosen; smoke_v2 and A2 and A3 costs are done. A2 stays unfrozen until RQ0 and its two design points are reviewed.
4. Same-clip analysis: RQ1 pattern score on model frames is confirmatory, subject to the 5 percent clip rule (DECISIONS). The claim is "aligned with the human pattern". Cells are exploratory.
5. Vision gates: Gemma passes G2 to G4. Molmo fails G2 at bfloat16 and G4, so it is not eligible for A2 as drafted. Qwen full 39981392 still queued; pull `research/reproduction/vision_gates_qwen3vl.json` when it ends.
6. Scoring path: full forwards for all models unless DECISIONS says otherwise. Never patch Gemma image positions at L24 or later.

Merge with true merge commits, not squash: a squash on PR 50 caused a conflict on the next PR. Cluster quota: 1097.8 of 1099.5 GB soft limit at last check.

Do not cancel A1/A2/CAS jobs. Ablation 39709579, 39709583, 39709584 were cancelled 2026-11-02 and are not requeued.

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
