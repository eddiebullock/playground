# STATE

Single source of truth for the paper-v2 agent. Update after every meaningful step.

**Updated:** 2026-11-04 Europe/London

## Current phase

**Rebuilding on a clean task.** Prompt confound found 2026-11-04 (DECISIONS). Branch B stays as a record, but accuracy and last-token probes are confounded. Amendment A1 is the active confirmatory step. Literature-led redesign under way in `research_notes/`.

## Next actions

1. A1 rotation evals are queued: arrays 39982133 to 39982138, restarted on the pinned decoder. Nothing read yet.
2. `research/scripts/analyse_rotation.py` once all 30 runs are pulled.
3. Molmo ladder 39987784, then choose the shared (prompt, prefill) pair, update `BLIND_PROMPT` and `BLIND_PREFILL`, run timed smokes in `SMOKE_DIR=smoke_v2`, and fill A2 and A3 costs.
4. Same-clip analysis: confirmatory unit is families and the pattern score. Single family cells are provisional (family recovery, RESULTS_REGISTRY). A restricted-shuffle second attempt must be locked before any model matrix is read.
5. Vision gates G2 to G4: 39981392 still queued.
6. Scoring path: full forwards for all models unless DECISIONS says otherwise.

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
