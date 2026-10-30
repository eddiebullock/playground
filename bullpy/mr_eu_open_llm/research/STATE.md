# STATE

Single source of truth for the paper-v2 agent. Update after every meaningful step.

**Updated:** 2026-10-30 14:05 Europe/London

## Current phase

Phase 1 wrapping; preregistration freeze; GPU jobs submitting on CSD3. Phase 2 gate starts on CPU after `prereg-v1` is tagged, using pulled 3-layer last-token activations.

## Task in progress

Submit Qwen full-N ablation + all-layer extracts via ControlMaster SSH. Write/tag prereg. Then run Phase 2 probes locally.

## Last completed step

- Phase 0 tagged `v2-phase0` on `origin/main` and `origin/paper-v2` (`d47c0be`).
- Started Phase 1: Molmo wording in `docs/study3_brief_for_claude_agent.md`; `src/paper_v2` metrics + item schema + EU-Emotion adapter (243 items, 17 actors); behaviour CLI reproduces binomial + entropy ρ. Tests 14/14 pass. Original `scripts/` untouched.

## Next 5 actions

1. EU-Emotion dataset adapter → common Item schema from human_entropy JSON + manifest.
2. Behaviour entry point: binomial vs chance for every model, consensus terciles, entropy ρ with bootstrap CIs.
3. Option-position-bias plan + logprob scoring interface (run when GPU exists).
4. Write `research/PREREGISTRATION.md` and tag `prereg-v1` before any Phase 2 probe rerun.
5. Unit tests for RDM construction from the adapter; keep original scripts archived in place.

## Running jobs

CSD3 jobs listed after `sbatch` (fill job IDs in DIGEST). Local: none.

| name | pid / tmux / slurm | log | expected finish |
|---|---|---|---|
| (pending submit) | CSD3 ampere | `~/rds/hpc-work/study3/logs/` | see squeue |

## Branch / git

- Default branch: `main` (local checkout still stale/dirty; `origin/main` is at `v2-phase0`).
- Integration branch: `paper-v2` (created from `origin/main` at `90e7a86`).
- Task branch: `p1-pipeline` (from `paper-v2`). Merged Phase 0: `p0-locate-reproduce`.
- Repo visibility: **public**.

## Blockers (non-idle)

- CSD3 SSH resolved via ControlMaster. Queue is busy with A1/A2/CAS jobs (Priority); study3 jobs may wait.
- CREMA-D access form and MRMET option-count verification wait for Phase 6.
