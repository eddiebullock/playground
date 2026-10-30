# STATE

Single source of truth for the paper-v2 agent. Update after every meaningful step.

**Updated:** 2026-10-30 14:35 Europe/London

## Current phase

Phase 0 complete (`v2-phase0` on `origin/main` and `origin/paper-v2`). Starting Phase 1.

## Task in progress

Phase 1: Molmo chance-level wording, binomial helper, pipeline skeleton, unit tests. Preregistration draft (commit before any Phase 2 run).

## Last completed step

- Phase 0 tagged `v2-phase0` on `origin/main` and `origin/paper-v2` (`d47c0be`).
- Started Phase 1: Molmo wording in `docs/study3_brief_for_claude_agent.md`; `src/paper_v2` metrics + item schema; known-answer tests (11/11 pass). Original `scripts/` untouched.

## Next 5 actions

1. EU-Emotion dataset adapter → common Item schema from human_entropy JSON + manifest.
2. Behaviour entry point: binomial vs chance for every model, consensus terciles, entropy ρ with bootstrap CIs.
3. Option-position-bias plan + logprob scoring interface (run when GPU exists).
4. Write `research/PREREGISTRATION.md` and tag `prereg-v1` before any Phase 2 probe rerun.
5. Unit tests for RDM construction from the adapter; keep original scripts archived in place.

## Running jobs

None.

| name | pid / tmux | log | expected finish |
|---|---|---|---|
| — | — | — | — |

## Branch / git

- Default branch: `main` (local checkout still stale/dirty; `origin/main` is at `v2-phase0`).
- Integration branch: `paper-v2` (created from `origin/main` at `90e7a86`).
- Task branch: `p1-pipeline` (from `paper-v2`). Merged Phase 0: `p0-locate-reproduce`.
- Repo visibility: **public**.

## Blockers (non-idle)

- CSD3 SSH: `Permission denied (publickey)` — see `NEEDS_INPUT.md`. 6AFC activation tensors are not on this Mac.
- CREMA-D access form and any MRMET option-count verification wait for Phase 6; adapters can still be written.
