# STATE

Single source of truth for the paper-v2 agent. Update after every meaningful step.

**Updated:** 2026-10-30 14:20 Europe/London

## Current phase

Phase 0 git wrap-up. Next after push/tag: Phase 1 (correctness + pipeline refactor). CPU-feasible work is unblocked. GPU work is blocked on CSD3 SSH.

## Task in progress

Merging `p0-locate-reproduce` into `paper-v2`, pushing, tagging `v2-phase0`.

## Last completed step

- Located project at `/Users/eb2007/playground/bullpy/mr_eu_open_llm` (git root `/Users/eb2007/playground`, remote `eddiebullock/playground`).
- Reproduced headline numbers from cached eval/stats/mech JSON and axis `.npy` files via `research/scripts/reproduce_headline_numbers.py`.
- Wrote research scaffold, compute map, reproduction report, decisions, needs-input, results registry, phase-0 review, digest.
- Commit `2e2af355a4c4c776d2b9675aa4a5b55c56fc44c8` on `p0-locate-reproduce`.

## Running jobs

None.

| name | pid / tmux | log | expected finish |
|---|---|---|---|
| — | — | — | — |

## Next 5 actions

1. Commit Phase 0 on `p0-locate-reproduce`, push, merge into `paper-v2`, tag `v2-phase0`.
2. Phase 1: fix Molmo above-chance wording in docs; add binomial tests as a reusable function; do not treat Molmo mechanistic results as recognition.
3. Phase 1: archive original scripts; start config-driven `src/` pipeline (dataset + model adapters, option-logprob scoring). Unit tests for entropy / RDM / Spearman / LOO.
4. Write `research/PREREGISTRATION.md` and commit/tag `prereg-v1` **before** any Phase 2 probe rerun.
5. If CSD3 SSH is restored: pull 6AFC activation `.npy` (3 models x 3 layers, last-token) then run Phase 2 gate on CPU. Until then, draft adapters and paper methods from cached JSON.

## Branch / git

- Default branch: `main` (local is 11 commits behind `origin/main`; those commits are Study A1/A2, not this project).
- Integration branch: `paper-v2` (created from `origin/main` at `90e7a86`).
- Task branch: `p0-locate-reproduce` @ `2e2af35`.
- Repo visibility: **public**.

## Blockers (non-idle)

- CSD3 SSH: `Permission denied (publickey)` — see `NEEDS_INPUT.md`. 6AFC activation tensors are not on this Mac.
- CREMA-D access form and any MRMET option-count verification wait for Phase 6; adapters can still be written.
