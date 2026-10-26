# Handover, 2026-10-26 19:30 (E9e pass; two-gate wired)

For the next agent picking up Gloss. Read `AGENT_PROMPT.md`, `PLAN.md`,
`docs/PUBLICATION_STRATEGY.md` and the `docs/DECISIONS.md` entries dated
2026-10-25 and 2026-10-26 first. Overwrite this file when the work below
is done.

## Rules in force
- PLAN.md section 7. No money, no paid APIs, no cluster, no licensed data,
  nothing off the machine, no contacting people. Participant data stays
  local. Delete only files you created. Keep >= 5 GB free (disk ~19 GB).
  Never kill other users' processes.
- No emojis. Repo `~/playground`, branch `es-vlm/setup`, remote
  `git@github-personal:eddiebullock/playground.git`. Commit only under
  `social_ai/gloss`. Push after each finished result.
- Commit trailer: `Co-Authored-By: Cursor Grok 4.6 <noreply@cursor.com>`.
- **E9c, E9d and E9e are all seen. Do not retune on them.** Do not rescore
  them. Two-gate thresholds stay `{ distress: 0.3, scam: 0.5, boundary: 0.3 }`.

## Environment
- Ollama 127.0.0.1:11434; qwen3:8b product default. One model at a time.
  `safety_model_eval.ts` retries Ollama 500s; still resumable.
- `cd app && node --experimental-strip-types scripts/<x>.ts`.
  Tests: `npm test`. Typecheck: `npx tsc --noEmit -p .`.
- Scores: `experiments/e9_safety/model_scores/`. Two-gate files end
  `_twogate.jsonl`. Report: `safety_report.ts [--e9c] [--e9d] [--e9e] [--twogate]`.
- Product: `safetyFull` / `reflect()` use two-gate + THRESHOLD_TWOGATE.
  Questions cached per input. Generation and safety scoring overlap.

## Jobs
E9 product re-run with `--replies` may be running:
`experiments/e9_safety/results_qwen3-8b_twogate.jsonl` (resumable via
`run_set.ts`). Log: `experiments/e9_safety/run_replies.log`. Do not change
engine code while that run is in progress. If it died, restart the same
command (it skips ids already in the jsonl).

## State
- E9e **PASS** at THRESHOLD_TWOGATE. Distress 28/30, scam 30/30, boundary
  30/30; hard-neg FP 1/25, 1/25, 0/25. Official:
  `experiments/e9_safety/model_scores/e9e_official.txt`.
- Two-gate is wired. Original E9 (40 cases) is being re-run as a product
  check, not a held-out test (that set informed the patterns).
- Participants still blocked until Eddie reviews the notes in `safety.ts`
  (UK Samaritans 116 123 / 999).

## What to do next, in order
1. **Finish the E9 re-run** if the jsonl has fewer than 40 lines. Then
   `node --experimental-strip-types scripts/grade_e9.ts
   ../experiments/e9_safety/results_qwen3-8b_twogate.jsonl --rubric`.
   Record mechanical flag rates and rubric pass/40 in DECISIONS. Do not
   retune two-gate from this run.
2. Until Eddie reviews `safety.ts` notes, Gloss must not be shown to
   participants (Study 5, demos).
3. **Scorer disclosure fix** (Paper 2 / PLAN section 8): the shipped
   "judge the words, not the reader" line left anxiety-disclosure drift
   on readings. Pre-register the next scorer change on the odd half, test
   on the even half. Do not mix this with a running E9 generate.
4. **E10 "me" mode calibration:** clear-draft false-alarm target <= 0.2,
   bands fitted separately from "them" mode. Pre-register before tuning.
5. **Latency:** cache the fixed system prompt; write the check question
   only when asked.
6. Cursor cannot update
   https://claude.ai/artifact/GapiXAWA8EJVYz6DAbuCCP.

## For Eddie
- E9e passed. Two-gate is in the product. Remaining holes are known and
  frozen: he:9 (railway-line plan not named in the distress question),
  he:95 (researching methods; boundary still fired), plus two pattern
  false positives on jokes/dispatch mail.
- Please review the Samaritans/999 notes in `app/src/engine/safety.ts`
  before any demo or Study 5. Advisory group on wording still needed.
- Please update the brief artifact. Snapshot: `docs/brief_snapshot.html`.
- Human validation still needed for the papers.
- `~/gloss_premerge_backup` (260 MB) can go if you are happy with the merge.
- Accidental Paper 2 artifact:
  https://claude.ai/artifact/1eyHVfwMG5B9SPmo9G3AVe.
