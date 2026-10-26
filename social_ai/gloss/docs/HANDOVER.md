# Handover, 2026-10-26 09:55 (Cursor session after E9c)

For the next agent picking up Gloss. Read `AGENT_PROMPT.md`, `PLAN.md`,
`docs/PUBLICATION_STRATEGY.md` and the `docs/DECISIONS.md` entries dated
2026-10-25 and 2026-10-26 first. This file is the state at handover; overwrite
it when the work below is done.

## Rules in force
- PLAN.md section 7 autonomy rules. Ask Eddie before: spending money, paid
  APIs, cluster jobs, licensed data (Mindreading, EU-Emotions, RMET), sending
  anything off the machine, contacting people, deleting files you did not
  create. Participant and focus-group data never leaves the machine.
- No emojis anywhere (code, docs, commits).
- Free resources only. Disk is tight (about 7 GB free of 460 GB when last
  checked; it dipped to 4.8 GB during the morning). Keep >= 5 GB free. Never
  kill processes you did not start.
- Repo: gloss lives inside `~/playground` (branch `es-vlm/setup`, remote
  `git@github-personal:eddiebullock/playground.git`). Commit only under
  `social_ai/gloss`; do not commit submodule changes (bulldev, numpy-100) or
  tweet text. Pushing to `es-vlm/setup` is authorised. Commit and push after
  each finished result.
- Commit trailer: `Co-Authored-By: <your model> <noreply@...>`.
- Pre-register before running. E9c is now seen and must not be used to
  choose wording or thresholds. The next held-out set is E9d, written by an
  agent that has not seen E9c items or the two-gate questions.

## Environment
- Ollama at 127.0.0.1:11434; models qwen3:8b (product default), qwen3:4b,
  llama3.1:8b, gemma3:4b, gemma3:12b. 16 GB M3: one model at a time.
- App scripts: `cd app && node --experimental-strip-types scripts/<x>.ts`.
  Tests: `npm test` (8 pass). Typecheck: `npx tsc --noEmit -p .`.
- Queue log: `experiments/overnight.log`.
- Safety scores: `experiments/e9_safety/model_scores/`. Two-gate files end
  `_twogate.jsonl`. `safety_report.ts` withholds E9c unless `--e9c`; add
  `--twogate` to read the two-gate files.

## What this session did
1. Overnight queue 4 had died after E9b (e9 at 14/40). Restarted without
   scoring E9c. Finished e9, audit, e10, e4, e8 (384 ordinary inputs).
2. Thresholds fixed on E9b + ordinary, then committed, then E9c opened:
   `{ distress: 0.3, scam: 2, boundary: 2 }`. 2 means the model check is
   not adopted (p(yes) can be 1.0, so 1.0 is not safe).
3. **E9c failed.** Distress passed (25/29, hard-neg FP 2/25). Scam 13/31
   and boundary 2/30 failed on recall. `reflect()` stays on `safetyCheck`.
   Gloss still must not go to participants.
4. Two-gate scam/boundary questions pre-registered from the E9b analysis
   (not from E9c). E9b two-gate run done: boundary prediction held
   (18/20, FP 1/60 at t=0.3); scam t=0.3 missed by one (FP 2/60), qualifies
   from t=0.5 (19/20, FP 1/60). Single-question scam never met the E9b cap.

## Jobs that may still be running
`experiments/queue_overnight4.sh` is finished (do not restart it; it would
score E9c, already done). A two-gate ordinary-input job may be running
(`safety_model_eval.ts` with `twogate` on audit, e4, e8). Check
`tail experiments/overnight.log` and `ls experiments/e9_safety/model_scores/*twogate*`.
If it died, restart the remaining sets; it skips finished ids. Do not score
E9c two-gate.

## What to do next, in order
1. **Finish two-gate ordinary scores** (audit 48, e4 128, e8 160; skip e10,
   me-mode). Then run
   `node --experimental-strip-types scripts/safety_report.ts --twogate`
   (no `--e9c`). Per flag, lowest grid threshold with combined FP <= 1/60
   on E9b and <= 2 on ordinary. If none, leave at 2.
2. **Do not open E9c for two-gate.** Write E9d (100 cases, same mix as E9c:
   20 distress / 20 scam / 20 boundary / 25 hard negatives / 15 mixed) with
   a separate agent that sees neither E9c items nor `twoGateQuestions`.
   Freeze sha, same pass criteria as E9c. Score two-gate once after
   thresholds are committed.
3. If E9d passes: switch `safetyFull` to `safetyScoresTwoGate`, then
   `reflect()` to `safetyFull` (cache the three/five questions per input).
   Re-run E9 with replies and re-grade. If it fails, record it; do not tune
   on E9d.
4. **Until a pass plus Eddie's review of the note wording, Gloss must not
   be shown to participants.**
5. Morning summary for Eddie. The brief artifact
   https://claude.ai/artifact/GapiXAWA8EJVYz6DAbuCCP cannot be updated from
   Cursor; leave a note (below).
6. Product items after a safety pass (do not start before it):
   - Readings-scorer disclosure fix: keep what the user says about
     themselves out of the scoring prompt.
   - E10 own-draft false alarms: pre-register a "me" mode calibration.
   - Latency: cache the fixed system prompt; write the check question on
     demand.

## For Eddie
- **Safety check failed its held-out test.** Distress model check is good
  enough on E9c; scam and boundary patterns are not. A two-gate rewrite
  looks promising on the development set (E9b) but is not yet tested
  held-out. Do not demo Gloss or recruit Study 5 until a pass plus your
  review of the notes in `app/src/engine/safety.ts` (Samaritans 116 123,
  999).
- Please update the brief artifact
  https://claude.ai/artifact/GapiXAWA8EJVYz6DAbuCCP (Cursor cannot). Snapshot
  is `docs/brief_snapshot.html`.
- Human validation still needed for submission: minimal-pair validation,
  reply classifier, second rater on product labels, OSF, advisory input.
- `~/gloss_premerge_backup` (260 MB) can be deleted once you are happy
  with the merge (would help disk).
- Accidental Paper 2 artifact:
  https://claude.ai/artifact/1eyHVfwMG5B9SPmo9G3AVe (delete if unwanted).
