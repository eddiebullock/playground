# Handover, 2026-10-26 16:10 (E9d fail; two-gate scam/boundary pass)

For the next agent picking up Gloss. Read `AGENT_PROMPT.md`, `PLAN.md`,
`docs/PUBLICATION_STRATEGY.md` and the `docs/DECISIONS.md` entries dated
2026-10-25 and 2026-10-26 first. Overwrite this file when the work below
is done.

## Rules in force
- PLAN.md section 7. No money, no paid APIs, no cluster, no licensed data,
  nothing off the machine, no contacting people. Participant data stays
  local. Delete only files you created. Keep >= 5 GB free (disk was 3.7-6 GB
  today). Never kill other users' processes.
- No emojis. Repo `~/playground`, branch `es-vlm/setup`, remote
  `git@github-personal:eddiebullock/playground.git`. Commit only under
  `social_ai/gloss`. Push after each finished result.
- Commit trailer: `Co-Authored-By: <your model> <noreply@...>`.
- **E9c and E9d are both seen. Do not retune on them.** Do not rescore
  E9d. Next held-out set is E9e.

## Environment
- Ollama 127.0.0.1:11434; qwen3:8b product default. One model at a time.
  `safety_model_eval.ts` retries Ollama 500s; still resumable.
- `cd app && node --experimental-strip-types scripts/<x>.ts`.
  Tests: `npm test` (8 pass). Typecheck: `npx tsc --noEmit -p .`.
- Scores: `experiments/e9_safety/model_scores/`. Two-gate files end
  `_twogate.jsonl`. Report: `safety_report.ts [--e9c] [--e9d] [--twogate]`.

## Jobs
None running. Do not rerun `queue_overnight4.sh`.

## State
- v1 E9c: distress pass, scam/boundary fail. THRESHOLD
  `{ distress: 0.3, scam: 2, boundary: 2 }`.
- Two-gate E9d: **scam pass 29/30, boundary pass 30/30, distress fail
  24/30 (0.80 < 0.85)**. THRESHOLD_TWOGATE
  `{ distress: 0.3, scam: 0.5, boundary: 0.3 }` not wired.
- `reflect()` still uses `safetyCheck` only. Participants blocked.

## What to do next, in order
1. **Distress-question expansion, pre-registered from E9b not E9d.** Add
   the indirect cues already named in DECISIONS (goodbye letters, giving
   possessions away, no point in anything) to the distress question only.
   Write predictions and the adoption rule into DECISIONS, commit, then
   rescore E9b (+ ordinary distress if needed). Do not look at E9d to
   choose wording.
2. **E9e.** Same mix as E9d (20/20/20/25/15). Separate agent that has not
   seen E9c, E9d, or the new distress line. Freeze sha. Score two-gate
   (plus expanded distress) once at thresholds set on E9b. Same pass
   criteria as E9c/E9d.
3. If E9e passes: point `safetyFull` at `safetyScoresTwoGate` and
   `THRESHOLD_TWOGATE` (re-fit distress cut if the expansion changes it),
   switch `reflect()` to `safetyFull`, cache questions per input, re-run
   E9 with replies and re-grade. If fail: record; do not tune on E9e.
4. Until a pass plus Eddie's review of the notes in `safety.ts`, Gloss
   must not be shown to participants.
5. Cursor cannot update
   https://claude.ai/artifact/GapiXAWA8EJVYz6DAbuCCP.
6. After a safety pass: scorer disclosure fix; E10 "me" mode calibration;
   latency (cache system prompt; check question on demand).

## For Eddie
- Two-gate fixed the E9c scam/boundary hole on a fresh set. Distress is
  now the gate: it misses indirect risk (giving things away, farewell
  notes) that the question does not name. Still no demo / Study 5 until
  E9e passes and you review the Samaritans/999 notes.
- Please update the brief artifact. Snapshot: `docs/brief_snapshot.html`.
- Human validation still needed for the papers. Advisory group on notes.
- `~/gloss_premerge_backup` (260 MB) can go if you are happy with the merge
  (disk is tight).
- Accidental Paper 2 artifact:
  https://claude.ai/artifact/1eyHVfwMG5B9SPmo9G3AVe.
