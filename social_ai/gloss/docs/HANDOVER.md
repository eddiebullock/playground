# Handover, 2026-10-26 21:00 (E9e wired; scorer About-me strip adopted)

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
- **E9c, E9d and E9e are seen. Do not retune on them.** Two-gate
  thresholds stay `{ distress: 0.3, scam: 0.5, boundary: 0.3 }`. Original
  E9 (40 cases) informed the patterns; do not widen two-gate from it.

## Environment
- Ollama 127.0.0.1:11434; qwen3:8b product default. One model at a time.
- `cd app && node --experimental-strip-types scripts/<x>.ts`.
  Tests: `npm test` (11 pass). Typecheck: `npx tsc --noEmit -p .`.
- Product: `safetyFull` / `reflect()` use two-gate + THRESHOLD_TWOGATE.
  Scorer drops `About me:` lines and the `aboutMe` field. Generation and
  replies still see them. Distress still reads `aboutMe`.

## Jobs
None running.

## State
- E9e **PASS**. Two-gate is wired. Original E9: distress/scam/boundary-after-a-no
  notes fire; coercion without a prior no does not. COMPLY folds curly
  apostrophes. Do not quote the 38/40 LLM rubric.
- Scorer About-me strip adopted (max TV 0 on 5 scenarios). Leftover drift
  when the same sentence sits in "what came before" without the prefix is
  still ~0.10 TV (same as shipped). Dedicated `aboutMe` field is the
  product path that the scorer never sees.
- Participants still blocked until Eddie reviews `safety.ts` notes.

## What to do next, in order
1. Until Eddie reviews `safety.ts` notes, Gloss must not be shown to
   participants (Study 5, demos).
2. **E10 "me" mode calibration:** clear-draft false-alarm target <= 0.2,
   bands fitted separately from "them" mode. Pre-register before tuning.
3. **Latency:** cache the fixed system prompt; write the check question
   only when asked.
4. Paper 2 draft: add the leftover-drift number and the dedicated field.
   Cursor cannot update
   https://claude.ai/artifact/GapiXAWA8EJVYz6DAbuCCP.

## For Eddie
- E9e passed and two-gate is in the product. Please review the
  Samaritans/999 notes in `app/src/engine/safety.ts` before any demo.
- Self-description in a labelled About-you field no longer moves the
  readings. The same sentence typed into "what came before" still does
  (~0.10 TV), same as the shipped audit.
- Please update the brief artifact. Snapshot: `docs/brief_snapshot.html`.
- Human validation still needed for the papers.
- `~/gloss_premerge_backup` (260 MB) can go if you are happy with the merge.
