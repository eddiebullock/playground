# Handover, 2026-10-27 11:20 (E10 me-mode bands adopted)

For the next agent picking up Gloss. Read `AGENT_PROMPT.md`, `PLAN.md`,
`docs/PUBLICATION_STRATEGY.md` and the `docs/DECISIONS.md` entries dated
2026-10-25 to 2026-10-27 first. Overwrite this file when the work below
is done.

## Rules in force
- PLAN.md section 7. No money, no paid APIs, no cluster, no licensed data,
  nothing off the machine, no contacting people. Participant data stays
  local. Delete only files you created. Keep >= 5 GB free (disk ~9 GB).
  Never kill other users' processes.
- No emojis. Repo `~/playground`, branch `es-vlm/setup`, remote
  `git@github-personal:eddiebullock/playground.git`. Commit only under
  `social_ai/gloss`. Push after each finished result.
- Commit trailer: `Co-Authored-By: Cursor Grok 4.6 <noreply@cursor.com>`.
- **E9c, E9d and E9e are seen. Do not retune on them.** Two-gate
  thresholds stay `{ distress: 0.3, scam: 0.5, boundary: 0.3 }`.
- E10 even half has been used to test me-mode bands. Do not retune those
  cuts on E10.

## Environment
- Ollama 127.0.0.1:11434; qwen3:8b product default. One model at a time.
- `cd app && node --experimental-strip-types scripts/<x>.ts`.
  Tests: `npm test`. Typecheck: `npx tsc --noEmit -p .`.
- Product: two-gate safety; scorer drops About-you; me-mode abstain at 0.4.

## Jobs
None running.

## State
- E9e pass, two-gate wired. Participants blocked until Eddie reviews notes.
- E10 me-mode: ABSTAIN_BELOW_ME 0.4, CLEAR_AT_ME 0.7. Even FA 2/12.
  Them-mode still 0.7 / 0.5.

## What to do next, in order
1. Until Eddie reviews `safety.ts` notes, Gloss must not be shown to
   participants.
2. **Latency:** cache the fixed system prompt between calls; write the
   check question only when asked. Pre-register a latency remeasure on
   20 E4 items (medians vs the 2026-10-25 v2 numbers).
3. Paper 2 draft leftover-drift numbers are in; brief artifact still
   needs Eddie. Cursor cannot update
   https://claude.ai/artifact/GapiXAWA8EJVYz6DAbuCCP.

## For Eddie
- E9e passed. Please review Samaritans/999 notes before any demo.
- Me-mode no longer labels as many short plain drafts "open to different
  readings" (false alarm 6/12 even -> 2/12). Them-mode bands unchanged.
- Please update the brief artifact. Snapshot: `docs/brief_snapshot.html`.
