# Handover, 2026-10-27 11:40 (E10 bands; check on demand)

For the next agent picking up Gloss. Read `AGENT_PROMPT.md`, `PLAN.md`,
`docs/PUBLICATION_STRATEGY.md` and the `docs/DECISIONS.md` entries dated
2026-10-25 to 2026-10-27 first. Overwrite this file when the work below
is done.

## Rules in force
- PLAN.md section 7. No money, no paid APIs, no cluster, no licensed data,
  nothing off the machine, no contacting people. Keep >= 5 GB free
  (disk ~9 GB). Never kill other users' processes. No emojis.
- Repo `~/playground`, branch `es-vlm/setup`, remote
  `git@github-personal:eddiebullock/playground.git`. Commit only under
  `social_ai/gloss`. Trailer: `Co-Authored-By: Cursor Grok 4.6 <noreply@cursor.com>`.
- Do not retune E9c/d/e or E10 even-half bands.

## Environment
- Ollama 127.0.0.1:11434; qwen3:8b. `cd app && npm test` (12 pass).
- Me-mode abstain 0.4, clear 0.7. Them-mode 0.7 / 0.5.
- Check question is on demand (`writeCheck`). Two-gate safety still runs
  inside `reflect()`.

## Jobs
None running.

## State
- E9e pass, two-gate wired. Participants blocked on Eddie's note review.
- E10 me-mode bands adopted (even FA 2/12).
- On-demand check shipped. Latency on 10 E4 items was *slower* than v2
  (done 36.7 s vs 25.4 s), likely two-gate in the same call.

## What to do next, in order
1. Eddie reviews `safety.ts` notes before any demo / Study 5.
2. **Latency:** overlap or defer two-gate so readings appear before
   safety notes, or cut two-gate questions. Pre-register before changing
   when notes appear. Do not retune safety thresholds.
3. Generation quality (white lies, metaphors) on odd/even E4 halves;
   E4 even has been used once.
4. Brief artifact still needs Eddie.

## For Eddie
- E9e passed. Please review Samaritans/999 notes.
- Me-mode is less alarmist on short drafts. Check questions are optional.
- Please update https://claude.ai/artifact/GapiXAWA8EJVYz6DAbuCCP
  (snapshot `docs/brief_snapshot.html`).
