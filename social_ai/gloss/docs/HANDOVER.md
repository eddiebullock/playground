# Handover, 2026-10-26 20:30 (E9e pass wired; original E9 re-run done)

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
  Tests: `npm test` (10 pass). Typecheck: `npx tsc --noEmit -p .`.
- Product: `safetyFull` / `reflect()` use two-gate + THRESHOLD_TWOGATE.
  Questions cached per input. Generation and safety scoring overlap.
  Apostrophes in `safety.ts` are folded so `I’ll send` matches COMPLY.

## Jobs
Leftover-drift audit may be running: `audit_v2.ts qwen3:8b stance --bare`
writing `experiments/e4_audit/v2_qwen3-8b_bare.json`. Do not change the
scorer while it runs. If it died, restart the same command.

## State
- E9e **PASS**. Official:
  `experiments/e9_safety/model_scores/e9e_official.txt`.
- Two-gate is wired. Original E9 re-run: distress 5/5, scam 5/5,
  boundary_pressure 5/5, coercion 0/5 (no prior no). Generation still
  offers complying replies; COMPLY unicode leak is fixed going forward.
  Do not quote the 38/40 LLM rubric as a result.
- Participants still blocked until Eddie reviews `safety.ts` notes.

## What to do next, in order
1. Until Eddie reviews `safety.ts` notes, Gloss must not be shown to
   participants (Study 5, demos).
2. **Finish leftover-drift** (`v2_qwen3-8b_bare.json`, 48 scenarios).
   Analyse with `python3 experiments/e4_audit/analyze_v2.py`. Record TV
   vs none. Do not put self-description back into the scorer from that
   run. Then E10 me-mode calibration.
3. **E10 "me" mode calibration:** clear-draft false-alarm target <= 0.2,
   bands fitted separately from "them" mode. Pre-register before tuning.
4. **Latency:** cache the fixed system prompt; write the check question
   only when asked.
5. Cursor cannot update
   https://claude.ai/artifact/GapiXAWA8EJVYz6DAbuCCP.

## For Eddie
- E9e passed and two-gate is in the product. Distress and scam notes now
  fire on the original red-team; boundary notes fire after a no, not on
  controlling asks with no prior refusal. Generation can still suggest
  complying replies; the filter now treats curly apostrophes as ASCII.
- Please review the Samaritans/999 notes in `app/src/engine/safety.ts`
  before any demo or Study 5. Advisory group on wording still needed.
- Please update the brief artifact. Snapshot: `docs/brief_snapshot.html`.
- Human validation still needed for the papers.
- `~/gloss_premerge_backup` (260 MB) can go if you are happy with the merge.
- Accidental Paper 2 artifact:
  https://claude.ai/artifact/1eyHVfwMG5B9SPmo9G3AVe.
