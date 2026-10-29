# Handover, 2026-10-29 (Live HUD + E11 protocol)

For the next agent. Read `docs/LIVE_BUILD.md`, `docs/research/05_frontier_safety.md`, then `docs/DECISIONS.md` 2026-10-29.

## Rules in force
- No money, no paid APIs unless `E11_ALLOW_PAID=1`. Nothing off the machine
  for call video. Keep >= 5 GB free. No emojis.
- Repo `~/playground`, branch `es-vlm/setup`, remote
  `git@github-personal:eddiebullock/playground.git`. Commit only under
  `social_ai/gloss`. Trailer: `Co-Authored-By: Cursor Grok 4.6 <noreply@cursor.com>`.
- E9c/d/e frozen. E11 holdout frozen until OSF + `HOLD_OUT_RELEASED`.

## Environment
- Product: `cd social_ai/gloss/live && npm run dev` then Chrome at
  http://127.0.0.1:5174/ (tab audio).
- Research tooling still at `social_ai/gloss/app` (Ollama qwen3:8b).
- E11 harness: `experiments/e11_frontier_safety/run.ts`.

## Jobs
None running.

## State
- Eddie pivoted: live HUD is the product; E11 is the safety paper.
- `live/` has the three-tab HUD (screen share, camera marker, turn
  meters, hedged emotion). Electron / system audio are not built.
- E11 protocol, codebook, prompts, 36 dev + 36 sealed holdout items,
  harness. Holdout not scored. No paid runs.

## What to do next, in order
1. Click through Gloss Live in Chrome on a Meet tab or a local video.
   Fix capture, face boxes, camera marker, two-channel audio.
2. Expand confirmatory items to 180 without scoring them. Hash the file.
3. Eddie: OSF snapshot of `prereg.md`; API keys if he wants frontier
   dry-runs on *dev* only.
4. Do not recruit. Do not demo to participants.

## For Eddie
- Try http://127.0.0.1:5174/ in Chrome. Share a Meet *tab* (tick Share
  audio) for turn-taking; a Zoom window is usually video-only.
- E11 will not run against GPT/Claude/Gemini/Grok until you set keys
  and `E11_ALLOW_PAID=1`. The protocol is ready for OSF.
- Please file the OSF snapshot when you are happy with `prereg.md`.
