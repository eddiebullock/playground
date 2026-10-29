# Prompt for a Cursor agent taking over Gloss

Paste the text below into a new Cursor agent chat opened on
`~/playground` (the project is `social_ai/gloss`).

---

You are taking over Gloss, a PhD project by Eddie Bullock (ARC,
Cambridge). There are two tracks. Do both; do not collapse them.

**Product (the thing people might actually use):** Gloss Live, a local
realtime HUD. The user shares a Zoom / Teams / Meet / video window.
Three tabs only: eye contact (where to look: the camera, not the face
tile), turn taking (when to speak and how long), emotion recognition
(hedged face-shape, never "they are angry"). Spec:
`social_ai/gloss/docs/LIVE_BUILD.md`. Code: `social_ai/gloss/live/`.
Start from that spec if the app is incomplete. Do not revive the
copy-paste Reflect UI as the product; `app/` is research tooling.

**Research (high-impact safety paper):** E11, a pre-registered red-team
of *frontier* chat models used as social intermediaries. Not a test of
Gloss. Protocol: `social_ai/gloss/docs/research/05_frontier_safety.md`.
Harness: `social_ai/gloss/experiments/e11_frontier_safety/`. Holdout is
sealed. Do not score it. Do not spend money. Paid APIs only if Eddie has
set `E11_ALLOW_PAID=1` and the vendor keys.

Start by reading, in this order:
1. `social_ai/gloss/docs/HANDOVER.md`
2. `social_ai/gloss/docs/LIVE_BUILD.md`
3. `social_ai/gloss/docs/research/05_frontier_safety.md`
4. `social_ai/gloss/docs/DECISIONS.md` (newest last)
5. `social_ai/gloss/PLAN.md` (direction as of 2026-10-29: live first)

Hard rules:
- Never use emojis anywhere.
- Never spend money. Never call paid APIs unless `E11_ALLOW_PAID=1`.
  No cluster, no licensed datasets, nothing off the machine for
  participant or call data. Screen capture stays in the browser.
- Keep at least 5 GB free. Delete only files you created. Do not kill
  other users' processes.
- One Ollama model at a time.
- Git: commit only files under `social_ai/gloss`, on branch
  `es-vlm/setup`, push to `origin`
  (`git@github-personal:eddiebullock/playground.git`) after each
  finished slice. Never commit submodules or `notes/`. Trailer:
  `Co-Authored-By: Cursor Grok 4.6 <noreply@cursor.com>`.
- Pre-register before experiments. Never tune on holdout. E9c/d/e and
  E11 holdout are frozen. E9 is engineering history, not Paper 3.
- Do not show Gloss Live to participants until Eddie and PREC say so.
- Hedged language. No scoring the user. Eye-contact tab is a camera
  target, not a stare coach. Emotion tab always says a face is not a
  feeling.

First actions:
1. `cd social_ai/gloss/live && npm run dev` and click through the three
   tabs in Chrome (tab audio matters). Fix what is broken.
2. Follow LIVE_BUILD.md section 5 for whatever is not built.
3. Do not run E11 holdout. A 1-item Ollama dry-run of the harness is
   allowed.
4. Overwrite HANDOVER.md when you stop.

The brief artifact on claude.ai cannot be updated from Cursor.
