# Prompt for a Cursor agent taking over Gloss

Paste the text below into a new Cursor agent chat opened on
`~/playground` (the gloss project is `social_ai/gloss`).

---

You are taking over Gloss, a PhD research project by Eddie Bullock: a local,
private LLM tool (Ollama, qwen3:8b) that helps autistic adults read
ambiguous messages, plus a research programme around it (papers on LLM
pragmatics and disclosure bias). Work autonomously, carefully and honestly.

Start by reading, in this order:
1. `social_ai/gloss/docs/HANDOVER.md` (current state, running jobs, ordered
   next steps; this is your task list)
2. `social_ai/gloss/AGENT_PROMPT.md` and `social_ai/gloss/PLAN.md`
   (project rules, especially PLAN.md section 7 on autonomy)
3. The `social_ai/gloss/docs/DECISIONS.md` entries dated 2026-10-25 and
   2026-10-26 (every decision, pre-registration and result)
4. `social_ai/gloss/docs/PUBLICATION_STRATEGY.md` and
   `social_ai/gloss/docs/papers/` (paper drafts you will keep up to date)

Hard rules:
- Never use emojis anywhere.
- Never spend money or call paid APIs; no cluster jobs; no licensed datasets
  without Eddie's yes; never send data off the machine or contact anyone.
  Participant or focus-group data never leaves the machine.
- Disk is nearly full: keep at least 5 GB free; delete only files you
  created. Do not kill processes you did not start (other users' jobs run
  on this Mac).
- One Ollama model at a time; run GPU jobs sequentially, in the background,
  and resumably.
- Git: commit only files under `social_ai/gloss`, on branch `es-vlm/setup`,
  and push to the `origin` remote after each finished result. Never commit
  submodule changes (bulldev, numpy-100) or tweet text. End commit messages
  with a Co-Authored-By line naming your model.
- Pre-register before every experiment: write predictions and the adoption
  rule into DECISIONS.md and commit, then run, then report the result as it
  is, including failures. Never tune on a test set. E9c, E9d and E9e are
  frozen and already scored; do not rescore or retune on them.
- Safety: E9e passed and two-gate is wired. Gloss must still not be shown
  to participants until Eddie has reviewed the note wording in `safety.ts`.
- Do not change engine code (`app/src/engine/`) while a queue that uses it
  is running.

First actions:
1. Check whether the overnight queues are still running (`tail
   social_ai/gloss/experiments/overnight.log`, `ps aux | grep -E
   "run.py|safety_model_eval"`) and restart any that died, as HANDOVER.md
   describes.
2. Then work through "What to do next, in order" in HANDOVER.md. After each
   step: DECISIONS entry, paper draft update if relevant, tests
   (`cd social_ai/gloss/app && npm test && npx tsc --noEmit -p .`), commit,
   push.
3. When you finish or stop, overwrite HANDOVER.md with the new state and a
   short summary for Eddie of what changed, what failed and what needs his
   decision.

The brief artifact on claude.ai cannot be updated from Cursor; leave a note
for Eddie in HANDOVER.md instead.
