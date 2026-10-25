# Handover, 2026-10-25 16:45 (laptop restart)

For the next agent picking up Gloss. Read `AGENT_PROMPT.md`, `PLAN.md` and
the newest `docs/DECISIONS.md` entries (2026-10-25) as usual; this file is
only the state at the moment of the restart. Delete or overwrite it once
the work below is done.

## Rules in force
- PLAN.md section 7 autonomy rules. Ask Eddie before: spending money,
  cluster jobs, licensed data (Mindreading, EU-Emotions), sending anything
  off the machine, contacting people, committing to the main playground
  repo, deleting files Claude did not create.
- No emojis anywhere. Free resources only; mind disk (about 25 GB free).
- `gloss` is now part of the playground repo (merged with history on
  2026-10-25, branch `es-vlm/setup`); commit there, only under `social_ai/gloss`. `experiments/*/results/
  *.jsonl` and `*.log` are gitignored on purpose.
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Environment
- Ollama at 127.0.0.1:11434 with qwen3:8b (temperature 0.3, seed settable
  via `setSeed` in `app/src/engine/llm.ts`). After the restart check it is
  up (`curl -s 127.0.0.1:11434/api/tags`); start it with `ollama serve` if not.
- Scripts run from `app/` with `node --experimental-strip-types scripts/...`.

## Interrupted job: fix-2 replies re-audit
What: the stereotype audit of "Ideas for replying" (Wohn et al. CHI 2026
design rule) after fix 2. Fix 2 is committed (b6f72b6) in
`app/src/engine/reflect.ts`, line about "Offer not replying, or replying
later, only if their message does not ask me anything or need an answer."

State at restart:
- seed 0 finished: `experiments/e4_audit/replies_fix2_seed0.json` (committed).
- seed 1 was at message 8 of 12 and is lost (the script writes only at the
  end). Seed 2 not started.

Resume: `experiments/e4_audit/resume_fix2.sh` (runs seeds 1 and 2, about
30 min each, writes FIX2_DONE to `replies_fix2_seed2.log`). Run it in the
background.

Then analyse:
```
cd experiments/e4_audit
python3 noise.py replies_fix2_seed0.json replies_fix2_seed1.json replies_fix2_seed2.json
```
Compare with (anxious vs no disclosure, 3 seeds, bootstrap 95% CI):

| Engine | anxious apologise | anxious hold back |
|---|---|---|
| pre-fix (`replies_qwen3-8b*.json`) | +0.054 [+0.019, +0.093] | +0.017 [-0.007, +0.046] |
| fix 1 (`replies_fix_seed*.json`) | +0.011 [-0.038, +0.067] | +0.080 [+0.018, +0.153] |
| fix 2 | to do | to do |

Noise floor (seed-to-seed shift of a condition mean): about 0.03 on average,
max about 0.07-0.09. Autism disclosure has shown no drift in every run.
Success for fix 2: anxious hold back and apologise CIs both include 0, and
the "check" and "engage" shares under no disclosure do not drop by more than
the noise floor (fix 1 none: engage 0.47, check 0.45). Also read the
hold-back replies per condition (replies whose `kinds` argmax is index 2,
under `scenarios[i][condition]`). If fix 2 fails, consider
reverting to fix 1 or trying a different wording, then re-audit.

After analysis:
1. Add a dated entry to `docs/DECISIONS.md` and commit with the jsons.
2. Update the brief artifact https://claude.ai/artifact/GapiXAWA8EJVYz6DAbuCCP
   (audit paragraph in the "What the product itself gets right" section,
   and the E4 audit row in the programme table, now reading "autism clear;
   anxiety fix 2 re-testing"). From a new session: Artifact `read` with that
   url first, then publish with `url`. A copy of the source as
   published (Version 4) is `docs/brief_snapshot.html`.

## Next steps after that (PLAN.md Sequence)
1. Readings-scorer anxiety check: does disclosure change the stance
   weights or top reading (same 12 messages, `app/scripts/audit_scenarios.ts`).
2. Generation misses: white lies, metaphors, who-said-what. Develop only on
   the odd E4 half.
3. Latency cuts (v2 medians: first reading 9.1 s, done 26.6 s): keep the
   fixed system prompt cached between calls; write the check question only
   on demand. Measure with `app/scripts/latency.ts`.
4. New held-out items: both E4 halves have now been seen.
5. Keep the brief in step.

## Open questions for Eddie (in the brief)
- Should a review pack be prepared for an autistic-led organisation before
  Study 5? Contacting them is his call.
- The direction (language first, reflective first, faces last) still needs
  his explicit agreement.
