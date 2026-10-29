# E11: frontier social-assistant red-team

This experiment evaluates **other companies' chat models**, not Cue.
Protocol: `docs/research/05_frontier_safety.md`. Paper:
`docs/papers/paper3_frontier_safety.md`.

Do not score `items_holdout.json` until an OSF snapshot exists and this
folder contains a file named `HOLD_OUT_RELEASED` (Eddie creates that).

## Dry-run (local, free)

```
cd social_ai/cue/experiments/e11_frontier_safety
node --experimental-strip-types run.ts --split dev --provider ollama --model qwen3:8b --persona generic --limit 1
```

## Paid providers

Requires Eddie's yes, `E11_ALLOW_PAID=1`, and the vendor key.

```
E11_ALLOW_PAID=1 OPENAI_API_KEY=... node --experimental-strip-types run.ts \
  --split dev --provider openai --model gpt-4o --persona social_autistic --limit 1
```

Providers: `ollama`, `openai`, `anthropic`, `google`, `xai`.

## After a run

Humans code `runs/*.jsonl` with `codebook.md`. Do not use an LLM as the
confirmatory label.
