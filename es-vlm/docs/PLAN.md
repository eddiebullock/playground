# One-paper plan

Locked in `docs/PROPOSAL.md`. This file is the execution order and
the go / no-go gates. Do not add SAEs, EQ/SQ-R slopes, human RSA,
extra Qwens, or Mind Reading here.

## What one paper is

Task-conditioned social vs mechanical residual directions in a VLM,
with matched-modality controls, a same-image question control, a
documented cross-modal confound, and a confirmatory steer on held-out
items if the extract is non-degenerate. Transfer is 2B + Gemma only if
8B works. Narrative E/S prose is a closed pilot, cited not re-run.

## Already closed

- Narrative extract + personas: linear leftover in Qwen, chance in
  Gemma, personas not independent.
- Narrative causal at 8B L27: add hurts RMET on every vector; ablate
  is a no-op; position=all is a generic CLEVRER shove.
- Design C (RMET faces vs CLEVRER videos), job 38807101: hold-out
  AUROC 1.0 at every layer including 0. Confound check. Do not steer.

## Gate 1 — 8B extract (now)

One GPU hour, `PACK=paper8b`, `MODEL=qwen3vl_8b`:

1. Design A: same-image RMET classic vs appearance, plus text-only
   residualize. Inspect `visual_residual_auroc_by_layer`,
   `wording_only`, `steer_ok`.
2. Design B: EU-Emotions video vs CLEVRER video. Inspect layer-0
   AUROC and `steer_ok`.

Do not queue 2B, Gemma, or steer in the same wave.

## Gate 2 — 8B causal (only if Gate 1 is non-degenerate)

If a design has `steer_ok=true` (candidate hold AUROC >= 0.60 and the
score used to pick the layer is not a layer-0 0.95+ confound):

- `PACK=task_same_image` and/or `PACK=task_video`
- add at the extract-locked candidate layer
- alphas {-2,-1,0,1,2}
- hold-out social items and a mechanical readout
- alpha 0 must match unsteered on that slice

Predicted: social+ raises the social task more than the mechanical
task; mechanical+ does the reverse. If the leftover is real but
steering is null, that is still the paper.

## Gate 3 — transfer (only if Gate 2 moves as predicted)

Repeat extract + steer on Qwen3-VL-2B and Gemma 4 E4B. Stop if 8B
steering is null or generic.

## Stop rules

- Do not re-run Design C or narrative add-sweeps.
- Do not introduce SAEs unless a causal task direction exists; that
  is a second paper.
- Two failed fixes on the same cluster problem, then stop.
- Never write into study2 / study3 / study4_rmet.
