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

## Gate 1 — 8B extract (closed, job 38817548)

1. Design A: wording only. Raw and text-only AUROC 1.0; unique vs
   text ~0; leftover AUROC chance except a last-layer spike on n=11.
   Do not steer.
2. Design B: layer-0 AUROC 1.0. Content/modality confound. Do not
   steer.

## Gate 2 — 8B causal (skipped)

Would have run only if a design had `steer_ok=true` and not
`wording_only`. Neither did.

If a later extract is non-degenerate (candidate hold AUROC >= 0.60 and the
score used to pick the layer is not a layer-0 0.95+ confound):

- `PACK=task_same_image` and/or `PACK=task_video`
- add at the extract-locked candidate layer
- alphas {-2,-1,0,1,2}
- hold-out social items and a mechanical readout
- alpha 0 must match unsteered on that slice

Predicted: social+ raises the social task more than the mechanical
task; mechanical+ does the reverse. If the leftover is real but
steering is null, that is still the paper.

## Gate 3 — transfer (skipped)

8B extract did not produce a steerable direction. Do not run 2B or
Gemma.

## Stop rules

- Do not re-run Design C or narrative add-sweeps.
- Do not introduce SAEs unless a causal task direction exists; that
  is a second paper.
- Two failed fixes on the same cluster problem, then stop.
- Never write into study2 / study3 / study4_rmet.
