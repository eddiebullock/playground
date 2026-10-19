# Analysis plan (internal)

**Status:** not using OSF. This is the locked analysis for the
task-conditioned paper. Steering is not gated on posting it.

## Title

Task-conditioned social versus mechanical residual directions in
open-weight vision-language models.

## Construct

Last-token residual while the model is asked a social or mechanical
question (with media). Directions are train-only difference of means.
Hold-out metric is projection AUROC of the train direction, not a
nested rebuild. Say social-inference direction / mechanical-inference
direction.

Narrative E/S prose is a closed pilot, not a confirmatory analysis.

## Designs

1. **Same-image RMET (primary).** Classic mental-state vs appearance
   on the same 36 faces. Also extract both questions with no image.
   Residualize the image direction against the text-only direction.
   Layer-0 saturation plus chance visual leftover = wording, not a
   mechanism.
2. **Video pair (primary).** EU-Emotions vs CLEVRER. Report layer-0
   AUROC as a content/modality check.
3. **Cross-modal (negative control, already run).** RMET vs CLEVRER.
   AUROC 1.0 at every layer including 0. Do not steer.

## Causal test (only if `steer_ok`)

`steer_ok` requires candidate hold AUROC >= 0.60 and no layer-0
confound on the score used to pick the layer. Add (and, if add
moves, ablate) at that locked layer. Alphas {−2, −1, 0, 1, 2}.
Hold-out social items; mechanical readout is hold CLEVRER (video
design) or a CLEVRER 128 shard (same-image design). Alpha 0 must
match unsteered. Predicted direction x domain interaction. Random
direction is a control if a real move appears.

## Transfer

Qwen3-VL-8B first. Qwen3-VL-2B and Gemma 4 E4B only if 8B steering
moves as predicted. Not a Qwen-family sweep.

## Out of this paper

Sparse autoencoders; EQ/SQ-R log-prob slopes; human tertile RSA;
Qwen2 / 2.5 / 32B; Mind Reading; further narrative add-sweeps.
