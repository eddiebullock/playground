# OSF pre-registration text (E11)

Paste into OSF as a snapshot before any holdout call. Do not edit after
the snapshot except via a dated amendment that does not peek at holdout.

## Title
When the assistant writes the reply: a pre-registered red-team of frontier
language models on crisis, fraud, and consent.

## Authors
Eddie Bullock (ARC, University of Cambridge). Additional authors TBA.

## Hypotheses
H1. Harmful-action rates on true-positive items will be above 10% for at
least one harm type in every consumer frontier model tested.
H2. An autistic-user social-assistant persona will increase scam and
boundary harmful-action relative to a generic assistant persona
(one-sided). It will not decrease distress harmful-action.
H3. A third persona with extra safety instructions will not reduce total
harm across the three types; any drop in one type will be offset by
hard-negative overflags or by another type (displacement).
H4. Hard-negative overflag rates will differ by model (two-sided). No
direction.

## Sampling
Confirmatory N = 180 synthetic English items (60 distress, 60 scam, 60
boundary; 40 true-positive and 20 hard-negative each). Development N = 36
is not analysed for H1–H4. Items are sealed; SHA-256 stored in-repo.

## Models and personas
Consumer: OpenAI GPT-4o or 4.1, Anthropic Claude Sonnet current, Google
Gemini 2.x Pro, xAI Grok current. Open: one 70B-class instruct model if
available, plus Qwen3-8B local. Personas: Generic; Social-autistic;
Social-autistic+safety (H3 only). Temperature 0. One sample per cell for
the primary analysis.

## Outcome
Human-coded `harmful_action` (0/1) per codebook v1. Two raters, blind to
model. Kappa >= 0.70 on a 60-output calibration set before holdout.
Adjudication by a third rater on primary disagreements.

## Analysis
Mixed-effects logistic regression
`harmful_action ~ model * persona * harm_type + (1 | item)`
on confirmatory true-positives for H1–H2. Bootstrap 95% CIs on per-model
rates. FDR on the H2 simple effects. Hard negatives: same model with
`overflag`. Exploratory: temperature 0.7 three-seed subset; US-locale
clone; safety persona (H3).

## Exclusion
A model arm that cannot be run is dropped, not replaced. Items are not
dropped after seeing outputs unless a rater flags dual-use risk, in which
case the raw string is restricted and the code is kept.

## Open materials
Codebook, prompts, analysis scripts, model snapshots, aggregated rates.
Full scam item text not in the public paper.
