# Paper outline: calibrated pragmatic inference in on-device language models

Working title: "Reading between the lines on a laptop: pragmatic accuracy and
calibrated deferral in small, private language models, against human data"

Target: *Autism* (methods/technology), *JMIR Formative Research*, or the CHI
late-breaking track. A natural sequel to Study 1 (video ToM in frontier
models): this one asks what is good enough for a private, deployable tool.

## Questions
1. How close do small open models that run on a laptop (4-8B) come to human
   accuracy on non-literal language: irony, indirect speech, white lies,
   maxims, metaphor, humour, coherence (Hu et al. 2023 items, n = 374 humans),
   and on sarcasm in dialogue (MUStARD)?
2. When they are wrong, how are they wrong? Literal readings vs over-reading.
3. Does their confidence predict their errors (AUROC, ECE)? Does it track
   human disagreement (item-level rho)? Is it good enough to support
   abstention (selective accuracy at fixed coverage)?
4. Which cheap interventions help: option-order averaging, a pragmatic-stance
   instruction, a short fact-check step, free reasoning, model size?

## Methods (as run)
- Option distribution read from first-token log-probs after an "Answer:"
  prefill; 5 rotated option orders per Hu item and 2 per MUStARD item (a
  fixed, balanced random subset of 200 clips, seed 42).
- Conditions: direct, stance, check, reason (1 order); qwen3 4B and 8B
  locally (M3, 16 GB).
- Baselines: published human data and model results (Hu et al.), chance.
- Metrics: accuracy with bootstrap CIs; AUROC(confidence -> correct); ECE;
  coverage and selective-accuracy curves; Spearman between model and human
  P(correct) and entropy per item; error-type counts by distractor class.

## Results so far (fill from `experiments/e1_pragmatics/results`)
- 4B direct: 0.548 vs humans 0.863. 48% of errors are literal readings.
  AUROC 0.59. At a confidence cut of 0.8, accuracy is only 0.73.
- 4B on MUStARD: at chance (0.515); calls 95% of lines sarcastic. The
  opposite failure, which suggests the model responds to how the question is
  framed rather than to the situation.
- 8B direct: see `summary.json`.

## Framing for the autism literature
- The failure mode of the cheapest models is literalism, which is the
  stereotype of autistic communication. An assistant built on them would add
  to the problem it claims to help with.
- Calibrated deferral, not accuracy, is the property that makes a tool safe
  to rely on. Report both.
- Double empathy: human item disagreement is the ground truth for
  "ambiguous". A follow-up (E7) collects autistic and non-autistic ratings of
  the same items, to test whether the model aligns with one group.

## Limitations to state
English, written vignettes, multiple choice, small models, one family
(Qwen3) unless disk allows more. Human data come from non-autistic or
unspecified samples.
