# Paper 2 draft: disclosure effects on AI social readings and advice

Working title: **"Telling the assistant you are autistic: disclosure
effects on readings and advice, and why prompt fixes move them rather than
remove them"**

Target: FAccT (first choice; January deadline), ASSETS or CHI as
alternatives; CHI late-breaking work for an early version. Status:
skeleton, 2026-10-25. [ON] = overnight queue; [EB] = needs Eddie.

## Question
Wohn et al. (CHI 2026) report that frontier chat models give autistic users
more avoidant social advice after disclosure. Two open questions follow:
(1) do the *readings* of other people's messages move too, or only the
advice; (2) can a tool's prompt remove such drift, or does a fix just move
it somewhere else?

## Design
- 48 everyday ambiguous messages (relation and channel balanced), each with
  four fixed readings (plain, mildly negative, hostile, joking). The first
  12 were used for development; 36 are new.
- Six reader conditions in the user's context: none; "I am autistic"; "I
  often get anxious about what messages mean"; "I have ADHD"; "I have
  depression"; **placebo "I am left-handed"**. The placebo separates the
  effect of a condition from the effect of saying anything about oneself.
- Readings audit: the scorer's probability over the four fixed readings
  (deterministic; 4 rotated orders). Outcomes: total variation from no
  disclosure, top-1 flips, negativity shift, sincere-reading shift, each
  also net of the placebo. Bootstrap CIs over messages.
- Advice audit: the full pipeline writes readings and 2-3 replies; each
  reply is classed engage / check / hold back / apologise-or-explain.
  Three generation seeds per condition give a noise floor.
- Models: qwen3 4B and 8B, llama3.1 8B, gemma3 4B and 12B. [EB:
  frontier models need paid APIs.]

## Results so far (qwen3:8b, 12 development messages)
- Readings: autism disclosure TV 0.075, 1/12 top-1 flips, negativity shift
  -0.04 (n.s.). Anxiety: TV 0.151, 3/12 flips, negativity -0.12 [-0.22,
  -0.03] (readings moved towards reassuring).
- Advice, 3 seeds: autism no drift (hold back +0.019 [-0.013, 0.053]);
  anxiety apologise +0.054 [+0.019, +0.093].
- **Fix 1** ("base replies on what happened, not on what I say about
  myself"): apologise +0.011 [-0.038, 0.067], but hold back +0.080 [+0.018,
  +0.153]. Apologise + hold back together +0.07 -> +0.09. The fix relabelled
  the drift.
- **Fix 2** (tie "not replying" to whether the message asks anything):
  anxious hold back +0.026 [-0.010, +0.066], apologise +0.002 [-0.041,
  +0.034]. Hold-back replies now follow the message (they rise on messages
  that ask nothing, in every condition) rather than the reader.
- **48-message readings audit with placebo (qwen3:8b).** Any sentence about
  the reader moves the readings (placebo "I am left-handed": TV 0.096,
  negativity -0.074 [-0.110, -0.040], i.e. softer readings). Net of the
  placebo, autism (+0.081 [+0.039, +0.131]), ADHD (+0.112) and depression
  (+0.144) make readings *more negative*; anxiety matches the placebo.
  Against no disclosure the components cancel for autism (+0.008), which is
  why a no-placebo audit reports "no drift".
- A "judge the words, not the reader" scorer instruction left all of this
  unchanged and moved the no-disclosure readings (6/48 top-1 flips).
- **Across 5 models** (negativity vs placebo; see DECISIONS 2026-10-26):
  depression more negative in all 5 (+0.03 to +0.14, every CI excludes 0);
  autism more negative only in qwen3:8b (+0.081) and llama3.1:8b (+0.038),
  null in qwen3:4b and both Gemmas; ADHD positive in 3 of 5; anxiety softer
  or null in all 5 (significantly softer in qwen3:4b and gemma3:12b). The
  placebo's own softening is large only in qwen3:8b (-0.074).

## Contribution claims (to be confirmed)
1. Disclosure drift appears in readings, not only advice. Against a placebo
   it is condition-specific and partly model-specific: depression makes
   readings more negative in every model tested, autism in 2 of 5, anxiety
   makes them softer or leaves them unchanged.
2. Prompt-level fixes can displace rather than remove drift; evaluations of
   fairness fixes must measure the whole outcome distribution, not the
   targeted category.
3. A placebo disclosure is a cheap, necessary control. In the shipped
   model it reveals a generic softening response to any self-description
   that hides the autism shift; this masking is large in one of five
   models, so audits cannot assume it away or assume it is present.

## Needed for submission
- [EB] Human validation of the reply classifier (two raters, 150 replies,
  kappa). Claude can provide a first pass.
- More seeds for the advice audit at 48 messages (about 3 h per seed per
  model on the M3; run over several nights).
- [EB] Positioning with an autistic co-author or advisory input (ASSETS
  expects it).
