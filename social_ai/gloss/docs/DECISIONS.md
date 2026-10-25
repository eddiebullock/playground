# Decisions log

Dated, with the evidence. Newest last.

## 2026-10-24

- **New repo `~/playground/gloss`**, separate from `social_ai` (concept
  scenes) and `es-vlm`. It is its own git repo, so the playground branch
  `es-vlm/setup` and its uncommitted changes are untouched.
- **Order of work: language first, reflective first, faces last.** Evidence:
  `docs/research/01_landscape.md` sections 1, 4 and 5 and
  `03_study_spinout.md` section 2. EU AI Act Art. 5(1)(f) covers face and
  voice only. Barrett 2019. The community critiques. No reflective or
  calibrated tool exists. Pending Eddie's agreement and the focus-group
  transcripts.
- **Local-only inference for participant data.** Free cloud tiers either
  exclude UK/EEA apps or allow human review of inputs (`02_tech.md` section 5).
  No paid APIs (standing rule).
- **Confidence comes from first-token option probabilities averaged over
  rotated orders**, not from verbalised confidence. This is the same mechanism
  E1 measures, so the product's ambiguity rating has a measured calibration.
- **The literal or sincere reading is always a candidate.** This guards
  against over-reading and against reinforcing anxious interpretations (the
  sycophancy risk in Ma et al. 2026).
- **E1 result, qwen3:4b direct (Hu et al. items, 169 x 5 orders):** accuracy
  0.548 [0.514, 0.580] vs human 0.863. Confidence barely predicts errors
  (AUROC 0.594, ECE 0.237). 182 of 382 errors (48%) are literal readings.
  Published references on the same items: text-davinci-002 0.697,
  flan-t5-xxl 0.580. **Implication: a 4B on-device model is not good enough
  to show readings with confidence. Study 5 needs a stronger model, or the
  product must lean on abstention.** E2 (8B) is blocked by disk: 5.3 GB free,
  and other sessions are writing.
- **Provisional thresholds:** "clear" when the top reading has p >= 0.75,
  abstain below 0.5. To be re-set from the E1/E2 selective-accuracy curves.
- **E1/E2, qwen3:8b direct (169 x 5 orders):** accuracy 0.683 [0.651, 0.712]
  per trial, 0.692 [0.621, 0.763] order-averaged, about the level of
  text-davinci-002. AUROC 0.72, ECE 0.21. Item entropy alignment with humans
  rho 0.37 (4B: 0.14). Overconfident: 67% of items get >= 0.9 confidence and
  0.78 of those are right. **Irony is unsolved at 0.51** (humans 0.92), and
  53% of errors are still literal. Model size helps calibration and most
  phenomena but not literalism.
  Implication: the 8B model is the floor for Study 5. Thresholds must be set
  on 8B after calibration fixes, because the provisional 0.75 cut would give
  only 0.73 accuracy.
- **E1, qwen3:4b MUStARD (balanced 200 x 2):** 0.515, chance. Recall 0.955,
  precision 0.508: it over-calls sarcasm when the question names it.
- **Style note:** 4B echoes the user's text back; 8B writes a usable,
  reader-facing note. A third sign that the 4B tier is below the product bar.
- **E3, temperature scaling (5-fold CV over items):** qwen3:8b T = 4.4 (folds
  4.0-4.7), ECE 0.21 -> 0.08. With calibrated cuts: >= 0.7 covers 56% of items
  at 0.82 accuracy, and >= 0.6 covers 66% at 0.79. qwen3:4b T = 1.7, but its
  ranking is too weak (AUROC 0.62) for any cut to reach 0.7 accuracy.
  **Product change:** per-model temperature in the engine, "clear" at
  calibrated >= 0.7, abstain below 0.5, default model qwen3:8b. Caveat: the
  temperature was fitted on test items, not on Gloss's own generated
  readings. Study 5 checks the transfer by comparing diary ratings across
  clarity bands.
- **E1b, pragmatic-stance instruction (the largest effect so far):**
  qwen3:8b 0.692 -> 0.781 [0.716, 0.840] order-averaged. Irony 0.51 -> 0.85.
  AUROC 0.72 -> 0.82. Most-confident-half accuracy 0.95. Literal errors
  182 -> 98, other errors 86 -> 88, so there is no rise in non-literal
  mistakes on these items. Calibrated (T = 3.4, ECE 0.044): a cut at >= 0.7
  covers 71% of items at 0.91 accuracy, above the human mean of 0.86.
  qwen3:4b 0.548 -> 0.646 per trial.
  **Caveat:** every Hu item has a non-literal correct answer, so this cannot
  detect over-reading of sincere messages. The MUStARD stance run (50%
  sincere) decides whether the product adopts the stance prompt. Watch the
  sincere-item false positives.
- **Sincere minimal pairs (25 irony items with the context rewritten so the
  remark is sincere; written by Claude, not yet human-validated):** 8B direct
  0.976, with stance 0.904. Balanced accuracy over the irony/sincere pairs:
  direct 0.74, stance 0.88. The over-readings cluster on remarks that
  conventionally signal sarcasm ("their generosity knows no bounds", "I see
  your revision is going well"). **Decision: the product scorer uses the
  stance prompt, with T = 3.4 (8B) and 1.8 (4B).** MUStARD stays at chance
  for 8B direct (0.535, recall 0.96). TV sarcasm without audio or visual
  context is a weak test; E5 (prosody) would address it.
- **Generation fix:** readings must differ in what they mean for the user, and
  the first is the plain meaning in its ordinary, kind sense. Before the fix,
  near-paraphrased readings split the probability, and clear messages were
  wrongly flagged "Hard to tell". After it, the three development cases read
  correctly (sarcasm 0.93; sincere work messages 0.87 and 0.89). These three
  cases are a development set, not an evaluation.
- **Moved to `~/playground/social_ai/gloss` (2026-10-25)** at Eddie's request, so
  the work lives under the personal `eddiebullock/playground` GitHub repo rather
  than the work `minimal` repo. The earlier "separate repo" entry above is superseded.

## 2026-10-25

- **Focus groups 2-4 read** (`docs/research/04_focus_groups.md`, participants
  coded, no quotes). The decisions it drives are tabled there. Biggest changes:
  an optional "your read first" step (learning and autonomy without forcing
  either), a way to correct Gloss with your own reading, private people notes
  as context, opt-in ways to reply with how each might land, boundary
  affirmation, and a guardrail against ever advising the user to suppress
  interests or mask. Glasses are disliked by most of this group; phone first,
  haptics or earpiece for live.
- **E4 product eval, hypothesis (written before running).** The full Reflect
  pipeline (generate readings, then weigh them) on 103 Hu stories that look
  like messages (Irony, IndirectSpeech, Deceits, Maxims, Metaphor) plus the 25
  sincere minimal pairs. Each generated reading is mapped to the story's
  original options by a judge call, and Claude hand-checks a sample of the
  judge. Prediction: top reading matches the intended meaning on 0.65-0.75 of
  Hu items (below E1's 0.78, because generation can miss the right reading),
  some reading matches it on >= 0.85, and the plain reading comes top on
  >= 0.8 of the sincere items. **What would change the plan:** top-1 < 0.6, or
  sincere < 0.7, makes reading generation the bottleneck and the next work is
  the generation prompt, not the UI. High recall but low top-1 points at the
  scorer instead.
- **E1 runs finished after the interruption.** MUStARD with the stance
  instruction stays at chance for both sizes: 8B 0.530 [0.477, 0.580],
  recall 0.98, precision 0.52; 4B 0.510. When the question itself names
  sarcasm, both models say "sarcastic" to almost every TV line, with or
  without the instruction. This supports the product's design of never asking
  "is this sarcastic?" and instead weighing concrete readings, one of which is
  always the plain one. It also leaves E5 (prosody) as the only route to the
  TV-sarcasm question. Sincere minimal pairs on 4B: direct 0.832, stance 0.888
  (the stance helps 4B on sincere items, unlike 8B: 0.976 -> 0.904).
- **E1c fact-check mode is dropped.** The model writes one or two sentences
  comparing the words with the situation, then answers. On the same 169 Hu
  items (seed 0) it is worse than the stance instruction for both sizes: 8B
  0.716 vs 0.775 (stance-only correct 18, check-only 8, sign test p = 0.08);
  4B 0.586 vs 0.627 (19 vs 12, p = 0.28). It also costs 8-13 s per item. The
  scorer stays on the stance instruction with no reasoning step.
- **E4 stereotype audit (readings).** Twelve ambiguous messages, four fixed
  readings each (sincere, mildly negative, hostile, joking), scored with no
  disclosure, with "I am autistic" and with "I get anxious" in the user's
  context. Measure: total variation (TV) distance from no disclosure, top-1
  flips, and a negativity index (mild = 1, hostile = 2).
  | Model | Disclosure | mean TV | top-1 flips | negativity shift (t, df 11) |
  |---|---|---|---|---|
  | 4B | autistic | 0.067 | 1/12 | +0.036 (1.69) |
  | 4B | anxious | 0.046 | 0/12 | +0.012 (0.94) |
  | 8B | autistic | 0.075 | 1/12 | -0.040 (-1.27) |
  | 8B | anxious | 0.151 | 3/12 | -0.122 (-2.41) |
  Disclosing autism does not move readings much on either model, and there is
  no significant shift in either direction. Disclosing anxiety on 8B pushes
  readings towards the reassuring one. That is
  kinder, but it is still a shift: the words have not changed, so the
  reading should not either. Not fixed yet; noted for the scorer prompt
  ("judge the words, not the reader"). What this audit does not cover is
  *advice*: the Wohn et al. effect is about advice drifting towards avoidance,
  and "ideas for replying" is now the place where that could happen. Next
  audit: run waysToReply with and without disclosure and count avoidance
  options (decline, withdraw, say less).
- **E4 stereotype audit (ideas for replying, 8B).** The same twelve
  messages through the full pipeline plus waysToReply (three replies each),
  with no disclosure, "I am autistic" and "I get anxious". Each reply is
  sorted by the model into engage / check / hold back / apologise-or-explain
  (first-token probabilities over rotated orders). Mean share, n = 36 replies:
  | Disclosure | engage | check | hold back | apologise |
  |---|---|---|---|---|
  | none | 0.47 | 0.43 | 0.07 | 0.04 |
  | autistic | 0.45 | 0.42 | 0.10 | 0.03 |
  | anxious | 0.44 | 0.38 | 0.06 | 0.13 |
  Paired by message, with bootstrap 95% CIs: autistic, hold back +0.03
  [-0.05, 0.13] and apologise 0.00 [-0.07, 0.07]. So the Wohn et al. drift
  towards avoidance does not show for autism disclosure here. Anxious,
  apologise +0.09 [-0.01, 0.21] (4 messages up, 1 down). That is borderline,
  and it goes the same way as the anxiety shift in the readings audit: the
  model reassures and over-explains for an anxious user (e.g. "I didn't
  realize it was a problem. I'll try to be earlier next time." for "Nice of
  you to show up."). Caveat: one generation per condition, so sampling noise
  is not separated from the disclosure effect, and readings differ between
  conditions too. Next: repeat runs per condition (3 seeds) to measure the
  noise floor (queued: `experiments/e4_audit/queue_seeds.sh`). The reply
  prompt already says "Do not tell me to apologise unless I clearly did
  something wrong", so the anxious drift gets past that rule. If the noise
  floor confirms it, the fix is to keep the disclosure out of the reply call
  (or the scorer) rather than add more wording.
- **E4 baseline (pre-focus-group engine, 8B): the prediction failed.** The
  8B judge that maps generated readings to the story's options turned out
  unreliable: against Claude's hand labels of all 388 readings it agrees on
  0.72, with precision 0.58 (it calls loose paraphrases matches and plain
  sincere readings "None of the above"). So every number below uses the hand
  labels (`experiments/e4_product/hand_labels/`, one rater, not blind to the
  engine; the next run is labelled blind, mixed with these). The judge is
  kept only as a rough first pass.
  | Measure | Predicted | Result (hand labels) |
  |---|---|---|
  | Top reading matches intended meaning (103 Hu items) | 0.65-0.75 | **0.563** [0.456, 0.660] |
  | Some reading matches | >= 0.85 | 0.748 [0.660, 0.835] |
  | Top-1 when the right reading was generated | - | 0.753 |
  | Sincere: plain reading comes top (25) | >= 0.8 | 0.720 [0.520, 0.880] |
  | Sincere: plain reading generated | - | 1.000 |
  By clarity band: "Fairly clear" 0.76 (36% of items), "Some" 0.53, "Hard to
  tell" 0.46. So the band means something: the clear band is close to E1's
  overall accuracy. Spread vs human disagreement: rho 0.28. Latency median
  39 s (p90 45 s). By task: Irony 0.76, IndirectSpeech 0.55, Metaphor 0.55,
  Maxims 0.47, Deceits 0.42.
  **Reading.** Both steps lose accuracy. Generation misses the intended
  meaning on a quarter of items, and the scorer then ranks it top only three
  times in four. The generation misses cluster: (a) replies that dodge the
  question or say too much or too little (Maxims: the meaning is in what is
  *not* said, which is focus-group theme T8); (b) the kind motive behind a
  white lie (Deceits: "does not want to hurt her"); (c) metaphors glossed
  near-literally ("the kids are stretchy"); (d) confusing who said what in
  stories with three people (partly an artefact of the story format, which
  real messages do not have). On the sincere items the plain reading is always
  offered, but a plausible sarcastic reading sometimes outranks it. The
  stance instruction pulls towards non-literal readings once both are on
  the table. As pre-registered, the next work is the generation prompt, not
  the UI. Caveat: Claude has now seen failures across all items, so any
  prompt change made after this is checked on the even-numbered items only
  (odd = development) and should be confirmed on new items.
- **E4 blind labels, current engine, and generation prompt v2.** All
  readings from three runs were hand-labelled blind (`experiments/e4_product/
  blind.py`: runs mixed, cases shuffled, readings shuffled within a case, no
  engine, rank or probability shown). Label rule: 1 if the reading gives the
  speaker the intended belief or motive, even when framed as hidden or
  polite; for indirect requests the reading must contain the request.
  Test-retest: the 388 baseline readings relabelled blind agree with the
  earlier unblinded labels on 0.93 (kappa 0.85); one rater, so this is
  consistency, not validity. The 8B judge still agrees with hand labels on
  only 0.74 (precision 0.62).
  Current engine (after the focus-group changes) against the baseline, all
  128 items, blind labels:
  | Measure | Baseline | Current |
  |---|---|---|
  | Top reading = intended (103 Hu) | 0.534 [0.437, 0.631] | 0.495 [0.398, 0.592] |
  | Some reading = intended | 0.738 | 0.680 |
  | Sincere: plain reading top (25) | 0.720 | 0.720 |
  | Sincere: plain reading offered | 0.960 | 0.960 |
  | Clarity "Fairly clear": top-1 | 0.74 (36% of items) | 0.67 (30%) |
  | Spread vs human disagreement (rho) | 0.28 | 0.30 |
  Paired on the odd half: no significant difference (top-1 5 vs 3, recall
  6 vs 1, p = 0.13). The focus-group changes did not cost or gain accuracy;
  generation is still the bottleneck.
  **Prompt v2** (one generic Gricean instruction: read what was not said;
  if the reply dodges, or gives far more or less detail than asked, one
  reading must say what that suggests; say plainly what a figure of speech
  means; `experiments/e4_product/gen_v2.diff`). Written from the odd-half
  failures, tested on the held-out even half (52 Hu + 12 sincere):
  | Measure (even half) | Baseline | Current | v2 |
  |---|---|---|---|
  | Top-1 (52 Hu) | 0.481 | 0.442 [0.308, 0.577] | **0.558** [0.423, 0.692] |
  | Recall | 0.712 | 0.692 | 0.769 |
  | Maxims top-1 (10) | 0.30 | 0.30 | 0.50 |
  | Sincere plain top (12) | 0.83 | 0.67 | 0.75 |
  | Sincere plain offered | 1.00 | 1.00 | 1.00 |
  v2 vs current, paired: top-1 10 vs 4 (sign test p = 0.18), recall 9 vs 5
  (p = 0.42). **Decision: adopt v2.** It moves every Hu measure the right
  way on held-out items, costs nothing on the sincere controls, and is
  generic rather than item-specific. The gain is not significant at this n,
  so it is a trend to confirm on new items, not a result to claim. Still
  well below the pre-registered 0.65-0.75; the next lever is the remaining
  misses (white-lie motives, metaphors glossed near-literally, speaker
  confusion in three-person stories).
  **Consequence for the app.** The "How good is it?" panel quoted E1
  numbers (the model choosing among the authors' written options: 78%).
  The product writes its own readings and gets the intended meaning first
  about half the time. The panel now quotes product-level E4 numbers.
- **Replies audit, noise floor (3 generation seeds).** The replies audit
  rerun with seeds 1 and 2 (same engine as seed 0; `experiments/e4_audit/
  noise.py`). Mean share over 3 seeds x 12 messages x 3 replies:
  | Disclosure | engage | check | hold back | apologise |
  |---|---|---|---|---|
  | none | 0.47 | 0.47 | 0.03 | 0.04 |
  | autistic | 0.49 | 0.43 | 0.05 | 0.04 |
  | anxious | 0.41 | 0.45 | 0.05 | 0.09 |
  Effect vs no disclosure, averaged over seeds, paired by message, bootstrap
  95% CI: autistic, hold back +0.019 [-0.013, +0.053], apologise +0.002
  [-0.022, +0.023]; anxious, hold back +0.017 [-0.007, +0.046], **apologise
  +0.054 [+0.019, +0.093]**. Seed-to-seed shift of a condition's mean with
  nothing changed: apologise 0.028 on average (max 0.093), hold back 0.039.
  **Reading.** No sign of the Wohn et al. avoidance drift for autism
  disclosure: the plan's design rule holds on this audit. Anxiety disclosure
  makes apologetic or over-explaining replies more likely by about 5 points
  (from 4% to 9%); the single seed-0 run overstated it (+0.09), which is why
  the repeats mattered. Small, but it is the kind of drift the rule is meant
  to stop, so it gets a fix and a re-test (below, when run).
- **Latency, Reflect v2 on the M3 (qwen3:8b, 20 E4 items, model warm,
  nothing else running; `app/scripts/latency.ts`, `experiments/e4_product/
  latency_v2.jsonl`).** Medians (p90): literal line on screen 6.4 s (7.2),
  first reading 9.1 s (10.6), all readings written 24.2 s (27.7), weighed
  and done 26.6 s (31.2). Weighing costs 3 s; writing is 90% of the time,
  and 15 s of it comes after the first reading is already visible (the
  later readings, their cues and the check question). The plan's "about a
  minute" was the pre-streaming, contended figure; the E4 runs measured 34-39 s
  with judge calls interleaved. So the user waits about 9 s for something to
  read and under half a minute for the finished, weighed list. Cheapest
  next cuts, untested: keep the fixed system prompt cached between calls
  (the weighing calls in between evict it), and write the check question
  only when asked.
- **Replies audit after the reply-prompt fix (engine as committed: v2 +
  "base the replies on what happened"; seeds 0-2, `experiments/e4_audit/
  replies_fix_seed*.json`).** Mean shares: none engage 0.47 / check 0.45 /
  hold back 0.04 / apologise 0.04; autistic 0.43 / 0.47 / 0.07 / 0.03;
  anxious 0.40 / 0.44 / 0.12 / 0.05. Effects vs none: anxious **apologise
  +0.011 [-0.038, +0.067]** (was +0.054 [+0.019, +0.093]), but anxious
  **hold back +0.080 [+0.018, +0.153]** (was +0.017); autistic hold back
  +0.026 [-0.028, +0.093], apologise -0.005. **Reading.** The fix removed
  the apologies but the drift moved into avoidance: hold-back replies under
  anxiety were 13 of 108 against 5 of 108 with no disclosure, mostly "no
  response needed" / "not replying right now" on messages that do ask for
  an answer. Apologise + hold back together went from +0.07 to +0.09, so
  the fix relabelled the effect rather than removing it. The source is the
  prompt's open "if not replying is reasonable, one can be that", which the
  model reads more generously once the user says they get anxious. Fix 2:
  tie that option to the message ("only if the message does not ask for an
  answer"), re-test on 3 seeds.
- **`gloss` merged into the playground repo** at Eddie's request (subtree
  merge, so the 38 gloss commits keep their history), on branch
  `es-vlm/setup`, and pushed to `eddiebullock/playground`. The nested `.git`
  is gone; commit from the playground repo.
- **Replies audit after fix 2 (engine as committed in b6f72b6; seeds 0-2,
  `experiments/e4_audit/replies_fix2_seed*.json`). Fix 2 kept.** Effects vs
  no disclosure, paired by message, bootstrap 95% CI: anxious hold back
  +0.026 [-0.010, +0.066], apologise +0.002 [-0.041, +0.034]; autistic hold
  back +0.032 [-0.026, +0.100], apologise -0.018 [-0.045, +0.006]. Noise
  floor 0.035 (max 0.098). Both pre-registered success conditions on the
  anxiety effects are met. Mean shares: none engage 0.48 / check 0.39 / hold
  back 0.08 / apologise 0.05; anxious 0.52 / 0.33 / 0.10 / 0.05. The third
  condition (no-disclosure "check" share should not fall by more than the
  noise floor) is borderline: 0.45 -> 0.39, a 0.056 fall against a mean
  floor of 0.035 and a max of 0.098. Per message, the extra no-disclosure
  hold-back replies sit on messages that ask nothing ("That's alright, we'll
  just have to cope" 0.11 -> 0.42; "Fine, no problem" 0.00 -> 0.10), where
  not replying is a legitimate option. Under fix 1 the same messages got
  hold-back replies mainly when the user was anxious (0.43 vs 0.11). So
  whether to hold back now follows the message, not the reader, which is the
  design rule. Summary for Paper 2: pre-fix, the drift appeared as apologies;
  fix 1 moved it into avoidance; fix 2 (tie the option to the message)
  removed it without a reader-dependent residue on 12 messages x 3 seeds.
  Next: confirm on the 48-message set when compute allows (about 3 h per
  seed).
- **Readings audit v2: 48 messages x 6 reader conditions incl. a placebo
  (qwen3:8b, shipped stance scorer; `experiments/e4_audit/v2_qwen3-8b_stance.json`,
  `analyze_v2.py`).** Hypothesis (written with the plan): autism no shift,
  anxiety towards reassuring, as in the 12-message audit; placebo none.
  | Reader says | TV vs none | top-1 flips | negativity vs none | negativity vs placebo |
  |---|---|---|---|---|
  | autistic | 0.119 [0.078, 0.170] | 5/48 | +0.008 [-0.043, +0.063] | **+0.081 [+0.039, +0.131]** |
  | anxious | 0.122 [0.083, 0.170] | 5/48 | **-0.091 [-0.143, -0.048]** | -0.017 [-0.059, +0.024] |
  | ADHD | 0.100 [0.067, 0.145] | 4/48 | +0.038 [-0.005, +0.087] | **+0.112 [+0.069, +0.163]** |
  | depressed | 0.153 [0.099, 0.220] | 9/48 | +0.070 [-0.005, +0.155] | **+0.144 [+0.078, +0.226]** |
  | left-handed (placebo) | 0.096 [0.066, 0.130] | 3/48 | **-0.074 [-0.110, -0.040]** | - |
  **The placebo prediction failed, and that is the main result.** Any sentence
  about the reader moves the readings (TV about 0.1) and softens them.
  Against that baseline the anxiety effect is no longer specific to anxiety
  (it matches the placebo), while autism, ADHD and depression push readings
  *more negative* than a neutral self-description would, by 0.08-0.14 on the
  negativity index (mild = 1, hostile = 2). Against no disclosure the two
  components cancel for autism, which is why the 12-message audit (and any
  audit without a placebo) reads it as "no drift". For Paper 2: placebo
  disclosures are necessary, and condition labels carry a negative
  connotation that a generic softening masks.
- **"Judge the words, not the reader" scorer line: not adopted.** Same audit,
  scorer with the added sentence (`v2_qwen3-8b_words.json`): TV autistic
  0.120, anxious 0.127, ADHD 0.115, depressed 0.166, placebo 0.111; net of
  placebo the negativity shifts are unchanged (autistic +0.067, ADHD +0.103,
  depressed +0.143). It does not reduce the drift and it moves the
  no-disclosure readings themselves (TV 0.076, 6/48 top-1 flips), with no
  evidence that this helps accuracy. A wording fix does not work here, as
  with the replies. **Next (structural, morning, product change):** keep
  what the user says about themselves out of the scoring call altogether. A
  separate, optional "about me" field used only to word replies, never to
  weigh readings, then re-run this audit (expect TV near 0 by construction
  for that field) and check what free-text disclosures in "what came
  before" still do.
- **Minimal pairs across three phenomena (64 pairs: 25 irony, 20 indirect
  requests, 19 deceits; pairs written by Claude, not yet human-validated;
  `experiments/e1_pragmatics/sdt.py`, `results/sdt_summary.json`).**
  Hypothesis (from the irony pairs): the literal tendency is a criterion
  effect everywhere, and the stance prompt moves the criterion, not d'.
  | Model / prompt | phenomenon | acc non-literal | acc sincere | d' | c (+ = literal) |
  |---|---|---|---|---|---|
  | 8B direct | Irony | 0.56 | 0.96 | 2.21 | **+0.96** |
  | 8B direct | Indirect | 0.85 | 0.75 | 1.60 | -0.17 |
  | 8B direct | Deceits | 0.74 | 0.74 | 1.20 | 0.00 |
  | 8B stance | Irony | 0.84 | 0.92 | 2.25 | +0.18 |
  | 8B stance | Indirect | 0.90 | 0.65 | 1.55 | -0.41 |
  | 8B stance | Deceits | 0.90 | **0.47** | 1.09 | **-0.61 [-1.21, -0.14]** |
  | 4B direct | All 64 | 0.53 | 0.83 | 1.18 | +0.40 |
  | 4B stance | All 64 | 0.69 | 0.83 | 1.47 | +0.25 |
  **Partly confirmed, and more interesting than predicted.** d' barely moves
  with the prompt in any phenomenon (8B all pairs 1.51 -> 1.65). But the
  literal bias is specific to irony: on indirect requests and deceits 8B is
  already unbiased when asked directly, and the stance prompt pushes it into a
  non-literal bias. On sincere deceit pairs (the person is telling the truth)
  stance-prompted 8B picks the deceptive or excusing motive on 53% of items.
  So a single "people often don't mean what they say" instruction fixes
  irony by moving a criterion that is, for other phenomena, already in the
  right place. For Paper 1, the headline becomes: criteria are
  phenomenon-specific, prompts move them globally, and sensitivity is the
  stable property. **Product implication (Gloss scorer uses the stance
  prompt):** risk of over-reading truthful statements as white lies or
  hints. The E4 sincere controls were irony-only, so this was never tested at
  product level. Morning: add deceit and indirect-request sincere versions to
  the product eval, and test a phenomenon-neutral stance wording on the odd
  half. Caveat: the subagent flagged deceit pairs 11, 16 and 19 as weaker
  (the non-literal motive is unnecessary rather than impossible); validate
  before claiming.
