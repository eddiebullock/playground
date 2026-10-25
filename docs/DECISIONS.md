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
