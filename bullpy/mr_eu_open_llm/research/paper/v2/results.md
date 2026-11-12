<!--
Draft v2 scaffold, 2026-11-08 (ROADMAP step 0.8). No A1 rotation file, A2 cell or A4 model output was read to write this.
Sections follow research/paper/v2/outline.md, "Results structure", in order. Subheadings are provisional.
Every stated number is copied from research/RESULTS_REGISTRY.md, and the row is named beside it as [REG: section, claim].
Anything not in the registry is a slot: [PENDING: output file, key]. A slot is filled only when its analysis is locked and its row is in the registry.
Decision rules are quoted or closely paraphrased from the frozen amendment named, with file:line.
Results carry topical subheadings, as the journal asks for Articles.
-->

# Results

## 1. Several published designs, and our earlier one, cannot rule out option order or prompt text

**Question.** Can the task we and others used tell face reading apart from option position and prompt text?

**Rule.** None. The audit is descriptive. The delivery rule is in methods.md, "Delivery audit", and the vision gate tolerances were fixed before any model ran (DECISIONS 2026-11-04, "Vision gates G2 to G4").

**Locked text.**

In the v1 prompt the acted label was the first option on 243 of 243 items [REG: "Target position in 6AFC prompt"]. So accuracy equals the share of first-option choices. There were 21 option sets for 21 labels, one per label [REG: "Option sets per target label"]. The options therefore identify the category without the image. Category alone gave a grouped correlation of .503 with human response entropy, using no activations [REG: "Category-only grouped T1 rho"].

The face did not always reach the model. On Gemma 4 the composite grid left a median of 1.08 tokens across the face, against 9.01 on a single face crop [REG: Input delivery, "Gemma 4 tokens across the face, median"]. The gate is 8. On Qwen3-VL the values were 11.54 and 13.52, and all four routes passed [REG: Input delivery, "Qwen3-VL"]. On Molmo2 the single face crop passed at 12.96, and the native video route (5.07) and the grid (2.01) failed [REG: Input delivery, "Molmo2"]. The grid keeps the first 8 frames, so 68 of 243 clips lost frames at 1 frame per second [REG: Input delivery, "Clips losing frames to the 8-cell grid"].

Weak labels. On three labels, face-only human observers chose the acted label little more often than chance: Kind .229 (9 items), Unfriendly .241 (10 items) and Jealous .257 (7 items), 26 items in all, against 1/6 [REG: "Rows added 2026-11-08", "Weak labels"]. The acted label is a poor ground truth for these items, so results are also read against the human response distribution (DECISIONS 2026-11-04, "Keep acted labels, add human-referenced scoring").

Vision gates. Gemma 4 passed the option-independence, replay-and-cut and token-swap gates, with a token-swap change of 4.87 nats and a flip rate of .675 at 100% [REG: Vision gates, "Gemma 4"]. Qwen3-VL passed all three at bfloat16 only, at 9.30 nats and .55 [REG: Vision gates, "Qwen3-VL"]. Molmo2 failed option independence at bfloat16 (worst relative L2 .0406 against .01) and failed token swap (2.28 nats, flip rate 0 against .25) [REG: Vision gates, "Molmo2"]. Gemma 4 and Qwen3-VL are therefore eligible for A2 and A4, and Molmo2 is not [REG: Vision gates, "Eligibility"].

**Literature check.** We coded eight recent studies of emotion or mental-state reading by vision-language models [1, 2, 8-13] for four design features, as far as our reading of each went; the level of reading is given per row in `research/reproduction/literature_table.csv`. Target position was controlled in one study and partly in one, fixed in two, not stated in one, and did not arise in three that used no option list. One option set served every item in three studies and partly in one; this was not stated in two and did not arise in two. None stated a check that the images reached the model. Five had no image-free, mismatched or swapped-face control, and three had a partial one [REG: "Rows added 2026-11-12", "Literature table"]. "Not stated" means not stated in what we read.

**Figure.** Figure 1, panels a to e (`research/paper/v2/figures/fig1.pdf`). Panel e is the eight-paper table, drawn from `literature_table.csv`.

## 2. Dependence on the face (RQ0)

**Question.** With option position rotated, how much accuracy disappears when the prompt stays the same and the face is swapped?

**Rule (A1).** "A model is *vision-dependent* if the one-sided actor-level sign-flip p (100,000 draws) has Holm p < 0.05 across the three models and the actor-bootstrap 95% CI (10,000 draws) of delta excludes 0 from above. No new mental-state claim is made for a model that fails, whatever its accuracy versus 1/6." (`research/PREREGISTRATION.md:137`; estimand at `:135`.)

**Rule (A3).** The same test on `single_face`, its own Holm family (`research/PREREG_A3_DRAFT.md:40`). Reading against A1 (`research/PREREG_A3_DRAFT.md:44-49`):

| A1 | A3 | Reading |
|---|---|---|
| Pass | Pass | Face use on both inputs |
| Fail | Pass | The grid limited A1; use `single_face` from here |
| Pass | Fail | Evidence from A1 needs inspection |
| Fail | Fail | No demonstrated face use on either route; no RQ1 to RQ5 claims for that model |

**Locked text (A3).** On a single face crop, all three models passed A3, on 243 of 243 items each [REG: A3 face input validity]. For Gemma 4, accuracy was .371 on the item's own face and .156 on a mismatched face, a delta of .216 (95% CI .139 to .292, Holm *P* = .00014). For Qwen3-VL the values were .282 and .038, delta .244 (.212 to .276, Holm *P* = .00009). For Molmo2 they were .072 and .010, delta .063 (.040 to .087, Holm *P* = .00010) [REG: A3 table, rows Gemma 4, Qwen3-VL, Molmo2]. A pass tests dependence on the image, not competence. Molmo2 chose "None of the above" on .863 of real cells and .886 of mismatch cells [REG: A3, "Trace (2026-11-08)"]. Its accuracy was .140 with the target in the first position and .025 in the second [REG: A3, "Trace (2026-11-08)", accuracy by target position], so its delta rests on a small share of cells. For Gemma 4 the "None of the above" shares were .206 and .305, and for Qwen3-VL .534 and .688 [REG: A3, "Trace (2026-11-08)"]. Spearman correlations of model with human target probability were .376, .392 and .342 [REG: A3 table, column "Spearman with human p_target"]. These are secondary.

**Locked text (A1, read 9 November 2026).** With option position rotated over five runs and the prompt held fixed, all three models passed A1 [REG: A1 rotated 6AFC]. Qwen3-VL answered correctly on .469 of real cells and .067 when the face came from a mismatched clip, a delta of .402 (95% CI .350 to .461, Holm *P* = .00009). Gemma 4 gave .402 and .174, delta .228 (.172 to .283, Holm *P* = .00009). Molmo2 gave .163 and .092, delta .071 (.049 to .094, Holm *P* = .00009) [REG: A1 table]. Holm *P* is at the floor of 100,000 sign-flip draws for all three. With A3 also passed, every model is in the first row of the table above: face use on both inputs. A pass tests dependence on the image, not competence. Molmo2's accuracy on the item's own face was at chance (.163 against 1/6; pooled *P* = .65, descriptive), and its pass comes from mismatched faces pulling accuracy below chance. Accuracy by target position on real cells ranged from .444 to .510 for Qwen3-VL, .354 to .444 for Gemma 4 and .132 to .198 for Molmo2, so no model favoured the first slot once options were rotated [REG: A1, accuracy by target position]. Spearman correlations with human target probability were .409, .444 and .380, and within-category correlations .288, .138 and .233 (secondary). One Qwen3-VL trial of 7,290 failed to load its clip on the cluster; it was scored as wrong, under a rule fixed before any A1 file was read [REG: A1, errored trials].

**Writing rule.** A Pass is never written as "reads faces" (outline.md, RQ0). If A1 is incomplete, nothing is stated, because Holm covers all three models (outline.md, R0-d).

**Figure.** Figure 2 (`research/paper/v2/figures/fig2.pdf`), drawn from the locked `rotation_summary.json` and `face_input_validity_summary.json`: delta per model and input with its actor bootstrap CI (a), accuracy on real and mismatched faces against 1/6 (b), and real-face accuracy by target position on A1 (c) and A3 (d).

## 3. Behaviour cannot settle cue use

**Question.** Can agreement on natural items identify which cues a system uses?

**Rule.** None. These are mathematical results and design facts, with proofs in Supplementary Note 1 and in `research/theory/causal_feature_alignment.md`. They use no model output.

**Locked text.** If two systems give the same responses on every natural item, every behavioural metric on those items is equal, yet their cue effects can take any pattern (Theorem 1, Methods). For linear observers, how far behaviour constrains cue effects depends on the smallest eigenvalue of the natural cue correlation (Proposition 3, Methods).

The natural clips are poorly conditioned. On the 243 EU-Emotion clips, the smallest eigenvalue of the correlation of 15 LibreFace cues was .054, with a condition number of 76.1 and 1 / lambda_min of 18.4 [REG: Cue spectra, "EU-Emotion LibreFace"]. For 10 OpenFace 3.0 cues the values were .125, 22.4 and 8.0 [REG: Cue spectra, "EU-Emotion OpenFace 3.0"]. The randomised Snoek designs, with 33 cues as coded, had smallest eigenvalues of .178 to .209 and condition numbers of 11.0 to 13.7 [REG: Cue spectra, "Snoek as coded, 33 cues"]. With left and right merged into 25 cues, the smallest eigenvalue was .44 to .47 and the condition number 3.4 to 3.5 [REG: Cue spectra, "Snoek left and right merged, 25 cues"]. So most of the Snoek correlation comes from bilateral pairs. Cue counts differ between rows.

On synthetic stimuli with paired cues correlated at a constructed .9, pair-swapped observers' outputs correlated .83 to .95 with the human-weighted observer, while their raw CEA on the randomised design was .42 to .56 [REG: A4 recovery, "Pair-swapped, run 2"]. The .9 is ours, not a measurement.

**Figure.** Figure 3, panels a to e (`research/paper/v2/figures/fig3.pdf`). Panel b shows the Proposition 2 check: for two observers that weight different but correlated cues, the correlation of their outputs equals the cue correlation (.5, .9 and .99), while the cosine between their cue effects stays 0 [REG: "Rows added 2026-11-12", "Proposition 2 check"].

## 4. Pattern alignment on natural clips (RQ1)

**Question.** Does each model's response pattern on natural clips align with the human pattern, over the same measured cues?

**Rule.** The confirmatory unit is the family pattern score on the model-frames route, at human ridge 10, with the attempt 1 null of record (DECISIONS 2026-11-04, "RQ1 inference after two failed calibration attempts", `research/DECISIONS.md:232`). It is confirmatory only if size and power pass at both 0.75 and 1 times the human effect (`research/DECISIONS.md:260`), which they did (`research/DECISIONS.md:270`). If more than 5% of clips are lost, the result is descriptive (`research/DECISIONS.md:264`). A clip counts only with a usable response on all 5 rotations (`research/DECISIONS.md:282`). The reading depends on RQ0 (`research/DECISIONS.md:327-332`):

| A1 | A3 | Natural-clip RQ1 |
|---|---|---|
| Pass | Pass | Confirmatory, subject to the 5 percent clip rule |
| Fail | Pass | Computed and reported as descriptive |
| Pass | Fail | Computed and reported as descriptive, pending inspection |
| Fail | Fail | Computed for the record, with no claim |

A positive score supports "aligned with the human pattern", not "uses the same cues" (`research/DECISIONS.md:234`).

**Locked text (recovery).** On this route, power was .995 at 0.75 times and 1.0 at 1 times the human effect, against a limit of .80. Size was .035 for the prior-only observer and .06 for the actor-prior observer at 0.75 times, against .081 [REG: "Pattern score, model-frames route (2026-11-04)"]. Power was .775 at 0.5 times. For an observer sharing half the human weights, power was .60 at 0.75 times [REG: same section]. The different-cues observer had size .055 at 0.75 times [REG: same section]. At 1 times it was .83 [REG: "Rows added 2026-11-12", "Half-shared pattern power"].

**Locked text (models, read 9 November 2026).** All three models are in the Pass and Pass row, and none lost more than 5% of clips, so the score is confirmatory for each [REG: Natural-clip RQ1]. On the composite input, the family pattern score was .549 for Qwen3-VL (permutation *P* = .0002, 242 of 243 clips), .418 for Gemma 4 (*P* = .0006, 243 clips) and .543 for Molmo2 (*P* = .0002, 243 clips) [REG: Natural-clip RQ1 table]. *P* = .0002 is the floor at 4,999 permutations. On each detector alone, a sensitivity, the scores were .548 and .520, .411 and .350, and .522 and .469. Qwen3-VL lost one clip, the one whose trial failed to load. Each model's responses on these clips are therefore aligned with the human pattern of AU families. This does not show that the models use the same cues. The score is a regression on measured cues, and a system responding to an unmeasured cue correlated with them can produce the same slopes (Proposition 4), so the claim stays at alignment with the human pattern, as fixed in advance. Molmo2's alignment sits beside chance-level accuracy in A1, which shows that the pattern score and accuracy measure different things. Single and family-wise cells are exploratory.

**Figure.** Figure 4 (`research/paper/v2/figures/fig4.pdf`), drawn from the three locked `same_clip_family_model_*.json` files and the route recovery files: the composite pattern score per model with each detector alone, against its permutation null and the recovery range of human-like and prior-only observers.

## 5. Cue effects under randomisation (RQ1, A4)

**Question.** Do a model's effects of randomised facial action units, on rendered faces, align with randomised human effects?

**Rule (A4, residual CEA, per model and dataset).** (`research/PREREG_A4_DRAFT.md:121-125`.)

| Outcome | Requires |
|---|---|
| Aligned | All gates; signal p < .05; alignment Holm p < .05; CEA at least .30; CI lower bound above 0 |
| Not aligned | All gates; signal p < .05; CI upper bound divided by the square root of R_S below .30 |
| Inconclusive | Anything else, including any failed or missing gate |

Not aligned is always written "not aligned on these renders" (`:127`). A model with no signal cannot be Not aligned (`:129`). Headline per model per task: Aligned or Not aligned only if WC and EA agree, Mixed if one of each, Inconclusive otherwise; basic and conversational are never pooled (`:137`). Human-level only if the interval reaches the WC to EA reference, which is descriptive (`:135`). Validity gates V1 to V4 run before any human comparison, and missing means not passed (`:141-148`). Eligibility runs through the A1 and A3 table, as in A2 (methods.md, "Eligibility").

**Locked text (design and recovery).** The human ceilings, raw, were .990, .989, .978 and .974 on basic WC, basic EA, conversational WC and conversational EA, and residual .970, .983, .974 and .960 [REG: A4 human ceiling, "Ceiling, raw" and "Ceiling, residual"]. The WC to EA reference on residual CEA was .780 for basic and .732 for conversational, and on raw CEA .877 and .767 [REG: "WC to EA reference, residual", "WC to EA reference, raw"]. The common profile held .26 to .35 of human squared effects [REG: "Common-profile share"].

Residual CEA passed recovery on 66 of 66 checks and raw CEA on 62 of 66, so residual CEA is confirmatory [REG: A4 recovery, "Run 4 (frozen)"]. In run 3, a system with no cue-specific agreement reached Aligned in up to .40 of replicates on raw CEA, against .03 on residual CEA [REG: "Run 3 rates, raw / residual"]. Two accuracy cells at conversational 0.25 times are exempt, at .85 and .62, so A4 makes no accuracy claim there [REG: "Exempt cells, residual pair-swapped within-limit share"].

The detector check passed thinly: 113 of 144 covered pairs rose, .785 against .70 [REG: A4 V1 pilot, "V1 share rising"]. Only 22 of 96 intensity pairs gained at least .5, all in AU01, 05, 25 and 26 [REG: "Intensity pairs with reading at amplitude 1 minus .25 of at least .5"]. Those four families hold .14, .13, .09 and .20 of residual human energy [REG: "Detector-confirmed share of residual energy"].

**Slots.** From `results/a4/summary.json` (written by `scripts/a4_analyse.py`), per model and dataset (`datasets.<dataset>.<model>`):
- residual CEA, CI, R_S: [PENDING: `residual.cfa`, `residual.ci`, `residual.r_s`]
- alignment Holm *P*: [PENDING: `p_holm`]
- signal *P*: [PENDING: `p_signal`]
- gates V1 to V4 and RQ0: [PENDING: `gates`]
- decision: [PENDING: `decision`]
- ratio to ceiling, descriptive: [PENDING: `residual.over_ceiling`, `residual.lower_bound_only`]
- raw CEA, secondary: [PENDING: `raw.cfa`, `raw.ci`]
- sensitivities (without AU22 and AU27, without abstain row, flagged against confirmed families, amplitude sets, direct contrast): [PENDING: sensitivity blocks and `models.<model>.scale`, `direct_cfa_residual`]
- headline per model per task: [PENDING: `models.<model>.headline`]
- blind readout, secondary, only if its gate passes: [PENDING: `models.<model>.blind`]
- Molmo2, descriptive if run at all: [PENDING]

**Figure.** Figure 5: residual CEA per model and dataset with its between-group reference, raw CEA as secondary. [PENDING: Figure 5 function (ROADMAP 0.7).]

## 6. Do behaviour and cue effects disagree?

**Question.** On the same rendered stimuli, does behavioural similarity say something different from CEA?

**Rule.** Descriptive, no decision (`research/PREREG_A4_DRAFT.md:162-169`). Output similarity is Pearson *r* between model and mean human response, option means removed. Modal agreement is reported with chance-corrected kappa. "Whether these disagree with CEA is an empirical result" (`:169`).

**Slots.** From `results/a4/summary.json`, `datasets.<dataset>.<model>.behaviour`:
- output similarity: [PENDING: `behaviour.output_r`]
- modal agreement and kappa: [PENDING: keys as written by `behaviour_metrics`]

**Writing rule.** Agreement is not a failure of the theory, nor proof of use (outline.md, section 6).

**Figure.** Part of Figure 5.

## 6b. The randomised cue test on a second human sample (A9)

<!-- Added 2026-11-09. A9 was a DRAFT when this was written (research/PREREG_A9_DRAFT.md:3); line numbers are as of that text and must be rechecked against the frozen file. No A9 model output exists. No result is stated. -->

**Question.** Does A4's answer hold against a second set of randomised human observers, Liu et al.'s Experiment II?

**Status.** [PENDING: A9 freeze date and tag `a9-freeze`.] Until A9 is frozen, nothing in this section is preregistered.

**Relation to A4.** A4's verdicts are final as frozen. A9 never enters A4's Holm family, headline or decision table, and is reported here, after A4. A9 and A4 estimates are never pooled. If they disagree for a model, both stand. An A9 Aligned or Not aligned does not make an A4 Inconclusive conclusive (`research/PREREG_A9_DRAFT.md:423-429`).

**Rule (A9, residual CEA, per model, one dataset).** (`research/PREREG_A9_DRAFT.md:553-561`.)

| Outcome | Requires |
|---|---|
| Aligned | All gates; signal p < .05; alignment Holm p < .05 within A9 (Gemma 4 and Qwen3-VL); residual CEA at least .30; CI lower bound above 0 |
| Not aligned | All gates; signal p < .05; CI upper bound divided by the square root of R_S below .30 |
| Inconclusive | Anything else, including any failed or missing gate, or no recovery-selected design |

Not aligned is written "not aligned on these renders". With one dataset there is no WC and EA headline, so the per-model decision is A9's result (`:561`). Human-level only if the interval reaches the residual Liu against Snoek basic WC reference, which is descriptive (`PREREG_A9_DRAFT.md:522`; value in `research/reproduction/a9_human_ceiling.json`, `human_reference_cfa.basic_WC.residual`, no registry row yet: [PENDING: registry row]). Gates: RQ0 at `single_face`, G2 to G4, V1 inherited from A4's pilot, V2 to V4 recomputed on Liu renders (`PREREG_A9_DRAFT.md`, Validity gate). Design B, by recovery. Decisions stay confirmatory whatever a model's correlation across faces. The accuracy claim for residual CEA on weak effects (0.25 times) is withdrawn under correlated face noise (`:442`; block `withdrawn_under_correlated_face_noise`).

**Slots.** From `results/a9/summary.json` (written by `scripts/a9_analyse.py`), per model (`models.<model>`):
- residual CEA, CI, R_S: [PENDING: `residual.cfa`, `residual.ci`, `residual.r_s`]
- alignment Holm *P* within A9, signal *P*: [PENDING: keys as written by `a4_analyse.apply_rule`]
- gates (RQ0, G2 to G4, V1 to V4): [PENDING: gates block]
- decision: [PENDING: `decision`]
- configurations kept and share lost under design B: [PENDING: `n_configurations`, `share_lost`, `lost_ok`]
- residual correlation across faces, descriptive, with no label on the decision: [PENDING: `face_shared_residual_correlation`]
- the reading the replaced first operationalisation of lead decision 2 would have given, descriptive, reported beside the decision so readers have both: [PENDING: `earlier_rule_reading_descriptive`]
- replication reading against the same model's A4 basic WC decision, with A4's basic headline beside it: [PENDING: `replication.reading`, `replication.reason`, `replication.a4_basic_WC_decision`, `replication.a4_basic_headline_beside`]
- ratio to ceiling, descriptive: [PENDING: `residual.over_ceiling`, `residual.lower_bound_only`]
- raw CEA, secondary: [PENDING: `raw.cfa`, `raw.ci`]
- sensitivities (presence coding, without AU22 and AU27, without the abstain row, flagged against confirmed families, `frame_max`): [PENDING: `sensitivities`]
- behaviour beside CEA, per configuration: [PENDING: `behaviour`]
- Molmo2, descriptive if run: [PENDING]
- side-by-side with A4, labelled as separate tests, descriptive only: [PENDING]

**Replication wording, fixed before data** (`PREREG_A9_DRAFT.md`, Relation to A4; block `replication`). Comparator: the same model's A4 decision on basic WC. A4's basic headline is reported beside it, never used as the comparator.

| A4 basic WC | A9 | Words used |
|---|---|---|
| Aligned | Aligned | "replicated on a second human sample (same lab and generator)" |
| Not aligned on these renders | Not aligned on these renders | "the Not aligned result replicated" |
| Aligned | Not aligned | "did not replicate"; report both, no pooled claim |
| Not aligned | Aligned | "did not replicate"; report both, no pooled claim |
| Inconclusive, or not run for that model | any | "replication not testable", naming the gate or test |
| any | Inconclusive | "replication not testable", naming the gate or test |

A4's per-dataset decision cannot be Mixed, so no Mixed row arises. An A9 Inconclusive is never a failure to replicate. A9 is never called independent of the GFG platform.

**Withdrawn claim.** Under face-correlated noise the claim that residual CEA is estimated accurately on weak effects (0.25 times human strength) is withdrawn. This was decided after recovery run 2 was seen, replacing an earlier operationalisation that would have labelled every model outside the validated conditions (`PREREG_A9_DRAFT.md`, lead decisions 2 and 8).

**Writing rule.** Never "A9 confirms A4" or "A9 overturns A4". Say A9 is a second sample on the same platform (lab, generator, protocol, culture) and the same renders, not a second method (`PREREG_A9_DRAFT.md:421`).

## 7. Interventions on the stimulus (RQ2, A2)

**Question.** Do region masks, spatial-frequency filters and changes of frame order move a model's judgement in the direction of a stated human prediction?

**Rule (A2, per prediction).** (`research/PREREG_A2_DRAFT.md:188-193`.)

| Outcome | Requires |
|---|---|
| Matches prediction | RQ0, G2 to G4, Gate B, route delivery and the sham all passed; step-down p < .05 in the predicted direction on the blind readout; fill rule met |
| Opposes prediction | The same gates passed; step-down p < .05 two-sided in the opposite direction |
| Null | The same gates passed; neither test significant; all null conditions met |
| Inconclusive | Anything else, including a sham or delivery failure |

Gate B is B1 to B5 (`research/PREREG_A2_DRAFT.md:109-117`). A model and route that fail it go no further on the blind readout. Null needs all conditions at `:201-206`. P5, P6, P9, P10, P14 and P15 cannot reach Null, and on Qwen3-VL P12 and P13 cannot either (`:199`). Every result carries the BASIC ONLY scope (`:195`). After the freeze, two dated additions govern how results are read: source support for P1 to P15 was corrected before any data (`:332-353`), and each Matches or Opposes verdict is also reported against step-down *P* < .01, fixed before any cell (DECISIONS 2026-11-08, `research/DECISIONS.md:398`).

**Locked text (recovery).** All five synthetic observers passed recovery, 100 replicates each, with expected contrasts recovered at 1.0 [REG: A2 recovery]. Recovery used large effects and independent noise, so it shows that the pipeline works, not its power. Under a skewed zero-mean null with its tail opposite the prediction, the frozen test is liberal, and the .01 reading holds familywise error under .05 (DECISIONS 2026-11-08). [PENDING: per-route familywise error at .05 and at .01, from `research/reproduction/a2_gain_skew_sim.json`. The values are in DECISIONS and methods.md but have no registry row, so none is stated here.]

**Slots.** Per eligible model and route, from the A2 analysis output (`scripts/analyse_stimulus_interventions.py --output <path>`; path set at run time: [PENDING: path]):
- Gate B, B1 to B5: [PENDING: gate block, `pass`]
- per prediction P1 to P15 (16 contrasts): effect, actor-bootstrap CI, step-down *P*, verdict: [PENDING: rows, `verdict`]
- the .01 reading for each Matches or Opposes verdict: [PENDING]
- equivalence intervals for capped contrasts: [PENDING]
- shuffle against `static_apex`, descriptive: [PENDING]
- sensitivities (without smoke items, canonical words, clips at 8 tokens or more, human modal label): [PENDING]

**Figure.** Figure 6: effect and decision class per prediction, by model and route. [PENDING: Figure 6 function (ROADMAP 0.7).]

## 8. What each model has shown

**Question.** Across RQ0, RQ1 and RQ2, which degree of similarity does each model reach?

**Rule.** None of its own. Each cell copies the class from Sections 2, 4, 5 and 7. Models are never pooled (methods.md, "Models").

**Locked cells.** A1 Pass and A3 Pass for all three models, so face use on both inputs (Section 2). Natural clips: aligned with the human pattern, confirmatory, for all three (Section 4). Molmo2 is not eligible for A2 or A4, on its gate failures (Section 1), so its row ends there.

**Slots.** Gemma 4 and Qwen3-VL: [PENDING: A4 headline per task], [PENDING: A2 classes per route].

**Figure.** Table 1.

## 8b. Larger models in the same families (A8)

<!-- Added 2026-11-09. A8 is FROZEN, tag a8-freeze (research/PREREG_A8_DRAFT.md:3). No A8 output exists. No result is stated. -->

**Question.** Do the frozen tests give the same answers on Qwen3-VL-32B and Gemma 4 31B?

**Rule.** Every step is judged by its frozen rule, with no new threshold (`research/PREREG_A8_DRAFT.md:678-690`). A1 and A3 as in Section 2, with A8's own Holm family; the RQ0 reading as in the A1 and A3 table. A model that fails G2 to G4, or whose A3 result is not Pass, does not run A1 and is reported "A1 not run, by the A8 stop rule" (`:690`). A2 and A4 by their frozen rules, run only on eligible models. Natural-clip RQ1 is descriptive only. Families are A8 models only (`:694-701`); with one model Holm is the identity, and a smaller family is less strict per model than the frozen three-model family, which the text must say. No pooling with the frozen models, with each other or with the small siblings (`:703`).

**Reporting rule.** A8 rows are labelled A8 and placed under the frozen rows, whatever they show. Where an A8 model and its sibling disagree, both verdicts are reported and the difference is described, not tested (`:705`). A8 cannot change any frozen verdict: a frozen Inconclusive stays Inconclusive, a frozen null stays a null, and Molmo2 stays ineligible (`:707`). A8 may support a statement of scope only.

**Pre-data steps.** [PENDING: date of the A8 code pins; GPU A1 equivalence result (frozen command twice, 2-shard merge, record for record), from the dated pre-data note in `PREREG_A8_DRAFT.md`.]

**Slots,** per A8 model (`qwen3vl_32b`, `gemma4_31b`):
- vision gates G2 to G4, and whether bfloat16 G2 or G3 was non-zero: [PENDING: `research/reproduction/a8/<a8_id>/vision_gates_<key>.json`]
- input delivery: [PENDING: `research/reproduction/a8/<a8_id>/input_delivery_<key>.json`]
- readout ladder, descriptive: [PENDING: `research/reproduction/a8/<a8_id>/readout_ladder_<key>.json`]
- A3, the same fields as Section 2: [PENDING: `research/reproduction/a8/face_input_validity_summary_a8.json`]
- A1, the same fields as Section 2, or "A1 not run, by the A8 stop rule": [PENDING: `research/reproduction/a8/rotation_summary_a8.json`; errored trials from `rotation_errored_trials_a8.json`]
- natural-clip pattern score, descriptive: [PENDING: `research/reproduction/a8/<a8_id>/same_clip_family_model_<a8_id>.json`]
- Gate B and A2 verdicts, with the .01 reading: [PENDING: A2 output of `research/scripts/a8_analyse_a2.py`, path set at run time]
- A4 per dataset and headline per task: [PENDING: `results/a8/a4/summary.json`]
- steps not run, and why (stop rule, not tested after retry, storage, deferral of A4 by a dated note): [PENDING: launcher records under `research/reproduction/a8/<a8_id>/launch/`]

**Figure.** A8 rows join Figures 2, 5 and 6 and Table 1 below the frozen rows, marked A8. [PENDING: figure functions.]

## 9. The route through the model (RQ3), a scoped limit

**Question.** Is a cue's effect carried from image tokens through the language model?

**Rule.** Not a test (DECISIONS 2026-11-07, "G7 becomes a scoped limit", `research/DECISIONS.md:355-363`). A7 was frozen on 2026-11-08, tag `a7-freeze`, before any full A2 cell was read (`research/PREREG_A7_DRAFT.md:3`). Its reporting table is `research/PREREG_A7_DRAFT.md:391-398`. No site is called a cut unless Stage 0 shows it complete.

**Locked text (Stage 0, 10 November 2026).** On the 24 calibration clips, on `single_face` with the blind readout, the exit cut, which removes the whole vision path, was complete on both models: in every check the worst change from the image-free reference was 0.0 nats, against a limit of 0.001 [REG: "A7 Stage 0, Gemma 4"; "A7 Stage 0, Qwen3-VL"]. Its two blocks were both live (ratios .230 and .885 on Gemma 4, .802 and .273 on Qwen3-VL, against a floor of .10) and together exhaust it, but a block is not a cut. No site inside the language model was shown complete, so RQ3 is not testable beyond the interface on either model. The region test, which asks whether the tokens at a cue's location carry its effect, failed on both. On Gemma 4 the effect leaked past those tokens (leak .139 and .088 at buffers 0 and 1). On Qwen3-VL the token map located the cue on .760 of 96 trials, below its rule, so no leak was computed. Cue-level mediation is therefore not testable by a token cut on either model. The route arm may run in Stage 1 only as a secondary split of the complete exit cut, and is never reported as an RQ3 result (DECISIONS 2026-11-10).

**Figure.** None in the main text. Stage 0 table in the Supplement, from `research/reproduction/a7_stage0_gemma4.json` and `a7_stage0_qwen3vl.json`.

## 10. RQ4, and RQ5 as future work

**RQ4 (A5).** A5 is frozen, tag `a5-freeze`, on 2026-11-08, before any A2 cell existed (`research/PREREG_A5_DRAFT.md:3`). Only the evidence classes are confirmatory (`:246`, `:576`):

| Class | Requires |
|---|---|
| Shown | E* passes (step-down p < .05, 95% CI lies above 0), and Gate B passed |
| Unmapped | E* passes, and any of B3, B4 or B5 failed |
| Absent | E* not significant, its 90% CI inside the margin, B4 failed, and the positive control holds |
| Unresolved | Anything else |

(`research/PREREG_A5_DRAFT.md:226-231`.) Balance classes and verdict names are descriptive, not error-controlled under skewed portrayal gain (`:574`).

Slots, per model and route, from the A5 output (`scripts/a5_analyse.py --output <path>`): [PENDING: `confirmatory` block, evidence class], [PENDING: `descriptive` block, balance, Phi with 90% CI, word share, `family_gain_log_sd`].

**RQ5 (A6), closed as a limit.** A6 was never frozen. Its own pre-freeze rule, fixed before any model output, required the first frame of each clip to carry at most half of the apex value of AU9 and AU10 on every detector. ME-GraphAU's AU10 gave .801 [REG: "A6 pre-freeze cluster checks"], so A6 did not freeze or run (DECISIONS 2026-11-10), and RQ5 is named as future work in the Discussion. No CREMA-D model output exists. The human target sample, 819 clips and 91 actors [REG: A6 human target, "Sample"], is a design fact.

**Figure.** The free display slot (outline.md, "Display items"), only if A5 appears.
