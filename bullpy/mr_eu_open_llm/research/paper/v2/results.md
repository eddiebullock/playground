<!--
Draft v2 scaffold, 2026-11-08 (ROADMAP step 0.8). No A1 rotation file, A2 cell or A4 model output was read to write this.
Every stated number is copied from research/RESULTS_REGISTRY.md, and the row is named beside it as [REG: section, claim].
Anything not in the registry is a slot: [PENDING: output file, key]. A slot is filled only when its analysis is locked and its row is in the registry.
Decision rules are quoted or closely paraphrased from the frozen amendment named, with file:line.
Results carry topical subheadings, as the journal asks for Articles.

2026-11-21, reframe and length cut 7 (the author's instruction of 2026-11-21; research/paper/v2/IMPACT_PLAN.md; DECISIONS 2026-11-21):
- Sections are reordered, merged and retitled as findings: old 1 and 2 keep their place; old 3 and 4 are now 3; old 5 and 6b are now 4; old 6 is 5; old 7 is 6; old 10's A5 part is 7; old 8b is 8; old 9 and old 10's RQ5 part are 9. Old 8 ("What each model has shown") restated the classes of the other sections and is replaced by a pointer to Table 1, which holds every cell.
- The drafting scaffold of each section (Question, Rule, Writing rule, Figure notes, the replication table) is kept in comments beside it, unchanged. The rules are also in Methods.
- Amendment labels (A1 to A9, RQ0 to RQ5, G, V, B and P numbers) are replaced in the visible text by plain names. Methods keeps the labels and their meaning.
- No class, number, threshold or disclosure changed, and every value keeps its registry marker. Text that left this file moved word for word to Supplementary Methods, in the section named in a comment where it stood.
- Corrections, each more conservative: Section 5 now states that the behaviour comparison has no noise ceiling and that per-stimulus human responses are less reliable than cue effects, so the gap may reflect reliability; and it drops "The identifiability result allows this", because on a randomised design Proposition 3 says behaviour does constrain cue effects for linear observers. Section 2 now says Molmo2 was at chance on the composite input and below chance on the single crop, where the old text said only "at chance". Section 4 now says the models saw the same AU configurations, not the stimuli people saw, as Methods states.
- The full text before this rewrite is in the git history at bb3f90b.
-->

# Results

## 1. Accuracy alone cannot show that a model used the face

<!-- Scaffold, kept from before 2026-11-21.
**Question.** Can the task we and others used tell face reading apart from option position and prompt text?
**Rule.** None. The audit is descriptive. The delivery rule is in methods.md, "Delivery audit", and the vision gate tolerances were fixed before any model ran (DECISIONS 2026-11-04, "Vision gates G2 to G4").
**Figure.** Figure 1, panels a to e (`research/paper/v2/figures/fig1.pdf`). Panel e is the eight-paper table, drawn from `literature_table.csv`.
-->

<!-- 2026-11-13, length plan cut 3 (outline.md, "Length check"): the full coding of the literature check moved word for word to Supplementary Methods, Section S2, "Literature check". 2026-11-21, length cut 7: the delivery values for Qwen3-VL and Molmo2, the frames lost to the grid, the per-label weak-label values and the vision-gate values moved word for word to Supplementary Methods, Section S2, "Results, Section 1, in full". -->

A multiple-choice score measures face reading only if the answer cannot be found without the face. Our earlier design failed on two counts. The acted label was the first option on 243 of 243 items [REG: "Target position in 6AFC prompt"], so accuracy equals the share of first-option choices. And there were 21 option sets for 21 labels, one per label [REG: "Option sets per target label"], so the options identify the category without the image: category alone gave a grouped correlation of .503 with human response entropy, using no activations [REG: "Category-only grouped T1 rho"]. We withdrew that design (Methods). We then coded eight recent studies of emotion or mental-state reading by vision-language models [1, 2, 8-13] for four design features, as far as our reading of each went (Fig. 1e; Supplementary Methods, Section S2). Target position was controlled in one and partly in one. None estimated the accuracy lost when the face is swapped under a fixed prompt: five had no image-free, mismatched or swapped-face control, and three had a partial one [REG: "Rows added 2026-11-12", "Literature table"].

Two more conditions must hold before accuracy can mean face reading. The face has to reach the model: on Gemma 4, the composite frame grid of our earlier design left a median of 1.08 tokens across the face, against 9.01 on a single face crop and a gate of 8 (Fig. 1c) [REG: Input delivery, "Gemma 4 tokens across the face, median"]. And the target has to be fair: on items acted as Kind, Unfriendly or Jealous, face-only human observers chose the acted label little more often than chance (.229 to .257, against 1/6; Fig. 1d) [REG: "Rows added 2026-11-08", "Weak labels"], so results are also read against the human response distribution. Three vision gates checked that each model's output depends on the image it is given. Gemma 4 and Qwen3-VL passed them; Molmo2 failed two, so it was not eligible for the cue-level tests below (Supplementary Methods, Section S2) [REG: Vision gates, "Eligibility"].

## 2. Answers depend on the face once option order is controlled

<!-- Scaffold, kept from before 2026-11-21.
**Question.** With option position rotated, how much accuracy disappears when the prompt stays the same and the face is swapped?
**Rule (A1).** "A model is *vision-dependent* if the one-sided actor-level sign-flip p (100,000 draws) has Holm p < 0.05 across the three models and the actor-bootstrap 95% CI (10,000 draws) of delta excludes 0 from above. No new mental-state claim is made for a model that fails, whatever its accuracy versus 1/6." (`research/PREREGISTRATION.md:137`; estimand at `:135`.)
**Rule (A3).** The same test on `single_face`, its own Holm family (`research/PREREG_A3_DRAFT.md:40`). Reading against A1 (`research/PREREG_A3_DRAFT.md:44-49`):

| A1 | A3 | Reading |
|---|---|---|
| Pass | Pass | Face use on both inputs |
| Fail | Pass | The grid limited A1; use `single_face` from here |
| Pass | Fail | Evidence from A1 needs inspection |
| Fail | Fail | No demonstrated face use on either route; no RQ1 to RQ5 claims for that model |

**Writing rule.** A Pass is never written as "reads faces" (outline.md, RQ0). If A1 is incomplete, nothing is stated, because Holm covers all three models (outline.md, R0-d).
**Figure.** Figure 2 (`research/paper/v2/figures/fig2.pdf`), drawn from the locked `rotation_summary.json` and `face_input_validity_summary.json`: delta per model and input with its actor bootstrap CI (a), accuracy on real and mismatched faces against 1/6 (b), and real-face accuracy by target position on A1 (c) and A3 (d).
-->

<!-- 2026-11-21, length cut 7: the single-face values beyond each delta, the answer shares of "None of the above", accuracy by target position, the correlations with human target probability, the token-floor sensitivity and the errored trial moved word for word to Supplementary Methods, Section S3, "Results, Section 2, in full". -->

We then measured face dependence directly. Over five runs the acted label was rotated through every position, the prompt was held fixed, and the face was the item's own or one from another clip, on the composite input and on a single face crop (Methods). By the preregistered rule, answers depend on the face if accuracy falls reliably when the face is swapped.

All three models met the rule on both inputs (Fig. 2) [REG: A1 rotated 6AFC; A3 face input validity]. On the composite input, Qwen3-VL answered correctly on .469 of real cells and .067 when the face came from a mismatched clip, a delta of .402 (95% CI .350 to .461, Holm *P* = .00009). Gemma 4 gave .402 and .174, delta .228 (.172 to .283, Holm *P* = .00009). Molmo2 gave .163 and .092, delta .071 (.049 to .094, Holm *P* = .00009) [REG: A1 table]. On a single face crop the deltas were .244 for Qwen3-VL, .216 for Gemma 4 and .063 for Molmo2 [REG: A3 table, rows Gemma 4, Qwen3-VL, Molmo2]. Only Molmo2 was most accurate with the target in the first position [REG: A1, accuracy by target position].

Dependence is not competence. Molmo2's accuracy on the item's own face was at chance on the composite input (.163 against 1/6; pooled *P* = .65, descriptive) and below it on the single face crop (.072) [REG: A1 table; A3 table, row Molmo2], and its pass comes from mismatched faces pulling accuracy further down; on the composite input its answer was "None of the above" on .579 of real cells [REG: "Rows added 2026-11-12, late", A1 share of answers "None of the above"]. For Qwen3-VL and Gemma 4, accuracy with the right face was well above chance, and it fell by 40 and 23 points when the face was swapped.

## 3. Agreement on natural faces cannot identify which cues a model uses

<!-- Scaffold, kept from before 2026-11-21 (old Section 3).
**Question.** Can agreement on natural items identify which cues a system uses?
**Rule.** None. These are mathematical results and design facts, with proofs in Supplementary Note 1 and in `research/theory/causal_feature_alignment.md`. They use no model output.
**Figure.** Figure 3, panels a to e (`research/paper/v2/figures/fig3.pdf`). Panel b shows the Proposition 2 check: for two observers that weight different but correlated cues, the correlation of their outputs equals the cue correlation (.5, .9 and .99), while the cosine between their cue effects stays 0 [REG: "Rows added 2026-11-12", "Proposition 2 check"].
-->

<!-- Scaffold, kept from before 2026-11-21 (old Section 4).
**Question.** Does each model's response pattern on natural clips align with the human pattern, over the same measured cues?
**Rule.** The confirmatory unit is the family pattern score on the model-frames route, at human ridge 10, with the attempt 1 null of record (DECISIONS 2026-11-04, "RQ1 inference after two failed calibration attempts", `research/DECISIONS.md:232`). It is confirmatory only if size and power pass at both 0.75 and 1 times the human effect (`research/DECISIONS.md:260`), which they did (`research/DECISIONS.md:270`). If more than 5% of clips are lost, the result is descriptive (`research/DECISIONS.md:264`). A clip counts only with a usable response on all 5 rotations (`research/DECISIONS.md:282`). The reading depends on RQ0 (`research/DECISIONS.md:327-332`):

| A1 | A3 | Natural-clip RQ1 |
|---|---|---|
| Pass | Pass | Confirmatory, subject to the 5 percent clip rule |
| Fail | Pass | Computed and reported as descriptive |
| Pass | Fail | Computed and reported as descriptive, pending inspection |
| Fail | Fail | Computed for the record, with no claim |

A positive score supports "aligned with the human pattern", not "uses the same cues" (`research/DECISIONS.md:234`).
**Figure.** Figure 4 (`research/paper/v2/figures/fig4.pdf`), drawn from the three locked `same_clip_family_model_*.json` files and the route recovery files: the composite pattern score per model with each detector alone, against its permutation null and the recovery range of human-like and prior-only observers.
-->

<!-- 2026-11-13, length plan cut 3: the recovery values that stood in old Section 4 are given in Methods ("RQ1 on natural clips": power .995 and 1.0, size at most .06 against .081) and in Supplementary Methods, Section S5, "Recovery and detectors (continued)" (power .775 at 0.5 times; half-shared observer .60 and .83; different-cues size .055), with the prior-only (.035) and actor-prior (.06) sizes in RESULTS_REGISTRY, "Pattern score, model-frames route (2026-11-04)". 2026-11-21, length cut 7: the OpenFace 3.0 and as-coded Snoek spectra moved word for word to Supplementary Methods, Section S11, "Results, Section 3, in full"; the per-detector scores, the permutation floor, the lost clip and the two limits on the scores moved word for word to Section S5, "Results, Section 3, natural clips, in full". -->

Face dependence shows that an answer uses the image, not which facial information it uses. The usual next step compares a model's responses with people's on natural items, and there this comparison cannot settle cue use. If two systems give the same responses on every natural item, every behavioural metric on those items is equal, yet their cue effects, meaning the change in response when one cue changes and the others do not, can take any pattern (Theorem 1, Methods). For linear observers, how far behaviour constrains cue effects depends on the smallest eigenvalue of the natural cue correlation (Proposition 3, Methods). The natural clips are poorly conditioned. On the 243 EU-Emotion clips, the smallest eigenvalue of the correlation of 15 LibreFace cues was .054, with a condition number of 76.1 [REG: Cue spectra, "EU-Emotion LibreFace"], against .44 to .47 and 3.4 to 3.5 for the randomised human design with left and right merged (Fig. 3d,e) [REG: Cue spectra, "Snoek left and right merged, 25 cues"]. Supplementary Methods, Section S11 gives a synthetic case in which outputs agree closely while cue effects do not.

The natural clips show the same limit in practice. Our preregistered natural-clip test regresses each system's graded responses on detector-measured families of facial action units and correlates the model's pattern with the human one on the same clips (the family pattern score; Methods). Recovery on synthetic observers passed the size and power rules fixed in advance, so the score is confirmatory [REG: "Pattern score, model-frames route (2026-11-04)"]. All three models were aligned with the human pattern: the score was .549 for Qwen3-VL (permutation *P* = .0002, 242 of 243 clips), .418 for Gemma 4 (*P* = .0006, 243 clips) and .543 for Molmo2 (*P* = .0002, 243 clips) (Fig. 4) [REG: Natural-clip RQ1 table]. That includes Molmo2, whose accuracy with the right face was at chance (Section 2), and two models for which the face spanned one or two image tokens on this input [REG: Input delivery]. The score supports alignment with the human pattern, as fixed in advance, and not use of the same cues: a system responding to an unmeasured cue correlated with the measured ones produces the same slopes (Proposition 4). To learn which cues a model uses, the correlations between cues have to be broken.

## 4. With cues randomised, Gemma 4's cue effects align with people's for basic emotions

<!-- Scaffold, kept from before 2026-11-21 (old Section 5).
**Question.** Do a model's effects of randomised facial action units, on rendered faces, align with randomised human effects?
**Rule (A4, residual CEA, per model and dataset).** (`research/PREREG_A4_DRAFT.md:121-125`.)

| Outcome | Requires |
|---|---|
| Aligned | All gates; signal p < .05; alignment Holm p < .05; CEA at least .30; CI lower bound above 0 |
| Not aligned | All gates; signal p < .05; CI upper bound divided by the square root of R_S below .30 |
| Inconclusive | Anything else, including any failed or missing gate |

Not aligned is always written "not aligned on these renders" (`:127`). A model with no signal cannot be Not aligned (`:129`). Headline per model per task: Aligned or Not aligned only if WC and EA agree, Mixed if one of each, Inconclusive otherwise; basic and conversational are never pooled (`:137`). Human-level only if the interval reaches the WC to EA reference, which is descriptive (`:135`). Validity gates V1 to V4 run before any human comparison, and missing means not passed (`:141-148`). Eligibility runs through the A1 and A3 table, as in A2 (methods.md, "Eligibility").
**Figure.** Figure 5 (`research/paper/v2/figures/fig5.pdf`), drawn by `research/scripts/make_figures.py --fig 5 --real` from the locked `results/a4/summary.json`: residual CEA per model and dataset with its 95% CI, the .30 bound and the human WC to EA reference, with the decision (a); raw CEA, secondary (b); and output correlation with the mean human response (c).
-->

<!-- Scaffold, kept from before 2026-11-21 (old Section 6b, A9). Added 2026-11-09; status and references brought up to date 2026-11-12 after the independent draft review (finding 10).
**Question.** Does A4's answer hold against a second set of randomised human observers, Liu et al.'s Experiment II?
**Status.** A9 was frozen on 2026-11-09 at 17:33 UTC, tag `a9-freeze` (commit `bc41374`), before any model saw a Liu render (Methods, Disclosure 11).
**Relation to A4.** A4's verdicts are final as frozen. A9 never enters A4's Holm family, headline or decision table, and is reported here, after A4. A9 and A4 estimates are never pooled. If they disagree for a model, both stand. An A9 Aligned or Not aligned does not make an A4 Inconclusive conclusive (`research/PREREG_A9_DRAFT.md`, "Relation to A4").
**Rule (A9, residual CEA, per model, one dataset).** (`research/PREREG_A9_DRAFT.md`, "Decision rule (unchanged from A4), per model".)

| Outcome | Requires |
|---|---|
| Aligned | All gates; signal p < .05; alignment Holm p < .05 within A9 (Gemma 4 and Qwen3-VL); residual CEA at least .30; CI lower bound above 0 |
| Not aligned | All gates; signal p < .05; CI upper bound divided by the square root of R_S below .30 |
| Inconclusive | Anything else, including any failed or missing gate, or no recovery-selected design |

Not aligned is written "not aligned on these renders". With one dataset there is no WC and EA headline, so the per-model decision is A9's result (same section). Human-level only if the interval reaches the residual Liu against Snoek basic WC reference, which is descriptive (`PREREG_A9_DRAFT.md`, "Human target", "Human reference, descriptive"; value .690, `research/reproduction/a9_human_ceiling.json`, `human_reference_cfa.basic_WC.residual` [REG: "Rows added 2026-11-12, late", A9 human ceiling and reference]). Gates: RQ0 at `single_face`, G2 to G4, V1 inherited from A4's pilot, V2 to V4 recomputed on Liu renders (`PREREG_A9_DRAFT.md`, Validity gate). Design B, by recovery. Decisions stay confirmatory whatever a model's correlation across faces. The accuracy claim for residual CEA on weak effects (0.25 times) is withdrawn under correlated face noise (lead decision 8; block `withdrawn_under_correlated_face_noise`).

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

**Withdrawn claim.** Under face-correlated noise the claim that residual CEA is estimated accurately on weak effects (0.25 times human strength) is withdrawn. This was decided after recovery run 2 was seen, replacing an earlier operationalisation that would have labelled every model outside the validated conditions (`PREREG_A9_DRAFT.md`, lead decisions 2 and 8). Methods, Disclosure 10 and "The narrowing", carry this in the paper.

**Writing rule.** Never "A9 confirms A4" or "A9 overturns A4". Say A9 is a second sample on the same platform (lab, generator, protocol, culture) and the same renders, not a second method (`PREREG_A9_DRAFT.md`, "Relation to A4").
-->

<!-- 2026-11-14, step 4.1: the slots and the per-outcome text drafted on 2026-11-13, before any A4 output existed, were replaced by the branches that match the locked summary (`results/a4/summary.json`, sha256 08148303...); the drafts are in the git history at 3ced4a5. Length plan cut 2 (2026-11-14): the per-dataset account is in Supplementary Methods, Section S6, "A4 results in full". The A9 slot list was replaced by the locked text (`results/a9/summary.json`, sha256 1d1a9995...; run note `research/reproduction/a9_run_note.json`); the slot list is in the git history at 453d69e. Length cut 6 (2026-11-14): A9's per-model detail moved to Section S7, "A9 results in full". 2026-11-21, length cut 7: the human ceilings and the common-profile share, raw CEA's recovery and the exempt cells, the detector-check values, the pre-run check, the shared configurations of the two tasks, the amplitude-set values and the pointers to raw CEA, the direct contrast and the blind readout moved word for word to Section S6, "Results, Section 4, randomised cues, in full"; A9's reliabilities, Holm P, sensitivity range and the replication sentence's own wording moved word for word to Section S7, "Results, Section 4, second human sample, in full". -->

Human data with the correlations broken exist. In reverse-correlation experiments, observers judge animated faces whose facial action units (AUs) are set at random, which identifies the effect of each AU on human judgement [19,42]. We used one such public data set, in which basic emotions and conversational signals were each judged by a Western Caucasian (WC) and an East Asian (EA) observer group [19,25]. People saw short animations from the original face generator, which the public deposit does not hold, so we rendered the same AU configurations as a new set of static faces at their peak, and measured each model's effect of every AU family on its responses. The models therefore saw the same configurations, not the same stimuli (Methods). We compare model and human effects by residual Cue Effect Alignment (CEA): the cosine between the two matrices of cue effects after removing from each its own average effect, so that the score reflects which action moves which label (Supplementary Note 1). By the preregistered rule, a model is aligned on a dataset if it passes four validity gates, the rendered AUs move its responses, and residual CEA is significant after Holm correction and at least .30, with a 95% CI above 0; it is not aligned on these renders if the reliability-corrected upper bound is below .30, and inconclusive otherwise. Residual CEA passed recovery on 66 of 66 checks [REG: A4 recovery, "Run 4 (frozen)"]. A face detector confirmed the rendered movement clearly for only four AU families, AU01, 05, 25 and 26, so alignment is with the rendered AU recipes [REG: "Intensity pairs with reading at amplitude 1 minus .25 of at least .5"].

For both models on all four datasets, every main stimulus entered with a complete rotation set, and the rendered action units moved the responses (signal *P* = .001, the floor at 999 permutations) [REG: "A4 share of main stimuli lost"; A4 table]. Gemma 4 passed every gate [REG: "A4 gates, Gemma 4"]. Its cue effects were aligned with people's for basic emotions in both observer groups, with residual CEA .386 (95% CI .372 to .397) on WC and .516 (.497 to .535) on EA (Fig. 5a). For conversational signals the result was mixed: aligned on EA at .369 (.348 to .385), and not aligned on these renders on WC at .025, whose upper bound after correction for reliability, .053, is below .30 [REG: A4 table, headlines; "A4 residual CI upper bound over the square root of R_S"]. Conversational EA is the least secure of the aligned results: restricted to the detector-confirmed families, its residual CEA was .158 [REG: "A4 sensitivities"]. No sensitivity changed Gemma 4's reading on basic labels or on conversational WC. No interval reaches the agreement between the two human groups (.780 for basic emotions, .732 for conversational signals), so none is human-level by that descriptive standard [REG: "WC to EA reference, residual"].

Qwen3-VL was inconclusive on both tasks [REG: A4 table, headlines]. On three datasets a validity gate failed, so no decision is made there, whatever the interval. Basic WC, conversational WC and conversational EA failed the domain limit, and conversational WC also failed the check that a configuration carries across faces [REG: "A4 gates, Qwen3-VL"]. On basic EA every gate passed, but the interval spans the floor: .264 (.219 to .299), with an upper bound of .306 after correction.

We repeated the test, as preregistered, on a second human sample: 20 further observers in Liu et al.'s Experiment II, with other faces from the same laboratory and face generator [60]. Both models passed every gate on these renders, all 1,200 configurations entered, and the rendered action units moved the responses (signal *P* = .001) [REG: "A9 gates"; "A9 configurations and share lost"; A9 table]. Gemma 4 was aligned again, with residual CEA .483 (95% CI .469 to .496), so its result on basic emotions replicated on a second human sample (same lab and generator) [REG: A9 table; A9 table, replication]. Its interval does not reach the residual reference between the two human samples (.690), so it is not human-level by that standard [REG: "A9 human references"]. Qwen3-VL's cue effects lined up with people's above chance (Holm *P* = .037), but by less than the smallest value of interest: residual CEA .230 (.189 to .269), with an upper bound of .273 after correction for reliability, so it is not aligned on these renders [REG: A9 table; "A9 residual CI upper bound over the square root of R_S"]. Its first decision on basic WC was inconclusive, because the domain gate failed there, so replication is not testable [REG: A9 table, replication]. Beside each decision, as fixed before data, the estimated correlation across faces was .192 for Gemma 4 and .338 for Qwen3-VL, both within the range recovery tested (up to .5); under the replaced first operationalisation, both decisions would have been labelled "outside the recovery-validated conditions" (Methods, "The narrowing") [REG: "A9 descriptive fields"]. Molmo2, excluded by its vision gates, was not run on either sample. Table 1 gathers every model's results.

## 5. Answers on the same faces track people's only weakly

<!-- Scaffold, kept from before 2026-11-21 (old Section 6).
**Question.** On the same rendered stimuli, does behavioural similarity say something different from CEA?
**Rule.** Descriptive, no decision (`research/PREREG_A4_DRAFT.md:162-169`). Output similarity is Pearson *r* between model and mean human response, option means removed. Modal agreement is reported with chance-corrected kappa. "Whether these disagree with CEA is an empirical result" (`:169`).
**Writing rule.** Agreement is not a failure of the theory, nor proof of use (outline.md, section 6).
**Figure.** Part of Figure 5.
-->

<!-- 2026-11-14, step 4.1: the slots and the per-outcome text drafted on 2026-11-13, before any A4 output existed, were replaced by the branch that matches the locked summary; the drafts are in the git history at 3ced4a5. 2026-11-21: the reliability caveat (third-to-last and second-to-last sentences) is new, a correction that narrows the reading; nothing left this section. -->

Comparing answers on the same rendered faces gives a weaker picture (Fig. 5c) [REG: "A4 behaviour on the same renders"]. With option means removed, Gemma 4's responses correlated with the mean human response at r = .140, .229 and .138 where its cue effects are aligned (basic WC and EA, conversational EA), each less than half its residual CEA. On conversational WC both were near zero (r = -.030, CEA .025). Its modal answer agreed with the human mode on .096 to .152 of stimuli (kappa -.021 to .096). Qwen3-VL's correlations ran from -.025 to .084. So the two measures came apart in size, not in sign: judged by its answers alone, Gemma 4 would have looked only weakly similar to people where its cue effects met the alignment bound. Whether this gap means anything is open. The comparison is descriptive, as preregistered, and it has no noise ceiling. Each stimulus's human response is the mean of that configuration's own trials, far fewer than the trials behind each cue effect, so the gap may reflect, in part or in whole, the lower reliability of per-stimulus human responses. And on a randomised design, unlike on natural items, behaviour does constrain cue effects for linear observers (Proposition 3), so a gap that survived a noise ceiling would need an explanation other than the identifiability limit of Section 3, such as nonlinear responses or the common profile that residual CEA removes. An aligned CEA still means a shared pattern of cue effects, not the same strategy (Methods).

## 6. Masks and filters move judgements as predicted on about half the single-face contrasts

<!-- Scaffold, kept from before 2026-11-21 (old Section 7).
**Question.** Do region masks, spatial-frequency filters and changes of frame order move a model's judgement in the direction of a stated human prediction?
**Rule (A2, per prediction).** (`research/PREREG_A2_DRAFT.md:188-193`.)

| Outcome | Requires |
|---|---|
| Matches prediction | RQ0, G2 to G4, Gate B, route delivery and the sham all passed; step-down p < .05 in the predicted direction on the blind readout; fill rule met |
| Opposes prediction | The same gates passed; step-down p < .05 two-sided in the opposite direction |
| Null | The same gates passed; neither test significant; all null conditions met |
| Inconclusive | Anything else, including a sham or delivery failure |

Gate B is B1 to B5 (`research/PREREG_A2_DRAFT.md:109-117`). A model and route that fail it go no further on the blind readout. Null needs all conditions at `:201-206`. P5, P6, P9, P10, P14 and P15 cannot reach Null, and on Qwen3-VL P12 and P13 cannot either (`:199`). Every result carries the BASIC ONLY scope (`:195`). After the freeze, dated additions made before any data govern how results are read: source support for P1 to P15 was corrected (`:332-353`); each Matches or Opposes verdict is also reported against step-down *P* < .01, and a verdict holding only at .05 is reported as not robust to the skewed null described below (DECISIONS 2026-11-08, `research/DECISIONS.md:398`); and the rulings of 2026-11-12 apply the fill rule to Opposes verdicts and give the contrasts that cannot reach Null as 90% equivalence intervals (DECISIONS 2026-11-12).
**Figure.** Figure 6 (`research/paper/v2/figures/fig6.pdf`), drawn by `research/scripts/make_figures.py --fig 6 --real` from A2's locked outputs and declared extras: the change in the predicted direction per contrast, with its actor 95% interval and its class after the fill rule, by model and route.
-->

<!-- 2026-11-14, step 4.1: the slots and the per-outcome text drafted on 2026-11-13, before any A2 output existed, were replaced by the branches that match the locked outputs (`research/reproduction/stimulus_interventions_{gemma4,qwen3vl}.json` and `a2_extras_*.json`; run note `a2_run_note.json`); the drafts are in the git history at e36fd7a. Length plan cut 2: the per-contrast account is in Supplementary Methods, Section S8, "A2 results in full". Length cut 6 (2026-11-14): the per-route familywise errors, "no repair was needed" and the per-contrast sensitivity changes moved word for word to Section S8. 2026-11-21, length cut 7: the recovery sentence, the lost cells, the P values of the non-robust Matches, the other readouts and the detail of the sensitivities moved word for word to Section S8, "Results, Section 6, in full". -->

We also edited the clips and asked whether each model's option-free judgement, read from its probabilities for 106 emotion words when asked what the person feels with no options offered (Methods), moves as stated predictions from human studies say it should. The predictions concern basic emotions only. For both models on both inputs every gate passed [REG: "A2 gates"]. On a single face, region masks and spatial-frequency filters moved the judgement as predicted on 5 of 10 contrasts for Gemma 4 and on 4 of 10 for Qwen3-VL, and the rest were inconclusive (Fig. 6) [REG: A2 tables; "A2 classes after the fill rule"]. Three matches per model also hold at step-down *P* < .01, which keeps familywise error under .05 under the skewed nulls we simulated [REG: "A2 sign-flip FWER under skewed gain nulls"]: the mouth mask on Happy and the mouth-minus-eye contrast for both models, with the low- minus high-pass filter on Sad for Gemma 4 and the brow mask on Angry for Qwen3-VL. The others hold only at .05. Only the mouth mask on Happy, for both models, and the brow mask on Angry, for Qwen3-VL, held under every sensitivity; the other single-face matches depend on which items are included [REG: "A2 sensitivities"]. On the video input, neither shuffled frames nor clips cut to their first or last third matched a prediction: for Gemma 4, shuffling frames on Happy and Surprised raised the acted label, against the prediction (two-sided step-down *P* = .024, not robust to this null), and every other video contrast on both models was inconclusive. Contrasts that cannot reach a null by design are given as 90% equivalence intervals in Supplementary Methods, Section S8 [REG: "A2 capped contrasts"]. A match is agreement with a stated prediction drawn from other stimuli, not with human data on these clips.

## 7. Face evidence, not word knowledge alone, moves the judgement

<!-- Scaffold, kept from before 2026-11-21 (old Section 10, RQ4).
**RQ4 (A5).** A5 is frozen, tag `a5-freeze`, on 2026-11-08, before any A2 cell existed (`research/PREREG_A5_DRAFT.md:3`). Only the evidence classes are confirmatory (`:246`, `:576`):

| Class | Requires |
|---|---|
| Shown | E* passes (step-down p < .05, 95% CI lies above 0), and Gate B passed |
| Unmapped | E* passes, and any of B3, B4 or B5 failed |
| Absent | E* not significant, its 90% CI inside the margin, B4 failed, and the positive control holds |
| Unresolved | Anything else |

(`research/PREREG_A5_DRAFT.md:226-231`.) Balance classes and verdict names are descriptive, not error-controlled under skewed portrayal gain (`:574`).
**Figure.** The free display slot (outline.md, "Display items"), only if A5 appears.
-->

<!-- 2026-11-14, step 4.1: the slots and the drafted per-outcome text were replaced by the branch that matches the locked output (`results/a5/a5_analysis.json`, sha256 3fb4f5de...; run note `research/reproduction/a5_run_note.json`; pre-run check passed); the drafts are in the git history at 6d01d23. 2026-11-21, length cut 7: the balance, Phi, word share, survival and H2 values and the note on H3 moved word for word to Supplementary Methods, Section S10, "Results, Section 7, in full". -->

A language model could answer from what an emotion word usually goes with, rather than from the face. Our preregistered prior-and-evidence test separates the two on the option-free judgement, with image-free, grey and mismatched-face baselines (Methods). On both inputs, both models' evidence class was Shown, with the judgement's gate passed [REG: "A5 confirmatory analysis (2026-11-14)"]: evidence from the face moved each model's judgement towards the portrayed emotion. The face-evidence estimate E* ran from .670 to 1.048 across models and inputs, every 95% CI above 0, with step-down *P* at most .00002 and no cell flagged. The face also moved each model's judgement towards people's response distribution beyond the default (Delta J .084 to .153, two-sided step-down *P* at most .00003) [REG: "A5 H1"]. Per-input values, and how each judgement balances prior and evidence, which is descriptive, are in Supplementary Methods, Section S10.

## 8. Larger models in the same families

<!-- Added 2026-11-09. A8 is FROZEN, tag a8-freeze (research/PREREG_A8_DRAFT.md:3). No A8 output has been read. No result is stated. The scaffold below is kept, unchanged, for the fill at step 4.1; the visible slot is the one placeholder.
Filled 2026-11-23 (DECISIONS 2026-11-23, row 105: A8 ended for this paper after Phase A, its scored steps deferred past the GPU shutdown by dated note). Slots used: A3 (both Pass, A8 family of two, read 2026-11-21); vision gates, no result for either model (rows 99 to 101; row 102's rerun deferred); A1, A2 and A4 not run by the stop rule; natural-clip RQ1 not run (needs A1). Delivery and the ladder are in Methods and Supplementary Methods, Section S9, to keep the main text under 5,000 words.
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
- Gate B and A2 verdicts, with the .01 reading: [PENDING: `research/reproduction/a8/<a8_id>/stimulus_interventions_<key>.json` (written by `research/scripts/a8_analyse_a2.py`, path fixed in the A8 runbook, ROADMAP Part 3), with verdicts after the fill rule from `a2_extras_<key>.json` (A8 dated note 2026-11-13)]
- A4 per dataset and headline per task: [PENDING: `results/a8/a4/summary.json`]
- steps not run, and why (stop rule, not tested after retry, storage, deferral of A4 by a dated note): [PENDING: launcher records under `research/reproduction/a8/<a8_id>/launch/`]
**Figure.** A8 rows join Figures 2, 5 and 6 and Table 1 below the frozen rows, marked A8. [PENDING: figure functions.]
-->

We ran the frozen tests on two larger models in the same families, Qwen3-VL-32B and Gemma 4 31B, as preregistered (Methods). On a single face crop, both models' answers depended on the face: the delta was .329 for Qwen3-VL-32B and .316 for Gemma 4 31B (Holm *P* = .00006 for each, over two models, a family less strict per model than the three-model one) [REG: A8 A3 table]. Neither has a validity-gate result, so by the preregistered stop rule neither ran the other tests (Methods; Table 1) [REG: "A8 Phase A records and final checks"].

## 9. What could not be tested

<!-- Scaffold, kept from before 2026-11-21 (old Section 9).
**Question.** Is a cue's effect carried from image tokens through the language model?
**Rule.** Not a test (DECISIONS 2026-11-07, "G7 becomes a scoped limit", `research/DECISIONS.md:355-363`). A7 was frozen on 2026-11-08, tag `a7-freeze`, before any full A2 cell was read (`research/PREREG_A7_DRAFT.md:3`). Its verdict table is `research/PREREG_A7_DRAFT.md:397-409` and its reporting table `:444-455`. No site is called a cut unless Stage 0 shows it complete.
**Figure.** None in the main text. Stage 0 table in the Supplement (`research/paper/v2/supplementary_table_a7_stage0.md`, built by `research/scripts/make_stage0_table.py` from `research/reproduction/a7_stage0_gemma4.json` and `a7_stage0_qwen3vl.json`).
-->

<!-- 2026-11-14, length cut 6 (the author's go-ahead; DECISIONS 2026-11-14): the block ratios, the leak values and the token map's location rate moved word for word to Supplementary Methods, Section S10, "A7 Stage 0 (continued)"; the route arm's path-fault sentence came out because Methods ("A7 Stage 1 runs"), Section S10 (the route arm) and decisions table row 90 give the full account. 2026-11-14, step 4.1: the drafted per-outcome text was replaced by the branch that matches the locked output (`research/reproduction/a7_analysis.json`; run note `a7_run_note.json`); the draft is in the git history at e0466c9. 2026-11-21, length cut 7: the Stage 0 values and the route arm's preconditions moved word for word to Section S10, "Results, Section 9, in full"; the RQ5 paragraph of old Section 10 moved word for word to Section S10 too, and Methods ("RQ4, frozen, and RQ5, closed as a limit") already gives its rule and value. -->

Two questions stayed out of reach. Whether a cue's effect is carried from image tokens through the language model is not testable here beyond the interface. On both eligible models the only complete cut of the vision path was the whole path, at its exit into the language model, whose contrast is zero by construction; no site inside the language model was shown to be a complete cut; and the tokens at a cue's location could not be shown to carry its effect (Supplementary Methods, Section S10) [REG: "A7 Stage 0, Gemma 4"; "A7 Stage 0, Qwen3-VL"]. A secondary split of that cut, with no human reading, is reported in Section S10 as preregistered [REG: "A7 Stage 1 route arm (2026-11-14)"]. Whether people and models make the same errors for the same reasons was not tested: the planned test failed its own pre-data check and was never run (Methods) [REG: "A6 pre-freeze cluster checks"].
