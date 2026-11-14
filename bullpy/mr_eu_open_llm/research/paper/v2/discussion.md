<!--
Draft v2 scaffold, 2026-11-08 (ROADMAP step 0.8). An outline, not conclusions. No A1, A2 or A4 model result has been read.
For each main question, one short paragraph per outcome class allowed by its frozen decision rule. The branch that matches the locked results is picked at ROADMAP step 4.3; the others are deleted.
Classes are taken from the amendments and DECISIONS, with no class added. The Discussion carries no subheadings in the journal; the headings here are for drafting only.

2026-11-12, late (new agent session): every branch still open is now drafted as prose, before any A2, A4, A5, A7 Stage 1, A8 or A9 output exists, so that at step 4.3 a branch is chosen and its slots filled, not written after the result is seen. Added: A9 (wording fixed by its block), A8 (scope only), the A7 route arm (secondary). Scenario S4 is struck (RQ0 locked as R0-a); S2 is folded into S3 (natural clips positive for all three models, so any A4 Not aligned is S3). Limitations brought up to date (A7 Stage 0 ran; A5 frozen; outline item 16 closed). Numbers stated here are only those already locked in RESULTS_REGISTRY; every other value is a [slot].
-->

# Discussion (outline)

## Opening paragraph

<!-- 2026-11-14, step 4.3: assembled from the fixed opening, the live scenario (S6, the models differ, with S1 for Gemma 4 on basic labels and S5 for Qwen3-VL) and the A2 sentences R2-b and R2-c, all drafted before the data. The scaffold is in the git history at 3f718bb. -->

We asked how similar vision-language models are to people in how they read emotion from faces, beyond giving the same answers. Three findings hold whatever the later tests showed. The design we used before, and several published designs, let option position or prompt text identify the answer, so accuracy on such designs is not evidence that a model read the face (Results, Section 1). Once option position was rotated and the prompt held fixed, the answers of all three models depended on the face shown (Section 2), and on natural clips their graded responses lined up with the human pattern over four families of facial actions (Section 4). And alignment of that kind cannot show which cues a model uses, because a model that responds to unmeasured cues correlated with the measured ones produces the same score (Section 3; Supplementary Note 1). The randomised test was built to reach past that limit. The two eligible models reached different degrees of similarity there, and we report them as profiles with no pooled statement. When facial actions were randomised on rendered faces, so that no cue could stand in for another, Gemma 4's cue effects lined up with those measured in people on basic labels, in two observer groups and again in a second human sample, beyond what chance and measurement noise allow. That is a shared pattern of cue effects on these renders, at a peak frame. It is not evidence that the model follows the same strategy, it falls short of the agreement between the two human groups, and no test here reaches computation. For Qwen3-VL the randomised test did not reach a decision, because its answers failed a validity gate on three of four datasets. On the second sample its cue effects were above chance but below the bound we set. Region masks and frequency filters moved the option-free judgement in the direction people's data predict for five of 16 contrasts on Gemma 4 and four on Qwen3-VL, on basic labels in this stimulus set. Changes of frame order matched no prediction, and on Gemma 4 shuffling frames moved the judgement against the stated prediction on one contrast.

## RQ0: dependence on the face

Branch R0-a, locked 9 November 2026 (RESULTS_REGISTRY, "A1 rotated 6AFC"). Drafted text:

With option position rotated and the prompt held fixed, accuracy on all three models fell when the face came from another clip, on the composite video input (A1) and on a single face crop (A3). So the answers these models give on this task depend on the face shown, and not only on the option text, which our earlier design could not exclude. This is a test of dependence, not of competence. Molmo2 shows it most clearly: its accuracy on the right face was at chance, and the face effect came from mismatched faces pulling it below chance. For Qwen3-VL and Gemma 4, accuracy was well above chance and fell by 40 and 23 points when the face was swapped. The composite grid did not prevent face use, despite its small number of image tokens per face (Methods), so the input limits we documented did not decide the outcome here. Molmo2 stays out of A2 and of A4's family on its gate failures, which A1 does not change.

## RQ1 on natural clips

Positive and confirmatory on all three models (RESULTS_REGISTRY, "Natural-clip RQ1"). Drafted text:

On natural clips, each model's responses lined up with the human pattern over four families of facial action units, with family pattern scores of .42 to .55, each well outside its permutation null. This is the claim the design supports, and only this one. The score is a regression on measured cues, so a model responding to an unmeasured cue that is correlated with them would produce the same result (Proposition 4); alignment with the human pattern is not evidence that the models use the same cues. The randomised test in the next section is the one built to reach that question. Two limits apply. For Gemma 4 and Molmo2 the scores come from a composite input on which the face spans one or two image tokens, below our own delivery gate, a limit recorded before any rotated result was read; they show alignment despite that input and say nothing about which visual information carried it. And Molmo2's score sits beside chance-level accuracy and a majority of "None of the above" answers: the score is computed on graded responses over all options, so a model can track the human pattern while rarely putting the most weight on the acted label.

## RQ1 under randomisation (A4)

Rule: `research/PREREG_A4_DRAFT.md:121-137`. Each dataset alone, then one headline per model per task. Basic and conversational are separate claims.

<!-- 2026-11-14, step 4.3: branches picked for the locked A4 summary (RESULTS_REGISTRY, "A4 confirmatory analysis (2026-11-14)"; Results, Sections 5 and 6). The drafted branches, written before any A4 output existed, are in the git history at be63deb. For the opening paragraph, A4's part of the scenario is S6 (the models differ), with S1 for Gemma 4 on basic labels and S5 for Qwen3-VL; the opening is assembled once A2 is locked. -->

**Gemma 4, basic labels: Aligned.** With facial actions set at random on rendered faces, Gemma 4's responses to basic labels changed with each action in the same pattern as people's, in both cultures (residual CEA .386 and .516, against a smallest value of interest of .30). "Residual" means that the part of each effect shared by all actions, such as a general move away from "none of the above" whenever the face moves, was removed first, so the score reflects which action moves which label. This is a reliable shared pattern of cue effects on these avatars, at a peak frame. It is not evidence of the same strategy: in recovery, an observer with 16 of 25 action families swapped in pairs could still clear .30 (Methods). It is not human-level, and most action families on these renders are not confirmed by a detector, so the pattern is one of rendered recipes.

**Gemma 4, conversational labels: Mixed.** Gemma 4 was aligned on the East Asian set and not aligned on the Western Caucasian set, so its headline is Mixed. The two sets are not independent tests of a model, and we draw no claim about culture from the difference: the East Asian faces were not based on East Asian people. The East Asian alignment is also the least secure of Gemma 4's results (Results, Section 5). For the Western set, A4 cannot separate three explanations: the domain of the avatars, the static peak frame against 1.25 s of human-viewed motion, and the linear mapping of intensity to amplitude. The result says nothing about real faces. Beside Gemma 4's positive natural-clip score, that dataset gives the pair the identifiability result allows, though it cannot exclude the avatar domain as the reason.

**Qwen3-VL: Inconclusive.** Validity gates failed on three datasets, and the fourth spanned both floors. An Inconclusive is not a null. A decision would need renders on which Qwen3-VL passes the failing gates, or the original stimuli.

**Behaviour beside cue effects.** Behaviour and cue effects came apart in size on the same stimuli (Section 6). Judged by its outputs alone, Gemma 4 would have looked only weakly similar where its cue effects were aligned, the opposite direction to Proposition 2's example. That is the case the theory needs, observed. It is not proof of cue use.

## A second human sample (A9)

Rule and wording fixed before any model data (`research/PREREG_A9_DRAFT.md`, "Relation to A4" and block `replication`). Comparator: the same model's A4 decision on basic WC; A4's basic headline beside it. A9 is a second sample on the same platform (lab, generator, protocol, culture) and the same renders, not a second method, and is never called independent of the GFG platform. Never "A9 confirms A4" or "A9 overturns A4".

<!-- 2026-11-14, step 4.3: branches picked for the locked A9 summary (RESULTS_REGISTRY, "A9 confirmatory analysis (2026-11-14)"; Results, Section 6b). The drafted branches are in the git history at 453d69e. -->

**Gemma 4: both Aligned.** Gemma 4's basic-label alignment replicated on a second human sample (same lab and generator), Liu et al.'s 20 observers, with residual alignment .483 (.469 to .496). That is a second sample on the same platform and the same renders, not a second method, so it strengthens the claim about these renders and no more.

**Qwen3-VL: replication not testable.** Qwen3-VL was Not aligned on Liu's renders, but its A4 basic WC decision was Inconclusive because a validity gate failed, so the replication is not testable. The A9 result stands on its own terms: on the second sample, every gate passed, and its cue effects were above chance but below the smallest value of interest.

An A9 Inconclusive is never a failure to replicate. A9 cannot change an A4 verdict.

## RQ2: stimulus interventions (A2)

Rule: `research/PREREG_A2_DRAFT.md:188-206`, the deviation at `:332-353`, the .01 reading (DECISIONS 2026-11-08) and the pre-data rulings of 2026-11-12 (the fill rule also governs Opposes). BASIC ONLY scope on every result. RQ2 here is A2's stimulus arm, which says nothing on necessity or sufficiency of an AU, and nothing about how people would respond to these edits, which nobody has measured.

<!-- 2026-11-14, step 4.3: R2-b (mixed classes, by cue type) with R2-c for Gemma 4's P12 and the R2-d note on capped contrasts, for the locked A2 outputs (RESULTS_REGISTRY, "A2 confirmatory analysis (2026-11-14)"; Results, Section 7). The other drafted branches are in the git history at 3f718bb. -->

**Mixed classes, by cue type (R2-b).**
- *Region masks.* These matched where the predictions are strongest: the mouth mask on Happy (P1, both models), the mouth-minus-eye contrast (P6, both models) and the brow mask on Angry (P5, both). Qwen3-VL also matched the eye mask on Sad (P4).
- *Frequency filters.* These matched on Gemma 4 (P9 and P10) and were not decided on Qwen3-VL.
- *Frame order.* This matched nothing.
- *The stricter level.* At *P* < .01, which keeps the familywise error under .05 under the skewed nulls we simulated (Methods), three Matches hold per model: P1, P6 and P10 on Gemma 4, and P1, P5 and P6 on Qwen3-VL.
- *Robustness.* Only P1 stayed Matches on both models under every sensitivity, so beyond the mouth mask on Happy these agreements depend on which items are included.

The predictions come from other stimuli, so a Match is agreement with a stated prediction, not with human data on these clips.

**One Opposes (R2-c).** For Gemma 4, shuffling frames on Happy and Surprised raised the acted label, against the stated prediction (P12; not at *P* < .01). An Opposes on P12 says nothing against human alignment, since the closest human test found no cost of shuffling.

The contrasts that cannot reach Null by design are given as 90% equivalence intervals, not as nulls (Section S8). No contrast met every condition of the null rule.

## RQ4: prior and evidence (A5)

Rule: `research/PREREG_A5_DRAFT.md:226-231`, with balance descriptive (`:574-576`), and the pre-data rulings of 2026-11-12 (input errors void a run; Absent reported with E*'s interval, "below the margin" where it excludes 0). A5 is a reanalysis of A2's cells, so it exists only where A2 ran.

<!-- 2026-11-14, step 4.3: the Shown branch, which matches the locked A5 output on every model and route (RESULTS_REGISTRY, "A5 confirmatory analysis (2026-11-14)"; Results, Section 10). The other drafted branches are in the git history at 3f718bb. -->

**Shown, on both models and both routes.** Evidence from the face moved each model's option-free judgement towards the portrayed word, with Gate B passed, so the judgement is not word knowledge alone. The balance between prior and evidence (Even for Gemma 4, Face for Qwen3-VL) and the share carried by words are description, not claims.

## Larger models (A8)

Rule: `research/PREREG_A8_DRAFT.md:678-707`. A8 rows sit under the frozen rows, may support a statement of scope only, and cannot change a frozen verdict. Families are A8's own, smaller than the frozen ones, which the text must say.

- **A8 model agrees with its sibling.** On [test], [larger model] reached the same class as [small sibling], which extends that result's scope within the family to [size]. It is not a new test of the claim.
- **A8 model differs from its sibling.** On [test], [larger model] was [class] where [sibling] was [class]. Both are reported and the difference is described, not tested. It bounds the frozen result's scope to the model size tested.
- **Not run.** [Step] was not run on [model] because [the A8 stop rule, a failed retry, storage, or deferral by a dated note], and nothing is inferred from that.

## RQ3: the route through the model

A scoped limit in every branch (DECISIONS 2026-11-07). Stage 0 is locked (RESULTS_REGISTRY, "A7 Stage 0", 2026-11-10). Drafted text:

On both eligible models the only complete cut of the vision path was the whole path, at its exit into the language model, and the contrast of a whole-path cut is zero by construction. No site inside the language model was shown to be a complete cut, so whether a cue's effect is carried through the language model is not testable here beyond the interface. The tokens at a cue's location did not hold its effect on Gemma 4, where it leaked past them, and could not be located reliably on Qwen3-VL, so cue-level mediation is not testable by a token cut on either model.

Then one sentence for the route arm, a secondary split of the complete exit cut reported in the supplement, which has no human reading and never answers RQ3:

- **Carried by one block, or Both blocks.** On this edit and readout, [model]'s change was carried by [block]. The interaction rider, [rider], is descriptive and is given with its recovery rates. This is about which tensor carries the effect of one eyes-and-mouth edit, not about how the model reads faces.
- **No effect to carry.** The joint eyes-and-mouth edit did not move [model]'s option-free judgement enough to trace; a statement about the edit, reported with its recovery reach (.62 to .65 at the null).
- **Not tested.** The route arm's preconditions (Gate B, the sham or the gates) failed for [model], applied by code as recorded before any A2 output existed.
- **Inconclusive.** [reason code].

Never "the model does not use route R" without "on this edit"; never "blocking route R"; no depth statement about the language model.

## RQ5

Not tested. A6 did not freeze: its own pre-freeze check on the first frame, fixed before any model output, failed on one of three detectors (GraphAU AU10 ratio .80 against .50; DECISIONS 2026-11-10). Named as future work, with the fill design as the thing to change: a fill taken from a neutral clip of the same actor rather than the first frame, in a new amendment frozen before any data.

## What would settle it

Original stimuli; new human data on the same edited clips, offered as future work with no claim about the result; playback speed once a route delivers it (outline.md, Discussion item 6).

## A methods recommendation (benchmark audit)

<!-- Added 2026-11-09. Fixed whatever the results. Source: research/BENCHMARK_LEAK_AUDIT.md and research/reproduction/benchmark_leak_audit.json (exploratory, post hoc, annotation text only, no model run); DECISIONS 2026-11-09, "benchmark leak audit is a methods note, not a headline". Not a headline and not a field-level claim. 2026-11-13, length plan cut 5 (outline.md, "Length check"): the audit's detail (the two partial cases, the source of the majority-class gap, and why these accuracies are lower bounds) moved word for word to Supplementary Methods, Section S2; the recommendation and its scope here are unchanged. -->

Our own task let option position and option sets identify the answer. To see whether public benchmarks share that problem, we read the annotation files of 11 multiple-choice affect benchmarks for vision-language and multimodal models, without running any model, in an exploratory audit done after the fact (Supplementary Methods, Section S2). None of the 11 combined a constant answer position with one option set per label, in the order its own evaluation code presents items. So our confound is not shown to be common, and we do not claim it is. We found two partial cases [63,64] and weaker text-only shortcuts: where every item shows the same label list, answering the majority class beat the nominal 1/*k* chance by 9 to 32 percentage points in several benchmarks. The lesson we draw is narrow. Papers that compare models or people on such tests should report text-only baselines, for answer position, option set, answer prior and majority class, beside 1/*k*, and should remove rotated copies of an item before any leave-one-out check. This is not a claim that affect benchmarks are invalid, and several of the subsets we checked were balanced.

## Limitations

<!-- 2026-11-14, length plan cut 1 (the author's decision): the numbered list moved word for word to Supplementary Methods, Section S12. This paragraph keeps the limits that bear on the claims made. -->

The claims are bounded by five limits. Every dated decision and deviation is listed in the Supplementary Information (Section S12 and the table of dated decisions), and none moved a threshold. First, the cue-level results hold on rendered avatars at a static peak frame, with intensity mapped linearly to amplitude and two actions approximated. Most action families on these renders are not confirmed by a detector, so alignment is with rendered recipes, and nothing here speaks to real faces. Second, the East Asian sets use faces not based on East Asian people, and the second human sample shares the first's lab, generator, protocol and culture, so A9 is a second sample, not a second method. Third, the models are open and mid-sized. Molmo2 left the randomised tests on its vision gates, Qwen3-VL's randomised results were limited by a validity gate on three of four datasets, and larger models in the same families are tested only as far as A8 reaches. Fourth, dependence on the face is not competence, and pattern alignment on natural clips is not evidence of shared cues (Proposition 4). Fifth, the human predictions for the stimulus interventions come from other stimuli, and A2 cannot reach a null on half of its contrasts by design.
