<!--
Draft v2 scaffold, 2026-11-08 (ROADMAP step 0.8). An outline, not conclusions. No A1, A2 or A4 model result has been read.
For each main question, one short paragraph per outcome class allowed by its frozen decision rule. The branch that matches the locked results is picked at ROADMAP step 4.3; the others are deleted.
Classes are taken from the amendments and DECISIONS, with no class added. The Discussion carries no subheadings in the journal; the headings here are for drafting only.

2026-11-12, late (new agent session): every branch still open is now drafted as prose, before any A2, A4, A5, A7 Stage 1, A8 or A9 output exists, so that at step 4.3 a branch is chosen and its slots filled, not written after the result is seen. Added: A9 (wording fixed by its block), A8 (scope only), the A7 route arm (secondary). Scenario S4 is struck (RQ0 locked as R0-a); S2 is folded into S3 (natural clips positive for all three models, so any A4 Not aligned is S3). Limitations brought up to date (A7 Stage 0 ran; A5 frozen; outline item 16 closed). Numbers stated here are only those already locked in RESULTS_REGISTRY; every other value is a [slot].
-->

# Discussion (outline)

## Opening paragraph

Built from three parts: a fixed opening, one A4 sentence group from the live scenario, and one A2 sentence. Scenarios from outline.md, "How the whole paper reads". [PENDING: scenario, once A4 and A2 are locked.]

**Fixed opening (locked parts only).** We asked how similar vision-language models are to people in how they read emotion from faces, beyond giving the same answers. Three findings hold whatever the later tests showed. The design we used before, and several published designs, let option position or prompt text identify the answer, so accuracy on such designs is not evidence that a model read the face (Results, Section 1). Once option position was rotated and the prompt held fixed, the answers of all three models depended on the face shown (Section 2), and on natural clips their graded responses lined up with the human pattern over four families of facial actions (Section 4). And alignment of that kind cannot show which cues a model uses, because a model that responds to unmeasured cues correlated with the measured ones produces the same score (Section 3; Supplementary Note 1). The randomised test was built to reach past that limit.

**S1 (A4 Aligned on a task for [models]).** When facial actions were randomised on rendered faces, so that no cue could stand in for another, the cue effects of [models] lined up with those measured in people on [task] labels, beyond what chance and measurement noise allow. That is a shared pattern of cue effects on these renders, at a peak frame; it is not evidence that the models follow the same strategy, and no test here reaches computation.

**S3 (A4 Not aligned on these renders for [models]).** When facial actions were randomised on rendered faces, the cue effects of [models] stayed below the floor we set for alignment, even after correction for measurement noise: not aligned on these renders. Agreement on natural clips therefore did not carry over to randomised cue effects on rendered faces. The identifiability result allows exactly this, since a model can match the human pattern on natural clips through cues that are correlated with the ones people use. It does not show that this is what happened, because rendered faces differ from natural ones in domain, in motion and in how intensity maps to amplitude.

**S5 (A4 Inconclusive for [models]).** The randomised test did not reach a decision for [models], because [gate or test]. The audit and the identifiability results stand without it, and the cue-level question stays open with its requirements stated: [a render validated by a detector for the failing families, or the original stimuli].

**S6 (the models differ).** The two eligible models reached different degrees of similarity: [model] [degree], [model] [degree]. We report them as profiles and make no pooled statement.

**A2 sentence, one of.** (R2-a or b) Region masks, frequency filters and changes of frame order moved the option-free judgement in the direction people's data predict for [n of 16] contrasts, on basic labels in this stimulus set. (R2-c) For [contrasts], the edits moved the judgement against the stated human prediction. (R2-d) Most stimulus edits were not decided, and capped contrasts cannot reach a null by design. (R2-f) The option-free readout carried no usable face signal for [model and route], so the stimulus edits could not be tested there.

## RQ0: dependence on the face

Branch R0-a, locked 9 November 2026 (RESULTS_REGISTRY, "A1 rotated 6AFC"). Drafted text:

With option position rotated and the prompt held fixed, accuracy on all three models fell when the face came from another clip, on the composite video input (A1) and on a single face crop (A3). So the answers these models give on this task depend on the face shown, and not only on the option text, which our earlier design could not exclude. This is a test of dependence, not of competence. Molmo2 shows it most clearly: its accuracy on the right face was at chance, and the face effect came from mismatched faces pulling it below chance. For Qwen3-VL and Gemma 4, accuracy was well above chance and fell by 40 and 23 points when the face was swapped. The composite grid did not prevent face use, despite its small number of image tokens per face (Methods), so the input limits we documented did not decide the outcome here. Molmo2 stays out of A2 and of A4's family on its gate failures, which A1 does not change.

## RQ1 on natural clips

Positive and confirmatory on all three models (RESULTS_REGISTRY, "Natural-clip RQ1"). Drafted text:

On natural clips, each model's responses lined up with the human pattern over four families of facial action units, with family pattern scores of .42 to .55, each well outside its permutation null. This is the claim the design supports, and only this one. The score is a regression on measured cues, so a model responding to an unmeasured cue that is correlated with them would produce the same result (Proposition 4); alignment with the human pattern is not evidence that the models use the same cues. The randomised test in the next section is the one built to reach that question. Molmo2's alignment beside chance-level accuracy also shows that the two measures can come apart: a model can track the human pattern of graded responses while rarely putting the most weight on the acted label.

## RQ1 under randomisation (A4)

Rule: `research/PREREG_A4_DRAFT.md:121-137`. Each dataset alone, then one headline per model per task. Basic and conversational are separate claims. Drafted text per branch; use the one that matches each model and task, and say which datasets each headline rests on.

**Aligned.** On faces where the facial actions were set at random, [model]'s responses to [task] labels changed with each action in the same pattern as people's did, with a residual cue effect alignment of [value] (95% CI [lower, upper]) on [datasets], against a smallest value of interest of .30. "Residual" means that the part of each effect shared by all actions, such as a general move away from "none of the above" whenever the face moves, was removed first, so the score reflects which action moves which label. This is a reliable shared pattern of cue effects on these avatars, at a peak frame. It is not evidence of the same strategy: in recovery, an observer with 16 of 25 action families swapped in pairs could still clear .30 (Methods). It is human-level only if the interval reaches the agreement between the two human groups on the same task, .780 for basic and .732 for conversational labels; [it does / it does not]. Most action families on these renders are not confirmed by a detector, so the pattern is one of rendered recipes, each of which may not produce the movement it names.

**Not aligned on these renders.** [Model] responded to the rendered actions (signal *P* [value]), so the cues reached it, yet the pattern of its cue effects stayed below .30 even at the upper end of its interval after correction for its own reliability ([value]). Its cue effects are not aligned with people's on these renders. Three explanations remain open, and A4 cannot separate them: the domain of the avatars, the static peak frame against 1.25 s of human-viewed motion, and the linear mapping of intensity to amplitude. The result says nothing about real faces.

**Inconclusive.** A4 did not decide for [model] on [dataset or task], because [the gate or test that failed, for example no signal, a failed V-gate, or an interval spanning both floors]. An Inconclusive is not a null. Reaching a decision would need [a render that passes the failing gate, or the original stimuli].

**Mixed (WC and EA disagree).** [Model] was [class] on the Western Caucasian set and [class] on the East Asian set for [task], so its headline is Mixed. The two sets share neither observers nor most configurations, but they are not independent tests of a model, and we draw no claim about culture from the difference: the East Asian faces were not based on East Asian people.

**Natural clips beside A4.** With natural clips positive for both eligible models, the informative reading is a positive natural-clip score beside Not aligned on these renders (scenario S3). That pair fits the identifiability result and is the empirical case it predicts can occur. It cannot exclude the avatar domain as the reason. A positive natural-clip score beside Aligned is consistent all the way down to the renders, and still says nothing about strategy.

**Behaviour beside cue effects (Section 6).** If behaviour and cue effects disagree on the same stimuli, that is the case the theory needs, observed: [describe]. If they agree, the proposition still stands as a limit of logic, and these models on these stimuli simply did not show the gap. Agreement is not a failure of the theory or proof of cue use.

## A second human sample (A9)

Rule and wording fixed before any model data (`research/PREREG_A9_DRAFT.md`, "Relation to A4" and block `replication`). Comparator: the same model's A4 decision on basic WC; A4's basic headline beside it. A9 is a second sample on the same platform (lab, generator, protocol, culture) and the same renders, not a second method, and is never called independent of the GFG platform. Never "A9 confirms A4" or "A9 overturns A4".

- **Both Aligned.** The basic-label alignment for [model] replicated on a second human sample (same lab and generator), Liu et al.'s 20 observers, with residual alignment [value] ([CI]).
- **Both Not aligned on these renders.** For [model], the Not aligned result replicated on Liu et al.'s observers.
- **One of each.** For [model] the result did not replicate: A4 basic WC was [class], A9 was [class]. Both are reported and no pooled claim is made. The two samples differ in observers and in configurations (Liu's faces carry 1 to 6 actions).
- **Either Inconclusive, or A4 basic WC not run for the model.** Replication not testable for [model], because [gate or test].

An A9 Inconclusive is never a failure to replicate. A9 cannot change an A4 verdict.

## RQ2: stimulus interventions (A2)

Rule: `research/PREREG_A2_DRAFT.md:188-206`, the deviation at `:332-353`, the .01 reading (DECISIONS 2026-11-08) and the pre-data rulings of 2026-11-12 (the fill rule also governs Opposes). BASIC ONLY scope on every result. RQ2 here is A2's stimulus arm, which says nothing on necessity or sufficiency of an AU, and nothing about how people would respond to these edits, which nobody has measured.

- **Most predictions Match, Gate B passed (R2-a).** On basic labels, masking regions, filtering spatial frequencies and changing frame order moved [model]'s option-free judgement in the direction human studies predict for [n] of 16 contrasts on [route]. [n] of these also hold at *P* < .01, the level that keeps the familywise error under .05 under the skewed nulls we simulated (Methods). The predictions come from other stimuli, so a Match is agreement with a stated prediction, not with human data on these clips. A Match on P12 or P13 is a match to a weakly sourced prediction, against which the closest human test found no cost of shuffling.
- **Mixed classes (R2-b).** Report by cue type. [Regions] matched, [frequency bands] [class], [temporal order] [class]. Say which hold at *P* < .01.
- **Several Oppose (R2-c).** For [contrasts], the edits moved [model]'s judgement against the stated prediction, with the other two fills agreeing in sign with the neutral fill, as the fill rule requires. An Opposes on P12 or P13 says nothing against human alignment, since the closest human test found no cost of shuffling.
- **Mostly Inconclusive (R2-d).** Most contrasts were not decided. Eight of 16 contrasts, and 10 of 16 on Qwen3-VL, cannot reach Null by design, because their predictions are capped; we give their 90% equivalence intervals ([range]) and do not call them nulls.
- **Null on eligible contrasts (R2-e).** For [contrasts] on [model and route], the edit left the option-free judgement unchanged within the margin. Null is hard to reach under A2's rules, so this is a real result, for that cue, model and route only.
- **Gate B fails (R2-f).** For [model and route], the option-free readout carried no usable face signal (failed [B-check]), so no RQ2 claim is made there. This is also a finding for RQ4 (A5).
- **No model eligible (R2-g).** Report the gates and stop; RQ2 is not tested.

## RQ4: prior and evidence (A5)

Rule: `research/PREREG_A5_DRAFT.md:226-231`, with balance descriptive (`:574-576`), and the pre-data rulings of 2026-11-12 (input errors void a run; Absent reported with E*'s interval, "below the margin" where it excludes 0). A5 is a reanalysis of A2's cells, so it exists only where A2 ran.

- **Shown.** On [route], evidence from the face moved [model]'s option-free judgement towards the portrayed word, with Gate B passed, so the judgement is not word knowledge alone. Balance between prior and evidence ([class]) and the share carried by words ([value]) are description, not claims.
- **Unmapped.** Portrayals moved [model]'s judgement consistently, but not towards the acted word. [If only B3 failed: the shift is too small on the image scale to be called mapped.]
- **Absent.** On [route], we found no face evidence in [model]'s option-free judgement, with E* at [value] ([interval]), even though face information was shown to reach the output. The word-knowledge reading needs Absent together with Default balance, and balance is descriptive here, so we state it as absent evidence with the balance beside it ([class]).
- **Unresolved.** Not decided, [reason code].
- **A2 not run for a model and route.** A5 has nothing to classify there; not tested.

## Larger models (A8)

Rule: `research/PREREG_A8_DRAFT.md:678-707`. A8 rows sit under the frozen rows, may support a statement of scope only, and cannot change a frozen verdict. Families are A8's own, smaller than the frozen ones, which the text must say.

- **A8 model agrees with its sibling.** On [test], [larger model] reached the same class as [small sibling], which extends that result's scope within the family to [size]. It is not a new test of the claim.
- **A8 model differs from its sibling.** On [test], [larger model] was [class] where [sibling] was [class]. Both are reported and the difference is described, not tested. It bounds the frozen result's scope to the model size tested.
- **Not run.** [Step] was not run on [model] because [the A8 stop rule, a failed retry, storage, or deferral by a dated note], and nothing is inferred from that.

## RQ3: the route through the model

A scoped limit in every branch (DECISIONS 2026-11-07). Stage 0 is locked (RESULTS_REGISTRY, "A7 Stage 0", 2026-11-10). Drafted text:

On both eligible models the only complete cut of the vision path was the whole path, at its exit into the language model, and the contrast of a whole-path cut is zero by construction. No site inside the language model was shown to be a complete cut, so whether a cue's effect is carried through the language model is not testable here beyond the interface. The tokens at a cue's location did not hold its effect on Gemma 4, where it leaked past them, and could not be located reliably on Qwen3-VL, so cue-level mediation is not testable by a token cut on either model.

Then one sentence for the route arm, a secondary split of the complete exit cut reported in the supplement, which has no human reading and never answers RQ3:

- **Carried by one block, or Both blocks with its rider.** On this edit and readout, [model]'s change was carried by [block], [rider]. This is about which tensor carries the effect of one eyes-and-mouth edit, not about how the model reads faces.
- **No effect to carry.** The joint eyes-and-mouth edit did not move [model]'s option-free judgement enough to trace; a statement about the edit.
- **Not tested.** The route arm's preconditions (Gate B, the sham or the gates) failed for [model], applied by code as recorded before any A2 output existed.
- **Inconclusive.** [reason code].

Never "the model does not use route R" without "on this edit"; never "blocking route R"; no depth statement about the language model.

## RQ5

Not tested. A6 did not freeze: its own pre-freeze check on the first frame, fixed before any model output, failed on one of three detectors (GraphAU AU10 ratio .80 against .50; DECISIONS 2026-11-10). Named as future work, with the fill design as the thing to change: a fill taken from a neutral clip of the same actor rather than the first frame, in a new amendment frozen before any data.

## What would settle it

Original stimuli; new human data on the same edited clips, offered as future work with no claim about the result; playback speed once a route delivers it (outline.md, Discussion item 6).

## A methods recommendation (benchmark audit)

<!-- Added 2026-11-09. Fixed whatever the results. Source: research/BENCHMARK_LEAK_AUDIT.md and research/reproduction/benchmark_leak_audit.json (exploratory, post hoc, annotation text only, no model run); DECISIONS 2026-11-09, "benchmark leak audit is a methods note, not a headline". Not a headline and not a field-level claim. -->

Our own task let option position and option sets identify the answer. To see whether public benchmarks share that problem, we read the annotation files of 11 multiple-choice affect benchmarks for vision-language and multimodal models, without running any model. This audit was exploratory and done after the fact. None of the 11 combined a constant answer position with one option set per label, in the order its own evaluation code presents items. So our confound is not shown to be common, and we do not claim it is. Two partial cases are worth naming. In HumanVBench's released files, most items have four options, and whenever an item has five the correct answer is the fifth: in 61 of 62 such items across three emotion tasks (Holm-corrected *P* .011 to .039 for answer position) [63]. We did not find its evaluation code, so whether options are shuffled when the benchmark is run is unknown. Face-Human-Bench's raw files store the correct answer first in all 50 items of each expression file, but its official evaluation script rotates the options, and replaying that rotation gives balanced positions [64]. Results from its own pipeline are therefore not affected by position. A harness that read the raw files without that step would be; we did not check whether any does. Weaker text-only shortcuts were more common. Where every item shows the same label list, answering the majority class beat the nominal 1/*k* chance by 9 to 32 percentage points in several benchmarks, because the source datasets are imbalanced. Our predictors were simple and used no question text, so these accuracies are lower bounds on what a language model could exploit (`BENCHMARK_LEAK_AUDIT.md`, Results and Limitations). The lesson we draw is narrow. Papers that compare models or people on such tests should report text-only baselines, for answer position, option set, answer prior and majority class, beside 1/*k*, and should remove rotated copies of an item before any leave-one-out check. This is not a claim that affect benchmarks are invalid, and several of the subsets we checked were balanced.

## Limitations

Each item is a disclosed deviation or limit, with its DECISIONS date. Order by weight at step 4.3. Items marked (methods) are already disclosed in methods.md.

1. **v1 task withdrawn** (2026-11-04, "Prompt confound"). The target was first on all items and option sets identified the category. Its accuracy and probe results are not evidence. The v1 gate also hit a Holm floor and was not retuned (`research/PREREGISTRATION.md`, deviation 2026-11-02). (methods)
2. **Composite-grid input limits A1** (2026-11-04, "Input limits on A1"). About 1 token across the face on Gemma 4; 68 of 243 clips lose frames; 16:9 frames squashed. A model that fails A1 on the grid has not shown that it cannot use faces. (methods)
3. **Decoder fault, A1 rerun** (2026-11-04, "Decoder fault"; amendment A1.1). All earlier image-dependent results excluded. (methods)
4. **Partial rotation files opened by a loader check** (2026-11-04, "RQ1 same-clip analysis moves to AU families"). No model values saved, but not a clean separation. (methods)
5. **Smoke run before A2's predictions were committed** (2026-11-04, "A2 is a draft"). A2 repeats its analysis without those items. (methods)
6. **Natural-clip unit changed, cells exploratory, strength rule added late** (2026-11-04: "moves to AU families", "Second attempt fails", "RQ1 inference after two failed calibration attempts", "Strength rule"). The strength rule was set after power at 0.5, 1 and 2 times was known, and can only downgrade. Treated as a deviation; the 0.75 times run passed, so the verdict is the same as a 1 times reading alone (outline.md, inconsistency 23). (methods)
7. **Detectors share AU training data; rule 8 unmet for AU15 and AU20** (2026-11-04, "Family recovery result" and "Family recovery criteria"). (methods)
8. **Gate 1 read as the median; clips under 8 tokens are a sensitivity** (2026-11-04, "Gate 1 is read as the median").
9. **Molmo2 out of A2 and A4's family** (2026-11-04, "Vision gates G2 to G4, Gemma and Molmo"). Failed option independence at bfloat16 and token swap on the flip rate alone. The G4 thresholds were set by the implementer. (methods)
10. **Qwen3-VL gates at bfloat16 only** (2026-11-05, "Vision gates G2 to G4, Qwen"). float32 ran out of memory. (methods)
11. **Gemma 4 layers 24 to 41 reuse keys and values** (2026-11-04, vision gates entry). Patches there are false nulls. Bears on any cut claim, not on the reported tests.
12. **Natural-clip reading under each RQ0 outcome added before A1 was read** (2026-11-07). The frozen A3 table overrides RQ0's text. (methods)
13. **A2 cannot reach Null on 8 of 16 contrasts, 10 of 16 on Qwen3-VL** (2026-11-07, "A2 frozen before A1 is read"). (methods)
14. **A2 source support weakened after the freeze** (2026-11-07, "Source check after the A2 freeze"). P7, P12 and P13 chiefly; contrary evidence for P12 and P13. Predictions unchanged. (methods)
15. **A2 test liberal under one skewed null; .01 sensitivity** (2026-11-08). Verdicts stay primary; each also reported at *P* < .01. Whether real portrayal gain is skewed this way is unknown. (methods)
16. **A2, A4 and A5 pre-data rulings after independent code audits** (2026-11-12). No code or hash changed; each ruling reads the frozen output more strictly or adds a declared calculation (A2's fill rule also on Opposes; A4's Holm family fixed at two models and non-finite rows unusable; Aligned and Not aligned both met read as Inconclusive; input errors void a run). Made before any of their data existed.
17. **A4 frozen with two exempt accuracy cells and a thin V1 pass** (2026-11-07, "Freeze A4 before A1 is read"). No accuracy claim for residual CEA on weak conversational effects; most families unconfirmed by a detector. (methods)
18. **A4 ceiling uses a related estimator** (2026-11-08; quantified by the 2026-11-12 audit at .0003 or less on the basic ceilings). Corrected values are approximate; decisions use uncorrected values. (methods)
19. **A4 domain limits** (A4 frozen text, `research/PREREG_A4_DRAFT.md:221-250`). Avatars, static peak frames against 1.25 s of motion, linear amplitude mapping, AU22 and AU27 approximated, four faces per set, the EA faces not based on East Asian people. (methods)
20. **A9's narrowing was chosen after a recovery run was seen** (lead decision 8, 2026-11-09). The replacement keeps every decision confirmatory and withdraws the weak-effect accuracy claim under correlated face noise; it is more favourable to confirmatory decisions than the rule it replaced, whose reading is reported beside each decision. A9 shares lab, generator, protocol and culture with A4.
21. **G7 is a scoped limit** (2026-11-07). The literal route test is vacuous; Stage 0 (2026-11-10) found the exit cut complete and no complete cut inside the language model, so depth is not testable, and the region test failed on both models. The route arm is a secondary split, submitted before A2's output existed with its preconditions applied by code (2026-11-12). (methods)
22. **A5 balance is descriptive** (2026-11-08, two entries). `min_balance_cells` of 4; balance verdicts left the confirmatory claims after run 6; five recovery failures stay on record. Only the evidence classes are confirmatory.
23. **A6 closed as a limit** (2026-11-10). Its first-frame AU check blocked the freeze on GraphAU AU10 (.80 against .50), so RQ5 was not run. Earlier, A6 had given up its cue-absent nulls (2026-11-08).
24. **Full forwards on all models** (2026-11-04). One scoring path; bfloat16 rounding between paths shown, so cached scoring was dropped. Not a limit on results, but a design choice to state.
25. **Data and scope limits** (methods.md, Human data). Human predictions come from other stimuli; about 8 of 21 labels overlap the Snoek sets; 17 actors only partly crossed with label; option sets still identify category on the natural task; author-set thresholds; open models only. Seven workbook items have no edited clip in the collaborators' working set and are left out (DECISIONS 2026-11-12; IDs in `eu_missing_items_check.json`).
26. **EU-Emotion permissions** (2026-11-08, "EU-Emotion permissions settled"). Norms public and in-memory edits allowed; videos withheld by choice. A data statement, not a limit.
