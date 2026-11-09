<!--
Draft v2 scaffold, 2026-11-08 (ROADMAP step 0.8). An outline, not conclusions. No A1, A2 or A4 model result has been read.
For each main question, one short paragraph per outcome class allowed by its frozen decision rule. The branch that matches the locked results is picked at ROADMAP step 4.3; the others are deleted.
Classes are taken from the amendments and DECISIONS, with no class added. The Discussion carries no subheadings in the journal; the headings here are for drafting only.
-->

# Discussion (outline)

## Opening paragraph

One paragraph from the scenario table in outline.md, "How the whole paper reads" (S1 to S6). [PENDING: scenario, once RQ0, RQ1 and RQ2 are locked.] The audit, the identifiability results and the estimand stand in every scenario.

Fixed whatever the results: the v1 task put the target first on every item and gave each label one option set (Results, Section 1). Behaviour on natural items cannot settle cue use (Results, Section 3). These are locked and can be discussed now.

## RQ0: dependence on the face

A3 passed on all three models (RESULTS_REGISTRY, A3), so only the A1 column varies. Rule: `research/PREREGISTRATION.md:137` and `research/PREREG_A3_DRAFT.md:44-49`.

- **If A1 passes on Gemma 4 and Qwen3-VL (R0-a).** Accuracy needs the matching face on both inputs. Say how much survives the face swap, and compare it with accuracy against 1/6. Stress that this tests dependence on the image, not competence. Natural-clip RQ1 is confirmatory for both, subject to the clip rule.
- **If A1 passes on one and fails on the other (R0-b).** The models differ. The failing model showed face dependence only on a face crop. Its natural-clip RQ1 is descriptive, and its A2 runs on `single_face` only.
- **If A1 fails on both (R0-c).** The composite input limits face use, and accuracy on that format cannot be read as face reading. This becomes the audit's main finding (scenario S4). Natural-clip RQ1 is descriptive for both, and temporal predictions P12 to P15 have no route.
- **If A1 is not complete (R0-d).** Nothing can be stated, because Holm covers all three models. No submission on this branch.
- **Molmo2, in every branch.** Reported in RQ0. It stays out of A2 and of A4's Holm family on its gate failures. Its A3 delta rests on a small share of cells, since it chose "None of the above" on most of them (RESULTS_REGISTRY, A3 trace).

## RQ1 on natural clips

Rule: DECISIONS 2026-11-04 (`research/DECISIONS.md:232-282`) and 2026-11-07 (`research/DECISIONS.md:327-332`).

- **If the pattern score is positive and confirmatory.** The model is aligned with the human pattern on these clips. It does not show that it uses the same cues (Proposition 4). Point to Section 5 for the cue-level test.
- **If the score is null.** Alignment is not shown. It is not shown absent: power was .775 at half the human effect, and an observer sharing half the human weights would often be missed (RESULTS_REGISTRY, pattern score on the model-frames route).
- **If more than 5% of clips are lost, or RQ0 makes it descriptive.** Report the score with no claim, and say which rule applied.
- **If the model failed both A1 and A3.** Cannot occur, because A3 passed on all three models. Delete at 4.3.

## RQ1 under randomisation (A4)

Rule: `research/PREREG_A4_DRAFT.md:121-137`. Each dataset alone, then one headline per model per task.

- **If Aligned.** A reliable shared pattern of cue effects on these avatars, on a peak frame. Not the same strategy. Human-level only if the interval reaches the residual WC to EA reference, .780 for basic or .732 for conversational (RESULTS_REGISTRY, A4 human ceiling). Note that most families are not confirmed by a detector.
- **If Not aligned on these renders.** The model responds to the AUs, yet its cue pattern stays below the floor after disattenuation. Always in those words. Domain, the static route and the amplitude mapping remain open. It says nothing about real faces.
- **If Inconclusive.** Name the gate or test that failed. Say what would be needed: a validated render, or the original stimuli. The paper then rests on the audit, the theory and RQ2 (scenario S5).
- **If Mixed (WC and EA disagree).** Report each dataset and the Mixed headline. Adds no scenario. WC and EA are not independent tests of a model.
- **Combined with natural clips.** Use outline.md's table "Natural clips by A4". The most informative case is a positive natural-clip score with Not aligned on these renders (scenario S3): it fits the identifiability result, but cannot exclude the avatar domain.
- **Behaviour beside CEA (Section 6).** If they disagree on the same stimuli, that is the empirical case the theory needs. If they agree, the theorem stands as a logical limit, and these models on these stimuli do not show the gap. Agreement is not a failure of the theory or proof of use.

## RQ2: stimulus interventions (A2)

Rule: `research/PREREG_A2_DRAFT.md:188-206`, the deviation at `:332-353`, and the .01 reading (DECISIONS 2026-11-08). BASIC ONLY scope on every result. RQ2 here is A2's stimulus arm, which says nothing on necessity or sufficiency of an AU.

- **If Gate B passes and most predictions Match (R2-a).** Judgement moves with the stated human predictions on basic labels in this stimulus set. Say which verdicts also hold at *P* < .01. No claim that people would respond this way to these edits. A Match on P12 or P13 is a match to a weakly sourced prediction.
- **If classes are mixed (R2-b).** Report by cue type: regions, frequency bands, temporal order.
- **If several Oppose (R2-c).** The model moves against the stated prediction. Check the fill rule and the bridge readout. An Opposes on P12 or P13 says nothing against human alignment, since the closest human test found no cost of shuffling.
- **If most are Inconclusive (R2-d).** Capped contrasts cannot reach Null by design (8 of 16, or 10 of 16 on Qwen3-VL). Give equivalence intervals and do not call them nulls.
- **If Null on eligible contrasts (R2-e).** Hard to reach, so reportable as a real result for that cue, model and route only.
- **If Gate B fails (R2-f).** The option-free readout carries no usable face signal for that model and route. No RQ2 claim there. This is also a finding for RQ4.
- **If no model is eligible (R2-g).** Report the gates and stop. RQ2 is not tested.

## RQ4: prior and evidence (A5)

Rule: `research/PREREG_A5_DRAFT.md:226-231`, with balance descriptive (`:574-576`). A5 is a reanalysis of A2's cells, so it exists only where A2 ran.

- **If Shown.** Face evidence moves the blind judgement towards the portrayed word, with Gate B passed. Balance and word share are given as description, not as a claim.
- **If Unmapped.** Portrayals move the judgement consistently, but not to the acted word. If only B3 failed, the shift is too small on the image scale to be called mapped.
- **If Absent.** No face evidence on this route, with face information shown to reach the output. The word-knowledge reading needs Absent with Default, and the Default part is descriptive, so state it as Absent evidence with the balance beside it.
- **If Unresolved.** Inconclusive, with the reason code.
- **If A2 did not run for a model and route.** A5 has nothing to classify there. Name it as not tested.

## RQ3 and RQ5 (one sentence each in the Discussion)

- **RQ3.** A scoped limit in every branch (DECISIONS 2026-11-07). The literal test cannot fail. Pick the A7 reporting row once Stage 0 exists (`research/PREREG_A7_DRAFT.md:391-398`), or R3-e if it has not run.
- **RQ5.** Only if A6 freezes before its data and finishes. Otherwise named as future work.

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
6. **Natural-clip unit changed, cells exploratory, strength rule added late** (2026-11-04: "moves to AU families", "Second attempt fails", "RQ1 inference after two failed calibration attempts", "Strength rule"). The strength rule was set after power at 0.5, 1 and 2 times was known, and can only downgrade. Whether this counts as a deviation under journal policy is open (outline.md, inconsistency 23). (methods)
7. **Detectors share AU training data; rule 8 unmet for AU15 and AU20** (2026-11-04, "Family recovery result" and "Family recovery criteria"). (methods)
8. **Gate 1 read as the median; clips under 8 tokens are a sensitivity** (2026-11-04, "Gate 1 is read as the median").
9. **Molmo2 out of A2 and A4's family** (2026-11-04, "Vision gates G2 to G4, Gemma and Molmo"). Failed option independence at bfloat16 and token swap on the flip rate alone. The G4 thresholds were set by the implementer. (methods)
10. **Qwen3-VL gates at bfloat16 only** (2026-11-05, "Vision gates G2 to G4, Qwen"). float32 ran out of memory. (methods)
11. **Gemma 4 layers 24 to 41 reuse keys and values** (2026-11-04, vision gates entry). Patches there are false nulls. Bears on any cut claim, not on the reported tests.
12. **Natural-clip reading under each RQ0 outcome added before A1 was read** (2026-11-07). The frozen A3 table overrides RQ0's text. (methods)
13. **A2 cannot reach Null on 8 of 16 contrasts, 10 of 16 on Qwen3-VL** (2026-11-07, "A2 frozen before A1 is read"). (methods)
14. **A2 source support weakened after the freeze** (2026-11-07, "Source check after the A2 freeze"). P7, P12 and P13 chiefly; contrary evidence for P12 and P13. Predictions unchanged. (methods)
15. **A2 test liberal under one skewed null; .01 sensitivity** (2026-11-08). Verdicts stay primary; each also reported at *P* < .01. Whether real portrayal gain is skewed this way is unknown. (methods)
16. **A4 frozen with two exempt accuracy cells and a thin V1 pass** (2026-11-07, "Freeze A4 before A1 is read"). No accuracy claim for residual CEA on weak conversational effects; most families unconfirmed by a detector. (methods)
17. **A4 ceiling uses a related estimator** (2026-11-08). Corrected values are approximate; decisions use uncorrected values. (methods)
18. **A4 domain limits** (A4 frozen text, `research/PREREG_A4_DRAFT.md:221-250`). Avatars, static peak frames against 1.25 s of motion, linear amplitude mapping, AU22 and AU27 approximated, four faces per set, the EA faces not based on East Asian people. (methods)
19. **G7 is a scoped limit** (2026-11-07). The literal route test is vacuous; depth is not testable; Stage 0 has not run. (methods)
20. **A5 balance is descriptive** (2026-11-08, two entries). `min_balance_cells` of 4; balance verdicts left the confirmatory claims after run 6; five recovery failures stay on record. Only the evidence classes are confirmatory. methods.md still calls A5 unfrozen: see the report on proposed changes.
21. **A6 gives up its cue-absent nulls** (2026-11-08). Not this cue and No tested feature read as Inconclusive on the cue question; adjacency over all pairs descriptive. A6 is not frozen.
22. **Full forwards on all models** (2026-11-04). One scoring path; bfloat16 rounding between paths shown, so cached scoring was dropped. Not a limit on results, but a design choice to state.
23. **Data and scope limits** (methods.md, Human data). Human predictions come from other stimuli; about 8 of 21 labels overlap the Snoek sets; 17 actors only partly crossed with label; option sets still identify category on the natural task; author-set thresholds; open models only; seven workbook items with no clip and one item with two files (outline.md, inconsistency 16, still open).
24. **EU-Emotion permissions** (2026-11-08, "EU-Emotion permissions settled"). Norms public and in-memory edits allowed; videos withheld by choice. A data statement, not a limit.
