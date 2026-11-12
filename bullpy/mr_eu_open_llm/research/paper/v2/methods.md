<!--
Draft v2, 2026-11-07, updated the same day to the frozen A4. Citation numbers continue from introduction.md; see references.md.
Sources: PREREGISTRATION.md, PREREG_A2_DRAFT.md (frozen), PREREG_A3_DRAFT.md (frozen), PREREG_A4_DRAFT.md (frozen, tag a4-freeze, despite the filename),
PREREG_A7_DRAFT.md (frozen 2026-11-08, used for the scoped limit and its Stage 0 table), DECISIONS.md, theory/causal_feature_alignment.md and the scripts they name.
A5 and A7 were frozen on 2026-11-08. A6 was closed as a limit on 2026-11-10 by its own pre-freeze check and never frozen; none of its thresholds is copied.
Numbers here are design facts only. Results, including the A1, A3, gate and Stage 0 outcomes, are in Results; no A2 or A4 model result exists at writing.
Reused text from research/paper/*_v1.md: none.
Added 2026-11-09: A8 (PREREG_A8_DRAFT.md, FROZEN, tag a8-freeze) and A9 (PREREG_A9_DRAFT.md, frozen 2026-11-09 17:33 UTC, tag a9-freeze; the file's own status line still reads DRAFT, and the tag governs).
A9 values come from research/reproduction/a9_human_target.json, a9_human_ceiling.json, a9_recovery.json (recovery run 4 on the final code, sha256 0ca7bd24..., design hash 21341eb4...) and a9_chen_overlap.json. Earlier recovery runs are kept as a9_recovery_run1_draft.json, a9_recovery_run2.json (design hash b548bc82, the only record of the pre-decision-8 text) and a9_recovery_run3.json.
No A8 output and no A9 model output existed when this was written.
Split 2026-11-09: to fit Nature Human Behaviour Methods length, detail was moved verbatim to supplementary_methods.md (Supplementary Methods, Sections S1 to S11); no content was changed or deleted.
-->

# Methods

### Preregistration and order of work

Each confirmatory analysis was fixed in a dated amendment before its data were read. Amendments live in the public repository [PENDING: repository archive DOI, and a dated link for each amendment. The journal strongly encourages preregistration for this kind of work, and requires a link and date for any reported one. Whether a tagged commit in a public repository counts as a preregistration is not stated by the journal: ask the editors, or deposit the amendments on a registry]. Thresholds marked as proposed below were set by the authors and have no external source.

| Amendment | Content | Frozen | Known when frozen |
|---|---|---|---|
| v1 | Original task and hypotheses | 2026-10-30, tag `prereg-v1` | Pilot data. Its tests are not used here (see Disclosures) |
| A1, A1.1 | Rotated task, real versus mismatch, composite input; decoder fix | 2026-11-04, commits 410da27 (A1) and b3af2cc (A1.1), before any rotated data existed (A1) and before any A1 result was read (A1.1) | Nothing from rotated runs |
| A3 | A1's design on a single face crop | 2026-11-04, commit b6c712d, before any scored cell and before any rotation result was read | Nothing from rotated runs |
| RQ1 natural-clip rules | Unit, null, strength rule, clip rule; reading under each RQ0 outcome | 2026-11-04 (DECISIONS log), before any model matrix existed; RQ0 reading added 2026-11-07, before any A1 result was read | Human matrices and recovery results |
| A2 | Blind readout gate and stimulus interventions | 2026-11-07, tag `a2-freeze` (commit ca38d89), `design_sha256` 0ae22118, before any A1 result was read and before any full intervention run | A3 result, vision gate results, CEA spectra. The draft predates the A3 result; two audits' corrections do not |
| A4 | Randomised cue test on rendered faces | 2026-11-07, tag `a4-freeze` (commit 568b6f5), `design_sha256` 12c454da, pinned manifest 2ef1161e, before any A1 result was read and before any model saw a render | A3 result, vision gate results, human Snoek effects and ceilings, the detector-only V1 pilot, and four recovery runs on synthetic observers, which shaped the choice of statistic |
| A5 | RQ4: prior and evidence on the blind readout | 2026-11-08, tag `a5-freeze` (commit 9b12720), `design_sha256` fd20495b, before any A2 cell existed. Evidence classes confirmatory; balance verdicts descriptive after recovery run 6 | A2 design and recovery, A5 recoveries 1 to 6 and their diagnostics |
| A7 | RQ3: vision-path cuts, Stage 0 mechanics and an optional route arm | 2026-11-08, tag `a7-freeze` (commit 62c6474), design 9e40cfcf, before any full A2 cell was read | A2's design, the vision gates and the cut results on the legacy input. No A2 cell |
| A6 | RQ5: shared errors on CREMA-D | Not frozen. Closed as a limit on 2026-11-10, when its own pre-freeze first-frame check failed; never run on a model | The human CREMA-D target and A6's recovery runs. No model output |
| A8 | Two larger open models, Qwen3-VL-32B and Gemma 4 31B, through the frozen pipelines, with their own families | 2026-11-09, tag `a8-freeze` (commit cbf0826), `design_sha256` 95dd41b5, on the design text, before the Qwen3-VL-8B A1 result was read and before any A8 model ran on a study stimulus. Code pins follow as a dated pre-data step | A3 results of the three first models, vision gates, delivery, the readout ladder, and A2, A4, A5 and A7 as frozen. No A1 result of any model |
| A9 | A4's confirmatory test repeated on a second human sample (Liu et al. 2022, Experiment II) | 2026-11-09 17:33 UTC, tag `a9-freeze` (commit bc41374), `design_sha256` 21341eb4, before any model saw a Liu render | A4's frozen text, code and outputs; the Liu human target and ceiling; recovery on synthetic observers, including the run after which the correlated-noise claim was narrowed (see the A9 section); and the A1 result, computed 37 minutes before the freeze (Disclosure 11). No model had seen a Liu render |

A1 and A3 have no tag, so their freezes are identified by the commits above. No tag was added to them later, since a later tag could be misread as a freeze date. The frozen A1 analysis ran once, on 2026-11-09 at 16:56 UTC, and its result was first read then (RESULTS_REGISTRY, "A1 rotated 6AFC").

**Disclosures.**

1. *Withdrawn task.* An audit on 2026-11-04 found two defects in the v1 task (see Validity audit). Its accuracy and activation results are not reported as evidence.
2. *Decoder fault.* OpenCV 4.13.0.92 returned junk frames on clips flagged interlaced: 49 of 243 clips were unstable between reads in one environment and 51 in the other, against 0 under OpenCV 4.11. All A1 runs were discarded unread and rerun under 4.11. Every job since runs a preflight that stops it on any other decoder. Discarded runs may be read only after the A1 result is locked, as an exploratory measure of the fault. All earlier image-dependent results are excluded from this paper.
3. *Partial files.* While testing its loader on 2026-11-04, one analysis job opened partial rotation files for frame indices and a loader check. No saved output holds model values, but this was not a clean separation.
4. *Smoke runs.* A three-item smoke run on Gemma 4 ran before A2's predictions were committed, which departs from our rule that human targets are fixed first. Only mechanics were read, and no contrast between conditions was computed by label or direction. A2 repeats its analysis without these items. Coverage on the same items was read before the readout ladder ran, although selection rests on separate calibration clips.
5. *RQ1 unit.* The natural-clip analysis moved from single AUs to AU families after recovery failed in every condition. Two attempts at family-wise calibration then failed, and no third was run. Single cells are therefore exploratory. The strength rule below was added after power at 0.5, 1 and 2 times the human effect was known. The earlier rule named no strength, and the added rule can only downgrade the reading at the calibrated effect (DECISIONS 2026-11-04). We treat this as a deviation. The 0.75 times run passed, so the added rule and a reading of the calibrated 1 times row alone give the same verdict, confirmatory, and both calibrations are reported.
6. *Gate thresholds.* The size threshold of the token-swap gate was set by the implementer before any model ran, with no source in the study brief.
7. *Numerical precision.* Qwen3-VL's gates ran at bfloat16 only, because float32 ran out of memory.
8. *Source checks after the A2 freeze.* A full read of the sources on 2026-11-07 weakened the stated support for several A2 predictions, chiefly P7, P12 and P13. The predictions, family and code are unchanged, and the change is logged as a pre-data deviation in A2.
9. *Model revisions.* Models were loaded from local copies of Gemma 4 E4B (commit 83df0a88), Qwen3-VL 8B (0c351dd0) and Molmo2-O 7B (78441065). The A8 models' revisions are in "Larger models (A8)".
10. *A9 narrowing after a recovery run.* A general rule for correlated face noise was stated before the recovery run that tested it: narrow the claim to the conditions that pass. Its first operationalisation, a scope label that would have placed every model outside the validated conditions, was replaced after that run was seen by narrowing by check, which is more favourable to confirmatory decisions. The replaced text was never committed. No threshold moved, and the earlier reading is reported beside each A9 decision (see "Replication on a second human sample (A9)").
11. *A9 frozen after the A1 analysis ran.* The frozen A1 analysis wrote its result at 16:56 UTC on 2026-11-09, and A9 was tagged at 17:33 UTC the same day. A9's last design changes, committed at 17:29 UTC, applied the findings of an independent audit completed at 15:27 UTC, before A1 ran, among them the replication comparator and its fixed wording, and hashing of every input. None of them refers to an A1 value, and A9's eligibility rule is A4's, frozen on 2026-11-07. No model had seen a Liu render.
12. *Pre-data rulings after independent code audits (2026-11-12).* Before any A2, A4 or A5 data existed, an independent audit compared each amendment's frozen code with its text. No code, threshold or hash changed; each ruling reads the frozen output more strictly or adds a declared calculation, and each is appended to its amendment as a dated deviation. A2: the fill rule also governs Opposes (see Decision rule); an unparseable cells line is removed and logged, and its cells are recomputed by resubmitting the shard; model identity is checked from pixel shapes; failed cells are not rerun. A4: the Holm family is fixed at the two named models, so all twelve shards of both must be complete before the single run; rows with non-finite log probabilities are not usable and count toward the 5% loss limit; where Aligned and Not aligned both hold, the result is reported Inconclusive (see the rule table); *R_S* is the Spearman-Brown stepped value, as the code computes it. A5: a model with no A2 cells, or an A1 or A3 gate not read as passed, is an input error that voids the run rather than a result; a route with no usable original cell is Not tested. A9 carries A4's rulings for the A4 functions it imports. Each analysis starts with a pre-run check that records the sha256 of every input. The audits also found reporting limits, disclosed rather than changed: two of A2's descriptive outputs are narrower than its text (Supplementary Methods S8), and A4's V2 for the East Asian datasets is measured on Western renders (Supplementary Methods S6).

### Models

We tested three open models: Gemma 4 E4B (`google/gemma-4-E4B-it`), Qwen3-VL-8B (`Qwen/Qwen3-VL-8B-Instruct`) and Molmo2-O-7B (`allenai/Molmo2-O-7B`). They ran at bfloat16 on A100 GPUs with full forward passes, so that every model used one scoring path. Molmo2 is in the RQ0 family. It failed the option-independence gate at bfloat16, the precision of every run (it passed at float32), and the token-swap gate, and is therefore outside the A2 and A4 families; A4 reports it descriptively if it is run at all. Models are never pooled. Two larger models in the same families were added by A8 (see "Larger models (A8)"). They form their own families and cannot change any verdict on the three models above.

### Human data

No new human data were collected. [PENDING: ethics or exemption statement.]

**EU-Emotion.** We used 243 silent face clips with six response options per item (five named labels and None of the above), drawn from the 249 face items of the published set [14]. Clips show 21 labels (20 states plus neutral), acted by 17 face actors.

Kind (9 items), Unfriendly (10) and Jealous (7) are flagged as weak labels. No flagged label is the target of a confirmatory contrast.

**Snoek AU data.** Human cue effects come from public trial-level data [19,25]; the deposit [25] is CC BY 4.0. Western (WC) and East Asian (EA) groups took part: 60 observers per group for basic emotions (140,910 and 139,728 trials) and 20 per group for conversational states (47,300 and 47,460 trials), counted in the public files with "other" responses kept.

**Calibration clips.** The blind readout was chosen on 24 CREMA-D [26] clips outside any analysis set, with labels and votes unread.

Full details: Supplementary Methods, Section S1.

### Validity audit

**Prompt audit.** Using the prompts alone, we recorded the target's position, the number of option sets per label, and how much of the human ambiguity measure (response entropy) a category label predicts. These findings do not depend on decoded pixels. The target was first on 243 of 243 items, and there were 21 option sets for 21 labels. Category alone gave a correlation of .503 with human entropy, cross-validated by actor (`research/reproduction/prompt_confound_audit.json`).

**Delivery audit.** For each route and model we logged media token counts, tensor shapes and estimated tokens across the face. On Gemma 4 the grid gives about 1 token across the face, against about 9 for a single face crop. A1 keeps this input as frozen, so a model that fails A1 on a grid has not shown that it cannot use faces. A3 exists for this reason.

**Vision gates.** Three gates checked that image information reaches the output as expected. They are labelled G2 to G4 in the amendments, and we name them by function. Tolerances were fixed before any model ran. Their outcomes, which decided eligibility, are in Results, Section 1.

Full details: Supplementary Methods, Section S2.

### RQ0: does the answer depend on the face?

**Design (A1).** Each of the 243 items ran under five cyclic rotations of its five named labels, with None of the above last, so that each target took each named position once. In `real`, the model saw the item's own video. In `mismatch`, it saw the same prompt with a partner item's video, from a different target label and a different actor (fixed partners, seed 42). Models saw composite grids (Gemma 4, Molmo2) or an image list (Qwen3-VL), at 1 frame per second with a cap of 16.

Each cell took 20 draws at temperature 1.0, and the modal answer was scored.

**Estimand and decision.** For item *i*, *a_i* is mean correctness over the five real rotations and *b_i* the same over mismatch. The estimand is delta, the mean of *a_i* minus *b_i*: accuracy that requires the matching face. A model is vision-dependent if the one-sided actor-level sign-flip test [53] (100,000 draws) has Holm [54] *P* < .05 across the three models and the actor-bootstrap 95% confidence interval (10,000 draws) excludes 0 from above. A model that fails makes no mental-state claim on that route, whatever its accuracy against 1/6.

**Face-input retest (A3).** A3 repeats this design with one landmark-selected apex frame per clip, face-cropped and sent at 768 by 768 pixels. Scoring uses exact option probabilities: each option's joint log probability after the prefill "EMOTION:", renormalised over the six options. A3 is its own Holm family.

**Reading.**

| A1 | A3 | Reading |
|---|---|---|
| Pass | Pass | Face use on both inputs |
| Fail | Pass | The grid limited A1; `single_face` is used from here |
| Pass | Fail | A1 evidence needs inspection |
| Fail | Fail | No demonstrated face use; no further claims for that model |

Rotation does not fix the second audit defect: label sets still identify category.

Full details: Supplementary Methods, Section S3.

### Blind readout and Gate B

Later tests use a readout that names no label. The user text was "In one plain word, with no formatting or punctuation, what is this person feeling?", with the prefill "They feel". It was chosen on calibration clips by a stated rule: among candidates passing on every model, take the one with the highest lowest median lexicon mass. Each word's joint log probability is read over a 106-word lexicon in 21 families. A family's score is the log sum over its words. The outcome is the centred log-ratio (CLR) [55] of a family: its score minus the mean over the 21 families.

Gate B runs per model and route on all 243 originals. Thresholds are author-set values, fixed at the freeze. A model and route that fail Gate B go no further on the blind readout. Lexicon mass alone is not image signal, so B3 to B5 carry the image claim.

Full details: Supplementary Methods, Section S4.

### RQ1 on natural clips

This analysis regresses human and model response probabilities on the same measured cues on the same clips. It is observational on both sides. Four AU families are the cue set: brow raise (AU1, 2), brow lower and nose wrinkle (AU4, 9), smile (AU6, 12) and mouth open (AU25, 26).

One fractional ridge logistic is fitted per option on the clips offering it, with a category intercept and with cues residualised on category and actor. The confirmatory statistic is the family pattern score: the Pearson correlation over all cells between the model's and the human's matrix. Its null shuffles cue rows within category for the observer only (4,999 permutations).

Model responses come from a complete real rotation run. If more than 5% of clips are lost, that model's RQ1 result is descriptive. The rule below was set before any A1 result was read (DECISIONS 2026-11-07). The analysis runs once per model in every case, and rows 2 to 4 change only how the score is read.

| A1 | A3 | Natural-clip RQ1 |
|---|---|---|
| Pass | Pass | Confirmatory, subject to the 5% clip rule |
| Fail | Pass | Computed and reported as descriptive. The grid limited that input, so the pattern score cannot carry a claim |
| Pass | Fail | Computed and reported as descriptive, pending the inspection the A3 table requires |
| Fail | Fail | Computed for the record, with no claim. No RQ1 to RQ5 claims for that model |

Recovery with synthetic observers (200 observers, 199 shuffles per strength, on this route) gave power of .995 at 0.75 times and 1.0 at 1 times the human effect, and size of at most .06 against a limit of .081. A positive score therefore supports "aligned with the human pattern", not "uses the same cues", and a null does not show absence of alignment.

Full details: Supplementary Methods, Section S5.

### RQ1 on randomised cues (A4)

A4 was frozen on 2026-11-07, before any A1 result was read and before any model saw a render (see the table above). Rules below follow the frozen text. Thresholds marked proposed were set by the authors and have no external source.

**Stimuli.** A headless renderer built each AU configuration on MakeHuman meshes using FACSHuman targets [29], at 768 by 768 pixels with a frontal view, flat lighting and a black background. This is a new stimulus set of the same AU recipes. It does not reproduce what humans saw: the public deposit [25] holds neither the base faces nor timing parameters, so models see the same AU configurations, not the same stimuli.

The render sets (`main`, `replicate`, `one_au` and `neutral`, `scale_0.75` and `scale_1.25`, and `blind`) are listed in Supplementary Methods, Section S6. That is 15,225 rows and 11,761 unique images (`research/PREREG_A4_DRAFT.md`).

**Task.** Each model saw one image, sent as on `single_face` and scored as in A3, with A1's prompt unchanged. A stimulus without a usable response on every rotation leaves the analysis, and losing over 5% of a dataset's stimuli fails that dataset.

A blind readout on basic labels is secondary, and it never replaces or rescues the main result.

**Human target.** The human matrix is the existing output in `research/reproduction/human_au_effects.json`, computed before any render existed and not refitted for A4. Subject estimates were averaged, and G-computation [56] gave the difference in probability between do(AU = 1) and do(AU = 0), other AUs as observed, renormalised over options, on 25 AU families.

**Estimand.** The statistic is called Cue Effect Alignment (CEA). Amendments A2, A4 and A7, their analysis code and their output files were frozen before the name was changed on 9 November 2026 and call it CFA; the quantity, its computation and every threshold are unchanged. Two statistics are computed for each model and dataset, each with its own null, interval and reliability:

raw CEA = cos(*tau_model*, *tau_human*)

residual CEA = cos(R(*tau_model*), R(*tau_human*))

over options by 25 families. R takes out each cue's component along that matrix's own mean profile across cues.

**Residual CEA is confirmatory, and raw CEA is secondary.** This was fixed before any model data. Raw CEA was tried first and failed recovery on a decision (see Recovery). If residual CEA failed recovery, the analysis would refuse to run.

**Inference.** Signal: stimuli are shuffled against responses (999 draws) on the summed squared OLS AU coefficients, with *P* < .05 required, so that a prior-only system fails here. Alignment: AU labels are shuffled across the 25 columns of the model's matrix (9,999 draws, one-sided on CEA), which keeps each cue's profile and any effect common to all cues. Smallest effect of interest: CEA of .30 (proposed), used as a floor and not a target.

The rule below applies to residual CEA, per model and dataset.

| Outcome | Requires |
|---|---|
| Aligned | All gates; signal *P* < .05; alignment Holm *P* < .05; CEA of at least .30; lower interval bound above 0 |
| Not aligned | All gates; signal *P* < .05; upper interval bound divided by the square root of *R_S* below .30, where *R_S* is the model's split-half reliability across stimuli on that statistic |
| Inconclusive (conflicting interval) | Aligned and Not aligned both met, which needs the point estimate above the bootstrap interval's upper bound; the frozen decision value is shown beside it (ruled before any A4 output, 2026-11-12) |
| Inconclusive | Anything else, including any failed or missing gate |

Not aligned is always written "not aligned on these renders", since most families are unconfirmed by a detector and the avatars are a new stimulus set.

Each dataset is reported alone, then one headline per model per task: Aligned or Not aligned on these renders only if WC and EA agree, Mixed if one of each, and Inconclusive otherwise. Basic and conversational are separate claims and are never pooled. They are not independent tests: within each culture, every configuration of the conversational experiment, coded in the 33 AU amplitudes the analysis uses, also occurs in the basic experiment, the same 2,322 on EA and 2,314 of basic WC's 4,659 on WC, although on WC the configuration identifiers differ [REG: "Rows added 2026-11-12, late", Snoek AU configurations shared]. So a model's basic and conversational results rest largely on the same configurations, read with different option sets against different observers.

**Validity gate.** None of these checks uses a human target. All run per model and task, before any human comparison, and a missing check counts as not passed. Rules are proposed values.

The four checks, V1 (detector), V2 (image dependence), V3 (configuration carries) and V4 (domain limit), and their rules are tabulated in Supplementary Methods, Section S6.

*V1 pilot, before the freeze, with no model.* A face was found on all 432 images, and 113 of 144 covered pairs of column and identity rose, which is .785 against the rule of .70 (`research/reproduction/a4_v1_pilot.csv`). That pass is thin. V1 therefore shows that the renders are monotonic where the detector responds. It does not show that a natural-face detector reads them as those AUs. The threshold was not changed after this was seen.

**Eligibility.** RQ0 is read through the A1 and A3 table on `single_face`, by the rule A2 uses. A Pass on A3, with either outcome on A1, admits a model, subject to the vision gates. Gemma 4 and Qwen3-VL are the Holm family.

**Recovery.** Before any run, synthetic observers answered the real configuration design through the analysis code, at effect sizes of 0.25, 0.5 and 1 times the human log odds, with 100 replicates per cell. Every attempt is kept.

*Exempt cells.* After run 1 and before any model, raw accuracy at conversational 0.25 times was withdrawn as a claim in both conversational datasets. A4 therefore makes no accuracy claim for residual CEA on weak conversational effects, and a decision there rests on the corrected rule.

Raw CEA can be fooled. Residual CEA made no false Aligned above .03. The number of configurations was raised after failures, so the design was tuned on recovery, on synthetic data only.

**Disclosures and limits.** AU22 and AU27 are approximated, yet they hold .109, .119, .183 and .103 of human squared effects on basic WC, basic EA, conversational WC and conversational EA. The common profile holds .26 to .35 of human squared effects, and each statistic must be read against its own reference. The bootstrap is over stimuli only, conditional on the human matrix and on these identities. A difference from humans may reflect the avatar domain, the static route or the amplitude mapping, because humans saw 1.25 s of movement and their effects average over time course. V1 to V4 only partly address this. So this compares a model's reading of a peak frame with human effects averaged over motion. Patterns are compared, not sizes. The comparison covers these 33 columns, and additive effects are assumed on both sides. Models read a text prompt with listed options, which is not how humans were asked. Each set has four faces. On conversational data each human group has 20 subjects, and no claim extends to other populations. The human target averages subject estimates, whereas the framework of [4] warns against averaging over individuals' information-processing methods. That framework also assumes the same stimuli for both systems, so A4 approximates its second degree and does not meet it as framed.

Full details: Supplementary Methods, Section S6.

### Replication on a second human sample (A9)

<!-- Source: research/PREREG_A9_DRAFT.md (DRAFT at writing; status line, Relation to A4, Lead decisions, Human data, AU map, Human target, Recovery, Decision rule). Values from the JSON files named inline. -->

A9 repeats A4's confirmatory test against a second set of human observers. It was frozen on 2026-11-09 (tag `a9-freeze`) before any model saw a Liu render (Disclosure 11 gives the order relative to A1). It cannot change any A4 verdict. It never enters A4's Holm family, headline or decision table, and A9 and A4 estimates are never pooled. If the two disagree for a model, both stand. An A9 result does not make an A4 Inconclusive conclusive. A9 shares the lab, the generator and its AU library, the response protocol and the culture of Snoek et al.'s data, and we never call it independent of that platform.

**What counts as replication, fixed before data.** Each model's A9 decision is compared with the same model's A4 decision on basic WC, because Liu's observers are Western and answered the same seven options; A4's basic headline is reported beside it and is not the comparator. Both Aligned reads "replicated on a second human sample (same lab and generator)". Both Not aligned on these renders reads "the Not aligned result replicated". One Aligned and the other Not aligned reads "did not replicate", and both decisions are reported with no pooled claim. If either is Inconclusive, or A4 basic WC was not run for that model, it reads "replication not testable", naming the gate or test. An A9 Inconclusive is never called a failure to replicate. A4's per-dataset decision cannot be Mixed, so no Mixed case arises (`PREREG_A9_DRAFT.md`, Relation to A4; block `replication`).

**Human data.** Liu et al. [60], Experiment II, from the public deposit (CC BY 4.0). Twenty white Western observers each sorted the same 1,200 random facial animations into happy, surprise, fear, disgust, anger, sad or "other", where "other" included blends [60].

**Human target.** A4's human fit, through the same functions: one ridge logistic per subject and label, one-vs-rest, on the 33 columns, with the intercept as the only unpenalised covariate, because Liu released no face gender.

Liu has one group, so A4's yardstick of one group against the other is replaced by Liu against Snoek basic WC. A model is called human-level on A9 only if its interval reaches .690 on residual CEA, and that too is descriptive.

**Stimuli and design.** All 1,200 configurations are rendered at peak amplitude by A4's renderer on A4's four Western faces, with gender assigned 600 and 600 by a seeded hash. Two designs were fixed before recovery. Design A shows each configuration once, on one face, as in A4. Design B shows each configuration on all four faces, and a configuration's response is the mean of its four faces' rotation-averaged probabilities, fitted with gender 0. The rule was: use A if residual CEA passes recovery under A, otherwise B if it passes under B, otherwise no confirmatory analysis. Design B has 6,200 rows and 5,992 unique images, including 200 replicate configurations for V3 and a `frame_max` set (`PREREG_A9_DRAFT.md`, Stimuli and render manifest; manifest sha256 4b2f04ae).

**Task, statistic and decision.** Unchanged from A4 basic: one image on `single_face`, the same prompt and seven options, all six rotations, exact option probabilities. Residual CEA is confirmatory and raw CEA secondary, through A4's frozen estimator at ridge 1.0. The decision rule is A4's, per model: Aligned, Not aligned on these renders, or Inconclusive, with the smallest effect of interest at .30.

**Recovery.** Synthetic observers answered Liu's real design through A4's estimator: A4's eight observer types built from Liu's human matrix, A4's noise and gain error, strengths of 0.25, 0.5 and 1 times, 100 replicates per cell, and A4's pass rule unchanged. To test whether a model's noise is independent across the four faces, which design B relies on, B was also run with noise correlated across faces at rho .25 and .5, set before the run. Design A failed one check (pair-swapped accuracy at 0.25 times); design B passed all 17; under correlated noise B failed the same single check on residual CEA (89% and 87% against 90%) and passed every decision check (Supplementary Methods, Section S7).

Design B was therefore selected, with residual CEA confirmatory (`a9_recovery.json`, `choice`). Only the accuracy of the estimate for a pair-swapped observer on weak effects fell below its bar.

**The narrowing, and when it was decided.** Before the correlated runs, the study lead stated a general rule: if B failed under correlation, the claim would be narrowed to the conditions that pass, with thresholds unchanged. Its first operationalisation, also written before the run, labelled each decision by r_star, the largest correlation at which B passed every check; with r_star = 0, every model would have been labelled outside the validated conditions. After that run was seen, the lead replaced it with narrowing by check, which is more favourable to confirmatory decisions. All A9 decisions stay confirmatory. The claim that residual CEA is estimated accurately on weak effects (0.25 times) is withdrawn under correlated face noise, the same kind of cell A4 withdrew. Beside each decision we report the model's estimated correlation across faces and the reading the replaced rule would have given. The replaced text was never committed; the recovery file of that run, which carries its design hash, is its only record (`PREREG_A9_DRAFT.md`, lead decisions 2 and 8). Two facts bear on correlations above the tested .5, though neither is a tested condition: design A, which approximates the fully correlated limit, passed every decision check, and the failing accuracy check reflects estimates biased low, not wrong decisions. Recovery was rerun on the final code (run 4) with the same verdicts (`a9_recovery.json`).

**Disclosures and limits.** A4's limits apply, with these added. One culture and 20 observers. Four faces against the 1,200 humans saw. The human fit has no gender column. The suffix mapping is inferred. Some stimuli carry more AUs than the article states. The peak frame combines AUs at their own peaks. AU16 is active on only 58 stimuli (`a9_human_target.json`, `family_active_count_min_max`). "Other" included blends in Liu's instructions, which may not be what None of the above means to a model. All of A9's thresholds are A4's, set by the authors with no external source.

Full details: Supplementary Methods, Section S7.

### RQ2: stimulus interventions (A2)

A2 tests a narrower thing, its stimulus arm: whether region masks, spatial-frequency filters and changes to frame order move a model's judgement in the direction of a stated human prediction. RQ2 in this paper means this narrower test. It cannot establish necessity or sufficiency of an AU.

**Estimand.** For item *i* and condition *c*, delta(*i*, *c*) is *y*(*i*, *c*) minus *y*(*i*, original), within item and route. The primary outcome *y* is the CLR of the acted family. Effects are averaged per actor.

**Inference.** The actor is the unit. We used an exact sign-flip test of actor effects with Westfall-Young step-down maxT [52] within one model and route, at familywise alpha .05, and actor-bootstrap 95% intervals (10,000 draws). A check run after the freeze, and before any A2 cell, found the test exact without per-actor gain and under symmetric gain-scaled noise, but liberal when actor effects average zero yet are skewed, with the long tail opposite the prediction: familywise error per route reached .09 to .10 at portrayal-gain SD .3 and .13 to .14 at SD .5 (`research/reproduction/a2_gain_skew_sim.json`). The verdicts stand as frozen. As a sensitivity fixed before the run, each Matches prediction or Opposes verdict is also reported against step-down *P* < .01, which kept familywise error of the one-sided test at or under .046 in every simulated case. The two-sided test behind Opposes was not simulated; it takes absolute *t* and so always sees the liberal tail, and the same .01 reading is applied to it by analogy (DECISIONS 2026-11-08).

**Predictions.** The contrasts in Supplementary Methods, Section S8, are the only confirmatory ones. That section also gives the sources checked for each prediction. All carry a BASIC ONLY scope tag, so they make no claim about complex labels. These 14 predictions make 16 contrasts. These source checks came after the freeze. They change the stated support and not the predictions (see Disclosures).

**Decision rule.**

| Outcome | Requires |
|---|---|
| Matches prediction | RQ0, the vision gates, Gate B, route delivery and the sham passed; step-down *P* < .05 in the predicted direction on the blind readout; the first-frame and blur fills share the neutral fill's sign |
| Opposes prediction | The same gates passed; step-down *P* < .05 two-sided in the opposite direction; for a masked cue, the first-frame and blur fills share the neutral fill's sign (ruled before any A2 cell existed, 2026-11-12) |
| Null | The same gates passed; neither test significant; every null condition below met |
| Inconclusive | Anything else, including a gate with no result |

**Null rule.** A null needs delivery to pass, a positive control showing that the cue type can move this model on this route, a null joint ablation of eyes and mouth where one exists, and equivalence: the 90% actor-bootstrap interval lies inside a margin of 10% of that model's mean absolute CLR change under the full image mismatch of B4 on that route. By design, P5, P6, P9, P10, P14 and P15 cannot reach Null, which is 8 of the 16 contrasts, or 10 of 16 on Qwen3-VL, whose P12 and P13 are also capped because frame order cannot be verified there.

Full details: Supplementary Methods, Section S8.

### Larger models (A8)

<!-- Source: research/PREREG_A8_DRAFT.md (FROZEN 2026-11-09, tag a8-freeze; Models, Storage, Design block, Eligibility, Statistics, Compute "A1 sharding", Freeze, Limits); DECISIONS 2026-11-09, "A8 frozen". -->

The two eligible models are small, and both were weak on the A3 task (Results, Section 2), so confirmatory claims rest on two weak models. A8 asks whether the frozen tests give the same answers on larger, newer open models in the same two families. It was frozen on 9 November 2026, tag `a8-freeze`, on the design text, before the Qwen3-VL-8B A1 result was read and before any A8 model ran on any study stimulus. No A1 output of any model had been read when the tag was made. A8 changes no frozen threshold, family, estimand, prompt, readout, item set, code file or verdict.

**Models.** Qwen3-VL-32B (`Qwen/Qwen3-VL-32B-Instruct`, revision 0cfaf481, 33.36 B parameters) and Gemma 4 31B (`google/gemma-4-31B-it`, revision 842da379, 31.27 B parameters), both Apache-2.0 and not gated (`PREREG_A8_DRAFT.md`, Models).

**Sharded A1, with an equivalence test.** At 32 B, one A1 task is projected at about 29 GPU-hours against a 12-hour limit, and the frozen evaluation script cannot resume (`PREREG_A8_DRAFT.md`, Compute). On the GPU, the frozen command runs twice on Qwen3-VL-8B (rotation 0, 6 items, both conditions), and the shard runner in 2 shards must then match it record for record. If the two frozen runs differ, sharding is not shown, and A1 on A8 models waits on a dated ruling made before any A8 A1 output exists. [PENDING: GPU equivalence result and date; date of the A8 code pins.]

**Eligibility, by the frozen rules only.** No new threshold is set. A model runs A1 only if it passes G2 to G4 and A3, as A8's stop rule states.

**Families.** A8 forms its own A3, A1 and A4 Holm families of A8 models only, computed from the A8 output files with no hand-picked list.

**Limits.** A8 tests scale within two families, not generality across families. For Gemma 4, scale and model generation change together. Both models run at bfloat16 only. The float32 checks of G2 and G3 are expected to run out of memory, and the frozen gate script then keeps the bfloat16 verdict, as was accepted for Qwen3-VL-8B partly because its bfloat16 values were exactly zero. If an A8 model shows a non-zero bfloat16 value on G2 or G3, no float32 check can separate rounding from leakage, and we say so for that model.

Full details: Supplementary Methods, Section S9.

### Scoped and provisional questions

**RQ3, reported as a scoped limit.** RQ3 asks whether a cue's effect is carried from image tokens through the language model, and whether blocking that path removes it. We report this as a limit, not as a test (DECISIONS 2026-11-07). The reason is logical and not one of cost. A cut that takes out the whole vision path gives the same output for every image, so a cue effect under it is zero by construction, and the literal test cannot fail.

Amendment A7, frozen on 2026-11-08 (tag `a7-freeze`) before any full A2 cell was read, defines a Stage 0 table on the 24 CREMA-D calibration clips, which are outside the analysis set, on `single_face`, with the blind readout. A site counts as a complete cut only if every clip and every one of the 106 lexicon words stays within 0.001 nats of the image-free reference; a site not run, failed or without output counts as not shown. Stage 0 ran on Gemma 4 and Qwen3-VL on 2026-11-10 (Results, Section 9). Its route arm, a secondary split of the complete exit cut, may run in Stage 1 for a model and route that pass Gate B and the sham; that scope was fixed before any A2 output existed (DECISIONS 2026-11-10), and it is never reported as an RQ3 result. Its verdicts (carried by one block, both blocks, opposed blocks, no effect to carry) are confirmatory for that decomposition. After A7's recovery, and before its freeze, the interaction rider (additive, redundant, synergic) was made descriptive, reported with its recovery rates, because it failed recovery; and no effect to carry stayed confirmatory with its margin unchanged, its recovery reach of .62 to .65 at the null reported beside it, since its misses end Inconclusive rather than as a wrong named verdict (`research/PREREG_A7_DRAFT.md`, Recovery ruling, 2026-11-08). This paper makes no claim about the route a cue takes through a model, and no site is described as a cut unless Stage 0 shows it complete.

**RQ4, frozen, and RQ5, closed as a limit.** A5 (RQ4) was frozen on 2026-11-08 with tag `a5-freeze`, before any A2 cell existed; its evidence classes are confirmatory and its balance verdicts descriptive, because two related false-positive recovery checks failed under skewed portrayal gain (.094 and .072 against .07). A6 (RQ5) was never frozen. Its own pre-freeze rule, fixed before any model output, requires the first frame of each clip to carry at most half of the apex value of AU9 and AU10 on every detector, so that the frame used as a fill does not keep the cue it should hide. One detector failed it (ME-GraphAU AU10, .80 against .50), so A6 did not freeze or run, and RQ5 is left as future work (DECISIONS 2026-11-10). No CREMA-D model output exists.

Full details: Supplementary Methods, Section S10.

### Identifiability results

Statements, proofs and their extension to residual CEA are in Supplementary Note 1, which draws the proofs of Theorem 1 and Propositions 2 to 5 from the theory note (`research/theory/causal_feature_alignment.md`). Numerical checks are in `research/scripts/cfa_identifiability.py`.

**Setup.** A stimulus has cues *z* in R^*p*. A system *S* maps *z* to a response distribution *f_S*(*z*) on *K* options. Natural stimuli have distribution *P*, and a design *Q* gives the stimuli used for measuring effects. For cue *j*, dose *delta_j* and option *k*, the effect matrix is

*tau_S*(*j*, *k*) = E over *z* in *Q* of [*f_S^k*(*z* + *delta_j* *e_j*) minus *f_S^k*(*z*)].

**Theorem 1.** Let *P* have finite support, take *Q* = *P*, and let the shifted points be distinct and off the support. For any system *T* and any admissible zero-row-sum matrix, there is a system *S* equal to *T* on every natural item whose effect matrix is that matrix. As a corollary, for any reference system *H* with non-zero effects and any *r* in [-1, 1], a system exists that is indistinguishable from *H* on every behavioural metric on those items, with CEA(*S*, *H*) = *r*. The construction changes the system off the natural support and is not smooth. It shows that behaviour imposes no logical constraint. It does not show that trained networks behave this way.

**Proposition 3.** For linear observers with matched means, equal doses *delta*, and natural cue covariance with extreme eigenvalues lambda_min and lambda_max,

(*delta*^2 / lambda_max) E||*f_S* minus *f_T*||^2 <= ||*tau_S* minus *tau_T*||_F^2 <= (*delta*^2 / lambda_min) E||*f_S* minus *f_T*||^2.

Both bounds are attained. So 1 / lambda_min measures how far behavioural agreement can understate a difference in cue effects. A randomised design pushes the cue covariance towards the identity, and this is why it identifies cue effects.

Full details: Supplementary Methods, Section S11.

### Statistics and sample size

We ran no power analysis to set sample size, because the stimulus sets are fixed: 243 items and 17 actors for EU-Emotion, and existing human samples for Snoek. A4 uses every available configuration, and recovery tested that count. Claims stop at these actors, these avatars and these human samples. Tests are one-sided where a direction was stated. Multiplicity is handled by Holm correction across models (A1, A3 and A4, each its own family) and by Westfall-Young step-down within model and route (A2). A8 forms separate A1, A3 and A4 families of A8 models only, and A9 is its own Holm family on its one dataset. Neither enters a frozen family. Seeds are fixed in the amendments. Each output carries a hash of its design, and analysis refuses cells whose hash differs from the code.

### Data availability

Snoek data are public [25]. Liu et al.'s Experiment II data are public under CC BY 4.0 [60] and are not copied into the repository. EU-Emotion is a collaborator dataset. Its item-level norms are public in the repository, and in-memory edits as in A2 are permitted (confirmed 2026-11-08). Clips, frames and edited images are not in the public repository, for size; access is through the dataset's authors. Renders are not published, because FACSHuman's targets are LGPL-3.0 with no output clause. The full statement, with licences and access routes, is drafted in `research/paper/v2/submission/data_code_availability.md`; the EU-Emotion access route is the authors' to supply [USER].

### Code availability

Analysis code, frozen hashes and derived outputs are in the public repository. [PENDING: DOI-granting code archive; clean-checkout reproduction test.]

### Reporting summary

A Nature Portfolio reporting summary is required for this journal and is to be supplied at submission. [PENDING: the summary itself.]
