<!--
Draft v2, 2026-11-07. Citation numbers continue from introduction.md; see references.md.
Sources: PREREGISTRATION.md, PREREG_A2_DRAFT.md (frozen), PREREG_A3_DRAFT.md (frozen), PREREG_A4_DRAFT.md (DRAFT, not frozen),
DECISIONS.md, theory/causal_feature_alignment.md and the scripts they name.
Numbers here are design facts only. No A1 result is stated. A3 and gate outcomes are [PENDING] and belong in Results.
Reused text from research/paper/*_v1.md: none.
-->

# Methods

### Preregistration and order of work

Each confirmatory analysis was fixed in a dated amendment before its data were read. Amendments live in the public repository [PENDING: repository archive DOI, and an external registration link if the journal requires one]. Thresholds marked as proposed below were set by the authors and have no external source.

| Amendment | Content | Frozen | Known when frozen |
|---|---|---|---|
| v1 | Original task and hypotheses | 2026-10-30, tag `prereg-v1` | Pilot data. Its tests are not used here (see Disclosures) |
| A1, A1.1 | Rotated task, real versus mismatch, composite input; decoder fix | 2026-11-04, before any rotated data existed (A1) and before any A1 result was read (A1.1) | Nothing from rotated runs |
| A3 | A1's design on a single face crop | 2026-11-04, before any scored cell and before any rotation result was read | Nothing from rotated runs |
| RQ1 natural-clip rules | Unit, null, strength rule, clip rule | 2026-11-04 (DECISIONS log), before any model matrix existed | Human matrices and recovery results |
| A2 | Blind readout gate and stimulus interventions | 2026-11-07, tag `a2-freeze`, before any A1 result was read and before any full intervention run | A3 result, vision gate results, CFA spectra. The draft predates the A3 result; two audits' corrections do not |
| A4 | Randomised cue test on rendered faces | [PENDING: A4 freeze date, tag and design hash]. Drafted 2026-11-07, before any A1 result was read and before any model saw a render | A3 result, vision gate results, human Snoek effects |

[PENDING: commit hashes for the A1 and A3 freezes; neither has a tag.] [PENDING: date A1 was read.]

**Disclosures.**

1. *Withdrawn task.* An audit on 2026-11-04 found two defects in the v1 task (see Validity audit). Its accuracy and activation results are not reported as evidence.
2. *Decoder fault.* OpenCV 4.13.0.92 returned junk frames on clips flagged interlaced: 49 of 243 clips were unstable between reads in one environment and 51 in the other, against 0 under OpenCV 4.11. All A1 runs were discarded unread and rerun under 4.11. Every job since runs a preflight that stops it on any other decoder. Discarded runs may be read only after the A1 result is locked, as an exploratory measure of the fault. All earlier image-dependent results are excluded from this paper.
3. *Partial files.* While testing its loader on 2026-11-04, one analysis job opened partial rotation files for frame indices and a loader check. No saved output holds model values, but this was not a clean separation.
4. *Smoke runs.* A three-item smoke run on Gemma 4 ran before A2's predictions were committed, which departs from our rule that human targets are fixed first. Only mechanics were read, and no contrast between conditions was computed by label or direction. A2 repeats its analysis without these items. Coverage on the same items was read before the readout ladder ran, although selection rests on separate calibration clips.
5. *RQ1 unit.* The natural-clip analysis moved from single AUs to AU families after recovery failed in every condition. Two attempts at family-wise calibration then failed, and no third was run. Single cells are therefore exploratory. The strength rule below was tightened after power at 0.5, 1 and 2 times the human effect was known.
6. *Gate thresholds.* The size threshold of gate G4 was set by the implementer before any model ran, with no source in the study brief.
7. *Numerical precision.* Qwen3-VL's gates ran at bfloat16 only, because float32 ran out of memory.
8. *Model revisions.* Models were loaded by repository name. [PENDING: confirm the commit hash of each checkpoint used; the repo records none.]

### Models

We tested three open models: Gemma 4 E4B (`google/gemma-4-E4B-it`), Qwen3-VL-8B (`Qwen/Qwen3-VL-8B-Instruct`) and Molmo2-O-7B (`allenai/Molmo2-O-7B`). They ran at bfloat16 on A100 GPUs with full forward passes, so that every model used one scoring path. Molmo2 is in the RQ0 family. It failed gates G2 and G4 and is therefore outside the A2 and A4 families; A4 may report it descriptively. Models are never pooled.

### Human data

No new human data were collected. [PENDING: ethics or exemption statement.]

**EU-Emotion.** We used 243 silent face clips with six response options per item (five named labels and None of the above), from the 249 face clips of the published set [14]. [PENDING: why 6 of the 249 are absent from the local norms.] Clips show 21 labels (20 states plus neutral), acted by 17 face actors. Actor and label are only partly crossed: each actor portrayed 10 emotions plus neutral. Human response distributions are the face-only norms [14]. Human presentation order is not confirmed from the source, so human position effects are unknown. Happy, Afraid and Disgusted came from one set of 8 actors, and Sad, Angry and Surprised from a separate set of 8 or 9. A contrast across those sets is also a contrast between actors, so it is reported only descriptively.

Kind (9 items), Unfriendly (10) and Jealous (7) are flagged as weak labels. Face-only humans chose the target on 23 to 26% of these responses, against chance at 17%. Every result is reported against the acted label and against the human response distribution. Model accuracy far above human on these items is treated as possible non-facial leakage, not as skill. No flagged label is the target of a confirmatory contrast.

**Snoek AU data.** Human cue effects come from public trial-level data [19,25] (CC BY 4.0). Observers judged 1.25 s animations of random AU combinations on generated faces, with a frontal view, flat lighting and a black background. Basic emotions used seven options, including "other", and conversational states used five. Western (WC) and East Asian (EA) groups took part: 60 observers per group for basic emotions (140,910 and 139,728 trials) and 20 per group for conversational states (47,300 and 47,460 trials). The public files hold 33 AU amplitude columns, recoded from 42 raw AUs. They hold no timing parameters, pose or gaze, and no stimuli. Six basic emotions and four conversational states overlap about 8 of the 21 EU-Emotion labels, and include none of Worried, Disappointed, Kind or Unfriendly.

**Calibration clips.** The blind readout was chosen on 24 CREMA-D [26] clips outside any analysis set, with labels and votes unread.

### Validity audit

**Prompt audit.** Using the prompts alone, we recorded the target's position, the number of option sets per label, and how much of the human ambiguity measure (response entropy) a category label predicts. These findings do not depend on decoded pixels. The target was first on 243 of 243 items, and there were 21 option sets for 21 labels. Category alone gave a correlation of .503 with human entropy, cross-validated by actor (`research/reproduction/prompt_confound_audit.json`).

**Literature check.** We tabulated option handling and image-free or mismatch controls in eight emotion papers [1,2,8-13]. Most were read as targeted passages. Search dates and coverage are in the Introduction and references.

**Delivery audit.** For each route and model we logged media token counts, tensor shapes and estimated tokens across the face. Rule: at least 8 tokens across the face (median over clips), a change in output when any sent frame is blanked, logged frame indices equal to the frames sent, and input ids that do not change with frame content. On composite-grid models the grid keeps the first 8 frames, so 68 of 243 clips lose frames at 1 frame per second with a cap of 16, and 16:9 frames are squashed into square cells. On Gemma 4 the grid gives about 1 token across the face, against about 9 for a single face crop. A1 keeps this input as frozen, so a model that fails A1 on a grid has not shown that it cannot use faces. A3 exists for this reason.

**Vision gates.** Three gates checked that image information reaches the output as expected. G2: activations at image positions do not change with the option list. G3: replaying a captured value changes logits by zero, and a full cut reproduces the image-free output. G4: swapping image tokens with a different-label clip at 10, 25, 50 and 100% has a monotonic effect, of at least 1.0 nat at 100% with an answer flip rate of at least .25. Tolerances were fixed before any model ran. [PENDING: gate results.]

### RQ0: does the answer depend on the face?

**Design (A1).** Each of the 243 items ran under five cyclic rotations of its five named labels, with None of the above last, so that each target took each named position once. Each rotation ran twice, on the prompt below. In `real`, the model saw the item's own video. In `mismatch`, it saw the same prompt with a partner item's video, from a different target label and a different actor (fixed partners, seed 42). Models saw composite grids (Gemma 4, Molmo2) or an image list (Qwen3-VL), at 1 frame per second with a cap of 16. Prompts were:

> You are performing a 6-alternative forced-choice mental state recognition task. Analyze video frames only. Choose exactly one label. OPTIONS: [numbered options]. Respond with: EMOTION: <one of the option labels exactly>, REASONING: <brief justification>

Each cell took 20 draws at temperature 1.0, and the modal answer was scored.

**Estimand and decision.** For item *i*, *a_i* is mean correctness over the five real rotations and *b_i* the same over mismatch. The estimand is delta, the mean of *a_i* minus *b_i*: accuracy that requires the matching face. A model is vision-dependent if the one-sided actor-level sign-flip test (100,000 draws) has Holm *P* < .05 across the three models and the actor-bootstrap 95% confidence interval (10,000 draws) excludes 0 from above. A model that fails makes no mental-state claim on that route, whatever its accuracy against 1/6.

**Face-input retest (A3).** A3 repeats this design with one landmark-selected apex frame per clip, face-cropped and sent at 768 by 768 pixels. Scoring uses exact option probabilities: each option's joint log probability after the prefill "EMOTION:", renormalised over the six options. An item is correct on a rotation if the target alone has the highest probability. This is the limit of A1's modal answer as draws increase, so a difference between A1 and A3 could reflect scoring as well as input. A3 is its own Holm family. The prompt still says "video frames" although one image is sent; this was kept identical on purpose.

**Reading.**

| A1 | A3 | Reading |
|---|---|---|
| Pass | Pass | Face use on both inputs |
| Fail | Pass | The grid limited A1; `single_face` is used from here |
| Pass | Fail | A1 evidence needs inspection |
| Fail | Fail | No demonstrated face use; no further claims for that model |

Both tests measure dependence on the image. Neither measures competence. Secondary outcomes are accuracy by target position, correlation with human target probability overall and within category, and A3 repeated without the three weak labels and against the human modal label. The accuracy-against-1/6 *P* value pools cells as independent and is descriptive only. Rotation does not fix the second audit defect: label sets still identify category.

### Blind readout and Gate B

Later tests use a readout that names no label. The user text was "In one plain word, with no formatting or punctuation, what is this person feeling?", with the prefill "They feel". It was chosen on calibration clips by a stated rule: among candidates passing on every model, take the one with the highest lowest median lexicon mass. Each word's joint log probability is read over a 106-word lexicon in 21 families. A family's score is the log sum over its words. The outcome is the centred log-ratio (CLR) of a family: its score minus the mean over the 21 families.

Gate B runs per model and route on all 243 originals. B1: a repeat forward changes log probabilities by at most 1e-3. B2: median lexicon mass is at least .05. B3: mean root-mean-square change in family CLR, real against image-free, is at least 0.25 nats. B4: the acted family's CLR is higher on the item's own face than on a mismatched face (one-sided actor-level sign-flip *P* < .01, and a 95% interval excluding 0 from above). B5: the same test for the acted family's CLR on the item against that family's mean over clips of other labels. Thresholds are author-set values, fixed at the freeze. A model and route that fail Gate B go no further on the blind readout. Lexicon mass alone is not image signal, so B3 to B5 carry the image claim.

### RQ1 on natural clips

This analysis regresses human and model response probabilities on the same measured cues on the same clips. It is observational on both sides. AU intensities came from LibreFace [27] and OpenFace 3.0 [28] on the frames models saw. Four AU families are the cue set: brow raise (AU1, 2), brow lower (AU4, 9), smile (AU6, 12) and mouth open (AU25, 26). A family score is the mean of its members' z-scores, rescaled to unit variance.

One fractional ridge logistic is fitted per option on the clips offering it, with a category intercept and with cues residualised on category and actor. Ridge strength is frozen at the human value, 10. A category intercept absorbs the label prior and the foils offered. The confirmatory statistic is the family pattern score: the Pearson correlation over all cells between the model's and the human's matrix. Its null shuffles cue rows within category for the observer only (4,999 permutations).

Model responses come from a complete real rotation run. A clip needs a usable response on all five rotations. If more than 5% of clips are lost, that model's RQ1 result is descriptive. [PENDING: rule for this analysis when a model fails A1 and passes A3; no document sets one.]

Recovery with synthetic observers (200 observers, 199 shuffles per strength, on this route) gave power of .995 at 0.75 times and 1.0 at 1 times the human effect, and size of at most .06 against a limit of .081. Power was .775 at 0.5 times. For an observer sharing half the human weights, power was .60 at 0.75 times and .83 at 1 times. Size for an observer using different cues was .055 at 0.75 times. A positive score therefore supports "aligned with the human pattern", not "uses the same cues", and a null does not show absence of alignment. Proposition 4 below gives the reason. Single cells are descriptive. Detection is weak for AU15 and AU20, which are outside the four families. [PENDING: training-data lineage of the two detectors.]

### RQ1 on randomised cues (A4)

*A4 is a draft. Every value in this section is provisional until [PENDING: A4 freeze].*

**Stimuli.** A headless renderer built each AU configuration on MakeHuman meshes using FACSHuman targets [29], at 768 by 768 pixels with a frontal view, flat lighting and a black background. This is a new stimulus set of the same AU recipes. It does not reproduce what humans saw: the original generated faces and timing parameters are not public. Avatars are bald and static, and identity coverage is narrower than for humans. Of the 33 columns, 31 map directly to targets, and AU22 and AU27 are approximated. Amplitude maps linearly to target weight, which is an assumption. No model saw a render before the freeze, and no image is published.

The draft manifest holds 1,200 configurations per dataset, chosen by seeded hash of the configuration string with no use of responses, each on one own-group identity; the first 200 on a second identity; each column alone at amplitudes .25, .5 and 1 on six identities, with their neutral faces; 400 configurations at 0.75 and 1.25 times amplitude for the two EA datasets; and 600 for the blind route. That is 8,400 rows and 7,102 unique images. [PENDING: final manifest hash.] The cap of 1,200 is set by cost, and recovery tests whether it is enough.

**Task.** Each model saw one image, with A1's prompt and A3's scoring. There is one fixed option set per task, so options carry no information about the stimulus: six basic labels or four conversational labels, plus None of the above. Human "other" maps to None of the above, and a sensitivity analysis drops that row. All cyclic rotations are used, six for basic and four for conversational. Responses are mean option probabilities over a complete rotation set. A dataset that loses over 5% of stimuli fails. A blind readout on basic labels is secondary and never rescues the main result.

**Human target.** The effect matrix was computed before any render existed. One ridge logistic per subject and label, one-vs-rest, was fitted on the 33 AU amplitudes plus centred face gender, with ridge strength chosen by held-out log loss (0.1 for basic data, 1.0 for conversational data). Subject estimates were averaged, and G-computation gave the difference in probability between do(AU = 1) and do(AU = 0), renormalised over options, on 25 AU families with left and right merged.

**Estimand.** The model is treated as one subject with fractional responses, through the same fit at the human ridge strength. The confirmatory statistic for each model and dataset is CFA, the cosine between model and human effect matrices over options by 25 families. Rows sum to zero, so this equals Pearson *r* over entries. It is reported raw and over the noise ceiling, and the corrected value is never used for a decision. Each human group against the other is the yardstick, because a model cannot be asked to agree with one human group better than the other group does. [PENDING: A4 ceilings and between-group reference; draft values .974 to .990 and .877 (basic), .767 (conversational); source file not yet committed.]

**Validity gate.** None of these checks uses a human target, and a missing check counts as not passed. V1: at least 70% of AU and identity pairs show detected intensity rising across the three doses (AU22 and AU27 cannot be confirmed by the detector). V2: mean root-mean-square change in option log probability, render against no image, of at least 0.25 nats. V3: output is closer for one configuration on two identities than for mismatched configurations (one-sided sign-flip *P* < .01). V4: mean mass on None of the above is at most .90, and on any option at most .95.

**Inference.** Signal: stimuli are shuffled against responses (999 draws), with *P* < .05 required, so that a prior-only system fails here. Alignment: AU labels are shuffled across the 25 columns of the model's matrix (9,999 draws, one-sided), which keeps each cue's profile and any effect common to all cues; Holm correction runs across the two eligible models within a dataset. Interval: stimulus bootstrap, 2,000 draws. Smallest effect of interest: CFA of .30 [PENDING: A4 freeze; proposed value with no external source].

| Outcome | Requires |
|---|---|
| Aligned | All gates; signal *P* < .05; alignment Holm *P* < .05; CFA of at least .30; lower interval bound above 0 |
| Not aligned | All gates; signal *P* < .05; upper interval bound below .30 |
| Inconclusive | Anything else, including any failed or missing gate |

A model with no measurable signal cannot be Not aligned, because nothing shows the cues were delivered. Aligned means a reliable shared pattern, not the same strategy. Each dataset is reported alone. A model is called human-level only if its interval reaches the between-group reference, and that reading is descriptive.

**Behaviour beside CFA.** On the same stimuli we report output similarity and chance-corrected modal agreement, descriptively. Snoek data have no correct answer, so error consistency proper is undefined, and modal agreement is its closest analogue. Whether these metrics disagree with CFA is an empirical result.

**Eligibility.** A4 uses one image, so a Pass on A3 with either outcome on A1 admits a model, subject to gates G2 to G4. Any other outcome leaves it Inconclusive.

**Recovery.** Before any run, synthetic observers answer the real design, with logit noise of SD 1 per stimulus and option and a lognormal gain error of log SD 0.5 per AU. Observers are human-like, half-shared, different-cues, prior-only, generic and pair-swapped. [PENDING: A4 recovery table.]

**Sensitivity and limits.** Sensitivities drop AU22 and AU27, drop the abstain row, rescale amplitude, and use the direct one-AU contrast. A difference from humans may reflect the avatar domain, the static route or the amplitude mapping, because humans saw 1.25 s of movement and their effects average over time course. The comparison covers these 33 columns, and additive effects are assumed on both sides.

### RQ2: stimulus interventions (A2)

**Eligibility.** RQ0 is read through the A1 and A3 table. Pass on both admits both routes. Fail on A1 with Pass on A3 admits `single_face` only, and any other outcome admits neither. G2 to G4 must also pass. Gemma 4 and Qwen3-VL are eligible, subject to RQ0.

**Routes.** Frames were decoded and edited in memory. `single_face` sends the apex frame as a face crop at 768 by 768 pixels, with one fixed square per clip at 1.5 times the largest landmark extent. `video_face` sends 8 uniformly spaced frames through the native video processor, on a synthetic 1 frame-per-second clock, so that timestamps depend only on position in the sent order. On Qwen3-VL, native video merges pairs of frames, so order within a pair is partly lost. Landmarks came from MediaPipe FaceMesh 0.10.14, and the apex frame is chosen from landmark movement alone. A route that fails delivery checks is not analysed. A sham that runs the mask pipeline with the frame's own pixels must equal the original exactly.

**Conditions.** Spatial conditions, on `single_face`: masks of mouth, eyes, brows, and eyes and mouth together, each with three fills (the same actor's neutral clip, the clip's first frame, and a heavy blur of the region); two eye-only views; and low-pass and high-pass filters (half amplitude at 8 and complement at 32 cycles per face width). Eye-aperture warps are run but are out of the confirmatory family, because the dose is unverified. Temporal conditions, on `video_face`: two fixed shuffles, reversal, truncation to the first or last third, and the apex frame repeated. No generator is used, and this suite contains no AU edit, gaze edit, head-angle edit or playback-speed test.

**Estimand.** For item *i* and condition *c*, delta(*i*, *c*) is *y*(*i*, *c*) minus *y*(*i*, original), within item and route. The primary outcome *y* is the CLR of the acted family. Predictions that another label rises use that family's CLR with the acted label left out of the reference mean, since otherwise a plain fall in the target would read as a predicted rise. Effects are averaged per actor. A bridge readout using rotated options never replaces a blind result.

**Inference.** The actor is the unit. We used an exact sign-flip test of actor effects with Westfall-Young step-down maxT [CITE NEEDED: Westfall and Young] within one model and route, at familywise alpha .05, and actor-bootstrap 95% intervals (10,000 draws). With 8 actors a one-sided *P* value cannot go below .0039.

**Predictions.** The contrasts below are the only confirmatory ones. Almost none of the human baselines were measured on EU-Emotion, and none on these edits. All carry a BASIC ONLY scope tag, so they make no claim about complex labels. Most sources were read at abstract level (references.md). Human-source agreement is recorded per prediction in A2.

| ID | Contrast, within item | Predicted direction | Sources |
|---|---|---|---|
| P1 | Mouth mask on Happy | Acted label falls | [30,31] |
| P2 | Mouth mask on Disgusted | Acted label falls | [30] |
| P3 | Eye mask on Afraid | Acted label falls | [30] |
| P4 | Eye mask on Sad | Acted label falls | [30] |
| P5 | Brow mask on Angry | Acted label falls | [32] |
| P6 | Mouth minus eye mask, Happy and Disgusted against Afraid, same 8 actors | More negative on Happy and Disgusted | [30] |
| P7 | Mouth mask on Happy, Sad and Angry, Neutral family | Neutral rises | [33-35] |
| P8 | Mouth mask on Disgusted, Angry family | Angry rises | [33] |
| P9 | Low-pass minus high-pass on Happy | Positive | [36] (use depends on task [37]) |
| P10 | Low-pass minus high-pass on Sad | Negative | [36] |
| P12 | Mean of shuffles on Happy and Surprised | Acted label falls | [38,39] |
| P13 | Mean of shuffles on 48 Low Intensity items | Acted label falls | [38,40,41] |
| P14 | First minus last third, Surprised family on Afraid and Afraid family on Surprised | Confusable label higher on the early third | [42] |
| P15 | First minus last third, Disgusted family on Angry and Angry family on Disgusted | Confusable label higher on the early third | [42] |

P11, on eye aperture [43,44], is outside the family until its dose is verified. These 14 predictions make 16 contrasts. P14 and P15 are analogues of a study on synthetic faces, not replications. Eye-only views have no directional prediction, because accounts conflict [31,45,46]; masks on complex labels are descriptive [46,47]; reversal describes the model's temporal code [48]. No category flip is predicted from head angle, here or later, because we located no human study showing one. A downward tilt raising perceived dominance is the located evidence [49], and this suite does not edit head angle.

**Decision rule.**

| Outcome | Requires |
|---|---|
| Matches prediction | RQ0, G2 to G4, Gate B, route delivery and the sham passed; step-down *P* < .05 in the predicted direction on the blind readout; the first-frame and blur fills share the neutral fill's sign |
| Opposes prediction | The same gates passed; step-down *P* < .05 two-sided in the opposite direction |
| Null | The same gates passed; neither test significant; every null condition below met |
| Inconclusive | Anything else, including a gate with no result |

**Null rule.** A null needs delivery to pass, a positive control showing that the cue type can move this model on this route, a null joint ablation of eyes and mouth where one exists, and equivalence: the 90% actor-bootstrap interval lies inside a margin of 10% of that model's mean absolute CLR change under the full image mismatch of B4 on that route. By design, P5, P6, P9, P10, P14 and P15 cannot reach Null, which is 8 of the 16 contrasts, or 10 of 16 on Qwen3-VL, whose P12 and P13 are also capped because frame order cannot be verified there. Those contrasts read as a match, an opposition or Inconclusive, and A2 cannot show that a model ignores those cues.

**Recovery.** Synthetic observers (prior-only, frame-bag, mouth-only, eyes-only, human-like) ran through the analysis on the real 243 items and actors, 100 replicates each. All five passed the stated rule. Recovery used large effects and independent noise, so it shows that the pipeline works, not its power at realistic effects.

**Sensitivity.** Reported beside the primary analysis: without smoke items, without flagged alignment, on canonical words only, and on clips with at least 8 tokens across the face. Results are also shown for the human modal label and as Jensen-Shannon divergence from the human distribution on the item's options. Humans did not see these edits, so a change in divergence is descriptive and says nothing about what they would do.

### Identifiability results

Full statements and proofs are in [PENDING: Supplementary Note 1]. Numerical checks are in `research/scripts/cfa_identifiability.py`.

**Setup.** A stimulus has cues *z* in R^*p*. A system *S* maps *z* to a response distribution *f_S*(*z*) on *K* options. Natural stimuli have distribution *P*, and a design *Q* gives the stimuli used for measuring effects. For cue *j*, dose *delta_j* and option *k*, the effect matrix is

*tau_S*(*j*, *k*) = E over *z* in *Q* of [*f_S^k*(*z* + *delta_j* *e_j*) minus *f_S^k*(*z*)].

Each row sums to zero. CFA(*S*, *T*) is the cosine between *tau_S* and *tau_T*, and it is undefined if either matrix is zero. A behavioural metric is any functional of the joint distribution of items and responses on *P*. This covers accuracy, confusion matrices, error consistency and representational similarity on outputs. A4 estimates the closely related contrast do(AU = 1) minus do(AU = 0), with other AUs at their observed values.

**Theorem 1.** Let *P* have finite support, take *Q* = *P*, and let the shifted points be distinct and off the support. For any system *T* and any admissible zero-row-sum matrix, there is a system *S* equal to *T* on every natural item whose effect matrix is that matrix. As a corollary, for any reference system *H* with non-zero effects and any *r* in [-1, 1], a system exists that is indistinguishable from *H* on every behavioural metric on those items, with CFA(*S*, *H*) = *r*. The construction changes the system off the natural support and is not smooth. It shows that behaviour imposes no logical constraint. It does not show that trained networks behave this way.

**Proposition 2.** With two standardised cues correlated at rho, a linear observer on each cue gives output correlation rho, yet CFA of 0 at equal doses. As rho approaches 1, behaviour converges while alignment stays at 0.

**Proposition 3.** For linear observers with matched means, equal doses *delta*, and natural cue covariance with extreme eigenvalues lambda_min and lambda_max,

(*delta*^2 / lambda_max) E||*f_S* minus *f_T*||^2 <= ||*tau_S* minus *tau_T*||_F^2 <= (*delta*^2 / lambda_min) E||*f_S* minus *f_T*||^2.

Both bounds are attained. So 1 / lambda_min measures how far behavioural agreement can understate a difference in cue effects. A randomised design pushes the cue covariance towards the identity, and this is why it identifies cue effects.

**Proposition 4.** If a system responds to an unmeasured cue *U* that is correlated with measured cues, a regression on measured cues gives non-zero slopes, while intervening on the measured cues alone has no effect. Regression on natural clips therefore cannot reach a claim about cue use.

**Proposition 5.** With independent zero-mean errors in both matrices, the observed cosine approximates CFA times the square root of the product of their reliabilities, which is Spearman's attenuation correction applied to a matrix [CITE NEEDED: Spearman].

**Cue spectra.** These describe the designs and use no model output. Correlation matrices of each cue table gave:

| Cue table | Rows | Cues | lambda_min | Condition number | 1 / lambda_min |
|---|---|---|---|---|---|
| EU-Emotion, LibreFace | 243 clips | 15 | .054 | 76.1 | 18.4 |
| EU-Emotion, OpenFace 3.0 | 243 clips | 10 | .125 | 22.4 | 8.0 |
| Snoek basic WC | 4,659 configurations | 33 | .178 | 13.7 | 5.6 |
| Snoek basic EA | 2,322 | 33 | .199 | 11.3 | 5.0 |
| Snoek conversational WC | 2,314 | 33 | .209 | 11.0 | 4.8 |
| Snoek conversational EA | 2,322 | 33 | .199 | 11.3 | 5.0 |

With left and right merged, the Snoek designs have 25 cues, lambda_min of .44 to .47, condition numbers of 3.4 to 3.5 and 1 / lambda_min of 2.1 to 2.3. Most of the Snoek spread in the table therefore comes from bilateral pairs, which are correlated by design. Cue counts differ between rows. EU-Emotion cues are detector estimates, so measurement noise changes their spectrum, and clip-level means ignore within-clip variation.

### Statistics and sample size

We ran no power analysis to set sample size, because the stimulus sets are fixed: 243 items and 17 actors for EU-Emotion, and existing human samples for Snoek. A4's stimulus count is set by cost and tested by recovery. Claims stop at these actors, these avatars and these human samples. Tests are one-sided where a direction was stated. Multiplicity is handled by Holm correction across models (A1, A3 and A4, each its own family) and by Westfall-Young step-down within model and route (A2). Seeds are fixed in the amendments. Each output carries a hash of its design, and analysis refuses cells whose hash differs from the code.

### Data availability

Snoek data are public [25]. EU-Emotion stimuli cannot be redistributed, and no clip, frame or edited image is public. [PENDING: confirm that the licence covers in-memory edits for A2; confirm which aggregated norms may be shared.] Renders are not published. [PENDING: data availability statement.]

### Code availability

Analysis code, frozen hashes and derived outputs are in the public repository. [PENDING: DOI-granting code archive; clean-checkout reproduction test.]

### Reporting summary

[PENDING: confirm whether the journal requires a reporting summary for this design.]
