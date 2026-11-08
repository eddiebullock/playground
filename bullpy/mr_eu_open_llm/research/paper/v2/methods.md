<!--
Draft v2, 2026-11-07, updated the same day to the frozen A4. Citation numbers continue from introduction.md; see references.md.
Sources: PREREGISTRATION.md, PREREG_A2_DRAFT.md (frozen), PREREG_A3_DRAFT.md (frozen), PREREG_A4_DRAFT.md (frozen, tag a4-freeze, despite the filename),
PREREG_A7_DRAFT.md (draft, used only for the scoped limit), DECISIONS.md, theory/causal_feature_alignment.md and the scripts they name.
A5 and A6 are unfrozen and changing: they appear at outline level only, and none of their thresholds is copied.
Numbers here are design facts only. No A1 result, A2 result or A4 model result is stated. A3 and gate outcomes are [PENDING] and belong in Results.
Reused text from research/paper/*_v1.md: none.
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
| A2 | Blind readout gate and stimulus interventions | 2026-11-07, tag `a2-freeze` (commit ca38d89), `design_sha256` 0ae22118, before any A1 result was read and before any full intervention run | A3 result, vision gate results, CFA spectra. The draft predates the A3 result; two audits' corrections do not |
| A4 | Randomised cue test on rendered faces | 2026-11-07, tag `a4-freeze` (commit 568b6f5), `design_sha256` 12c454da, pinned manifest 2ef1161e, before any A1 result was read and before any model saw a render | A3 result, vision gate results, human Snoek effects and ceilings, the detector-only V1 pilot, and four recovery runs on synthetic observers, which shaped the choice of statistic |
| A5, A6, A7 | RQ4, RQ5 and RQ3 | Not frozen. Provisional, and not preregistered unless a freeze is added here | See Scoped and provisional questions |

A1 and A3 have no tag, so their freezes are identified by the commits above. [PENDING: dated public links for each amendment, and whether to tag A1 and A3.] [PENDING: date A1 was read.]

**Disclosures.**

1. *Withdrawn task.* An audit on 2026-11-04 found two defects in the v1 task (see Validity audit). Its accuracy and activation results are not reported as evidence.
2. *Decoder fault.* OpenCV 4.13.0.92 returned junk frames on clips flagged interlaced: 49 of 243 clips were unstable between reads in one environment and 51 in the other, against 0 under OpenCV 4.11. All A1 runs were discarded unread and rerun under 4.11. Every job since runs a preflight that stops it on any other decoder. Discarded runs may be read only after the A1 result is locked, as an exploratory measure of the fault. All earlier image-dependent results are excluded from this paper.
3. *Partial files.* While testing its loader on 2026-11-04, one analysis job opened partial rotation files for frame indices and a loader check. No saved output holds model values, but this was not a clean separation.
4. *Smoke runs.* A three-item smoke run on Gemma 4 ran before A2's predictions were committed, which departs from our rule that human targets are fixed first. Only mechanics were read, and no contrast between conditions was computed by label or direction. A2 repeats its analysis without these items. Coverage on the same items was read before the readout ladder ran, although selection rests on separate calibration clips.
5. *RQ1 unit.* The natural-clip analysis moved from single AUs to AU families after recovery failed in every condition. Two attempts at family-wise calibration then failed, and no third was run. Single cells are therefore exploratory. The strength rule below was added after power at 0.5, 1 and 2 times the human effect was known. The earlier rule named no strength, and the added rule can only downgrade the reading at the calibrated effect (DECISIONS 2026-11-04).
6. *Gate thresholds.* The size threshold of the token-swap gate was set by the implementer before any model ran, with no source in the study brief.
7. *Numerical precision.* Qwen3-VL's gates ran at bfloat16 only, because float32 ran out of memory.
8. *Source checks after the A2 freeze.* A full read of the sources on 2026-11-07 weakened the stated support for several A2 predictions, chiefly P7, P12 and P13. The predictions, family and code are unchanged, and the change is logged as a pre-data deviation in A2.
9. *Model revisions.* Models were loaded from local copies of Gemma 4 E4B (commit 83df0a88), Qwen3-VL 8B (0c351dd0) and Molmo2-O 7B (78441065).

### Models

We tested three open models: Gemma 4 E4B (`google/gemma-4-E4B-it`), Qwen3-VL-8B (`Qwen/Qwen3-VL-8B-Instruct`) and Molmo2-O-7B (`allenai/Molmo2-O-7B`). They ran at bfloat16 on A100 GPUs with full forward passes, so that every model used one scoring path. Molmo2 is in the RQ0 family. It failed the option-independence and token-swap gates and is therefore outside the A2 and A4 families; A4 reports it descriptively if it is run at all. Models are never pooled.

### Human data

No new human data were collected. [PENDING: ethics or exemption statement.]

**EU-Emotion.** We used 243 silent face clips with six response options per item (five named labels and None of the above), drawn from the 249 face items of the published set [14]. The item-level workbook we hold has all 249 (`eu_emotions_oreilly_item_level.xlsx`; `n_human_items_available` in `data/eu_emotions_human_entropy.json`). Our clip list is a scan of the edited face folders in the copy we hold (`scripts/build_eu_full_manifest.py`), with no exclusion by content or result. It has no clip for seven workbook items, six of High intensity and one of Low intensity. It has two files for one workbook item, one in each of two packs, and both take that item's norms. The 243 clips therefore map to 242 norm rows, and we have 48 Low intensity items against the published 49. None of the seven carries a weak label (below). [PENDING: whether the seven are absent from the distributed set or only from our copy, and whether the two files for that item are the same video.] Clips show 21 labels (20 states plus neutral), acted by 17 face actors. Actor and label are only partly crossed: each actor portrayed 10 emotions plus neutral. Human response distributions are per-option face-only norms from that workbook (`data/eu_emotions_human_entropy.json`); the published paper reports recognition scores [14]. Response options were described as counterbalanced [14], but the scheme is not given and no results by position are reported, so human position effects are unknown. In our item metadata, Happy, Afraid and Disgusted came from one set of 8 actors, and Sad, Angry and Surprised from a separate set of 8 or 9. A contrast across those sets is also a contrast between actors, so it is reported only descriptively.

Kind (9 items), Unfriendly (10) and Jealous (7) are flagged as weak labels. In our item norms, face-only humans chose the target on 23 to 26% of these responses, against chance at 17%. Every result is reported against the acted label and against the human response distribution. Model accuracy far above human on these items is treated as possible non-facial leakage, not as skill. No flagged label is the target of a confirmatory contrast.

**Snoek AU data.** Human cue effects come from public trial-level data [19,25]; the deposit [25] is CC BY 4.0. Observers judged 1.25 s computer-generated animations of random AU combinations on eight base faces per group, matched to observer ethnicity, with a frontal view, flat lighting and a black background [19]. The public files do not show eight codes in every set: they hold 967 for Western basic, where 15 observers saw blended identities, 8 for East Asian basic, and 4 for each conversational set (`human_au_effects.json`). Within a stimulus set, every observer saw the same fixed stimuli. Basic emotions used seven options, including "other", and conversational states used five. Western (WC) and East Asian (EA) groups took part: 60 observers per group for basic emotions (140,910 and 139,728 trials) and 20 per group for conversational states (47,300 and 47,460 trials), counted in the public files with "other" responses kept. The public files hold 33 AU amplitude columns, recoded from 42 raw AUs. They hold no timing parameters, pose or gaze, and no stimuli. Six basic emotions and four conversational states overlap about 8 of the 21 EU-Emotion labels, and include none of Worried, Disappointed, Kind or Unfriendly.

**Calibration clips.** The blind readout was chosen on 24 CREMA-D [26] clips outside any analysis set, with labels and votes unread.

### Validity audit

**Prompt audit.** Using the prompts alone, we recorded the target's position, the number of option sets per label, and how much of the human ambiguity measure (response entropy) a category label predicts. These findings do not depend on decoded pixels. The target was first on 243 of 243 items, and there were 21 option sets for 21 labels. Category alone gave a correlation of .503 with human entropy, cross-validated by actor (`research/reproduction/prompt_confound_audit.json`).

**Literature check.** We tabulated option handling and image-free or mismatch controls in eight emotion papers [1,2,8-13]. Reading status is in references; some supplements stayed out of reach. Search dates and coverage are in the Introduction and references.

**Delivery audit.** For each route and model we logged media token counts, tensor shapes and estimated tokens across the face. Rule: at least 8 tokens across the face (median over clips), a change in output when any sent frame is blanked, logged frame indices equal to the frames sent, and input ids that do not change with frame content. On composite-grid models the grid keeps the first 8 frames, so 68 of 243 clips lose frames at 1 frame per second with a cap of 16, and 16:9 frames are squashed into square cells. On Gemma 4 the grid gives about 1 token across the face, against about 9 for a single face crop. A1 keeps this input as frozen, so a model that fails A1 on a grid has not shown that it cannot use faces. A3 exists for this reason.

**Vision gates.** Three gates checked that image information reaches the output as expected. They are labelled G2 to G4 in the amendments, and we name them by function. Option independence: activations at image positions do not change with the option list. Replay and cut: replaying a captured value changes logits by zero, and a full cut reproduces the image-free output. Token swap: swapping image tokens with a different-label clip at 10, 25, 50 and 100% has a monotonic effect, of at least 1.0 nat at 100% with an answer flip rate of at least .25. Tolerances were fixed before any model ran. [PENDING: gate results.]

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

This analysis regresses human and model response probabilities on the same measured cues on the same clips. It is observational on both sides. AU intensities came from LibreFace [27], and per-AU outputs recorded as intensity from OpenFace 3.0 [28], on the frames models saw. Four AU families are the cue set: brow raise (AU1, 2), brow lower and nose wrinkle (AU4, 9), smile (AU6, 12) and mouth open (AU25, 26). A family score is the mean of its members' z-scores, rescaled to unit variance.

One fractional ridge logistic is fitted per option on the clips offering it, with a category intercept and with cues residualised on category and actor. Ridge strength is frozen at the human value, 10. A category intercept absorbs the label prior and the foils offered. The confirmatory statistic is the family pattern score: the Pearson correlation over all cells between the model's and the human's matrix. Its null shuffles cue rows within category for the observer only (4,999 permutations).

Model responses come from a complete real rotation run. A clip needs a usable response on all five rotations. If more than 5% of clips are lost, that model's RQ1 result is descriptive. This route uses A1's composite-grid rotations, so its reading depends on RQ0. The rule below was set before any A1 result was read (DECISIONS 2026-11-07). The analysis runs once per model in every case, and rows 2 to 4 change only how the score is read.

| A1 | A3 | Natural-clip RQ1 |
|---|---|---|
| Pass | Pass | Confirmatory, subject to the 5% clip rule |
| Fail | Pass | Computed and reported as descriptive. The grid limited that input, so the pattern score cannot carry a claim |
| Pass | Fail | Computed and reported as descriptive, pending the inspection the A3 table requires |
| Fail | Fail | Computed for the record, with no claim. No RQ1 to RQ5 claims for that model |

A later analysis on `single_face` would be exploratory, or would need its own amendment before data.

Recovery with synthetic observers (200 observers, 199 shuffles per strength, on this route) gave power of .995 at 0.75 times and 1.0 at 1 times the human effect, and size of at most .06 against a limit of .081. Power was .775 at 0.5 times. For an observer sharing half the human weights, power was .60 at 0.75 times and .83 at 1 times. Size for an observer using different cues was .055 at 0.75 times. A positive score therefore supports "aligned with the human pattern", not "uses the same cues", and a null does not show absence of alignment. Proposition 4 below gives the reason. Single cells are descriptive. Intensity estimation is weak for AU15 and AU20 in LibreFace (PCC .36 and .34 on DISFA [27]), which are outside the four families. LibreFace's intensity outputs were fine-tuned on DISFA after pre-training on EmotioNet, AffectNet and FFHQ [27]. OpenFace 3.0's AU head was trained on DISFA and BP4D, with AffectNet, gaze and landmark sets in a shared multitask model [28]. The two detectors therefore share AU training data, so their agreement is not fully independent evidence, and neither reports training on EU-Emotion.

### RQ1 on randomised cues (A4)

A4 was frozen on 2026-11-07, before any A1 result was read and before any model saw a render (see the table above). Rules below follow the frozen text. Thresholds marked proposed were set by the authors and have no external source.

**Stimuli.** A headless renderer built each AU configuration on MakeHuman meshes using FACSHuman targets [29], at 768 by 768 pixels with a frontal view, flat lighting and a black background. This is a new stimulus set of the same AU recipes. It does not reproduce what humans saw: the public deposit [25] holds neither the base faces nor timing parameters, so models see the same AU configurations, not the same stimuli. Avatars are bald and static. Of the 33 columns, 31 map directly to targets, and AU22 and AU27 are approximated in our renders. That is a local finding, since FACSHuman's own Table 9 lists AU22 as usable [29]. Amplitude maps linearly to target weight, which is an assumption, so sets at 0.75 and 1.25 times amplitude repeat the fit. At 1.25 times, amplitudes stop at 1, as the renderer's range requires, so that set compresses large amplitudes. FACSHuman's targets are LGPL-3.0 with no output clause, so no image is published.

| Set | Content | Rows |
|---|---|---|
| `main` | Every available configuration in each dataset, each on one identity: 4,659 basic WC, 2,323 basic EA, 2,318 conversational WC, 2,325 conversational EA | 11,625 |
| `replicate` | The first 200 of those on a second identity | 800 |
| `one_au`, `neutral` | Each of 33 columns alone at .25, .5 and 1, on 6 identities, plus their neutral faces | 600 |
| `scale_0.75`, `scale_1.25` | 400 configurations, basic EA and conversational EA | 1,600 |
| `blind` | The first 600 basic EA configurations, under the blind readout | 600 |

That is 15,225 rows and 11,761 unique images (`research/PREREG_A4_DRAFT.md`). All available configurations are used, because 1,200 per dataset failed recovery. They are ranked by a seeded hash of the configuration string, with no use of responses, and that order picks the replicate, scale and blind subsets. Face gender comes from those shown to humans with that configuration, by hash.

Identities are own-group faces, as humans saw: four Western faces for WC sets and four East Asian faces for EA sets. Identity coverage is narrower than for humans, and it is not modelled beyond gender, as in the human fit. The East Asian faces are MakeHuman's macro sliders set to Asian, plus a seeded jitter for the second face. They are not based on East Asian people, and they are not independent of the Western meshes, so "own-group" is approximate for EA sets. Whether Snoek's left and right labels mean the same as the avatar's is unverified, so the confirmatory basis is the 25 families, which use the bilateral composite.

**Rendering and detector checks.** Rendering was checked with no vision-language model (`research/notes_au_render.md`): 231 checks on 33 columns by 7 identities at .25, .5 and 1 found mesh and pixel change monotonic and laterality correct. The detector check is gate V1, below, and its pilot result is disclosed there.

**Task.** Each model saw one image, sent as on `single_face` and scored as in A3, with A1's prompt unchanged. That prompt mentions video frames although one image is sent. There is one fixed option set per task, so options carry no information about the stimulus: Anger, Disgust, Fear, Happy, Sadness, Surprise and None of the above for basic labels; Bored, Confused, Interested, Thinking and None of the above for conversational labels. Human "other" maps to None of the above, the abstain option of A1 to A3, which stays last. The two are not the same word, so a sensitivity analysis drops that row. All cyclic rotations of the named labels are used, six for basic and four for conversational. Responses are exact option probabilities averaged over a stimulus's complete rotation set. A stimulus without a usable response on every rotation leaves the analysis, and losing over 5% of a dataset's stimuli fails that dataset.

A blind readout on basic labels is secondary, and it never replaces or rescues the main result. Six families are renormalised, with no "other" row, so its CFA is in different units from the option-based CFA and the two are not compared. It is reported only if a blind gate passes: repeat change of at most 1e-3, median lexicon mass of at least .05, mean root-mean-square change in family CLR, render against pixels removed, of at least 0.25 nats on 50 stimuli, and V3's test at *P* < .01 on 200 basic EA replicate configurations.

**Human target.** The human matrix is the existing output in `research/reproduction/human_au_effects.json`, computed before any render existed and not refitted for A4. One ridge logistic per subject and label, one-vs-rest, was fitted on the 33 AU amplitudes plus centred face gender, with ridge strength chosen by held-out log loss. Subject estimates were averaged, and G-computation gave the difference in probability between do(AU = 1) and do(AU = 0), other AUs as observed, renormalised over options, on 25 AU families.

| Dataset | Subjects | Ridge | Ceiling, raw | Ceiling, residual | WC to EA reference, raw | WC to EA reference, residual |
|---|---|---|---|---|---|---|
| Basic WC | 60 | 0.1 | .990 | .970 | .877 | .780 |
| Basic EA | 60 | 0.1 | .989 | .983 | .877 | .780 |
| Conversational WC | 20 | 1.0 | .978 | .974 | .767 | .732 |
| Conversational EA | 20 | 1.0 | .974 | .960 | .767 | .732 |

Ceilings use 200 splits across subjects and stimuli, on do effects, because models see other stimuli (`research/reproduction/a4_human_ceiling.json`). One human group against the other is the yardstick: a model cannot be asked to agree with Western observers better than East Asian observers do. The ceiling only says how far noise limits the score. It is computed from pooled fits at ridge 1.0 for every dataset, while the human target averages per-subject fits at the selected ridge (0.1 for basic), so it is the reliability of a closely related estimator, not of the target itself, and corrected values are approximate (Supplementary Note 1). The two groups saw faces of their own ethnicity, so the yardstick mixes observer culture with stimulus faces [19].

**Estimand.** The model counts as one subject with fractional responses, through the same fit at the human ridge strength for that dataset, fixed in advance. Two statistics are computed for each model and dataset, each with its own null, interval and reliability:

raw CFA = cos(*tau_model*, *tau_human*)

residual CFA = cos(R(*tau_model*), R(*tau_human*))

over options by 25 families. R takes out each cue's component along that matrix's own mean profile across cues. A system that answers every cue with one profile, scaled by salience, can score on raw CFA without any cue-specific agreement, and it has nothing left after R.

**Residual CFA is confirmatory, and raw CFA is secondary.** This was fixed before any model data. Raw CFA was tried first and failed recovery on a decision (see Recovery). If residual CFA failed recovery, the analysis would refuse to run. For raw CFA, each cue's effects sum to zero across options, so it equals Pearson *r* over entries. Each statistic is also reported over its ceiling times the square root of the model's split-half reliability across stimuli (20 splits), and that ratio is descriptive and decides nothing. The dose is the full amplitude range, on both sides. A model has no response noise, so its reliability *R_S* reflects which stimuli were sampled. The 200 replicates vary identity alone, and they serve V3, not reliability.

The one-AU set gives a direct contrast: an AU alone against that identity's neutral face. It is secondary, because it tests an AU at a neutral baseline, whereas the human effect is averaged over other active AUs. It is the only place dose can be read.

**Inference.** Signal: stimuli are shuffled against responses (999 draws) on the summed squared OLS AU coefficients, with *P* < .05 required, so that a prior-only system fails here. Alignment: AU labels are shuffled across the 25 columns of the model's matrix (9,999 draws, one-sided on CFA), which keeps each cue's profile and any effect common to all cues. Holm correction runs across the two eligible models within a dataset. Interval: stimulus bootstrap, 2,000 draws, 95% percentile. It treats stimuli as sampled and does not include error in the human matrix. Smallest effect of interest: CFA of .30 (proposed), used as a floor and not a target. It has no external source, and it is about 9% of shared variance.

The rule below applies to residual CFA, per model and dataset.

| Outcome | Requires |
|---|---|
| Aligned | All gates; signal *P* < .05; alignment Holm *P* < .05; CFA of at least .30; lower interval bound above 0 |
| Not aligned | All gates; signal *P* < .05; upper interval bound divided by the square root of *R_S* below .30, where *R_S* is the model's split-half reliability across stimuli on that statistic |
| Inconclusive | Anything else, including any failed or missing gate |

Not aligned is always written "not aligned on these renders", since most families are unconfirmed by a detector and the avatars are a new stimulus set. It is the null, and it requires the signal test, the gates and the equivalence bound after disattenuation. A model with no signal cannot be Not aligned: it is Inconclusive, because nothing shows the cues were delivered. A missing or non-positive *R_S* cannot support Not aligned.

The rule is asymmetric on purpose. Aligned uses uncorrected values of the confirmatory statistic, and Not aligned uses disattenuated ones. Noise can only lower a cosine, so it cannot create Aligned, whereas it could create a false null. The two claims therefore face different bars, and Aligned can be reached at a CFA where the interval is wide. Below *R_S* of .85, uncorrected CFA reads as a lower bound: Aligned still stands, and the corrected rule decides Not aligned.

Aligned means a reliable shared pattern, not the same strategy. A model at .31 is Aligned and far from human-level. In recovery, an observer with 16 of 25 cue families swapped in 8 pairs could still clear .30. CFA is therefore always reported against the WC to EA reference for its own statistic, and a model is called human-level only if its interval reaches that reference, which is descriptive. Each dataset is reported alone, then one headline per model per task: Aligned or Not aligned on these renders only if WC and EA agree, Mixed if one of each, and Inconclusive otherwise. Basic and conversational are separate claims and are never pooled. WC and EA share neither observers nor most configurations, but they are not independent tests of a model.

**Validity gate.** None of these checks uses a human target. All run per model and task, before any human comparison, and a missing check counts as not passed. Rules are proposed values.

| Check | Rule |
|---|---|
| V1 Detector | Across pairs of covered column and identity in the one-AU set, at least 70% have detected output rising across .25, .5 and 1 |
| V2 Image dependence | Mean root-mean-square change in option log probability, render against pixels removed, of at least 0.25 nats, on 50 stimuli |
| V3 Configuration carries | Output closer for one configuration on two identities than for a mismatch, on the Jensen-Shannon divergence, sign-flip over 200 configurations, one-sided *P* < .01. Mismatches are drawn within gender and ordered identity pair, so each compares the same two faces |
| V4 Domain limit | Mean mass on None of the above of at most .90, and on any option of at most .95 |

*V1 coverage.* LibreFace outputs 17 AUs, 12 with intensity (1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26) and presence alone for 7, 10, 14, 23 and 24 [27]. At least 8 of the 25 families therefore have no readout (AU11, 13, 16, 22, 27, 38, 39 and 43). The denominator is covered pairs only, and those columns are listed as unconfirmed, not as failures. LibreFace is bilateral, so laterality is not tested. It was trained on natural faces, so a failure may reflect the detector and not the render, and V1 does not separate them. Missing detector output stops the analysis and is not read as Inconclusive.

*V1 pilot, before the freeze, with no model.* A face was found on all 432 images, and 113 of 144 covered pairs of column and identity rose, which is .785 against the rule of .70 (`research/reproduction/a4_v1_pilot.csv`). That pass is thin. The rule counts any rise, and most are tiny. Of 96 intensity pairs, 22 rise by at least .5 on LibreFace's 0 to 5 scale, all in AU01, 05, 25 and 26. AU12 at full amplitude reads about .02, so LibreFace does not see the rendered smile, and the same holds for AU04, 06, 09, 15 and 20. Of the 43 rising presence pairs, 35 stay at negative logits, so they rise without reaching presence. V1 therefore shows that the renders are monotonic where the detector responds. It does not show that a natural-face detector reads them as those AUs. The threshold was not changed after this was seen.

*Family flag, not an exclusion.* Only AU01, 05, 25 and 26 are read by the detector with a real rise, and every other family is flagged. These four hold .14, .13, .09 and .20 of residual human energy on basic WC, basic EA, conversational WC and conversational EA (`a4_human_ceiling.json`), so most of what A4 compares is unconfirmed by a detector. All families stay in the confirmatory statistic, and a sensitivity reports the flagged and confirmed families apart.

*V4 is a weak guard.* Its .90 limit is on mean mass, not modal choice, so a heavily abstaining model can pass it.

**Behaviour beside CFA.** On the same stimuli, per model and dataset, we report two descriptive measures: output similarity (Pearson *r* between model and mean human response over stimuli and options, with option means removed), and modal agreement with chance-corrected kappa. Snoek data have no correct answer, so error consistency proper is undefined, and modal agreement is its closest analogue. Whether these measures disagree with CFA is an empirical result.

**Eligibility.** RQ0 is read through the A1 and A3 table on `single_face`, by the rule A2 uses. A Pass on A3, with either outcome on A1, admits a model, subject to the vision gates. Any other outcome, or a missing result, leaves it Inconclusive. Gemma 4 and Qwen3-VL are the Holm family. Molmo2 fails the vision gates, so it is descriptive if run at all.

**Recovery.** Before any run, synthetic observers answered the real configuration design through the analysis code, at effect sizes of 0.25, 0.5 and 1 times the human log odds, with 100 replicates per cell. Noise was logit SD 1 per stimulus and option, with a lognormal gain error of log SD 0.5 per AU that stands in for the unknown amplitude mapping. Observers were human-like, half-shared, boundary (human weights mixed at .35), different-cues, prior-only, generic, salience-profile and pair-swapped. Every attempt is kept.

| Run | Configurations | Datasets | Raw CFA | Residual CFA |
|---|---|---|---|---|
| 1 | 1,200 per dataset | Basic EA, conversational EA | Fail, 27 of 28 checks: pair-swapped accuracy at conversational EA 0.25 times, 66% | Not run |
| 2 | 1,200 per dataset | All four | Fail, 63 of 66: false Aligned on conversational WC, up to .33 | Fail, 62 of 66: pair-swapped accuracy in 4 cells, 76 to 88% |
| 3 | All available | All four | Fail, 62 of 66: false Aligned on conversational WC and at conversational EA 0.25 times, up to .40 | Pass, 66 of 66 |
| 4 | All available | All four | Fail, as run 3 | Pass, 66 of 66. Same tables and checks as run 3; only the design hash differs |

| Rate, run 3, all cells unless stated | Raw CFA | Residual CFA |
|---|---|---|
| Human-like Aligned | 1.0 | 1.0 |
| Half-shared Aligned at 1 times, lowest | .87 | .79 |
| Different-cues Not aligned at 1 times, lowest | .95 | .95 |
| False Aligned, highest | .40 | .03 |
| Prior-only signal, highest | .08 | .08 |
| Not aligned with true CFA at or above .30, highest | .01 | .03 |

*Exempt cells.* After run 1 and before any model, raw accuracy at conversational 0.25 times was withdrawn as a claim in both conversational datasets. That exemption matters to the pass, because it also covers residual CFA, where pair-swapped accuracy in those two cells is 85% on conversational WC and 62% on conversational EA (`a4_recovery.json`). Counting those two cells, residual CFA would pass 66 of 68. A4 therefore makes no accuracy claim for residual CFA on weak conversational effects, and a decision there rests on the corrected rule. Residual CFA also reads low on weak effects: at 0.25 times, the human-like observer reads .74 against a true .87 on conversational EA.

Raw CFA can be fooled. On conversational WC, a system with no cue-specific agreement reached Aligned in up to .40 of replicates, because about a third of human energy there lies along the common profile, and more stimuli did not help. Residual CFA made no false Aligned above .03. Recovery assumes additive log-odds observers, independent stimulus noise and the gain error above. It does not model domain shift, so it shows how the analysis behaves, not how a real model will respond. It used 100 bootstrap draws and 6 reliability splits per replicate, against 2,000 and 20 in the real analysis. The number of configurations was raised after failures, so the design was tuned on recovery, on synthetic data only.

On synthetic stimuli with paired cues at a constructed correlation of .9, pair-swapped observers' outputs correlated .83 to .95 with the human-weighted observer, whereas raw CFA on the randomised design was .42 to .56 (run 2). The correlation of .9 is ours, not a measurement. This shows only that the analysis can register the gap on synthetic data.

**Sensitivity.** Reported beside the primary analysis and never replacing it, each on residual and raw CFA: without AU22 and AU27; without the abstain row; flagged and confirmed families apart; amplitudes at 0.75 and 1.25 times, against the 1 times fit on the same configurations; and the direct contrast, by dose. Subsets take the common profile from the full matrix.

**Disclosures and limits.** AU22 and AU27 are approximated, yet they hold .109, .119, .183 and .103 of human squared effects on basic WC, basic EA, conversational WC and conversational EA. The common profile holds .26 to .35 of human squared effects, and each statistic must be read against its own reference. The bootstrap is over stimuli only, conditional on the human matrix and on these identities. A difference from humans may reflect the avatar domain, the static route or the amplitude mapping, because humans saw 1.25 s of movement and their effects average over time course. V1 to V4 only partly address this. So this compares a model's reading of a peak frame with human effects averaged over motion. Patterns are compared, not sizes. The comparison covers these 33 columns, and additive effects are assumed on both sides. Models read a text prompt with listed options, which is not how humans were asked. Each set has four faces. On conversational data each human group has 20 subjects, and no claim extends to other populations. The human target averages subject estimates, whereas the framework of [4] warns against averaging over individuals' information-processing methods. That framework also assumes the same stimuli for both systems, so A4 approximates its second degree and does not meet it as framed.

### RQ2: stimulus interventions (A2)

**What this tests.** Our research question RQ2 asks whether a human-diagnostic cue is necessary and sufficient to move a judgement, through graded edits of one cue with verified AUs, and whether the dose-response follows the human curve. A2 tests a narrower thing, its stimulus arm: whether region masks, spatial-frequency filters and changes to frame order move a model's judgement in the direction of a stated human prediction. It has no AU edit, no verified dose and no human dose-response curve, and its one graded cue, eye aperture, is outside the confirmatory family because the dose is unverified. RQ2 in this paper means this narrower test. It cannot establish necessity or sufficiency of an AU. Single rendered AUs appear only under A4, on avatars, as a secondary contrast.

**Eligibility.** RQ0 is read through the A1 and A3 table. Pass on both admits both routes. Fail on A1 with Pass on A3 admits `single_face` only, and any other outcome admits neither. The vision gates must also pass. Gemma 4 and Qwen3-VL are eligible, subject to RQ0.

**Routes.** Frames were decoded and edited in memory. `single_face` sends the apex frame as a face crop at 768 by 768 pixels, with one fixed square per clip at 1.5 times the largest landmark extent. `video_face` sends 8 uniformly spaced frames through the native video processor, on a synthetic 1 frame-per-second clock, so that timestamps depend only on position in the sent order. On Qwen3-VL, native video merges pairs of frames, so order within a pair is partly lost. Landmarks came from MediaPipe FaceMesh 0.10.14, and the apex frame is chosen from landmark movement alone. A route that fails delivery checks is not analysed. A sham that runs the mask pipeline with the frame's own pixels must equal the original exactly.

**Conditions.** Spatial conditions, on `single_face`: masks of mouth, eyes, brows, and eyes and mouth together, each with three fills (the same actor's neutral clip, the clip's first frame, and a heavy blur of the region); two eye-only views; and low-pass and high-pass filters (half amplitude at 8 and complement at 32 cycles per face width). Eye-aperture warps are run but are out of the confirmatory family, because the dose is unverified. Temporal conditions, on `video_face`: two fixed shuffles, reversal, truncation to the first or last third, and the apex frame repeated. No generator is used, and this suite contains no AU edit, gaze edit, head-angle edit or playback-speed test.

**Estimand.** For item *i* and condition *c*, delta(*i*, *c*) is *y*(*i*, *c*) minus *y*(*i*, original), within item and route. The primary outcome *y* is the CLR of the acted family. Predictions that another label rises use that family's CLR with the acted label left out of the reference mean, since otherwise a plain fall in the target would read as a predicted rise. Effects are averaged per actor. A bridge readout using rotated options never replaces a blind result.

**Inference.** The actor is the unit. We used an exact sign-flip test of actor effects with Westfall-Young step-down maxT [CITE NEEDED: Westfall and Young] within one model and route, at familywise alpha .05, and actor-bootstrap 95% intervals (10,000 draws). With 8 actors a one-sided *P* value cannot go below .0039.

**Predictions.** The contrasts below are the only confirmatory ones. Almost none of the human baselines were measured on EU-Emotion, and none on these edits. All carry a BASIC ONLY scope tag, so they make no claim about complex labels. Most sources were read at abstract level (references.md). Human-source agreement is recorded per prediction in A2.

| ID | Contrast, within item | Predicted direction | Sources |
|---|---|---|---|
| P1 | Mouth mask on Happy | Acted label falls | [30] (inferred); [31] pooled over expressions |
| P2 | Mouth mask on Disgusted | Acted label falls | [30] (inferred) |
| P3 | Eye mask on Afraid | Acted label falls | [30] (inferred) |
| P4 | Eye mask on Sad | Acted label falls | [30] (inferred; the Sad cue is a non-significant brow tendency) |
| P5 | Brow mask on Angry | Acted label falls | [32] (upper face, not brows alone) |
| P6 | Mouth minus eye mask, Happy and Disgusted against Afraid, same 8 actors | More negative on Happy and Disgusted | [30] (inferred) |
| P7 | Mouth mask on Happy, Sad and Angry, Neutral family | Neutral rises | [33], with a visible mask over nose and mouth |
| P8 | Mouth mask on Disgusted, Angry family | Angry rises | [33], descriptive, with the nose covered |
| P9 | Low-pass minus high-pass on Happy | Positive | [36] (abstract only). Task dependence on other labels [37] |
| P10 | Low-pass minus high-pass on Sad | Negative | [36] (abstract only) |
| P12 | Mean of shuffles on Happy and Surprised | Acted label falls | No direct shuffle source. Analogy from [39] (disrupted rhythm, subtle expressions, abstract only). Contrary: [50] |
| P13 | Mean of shuffles on 48 Low Intensity items | Acted label falls | No direct shuffle source. Analogy from [40,41] (dynamic against static, abstract only). Contrary: [50] |
| P14 | First minus last third, Surprised family on Afraid and Afraid family on Surprised | Confusable label higher on the early third | [42], ideal-observer classifier on Western observers' expectation models |
| P15 | First minus last third, Disgusted family on Angry and Angry family on Disgusted | Confusable label higher on the early third | [42], as P14 |

P11, on eye aperture [43], is outside the family until its dose is verified. These 14 predictions make 16 contrasts. Directions for P1 to P4 and P6 are inferred, because [30] is a reveal study in which no region was masked with the rest visible. [33] used a visible community mask over nose and mouth on static photographs, whereas our mouth masks use neutral, first-frame or blur fills and may leave the nose visible. No cited source tests frame shuffling for P12 or P13, and the closest human test found no cost of shuffled or reversed order [50]. A VLM study also found almost no change under frame shuffling [11]. P14 and P15 are analogues, not replications, of an ideal-observer analysis of expectation models on synthetic faces [42]. Those classifiers accumulate signals, so a stand-alone last third has no counterpart in that paper, and nothing shows that thirds of an EU-Emotion clip map onto its time bins. These source checks came after the freeze. They change the stated support and not the predictions (see Disclosures). Eye-only views have no directional prediction, because sources differ by label type: [45] finds the whole face better than the eyes for basic labels and the two equal for complex ones, and [31,46] are read at abstract level; masks on complex labels are descriptive [46,47]. Reversal describes whether order changes the model's judgement, and we give no human direction for it. No category flip is predicted from head angle, here or later, because we located no human study showing one. A downward tilt raising perceived dominance is the located evidence [49], and this suite does not edit head angle.

**Decision rule.**

| Outcome | Requires |
|---|---|
| Matches prediction | RQ0, the vision gates, Gate B, route delivery and the sham passed; step-down *P* < .05 in the predicted direction on the blind readout; the first-frame and blur fills share the neutral fill's sign |
| Opposes prediction | The same gates passed; step-down *P* < .05 two-sided in the opposite direction |
| Null | The same gates passed; neither test significant; every null condition below met |
| Inconclusive | Anything else, including a gate with no result |

**Null rule.** A null needs delivery to pass, a positive control showing that the cue type can move this model on this route, a null joint ablation of eyes and mouth where one exists, and equivalence: the 90% actor-bootstrap interval lies inside a margin of 10% of that model's mean absolute CLR change under the full image mismatch of B4 on that route. By design, P5, P6, P9, P10, P14 and P15 cannot reach Null, which is 8 of the 16 contrasts, or 10 of 16 on Qwen3-VL, whose P12 and P13 are also capped because frame order cannot be verified there. Those contrasts read as a match, an opposition or Inconclusive, and A2 cannot show that a model ignores those cues.

**Recovery.** Synthetic observers (prior-only, frame-bag, mouth-only, eyes-only, human-like) ran through the analysis on the real 243 items and actors, 100 replicates each. All five passed the stated rule. Recovery used large effects and independent noise, so it shows that the pipeline works, not its power at realistic effects.

**Sensitivity.** Reported beside the primary analysis: without smoke items, without flagged alignment, on canonical words only, and on clips with at least 8 tokens across the face. Results are also shown for the human modal label and as Jensen-Shannon divergence from the human distribution on the item's options. Humans did not see these edits, so a change in divergence is descriptive and says nothing about what they would do.

### Scoped and provisional questions

**RQ3, reported as a scoped limit.** RQ3 asks whether a cue's effect is carried from image tokens through the language model, and whether blocking that path removes it. We report this as a limit, not as a test (DECISIONS 2026-11-07). The reason is logical and not one of cost. A cut that takes out the whole vision path gives the same output for every image, so a cue effect under it is zero by construction, and the literal test cannot fail. Such a cut can only check that no route from pixels to the answer was missed. Holding image positions past the entry of the language model cannot be a complete cut, because text positions have by then read the image, and holding those too returns the image-free forward. The depth at which a cue is carried is therefore not testable on either eligible model. No site has yet been shown complete on `single_face` or on the blind readout, because the vision gates ran on the legacy input with the option prompt.

A draft amendment, A7, specifies a Stage 0 table on the 24 calibration clips, on `single_face`, with the blind readout. It would establish whether the whole-path cut and its two named blocks are complete on that input, and whether the tokens at a cue's location can act as a cut for that cue. **Stage 0 has not run.** A7 is not frozen, its code does not exist, and its recovery is specified but not run. Stage 0 is costed at about .71 GPU-hours, an upper bound. [PENDING: A7 freeze; Stage 0 table for Gemma 4 and Qwen3-VL.] Whether a block of a complete cut meets our rule that only complete cuts count is open to A7's audit. If it does not, the report is the Stage 0 table plus "not testable". A comparison of the two blocks would be an optional supplement, and it has no human reading. A pass on the region test would give a non-vacuous cue-level answer, and the question would be reopened by a dated entry. This paper makes no claim about the route a cue takes through a model, and no site is described as a cut unless Stage 0 shows it complete.

**RQ4 and RQ5, provisional.** Two further questions have draft amendments that are not frozen and are still changing, so nothing here binds and no threshold is given. [PENDING: A5 freeze.] A5 addresses RQ4, how much of a judgement comes from face evidence and how much from label priors, by separating prior from evidence with image-free, grey and mismatch baselines on the blind readout. [PENDING: A6 freeze.] A6 addresses RQ5, whether a shared error has the same cause in humans and models, on CREMA-D, across its fixed option set, with a cue-targeted test of one human confusion. Either is reported as confirmatory only if frozen before its data are read.

### Identifiability results

Full statements and proofs are in [PENDING: Supplementary Note 1]. Numerical checks are in `research/scripts/cfa_identifiability.py`.

**Setup.** A stimulus has cues *z* in R^*p*. A system *S* maps *z* to a response distribution *f_S*(*z*) on *K* options. Natural stimuli have distribution *P*, and a design *Q* gives the stimuli used for measuring effects. For cue *j*, dose *delta_j* and option *k*, the effect matrix is

*tau_S*(*j*, *k*) = E over *z* in *Q* of [*f_S^k*(*z* + *delta_j* *e_j*) minus *f_S^k*(*z*)].

Each row sums to zero. CFA(*S*, *T*) is the cosine between *tau_S* and *tau_T*, and it is undefined if either matrix is zero. A behavioural metric is any functional of the joint distribution of items and responses on *P*. This covers accuracy, confusion matrices, error consistency and representational similarity on outputs. A4 and the human matrix use the set-to form, do(AU = 1) minus do(AU = 0), with other AUs at their observed values. For a linear observer this equals the dose form at dose 1, and Propositions 2 and 3 hold unchanged. Theorem 1 is proved for the dose form. Its construction carries to the set-to form when neither intervened point is a natural item, and that case is not covered here. These results are stated for CFA as defined here. A4's confirmatory statistic, residual CFA, removes from each system's matrix the component along that system's own mean option profile, and is the cosine of what remains. Supplementary Note 1 defines it from the frozen code and states which results carry over: Corollary 1.1 carries when (K - 2)(p - 1) >= 2; Proposition 2 carries only in a three-option variant; the upper bound of Proposition 3 needs a shared mean direction and the lower bound fails; and Proposition 5 is approximate, because the projector is estimated from the data.

**Theorem 1.** Let *P* have finite support, take *Q* = *P*, and let the shifted points be distinct and off the support. For any system *T* and any admissible zero-row-sum matrix, there is a system *S* equal to *T* on every natural item whose effect matrix is that matrix. As a corollary, for any reference system *H* with non-zero effects and any *r* in [-1, 1], a system exists that is indistinguishable from *H* on every behavioural metric on those items, with CFA(*S*, *H*) = *r*. The construction changes the system off the natural support and is not smooth. It shows that behaviour imposes no logical constraint. It does not show that trained networks behave this way.

**Proposition 2.** With two standardised cues correlated at rho, a linear observer on each cue gives output correlation rho, yet CFA of 0 at equal doses. As rho approaches 1, behaviour converges while alignment stays at 0.

**Proposition 3.** For linear observers with matched means, equal doses *delta*, and natural cue covariance with extreme eigenvalues lambda_min and lambda_max,

(*delta*^2 / lambda_max) E||*f_S* minus *f_T*||^2 <= ||*tau_S* minus *tau_T*||_F^2 <= (*delta*^2 / lambda_min) E||*f_S* minus *f_T*||^2.

Both bounds are attained. So 1 / lambda_min measures how far behavioural agreement can understate a difference in cue effects. A randomised design pushes the cue covariance towards the identity, and this is why it identifies cue effects.

**Proposition 4.** If a system responds to an unmeasured cue *U* that is correlated with measured cues, a regression on measured cues gives non-zero slopes, while intervening on the measured cues alone has no effect. Regression on natural clips therefore cannot reach a claim about cue use.

**Proposition 5.** With independent zero-mean errors in both matrices, the observed cosine approximates CFA times the square root of the product of their reliabilities, which is Spearman's attenuation correction applied to a matrix [CITE NEEDED: Spearman]. A4 estimates human reliability by split-half across subjects and stimuli, and model reliability by split-half across stimuli, because a model has no response noise. When models see other stimuli than humans did, the subjects-and-stimuli ceiling applies. Corrected values above 1 can occur from sampling and are reported uncorrected beside them.

**Cue spectra.** These describe the designs and use no model output. Correlation matrices of each cue table gave the values below (`research/reproduction/cfa_identifiability.json`). Snoek rows count unique rows of the 33 amplitudes, whereas A4's manifest counts configuration strings, so basic EA, conversational WC and conversational EA have 2,322, 2,314 and 2,322 rows here against 2,323, 2,318 and 2,325 there:

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

We ran no power analysis to set sample size, because the stimulus sets are fixed: 243 items and 17 actors for EU-Emotion, and existing human samples for Snoek. A4 uses every available configuration, and recovery tested that count. Claims stop at these actors, these avatars and these human samples. Tests are one-sided where a direction was stated. Multiplicity is handled by Holm correction across models (A1, A3 and A4, each its own family) and by Westfall-Young step-down within model and route (A2). Seeds are fixed in the amendments. Each output carries a hash of its design, and analysis refuses cells whose hash differs from the code.

### Data availability

Snoek data are public [25]. EU-Emotion stimuli cannot be redistributed, and no clip, frame or edited image is public. [OPEN, for the user or the stimulus authors: whether the EU-Emotion licence covers in-memory edits as in A2, and which aggregated norms may be shared. Neither is settled by any document in the repository.] Renders are not published, because FACSHuman's targets are LGPL-3.0 with no output clause. [PENDING: data availability statement.]

### Code availability

Analysis code, frozen hashes and derived outputs are in the public repository. [PENDING: DOI-granting code archive; clean-checkout reproduction test.]

### Reporting summary

A Nature Portfolio reporting summary is required for this journal and is to be supplied at submission. [PENDING: the summary itself.]
