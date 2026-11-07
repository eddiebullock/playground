# DRAFT amendment A2: blind readout gate and stimulus interventions

**Status: DRAFT. Not frozen, not preregistered, not part of `research/PREREGISTRATION.md`.** Nothing here binds until it is reviewed, its open choices are settled, and it is committed and tagged before any full run. Numbers marked **[proposed]** were set by the implementer and need sign-off.

**Written:** 2026-11-04, before any rotation result was seen and before any full intervention run.
**Depends on:** amendment A1 (RQ0) and gates G2 to G4. No full run starts until these pass for that model.
**Code:** `scripts/stimulus_interventions.py`, `scripts/run_stimulus_interventions.py`, `slurm_jobs/study3_stimulus_interventions.sh`, `scripts/analyse_stimulus_interventions.py`, `scripts/recovery_stimulus_interventions.py`, `scripts/human_response_predictor.py`. Each output carries `design_sha256`, a hash of the prompt, prefill, lexicon, conditions and frame count.

**What was seen before writing this.** A smoke run of 3 items on Gemma 4 E4B ran the pipeline before these predictions were committed, which sits against rule 2 of `RESEARCH_QUESTIONS.md`. Only mechanics were read: run status, token counts, timing, determinism, whether a blanked frame changes the logits, lexicon coverage and top next tokens on unedited clips. No contrast between conditions was computed by label or direction. The confirmatory analysis is repeated without those 3 items (listed in the appendix) as a sensitivity check. The processor probes used synthetic noise, not clips.

## Questions

**Gate B.** Under a fixed prompt that names no label, does a model's readout depend on the face?

**RQ1 and RQ2 (stimulus arm).** Holding identity and clip fixed, do region masks, an eye-only view, spatial-frequency filters, eye-aperture scaling and changes to frame order or extent move a model's judgement, and in the direction of a stated human prediction?

**Scope.** Causal claims are about the model. Human predictions come from other stimuli and are fixed targets. This work never states how humans would respond to an edited EU-Emotion clip, only whether the model matches a stated prediction. EU-Emotion results concern recognition of acted conventions, within this stimulus set and these 17 face actors.

## Stimuli and sent frames

All 243 EU-Emotion face clips with six human options. Frames are decoded and edited in memory. No clip, frame or edited image is written, copied or uploaded.

- **Fixed frames.** k = 8 frames at uniform positions over the clip, by sequential decoding so that an index means the same frame every time. Every cell logs its sent indices in sent order, a hash of each image sent, and a hash of the model's pixel tensor.
- **Apex frame.** Among the 8, the frame furthest from the first usable frame in lip, lid and brow landmarks, after registration on stable points. It uses landmarks only, so no model output can influence it.
- **Face crop.** One fixed square per clip, 1.5 times the largest landmark extent, so head motion stays visible, sent at 768 by 768. Face crop is primary because faces fill about 18% of frame width in the clip measured so far, which leaves very few tokens on the face in a full frame (see Delivery).
- **Landmarks.** MediaPipe FaceMesh 0.10.14 with `static_image_mode` and refined landmarks, one independent detection per frame. All models get the same geometry, computed once.
- **Failures.** A frame with no face fails only the cells that need it. Every failed cell stays in the output with its reason, and failures are reported per condition and per label. No item is dropped silently and no landmarks are borrowed from another frame.

## Input routes

| Route | Content | Used for | Why |
|---|---|---|---|
| `single_face` | Apex frame, face crop | Spatial conditions | A lip corner is sub-token on a grid |
| `single_full` | Apex frame, whole frame, unedited | Bridge from crop to frame | Shows what the crop changed |
| `video_face` | 8 frames, native video processor, face crop | Temporal conditions | Only route where order is not a spatial position |
| `legacy_full` | 8 frames through `prepare_images_for_model`, `original` only | A1 bridge | Composite grid on Gemma 4 and Molmo2, image list on Qwen3-VL. Not for causal work |

Video timestamps enter the prompt on Gemma 4, Qwen3-VL and Molmo2, so all clips and conditions use a synthetic 1 fps clock. Timestamps then depend only on position in the sent order and the token sequence is the same in every condition. A model cannot read duration from the prompt. On Qwen3-VL, native video merges pairs of frames, so order inside a pair is partly lost; 8 frames become 4 temporal units.

## Conditions

**Spatial (on `single_face`).**

| Condition | Definition |
|---|---|
| `original` | Unedited apex frame |
| `sham_self_fill` | Full mask pipeline filled from the frame itself. Must equal `original` exactly |
| `mask_{mouth,eyes,brows,eyesmouth}_neutral` | Region filled from the same actor's neutral clip, middle sent frame, registered to the target frame |
| `mask_{mouth,eyes,brows,eyesmouth}_first` | Region filled from the clip's own first frame, registered |
| `mask_{mouth,eyes,brows,eyesmouth}_blur` | Region filled with a heavy blur of itself |
| `eye_crop` | Band from brow top to below the lower lid, enlarged by the processor |
| `eye_window` | The same band at face-crop scale, grey elsewhere, so scale is held fixed |
| `sf_low` | Gaussian low-pass, half amplitude at 8 cycles per face width |
| `sf_high` | Complement at 32 cycles per face width, channel mean kept |
| `aperture_wide`, `aperture_narrow` | Lids moved by a factor of 1.25 or 0.75 by a local warp with no generator |

Regions are convex hulls of landmark rings, dilated and feathered by fixed fractions of eye span. Cutoffs of 8 and 32 cycles per face are conventional values, not taken from the review notes **[proposed]**.

The aperture warp stretches the iris along with the lids. That is a known artefact. Doses are opposed so that an artefact common to both can be separated from a cue, and lid movement is checked by re-detecting landmarks on the warped frame.

**Temporal (on `video_face`).** All conditions send 8 frames.

| Condition | Definition |
|---|---|
| `original` | 8 frames in order |
| `shuffle_a`, `shuffle_b` | The same frames in two fixed derangements, keyed on the clip |
| `reverse` | The same frames reversed |
| `trunc_first_third`, `trunc_last_third` | 8 frames resampled inside that third of the clip |
| `static_apex` | The apex frame 8 times |

Truncation changes both content and frame density, so it is not a pure order test. Clip onset is unmarked, so "first third of the clip" only approximates the "first third of the rise" in the notes.

**Not in this suite.** The composite grid for causal work, playback speed (needs the shuffle result first, per the design constraints), gaze, generative AU edits, morphs, adaptation, and head angle.

## Readouts

**Blind prefill (primary).** User text: "In one plain word, with no formatting or punctuation, what is this person feeling?" Prefill: "They feel". Chosen on rung 1 of the readout ladder, on calibration clips only, by the rule in `research/DECISIONS.md`. The prompt names no label. Each word's joint logprob is read over the 106-word, 21-family lexicon in `LEXICON`, and a family's score is the log sum over its words. Word forms come from `data/eu_emotion_synonyms.json`, with multi-word entries, forms listed under two labels and forms with a mainly non-mental sense removed. The Neutral family is new.

**Rotated MCQ (bridge).** The item's six options under two rotations, never target first, assigned by item position and held fixed across conditions, so position cannot differ between an edit and its baseline. It uses `build_4afc_prompt` and scores as in the gates, by joint logprob after "EMOTION:". It never replaces or rescues a blind result, since its option sets identify the category (A1, defect 2).

**Baselines per model and route.** The blind readout with pixels removed, and with a uniform grey image.

## Delivery checks, recorded per model and route

Run on the first items. These are mechanics, not tests of cue use.

- Media token count, pixel tensor shape, and tokens per frame.
- Estimated tokens across the face.
- One frame blanked at a time: does the output change for every position?
- Repeat forward: is the change zero?
- Under shuffle: do input ids stay identical, and do pixels arrive in sent order?
- How many frames the existing 1 fps, 16-frame policy would drop at the composite grid (`make_frame_grid` keeps the first 8).

**A route failing these checks is not analysed.** A frame that cannot reach the output makes a temporal null uninformative.

## Gate B: blind readout validity

Run per model on `single_face` and `video_face` originals, all 243 items. Gate B sits after A1 and G2 to G4, and before RQ1 and RQ2 analysis. A1 stays as written.

Let s_i(x) be the centred log-ratio (CLR) of item i's acted-label family on image x: its family score minus the mean over 21 families. Let d(i) be a fixed donor clip with a different label and a different actor, chosen by `vision_mismatch_partners`, seed 42, as in A1.

| Check | Rule [proposed] |
|---|---|
| B1 Determinism | Repeat forward changes logprobs by no more than 1e-3 |
| B2 Coverage | Median lexicon mass on originals at least 0.05 |
| B3 Image dependence | Mean RMS change in family CLR, real versus image-free, at least 0.25 nats |
| B4 Face-specific signal | Mean of s_i(x_i) minus s_i(x_d(i)) above zero, one-sided actor-level sign-flip p < .01, and actor-bootstrap 95% CI excludes 0 from above |
| B5 Beyond label prior | B4 repeated after subtracting each family's mean over all originals from that model and route |

A model and route that fail Gate B go no further on the blind readout for RQ1 and RQ2, and are reported as such. Reading image effects off MCQ instead is exploratory. B4 uses acted labels for continuity. It is repeated against the human modal label and with the 26 flagged items removed.

## Estimands

Item i, condition c, model m, route r, and an outcome y on that item:

    delta(i, c) = y(i, c) - y(i, original)

The primary contrast is always within item against the unedited clip on the same route. It removes clip content, actor and most of the label prior. The primary outcome y is the CLR of the acted-label family, which is scale-free across labels.

Predictions that another label rises (P7, P8, P14, P15) use the named family's CLR with **the item's acted label left out of the reference mean**. With it included, a fall in the acted label lowers the mean and raises every other label by the same amount, so a plain loss of the target reads as a predicted rise. Recovery found this: the mouth-only observer triggered P7 in 23% of 30 replicates. The test changed; the expected sets and the pass rule did not.

Each outcome is also computed for the item's **human modal label** and, on the item's named options, as **Jensen-Shannon divergence** between the model and the human response distribution (None of the above removed from both, which the blind readout cannot express).

Actor-level effect for contrast k:

    e(a, k) = mean of delta over actor a's items in contrast k

Effects are per model and route. Models are never pooled, because their architectures differ and the question is about each one. Flip rate is secondary and reported against the rate between `shuffle_a` and `shuffle_b` on the same item, as a measured nuisance rate.

## Inference

- **Unit.** The actor. 17 actors; 8 or 9 per basic label, 7 to 9 for most others.
- **Test.** Sign-flip of e(a, k), flipping whole actor rows so dependence across contrasts is kept, on the one-sample t. All 2^n flips are enumerated. Implemented as `signflip_maxt`.
- **Multiplicity.** Westfall-Young step-down maxT within one model and route, over the contrasts in the confirmatory family below, at familywise alpha .05. Directions are as stated. Unlike the old Holm gate, the floor here depends on actors, not a permutation count.
- **Floors.** With 8 actors, one-sided p cannot go below 1/256 = .0039. Jealous, with 7, floors at .0078 one-sided. With 17 it is about 8e-6. A label-level contrast is thus testable at .05 but only just, and nothing here supports claims beyond these actors.
- **Partial crossing limits what can be tested.** Happy, Afraid and Disgusted were portrayed by the same 8 actors (C, E, G, I, K, M, O, S). Sad, Angry and Surprised came from the other 8 or 9 (B, D, F, H, J, L, N, P, R), and the two sets do not overlap. A contrast between Happy and Sad is therefore also a contrast between actors. Label-by-cue interactions are tested only inside an actor set. Contrasts across the sets are descriptive.
- **Fill rule.** The neutral fill is primary. A mask effect counts only if it passes and the first-frame and blur fills have the same sign, since the occluder is itself an intervention.
- **Interval.** Actor-bootstrap 95% CI for each contrast, 10,000 draws.
- **Recovery.** Before any confirmatory run, synthetic observers with known behaviour go through `scripts/analyse_stimulus_interventions.py`, by `scripts/recovery_stimulus_interventions.py`, on the real 243 items and actors: prior-only, frame-bag with no order code, mouth-only, eyes-only, and a human-like observer stating every prediction. Pass rule, fixed before running: expected contrasts recovered in at least 80% of replicates, and any other contrast in at most 10% of replicates. Results: `research/reproduction/recovery_stimulus_interventions.json` (see appendix).

## Human predictions by cue

The primary contrasts below are the only confirmatory ones. Evidence grades and scope are copied from `research_notes/VLM versus human emotion mechanisms/human_mechanisms.md`.

**Almost none of these baselines were measured on EU-Emotion, and none on these edits.** Most human papers were read at abstract level. **BASIC ONLY** marks a prediction whose human evidence covers basic emotions only. For those, any reading on complex labels is out of scope. Rule 4 of `RESEARCH_QUESTIONS.md` counts a cue as human-aligned only if two independent sources agree, ideally one randomised and one on natural clips. The last column records that.

| ID | Cue and contrast (within item) | Predicted direction | Source | Grade | Scope | Rule 4 |
|---|---|---|---|---|---|---|
| P1 | `mask_mouth` on Happy | Acted label falls | Wegrzyn 2017; Blais 2012 | Strong | **BASIC ONLY** | Two sources, no randomised AU source checked here |
| P2 | `mask_mouth` on Disgusted | Acted label falls | Wegrzyn 2017 | Strong | **BASIC ONLY** | One source |
| P3 | `mask_eyes` on Afraid | Acted label falls | Wegrzyn 2017 | Strong | **BASIC ONLY** | One source; Blais 2012 finds the mouth most informative across eight expressions, so size may be small |
| P4 | `mask_eyes` on Sad | Acted label falls | Wegrzyn 2017 | Strong | **BASIC ONLY** | One source; same caution |
| P5 | `mask_brows` on Angry | Acted label falls | Ponari 2012 (upper half for anger); notes' takeaway | Moderate | **BASIC ONLY** | One source, for the upper face rather than brows alone |
| P6 | `mask_mouth` minus `mask_eyes`, Happy and Disgusted against Afraid, same 8 actors | More negative on Happy and Disgusted | Wegrzyn 2017 | Strong | **BASIC ONLY** | Label-by-region interaction inside one actor set |
| P7 | `mask_mouth` on Happy, Sad and Angry, Neutral family | Neutral rises | Carbon 2020; Grundmann 2021; Pazhoohi 2021 | Strong, from face masks | **BASIC ONLY** | Multiple sources, natural occlusion, different occluder |
| P8 | `mask_mouth` on Disgusted, Angry family | Angry rises | Carbon 2020 | Strong, from face masks | **BASIC ONLY** | One source |
| P9 | `sf_low` minus `sf_high` on Happy | Positive: low-pass spares Happy | Kumar and Srinivasan 2011 | Moderate | **BASIC ONLY** | One source. Use depends on task (Smith and Merlusca 2014) |
| P10 | `sf_low` minus `sf_high` on Sad | Negative: high-pass spares Sad | Kumar and Srinivasan 2011 | Moderate | **BASIC ONLY** | One source. Low-pass results are never read as a subcortical analogue |
| P11 (out of the confirmatory family until the dose is verified) | `aperture_wide` minus `aperture_narrow`, all items, CLR of {Interested, Surprised, Afraid} minus CLR of {Unfriendly, Sneaky} | Positive | Lee and Anderson 2017; Lee et al. 2013 | Moderate, mainly one lab | Complex and basic | Same lab. Mapping their sensitivity and discrimination clusters onto EU-Emotion labels comes from the notes, not the paper |
| P12 | Mean of `shuffle_a`, `shuffle_b` on Happy and Surprised | Acted label falls | Richoz 2024; Bould 2008 | Strong | **BASIC ONLY** | Two sources. Richoz found no dynamic gain at full signal except for these two labels |
| P13 | Mean of shuffles on Low Intensity items (48 items, 17 actors) | Acted label falls | Ambadar 2005; Bould and Morris 2008; Richoz 2024 | Strong | **BASIC ONLY** | Multiple sources |
| P14 | `trunc_first_third` minus `trunc_last_third` on Afraid, Surprised family, and on Surprised, Afraid family | Confusable label higher on the early third | Jack 2014 | Moderate, synthetic faces; AU order checked in the main text | **BASIC ONLY** | One source; supplement unread. Main text read 2026-11-07: confusions at t3 of 10, Surprised and Afraid separate at t6, Disgusted and Angry at t7. Its classifiers are cumulative and use reverse-correlated models with time scaled to the animation, so these clip thirds are an analogue, not a replication. Afraid and Surprised come from separate actor sets, so tested separately |
| P15 | Same contrast on Angry, Disgusted family, and on Disgusted, Angry family | Confusable label higher on the early third | Jack 2014 | As P14 | **BASIC ONLY** | As P14 |

### No directional prediction: estimated and reported

| Cue | Reported as | Why |
|---|---|---|
| Eye-only views | `eye_crop` and `eye_window` by label, complex against basic | Contested. Baron-Cohen 1997 predicts complex labels hold under eyes alone and basic labels drop. Blais 2012 and Cunningham 2005 predict a mixed picture. Reported as which account the model resembles, not as a pass or fail |
| `reverse` | Change from `original` | Any order-sensitive observer should change and a frame-bag cannot (Reinl and Bartels 2015). This describes the model's temporal code. No human direction is given |
| `shuffle` overall | Change from `original`, with `static_apex` | First establishes whether temporal information reaches the judgement at all, which a speed test requires |
| Complex labels under any cue except P11 | Descriptive | Bored, Worried, Disappointed, Kind and Unfriendly have no located reverse-correlation model. Sneaky, Hurt, Joking, Excited and Disappointed have no specific cue literature |
| Masks on complex labels | Descriptive | Grade is contested for complex states. Cunningham 2005 and Nusseck 2008 suggest that the region varies by expression |

### Head angle: excluded as a category-flip prediction

**No prediction of a category flip from head angle is made, here or in any later amendment drawn from this one.** No located study shows head pose alone flipping happy to sad. The located evidence is that a downward tilt raises perceived dominance on neutral faces through apparent brow lowering (Witkower and Tracy 2019, abstract level), and Mignault 2003 and Hess 2007 were not retrievable. This is an absent study, not a shown human null. If head angle is tested later, a flip is reported as "no human precedent located", not as a failure. This suite does not edit head angle, because that needs a generator. Rigid head motion is present in every clip, since the face crop is fixed.

## Decision rule per prediction

| Outcome | Requires |
|---|---|
| **Matches prediction** | Gate B passed; route delivery passed; step-down p < .05 in the predicted direction on the blind readout; fill rule met |
| **Opposes prediction** | Step-down p < .05 two-sided in the opposite direction |
| **Null** | All conditions in the next section met |
| **Inconclusive** | Anything else, including a sham or delivery failure |

Every result states its scope tag. A BASIC ONLY match is a claim about basic labels in this stimulus set. The MCQ bridge is reported beside the blind result, and disagreement between the two is reported as such.

## Null-result rule

A null in a cue counts as a null only if all of these hold. Otherwise the result is inconclusive and described as such.

1. A1, G2 to G4, and Gate B passed for that model and route.
2. Delivery passed. For temporal cues, every frame position changes the output. For spatial cues, the edited region spans at least 1 token on that route.
3. The dose was shown effective elsewhere on the same route: the model's change under `mask_mouth_blur` plus `mask_eyes_blur` on the unedited item, or the full mismatch for a temporal cue, exceeds the margin below.
4. Joint ablation is also null: `mask_eyesmouth_neutral` for region cues, `static_apex` for temporal cues. A region null without it is reported as "single-region null", because redundancy is not ruled out.
5. Equivalence passes: the 90% actor-bootstrap CI of the contrast lies inside the margin. **Margin [proposed]:** 10% of that model's mean absolute CLR change under the full image mismatch of B4, fixed per model before any intervention contrast is computed.
6. Divergence from the human prediction is reported alongside.

## Human response distribution and weak labels

Every result appears twice: against the acted label, and against the human response distribution.

- **Flagged items.** Kind (9), Unfriendly (10) and Jealous (7), 26 items. Face-only humans chose the target on 23 to 26% of these (chance 17%; `data/eu_emotions_human_entropy.json`). Each output row carries `weak_label`.
- **Scoring.** The acted label is not ground truth for these items. A model that answers Bored on an Unfriendly clip agrees with people. Label-level tables show these three labels separately, with the human modal label beside the acted label.
- **Leakage flag.** Model accuracy well above human on these items is treated as possible non-facial leakage, not as skill.
- **No flagged label is a target of a confirmatory contrast.** P11, which includes them, is out of the family while the dose is unverified. If it returns, it is repeated without them.
- **Reading divergence.** Change in divergence from the human distribution under an edit is descriptive. Humans did not see the edited clip, so this does not show what they would do.
- **Item filtering.** Disgusted (item range .01 to .96) and Angry (.14 to .91) also hold failed items. Results are repeated without items whose human mode is not the target.

## Exclusions and sensitivity

No item is excluded from the primary analysis except by a logged failure. These are reported beside it:

- Excluding the 3 smoke items.
- Excluding cells with `align_flagged`, where the donor residual is above 5% of eye span.
- Excluding apex frames where the first-frame donor is the sent frame.
- Excluding Neutral clips acting as their own neutral donor.
- Canonical word only, instead of the family sum, to check family-size effects.

Family sizes differ, so raw between-family comparison is biased. Within-item change removes this to first order, and this check tests that assumption.

## Limits

- Human predictions come from other stimuli, often synthetic or still, mostly on basic labels. The review's rule stands: no located human baseline exists for these edits on EU-Emotion.
- Rule 5's human response predictor exists (`scripts/human_response_predictor.py`) and is validated on held-out Snoek only (appendix). It is a model of humans, never human data. It maps AU amplitudes to labels, so it cannot yet speak to any edit in this suite: masks, filters and frame order are not AU changes, and Snoek amplitudes and detector intensities do not share units. Predictions here therefore stay literature directions.
- Prefill changes recognition into completion, and lexicon coverage may be low. Gate B checks signal, not whether the task is the same.
- Masks fill a region with a neutral face rather than removing information, so a mask effect may mean adding a neutral cue. The three fills bound this.
- Frames 1 fps apart cannot carry onset velocity. Eight uniform frames test order, not speed.
- Actor and label are only partly crossed.

## Open choices before freezing

1. Thresholds marked **[proposed]**: Gate B rules, the equivalence margin, SF cutoffs.
2. Whether 243 items run on all three models or a reduced condition set is used; see Cost.
3. Whether blind prefill text is accepted as is. A prompt change needs a new smoke run and a new `design_sha256`.
4. Confirm Jack 2014 from the full text, or remove P14 and P15.

### Recommendations, awaiting sign-off

These are the implementer's recommendations, not decisions. They bind nothing until reviewed. A1 has not been read, so choice 2 may change.

1. Accept the **[proposed]** thresholds as drafted. Any change must come before any full run.
2. Run all 243 items on Gemma and Qwen: 28.1 GPU-hours of cell time, about 31 with the 10% allowance. Decide on Molmo after A1. It fails G2 and G4, so it is not eligible as drafted, and including it would need its own amendment.
3. Accept the blind prefill text. It was chosen by the DECISIONS rule on calibration clips, and changing it now would need a new smoke run and a new `design_sha256`.
4. Keep P14 and P15, at their current **BASIC ONLY** and moderate standing. The main text of Jack 2014 was read on 2026-11-07 from the Glasgow repository copy (https://eprints.gla.ac.uk/88928/), and it supports their direction. The supplement was not read. A result here would show the analogue on natural clips, not replicate the paper.

## Cost

Measured on `smoke_v2`: A100, bfloat16, full forwards on every model, the chosen readout, 3 items. An item has 29 cells: 20 on `single_face`, 7 on `video_face`, 1 each on `single_full` and `legacy_full`.

| Model | `single_face` s per cell | `single_full` | `video_face` | Forwards per cell | Peak GB | GPU-hours, 243 items |
|---|---|---|---|---|---|---|
| Gemma 4 | 3.3 | 3.2 | 14.6 | 31 | 18.3 | 11.6 |
| Qwen3-VL | 5.4 | 17.9 | 17.0 | 46 | 23.5 | 16.5 |
| Molmo2 | 7.7 | 11.1 | 8.1 | 46 | 17.1 | 15.0 |

Hours are cell time only, so allow about 10% for decoding and delivery checks: about 47 GPU-hours in total. `legacy_full` is left out because it is not a causal route. If Molmo's `video_face` stays out, since it fails gate 1, Molmo falls to 11.2 hours.

**Wall time.** `--shard` is enough, and no offset splitting is needed. Items go to shards by position in the round-robin actor order, so shards are balanced by actor. Three shards per model (`sbatch --array=0-2`) give about 4 to 6 hours each against the 12 h limit. Each shard writes its own cells file, which is concatenated before analysis.

## Appendix: recovery and predictor

**Recovery** (`research/reproduction/recovery_stimulus_interventions.json`, 100 replicates, after the leave-target-out fix): all five observers pass. Expected contrasts were recovered in 100% of replicates. Replicates with an unexpected contrast: prior-only 6%, frame-bag 7%, mouth-only 8%, eyes-only 8%, human-like 0%. Two families at alpha .05 sit near the 10% limit, so these rates are close to it by construction. Aperture is not in the family.

**Predictor** (`research/reproduction/human_response_predictor.json`). A model of human judgement, never human data.

| Dataset | Unseen subjects and stimuli: log loss (base rate), r | Slopes from the other group: log loss (base rate), r |
|---|---|---|
| basic_WC | 1.618 (1.935), .69 | 1.658 (1.928), .66 |
| basic_EA | 1.567 (1.906), .82 | 1.624 (1.903), .77 |
| conversational_WC | 1.494 (1.613), .59 | 1.479 (1.590), .63 |
| conversational_EA | 1.491 (1.591), .64 | 1.481 (1.555), .65 |

## Appendix: smoke run

Job 39974985, Gemma 4 E4B, 3 items: `emotions_38/EDITED/Hurt/OF10_cut.mp4`, `emotions_7/EDITED/Interested/JF21v1_cut.mp4`, `emotions_4/EDITED/Sneaky/HF26_cut.mp4`. Mechanics only.

**The prompt in this first smoke was replaced.** After "They feel", Gemma 4 put almost all probability on the markdown token " **" under the original prompt, and lexicon mass on cells was about 2e-5 to 0.004. The readout ladder then chose the current prompt on calibration clips. The B2 rule of 0.05 was not lowered.

**Aperture dose is not verified.** Re-detected lid gaps on 5 clips were 1.02 to 1.24 for nominal 1.25 and 0.77 to 0.93 for nominal 0.75. Direction was right, but size fell short of nominal. FaceMesh may under-read lids, or the warp may do less on real faces than on synthetic ones. Resolve this before freezing P11.

**Other.** 2 of 96 cells failed, both `eye_closed` on one apex frame, and were logged. One clip had donor alignment flagged on all neutral and first-frame masks (residual about 0.11 of eye span), so head pose at the apex can exceed the similarity fit.

## Appendix: smoke_v2

Jobs 39995347 to 39995349, the same 3 items, one run per model on the chosen readout. Mechanics only.

- **Run:** 85 of 87 cells ran on each model. The 2 failures are `aperture_wide` and `aperture_narrow` on `JF21v1`, reason `eye_closed`, as in the first smoke. Aperture is out of the family.
- **Checks:** the sham equals the original exactly, repeat forwards change nothing, and `design_sha256` starts `a1924d80` on all three.
- **Coverage on cells, median lexicon mass:** Gemma .58 to .71, Qwen .82 to .96, Molmo .66 to .71 by route.
- **Lexicon mass is not image signal.** Image-free mass on `single_face` is .49 on Gemma, .62 on Qwen and .42 on Molmo. Grey is .02, .08 and .46. Coverage (B2) can pass on the prior alone, so B3 to B5 carry the image claim.
- **The image-free baseline depends on route on Qwen.** Mass is .62 on `single_face` and `video_face`, but .001 on `single_full` and `legacy_full`. So its use as a prior in B3 needs review before freezing.
- **Alignment:** 8 cells flagged per model, all on `OF10`, as in the first smoke.
