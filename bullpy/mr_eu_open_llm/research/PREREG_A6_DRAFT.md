# DRAFT amendment A6: shared errors on CREMA-D (RQ5)

**Status: DRAFT. Not frozen, not tagged, not preregistered.** Nothing here binds until it is audited, committed and tagged. No model has seen a CREMA-D analysis clip, and no run starts until it is frozen and the wait below has ended.

Draft `design_sha256` of the block below: `a2c11e3b783f3c57e669b118750eb0b510c5ef82867ddeadb92ccf9a2001b8dd`. It changes with any edit to the block and is restated at the freeze.

**Written:** 2026-11-07, before any A1 result was read and before any model output on the 819 clips exists.
**Goal:** G8 in `research/GOALS.md`.
**Depends on:** G1. RQ0 is read through the frozen A1 and A3 table by A2's `rq0_status`, per model. Gates G2 to G4 as in A2.
**Models.** Gemma 4 and Qwen3-VL. Molmo2 fails G2 and G4, and any route back needs a dated rule before new data, as in A2.
**Code, not yet written:** `scripts/a6_design.py`, `scripts/a6_run.py`, `scripts/a6_analyse.py`, `scripts/a6_recovery.py`, `research/scripts/a6_human_target.py`, `slurm_jobs/study3_a6.sh`, `tests/test_a6.py`. All are new files. `a6_design.py` parses the design block from this file and refuses a hash mismatch. Frozen A2 and A3 code is imported unchanged, with hashes pinned below.

**What was seen before writing this.**

- CREMA-D face-only votes on the selected clips, to state the human target. The pair rule was written with the table below in view, so it is not blind to votes. No model output on the 819 clips exists, so it is blind to that.
- The 24 calibration clips, which are outside the 819 (checked: no overlap), were seen by models for the readout ladder, with no labels or votes read.
- No A1 result. A3's result as stated in `STATE.md`.

## Question

RQ5. When a human and a model make the same mistake, is it for the same reason? The candidates are the same facial cue, shared label semantics, and the options offered.

**Why CREMA-D.** Every clip offers the same six options, so human confusion does not depend on which foils appeared, unlike EU-Emotion. And all 91 actors portray every emotion, so label contrasts are within actor. Neither holds on EU-Emotion. Complex states are out of scope here; Bored-Unfriendly stays an EU-Emotion limit.

**Scope.** Causal claims are about the model. Humans are a fixed target from silent video. This work never states how humans would respond to an edited clip, only whether the model matches a stated prediction. Human cue claims come from other stimuli, and each carries its read status.

## Design block

All fixed constants are here and nowhere else. The hash is sha256 of the bytes between the fence lines, excluding the fences, with a final newline. Items marked PENDING are filled from outputs that involve no model, before freezing.

```json
{
  "amendment": "A6",
  "selection_sha256": "81bab36faadab729a4623d9761e93c8cd46bf3bceac497364246a7734c0bc3b5",
  "a2_code_sha256": {
    "scripts/stimulus_interventions.py": "93c601eb42d6700d4f0fc8ba72317ed5c2e354105d57fcac611c59660125deb3",
    "scripts/run_stimulus_interventions.py": "f7880c94e9671cc294cd4eaeece2108d7cd1d7a4c7e6f9e30a1c495964e6cd82",
    "scripts/analyse_stimulus_interventions.py": "aba4f533daad6210e87f7bb4e68fc4627805cb4dfd9d9308df734fada4edc8b1"
  },
  "a3_code_sha256": "8a681a652d8e34065372d4f45bce650ba5276b5138e3aa64cf4310d65e93d818",
  "geometry_sha256": "PENDING",
  "human_target_sha256": "PENDING",
  "models": ["gemma4", "qwen3vl"],
  "primary_route": "single_face",
  "bridge_route": "video_face",
  "n_items": 819,
  "n_actors": 91,
  "seed": 42,
  "options": ["Anger", "Disgust", "Fear", "Happy", "No Emotion", "Sad"],
  "family_map": {"Anger": "Angry", "Disgust": "Disgusted", "Fear": "Afraid", "Happy": "Happy", "No Emotion": "Neutral", "Sad": "Sad"},
  "prime_words": {"Anger": "angry", "Disgust": "disgusted", "Fear": "afraid", "Happy": "happy", "No Emotion": "neutral", "Sad": "sad"},
  "prime_prompt_prefix": "A person feels {word}.",
  "orders": [
    ["Anger", "Disgust", "Sad", "Fear", "No Emotion", "Happy"],
    ["Disgust", "Fear", "Anger", "Happy", "Sad", "No Emotion"],
    ["Fear", "Happy", "Disgust", "No Emotion", "Anger", "Sad"],
    ["Happy", "No Emotion", "Fear", "Sad", "Disgust", "Anger"],
    ["No Emotion", "Sad", "Happy", "Anger", "Fear", "Disgust"],
    ["Sad", "Anger", "No Emotion", "Disgust", "Happy", "Fear"]
  ],
  "edit_order_rows": [2, 5],
  "pseudo_count_human": 0.5,
  "target_pair": ["Anger", "Disgust"],
  "shared_family": [["Anger", "Disgust"], ["Fear", "Sad"], ["Disgust", "Sad"], ["No Emotion", "Sad"]],
  "stage3_emotions": ["Anger", "Disgust", "Fear", "Sad"],
  "control_emotions": ["Fear", "Sad"],
  "regions": {
    "upper": ["A2 eyes rings", "A2 brows rings"],
    "mid": {"hull": [133, 362, 129, 358, 2]},
    "lower": ["A2 mouth ring"],
    "midlower": ["mid", "lower"],
    "uplow": ["upper", "lower"],
    "all": ["upper", "mid", "lower"]
  },
  "region_dilate_fraction": 0.10,
  "conditions": [
    "original", "sham_self_fill",
    "mask_upper_neutral", "mask_upper_first", "mask_upper_blur",
    "mask_mid_neutral", "mask_mid_first", "mask_mid_blur",
    "mask_midlower_neutral", "mask_midlower_first", "mask_midlower_blur",
    "mask_lower_neutral", "mask_uplow_neutral", "mask_all_neutral"
  ],
  "blind_conditions": ["mask_upper_neutral", "mask_mid_neutral", "mask_midlower_neutral", "mask_all_neutral"],
  "alpha": 0.05,
  "ci_decision": 0.95,
  "ci_equivalence": 0.90,
  "signflip_draws": 100000,
  "bootstrap_draws": 10000,
  "shared_margin": 0.05,
  "cue_margin_fraction": 0.10,
  "lost_item_limit": 0.05,
  "text_neighbour_rank": 5,
  "min_span_tokens": 1.0,
  "token_floor": 8.0,
  "recovery": {
    "replicates": 500,
    "expected_verdict_min": 0.80,
    "other_named_verdict_max": 0.10,
    "inconclusive_max": 0.20,
    "seed": 6042
  }
}
```

`shared_margin`, `cue_margin_fraction`, `text_neighbour_rank` and the mid-face landmark list are **[proposed]**. The mid-face indices are drafted from the FaceMesh layout and need a check on calibration clips (see Fixed before freezing).

## Human target, fixed before model data

**Data.** Face-only votes in `research/reproduction/crema_d_selection.csv` (sha256 in the design block), 819 clips, 91 actors, about 10 raters per clip. Chance is 1/6, and a target is the actor's intended emotion. Per `CREMA_D_PLAN.md`, the votes come from `queryType` 2, checked against the raw responses. `research/scripts/a6_human_target.py` will reproduce the table below, writing `research/reproduction/a6_human_target.json`.

**Mean vote share**, row is the intended emotion:

| Intended | Anger | Disgust | Fear | Happy | No Emotion | Sad |
|---|---|---|---|---|---|---|
| Anger | .585 | .206 | .039 | .021 | .123 | .027 |
| Disgust | .080 | .540 | .090 | .015 | .058 | .217 |
| Fear | .080 | .100 | .467 | .013 | .196 | .143 |
| Happy | .003 | .014 | .013 | .874 | .085 | .010 |
| Neutral | .041 | .076 | .044 | .031 | .727 | .080 |
| Sad | .033 | .095 | .143 | .020 | .391 | .318 |

**Pair excess.** A confusion must not be a bias towards one option, or generic difficulty. So for system S, on clip i, a confusion of A with B is measured with A left out, and against the same measure on other emotions, within actor:

    r_i(B given not A) = p_i(B) / (1 - p_i(A))

    X_a(A to B) = mean r_i over actor a's clips of A,
                  minus mean r_i over actor a's clips of emotions other than A and B

    X(A, B) = mean over actors of [ X_a(A to B) + X_a(B to A) ] / 2

For humans, p_i are vote shares with `pseudo_count_human` added per option. For models, p_i are exact option probabilities, averaged over orders. The two scales are comparable, as each is a share of errors.

**Pair rule.** No model output enters.

1. Rank the 15 pairs by human X.
2. The **target pair** is the highest-ranked pair of two emotions. No Emotion is excluded from the target, because any occluder removes signal, and a rise in No Emotion cannot separate lost signal from a confused cue (A2, P7, and Carbon 2020).
3. The **shared family** is the top three pairs of two emotions, plus the top pair that includes No Emotion.

**Result, human X with actor-bootstrap 95% CI, 10,000 draws:**

| Pair | Human X | 95% CI | Role |
|---|---|---|---|
| No Emotion, Sad | .206 | .181 to .231 | Family |
| **Anger, Disgust** | **.198** | **.173 to .223** | **Target** |
| Fear, Sad | .131 | .101 to .160 | Family |
| Disgust, Sad | .130 | .104 to .156 | Family |
| Happy, No Emotion | .105 | .084 to .126 | Not selected |

Directed: Anger read as Disgust .273, Disgust read as Anger .123, with 87 and 69 of 91 actors above zero. So the human confusion runs mostly one way.

**Stability.** Anger and Disgust stay the top emotion pair with no pseudo-count (.300) and on all 7,442 clips (.184). Third place is close: Fear-Sad and Disgust-Sad swap on all clips (.115 against .132), and both are in the family either way.

**Clips that read as the other label.** RQ5's design constraint asks whether a confusion is a real ambiguity or clips that simply read as the other label. `a6_human_target.py` reports the share of Anger clips with a Disgust mode, and the concentration of Anger-to-Disgust votes across clips. Every result is repeated on clips whose human mode is the intended emotion.

**Human cue check on the same clips.** Under rule 3 of `RESEARCH_QUESTIONS.md`, within Anger and within Disgust clips, the human partner share is regressed on detector estimates of AU9 and AU10 at the apex, with both detectors, by the same-clip code. It involves no model and runs before any model. It is reported and gates nothing. If it fails to show the link, the mid-face prediction rests on the literature alone and is labelled so. Speech movement may weaken it.

## Stimuli and input

- **Clips.** 819 selected clips, `<clip_id>.flv`, 480 by 360, 29.97 fps, 1.40 to 4.87 s, on CSD3. No audio is used. Raters saw them silently. Frames are decoded and edited in memory, and no frame or edit is written, kept or published.
- **Routes.** `single_face` is primary, because A3 tested it and a lip corner is sub-token on a grid. `video_face` is a bridge, descriptive only, because humans saw motion.
- **Frames and geometry.** A2's rule: 8 uniform frames, landmark apex, one fixed square crop at 1.5 times the largest landmark extent, 768 by 768, MediaPipe FaceMesh 0.10.14. A new geometry file is pinned in the design block. A2's EU geometry check is not used for it.
- **Frame count.** `count_frames` counts decodable frames, so the three h264 clips that overstate their header stay usable and lose no index. The plan's expected loss of one frame applied to seeking, which this route does not do.
- **Neutral donor.** Every actor has one or two Neutral clips (46 and 45 actors), so A2's neutral fill applies. The donor is the middle sent frame of the first by clip id. Where a Neutral clip would donate to itself it is not edited, since Neutral clips are not in Stage 3.

## Task and scoring

- **MCQ.** `build_4afc_prompt(options, condition="video_only")`, as in A1 and A3, with prefill "EMOTION:". Each option's joint logprob is renormalised over the six options, as in A3. An item is correct if the target is strictly highest; ties are wrong.
- **Blind readout.** A2's prompt, prefill and lexicon. Six families are read through `family_map`. It adds a route with no option list, so it can show whether a confusion needs the options at all.
- **Option wording.** "No Emotion", as raters saw, not "Neutral".

**Option rotation.** Option order is known to move VLM emotion judgements (Bhattacharyya and Wang 2025, not faces). All clips run all six `orders` in Stage 1. These replace the cyclic rotation in the plan and adapter, which is not used here and is not edited. Cyclic rotation keeps neighbours adjacent: Anger and Disgust are next to each other in 5 of 6 cyclic rotations, so the target pair would be confounded with adjacency. The `orders` are a Williams square: each option takes each position once, and each ordered pair of options is adjacent exactly once. So every pair is adjacent in 2 of 6 orders. Cost is the same as six rotations.

The on-screen order raters saw is unknown (`CREMA_D_PLAN.md`), so human position effects stay unknown.

**Real against mismatch.** Partners come from `vision_mismatch_partners`, seed 42, on these clips. Checked: all 819 partners differ in emotion and actor, and each face is used once. **The prompt depends only on the order, not the clip, so a clip's mismatch cell is its partner's real cell on the same order.** Scoring is deterministic, so no extra forward is needed.

## Stages

Each stage is read only after the one before it has a result. Stage 3 runs only for a model that needs it.

**Stage 0: gates and the wait.** See the next section.

**Stage 1: validity on CREMA-D.** RQ0's design on `single_face`. For clip i, a_i is mean correct over six orders, and b_i the same on its partner's face. delta = mean(a_i - b_i). A model is vision-dependent on CREMA-D if the one-sided actor-level sign-flip p (`actor_sign_flip_p`, 100,000 draws) has Holm p < .05 across the eligible models, and the actor-bootstrap 95% CI lies above 0, the same rule as A3.

Also here: `image_free` and `grey` for each order, the blind readout on 819 originals, and the text primes. Secondary: accuracy against 1/6, accuracy by target position, within-emotion correlation with human p_target against the ceilings in `crema_d_reliability.json`, and Gate B's B1 to B5 on CREMA-D's six families, by the frozen rules. The B checks are reported and do not gate.

**Stage 2: is the error shared, and is it the options?**

- **Shared.** Model X on the four family pairs, one-sided above zero, step-down maxT, one model at a time, with an actor-bootstrap 95% CI. Only the target pair sets the verdict; the others say how general sharing is.
- **Pattern, descriptive.** Rank correlation of human and model X over the 30 directed cells, against a split-half ceiling from raters. Error consistency in the sense of Geirhos et al., on the modal correct pattern over shared clips.
- **Options.** O = X on the 2 orders where the pair is adjacent, minus X on the other 4. Blind X is X on the blind readout's six families.
- **Semantics.** Text-only primes: for each `prime_words` word, the prompt is `prime_prompt_prefix` plus the MCQ text, with no image, on all six orders. X_text is X with primes in place of clips. This is one deterministic value per directed cell, so it has a rank and no test.
- **Margin locked.** The cue margin for Stage 3 is computed here, and it is written to the output before Stage 3 starts.

**Stage 3: cue-targeted test.** Only for a model with a shared target pair not attributed to options. Clips of the four `stage3_emotions`, 547 in all, on `single_face`, with every `conditions` entry at both `edit_order_rows`. Those two orders follow a rule: the Williams rows in which the target pair is not adjacent and neither member sits at an end. Within-clip deltas hold order fixed.

## Edits

Edits come from A2's frozen code: `read_frames`, landmarks, `align_to`, `_fill_image`, `blend`, `crop`, `fixed_size`, the fill types, the feather, and `clr`. `upper` and `lower` are unions of A2's rings. `mid` is new, because A2 has no nose region and its disclosed limit on P8 is that a mouth mask may leave a nose wrinkle visible. `a6_run.py` holds a mask function matching `region_mask` line for line, because `REGIONS` is a frozen constant and cannot take a new key. A test holds its output equal to `si.region_mask` on `mouth`.

| Condition | Definition |
|---|---|
| `original` | Unedited apex frame |
| `sham_self_fill` | Full pipeline, filled from the frame itself. Must equal `original` exactly |
| `mask_upper_{neutral,first,blur}` | Eyes and brows |
| `mask_mid_{neutral,first,blur}` | Nose bridge to below the nose: AU9 wrinkle and the region above the lip |
| `mask_midlower_{neutral,first,blur}` | Mid and mouth, the analogue of a nose and mouth covering |
| `mask_lower_neutral` | A2's mouth, descriptive bridge to P8 |
| `mask_uplow_neutral` | Upper and mouth, so mid-face stays visible |
| `mask_all_neutral` | Upper, mid and mouth together |

**Outcome.** For a clip of emotion A with partner B: y = CLR of B, with A and No Emotion left out of the reference, so among B and the three other emotions. Leaving out A follows A2's fix, so a plain fall in A cannot read as a rise in B. Leaving out No Emotion removes the generic route by which an occluder raises No Emotion.

**Contrast.** delta(i, c) = y(i, c) - y(i, `original`), on the same order, averaged over the two orders. The confirmatory effect is a within-actor double difference, so that it moves this pair and not others:

    e(a) = mean delta over actor a's clips of A,
           minus mean delta over actor a's clips of the control emotions,
           both on the same outcome word B, with the same two options left out

All 91 actors are in every cell, so this is valid at actor level, unlike A2's cross-set contrasts.

### Predictions

| ID | Edit, neutral fill | Clips | Partner | Predicted | Basis | Standing |
|---|---|---|---|---|---|---|
| C1 | `mask_mid` | Anger | Disgust | Falls | Yan et al. 2025: confused because they share nose wrinkler and upper lip raiser. Jack et al. 2014: nose wrinkler carries the early confusion | Shared cue removed. Human evidence on synthetic faces, 10 participants; a classifier analysis in Jack |
| C2 | `mask_mid` | Disgust | Anger | Falls | As C1 | As C1 |
| C3 | `mask_midlower` | Disgust | Anger | Rises | Carbon 2020: 38% against 2% under a nose and mouth covering, descriptive, static photographs | Separator removed |
| C4 | `mask_upper` | Anger | Disgust | Rises | Ponari 2012 and Wegrzyn 2017 place anger's information in the upper face. The direction towards Disgust is inferred | Weakest. Inferred, abstract level for Ponari |

**BASIC ONLY** applies to all four. None was measured on CREMA-D or on these edits.

**The sources do not fully agree.** Yan lists upper lip raiser as shared, and Jack places its left form as a late discriminator. Carbon's covering hides the nose wrinkle and increases the confusion, whereas Yan implies that hiding it should reduce the confusion. C2 and C3 can both hold, since C3 also hides the mouth, but a model cannot be read against one settled human account. So C1 and C2 test Yan's stated cue, the cue-targeted test RQ5 asks for, and C3 and C4 are reported in the same family.

| C1, C2 | C3 | Reading |
|---|---|---|
| Fall | Rises | Fits Yan and Carbon together: the nose carries the confusion, and the mouth separates |
| Fall | No rise | Fits Yan. Carbon's analogue not shown |
| No fall | Rises | Fits Carbon's analogue. Not Yan's cue |
| Rise | Any | The mid-face separates this pair in this model, against Yan |

## Inference

- **Unit.** The actor, 91 actors, fully crossed with emotion: 1 or 2 clips per actor-emotion cell, 9 clips per actor.
- **Test.** Sign-flip of actor rows, Monte Carlo, because 2^91 cannot be enumerated: `signflip_maxt` with 100,000 draws, identity included, seed 42. Step-down maxT at familywise .05 within one model, on Stage 2's four pairs, and on Stage 3's C1 to C4. Directions as stated, and two-sided for Opposes as in A2. Stage 1 uses A3's test and Holm.
- **Floor.** About 1e-5.
- **Interval.** Actor-bootstrap, 10,000 draws, 95% for direction, 90% for equivalence.
- **Fill rule.** The neutral fill is primary. A match counts only if the first and blur fills have the same sign, and the simple change on A clips has the predicted sign.
- **Cue margin.** 10% of the model's mean absolute change in the same outcome on the same clips under full mismatch, fixed at Stage 2. This avoids A2's disclosed fault of a margin taken from another outcome.
- **Shared margin [proposed].** .05 on X, about a quarter of the human value. At the human actor SD for this pair (.12) the 90% half-width is .02, and a null stays attainable up to an actor SD of about .29.
- **Models.** Never pooled.

## Decision table

Per model, on the target pair. Rows are read in order, and the first that applies is the verdict.

| Verdict | Requires |
|---|---|
| **Not tested** | RQ0 on `single_face`, G2 to G4, delivery, or geometry not passed for this model, or over `lost_item_limit` of clips lost. Stops |
| **No face use on CREMA-D** | Gates passed, Stage 1 failed. Named, and stops: no shared-error claim |
| **Not shared** | Stage 1 passed; X not significant and its 90% CI inside `shared_margin`. The null at Stage 2. Stops |
| **Option-dependent** | X passes; O significant at step-down .05, X on non-adjacent orders with its 90% CI inside `shared_margin`, and blind X not significant. Stops |
| **Same cue** | X passes, not option-dependent; sham equal; C1 and C2 both match, with the fill rule |
| **Same cue, one direction** | As above, with exactly one of C1 and C2 matching, and the other not opposing |
| **Opposes the stated cue** | As above, with C1 or C2 significant in the opposite direction |
| **Not this cue** | The cue null in the next section met on both C1 and C2. The null at Stage 3 |
| **No tested feature** | C1 and C2 not significant, the dose passes, and `mask_all_neutral` leaves the pair inside the margin. Then the shared error does not need these regions |
| **Inconclusive** | A reason code from the list below |

**Rider: label semantics.** With Not this cue or No tested feature, the text-prime rank of the directed target cells is reported. A rank at or within `text_neighbour_rank` of 30 reads as "label neighbours are a live account". This is a correlational reading, and A6 does not show that semantics caused the error.

**Rider: redundancy.** `mask_uplow_neutral` keeps only the mid-face. If y stays within the margin of `original` on a null, the label `sufficient_not_necessary` is attached, and the verdict does not change.

Every verdict states its scope: a claim about this model, these 91 actors, this clip set, single apex frames. The bridge route and blind readout are reported beside it, and any disagreement is stated.

## Null and Inconclusive

**Null at Stage 2 (Not shared).** Stage 1 passed; X not significant; 90% CI inside `shared_margin`.

**Null at Stage 3 (Not this cue).** All of these, on both C1 and C2. All four contrasts can reach a null, on both models, so no contrast is capped.

1. Gates, Stage 1 and Stage 2 passed, and the sham equals the original exactly.
2. Delivery passed. The median over the contrast's clips of the edited region's narrowest side is at least `min_span_tokens` token wide. This uses span, not area, since A2 disclosed that an area rule can pass a long thin region.
3. The dose passed. The signed change on the acted label under the blur fill of the same region, as the mean over actors of actor means, has an absolute value above the margin. It is signed so that noise cancels, the correction from A2's second audit.
4. The pair can move. `mask_all_neutral` changes y by more than the margin in absolute value, on the same measure. Without this, the row is No tested feature.
5. Equivalence: the 90% CI of the double difference is inside the cue margin, and the test is not significant.
6. The mean and 95% CI in the predicted direction are reported.

A2 asks that a joint ablation is also null. That cannot apply here: a shared error is face-dependent, so masking the whole face should move it. So condition 4 asks the reverse, and redundancy is reported by the rider. This difference is deliberate, and it is why the null here means "not needed alone".

**Inconclusive.** Exactly these reason codes: `shared_unresolved` (X not significant, CI outside the margin), `option_unresolved` (O significant, but the other parts of the Option-dependent row unmet), `cue_unresolved` (C1 or C2 not significant, not equivalent), `sham_failed`, `region_too_small`, `dose_failed`.

**Inconclusive cannot be the default here, by construction.**

1. Every row is reachable for an eligible model, and all contrasts can reach a null, on 91 fully crossed actors.
2. The gate loader must return True on a fixture built from the real pinned paths, with a test that runs the verdict function end to end, so that no missing path makes every verdict Inconclusive.
3. A failed outside gate or Stage 1 gives Not tested or No face use, each a named result. Only checks inside Stage 3 can give Inconclusive.
4. Recovery must reach every named row, and hold Inconclusive under `inconclusive_max` for each observer not placed on a boundary. If not, the margin or rule is revised before freezing and dated.
5. A row with no verdict or reason code is a code fault and stops the analysis.

## Missing data

A face that cannot be localised fails its cells, and the reason is logged. A clip enters Stage 1 only with all six orders. Its partner must also be present. Failures are reported by emotion and actor. Over `lost_item_limit` of clips lost for a model gives Not tested. Lost clips are not replaced. A bad final JSON line is dropped and rescored, by A3's `read_cells`, since scoring is deterministic.

## Sensitivities

All are coded and reported beside the primary, and none changes a verdict.

- Clips whose human mode is the intended emotion.
- Outcome with No Emotion kept in the reference.
- Plain change on A clips, without the double difference.
- Control emotions one at a time.
- Each edit order separately.
- All other-emotion, other-actor faces in place of the single partner, on Stage 1, which is free since cells exist.
- Excluding `align_flagged` cells, and clips under `token_floor` tokens across the face.
- The three h264 clips with short decodes excluded.
- Actor sex, descriptive. Equal weighting of clips.
- The directed cells separately, since human confusion runs mostly from Anger to Disgust.

## Recovery

Before any model run, synthetic observers go through `a6_analyse.py` on the real 819 clips, 91 actors, partners and orders, by `a6_recovery.py`. The observers have option probabilities built from named components, and the same masks act on those components. The noise has actor and clip parts, and effects are graded at 0.5x, 1x and 2x of the human directed rates, since A2 disclosed that its recovery showed mechanics, not power.

| Observer | Construction | Expected verdict |
|---|---|---|
| Prior-only | One fixed distribution on every clip | No face use on CREMA-D |
| No sharing | Face use, errors spread evenly | Not shared |
| Adjacency | Pair excess only when the pair is adjacent | Option-dependent |
| Same cue | Pair excess carried by a mid-face component | Same cue |
| One-way cue | The same, on Anger clips only | Same cue, one direction |
| Reverse | Mid-face separates the pair | Opposes the stated cue |
| Other cue | Pair excess carried by an upper-face component | Not this cue |
| Spread | Error shares fixed by label neighbours, independent of region. Moves under `mask_all_neutral` | Not this cue, with the semantics rider |
| Global face | Pair excess from a whole-face component that survives all masks | No tested feature |
| Generic mask | Any mask raises No Emotion and lowers all foils | Must not reach Same cue |
| Generic pull | Any mask raises Disgust on all clips | Must not reach any match, since the double difference removes it |
| Human-like | Mid-face share set to the human rates, with Carbon's rise | Same cue |
| Boundary | Effects near each margin | Reported, not judged |

Pass rule: for each non-boundary observer at 1x and 2x, the expected verdict in at least `expected_verdict_min` of replicates, any other named verdict in at most `other_named_verdict_max`, and Inconclusive in at most `inconclusive_max`. The two Must not observers reach the forbidden verdict in at most `other_named_verdict_max`. 0.5x is reported as power and not judged. Every named row must be reached at least once. A failure is reported. Any change comes before any model run, and is dated.

## Order of work and the wait

Nothing below may change a file read by the pending rotation jobs before they finish (rule 4 in `GOALS.md`). Qwen arrays 39982137 and 39982138 were pending at the last entry in `STATE.md`, on 2026-11-07, with a Slurm estimate of 8 Nov about 19:55. Files they read include `scripts/evaluate.py`, `scripts/frame_sampling.py`, `scripts/prompts.py`, `config.py` and `src/`.

**`.flv` support: what is needed, and what is already there.**

- `CREMA_D_PLAN.md` says `load_stimulus_as_images` does not accept `.flv`. That line is out of date for the local tree: commit e969c05 (2026-11-04) added `.flv` to it. Whether the cluster copy has that commit was not checked, as no SSH was used here.
- `MEDIA_EXTS` in `scripts/evaluate.py` still lacks `.flv`.
- A6's route does not use either path. It decodes with `si.read_frames`, which takes any cv2-readable path, and the readout ladder has already decoded CREMA-D `.flv` this way on the cluster.

So A6 needs no edit to `evaluate.py`. **If the cluster's `frame_sampling.py` lacks `.flv`, syncing it must wait for the jobs to finish, like any other sync.** New files could be added earlier, but A6 does not do so, in order to keep one synced, hashed state.

**Order.**

1. Now, with no cluster change: audit this text, write the code, tests and recovery. Fill PENDING items that need no cluster.
2. After the rotation jobs finish and G1 is recorded: sync, and confirm by hash that A2's frozen files are unchanged. Confirm or update `frame_sampling.py` on the cluster.
3. CPU jobs with no model: landmarks on 819 clips in `venvs/au_tools`, and the AU step for the human cue check. Run a decoder preflight on `.flv` under cv2 4.11, since `decoder_preflight.py` reads EU-Emotion clips only.
4. Pin geometry and the human target hash. Freeze. Tag.
5. Mechanics run: 2 clips per model, scoring nothing readable, to replace the timings below with measured ones.
6. Stage 1, read, then Stage 2, read, then Stage 3 where a model needs it.

## Cost

Timings come from A2's cost table, per forward: `single_face` Gemma 3.3 s over 31 forwards and Qwen 5.4 s over 46; `video_face` 14.6 s and 17.0 s. One MCQ order takes 7 and 11 forwards (A3), so an A2 cell's two orders leave 17 and 24 for the blind readout. So one MCQ order costs 0.75 and 1.29 s on `single_face` and 3.30 and 4.07 s on `video_face`, and one blind cell costs 1.81 and 2.82 s. CREMA-D's options are shorter, so 7 and 11 are likely high.

| Arm | Cells per model | Gemma GPU-h | Qwen GPU-h |
|---|---|---|---|
| Stage 1, MCQ, 819 clips by 6 orders, real only | 4,914 | 1.02 | 1.76 |
| Mismatch | 0, reused | 0 | 0 |
| Stage 1, blind, 819 originals | 819 | 0.41 | 0.64 |
| `video_face` bridge, 819 by 2 orders | 1,638 | 1.50 | 1.85 |
| Baselines and text primes, 48 cells | 48 | under 0.02 | under 0.02 |
| Stage 3, MCQ, 547 clips by 14 conditions by 2 orders | 15,316 | 3.17 | 5.49 |
| Stage 3, blind, 547 clips by 4 conditions | 2,188 | 1.10 | 1.71 |
| **Total** | | **7.2** | **11.7** |

So about 18.9 GPU-hours of cell time, and **about 21 with A2's 10% allowance**, on both models. **If no model reaches Stage 3, it is about 8.** A3's measured rates (1.22 and 1.66 s per cell, decode included) would lift Stage 1's MCQ to 1.67 and 2.27 hours, which adds about 1.2 hours. Applied to Stage 3 as well, they add about 4.8 hours in all.

**Not in that total.** The AU step for the human cue check has no timing in the repo, and a 2-clip mechanics job sets it. Landmarks are CPU. Qwen fits two shards at the 12 h limit. Cells are JSON lines, with few files, so the cluster's file quota is not a concern. No image is written.

## Limits

- **Apex frame against video.** Humans saw motion, and models primarily see one frame. The bridge route is descriptive, and Qwen's native video merges frames in pairs.
- **Speech.** Every clip is a spoken sentence. The apex is the frame furthest from the first in lip, lid and brow landmarks, so it may be a speech shape. The mouth regions are affected most. CREMA-D's neutral donors also speak.
- **Resolution.** 480 by 360 video upsampled into a 768 crop carries less detail than EU-Emotion. Delivery checks token counts, not detail.
- **Occluder.** A neutral fill adds a neutral feature as well as removing one. The three fills bound this. Nose fill alignment is untested.
- **No AU edit.** Masks remove a region. They cannot add or isolate AU9, so `mid` covers more than one action unit. A generator would be a later amendment.
- **Human cue evidence** comes from other stimuli, mostly synthetic or static, with sources that disagree on upper lip raiser. Rule 4's two-source requirement is met for the confusion and not cleanly for the cue.
- **Raters** always did voice, then face, then audio-visual, so face ratings may carry practice from it. Filtered rating counts relate weakly to ambiguity (`CREMA_D_PLAN.md`).
- **The target pair** was chosen with the votes in view, though only by a stated rule.
- **Sharing is not same-reason on the human side.** Humans are observational here. A match says this model behaves as a stated human account predicts.
- **Text primes** are one wording, with no test, and a model's reading of "A person feels angry." is not a measure of semantics in general.
- **Basic emotions only.** Nothing here reaches complex states, and EU-Emotion's Bored-Unfriendly question stays open.

## Fixed before freezing

1. Audit of this text, then code, then recovery, with outputs under `research/reproduction/a6_recovery.json`.
2. Check the `mid` landmark indices against rendered landmarks on the 24 calibration clips, which are outside the analysis. Confirm that the hull excludes the lip ring and spans the nose wrinkle region.
3. Confirm the **[proposed]** values.
4. Write `a6_human_target.py`, and confirm that it reproduces the numbers above.
5. Pin the geometry and target hashes, and record the resulting `design_sha256`.

## Faults from A2's audit, and how A6 avoids them

| A2 fault | Here |
|---|---|
| Null rule uncoded or unexecutable | Null per stage, from named outputs, reachable on every contrast |
| Verdicts ignored gates | Gates are inputs; missing means not passed |
| Paths that made every verdict Inconclusive | End-to-end test on the real pinned paths |
| A significant contrast called null | A null requires a non-significant test |
| Dose passed on noise | Signed mean of actor means |
| Fill rule skipped | Applied to all four contrasts |
| A loss of the target read as a rise in another | Target and No Emotion left out, with a double difference |
| Area rule passing thin regions | Span rule |
| Margin from another outcome | Margin on the same outcome and clips |
| Shard and hash faults | `N_SHARDS` required, every output carries `design_sha256`, analysis refuses gaps |
| Recovery showed mechanics only | Graded effects, actor noise, adversarial observers |

## Read status of sources

F full text read, P partial, A abstract only, per `research/paper/v2/references.md` and `citation_check.md`.

| Source | Status | Used for |
|---|---|---|
| Cao et al. 2014 | A; README and data read directly | CREMA-D |
| Carbon 2020 | F; Figure 5 absent from the text read | C3, 38% against 2%, descriptive, static, nose and mouth covered |
| Yan et al. 2025, PNAS | F main text; SI not read; not in the 49 references | C1, C2. Confusion attributed to shared AUs, synthetic faces, 10 participants |
| Jack et al. 2014 | F main text; supplement not read | C1, C2. Ideal-observer analysis, not human responses over time |
| Wegrzyn et al. 2017 | F | C4. Anger at the eye (AU7). A reveal study, so a fall under a mask is inferred |
| Ponari et al. 2012 | A | C4. Upper half, not brows |
| Geirhos et al. 2020 | F | Error consistency |
| Bhattacharyya and Wang 2025 | F | Precedent for option-order sensitivity |
