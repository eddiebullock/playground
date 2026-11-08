# DRAFT amendment A6: shared errors on CREMA-D (RQ5)

**Status: DRAFT. Not frozen, not tagged, not preregistered.** Nothing here binds until it is audited, committed and tagged. No model has seen a CREMA-D analysis clip, and no run starts until it is frozen and the wait below has ended.

Draft `design_sha256` of the block below: `c1961048eaaf4ed9ff12d6c78ae46322975cc15dc9820d36e53094396adc04c8`. It changes with any edit to the block and is restated at the freeze.

**Written:** 2026-11-07, before any A1 result was read and before any model output on the 819 clips exists.
**Goal:** G8 in `research/GOALS.md`.
**Depends on:** G1. RQ0 is read through the frozen A1 and A3 table by A2's `rq0_status`, per model. Gates G2 to G4 as in A2.
**Models.** Gemma 4 and Qwen3-VL. Molmo2 fails G2 and G4, and any route back needs a dated rule before new data, as in A2.
**Code, written, not yet committed:** `scripts/a6_design.py`, `scripts/a6_run.py`, `scripts/a6_analyse.py`, `scripts/a6_recovery.py`, `research/scripts/a6_human_target.py`, `slurm_jobs/study3_a6.sh`, `tests/test_a6.py`. All are new files. `a6_design.py` parses the design block from this file and refuses a hash mismatch. Frozen A2 and A3 code is imported unchanged, with hashes pinned below. Every other file the A6 code imports is pinned in `other_code_sha256`, and the A6 code itself in `a6_code_sha256`. Two files are not pinned, by design: `scripts/a6_design.py`, which holds this block's own hash and so cannot be named in it, and `tests/test_a6.py`, which no result reads. Model weights are pinned by a manifest hash in `weights_sha256`, and runs refuse other weights, another dtype, or shards on mixed library versions.

**What was seen before writing this.**

- CREMA-D face-only votes on the selected clips, to state the human target. The pair rule was written with the table below in view, so it is not blind to votes. No model output on the 819 clips exists, so it is blind to that. The measure was corrected in the audit of 2026-11-07, with no model output in view, and the rule was then rerun unchanged: see Pair excess.
- The 24 calibration clips, which are outside the 819 (checked: no overlap), were seen by models for the readout ladder, with no labels or votes read.
- No A1 result. A3's result as stated in `STATE.md`.

## Question

RQ5. When a human and a model make the same mistake, is it for the same reason? The candidates are the same facial cue, shared label semantics, and the options offered.

**Why CREMA-D.** Every clip offers the same six options, so human confusion does not depend on which foils appeared, unlike EU-Emotion. And all 91 actors portray every emotion, so label contrasts are within actor. Neither holds on EU-Emotion. Complex states are out of scope here; Bored-Unfriendly stays an EU-Emotion limit.

**Scope.** Causal claims are about the model. Humans are a fixed target from silent video. This work never states how humans would respond to an edited clip, only whether the model matches a stated prediction. Human cue claims come from other stimuli, and each carries its read status.

## Design block

All fixed constants are here and nowhere else. The hash is sha256 of the bytes between the fence lines, excluding the fences, with a final newline. Items marked PENDING are filled before freezing, from outputs that involve no model output on the 819 clips. Recovery binds to `binding_sha256`, the same hash with the keys in `unbound_keys` blanked, since it reads neither the geometry nor the weights, so pinning those does not call for a rerun.

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
  "other_code_sha256": {
    "scripts/stimulus_design.py": "89d256c1072e0768fc747294553c5710adaf42184a175866d7bd75bcc6104978",
    "scripts/prompts.py": "871272d4aa6c86ff809f7631a4db81aeeb41564ba7f0a31809ea08e6dcf64ba7",
    "scripts/vision_gates.py": "6327b6f88e99409cba5c63d5aca0fe97348fbce4f3b08522158a8d9ba3ce8194",
    "research/scripts/analyse_rotation.py": "98890c5bff65f13abeb0549b61fe414dc9fa30082b84bcb564671683ec12e24f",
    "src/paper_v2/option_score.py": "ae949e4c1c3abc6a775a535c4e0a5042bce59ba14f7ac1afeefd476f032a5a2b",
    "src/paper_v2/metrics.py": "1fc15676cae999e5185cfd016e3dd9db616c71ec0bb4c5d38cd49a4e7b886568",
    "src/paper_v2/schema.py": "4acc626669439384115ae0efcb52cec4edee02c07662a242212d5f7b9c359b47",
    "scripts/__init__.py": "8eb06f2e1f687cde2a9d341830b5beb74ebbfe68881270d0d98f7b2ac2dabb07",
    "scripts/activation_forward.py": "695b1bd7dbb86c2d141621057dd7854a5f48703cb56d70403a4c0b0f5aedec4f",
    "scripts/multi_frame.py": "2cdc5c454f0fe9ecccbd167d7b145d29b7a9d602bbb7a6e5cf61c28cbe4e1b35",
    "scripts/frame_sampling.py": "11b1bb42f8565e66ff0be457bd76e4649410a8991cfe841280eba2444d90becb",
    "scripts/evaluate.py": "cc672f9438fe0b165dc2d620f5fad0e0c3a8845857a91e68871e0dad669a2481",
    "config.py": "75d4a11128aad7d026cbc66331ed9cfd0c51acee1b64a986c2b2138cd1ba77e8",
    "scripts/model_compat.py": "0d88a2d824c966b046d27c6201c5c585c361eb516f30ef51e2b54bfe782efd7c",
    "scripts/model_inference.py": "2ddb946f75db80de14299a77b0de682d1b16ce092185f64d3b3e2d7ae0e7fc7c",
    "scripts/emotion_parse.py": "19a8f875da09bdb4b7ba679eb90ee72adbb37c31c8f2b4e74e2c9e4d445ba93d",
    "scripts/eu_audio_resolver.py": "994947c7ef3a9b08dca7b1e0c007618daaadf37384c833f8defc2255a751d84a",
    "scripts/eu_emotion_definitions.py": "091b0129cea984596cd8b8675bc2b69e7f294810e0b66847ddea4013077bee0d",
    "scripts/eu_emotion_synonyms.py": "9f27ed1c9704ebb1ee59a50ac2f0030d80064e8c1c147f6964862663b76323f8",
    "scripts/forced_choice_entropy.py": "1578a7242f7217aaab6c2bcdaef8ce82df0261c05a249960df3646a6d1f2799f",
    "scripts/free_response_judge.py": "29803031809f90c98d5428a782e26e90edcada8b6f6c8e0dceb90efd460be4bb",
    "scripts/human_entropy.py": "e585bf2317a939aa5a494fdcf21aad7850d41a9be3e0166f2f721ec06206fc9c",
    "scripts/mindreading_audio_resolver.py": "a13abf00d8f5cf094c46a470a9ff15bde2d72cd0640afa6203dad110c69ee81c",
    "scripts/semantic_entropy.py": "058f47fcc7e9532fa1661f866f929de6449024664aacbda953aaa0ab4a0ab62f",
    "scripts/statistics.py": "2d9d361fb9087e206d8896f343707109bb531ae4bdacd0cfab4049630e6ee7b7",
    "scripts/tolerant_parse.py": "d4e31b4f94eb28480edb1f2a9e41ed12a6f83530dd5545881bd56d0fcfa58833",
    "scripts/trial_foils.py": "5e5d23a085ee4465d983e3bf12f7de48d95b0a2fd16287b6529bdf3c093d9ebf",
    "research/scripts/crema_reliability.py": "641136994fd4425c28aec7fe99ba21add0b7247a7b233f58aff9583c7954741f",
    "research/scripts/decoder_preflight.py": "ed5262d89562e2662f6aea3660edb4f1d9cb5306a08ffc68daddd4f669c20699",
    "research/scripts/extract_au_measures.py": "bc846844a3ea37fe39b80e128e0e0753b7680a2976e62180b24b296cabe00ab8",
    "research/reproduction/crema_d_reliability.json": "4ffd40cf62241fe71dc3977c6cbf3d46d80496e250c66127e732ae5ecb81d9d7",
    "research/reproduction/face_input_validity_summary.json": "c1c64e6975b32356a3314268c054f0dee483ddf74b11cd66d03b2faf972e64a4"
  },
  "a6_code_sha256": {
    "scripts/a6_analyse.py": "89f578e5a44a88d9526dea6d6e1f828433e63b6c5eb79e07002d5e2908d883ed",
    "scripts/a6_run.py": "9cb517006f05d2023c35225fa25b93654aee10053ae2f39cac887378b23fb958",
    "scripts/a6_recovery.py": "c77d42ca3c511742c63b0f26f2da733038905bd4448927c867308fe99d53d189",
    "research/scripts/a6_human_target.py": "f7424fa15484328c1c71c5534ff8724bd0a92c8bf0b366c823065f84ded4283d",
    "slurm_jobs/study3_a6.sh": "1cf329f486c058b72de68c0e0f10fa59a6a8a90c789b8fa1e8385573a584c1b4"
  },
  "geometry_sha256": "PENDING",
  "weights_sha256": {
    "gemma4": "PENDING",
    "qwen3vl": "PENDING"
  },
  "unbound_keys": ["geometry_sha256", "weights_sha256"],
  "runtime": {
    "dtype": "bfloat16",
    "cv2_prefix": "4.11.",
    "mediapipe_version": "0.10.14",
    "short_decode_clips": ["1039_WSI_SAD_XX", "1061_TIE_ANG_XX", "1072_WSI_ANG_XX"]
  },
  "human_target_sha256": "916d7a5633d23f9469e8cb414ee8408fb5ff54cb7c8bdeac8051fec11def8169",
  "models": ["gemma4", "qwen3vl"],
  "primary_route": "single_face",
  "bridge_route": "video_face",
  "n_items": 819,
  "n_actors": 91,
  "seed": 42,
  "options": ["Anger", "Disgust", "Fear", "Happy", "No Emotion", "Sad"],
  "family_map": {
    "Anger": "Angry",
    "Disgust": "Disgusted",
    "Fear": "Afraid",
    "Happy": "Happy",
    "No Emotion": "Neutral",
    "Sad": "Sad"
  },
  "prime_words": {
    "Anger": "angry",
    "Disgust": "disgusted",
    "Fear": "afraid",
    "Happy": "happy",
    "No Emotion": "neutral",
    "Sad": "sad"
  },
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
  "shared_family": [
    ["Anger", "Disgust"],
    ["Fear", "Sad"],
    ["Anger", "Happy"],
    ["No Emotion", "Sad"]
  ],
  "stage3_emotions": ["Anger", "Disgust", "Fear", "Sad"],
  "control_emotions": ["Fear", "Sad"],
  "regions": {
    "upper": ["A2 eyes rings", "A2 brows rings"],
    "mid": {
      "hull": [133, 362, 129, 358, 2]
    },
    "lower": ["A2 mouth ring"],
    "midlower": ["mid", "lower"],
    "uplow": ["upper", "lower"],
    "all": ["upper", "mid", "lower"]
  },
  "region_dilate_fraction": 0.1,
  "sham_region": "all",
  "outcome_reference": "fixed",
  "reference_check": "partner_beyond_zero_for_named_results_happy_inside_margin_for_nulls",
  "prime_separator": " ",
  "conditions": ["original", "sham_self_fill", "mask_upper_neutral", "mask_upper_first", "mask_upper_blur", "mask_mid_neutral", "mask_mid_first", "mask_mid_blur", "mask_midlower_neutral", "mask_midlower_first", "mask_midlower_blur", "mask_lower_neutral", "mask_uplow_neutral", "mask_all_neutral"],
  "alpha": 0.05,
  "ci_decision": 0.95,
  "ci_equivalence": 0.9,
  "signflip_draws": 100000,
  "bootstrap_draws": 10000,
  "pair_measure": "standardised_share",
  "shared_margin": 0.05,
  "cue_directions": "excess_only",
  "direction_rows": "edit_order_rows",
  "null_needs_edit_row_excess": true,
  "cue_margin_fraction": 0.1,
  "cue_margin": "signed_cells_by_partner_emotion",
  "dose_margin": "cue_margin_fraction_of_signed_acted_change_under_mismatch",
  "lost_item_limit": 0.05,
  "text_neighbour_rank": 5,
  "min_span_tokens": 1.0,
  "token_floor": 8.0,
  "first_frame_au_ratio_max": 0.5,
  "human_split_halves": 200,
  "recovery": {
    "replicates": 500,
    "expected_verdict_min": 0.8,
    "other_named_verdict_max": 0.1,
    "inconclusive_max": 0.2,
    "seed": 6042,
    "strengths": [0.5, 1.0, 2.0],
    "judged_strengths": [1.0, 2.0],
    "signflip_draws": 4999,
    "bootstrap_draws": 1000,
    "evidence": 2.0,
    "prior_sd": 1.0,
    "actor_sd": 0.3,
    "clip_sd": 0.5,
    "condition_sd": 0.2,
    "order_sd": 0.2,
    "evidence_share": {
      "upper": 0.3,
      "mid": 0.2,
      "lower": 0.3,
      "rest": 0.2
    },
    "mask_area": {
      "upper": 0.35,
      "mid": 0.15,
      "lower": 0.2,
      "midlower": 0.35,
      "uplow": 0.55,
      "all": 0.7
    },
    "generic_nats": 1.0,
    "noise_scales": [1.5, 2.0],
    "actor_region_sd": 0.3,
    "actor_region_ladder": [0.0, 0.15, 0.6, 1.0],
    "ladder_replicates": 200,
    "score_scale_log_sd": [0.15, 0.3, 0.5, 1.0],
    "first_fill_kept": [0.5, 0.8, 1.0],
    "partial_share": [0.5, 0.25],
    "calibration_draws": 1000,
    "tokens_across": 16.0,
    "span_fraction": 0.2,
    "too_small_fraction": 0.01,
    "lost_share": 0.08,
    "artefact_nats": 0.6,
    "happy_nats": 0.3,
    "pull_multiplier": 3.0,
    "option_prior_nats": 1.5,
    "position_effect": [1.5, 0.75, 0.0, 0.0, -0.75, -1.5],
    "split_actor_share": [0.35, 0.65],
    "split_gain": 3.0,
    "spread_prime_nats": 4.0,
    "boundary_cue_nats": 0.15,
    "adjacency_cap": 0.7,
    "carbon_lift": 2.0,
    "scenario_replicates": 50,
    "noise_replicates": 100,
    "draw_check_replicates": 15
  }
}
```

`shared_margin`, `cue_margin_fraction`, `text_neighbour_rank`, `outcome_reference`, `reference_check`, `first_frame_au_ratio_max` and the mid-face landmark list are **[proposed]**. The mid-face indices are drafted from the FaceMesh layout and need a check on calibration clips (see Fixed before freezing).

## Human target, fixed before model data

**Data.** Face-only votes in `research/reproduction/crema_d_selection.csv` (sha256 in the design block), 819 clips, 91 actors, about 10 raters per clip. Chance is 1/6, and a target is the actor's intended emotion. Per `CREMA_D_PLAN.md`, the votes come from `queryType` 2, checked against the raw responses. `research/scripts/a6_human_target.py` reproduces every table below, writing `research/reproduction/a6_human_target.json`, and that file's sha256 is `human_target_sha256`. So it carries no `design_sha256`, unlike every other output.

**Mean vote share**, row is the intended emotion:

| Intended | Anger | Disgust | Fear | Happy | No Emotion | Sad |
|---|---|---|---|---|---|---|
| Anger | .585 | .206 | .039 | .021 | .123 | .027 |
| Disgust | .080 | .540 | .090 | .015 | .058 | .217 |
| Fear | .080 | .100 | .467 | .013 | .196 | .143 |
| Happy | .003 | .014 | .013 | .874 | .085 | .010 |
| Neutral | .041 | .076 | .044 | .031 | .727 | .080 |
| Sad | .033 | .095 | .143 | .020 | .391 | .318 |

**Pair excess.** A confusion must not be a bias towards one option, generic difficulty, a by-product of accuracy, or of how sharp a system is on one emotion. So for system S, a confusion of A with B is measured against each other emotion C, on the four options left once A and C are set aside, on clips of A and on clips of C, within actor:

    z_i(k)  = log p_i(k), centred and divided by its SD over the four options k not A or C
    r_i(B without A, C) = exp z_i(B) / sum over the four of exp z_i(k)

    X_a(A to B) = mean over C, not A or B, of
                  [ mean r_i over actor a's clips of A
                    minus mean r_i over actor a's clips of C ]

    X(A, B) = mean over actors of [ X_a(A to B) + X_a(B to A) ] / 2

r_i is B's **standardised share** of the four: a quarter when the four are equal, and unchanged when a clip's scores are shifted by option priors or multiplied by a constant. For humans, p_i are vote shares with `pseudo_count_human` added per option, which the logarithm needs. For models, p_i are exact option probabilities, and r_i is taken on each order and then averaged over the orders in question, so that a position effect stays inside each order. Human and model X are on the same scale and computed by the same code, but not on the same footing: human shares come from about ten raters with a pseudo-count, and model shares from one exact distribution per order. So X is comparable in rank and sign, and its size is compared only with that caveat.

**Why this form.** The first draft set r_i = p_i(B) / (1 - p_i(A)) on every clip. On clips of A that removes the acted label. On other clips it leaves their acted label in the denominator, so B's share is smaller there for a reason that has nothing to do with B. An observer spreading its errors evenly at 60% accuracy then gets X = .113 for every pair, above `shared_margin`, so Not shared could not be reached by any model that uses the face, and 14 of the 15 human X values in the first table were above zero. Removing each clip's own label instead fixes that, but leaves a bias from option priors: with random log priors of SD 1 and 2 nats of acted evidence, 8% of 300 noise-free observers with no pair excess had X above .05 (`prior_bias_check` in `a6_recovery.json`; an earlier scratch check gave about 12% and does not trace to code). Taking the plain share on the same four options removes the prior bias exactly, which run 1 and run 2 used. The second audit then showed that it still fails when a model is sharper on some emotions than others: with a scale per acted emotion of log SD .15, .3 and .5, No sharing left Not shared in .23, .33 and .38 of replicates. Standardising the four scores within the clip removes that too, exactly, for scores built from priors, evidence and a scale per emotion (`tests/test_a6.py`). What it cannot remove is a scale that acts on the signal and not on the noise, or any change of shape that is not a shift or a scale. Those limits are in Recovery.

**Pair rule.** No model output enters.

1. Rank the 15 pairs by human X.
2. The **target pair** is the highest-ranked pair of two emotions. No Emotion is excluded from the target, because any occluder removes signal, and a rise in No Emotion cannot separate lost signal from a confused cue (A2, P7, and Carbon 2020).
3. The **shared family** is the top three pairs of two emotions, plus the top pair that includes No Emotion.

**Result, human X with actor-bootstrap 95% CI, 10,000 draws:**

| Pair | Human X | 95% CI | Role |
|---|---|---|---|
| **Anger, Disgust** | **.166** | **.132 to .199** | **Target** |
| No Emotion, Sad | .112 | .079 to .144 | Family |
| Happy, No Emotion | .104 | .074 to .136 | Not selected |
| Fear, Sad | .060 | .025 to .096 | Family |
| Anger, Happy | .014 | -.005 to .033 | Family |
| Disgust, Sad | .006 | -.026 to .039 | Not selected |

Directed: Anger read as Disgust .255, Disgust read as Anger .077, with 75 and 45 of 91 actors above zero. So the human confusion runs mostly one way.

The target pair has not changed under any of the three measures. The family changed once, under the second measure: Anger-Happy replaced Disgust-Sad, because Disgust-Sad is carried by Disgust read as Sad (.132) against Sad read as Disgust (-.120). The standardised share keeps the second family. Anger-Happy's interval now includes zero. The family says how general sharing is, and sets no verdict.

**Stability.** Anger and Disgust stay the top emotion pair at a pseudo-count of .1 (.158) and 1.0 (.169), and on all 7,442 clips (.156). With no pseudo-count the logarithm is undefined, so the check is at a tenth and twice the pinned one. The family's lower places are not stable: on all clips the third and fourth places become Disgust-Sad and Happy-No Emotion. Only the target pair sets the verdict.

**Clips that read as the other label.** RQ5's design constraint asks whether a confusion is a real ambiguity or clips that simply read as the other label. `a6_human_target.py` reports the share of Anger clips with a Disgust mode (.124 strictly, .153 with ties), and the concentration of Anger-to-Disgust votes across clips (76% of Anger clips have a Disgust vote; the top tenth of clips hold 30% of those votes). So the confusion is spread over many clips, and Disgust-to-Anger less so (35%, and 57%). Every result is repeated on clips whose human mode is the intended emotion, a tie for top counting as the mode (596 clips, 43 of them ties).

**Human cue check on the same clips.** Under rule 3 of `RESEARCH_QUESTIONS.md`, within Anger and within Disgust clips, the human partner share is regressed on detector estimates of AU9 and AU10 at the apex, with actor fixed effects, for each pairing of the two detectors of each unit (`a6_human_target.py --cue_frames`, not the same-clip code, whose loaders read EU-Emotion). The frames come from `extract_au_measures.py`, unchanged, on a manifest written by `a6_run.py`. It involves no model and runs before any model. It is reported and gates nothing. If it fails to show the link, the mid-face prediction rests on the literature alone and is labelled so. Speech movement may weaken it.

## Stimuli and input

- **Clips.** 819 selected clips, `<clip_id>.flv`, 480 by 360, 29.97 fps, 1.40 to 4.87 s, on CSD3. No audio is used. Raters saw them silently. Frames are decoded and edited in memory, and no frame or edit is written, kept or published.
- **Routes.** `single_face` is primary, because A3 tested it and a lip corner is sub-token on a grid. `video_face` is a bridge, descriptive only, because humans saw motion. It runs on the original clips at the two `edit_order_rows`, and is compared with `single_face` on the same two orders.
- **Frames and geometry.** A2's rule: 8 uniform frames, landmark apex, one fixed square crop at 1.5 times the largest landmark extent, 768 by 768, MediaPipe FaceMesh 0.10.14, which the landmark step asserts. A new geometry file is pinned in the design block. A2's EU geometry check is not used for it. **Geometry passes** if at most `lost_item_limit` of clips lack a face box or landmarks at the apex; the landmark step stops and pins nothing above that, and the analysis checks it again with the file's hash.
- **Frame count.** `count_frames` counts decodable frames, so the three h264 clips that overstate their header stay usable and lose no index. The plan's expected loss of one frame applied to seeking, which this route does not do.
- **Neutral donor.** Every actor has one or two Neutral clips (46 and 45 actors), so A2's neutral fill applies. The donor is the middle sent frame of the first by clip id. Where a Neutral clip would donate to itself it is not edited, since Neutral clips are not in Stage 3.

## Task and scoring

- **MCQ.** `build_4afc_prompt(options, condition="video_only")`, as in A1 and A3, with prefill "EMOTION:". Each option's joint logprob is renormalised over the six options, as in A3. An item is correct if the target is strictly highest; ties are wrong.
- **Blind readout.** A2's prompt, prefill and lexicon, all scored as in A2. Six families are read through `family_map`, and renormalised over those six. It adds a route with no option list, so it can show whether a confusion needs the options at all.
- **Option wording.** "No Emotion", as raters saw, not "Neutral".

**Option rotation.** Option order is known to move VLM emotion judgements (Bhattacharyya and Wang 2025, not faces). All clips run all six `orders` in Stage 1. These replace the cyclic rotation in the plan and adapter, which is not used here and is not edited. Cyclic rotation keeps neighbours adjacent: Anger and Disgust are next to each other in 5 of 6 cyclic rotations, so the target pair would be confounded with adjacency. The `orders` are a Williams square: each option takes each position once, and each ordered pair of options is adjacent exactly once. So every pair is adjacent in 2 of 6 orders. Cost is the same as six rotations.

The on-screen order raters saw is unknown (`CREMA_D_PLAN.md`), so human position effects stay unknown.

**Real against mismatch.** Partners come from `vision_mismatch_partners`, seed 42, on these clips. Checked: all 819 partners differ in emotion and actor, and each face is used once. **The prompt depends only on the order, not the clip, so a clip's mismatch cell is its partner's real cell on the same order.** Scoring is deterministic, so no extra forward is needed.

## Stages

Each stage is read only after the one before it has a result. Stage 3 runs only for a model that needs it.

**Stage 0: gates and the wait.** See the next section.

**Stage 1: validity on CREMA-D.** RQ0's design on `single_face`. For clip i, a_i is mean correct over six orders, and b_i the same on its partner's face. delta = mean(a_i - b_i). A model is vision-dependent on CREMA-D if the one-sided actor-level sign-flip p (`actor_sign_flip_p`, 100,000 draws) has Holm p < .05 across the eligible models, and the actor-bootstrap 95% CI lies above 0, the same rule as A3.

Also here: `image_free` and `grey` for each order, the same two for the blind readout (for B3), the blind readout on 819 originals, the `video_face` bridge, the text primes, and the delivery rows. Secondary: accuracy against 1/6, accuracy by target position, within-emotion correlation with human p_target against the ceilings in `crema_d_reliability.json`, and Gate B's B1 to B5 on CREMA-D's six families, by the frozen rules, with lexicon mass taken over those six. The B checks are reported and do not gate.

**Stage 2: is the error shared, and is it the options?**

- **Shared.** Model X on the four family pairs, one-sided above zero, step-down maxT, one model at a time, with an actor-bootstrap 95% CI. Only the target pair sets the verdict; the others say how general sharing is.
- **Pattern, descriptive.** Rank correlation of human and model X over the 30 directed cells, against a split-half ceiling from raters (half r .938, 200 splits, in `a6_human_target.json`). Error consistency in the sense of Geirhos et al., on the modal correct pattern over shared clips.
- **Options.** O = X on the 2 orders where the pair is adjacent, minus X on the other 4, one-sided above zero, by its own step-down maxT over the four pairs. X_far is X on those other 4 orders, tested the same way. Blind X is X on the blind readout's six families, tested the same way.
- **Semantics.** Text-only primes: for each `prime_words` word, the prompt is `prime_prompt_prefix`, then `prime_separator`, then the MCQ text, with no image, on all six orders. X_text is X with primes in place of clips, one prime per emotion and one actor. This is one deterministic value per directed cell, so it has a rank and no test.
- **Directions.** Each direction of the target pair gets a state here, from Stage 1 cells alone, written to the output before Stage 3 starts. The two directed X values are tested one-sided above zero by their own step-down maxT. **Carries:** significant and above zero. **Absent:** not that, and the upper limit of its 90% CI below `shared_margin`. **Unresolved:** neither. States are taken twice: on all six orders, to route Stage 2, and on the two `edit_order_rows`, to decide Stage 3, since those are the orders the edits are scored on (`direction_rows`). C1 belongs to Anger read as Disgust, and C2 to Disgust read as Anger. See Which directions decide.
- **Focus.** If exactly one direction carries on all six orders, O, X_far and blind X are taken on that direction alone, and otherwise on the pair, so that a negative direction cannot hide an option effect in a positive one.
- **Margins locked.** The cue margin and the dose margin for Stage 3 are computed here and written to the output before Stage 3 starts, with a report of whether a null could be reached (below).

Where Stage 2 leaves a model, on the target pair:

| X, and the direction states on all six orders | O | X_far | Blind X | Goes to |
|---|---|---|---|---|
| X not significant, upper 90% limit below `shared_margin`, and both directions absent | | | | Not shared |
| X not significant, no direction carries, and anything else | | | | Inconclusive, `shared_unresolved` |
| X passes, or a direction carries | Not significant | | | Stage 3 |
| As above | Significant | Significant | | Stage 3. The shared error survives without adjacency, and the edit orders are non-adjacent |
| As above | Significant | Not significant, upper limit below the margin | Not significant | Option-dependent |
| As above | Significant | Any other case | | Inconclusive, `option_unresolved` |

X passes if its step-down p is under .05 and its mean is above zero. X averages the two directions, so a negative direction can cancel a positive one: in the second audit, a human-rate forward direction with an equal negative reverse read as Not shared in .65 of replicates while the forward direction carried in 60 of 60. So a carrying direction is a shared error, and Not shared needs both directions absent. A null here needs only upper limits, since a negative X is no shared error.

**Can a null be reached?** Stage 2 also reports, for each contrast, the actor SD of the double difference of y under full mismatch, the 90% half-width it implies, and the actor SD above which the half-width exceeds the cue margin with 91 actors (`null_reach`, `sd_limit = margin * sqrt(91) / 1.645`). Full mismatch is the strongest edit there is, so this is a rough guide and gates nothing. A null at Stage 3 is equivalence inside a margin of about .05 to .09 on y, and **it may be out of reach on a real model**: recovery shows even with no actor-by-region variation the two-way nulls are reached in only .62 (Not this cue) and .70 (No tested feature) of replicates, and the one-way null in .87, falling to .29 at an SD of .3 (see Recovery result). `null_reach` is optimistic: its proxy gave an SD limit near .55, while the nulls failed at .3.

**Stage 3: cue-targeted test.** Only for a model with a shared target pair not attributed to options. Clips of the four `stage3_emotions`, 547 in all, on `single_face`, with every `conditions` entry at both `edit_order_rows`. There is no blind readout at Stage 3: no rule read it, and it would have cost 2.8 GPU-hours. Those two orders follow a rule: the Williams rows in which the target pair is not adjacent and neither member sits at an end. Within-clip deltas hold order fixed.

## Edits

Edits come from A2's frozen code: `read_frames`, landmarks, `align_to`, `_fill_image`, `blend`, `crop`, `fixed_size`, the fill types, the feather, and `clr`. `upper` and `lower` are unions of A2's rings, each dilated by `region_dilate_fraction` (.10), where A2 used .12 for the mouth and .08 for brows, so `lower` is not pixel-equal to A2's mouth. `mid` is new, because A2 has no nose region and its disclosed limit on P8 is that a mouth mask may leave a nose wrinkle visible. `a6_run.py` holds a mask function matching `region_mask` line for line, because `REGIONS` is a frozen constant and cannot take a new key. A test holds its output equal to `si.region_mask` on `mouth`.

| Condition | Definition |
|---|---|
| `original` | Unedited apex frame |
| `sham_self_fill` | Full pipeline on `sham_region` (`all`, the largest mask), filled from the frame itself. Must equal `original` exactly |
| `mask_upper_{neutral,first,blur}` | Eyes and brows |
| `mask_mid_{neutral,first,blur}` | Nose bridge to below the nose: AU9 wrinkle and the region above the lip |
| `mask_midlower_{neutral,first,blur}` | Mid and mouth, the analogue of a nose and mouth covering |
| `mask_lower_neutral` | A2's mouth ring, descriptive bridge to P8, reported as its double difference on C3's outcome |
| `mask_uplow_neutral` | Upper and mouth, so mid-face stays visible |
| `mask_all_neutral` | Upper, mid and mouth together |

**Outcome.** For a contrast of emotion A with partner B: y = the CLR of B, with A, No Emotion and both control emotions left out of the reference, on every clip in the contrast. That leaves B and Happy, so y = (log p(B) - log p(Happy)) / 2 on all of them (`outcome_reference: "fixed"`).

- Leaving out A follows A2's fix, so a plain fall in A cannot read as a rise in B.
- Leaving out No Emotion removes the generic route by which an occluder raises No Emotion.
- Leaving out the control emotions is new, from the audit. The first draft left only A and No Emotion out. On a control clip that keeps the clip's own label in the reference, so any mask that weakens the acted label raises y on control clips and not on clips of A, which reads as the predicted fall in C1 and C2 with no pair effect at all. Leaving each clip's own label out instead puts A in the reference on control clips, so a generic pull towards A reads as an opposed result. A single reference with no clip's own label in it has neither fault, and both are tests in `tests/test_a6.py`.
- Happy is noisier than a four-label reference, and it is one label. The two earlier references are coded as sensitivities.
- **Happy alone can move y.** In the second audit, Happy rising by .3 nats under `mask_mid` on clips of the pair gave Same cue in 1.00 of Global face replicates and .98 of Other cue, and Happy falling gave Opposes in 1.00. So `reference_check` adds two checks, each a double difference against a reference with neither the partner, nor Happy, nor No Emotion, nor the clip's own label in it, in y's units. **A match or an opposed result** needs the partner's own change to lie wholly beyond zero in that direction (95% CI). **A null** needs Happy's own change inside the cue margin (90% CI). Otherwise the contrast is `cue_unresolved`.
- The audit asked for Happy inside the margin for named results too. That was coded and tried first: at the noise recovery is judged at, Same cue then reached its own row in 0 of 12 replicates, because an equivalence test on Happy is as noisy as the effect itself. A named result says the partner moved, and the partner check tests exactly that, with Happy out of its reference. `--variant happy_equivalence` reruns the audit's version on the same seeds at 100 replicates (`a6_recovery_happy_equivalence.json`): Same cue reached its own row in .09 and either Same cue row in .64, against .99 under the partner check, and Happy up reached a false Same cue, one direction in .08, against .04 under the partner check.

**Contrast.** delta(i, c) = y(i, c) - y(i, `original`), on the same order, averaged over the two orders. The confirmatory effect is a within-actor double difference, so that it moves this pair and not others:

    e(a) = mean delta over actor a's clips of A,
           minus mean delta over actor a's clips of the control emotions,
           both on the same outcome word B, with the same options left out

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
- **Fill rule.** The neutral fill is primary. A match counts only if the 95% actor-bootstrap CIs of the first and blur fills each lie wholly on the predicted side of zero, and the simple change on A clips has the predicted sign. An opposed result needs the same in the opposite direction. The first draft asked only for the same sign, which two fills with no effect share one time in four, and in recovery an artefact of the neutral fill alone reached Same cue or Same cue, one direction in about half of replicates.
- **Cue margin.** 10% of the model's change in the same outcome on the same clips under full mismatch, fixed at Stage 2, one margin per contrast. The change is y on the partner's real cell minus y on the clip's own, at the `edit_order_rows`, over the contrast's clips of A and of the control emotions. It is signed within cells: clips are grouped by their own emotion and their partner's, the signed change is averaged in each group, and the margin is the clip-weighted mean of those group means' absolute values. The first draft took each clip's absolute change, which grows with noise, so a noisier model had a wider margin and an easier null; with pure noise of SD 2 against .5 that margin grew over 3.5 times (`tests/test_a6.py`). This avoids A2's disclosed fault of a margin taken from another outcome.
- **Dose margin.** The dose is a change in the acted label's CLR over six options, so its margin is 10% of that same quantity's change under full mismatch, signed within cells, on the contrast's clips of A. The first draft compared the dose with the cue margin, which is on another outcome.
- **Shared margin [proposed].** .05 on X, about three tenths of the human value under the standardised share (.166). At the human actor SD for this pair (.17) the 90% half-width is .03, and a null stays attainable up to an actor SD of about .29.
- **Models.** Never pooled.

## Decision table

Per model, on the target pair. Rows are read in order, and the first that applies is the verdict.

| Verdict | Requires |
|---|---|
| **Not tested** | RQ0 on `single_face`, G2 to G4, delivery, or geometry not passed for this model, or over `lost_item_limit` of clips lost at Stage 1 or Stage 3. Stops |
| **No face use on CREMA-D** | Gates passed, Stage 1 failed. Named, and stops: no shared-error claim |
| **Not shared** | Stage 1 passed; X not significant, the upper limit of its 90% CI below `shared_margin`, and both directions absent. The null at Stage 2. Stops |
| **Option-dependent** | X passes; O significant at step-down .05, X_far not significant with the upper limit of its 90% CI below `shared_margin`, and blind X not significant. Stops |
| **Same cue** | A shared error, not option-dependent; sham equal; two directions in play, C1 and C2 both matching, with the fill rule and the partner check. It says the mid-face carries part of the shared error, and the share of the excess the mask removes is reported beside it |
| **Same cue, one direction** | As above, with exactly one direction in play matching, and any other in play not opposing |
| **Opposes the stated cue** | As above, with C1 or C2 significant in the opposite direction, with the first and blur fills and the simple change on A clips all in that direction |
| **Not this cue** | The cue null in the next section met on every direction in play. The null at Stage 3 |
| **No tested feature** | C1 and C2 not significant in either direction, delivery and the dose pass, every direction in play carries on the edit orders, Happy is inside the margin, and `mask_all_neutral` leaves the pair inside the margin, by the 90% CI of the double difference. Then the shared error does not need these regions |
| **Inconclusive** | A reason code from the list below |

**Which directions decide (`cue_directions: "excess_only"`).** Stage 3 asks whether a cue carries a shared error. A direction with no excess has no shared error to carry, so it cannot test the cue. In the rows from Same cue to No tested feature, "C1 and C2" means the directions whose state is carries or unresolved. An absent direction does not enter them: it neither blocks a null, nor counts for a null, a match or an opposed result, and its results are reported beside the verdict.

- Same cue needs two directions in play, both matching. With one direction in play, a match is Same cue, one direction.
- An unresolved direction stays in play, so doubt about a direction never lightens a null. Only a direction shown to lack an excess at Stage 2 leaves. **A null also needs every direction in play to carry on the edit orders**, so that there is an excess for the mask to remove on the orders it is scored on.
- The rule is fixed at Stage 2, before any edited cell exists, so it cannot be chosen from Stage 3's result.
- **One guard.** If an absent direction's contrast is significant in either direction, the verdict is `cue_unresolved` unless a direction in play has already given a named cue verdict. A significant contrast contradicts the reading that the direction has no excess, so no null is read past it. This is the one way an absent direction can stop a null, and it is deliberate: without it a weak real cue misread as absent could give a false Not this cue.
- Plain "condition 4 on at least one of C1 and C2" was rejected, because it would let a direction with no excess supply the movement, or a null, for a direction that has one. With this rule, a null in the absent direction and no null in the direction with an excess is `cue_unresolved`. `tests/test_a6.py` holds that case.
- If no direction is in play, the verdict is `cue_unresolved`.

**Same cue and the share carried.** Same cue needs significance, the fill rule and the partner check, and no size floor. With half the excess on the mid-face Same cue was reached in .90 of replicates in the audit, at a quarter .32, and at an eighth .05 (`partial_share` in `a6_recovery.json` for the current rule). A floor would be a new constant with no human basis, so instead the share of the excess in y that the mask removes, with its parts, is reported with every Same cue row, and the verdict is read as "the mid-face carries part of the shared error in this model", never as all of it.

**Rider: label semantics.** With Not this cue or No tested feature, the text-prime rank of the directed target cells is reported. Rank 1 is the highest X_text. If both directed target cells rank within the top `text_neighbour_rank` of 30, it reads as "label neighbours are a live account". Both are required, since either of two cells would land in the top five about a third of the time by chance. This is a correlational reading, and A6 does not show that semantics caused the error.

**Rider: redundancy.** `mask_uplow_neutral` keeps only the mid-face. If the 90% CI of its double difference is inside the margin on both C1 and C2, on a null, the label `sufficient_not_necessary` is attached, and the verdict does not change.

Every verdict states its scope: a claim about this model, these 91 actors, this clip set, single apex frames. The bridge route and blind readout are reported beside it, and any disagreement is stated.

## Null and Inconclusive

**Null at Stage 2 (Not shared).** Stage 1 passed; X not significant; the upper limit of its 90% CI below `shared_margin`.

**Null at Stage 3 (Not this cue).** All of these, on every direction in play, which is at least one (see Which directions decide). No contrast is capped, and a one-way confusion can reach it on the direction with the excess. Reachable is not likely: see Recovery result, where this null was rarely reached at the judged noise.

1. Gates, Stage 1 and Stage 2 passed, and the sham equals the original exactly.
2. Delivery passed. The median over the contrast's clips of the edited region's narrowest side is at least `min_span_tokens` token wide. This uses span, not area, since A2 disclosed that an area rule can pass a long thin region.
3. The dose passed. The signed change on the acted label under the blur fill of the same region, as a CLR over all six options on clips of A, as the mean over actors of actor means, has an absolute value above the dose margin. It is signed so that noise cancels, the correction from A2's second audit.
4. The pair can move. `mask_all_neutral` changes the double difference by more than the margin in absolute value, as a mean over actors. Without this, the row is No tested feature, or `cue_unresolved` if neither applies.
5. Equivalence: the 90% CI of the double difference is inside the cue margin, and the test is not significant in either direction.
6. The mean and 95% CI in the predicted direction are reported.
7. Happy's own change is inside the cue margin (90% CI), so that Happy moving cannot hide a fall in the partner.
8. The direction carries an excess on the edit orders.

A2 asks that a joint ablation is also null. That cannot apply here: a shared error is face-dependent, so masking the whole face should move it. So condition 4 asks the reverse, and redundancy is reported by the rider. This difference is deliberate, and it is why the null here means "not needed alone".

**Inconclusive.** Exactly these reason codes, in the order the code reads them: `shared_unresolved` (no shared error shown and Not shared not met), `option_unresolved` (O significant, X_far not significant, and the rest of the Option-dependent row unmet), then at Stage 3 `sham_failed`, `cue_unresolved` if no direction is in play, then the named cue rows, then `region_too_small`, `dose_failed` (read only if no direction in play is significant), and last `cue_unresolved` (a significant contrast in an absent direction, a direction in play that does not carry on the edit orders, or no row that applies). Not tested for Stage 3 losses is read before `sham_failed`.

**Inconclusive is not the default by construction at Stages 1 and 2. At Stage 3 it is the likely result for a null model (see 1).**

1. Every row is reachable for an eligible model, on 91 fully crossed actors, and recovery reached each one. **This section's title holds for Stages 1 and 2, not for Stage 3**: at the noise recovery is judged at, a model with a shared error on another cue was mostly `cue_unresolved`. That is the failure recovery reports, and it is not repaired here by relaxing a rule.
2. The gate loader must return True on a fixture built from the real pinned paths, with a test that runs the verdict function end to end, so that no missing path makes every verdict Inconclusive.
3. A failed outside gate or Stage 1 gives Not tested or No face use, each a named result. Only checks inside Stages 2 and 3 can give Inconclusive.
4. Recovery must reach every named row and every reason code, and hold Inconclusive under `inconclusive_max` for each observer with an expected verdict. Observers judged on a forbidden verdict, and boundary observers, have no Inconclusive limit. A failure is reported, not hidden: any change of margin or rule comes before freezing and is dated.
5. A row with no verdict or reason code is a code fault and stops the analysis.

## Missing data

A face that cannot be localised fails its cells, and the reason is logged. A clip enters Stage 1 only with all six orders. Its partner must also be present. At Stage 3 a clip counts as lost if any condition or order failed, and it leaves only the contrasts that need that cell. Failures are reported by emotion and actor. Over `lost_item_limit` of clips lost for a model, at either stage, gives Not tested. Lost clips are not replaced. A bad final JSON line is dropped and rescored, by A3's `read_cells`, since scoring is deterministic.

## Sensitivities

All are coded and reported beside the primary, and none changes a verdict.

- Clips whose human mode is the intended emotion.
- Outcome with No Emotion kept in the reference, with the first draft's reference, and with each clip's own label left out.
- `mask_lower_neutral` on C3's outcome, as a bridge to A2's P8.
- MCQ with no image and with a grey image: mean probability by option and by position, as the option prior.
- Plain change on A clips, without the double difference.
- Control emotions one at a time.
- Each edit order separately.
- All other-emotion, other-actor faces in place of the single partner, on Stage 1, which is free since cells exist.
- Excluding `align_flagged` cells, and clips under `token_floor` tokens across the face.
- The three h264 clips with short decodes excluded.
- Actor sex, descriptive. Equal weighting of clips.
- The directed cells separately, since human confusion runs mostly from Anger to Disgust.

## Recovery

Before any model run, synthetic observers go through `a6_analyse.py` on the real 819 clips, 91 actors, partners and orders, by `a6_recovery.py`, as rows in `a6_run.py`'s format. The observers have option scores built from named components, and the same masks act on those components. Effects are graded at 0.5x, 1x and 2x of the human directed rates in `a6_human_target.json`, since A2 disclosed that its recovery showed mechanics, not power. Each replicate draws its own log option priors, and a boost in nats is solved per replicate so that X equals the rate on that prior, at the observers' noise.

**Noise.** A2's recovery noise (actor .3, clip .5, condition .2, with .2 for order on the same footing), not chosen for A6, plus one term the second audit showed was missing: **each actor responds to each masked region in its own way, per emotion and option, the same for all fills of that region** (`actor_region_sd`, .3, A2's actor SD on the same footing). Without it, actor and clip noise cancel in the within-clip delta, every actor has the same mask effect, and recovery overstated the power of the nulls: in the audit, Not this cue on Other cue fell from .95 to .32 and 0 as that SD went from 0 to .3 and .6, and No tested feature on Global face from .95 to .25 and 0. Nothing on CREMA-D sets this SD, so recovery is judged at .3 and reported at 0, .15, .6 and 1.0. Evidence for the acted label sits in the regions in `evidence_share`, so a mask always weakens it. 1.5x and 2x of the other noise are also reported, not judged. Recovery uses `signflip_draws` 4,999 and `bootstrap_draws` 1,000 from its own block, for time, checked verdict by verdict against the real draws on a small set of replicates, and `p` is doubled as Holm's worst case over two models at Stage 1, as in A4's recovery.

| Observer | Construction | Expected verdict |
|---|---|---|
| Prior-only | No evidence for the acted label on any clip | No face use on CREMA-D |
| No sharing | Face use, errors spread by the prior alone | Not shared |
| Adjacency | Pair excess only when the pair is adjacent | Option-dependent |
| Same cue | Pair excess carried by a mid-face component | Same cue, and must not reach Not this cue |
| One-way cue | The same, on Anger clips only | Same cue, one direction, and must not reach Not this cue |
| Reverse | Pair excess from outside the regions, and the mid-face separates the pair | Opposes the stated cue |
| Other cue | Pair excess carried by an upper-face component | Not this cue |
| One-way other cue | Upper-face pair excess on Anger clips only. A one-way null, which is how humans show this pair | Not this cue |
| Spread | Pair excess present while any region shows, gone only under `mask_all_neutral`, with text primes that place the pair as neighbours | Not this cue, with the semantics rider |
| Global face | Pair excess from a component outside the regions, which survives all masks | No tested feature |
| Human-like | Mid-face share set to the human rates, with Carbon's rise and an upper-face separator | Same cue |
| Generic mask | Any mask raises No Emotion and lowers all foils, by area | Must not reach Same cue |
| Generic pull | Any mask raises Disgust on all clips, by area | Must not reach any match |
| Signal loss | Any mask weakens each clip's own label, by area | Must not reach either Same cue row |
| Partner pull | Any mask raises Anger on all clips, by area | Must not reach either Same cue row, or Opposes |
| Fill artefact, down and up | The neutral fill of `mid` moves the partner, and the other fills do not | Must not reach a Same cue row, or Opposes |
| Option prior, Position bias, Adjacency on all pairs | No pair excess, under a strong prior, position effect, or adjacency for every pair | Not shared |
| One-way weak cue | Mid-face excess on Disgust clips only, the weaker human direction | Must not reach Not this cue or Not shared |
| Split cue | Upper-face excess on Anger clips and mid-face excess on Disgust clips | Must not reach Not this cue |
| Happy up, Happy down (second audit) | Global face, with Happy moving .3 nats under `mask_mid` on clips of the pair | Must not reach a Same cue row, or Opposes |
| Happy cancel (second audit) | Same cue, with Happy falling as far as the partner, so y does not move | Must not reach Not this cue or No tested feature |
| Negative reverse (second audit) | Mid-face excess at the human rate forward, and as far below zero in reverse | Same cue, one direction, and must not reach Not shared |
| Score scale (second audit) | No sharing, with each acted emotion's scores multiplied by a constant, log SD .3 | Not shared |
| Incomplete first fill (second audit) | Same cue, with the first frame keeping half of what its fill should hide | Same cue |
| Boundary | True X at `shared_margin`, and a mid-face effect near the cue margin | Reported, not judged |

**Reported, not judged:** the actor-by-region SD at 0, .15, .6 and 1.0; the score scale at log SD .15, .5 and 1.0; the first fill keeping .8 and all of what it should hide; a cue carrying a half and a quarter of the excess; and 1.5x and 2x noise. 200 replicates each, 100 for noise.

Scenarios reach what no observer does, at 50 replicates each, and show only that the row can be reached: a failed gate and 8% of clips lost (Not tested), a sham with different pixels (`sham_failed`), a region under `min_span_tokens` (`region_too_small`), acted evidence outside the regions (`dose_failed`), adjacency in the MCQ with the pair also in the blind readout (`option_unresolved`), and actors split in sign (`shared_unresolved`).

Pass rule: for each non-boundary observer at 1x and 2x, the expected verdict in at least `expected_verdict_min` of replicates, any other named verdict in at most `other_named_verdict_max`, and Inconclusive in at most `inconclusive_max`. The Must not observers reach the forbidden verdict in at most `other_named_verdict_max`. Spread must also reach its rider in `expected_verdict_min`. Every scenario must reach its row, and every named row and reason code must be reached. 0.5x is reported as power and not judged. A failure is reported, and the rule is not relaxed to hide it.

**Recovery result, run 4, 2026-11-08, on the binding hash of this block (`a6_recovery.json`). It does not pass.** 500 replicates per observer and strength. Every named row, every reason code and every scenario was reached, and verdicts at the recovery's draws matched the real draws in 60 of 60. The checks that failed, at 1x and 2x unless stated:

| Observer | Expected | 1x | 2x | Check |
|---|---|---|---|---|
| Other cue | Not this cue | .02 | .15 | FAIL, mostly `cue_unresolved` |
| One-way other cue | Not this cue | .29 | .84 | FAIL at 1x |
| Spread | Not this cue, and its rider | .01 | .11 | FAIL |
| Global face | No tested feature | .01 | .13 | FAIL |
| Adjacency on all pairs | Not shared, Option-dependent at most .10 | .15 | .15 | FAIL on other named verdicts |

Everything else passed at 1x and 2x:

| Observer | Expected | 1x | 2x |
|---|---|---|---|
| Prior-only | No face use | .97 | .98 |
| No sharing, Option prior, Position bias, Score scale | Not shared | .91 to .96 | .94 to .95 |
| Adjacency | Option-dependent | .98 | .99 |
| Same cue, Human-like | Same cue | .99, 1.00 | .99, .99 |
| Incomplete first fill | Same cue | .90 | .99 |
| One-way cue, Negative reverse, One-way weak cue, Split cue | Same cue, one direction | .99 to 1.00 | .99 to 1.00 |
| Reverse | Opposes | 1.00 | .99 |
| Happy up, Happy down, Happy cancel, Fill artefacts, Generic mask and pull, Signal loss, Partner pull | Their forbidden rows at most .10 | at most .05 | at most .05 |

So the named cue results are recoverable, with no false positives over the limit, and both nulls at Stage 3 are not, at the noise recovery is judged at. That is reported, and no rule is relaxed to hide it. Reported, not judged: the share of the excess the mask removes was read back at .50 and .25 for cues carrying a half and a quarter (Same cue .82 and .34 there); a first frame keeping .8 and all of what its fill should hide gave Same cue .39 and `cue_unresolved` .95; 0.5x is in the tables. The direction rule, on the same seeds with both directions forced into play (`a6_recovery_both_directions_paired.json`): Same cue's full row at 0.5x is .80 with both and .76 with the rule, Human-like's .82 and .78, and Split cue reads Not this cue in neither. The runs before this one are kept: run 1 and run 2 as `a6_recovery_run1.json` and `a6_recovery_run2.json`, and run 3, which gave the same tables on code with a broken audit switch, was replaced by this run.

## Changes after first drafting, dated

All of these are from 2026-11-07, before any model output on the 819 clips exists. **No numeric margin changed**: `shared_margin` stays .05 and `cue_margin_fraction` .10. But rules changed after recovery results were seen, and the fill rule, the direction rule, the partner and Happy checks, and the excess required on the edit orders all changed what a verdict needs. The changes to X and to the outcome y also changed what both margins mean: `shared_margin` is now about three tenths of the human X rather than a quarter, and the cue margin is now signed within cells and is smaller for the same model.

**.05 now means something different.** Under the first X it was about a quarter of the human value (.198); under the second, about two fifths (.131); under the standardised share, about three tenths (.166). So the Not shared null admits a different share of the human effect from the one first intended. This is stated so that it is not read as an unchanged standard.

**After the pilot of the draft as written** (30 replicates at 1x, `a6_recovery_draft_as_written.json`):

| Change | Failing number |
|---|---|
| X replaced by shares on the same four options, per order | X floor of .113 with evenly spread errors at 60% accuracy, so Not shared was out of reach; 8% of observers over .05 from priors alone under a simpler fix (`prior_bias_check`) |
| `shared_family` by the unchanged rule: Anger-Happy for Disgust-Sad. Target unchanged | Consequence of the new X |
| Reference of y: A, No Emotion and both controls out | With the first reference, signal loss alone read as the predicted fall |
| Fill rule: 95% CIs of the first and blur fills beyond zero, also for Opposes | An artefact of the neutral fill alone reached a Same cue row in .57 of replicates at 1x and .50 at 2x, with the second X and the new reference (a scratch pilot of 30 replicates). In the pilot of the draft as written it reached Same cue in 1.00, but that pilot also had the first X and reference |
| Together | No sharing reached a Same cue row in .47 of replicates, and Global face in .73 (scratch pilot). Regenerated by `--variant first_draft` on the current observers, 30 replicates at 1x (`a6_recovery_draft_as_written.json`): No sharing reached a Same cue row in .53, Global face .53, Other cue .57, Signal loss .60, and Fill artefact 1.00 |

The other audit changes, which fixed gaps and no failing number: the Stage 2 table, a one-sided Stage 2 null, Stage 3 losses in Not tested, and the definitions in the text above. The pilot itself ran from a scratch script on code that has since changed, so its numbers cannot be regenerated from the repository; `a6_recovery.py --variant first_draft` reruns the first draft's X, reference and fill rule on the current observers, and its numbers are the ones that trace.

**After run 1** (500 replicates, hash `d48b928e...`, `a6_recovery_run1.json`). Run 1 passed every check it judged. One-way other cue was then reported, not judged, and reached Not this cue in 0 of 1,500 replicates, because condition 4 was required on both C1 and C2 and nothing moves a direction with no excess. Humans show this pair at .206 against .056, so a one-way model is a realistic outcome, and a null it could not reach would bias the amendment. Decided by the coordinator on 2026-11-07:

| Change | Why |
|---|---|
| Direction states at Stage 2, and `cue_directions: "excess_only"` | A direction with no excess cannot test the cue |
| Guard: a significant contrast in an absent direction gives `cue_unresolved` | So a weak real cue misread as absent cannot pass as a null |
| One-way other cue judged against `expected_verdict_min` | It was only reported in run 1 |
| Same cue and One-way cue judged against a false Not this cue. One-way weak cue and Split cue added | To test that the change does not make a false null easier |

Run 1 and run 2 differ in seeds as well as rule, because the observer list changed. Run 2's paired comparison, regenerated on run 3's seeds by `a6_recovery.py --variant both_directions`, is in `a6_recovery_both_directions_paired.json`.

**After run 2** (500 replicates, hash `afb5eaef...`, `a6_recovery_run2.json`). Run 2 passed every check it judged. A second, independent audit then found six blockers and lesser faults, with stress code that I reran: every one of its 38 rows reproduced exactly. Each change, with the number that failed:

| Change | Failing number |
|---|---|
| Partner check for named results, Happy check for nulls (`reference_check`) | Happy +.3 nats under `mask_mid`: Same cue 1.00 on Global face, .98 on Other cue. Happy -.3: Opposes 1.00 |
| Stage 1 writes a verdict for every model that stops there, including a model with no cells | Not tested and No face use never reached an output file |
| A carrying direction is a shared error; Not shared needs both directions absent; focus on the carrying direction | Forward .206 and reverse -.206: Not shared .65, with the forward direction carrying in 60 of 60. One-way weak cue: Not shared .116 in run 2 |
| Actor-by-region noise in recovery, `null_reach` at Stage 2, and the statement that the null may be unreachable | At SD 0, .3 and .6: Not this cue .95, .32, 0; No tested feature .95, .25, 0. False Same cue rows on null observers at SD .6 and 1.0: .033 to .05 |
| A6 code, every imported file, weights, dtype and versions pinned; Stage 2's sha256 in Stage 3's output; a binding hash for recovery | The design hash covered no verdict code |
| X as the standardised share | A scale per acted emotion of log SD .15, .3, .5: No sharing left Not shared in .23, .33, .38 |
| First-frame AU check against the apex, a freeze blocker; an incomplete first fill in recovery | A first frame keeping .5, .8 and all of what it should hide: Same cue .97, .47, and `cue_unresolved` .95 |
| Share of excess removed reported with Same cue; no floor | A cue carrying a half, a quarter and an eighth of the excess: Same cue .90, .32, .05 |
| Stage 3 blind readout dropped; `mask_lower_neutral` and MCQ baselines reported; failures by actor | Output with no reader cost 2.8 GPU-hours |
| Direction states on the edit orders for Stage 3, and an excess there required before a null | States were taken on all six orders, while the edits use two |
| Dose margin on the dose's own outcome; cue margin signed within cells | The dose was compared with a margin on y; the absolute margin grew with noise, over 3.5 times from noise SD .5 to 2 |
| Draft text: line 157 on comparability, the Inconclusive limit aligned to the pass rule, the pilot number, the reason-code order, geometry passed, constants into the block, landmarks stopping above the loss limit, MediaPipe asserted, the baselines file checked | Text and code that did not match |

**After run 4** (500 replicates, design `c1961048...`, binding `f22c2253...`, `a6_recovery.json`). Run 4 fails. The named results recover (Same cue, one direction, Opposes, Not shared, No face use and Option-dependent all at .91 or more, and no must-not observer over .05 on its forbidden row). The Stage 3 nulls do not: Other cue reads Not this cue in .02 at 1x and .15 at 2x, One-way other cue .29 and .84, Spread .01 and .11, Global face reads No tested feature in .01 and .13, all against .80. Even at actor-by-region SD 0, Not this cue reaches only .62 and No tested feature .70. Adjacency on all pairs gives Option-dependent in .15, against a limit of .10. Decided by the coordinator on 2026-11-08, before any model output:

| Decision | Why |
|---|---|
| The partner check stays as `reference_check`. The audit's rule (Happy's CI inside the cue margin for named results too) is reported as a sensitivity, `--variant happy_equivalence` | The audit's rule took Same cue on its own row to .09, against .99, and still let Happy up give a false Same cue in .08 |
| The Stage 3 nulls stay in the decision table but carry no claim: a model that would read Not this cue or No tested feature is reported as Inconclusive for the cue question, with recovery's reach beside it | Recovery shows the nulls are rarely reachable at the noise A6 is judged at. No margin or limit is moved to reach them |
| Adjacency on all pairs is descriptive only, with no verdict | Its false-positive rate fails the limit |
| The run is still worth doing | A shared error that carries a named cue, or opposes it, is recoverable at .99, and that is the cue-targeted test G8 asks for. Only the cue-absent reading is lost |

Run 4's failures stand in the record as failures. These decisions change what A6 may claim, not any rule or threshold, and the recovery is not rerun for them.



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
3. CPU jobs with no model: landmarks on 819 clips in `venvs/au_tools`, and the AU step for the human cue check. Run a decoder preflight on `.flv` under cv2 4.11 (`a6_run.py --step decoder`), since `decoder_preflight.py` reads EU-Emotion clips only.
4. Pin geometry and the two weight manifests (`a6_run.py --step weights`), the last PENDING items. Neither enters the binding hash, so recovery is not rerun for them. Check the first-frame AU ratio. The human target hash is pinned already, and any rerun must give the same bytes. Freeze. Tag.
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
| Baselines and text primes, 50 cells | 50 | under 0.02 | under 0.02 |
| Stage 3, MCQ, 547 clips by 14 conditions by 2 orders | 15,316 | 3.17 | 5.49 |
| **Total** | | **6.1** | **10.0** |

So about 16.1 GPU-hours of cell time, and **about 18 with A2's 10% allowance**, on both models. The Stage 3 blind readout was dropped in the second audit, since no rule read it. **If no model reaches Stage 3, it is about 8.** A3's measured rates (1.22 and 1.66 s per cell, decode included) would lift Stage 1's MCQ to 1.67 and 2.27 hours, which adds about 1.2 hours. Applied to Stage 3 as well, they add about 4.8 hours in all. Hashing the weights once per job adds a few minutes.

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
- **The noise sizes are A2's, not measured on CREMA-D**, since there is no model output to set them from, and nor is the actor-by-region SD. The nulls depend on them most. At 1x effect, from `a6_recovery.json` (500 replicates at the judged noise, 200 on the actor-by-region ladder, 100 at 1.5x and 2x of the other noise):

| Observer | Row | Actor-by-region SD 0 | .15 | .3 (judged) | .6 | 1.0 | 1.5x noise | 2x noise |
|---|---|---|---|---|---|---|---|---|
| Other cue | Not this cue | .62 | .29 | .02 | 0 | 0 | .01 | 0 |
| One-way other cue | Not this cue | .87 | .72 | .29 | .10 | .08 | not run | not run |
| Global face | No tested feature | .70 | .27 | .01 | 0 | 0 | 0 | 0 |
| Same cue | Same cue | 1.00 | 1.00 | .99 | .89 | .65 | 1.00 | 1.00 |
| Human-like | Same cue | 1.00 | 1.00 | 1.00 | .92 | .65 | 1.00 | 1.00 |
| No sharing | Not shared | .93 | .95 | .93 | .95 | .88 | .94 | .91 |

  False named cue rows on null observers stay low as the SD grows: at .6 and 1.0, Global face and Other cue reached a Same cue row or Opposes in at most .01, against .03 to .05 in the audit before the partner check.

- **The null may be unreachable on a real model.** In recovery, at the judged actor-by-region SD of .3, Not this cue was reached in .02 (Other cue) and .29 (One-way other cue) at 1x, and No tested feature in .01; the rest was `cue_unresolved`. With no such variation they reached .62, .87 and .70. The binding conditions are equivalence inside a cue margin that is now small for C2 (Disgust read as Anger, the weak human direction), and Happy's own change inside that margin, which held in about a fifth of replicates on C2. So a model whose shared error does not use the mid-face will most likely be Inconclusive, not null, and A6 can then say only that it found no evidence either way. This is the main reason the amendment may not be worth running as it stands.
- **The standardised share** is exact for priors, evidence and a scale per emotion, and not for a scale on the signal alone or other changes of shape. In recovery No sharing stayed Not shared in .95, .94, .96 and .88 of replicates at a scale of log SD .15, .3, .5 and 1.0, so it bears a scale per emotion of at least a factor of e either way. **Adjacency on all pairs** is not cancelled: it read as Option-dependent in .15 at 1x, over the .10 limit, which is a failed check, presumably because the standardisation is not additive across the orders in which a pair is adjacent and those in which it is not.
- **First fill.** If the first frame is already expressive, the first fill hides less than it should, and the fill rule loses power. The first-frame AU check blocks the freeze above `first_frame_au_ratio_max`.
- **Text primes** are one wording, with no test, and a model's reading of "A person feels angry." is not a measure of semantics in general.
- **Basic emotions only.** Nothing here reaches complex states, and EU-Emotion's Bored-Unfriendly question stays open.

## Fixed before freezing

1. Audit of this text, then code, then recovery, with outputs under `research/reproduction/a6_recovery.json`.
2. Check the `mid` landmark indices against rendered landmarks on the 24 calibration clips, which are outside the analysis. Confirm that the hull excludes the lip ring and spans the nose wrinkle region.
3. Confirm the **[proposed]** values.
4. Done: `a6_human_target.py` is written and reproduces the numbers above.
5. Pin the geometry hash and the weight manifests, and record the resulting `design_sha256`.
6. The first-frame AU check from the no-model AU step: the mean first-frame AU9 and AU10 over the apex mean, on clips of the pair, at or under `first_frame_au_ratio_max` for every detector (`first_frame_blocks_freeze`).
7. A decision on whether a design whose nulls may be unreachable is worth running as it stands (see Recovery result).

On item 2, a partial check only, on synthetic renders and not on calibration clips: on six FACSHuman renders under MediaPipe 1.2.2, the hull holds no lip-ring landmark, its dilated mask does not reach the undilated lip hull, it meets the dilated lip hull by 1.5% to 4.4% of its area, and about 3% of it lies on the inner eye rings. So `mid` as drafted takes a little of the inner eye corners, and `midlower` is one connected region. This does not close item 2, which needs real speaking faces under 0.10.14.

## Faults from A2's audit, and how A6 avoids them

| A2 fault | Here |
|---|---|
| Null rule uncoded or unexecutable | Null per stage, from named outputs, reachable on every contrast, though rarely reached at Stage 3 in recovery |
| Verdicts ignored gates | Gates are inputs; missing means not passed |
| Paths that made every verdict Inconclusive | End-to-end test on the real pinned paths |
| A significant contrast called null | A null requires a non-significant test |
| Dose passed on noise | Signed mean of actor means |
| Fill rule skipped | Applied to all four contrasts |
| A loss of the target read as a rise in another | Target, No Emotion and control emotions left out, so no clip's own label is in its reference, with a double difference |
| Area rule passing thin regions | Span rule |
| Margin from another outcome | Margin on the same outcome and clips, for the cue and for the dose |
| Shard and hash faults | `N_SHARDS` required, every output carries `design_sha256`, analysis refuses gaps |
| Recovery showed mechanics only | Graded effects, actor noise at the level of the test, adversarial observers |

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
