# Plan: fix the readout and the model input

**Written:** 2026-11-04, before any rotated result was read and before any prompt-probe result was read.
**Problems:** (1) the option-free prompt puts almost no probability on emotion words; (2) the composite grid truncates frames and gives Gemma about one token across the face.
**Rule for both:** fix the instrument using criteria that never touch target labels, human responses or A1 results. Do not lower a gate to pass it.

## Problem 1: option-free readout

**Calibration set.** 24 CREMA-D clips outside the 819 analysis clips, balanced by emotion and fetched separately, plus a grey image. Choosing the readout on these keeps analysis items clean. Labels and votes for these clips are not read. The 3 smoke items are used only as a mechanics check.

**Try in order; stop at the first rung that passes.**

| Rung | Readout | Pass criterion on calibration clips |
|---|---|---|
| 1 | Prefill: fix the chat-template position first, then the six candidate prompts, scored over the lexicon | Lexicon mass at least 0.05 on at least 90% of clips |
| 2 | Single-label yes or no: one forward per label, "Does this person look {label}?", no option list, read P(yes) | P(yes) + P(no) at least 0.90 on at least 90% of clips |
| 3 | Label likelihood: score each label's full continuation against a fixed template | No coverage meaning; flagged as the weakest readout |

Every rung must also pass **image sensitivity**: readouts from two different clips differ more than readouts from one clip and a 1-pixel shifted copy of it, on at least 80% of pairs, and real clips differ from the grey image.

**Choosing across models.** Use the highest rung all three models pass, so results stay comparable. If none is shared, choose per model and report that as a limit on cross-model comparison.

**If no rung passes for a model,** that model has no option-free readout. It stays on rotated MCQ only, and RQ4 is not run on it.

**Costs noted in advance.** Rung 2 multiplies forwards by lexicon size, and yes-bias must be modelled per label. Rung 3 may reflect label frequency, so differences within a label across images are the only valid contrast.

## Problem 2: model input

**Stop using the grid for causal work.** `legacy_full` survives only to compare with A1.

| Route | Input | Use |
|---|---|---|
| `single_face` | One face-cropped frame at a fixed resolution | Primary for spatial cues |
| `video_face` | Fixed frames in a native video route where supported | Temporal cues |
| `legacy_full` | Composite grid | A1 bridge only |

**Frame choice.** The peak frame is chosen from landmark movement alone, without labels or human data.

**Delivery gate, per model and route, before any stimulus test.**
1. At least 8 tokens across the face.
2. Blanking any sent frame changes the output.
3. Logged frame indices equal the frames actually sent.
4. Input ids do not change with frame content.
A route that fails is not used for that cue. A cue smaller than one token on every route is reported as not testable.

**Metadata fix.** `n_frames_used` overstates frames on the grid. After all rotation jobs finish, add `n_frames_shown` and a test, leaving grid behaviour unchanged so that A1 stays reproducible. No file read by pending rotation jobs changes before then.

**Amendment A3: face-input retest.** A1 cannot say whether a failure comes from the model or the grid, so run its design again on `single_face`: same 243 items, 5 rotations, real versus mismatch, same partners, and the same estimand and decision rule, as its own Holm family. Frozen before results, and it does not replace A1.

| A1 | A3 | Reading |
|---|---|---|
| Pass | Pass | Face use on both inputs |
| Fail | Pass | The grid limited A1; use `single_face` from here |
| Pass | Fail | Evidence from A1 needs inspection: the face crop may remove useful context |
| Fail | Fail | No demonstrated face use on either route; no RQ1 to RQ5 claims for that model |

## Order

1. Fix template position; run the ladder on the calibration clips.
2. Run the delivery gate on all routes for all models.
3. Timed full-pipeline smoke on Qwen and Molmo.
4. Freeze A3 and run it.
5. Fill the A2 gaps: analysis script, recovery check, eyes-and-mouth mask, human response predictor. Then freeze A2.
6. After rotation jobs finish: sync the flv fix, add `n_frames_shown`, analyse A1.

**Failure risks.** Qwen's image-list route has 16,320 media tokens against 253 and could be slow. Face detection could fail on some clips; those are logged, not dropped silently. Aperture editing stays out of confirmatory tests until its dose is verified.
