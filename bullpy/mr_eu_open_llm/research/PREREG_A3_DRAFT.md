# DRAFT amendment A3: face-input retest of RQ0

**Status: DRAFT. Not frozen and not run.** The coordinator freezes this before any A3 result exists. `scripts/face_input_validity.py` refuses to run without `--a3_frozen`, which may be passed only after that.

**Written:** 2026-11-04, before any rotated result was read. This does not edit `research/PREREGISTRATION.md` and does not replace A1.

## Why

A1 uses the composite grid on Gemma 4 and Molmo2. On Gemma 4 that leaves about 1 token across the face and drops frames after the eighth on longer clips. A failed A1 therefore cannot separate the model from the input. A3 reruns A1's design on `single_face`, where each model sees one face-cropped frame.

## Design: what is the same as A1

- The same 243 EU-Emotion items and six-option sets.
- Five cyclic rotations of the named labels (r = 0..4), with None of the above last. Rotation 0 is included, as in A1.
- `real` (the item's face) and `mismatch` (the same prompt with its partner's face).
- The same partners: `vision_mismatch_partners`, seed 42, on the same labels and actors. This is tested.
- The same prompt, `build_4afc_prompt(..., condition="video_only")`.
- The same estimand, decision rule, and statistics, imported from `research/scripts/analyse_rotation.py`.

## What differs from A1, stated in advance

**Input.** One frame per clip: the landmark apex frame, face crop, sent at 768 by 768. The frame comes from landmark movement alone, with no label or human data. Partners use their own apex crop.

**Scoring: exact option probabilities, not 20 draws.** Each option's joint logprob is read after the prefill "EMOTION:" and renormalised over the six options. An item is correct on a rotation if the target has the strictly highest probability; ties are wrong. This is the limit of A1's modal answer as draws grow, with three differences:

- It has no sampling noise, so it is deterministic.
- It cannot produce an unparsed answer, which A1 scores as wrong or as no vote.
- It reads the answer at a fixed position, not free text with a format.

So a difference between A1 and A3 could reflect scoring as well as input. A secondary row reports the mean target probability in place of correctness.

**Holm family.** A3's own three models. Not pooled with A1.

## Estimand and decision

For item i, a_i = mean correct over the 5 real rotations, and b_i = the same for mismatch. delta = mean(a_i - b_i): accuracy that requires the matching face.

A model is **vision-dependent on `single_face`** if the one-sided actor-level sign-flip p (100,000 draws) has Holm p < 0.05 across the three models, and the actor-bootstrap 95% CI (10,000 draws) of delta excludes 0 from above.

## Reading against A1

| A1 | A3 | Reading |
|---|---|---|
| Pass | Pass | Face use on both inputs |
| Fail | Pass | The grid limited A1; use `single_face` from here |
| Pass | Fail | Evidence from A1 needs inspection: the face crop may remove useful context |
| Fail | Fail | No demonstrated face use on either route; no RQ1 to RQ5 claims for that model |

## Preconditions

- The `single_face` delivery gate passes for that model (`research/reproduction/input_delivery_{model}.json`). If not, the model is reported as not tested.
- Geometry for all 243 clips is `results/stimulus_interventions/geometry.json`, SHA-256 `2bc76dd60be233ee6072ba5b2049721899c5febcd953ec49bff1dcc2f2c72383`. The script refuses any other file.

## Missing data

A face that cannot be localised fails every cell using it, and the reason is logged. An item enters only with all 5 real and all 5 mismatch rotations. All such failures are reported by label. If over 5% of items are lost for a model, the result stands and carries that as a stated limit (`over_lost_item_limit`). The decision rule does not change, and lost items are not replaced.

Delivery and geometry were complete when this was written: 243 of 243 clips usable, so losses are not expected.

**Partial lines.** A row counts only if it is valid JSON ending in a newline. A killed job can leave one bad final line. Resume and `--summarise` read cells through the same function, `read_cells`: that line is dropped and counted (`n_partial_lines_dropped`), and on resume it is cut from the file and its cell rescored. Scoring is deterministic, so a rescored cell equals the lost one. A bad line anywhere else stops the run.

## Secondary

Accuracy against 1/6, accuracy by target position, and Spearman and within-category correlation with human p_target, as in A1. Results also repeated without Kind, Unfriendly and Jealous, and scored against the human modal label.

## Limits

- One frame cannot carry dynamics. A3 tests face use, not the use of motion.
- The apex is chosen by movement, so it may catch speech-like or head movement instead of expression.
- The crop removes framing and body cues. That is the point, but Pass then Fail reads ambiguously.
- The prompt says "Analyze video frames only" while the model sees one image. It is kept identical to A1 on purpose, so this mismatch cannot be separated from the input change.
- A3 pins cv2 4.11 through `venvs/molmo2_cv411`, and `decoder_preflight.py` stops a job before it runs on another decoder. Results hold for that decoder's pixels.
- A1's option-set defect stands: label sets still identify category. Rotation and mismatch control it only in the way they do for A1.

## Commands

A mechanics run scores nothing that can be read: `--mechanics_only N` writes only status, reason, seconds, forwards and sent frame indices, outside `results/face_input_validity`.

    MECHANICS_ONLY=2 MODEL=gemma4 sbatch --time=00:15:00 slurm_jobs/study3_face_input_validity.sh

After freezing, one job per model, from `~/rds/hpc-work/study3`:

    A3_FROZEN=1 MODEL=gemma4 sbatch slurm_jobs/study3_face_input_validity.sh
    A3_FROZEN=1 MODEL=qwen3vl sbatch slurm_jobs/study3_face_input_validity.sh
    A3_FROZEN=1 MODEL=molmo2 sbatch slurm_jobs/study3_face_input_validity.sh

Each writes `results/face_input_validity/cells_{model}.jsonl`. A job that stops is resubmitted with the same command and resumes.

After all three finish, one job combines them. This applies the Holm correction across the three models, so it needs all three cell files, and writes `results/face_input_validity/face_input_validity_summary.json`:

    A3_FROZEN=1 SUMMARISE=1 sbatch --time=00:10:00 slurm_jobs/study3_face_input_validity.sh

## Cost

Measured by `--mechanics_only 2` on each model (jobs 39999429 to 39999431): all 20 of 20 cells ran, partners included, on full forwards. 243 items by 5 rotations by 2 conditions is 2,430 cells.

| Model | Forwards per cell | s per cell | GPU-hours for 2,430 cells |
|---|---|---|---|
| Gemma 4 | 7 | 1.22 | 0.82 |
| Qwen3-VL | 11 | 1.66 | 1.12 |
| Molmo2 | 11 | 2.01 | 1.36 |

About 3.3 GPU-hours in total. These two-item means include one clip decode per face, so they are close to what a full run pays. Each model fits one job at the 1 h 45 min limit, which leaves about 25% margin on Molmo, and a job that stops resumes.
