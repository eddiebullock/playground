# DRAFT amendment A9: replication of A4's confirmatory test on Liu et al. 2022, Experiment II

**Status: DRAFT, 2026-11-09, prepared for freezing (code written, design block hashed, recovery rerun on the final code). Not frozen until tagged `a9-freeze`. No model has seen a Liu render, and no render exists.** Nothing here changes A4 (FROZEN, tag `a4-freeze`), and A9 can never rescue, overturn, qualify or otherwise change any A4 verdict (see "Relation to A4").

Current `design_sha256` of the block below: `41c95dae865b0ee0e3c61051ac6caeb1516dfea1633bd013bf4fe16ab1a9ec29`.

**Recovery outcome, stated up front:** on the final code, design A fails 1 of 17 checks and design B passes 17 of 17, so B is selected with residual CEA confirmatory. With noise correlated across faces (rho .25 and .5), B fails one residual check, pair-swapped estimation accuracy at 0.25x (89% and 87% against 90%); every decision check passes. By lead decision 8 (made after seeing that result), decisions stay confirmatory and only the 0.25x accuracy claim is withdrawn for correlated face noise.

**Written:** 2026-11-09. Known at writing: A4's frozen text, code and outputs; the Liu human data, its human target and ceiling computed here (human data only); the recovery runs reported below (synthetic observers only). No A1, A4 or any other model result on avatars was read; none exists for Liu.
**Depends on:** exactly A4's gates per model: RQ0 through `rq0_status` at `single_face`, G2 to G4, and V1 to V4 as stated under "Validity gate".
**Models:** A4's Holm family, Gemma 4 and Qwen3-VL (`a4_design.MODELS`). Molmo2 is descriptive if run, as in A4.
**Code (new, nothing existing edited):** `scripts/a9_design.py` (reads the design block below, checks its hash, the A4 pins, the A9 input pins and the code pins; reads every shared value from `scripts/a4_design.py`), `research/scripts/a9_human_target.py` (human target, ceilings, AU-map and label checks, Chen check), `research/scripts/a9_manifest.py` (render manifest; renders through `a4_stimuli.render_tar`), `scripts/a9_recovery.py` (recovery through `a4_recovery` and `a4_analyse`), `scripts/a9_run.py` (runner glue: A4's `RenderStore` and `score`; refuses without `--a9_frozen` and pinned code), `scripts/a9_analyse.py` (analysis through `a4_analyse`; replaces only `analyse_dataset`, whose dataset lookup cannot take A9's name), `slurm_jobs/study3_a9.sh`, `slurm_jobs/study3_a9_render.sh`, `tests/test_a9.py`. Outputs: `research/reproduction/a9_human_target.json`, `a9_human_ceiling.json`, `a9_recovery.json` (final code), `a9_recovery_run1_draft.json` (first run, draft code, kept), `a9_chen_overlap.json`. Notes: `research/notes_a9.md`.

## Design block

A9's own constants are here and nowhere else; values shared with A4 are restated under `shared_with_a4` and checked against `scripts/a4_design.py` at import. The hash is sha256 of the bytes between the fence lines, excluding the fences, with a final newline.

```json
{
 "amendment": "A9",
 "liu": {
  "dir": "data/external/liu2022",
  "files_sha256": {
   "resp_threetask_20par.mat": "52602f372f39660fa6b8117598255f74cabaa54564774799e6155b5a28c275d5",
   "parameters_dynamic_precise.mat": "79e24df6b74d2dc76ad8ae95655042f8a2840c12538bacdd410dfd72106c143e"
  },
  "source": {
   "article": "Liu M, Duan Y, Ince RAA, Chen C, Garrod OGB, Schyns PG, Jack RE (2022). Facial expressions elicit multiplexed perceptions of emotion categories and dimensions. Current Biology 32(1), 200-209.e6. DOI 10.1016/j.cub.2021.10.035",
   "data_doi": "10.17632/dr853shk56.2",
   "licence": "CC BY 4.0"
  },
  "n_stimuli": 1200,
  "n_subjects": 20,
  "au_labels": [
   "AU1",
   "AU1-2",
   "AU2",
   "AU2L",
   "AU4",
   "AU5",
   "AU6",
   "AU6L",
   "AU6R",
   "AU7L",
   "AU7R",
   "AU9",
   "AU10Open",
   "AU10LOpen",
   "AU10ROpen",
   "AU11L",
   "AU11R",
   "AU12",
   "AU25-12",
   "AU12L",
   "AU12R",
   "AU13",
   "AU14",
   "AU14L",
   "AU14R",
   "AU15",
   "AU16Open",
   "AU17",
   "AU20",
   "AU20L",
   "AU20R",
   "AU22",
   "AU23",
   "AU24",
   "AU25",
   "AU26",
   "AU27i",
   "AU38",
   "AU39",
   "AU43",
   "AU7",
   "AU12-6"
  ],
  "response_codes": {
   "1": "happy",
   "2": "surprise",
   "3": "fear",
   "4": "disgust",
   "5": "anger",
   "6": "sadness",
   "7": "other"
  },
  "response_code_evidence": [
   "PMI_exp2.m (Mendeley Data 10.17632/dr853shk56.3) declares emotion_label = {'happy','surprise','fear','disgust','anger','sad','other'} and drops other by emo_all<7",
   "Chen et al. 2024 WE_resp_data.mat label_responses: Happy, Surprise, Fear, Disgust, Anger, Sad, Other",
   "top AU family per code in the Liu fit: happy AU12, surprise AU27, fear AU20, disgust AU09, sad AU17 then AU15"
  ]
 },
 "au_map": {
  "suffixes_dropped": [
   "Open",
   "i"
  ],
  "suffix_meaning": "UNVERIFIED",
  "collision_rule": "max",
  "amplitude_source": "p",
  "amplitude_decimals": 4,
  "uncovered_rule": "drop from human design and renders alike; list in outputs",
  "uncovered": [],
  "columns": [
   "AU01",
   "AU02L",
   "AU02R",
   "AU04",
   "AU05",
   "AU06L",
   "AU06R",
   "AU07L",
   "AU07R",
   "AU09",
   "AU10L",
   "AU10R",
   "AU11L",
   "AU11R",
   "AU12L",
   "AU12R",
   "AU13",
   "AU14L",
   "AU14R",
   "AU15",
   "AU16",
   "AU17",
   "AU20L",
   "AU20R",
   "AU22",
   "AU23",
   "AU24",
   "AU25",
   "AU26",
   "AU27",
   "AU38",
   "AU39",
   "AU43"
  ],
  "map": {
   "AU1": [
    "AU01"
   ],
   "AU1-2": [
    "AU01",
    "AU02L",
    "AU02R"
   ],
   "AU2": [
    "AU02L",
    "AU02R"
   ],
   "AU2L": [
    "AU02L"
   ],
   "AU4": [
    "AU04"
   ],
   "AU5": [
    "AU05"
   ],
   "AU6": [
    "AU06L",
    "AU06R"
   ],
   "AU6L": [
    "AU06L"
   ],
   "AU6R": [
    "AU06R"
   ],
   "AU7L": [
    "AU07L"
   ],
   "AU7R": [
    "AU07R"
   ],
   "AU9": [
    "AU09"
   ],
   "AU10Open": [
    "AU10L",
    "AU10R"
   ],
   "AU10LOpen": [
    "AU10L"
   ],
   "AU10ROpen": [
    "AU10R"
   ],
   "AU11L": [
    "AU11L"
   ],
   "AU11R": [
    "AU11R"
   ],
   "AU12": [
    "AU12L",
    "AU12R"
   ],
   "AU25-12": [
    "AU25",
    "AU12L",
    "AU12R"
   ],
   "AU12L": [
    "AU12L"
   ],
   "AU12R": [
    "AU12R"
   ],
   "AU13": [
    "AU13"
   ],
   "AU14": [
    "AU14L",
    "AU14R"
   ],
   "AU14L": [
    "AU14L"
   ],
   "AU14R": [
    "AU14R"
   ],
   "AU15": [
    "AU15"
   ],
   "AU16Open": [
    "AU16"
   ],
   "AU17": [
    "AU17"
   ],
   "AU20": [
    "AU20L",
    "AU20R"
   ],
   "AU20L": [
    "AU20L"
   ],
   "AU20R": [
    "AU20R"
   ],
   "AU22": [
    "AU22"
   ],
   "AU23": [
    "AU23"
   ],
   "AU24": [
    "AU24"
   ],
   "AU25": [
    "AU25"
   ],
   "AU26": [
    "AU26"
   ],
   "AU27i": [
    "AU27"
   ],
   "AU38": [
    "AU38"
   ],
   "AU39": [
    "AU39"
   ],
   "AU43": [
    "AU43"
   ],
   "AU7": [
    "AU07L",
    "AU07R"
   ],
   "AU12-6": [
    "AU12L",
    "AU12R",
    "AU06L",
    "AU06R"
   ]
  }
 },
 "dataset": "liu_WC",
 "task": "basic",
 "seed": 20261109,
 "human": {
  "fit": "human_au_effects per-subject ridge logistic, one-vs-rest, 33 columns, intercept only covariate (no gender in Liu), do(AU=1) minus do(AU=0), family_25",
  "lambda_grid": [
   0.01,
   0.1,
   1.0,
   3.0,
   10.0,
   30.0
  ],
  "ridge_selected": 1.0,
  "ceiling_ridge": 1.0,
  "model_side_gender": "rendered gender, centred, as a4_analyse.fit_tau; design B fits gender 0",
  "reference": "Liu against Snoek basic_WC, raw and residual, descriptive only"
 },
 "renders": {
  "identity_group": "WC",
  "replicate_configs": 200,
  "frame_count": 75,
  "sets": [
   "main",
   "replicate",
   "frame_max",
   "all_identities"
  ],
  "designs": [
   "A",
   "B"
  ],
  "design_selected": "B",
  "design_b_genders": [
   -0.5,
   -0.5,
   0.5,
   0.5
  ],
  "design_b_lost_rule": "a configuration is kept only if all four faces have a complete rotation set; over 5 percent lost fails",
  "frame_max": "secondary, descriptive, fitted on the human peak design",
  "manifest_sha256": "4b2f04aee90a46723227429751bf7a068d36499bbd419279856113acb547250a"
 },
 "recovery": {
  "rule": "a4_design.RECOVERY unchanged",
  "withdrawn_cells": [],
  "runs": [
   "A",
   "B",
   "B_rho0.25",
   "B_rho0.5"
  ],
  "face_noise_correlations": [
   0.25,
   0.5
  ],
  "correlated_noise": "stimulus logit noise per face = sqrt(rho) * shared + sqrt(1 - rho) * own, per configuration and option",
  "selection": "A if residual passes under A; else B if residual passes under B; else no confirmatory analysis",
  "correlation_use": "Lead decision 2026-11-09, made after recovery run 2 and applying the rule stated before it (narrowed to the conditions that pass), narrowed by check: under face-correlated noise every decision check passes and only pair-swapped estimation accuracy at 0.25x fails. All A9 decisions stay confirmatory whatever the estimated face-shared correlation. The estimation-accuracy claim at 0.25x is withdrawn for correlated face noise. Each model's face-shared residual correlation is estimated and reported descriptively beside its result, with no label attached to decisions. Thresholds never move.",
  "withdrawn_under_correlated_face_noise": [
   "liu_WC|0.25|pair_error"
  ]
 },
 "shared_with_a4": {
  "statistic": "residual",
  "sesoi_cfa": 0.3,
  "alpha": 0.05,
  "models": [
   "gemma4",
   "qwen3vl"
  ],
  "options": [
   "Anger",
   "Disgust",
   "Fear",
   "Happy",
   "Sadness",
   "Surprise",
   "None of the above"
  ],
  "human_labels": [
   "anger",
   "disgust",
   "fear",
   "happy",
   "sadness",
   "surprise",
   "other"
  ],
  "rotations": 6,
  "r_s_lower_bound": 0.85,
  "max_lost_share": 0.05,
  "cue_permutations": 9999,
  "signal_permutations": 999,
  "bootstrap_draws": 2000,
  "split_half_splits": 20,
  "ceiling_splits": 200,
  "baseline_stimuli": 50
 },
 "gates": {
  "inherited": [
   "rq0 via rq0_status at single_face",
   "g2_g4",
   "v1_detector from research/reproduction/a4_v1_pilot.csv"
  ],
  "recomputed_on_liu": [
   "v2_image_dependence",
   "v3_config_carries on the 200 replicate configurations",
   "v4_domain"
  ]
 },
 "a4_pins": {
  "scripts/a4_design.py": "a0bf7bb1c69ec7af1f11fabb5a9f31a6c694a30b92e624f1e1c241f161d062a8",
  "scripts/a4_analyse.py": "30f8f529d128b555ccfbf4d31ac3de70d91f90b1f5bfd52360ae9f39305fe56d",
  "scripts/a4_recovery.py": "12b15e89a366d772078034fd56b105532b347f1b73afe0b4efd5548ad3302526",
  "scripts/a4_run.py": "f842e574bf5cdaebbe9057d0ecc0e311c900dd54ff0ca221d1f9baa3768a8dc6",
  "research/scripts/a4_stimuli.py": "d0dff6930ede7f908e8d7faee2bc647adeb02f366601d04bea47353833715875",
  "research/scripts/render_au_faces.py": "6bdfa4e947b06aa7c2213dc31fbd297df4f8b97ca045a40905c6de4800c001d4",
  "research/scripts/human_au_effects.py": "c02a945ba578156cc4394e7d3a49761ad6e74495585034fbb059a4ce40199a2b",
  "research/scripts/analyse_rotation.py": "98890c5bff65f13abeb0549b61fe414dc9fa30082b84bcb564671683ec12e24f",
  "scripts/analyse_stimulus_interventions.py": "aba4f533daad6210e87f7bb4e68fc4627805cb4dfd9d9308df734fada4edc8b1",
  "scripts/vision_gates.py": "6327b6f88e99409cba5c63d5aca0fe97348fbce4f3b08522158a8d9ba3ce8194",
  "scripts/face_input_validity.py": "8a681a652d8e34065372d4f45bce650ba5276b5138e3aa64cf4310d65e93d818",
  "scripts/prompts.py": "871272d4aa6c86ff809f7631a4db81aeeb41564ba7f0a31809ea08e6dcf64ba7",
  "scripts/run_stimulus_interventions.py": "f7880c94e9671cc294cd4eaeece2108d7cd1d7a4c7e6f9e30a1c495964e6cd82",
  "research/reproduction/human_au_effects.json": "58c0f082cbf6aa4827054188ac2fc47da58a254c38da2c7ac202e8f52399310c",
  "research/reproduction/a4_v1_pilot.csv": "c735615a034c281909c96894c700d224638e281f55f995a0959eeef8a6fb3994"
 },
 "input_pins": {
  "research/reproduction/a9_human_target.json": "3f93ccbe2506ad9e01d79c8f594da6eb0bcd33790d410fda2d3945aeaa526ab0",
  "research/reproduction/a9_human_ceiling.json": "a32e2dbe2e41b00b19594be1261d4aebe78dfb6170e3ab0837d3f612e62de669"
 }
}
```

## Question

The A4 question, on a second, independent human sample: when a face's action units are set at random, does a model's judgement change with each AU the way human judgements do? A9 asks it of Liu et al.'s Experiment II observers instead of Snoek et al.'s, with the same estimand, statistic, decision rule and smallest effect of interest.

**What A9 adds and what it does not.** Liu Experiment II has new observers (20) and new random stimuli (1,200), so it addresses "A4 rests on one human sample". It shares with Snoek the lab (Glasgow), the GFG generator and AU library, the response protocol and Western culture, so a shared artefact of that platform would replicate. Model-side, A9 uses the same renderer, identities, prompt and scoring as A4, so A9 does not address "A4 rests on one stimulus platform" either.

## Relation to A4

1. A4's verdicts are final as frozen. A9 never enters A4's Holm family, headline, decision table or any A4 sentence that states a verdict. A9 is reported in its own section, after A4's.
2. A9 has its own Holm family: the two eligible models on the one A9 dataset.
3. No pooling. A9 and A4 estimates are never averaged, meta-analysed or combined into one decision. A descriptive table may place them side by side, labelled as separate tests.
4. If A9 and A4 disagree for a model, both stand and the disagreement is reported as such; neither is reinterpreted.
5. If A4 is Inconclusive for a model, an A9 Aligned or Not aligned does not make A4 conclusive.

## Lead decisions, 2026-11-09

Recorded as given, before freezing and before any model saw a Liu render.

1. **Design B, with `frame_max`.** B is the selected design (recovery rule below); `frame_max` is run as a secondary, descriptive set.
2. **Face independence is tested, not assumed.** Recovery adds design B with model noise correlated across the four faces of a configuration at rho .25 and .5 (fixed in the block), under A4's unchanged rule, reported per check. Use, as stated before run 2: if B fails under correlation, the paper says so and the confirmatory claim is narrowed to the conditions that pass; thresholds never move. (The draft first operationalised this as a scope label on decisions, by the largest passing rho; decision 8 replaces that operationalisation.)
3. **Response-code order** rests on the `PMI_exp2.m` declaration, Chen et al.'s identical order and the AU signatures; the evidence is recorded in the block (`response_code_evidence`); no further fetch.
4. **Suffixes** "Open" and "i": disclosed as unverified (block: `suffix_meaning`), as A4 discloses its lip-parting approximation.
5. **Gender asymmetry** kept: the human fit has no gender column (Liu releases none); the model fit uses rendered gender as A4's frozen `fit_tau` does (gender 0 under design B, where faces are balanced). Disclosed.
6. **Liu against Snoek basic WC** is a descriptive yardstick only.
7. **The file is used as released**, including the 323 stimuli with 5 or 6 AUs; disclosed. Contacting the authors is the user's decision and optional.
8. **Narrowing by check (2026-11-09, made after seeing recovery run 2).** Applying decision 2's rule to run 2's result: under face-correlated noise every decision check passes (false Aligned, prior-only signal, boundary Not aligned, human-like Aligned, half-shared, different-cues Not aligned) and only pair-swapped estimation accuracy at 0.25x fails. So (a) all A9 decisions, Aligned, Not aligned and Inconclusive, stay confirmatory whatever a model's estimated face-shared correlation; (b) the estimation-accuracy claim at 0.25x is withdrawn for correlated face noise (block: `withdrawn_under_correlated_face_noise`), and the paper states this, as A4 withdrew its failing cells; (c) each model's face-shared residual correlation is estimated and reported descriptively beside its result, with the recovery table, and no label is attached to decisions. **Timing, disclosed:** this decision was taken after run 2's result was known, by the lead, applying the narrowing rule stated before the run; it was not part of the pre-run text. The block was changed accordingly (new design hash) and recovery rerun on the re-pinned code (run 3).

## Human data

**Source.** Liu M, Duan Y, Ince RAA, Chen C, Garrod OGB, Schyns PG, Jack RE (2022). Facial expressions elicit multiplexed perceptions of emotion categories and dimensions. *Current Biology* 32(1), 200-209.e6, DOI 10.1016/j.cub.2021.10.035. Data: Mendeley Data, DOI 10.17632/dr853shk56.2, CC BY 4.0. Files `resp_threetask_20par.mat` (sha256 `52602f37...c275d5`) and `parameters_dynamic_precise.mat` (sha256 `79e24df6...c143e`), pinned in `a9_design.LIU_FILES`, gitignored, never copied into a tracked file.

**Experiment II, as the article states (STAR Methods, read in full for these points).** 20 white Western observers (10 female, 10 male, mean age 20.3) each categorised the same 1,200 random facial animations, in random order, as happy, surprise, fear, disgust, anger, sad, or "other" ("if they perceived that it did not [represent an emotion], including blended/compound emotions"); valence and arousal were rated in separate blocks and are not used here. Each animation lasted 1.25 s and combined 1 to 4 AUs (median 3) from 42, each with six temporal parameters. Each was shown on a randomly generated white identity: 600 female and 600 male, aged 20 to 40.

**What the files hold (checked).** `resp_emo` 1,200 by 20, codes 1 to 7, no missing values (24,000 responses). `stimuli_raw` 42 by 1,200 binary. `p` 42 by 1,200 peak amplitudes, continuous in (0, 1], and `p > 0` equals `stimuli_raw` exactly. `pseq` 42 by 75 by 1,200, the per-frame amplitude of each AU; its maximum over frames equals `p` to within .0014. `params.labels` gives the 42 AU names, identical for all 1,200 stimuli. Active raw AUs per stimulus: 1 to 6, mean 3.83, median 4 (17, 112, 328, 420, 250 and 73 stimuli with 1 to 6). The article describes the procedure, which Experiment II says it reuses, as 1 to 4 AUs with median 3; 323 Liu stimuli have 5 or 6. The discrepancy is unexplained and the file is used as released. **Not in the files:** face gender and identity per stimulus, and trial order.

**Response codes.** 1 happy, 2 surprise, 3 fear, 4 disgust, 5 anger, 6 sad, 7 other. Source: Liu's released analysis code (Mendeley Data v3, `PMI_exp2.m`), which declares `emotion_label = {'happy','surprise','fear','disgust','anger','sad','other'}` and drops "other" by `emo_all<7`. That variable is declared but not used to index in the script, so the order rests on the declaration. It is corroborated, not set, by two further sources: the same lab's Chen et al. 2024 deposit labels its codes Happy, Surprise, Fear, Disgust, Anger, Sad, Other; and in the human fit the top AU family per code is AU12 for happy, AU27 for surprise, AU20 for fear, AU09 for disgust, AU17 then AU15 for sad (`label_corroboration_top3_families_by_beta`). Codes are mapped to A4's label order (anger, disgust, fear, happy, sadness, surprise, other) before any fit.

**Why not Chen et al. 2024.** Checked at trial level on 2026-11-09 (`a9_chen_overlap.json`): Chen et al.'s Western file (OSF 3m95w, `WE_resp_data.mat`) is the Snoek basic WC experiment. In Snoek's subjects 16 to 60, every one of the 2,342 rows per subject has a unique exact match, across all 45 subjects' responses, to a Chen trial, and per-subject agreement after alignment is 1.000 for all 45. In subjects 01 to 15, 2,211 of 2,368 rows match exactly and per-subject agreement is .964 to .9996 (median .998). Chen's AU coding differs only in labelling: it lacks AU7, AU12L and AU12R as raw codes, and codes AU41 and AU42, which Snoek merged into AU43. Chen et al. 2024 is therefore not a second sample and is not used. Chen's East Asian file was not downloaded; that it matches Snoek basic EA is UNVERIFIED.

## Differences from A4, and how each is handled

| Difference | Snoek (A4) | Liu (A9) | Handling, fixed now |
|---|---|---|---|
| AU coding | 42 raw AUs recoded by Snoek to 33 columns | Same 42 raw AUs, released raw | Recoded to A4's 33 columns by Snoek's own rule (below), verified against Snoek's data. Same 25 families |
| Amplitude | Six steps, rounded to one decimal in the columns | Continuous peak amplitude 0 to 1 | Peak amplitude, rounded to 4 decimals (one value rounds to zero and is dropped), on both sides. No rounding to one decimal. Presence coding is a sensitivity |
| Presence vs amplitude | Amplitude only | Both; Liu's own analysis used presence | Amplitude is primary, as in A4; presence-coded fit on both sides is a sensitivity |
| Motion | 1.25 s animations; time course not public | 1.25 s animations; full time course public (`pseq`) | Primary renders use each AU at its peak amplitude, as A4. Secondary set `frame_max`: the one frame with the largest summed activation |
| Culture and groups | WC and EA, two datasets per task | One Western group | One dataset, `liu_WC`; no WC and EA agreement rule; the per-model decision is A9's verdict |
| Observers | 60 (basic) | 20 | Same per-subject fit; ceiling recomputed; recovery on Liu's own effect sizes |
| Stimuli | 2,323 to 4,659 configurations, partly shared within sets | 1,200, all seen by every observer | All 1,200 used. A4 needed all configurations to pass recovery, so recovery decides between designs A and B (below) |
| Faces humans saw | 8 base faces per group (Western 01-15: blends) | 1,200 random identities, 600 per gender, gender per stimulus not released | Models see A4's four Western faces. Gender assigned by seeded hash, 600 and 600 |
| Face gender in the human fit | Centred covariate, unpenalised | Not available | Human fit has the intercept as its only unpenalised covariate. Model-side fit keeps A4's rendered-gender covariate unchanged; gender is randomised against AUs on both sides, so the marginal do effect is the same estimand |
| Options | 7AFC: six labels and other | Same labels; "sad" for sadness; "other" includes blends | Same seven options, same order, same abstain mapping as A4 basic |
| Ridge | Selected per dataset, 0.1 for basic | Selected by the same rule | Selected value 1.0, fixed now (below) |

## AU map and renderer coverage

**Rule.** Each Liu label becomes a Snoek raw token by dropping the suffixes "Open" and "i" (AU10Open is Snoek's AU10, AU16Open AU16, AU27i AU27). The 42 stripped names equal Snoek's 42 raw tokens exactly. Each token then sets A4 columns by Snoek's recoding (`human_au_effects.DATA_CAVEATS`): a bilateral AU sets both L and R, a one-sided AU its own side, a compound each part; where two raw AUs set one column, the column takes the maximum. Applied to the raw tokens kept in Snoek's own row labels (basic EA, conversational EA and WC), this rule reproduces Snoek's 33 columns for all 6,966 configurations, with zero mismatches and 55 collisions resolved by the maximum (`au_map_check_against_snoek`). The suffix correspondence itself is inferred from the names and the one-to-one match of the two sets; GFG's documentation was not available (UNVERIFIED that "Open" denotes added lip parting).

**Coverage.** All 42 Liu AUs map to A4's columns, and all 33 columns and 25 families are reached. The A4 renderer draws 31 columns from FACSHuman targets and approximates AU22 and AU27 (`research/notes_au_render.md`; `render_au_faces.COVERAGE`). No Liu AU is uncovered.

**Rule for uncovered AUs, fixed now.** A Liu AU with any part lacking a column or renderer target would be dropped from the human design and from the renders alike, and listed. With this renderer the list is empty (`a9_design.UNCOVERED`), and the tests fail if that changes.

**Approximations carried from A4, and two specific to Liu's raw names.** AU22 and AU27 are approximated in renders, as in A4. The Open variants (AU10Open, AU10LOpen, AU10ROpen, AU16Open) and AU27i are drawn with the A4 target of their column, so any lip parting GFG adds is not drawn; compounds are drawn as the sum of their parts at one amplitude. These are the renders A4 already uses for the same raw AUs in Snoek's data.

| Family | Liu AUs | A4 columns | Stimuli with family active (of 1,200) | Render | Share of residual human energy |
|---|---|---|---|---|---|
| AU01 | AU1, AU1-2 | AU01 | 240 | direct | .046 |
| AU02 | AU1-2, AU2, AU2L | AU02L, AU02R | 326 | direct | .031 |
| AU04 | AU4 | AU04 | 115 | direct | .018 |
| AU05 | AU5 | AU05 | 115 | direct | .012 |
| AU06 | AU6, AU6L, AU6R, AU12-6 | AU06L, AU06R | 407 | direct | .006 |
| AU07 | AU7L, AU7R, AU7 | AU07L, AU07R | 299 | direct | .009 |
| AU09 | AU9 | AU09 | 119 | direct | .137 |
| AU10 | AU10Open, AU10LOpen, AU10ROpen | AU10L, AU10R | 303 | direct (Open part not drawn) | .111 |
| AU11 | AU11L, AU11R | AU11L, AU11R | 247 | direct, barely visible | .008 |
| AU12 | AU12, AU25-12, AU12L, AU12R, AU12-6 | AU12L, AU12R | 479 | direct | .205 |
| AU13 | AU13 | AU13 | 109 | direct | .010 |
| AU14 | AU14, AU14L, AU14R | AU14L, AU14R | 289 | direct | .087 |
| AU15 | AU15 | AU15 | 134 | direct | .018 |
| AU16 | AU16Open | AU16 | 58 | direct (Open part not drawn) | .009 |
| AU17 | AU17 | AU17 | 111 | direct | .017 |
| AU20 | AU20, AU20L, AU20R | AU20L, AU20R | 309 | direct | .043 |
| AU22 | AU22 | AU22 | 127 | approximated | .001 |
| AU23 | AU23 | AU23 | 121 | direct | .004 |
| AU24 | AU24 | AU24 | 122 | direct | .014 |
| AU25 | AU25-12, AU25 | AU25 | 199 | direct | .010 |
| AU26 | AU26 | AU26 | 112 | direct | .074 |
| AU27 | AU27i | AU27 | 103 | approximated | .098 |
| AU38 | AU38 | AU38 | 118 | direct | .002 |
| AU39 | AU39 | AU39 | 120 | direct | .003 |
| AU43 | AU43 | AU43 | 118 | direct | .027 |

Families with a bilateral composite use it, as in A4 (`family_25`). Liu has no right-only AU2, as Snoek has none, so AU02 is read through the composite. AU16 is active on only 58 stimuli, the thinnest family.

## Human target

**Fit.** A4's human fit, through `human_au_effects`' own functions (`fit_ridge_logit`, `do_effects`, `effect_index`, `family_positions`, bootstrap and split-half): one ridge logistic per subject and label, one-vs-rest, on the 33 columns, intercept unpenalised and no gender column; subjects averaged with equal weight; G-computation of do(AU = 1) minus do(AU = 0), other AUs as observed, renormalised over labels; options by 25 families. Subject-bootstrap CIs, 2,000 draws.

**Ridge selection, the A4 rule.** Held-out log loss on unseen stimuli, two folds by seeded draw, summed over subjects and labels, over the grid .01, .1, 1, 3, 10, 30 (equal to `human_au_effects.LAMBDA_GRID`). Selected: **1.0** (loss .3382 against .3491 at .1 and .3435 at 3; intercept only .3962). Fixed now for the model-side fit. Recomputing it after any model data is not allowed.

**Result, human data only** (`a9_human_target.json`). Effects sum to zero over labels (maximum absolute column sum 0.0). One subject never chose "other". Correlation of the selected-ridge matrix with the lowest-ridge matrix .950. Common-profile share of squared effects .150 (Snoek basic: .26 to .35). AU22 and AU27 hold .123 of squared effects. The four detector-confirmed families of A4 (AU01, 05, 25, 26) hold .143 of residual energy. Presence-coded target against the amplitude target: raw .989, residual .955 (presence ridge selected 3.0 by the same rule). Prototype check (reported, never used): surprise 4 of 4 prototype AUs supported, disgust 3 of 4, anger and fear 2, happy and sad 1.

**Ceiling, subjects and stimuli split** (`a9_human_ceiling.json`), 200 splits, halves of 10 subjects and 600 stimuli sharing neither, pooled fits at ridge 1.0 as `a4_analyse.human_do_ceiling` does for A4 (here the selected ridge is also 1.0, so the A4 caveat that the ceiling is for a related estimator does not arise): split-half r .905, **ceiling .975 raw, .962 residual** (split-half residual r .862, SD .034). For comparison only, subjects split with stimuli shared: .987 raw, .983 residual.

**Human reference, descriptive.** A4's yardstick is one human group against another. Liu has one group, so the yardstick here is Liu against Snoek basic WC, on the same basis: **raw .861, residual .690**; Liu against Snoek basic EA: raw .815, residual .709; Snoek WC against EA, for scale: raw .877, residual .780. These include differences in stimuli, faces, sample and amplitude coding, not only in observers. A model is called human-level on A9 only if its CI reaches .690 (residual), and that is descriptive.

## Stimuli and render manifest

Built by `research/scripts/a9_manifest.py` from `p` through `a9_design.liu_to_columns`, rendered by `a4_stimuli.render_tar` with `render_au_faces.render_face`: headless, 768 by 768, frontal, flat light, black background, as A4. Rows have A4's manifest columns; `au_json` is a map over the 33 columns, amplitudes in 0 to 1. Images go into tar files, never loose and never into git, as A4.

| Set | Content | Rows |
|---|---|---|
| `main` | All 1,200 configurations, peak amplitudes, one face each | 1,200 |
| `replicate` | First 200 by rank on the other face of the same gender (V3) | 200 |
| `frame_max` | All 1,200 at the single frame of largest summed activation (of 75), on the `main` face; secondary, descriptive | 1,200 |
| `all_identities` | Design B only: each configuration on the three faces not used in `main` | 3,600 |

Design A: 2,600 rows, 2,592 unique images (manifest sha256 `dac1cd5a...` as built 2026-11-09). Design B: 6,200 rows, 5,992 unique images (`4b2f04ae...`). A9 renders design B, the design recovery selected (block: `design_selected`), including `frame_max`. Manifest sha256 `4b2f04aee90a46723227429751bf7a068d36499bbd419279856113acb547250a`, pinned in the block; the runner and the render job refuse any other.

**Rank.** Seeded hash of the stimulus index (`a9_design.SEED` 20261109), no use of responses. It picks the replicate set and the 50 image-free stimuli for V2.

**Identities.** A4's four Western faces (`female_a`, `female_b`, `male_a`, `male_b`). Gender: 600 and 600 by seeded hash of the index, matching the humans' totals but not their per-stimulus genders, which are unknown. Face within gender by seeded hash. Humans saw 1,200 different faces; models see four, so identity coverage is far narrower than for humans.

**Design A (primary) and design B (fallback), both fixed before recovery ran.** A: each configuration once, on one face, fitted with rendered gender, as A4. B: each configuration on all four faces; a configuration's response is the mean over its four faces' rotation-averaged probabilities, fitted with gender 0 (two faces of each gender). Rule: A is used if residual CEA passes recovery under A; otherwise B if it passes under B; otherwise A9 has no confirmatory analysis. The statistic is never switched to raw. B's gain depends on how independent a model's response noise is across faces, which the correlated-noise runs test (decision 2). Under B, a configuration is kept only if all four faces have a complete rotation set; losing over 5% of configurations fails the dataset, as A4's lost-share rule.

## Task, route and scoring

Unchanged from A4 basic: one image, sent as on `single_face`, `build_4afc_prompt(options, condition="video_only")`; options Anger, Disgust, Fear, Happy, Sadness, Surprise, None of the above; human "other" maps to None of the above, kept last; all six cyclic rotations; exact option probabilities averaged over the complete rotation set; a stimulus without a usable response on every rotation leaves, and losing over 5% fails the dataset. The prompt's mention of video frames is as in A3 and A4. The blind readout is not run for A9.

**Static against dynamic.** As in A4, models see one static frame and humans saw 1.25 s of motion. Liu's `main` renders set every AU at its own peak, a frame that may never have occurred (peak latencies range from 0.20 to 0.67 in the file's units). `frame_max` is the only frame that did occur; it is descriptive.

## Estimand, statistic and inference (unchanged from A4)

tau(j, k) is the average effect of setting cue family j on option k, do(AU = 1) minus do(AU = 0), other AUs as observed, from the model's responses treated as one subject with fractional responses, through `a4_analyse.fit_tau` at the human ridge (1.0). Residual CEA, cos(R(tau_model), R(tau_human)), over options by 25 families, is confirmatory; raw CEA is secondary. Signal: 999 shuffles, p < .05. Alignment: 9,999 shuffles of AU labels across the 25 columns, one-sided. Holm across Gemma 4 and Qwen3-VL within A9 only. Interval: stimulus bootstrap, 2,000 draws. R_S: split-half across stimuli, 20 splits. SESOI .30. Code: `a4_analyse.estimate`, `decide`, `remove_common`, imported unchanged.

## Decision rule (unchanged from A4), per model

| Outcome | Requires |
|---|---|
| **Aligned** | All gates; signal p < .05; alignment Holm p < .05 within A9; residual CEA at least .30; CI lower bound above 0 |
| **Not aligned** | All gates; signal p < .05; CI upper bound divided by the square root of R_S below .30 |
| **Inconclusive** | Anything else, including any failed or missing gate, or no recovery-selected design |

Not aligned is written "not aligned on these renders". A missing or non-positive R_S cannot support Not aligned. Below R_S .85, uncorrected CEA reads as a lower bound. With one dataset there is no WC and EA headline: the per-model decision is A9's result for that model.

## Validity gate

Missing means not passed. RQ0 and G2 to G4 as in A4. V1 (detector) is inherited from A4's frozen pilot (`a4_v1_pilot.csv`), because it tests the renderer and the four Western identities, which A9 uses unchanged (pinned by `render_au_faces.py`'s sha256 in A4); its thinness, disclosed in A4, carries. V2 (image dependence, 0.25 nats, 50 stimuli by rank), V3 (configuration carries across faces, sign-flip p < .01 on the 200 replicate configurations) and V4 (mean None of the above mass at most .90, any option at most .95) are recomputed on Liu renders with A4's rules and code.

## Sensitivity and descriptive analyses

Beside the primary, never replacing it, each on residual and raw CEA: presence coding on both sides; without AU22 and AU27; without the abstain row; flagged and confirmed families apart (A4's lists); `frame_max` renders fitted on the human peak design. Descriptive: output r, modal agreement and kappa against the mean human response per configuration (Liu observers all saw the same 1,200 configurations, so this is per configuration, though not per image).

## Recovery

Synthetic observers answer Liu's real design through A4's estimator, by `scripts/a9_recovery.py` importing `a4_recovery` and `a4_analyse`: the same eight observers, built from Liu's human log-odds matrix, the same realism (stimulus logit noise SD 1 per option, lognormal AU gain error log SD 0.5), strengths .25, .5 and 1, 100 replicates per cell, and A4's pass rule (`a4_design.RECOVERY`) unchanged. A4's two withdrawn cells name A4 datasets, so **no A9 cell is withdrawn** (block: `withdrawn_cells`), which makes A9's rule stricter than A4's as applied. Seventeen checks per statistic and run. Under the correlated runs, each face's stimulus noise is sqrt(rho) times a shared draw plus sqrt(1 - rho) times its own, per configuration and option, before the four faces are averaged.

**Every run, in order. The draft-code run is kept.**

| Run | Code | Design | Statistic | Checks passed | Result |
|---|---|---|---|---|---|
| 1 | draft (`a9_recovery_run1_draft.json`, sha256 `625675e8...0afe`) | A | Residual | 16 of 17 | FAIL: pair-swapped within .10 of truth 68% at 0.25x |
| 1 | draft | A | Raw | 16 of 17 | FAIL: same check, 86% |
| 1 | draft | B | Residual | 17 of 17 | PASS |
| 1 | draft | B | Raw | 17 of 17 | PASS |
| 2 | pre-decision-8 code (`a9_recovery_run2.json`, sha256 `79229b70c4a7d21cfcf9bd957669a899c9b503082932e77cddf94d4aced56e84`) | A | Residual | 16 of 17 | FAIL: pair-swapped 64% at 0.25x |
| 2 | final | A | Raw | 16 of 17 | FAIL: same check, 81% |
| 2 | final | B | Residual | 17 of 17 | **PASS** |
| 2 | final | B | Raw | 17 of 17 | PASS |
| 2 | final | B, rho .25 | Residual | 16 of 17 | **FAIL**: pair-swapped 89% at 0.25x (rule 90%) |
| 2 | final | B, rho .25 | Raw | 17 of 17 | PASS |
| 2 | final | B, rho .5 | Residual | 16 of 17 | **FAIL**: pair-swapped 87% at 0.25x |
| 2 | final | B, rho .5 | Raw | 17 of 17 | PASS |
| 3 | final, after decision 8 (`a9_recovery.json`, sha256 `01958263f95c6a2fb614d7e66093eb6ed61e11ede33415d14bc8fab8d076dd85`) | A, B, B rho .25, B rho .5 | Both | as run 2 | Every table and check identical to run 2 (same seeds and recovery code); only `design_sha256` and `code_sha256` differ |

Run 2 carried the block hash `b548bc82...e56c`. Decision 8 changed the block (new hash `41c95dae...ec29`) and `a9_analyse.py`, so recovery was rerun as run 3, which carries the new block hash and the code hashes it ran on, equal to the pins under "Code hashes"; its tables and checks are identical to run 2's, so the recovery verdicts did not change. Its seeds differ from run 1's (the run name is part of the seed), so the two runs are independent draws; their verdicts agree. Run 2 took 81 minutes and run 3 135 minutes on 7 CPU workers.

**Consequences.** Design B is selected, with residual CEA confirmatory (`choice`, equal to the block's `design_selected`). B fails one check at rho .25 and at .5, the same one: pair-swapped estimation accuracy on weak effects (0.25x human strength). No decision check fails under correlation: false Aligned at most .04, prior-only signal at most .07, boundary Not aligned 0 on residual CEA, human-like Aligned 1.00 at every strength, different-cues Not aligned at least .94 at 1x. By decision 8 (taken after this result, applying the rule stated before it), decisions stay confirmatory, the 0.25x estimation-accuracy claim is withdrawn for correlated face noise, and each model's face-shared residual correlation is reported descriptively with no label on its decision. The file still records `r_star` (0, the largest tested rho at which every check passes), now descriptive only. Thresholds and the decision rule are not changed.

| Rate, residual CEA, run 2 | A 0.25x | A 1x | B 0.25x | B 0.5x | B 1x | B rho .25, 0.25x | B rho .5, 0.25x |
|---|---|---|---|---|---|---|---|
| Human-like Aligned | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Human-like CEA, mean (truth) | .733 (.852) | .877 (.894) | .823 (.859) | .855 (.873) | .882 (.889) | .779 (.846) | .771 (.856) |
| Human-like R_S, mean | .721 | .953 | .901 | .967 | .983 | .842 | .799 |
| Half-shared Aligned | .72 | .84 | .94 | .84 | .84 | .79 | .75 |
| Different-cues Not aligned | .93 | .98 | .97 | .98 | .94 | .98 | .94 |
| False Aligned, highest of four observers | .02 | .01 | .01 | .01 | .06 | .01 | .02 |
| Pair-swapped within .10 of truth | .64 | .99 | .95 | 1.00 | 1.00 | .89 | .87 |

At 0.5x and 1x, B passes every check at both correlations (pair-swapped accuracy .96 to 1.00). Pair-swapped observers' outputs correlate .88 to .95 with the human-weighted observer on natural-like stimuli (pair r .9, our construction), while their residual CEA is about .38 to .40, as in A4.

**Disclosures on recovery.** Design B was written after A4's recovery history was known, before any A9 recovery number existed. The correlated-noise rhos (.25, .5) and their use were fixed by the lead before run 2. Recovery used 100 bootstrap draws and 6 splits per replicate, against 2,000 and 20 in the real analysis. The ridge was 1.0, the selected human value.

## Disclosures

- **Same platform.** Lab, generator, AU library and protocol are shared with Snoek; A9 is a second sample, not a second method.
- **Human reference** is cross-dataset (Liu against Snoek basic WC), not cross-group, mixes stimulus, face and coding differences with observer differences, and is descriptive only (decision 6).
- **Gender asymmetry** (decision 5): absent from the human fit, rendered gender on the model side as A4's frozen code fits it, gender 0 under design B.
- **Four faces** against 1,200 seen by humans.
- **Suffix mapping** (Open, i) is inferred from names, not documentation, and is UNVERIFIED (decision 4).
- **AU counts** (decision 7): 323 stimuli carry 5 or 6 AUs against the article's 1 to 4; the file is used as released.
- **Peak frame** combines AUs at their own peaks, which may never have co-occurred; `frame_max` is the descriptive check.
- **Design B** was added as a fallback after A4's recovery history was known, before any A9 recovery number existed; its benefit under correlated face noise is reported, not assumed (decision 2).
- **Narrowing by check** (decision 8) was decided after recovery run 2 was seen, by applying the narrowing rule stated before it; the withdrawn claim (0.25x estimation accuracy under correlated face noise) is an accuracy claim, not a decision.
- **The ridge, ceiling and AU map** were computed on human data only, before any model data, and are fixed by the block; recovery used synthetic observers only.
- **A9 inherits A4's thresholds**, all marked proposed in A4, with no external source.

## Limits

As A4's, plus: one culture; 20 observers; 1,200 configurations, against which A4 failed recovery at the same count on Snoek (A4 run 2); AU16 on 58 stimuli; the "other" option includes blends in Liu's instructions, which may not be what None of the above means to a model.

## Cost

Upper bounds from A4's 3.0 s per basic cell, design B with `frame_max`: 5,992 unique images at 6 rotations, 35,952 cells, plus 50 image-free cells: about 30 h per model, about 60 GPU-hours for both (lead decision: the hours are free). Renders: 5,992 images at 3.1 s, about 5.2 core-hours on 40 CPU tasks; at A4's 0.2 MB per image, about 1.2 GB in 40 tars. A timing smoke after the freeze replaces these figures.

## Freeze checklist

Done before this text (2026-11-09):

- [x] Lead decisions recorded (above).
- [x] Design block written; its hash is pinned in `scripts/a9_design.py` and stated in the header; import refuses any other.
- [x] A4 code and outputs pinned in the block (`a4_pins`, each value equal to A4's frozen list); A9 inputs pinned (`input_pins`); manifest pinned (`manifest_sha256`).
- [x] `scripts/a9_analyse.py`, `scripts/a9_run.py` and both Slurm scripts written; tests added.
- [x] Human target regenerated on the final code (identical values; runtime field removed so the file is reproducible).
- [x] Recovery and the correlated-noise sensitivity rerun on the final code after decision 8 (run 3; `a9_recovery.json` carries the block hash and the code hashes it ran on, equal to the pins below; tables identical to run 2).
- [x] A9 code pinned under "Code hashes"; `check_code_pins(require=True)` now passes, and `a9_run` and `a9_analyse` refuse to run without it.

Remaining, in order:

1. Audit this draft against A4 (the coordinator or the user).
2. Commit A9's files and tag `a9-freeze`. The code pins are already in place, so any later edit to pinned code makes the runner and the analysis refuse.
3. On the cluster, after the tag: copy `results/a9/manifest.csv` (gitignored; or rebuild it with `a9_manifest.py manifest`, which needs the Liu files) and check its sha256 against the block; render with `sbatch --array=0-39 slurm_jobs/study3_a9_render.sh`.
4. Timing smoke, then the scored run per eligible model after RQ0 and A4's gates: `A9_FROZEN=1 MODEL=... N_SHARDS=12 sbatch --array=0-11 slurm_jobs/study3_a9.sh`.
5. Analyse once, after all shards finish: `python -m scripts.a9_analyse --gates gates.json`.

## Code hashes

Recovery run 3 ran on exactly this code (its `code_sha256` equals these lines). `tests/test_a9.py` is not pinned here, so tests can still be added before the tag; at this text it is `4fefbb29ba8e7b35b51af59eb70fff643ecd45730e8e87c9d814117cfe6f3f60` (42 passed), and its final sha256 is recorded with the tag.

Other A9 files (sha256), not read by the code-pin check:

- `slurm_jobs/study3_a9.sh`: `d3011382f49dc6b18b48e29e9b2ba8efe113a665b2772c86f21b5ae695214d45`
- `slurm_jobs/study3_a9_render.sh`: `c4cb4c2e0dcdedba420bd83a984f557926218af08fccf21d88e0cc17d4a2aa76`
- `research/reproduction/a9_recovery.json`: `01958263f95c6a2fb614d7e66093eb6ed61e11ede33415d14bc8fab8d076dd85`
- `research/reproduction/a9_recovery_run2.json`: `79229b70c4a7d21cfcf9bd957669a899c9b503082932e77cddf94d4aced56e84`
- `research/reproduction/a9_recovery_run1_draft.json`: `625675e81eeda31f54f67e7fe2d1cf9b1330089f1b2edbf050f4ba326f680afe`
- `research/reproduction/a9_chen_overlap.json`: `793fa184215f566ef4899d8d3222a55401c03622c93b8950775a6b092f374a44`
- `research/reproduction/a9_human_target.json`: `3f93ccbe2506ad9e01d79c8f594da6eb0bcd33790d410fda2d3945aeaa526ab0`
- `research/reproduction/a9_human_ceiling.json`: `a32e2dbe2e41b00b19594be1261d4aebe78dfb6170e3ab0837d3f612e62de669`
- `results/a9/manifest.csv` (gitignored): `4b2f04aee90a46723227429751bf7a068d36499bbd419279856113acb547250a`, also pinned in the block

```text
scripts/a9_design.py 1dc0aeacfd5888849eed91ba82e45c8bf039334778534717151e8c6486ad8c87
scripts/a9_analyse.py 00c3b5faa8d94bf36d73fc073d54f4cd18a3940486a978c502235294191911af
scripts/a9_run.py e9d14fb6158918d2f806da1d877301392487e5f51460304b7d1b08616617b144
scripts/a9_recovery.py 273e0fbf0f11b2224c8513744351945690e14600c2c81fa0e259d1d5977cc785
research/scripts/a9_human_target.py a47dc63a59079235b64b7cbee28855f8edb3804bb925d07388e356055a265d24
research/scripts/a9_manifest.py 9abc55e6c69d06730c120d858cba07c7c975222cbeaae94311a84730ee9b77a9
```
