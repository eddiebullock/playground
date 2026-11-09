# DRAFT amendment A9: replication of A4's confirmatory test on Liu et al. 2022, Experiment II

**Status: DRAFT, 2026-11-09. Not frozen. No model has seen a Liu render, and no render exists.** Nothing here changes A4 (FROZEN, tag `a4-freeze`), and A9 can never rescue, overturn, qualify or otherwise change any A4 verdict (see "Relation to A4").

**Recovery outcome, stated up front:** design A failed one of 17 checks; design B passed all 17, and is selected by the rule written before the run.

**Written:** 2026-11-09. Known at writing: A4's frozen text, code and outputs; the Liu human data, its human target and ceiling computed here (human data only); the recovery runs reported below (synthetic observers only). No A1, A4 or any other model result on avatars was read; none exists for Liu.
**Depends on:** exactly A4's gates per model: RQ0 through `rq0_status` at `single_face`, G2 to G4, and V1 to V4 as stated under "Validity gate".
**Models:** A4's Holm family, Gemma 4 and Qwen3-VL (`a4_design.MODELS`). Molmo2 is descriptive if run, as in A4.
**Code (new, nothing existing edited):** `scripts/a9_design.py` (constants, AU map, design hash; reads every shared value from `scripts/a4_design.py`), `research/scripts/a9_human_target.py` (human target, ceilings, AU-map and label checks, Chen check), `research/scripts/a9_manifest.py` (render manifest; renders through `a4_stimuli.render_tar`), `scripts/a9_recovery.py` (recovery through `a4_recovery` and `a4_analyse`), `tests/test_a9.py`. Outputs: `research/reproduction/a9_human_target.json`, `a9_human_ceiling.json`, `a9_recovery.json`, `a9_chen_overlap.json`. Notes: `research/notes_a9.md`.

## Question

The A4 question, on a second, independent human sample: when a face's action units are set at random, does a model's judgement change with each AU the way human judgements do? A9 asks it of Liu et al.'s Experiment II observers instead of Snoek et al.'s, with the same estimand, statistic, decision rule and smallest effect of interest.

**What A9 adds and what it does not.** Liu Experiment II has new observers (20) and new random stimuli (1,200), so it addresses "A4 rests on one human sample". It shares with Snoek the lab (Glasgow), the GFG generator and AU library, the response protocol and Western culture, so a shared artefact of that platform would replicate. Model-side, A9 uses the same renderer, identities, prompt and scoring as A4, so A9 does not address "A4 rests on one stimulus platform" either.

## Relation to A4

1. A4's verdicts are final as frozen. A9 never enters A4's Holm family, headline, decision table or any A4 sentence that states a verdict. A9 is reported in its own section, after A4's.
2. A9 has its own Holm family: the two eligible models on the one A9 dataset.
3. No pooling. A9 and A4 estimates are never averaged, meta-analysed or combined into one decision. A descriptive table may place them side by side, labelled as separate tests.
4. If A9 and A4 disagree for a model, both stand and the disagreement is reported as such; neither is reinterpreted.
5. If A4 is Inconclusive for a model, an A9 Aligned or Not aligned does not make A4 conclusive.

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

Design A: 2,600 rows, 2,592 unique images (manifest sha256 `dac1cd5a...` as built 2026-11-09). Design B: 6,200 rows, 5,992 unique images (`4b2f04ae...`). A9 renders only the design that recovery selects: design B (see Recovery).

**Rank.** Seeded hash of the stimulus index (`a9_design.SEED` 20261109), no use of responses. It picks the replicate set and the 50 image-free stimuli for V2.

**Identities.** A4's four Western faces (`female_a`, `female_b`, `male_a`, `male_b`). Gender: 600 and 600 by seeded hash of the index, matching the humans' totals but not their per-stimulus genders, which are unknown. Face within gender by seeded hash. Humans saw 1,200 different faces; models see four, so identity coverage is far narrower than for humans.

**Design A (primary) and design B (fallback), both fixed before recovery ran.** A: each configuration once, on one face, fitted with rendered gender, as A4. B: each configuration on all four faces; a configuration's response is the mean over its four faces' rotation-averaged probabilities, fitted with gender 0 (two faces of each gender). Rule: A is used if residual CEA passes recovery under A; otherwise B if it passes under B; otherwise A9 has no confirmatory analysis. The statistic is never switched to raw. B only helps where a model's response noise is independent across faces, which recovery assumes and a real model may not satisfy.

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

Synthetic observers answer Liu's real design through A4's estimator, by `scripts/a9_recovery.py` importing `a4_recovery` and `a4_analyse`: the same eight observers, built from Liu's human log-odds matrix, the same realism (stimulus logit noise SD 1 per option, lognormal AU gain error log SD 0.5), strengths .25, .5 and 1, 100 replicates per cell, and A4's pass rule (`a4_design.RECOVERY`) unchanged. A4's two withdrawn cells name A4 datasets, so **no A9 cell is withdrawn**, which makes A9's rule stricter than A4's as applied. Seventeen checks per statistic. Run 2026-11-09, `a9_recovery.json` (sha256 `625675e8...0afe`, design hash `c3f899db...`, draft code), 99 minutes on 6 CPU workers.

| Run | Design | Statistic | Checks passed | Result |
|---|---|---|---|---|
| 1 | A, 1,200 images, one face each | Residual | 16 of 17 | **FAIL**: pair-swapped CEA within .10 of truth in 68% of replicates at 0.25x (rule: at least 90%) |
| 1 | A | Raw | 16 of 17 | FAIL: same check, 86% |
| 1 | B, 4,800 images, four faces averaged | Residual | 17 of 17 | **PASS** |
| 1 | B | Raw | 17 of 17 | PASS |

By the rule fixed before the run, **design B is selected, with residual CEA confirmatory** (`choice` in the file). Design A's failure is of the same kind as A4's run 2 at 1,200 configurations: estimation accuracy on weak effects, not a false decision.

| Rate, residual CEA | A 0.25x | A 0.5x | A 1x | B 0.25x | B 0.5x | B 1x |
|---|---|---|---|---|---|---|
| Human-like Aligned | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Human-like CEA, mean (truth) | .735 (.869) | .820 (.870) | .871 (.889) | .811 (.859) | .855 (.875) | .887 (.892) |
| Human-like R_S, mean | .711 | .895 | .952 | .896 | .969 | .984 |
| Half-shared Aligned | .77 | .85 | .84 | .85 | .87 | .89 |
| Different-cues Not aligned | .93 | .97 | .96 | .96 | .97 | .99 |
| False Aligned, highest of four observers | .01 | .02 | .02 | .02 | .01 | .01 |
| Prior-only signal | .05 | .07 | .04 | .03 | .08 | .07 |
| Boundary Not aligned with truth at or above .30 | 0 | 0 | 0 | 0 | 0 | 0 |
| Pair-swapped within .10 of truth | .68 | .93 | .99 | .94 | 1.00 | 1.00 |

**Reading.** Residual CEA makes no false Aligned above .02 on Liu. It reads low on weak effects (A: .735 against a true .869 at 0.25x; B: .811 against .859). Pair-swapped observers' outputs correlate .88 to .95 with the human-weighted observer on natural-like stimuli (pair r .9, our construction), while their residual CEA is about .39, as in A4.

**Disclosures on this choice.** Design B was written into the code and this draft after A4's history was known (A4 failed at 1,200 configurations in its run 2) but before any A9 recovery number existed. B's gain comes from averaging four independently noisy images per configuration; a real model's idiosyncratic response may be largely configuration-specific and shared across faces, in which case B's real precision is closer to A's. Recovery used 100 bootstrap draws and 6 splits per replicate, against 2,000 and 20 in the real analysis. The ridge was 1.0, the selected human value.

## Disclosures

- **Same platform.** Lab, generator, AU library and protocol are shared with Snoek; A9 is a second sample, not a second method.
- **Human reference** is cross-dataset (Liu against Snoek), not cross-group, and mixes stimulus, face and coding differences with observer differences.
- **Gender** is absent from the human fit and assigned by hash on the model side.
- **Four faces** against 1,200 seen by humans.
- **Suffix mapping** (Open, i) is inferred from names, not documentation.
- **Peak frame** combines AUs at their own peaks, which may never have co-occurred.
- **The ridge, ceiling and AU map** were computed on human data only, before any model data, and are fixed by this draft; recovery used synthetic observers only.
- **A9 inherits A4's thresholds**, all marked proposed in A4, with no external source.

## Limits

As A4's, plus: one culture; 20 observers; 1,200 configurations, against which A4 failed recovery at the same count on Snoek (A4 run 2); AU16 on 58 stimuli; the "other" option includes blends in Liu's instructions, which may not be what None of the above means to a model.

## Cost

Upper bounds from A4's 3.0 s per basic cell. Design A: main and replicate 8,400 cells (7.0 h per model), image-free 50, `frame_max` 7,200 cells (6.0 h per model): about 13 h per model, 26 GPU-hours for both. Design B adds 21,600 cells, about 18 h per model. Renders: 2,592 images (A) or 5,992 (B) at 3.1 s, 2.2 or 5.2 core-hours; at A4's 0.2 MB per image, about 0.5 or 1.2 GB in tars.

## Freeze checklist

1. Re-read this draft against A4 line by line; every shared value must come from `a4_design` (tests check the main ones).
2. Confirm the response-code order once more against a second Liu source if one can be found (for example the v3 `emotions_name.mat` or the article's figure order), and record it; do not change it from data.
3. Decide the open questions below and record each answer here as a dated decision.
4. Write `scripts/a9_analyse.py` (A4's `analyse_dataset` cannot be called as is: it looks up `a4_design.TASK_OF[dataset]`), importing `a4_analyse` functions unchanged; and a runner path or wrapper that sends `a9` manifest rows through `a4_run`'s scoring with A9's design hash; plus a Slurm script. Tests for each.
5. Pin the manifest of the selected design: rebuild it, record its sha256 here and in `a9_design.MANIFEST_SHA256` (to add).
6. Rerun `a9_human_target.py target` and `a9_recovery.py` on the pinned code, so that `a9_recovery.json` carries the frozen `design_sha256`; confirm the tables match this draft.
7. Record sha256 of every A9 file and output in this draft, as A4 does.
8. Confirm no model has seen a Liu render, then tag `a9-freeze`.
9. Only then render, then run models after RQ0 and the A4 gates.

## Open questions

1. **Design B cost.** It quadruples main renders and cells (about 31 h per model with `frame_max`, against 13 h for A). Accept, or drop `frame_max` to save about 6 h per model.
2. **Is B's gain real?** Recovery assumes independent noise across faces. A pre-freeze check with no human target would be informative but needs model outputs on Liu renders, which this draft forbids before freeze. Accept the assumption as a disclosed limit, or add a rule: if V3's configuration-carries effect is very strong, report that B's averaging may overstate precision (descriptive only).
3. **Response-code order** rests on one declared variable in Liu's code, corroborated by Chen's deposit and by prototype AUs. A second Liu source (for example `emotions_name.mat` in the v3 deposit, about whose content nothing is known) was not fetched.
4. **"Open" and "i" suffixes**: their meaning in GFG is unverified; the renders ignore any extra lip parting, as A4 already does for Snoek.
5. **Gender covariate asymmetry**: humans fitted without gender, model with rendered gender (A4's code unchanged); under design B the model fit uses gender 0 for all. Confirm this is acceptable, or fit the model without gender too (equivalent in expectation under randomisation).
6. **Human reference**: Liu against Snoek basic WC (residual .690) is the proposed descriptive yardstick. Alternatives: none, or the A4 WC to EA reference (.780).
7. **AU count discrepancy** (1 to 6 AUs in the file against 1 to 4 in the article): ask the authors, or proceed as released.
8. **Chen 2024**: same data as Snoek basic WC, so it adds no sample; its EA half is presumably Snoek basic EA (UNVERIFIED). Its only possible use is a coding check.
