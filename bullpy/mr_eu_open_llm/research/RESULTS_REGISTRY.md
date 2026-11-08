# RESULTS_REGISTRY

**2026-11-04: image-dependent values below came from a faulty decoder on about 20% of items; see amendment A1.1. Do not cite.**

Every paper number must trace to a script, a commit, and an output file. No estimates.

Status key: **verified** = recomputed or read from the named file this session; **read-only** = copied from a saved analysis JSON without rerunning the analysis that produced it; **missing** = brief/docs mention it, no file.

Phase 0 snapshot commit: `2e2af355a4c4c776d2b9675aa4a5b55c56fc44c8` (`p0-locate-reproduce`).

| claim | value | status | script | output file | notes |
|---|---|---|---|---|---|
| Qwen 6AFC accuracy | 92/243 = 0.378601 (37.9%) | verified | eval on CSD3 job 33520827; check `research/scripts/reproduce_headline_numbers.py` | `results/baseline/eu_emotions/qwen3vl/eval_v2_eu_emotions_qwen3vl_video_only_seed42.json` | Wilson [0.320, 0.441] |
| Qwen p vs chance (greater) | 2.252914e-15 | verified | `scipy.stats.binomtest(92, 243, 1/6, alternative='greater')` | same eval JSON `p_binom_gt_chance` | matches stored |
| Gemma 6AFC accuracy | 76/243 = 0.312757 (31.3%) | verified | CSD3 job 32447858; reproduce script | `results/baseline/eu_emotions/gemma4/eval_v2_eu_emotions_gemma4_video_only_seed42.json` | Wilson [0.258, 0.374] |
| Gemma p vs chance | 1.541805e-08 | verified | binomtest | same | above chance |
| Molmo 6AFC accuracy | 48/243 = 0.197531 (19.8%) | verified | CSD3 job 32447860; reproduce script | `results/baseline/eu_emotions/molmo2/eval_v2_eu_emotions_molmo2_video_only_seed42.json` | Wilson [0.152, 0.252] |
| Molmo p vs chance | 0.1155465 | verified | binomtest | same | **not above chance** |
| Qwen ρ(H_model, H_human) | −0.001325, p = 0.984, boot CI [−0.121, 0.120] | verified | `paper_v2.behaviour` | `research/reproduction/behaviour_summary.json` | CI includes 0 (H2) |
| Gemma ρ(H_model, H_human) | +0.117246, p = 0.068, boot CI [−0.013, 0.235] | verified | same | same | CI includes 0 (H2) |
| Molmo ρ(H_model, H_human) | −0.034423, p = 0.593, boot CI [−0.154, 0.097] | verified | same | same | CI includes 0 (H2) |
| Qwen acc high/mid/low consensus | 0.543 / 0.383 / 0.210; p = 1.20e-05 | verified | `scripts/human_calibration.py`; `scripts/reproduce.py --quick` | `results/stats/human_calibration.json` | n = 81 tercile; output commit `2e2af35`; rerun 2026-11-08 from the committed eval JSONs: accuracies equal, p within 2e-21 |
| Gemma acc high/mid/low | 0.506 / 0.309 / 0.123; p = 1.57e-07 | verified | same | same | rerun 2026-11-08, equal |
| Molmo acc high/mid/low | 0.346 / 0.148 / 0.099; p = 1.57e-04 | verified | same | same | not < 1e-4; rerun 2026-11-08, equal |
| Probe confusability ρ range | 0.467–0.611 | read-only | `scripts/confusability_probe_rsa.py` | `results/mech/{qwen3vl,gemma4,molmo2}_confusability_probe_rsa.json` | pilot LOO; npy now local |
| Probe human-entropy ρ range | 0.441–0.532 | read-only | same | same | confirmatory rerun is `paper_v2.probe` |
| Raw RSA ρ range | −0.079 to +0.027 | read-only | same | same | Qwen L4 p_perm = 0.045 uncorrected, ρ negative |
| Difficulty-matched RSA mean ρ | Qwen 0.031/0.050/0.104; Gemma 0.086/0.087/0.004; Molmo 0.155/0.187/0.185 | verified | `scripts/confusability_probe_rsa.py` | same JSON `rsa_difficulty_matched.rho_mean` | exploratory |
| Bored–Unfriendly matched RSA (early layer) | Qwen L4 0.269; Gemma L5 0.258; Molmo L4 0.473 | verified | same | same JSON per_pair | exploratory until replicated |
| Confusability vs entropy axis cosine | 0.950085 | verified | cosine of unit npy | `results/mech/axis_{confusability,entropy}_qwen3vl_layer4.npy` and `qwen3vl_eu_causal_axes_layer4.json` | |
| Pair own-vs-rest | 0.528, 0.584, 0.796 | read-only | `scripts/causal_eu_confusion_axes.py` | `results/mech/qwen3vl_eu_causal_axes_layer4.json` | |
| Pair vs entropy cosine (signed) | +0.477, +0.652, −0.527 | verified | npy cosine | `results/mech/axis_pair_*_qwen3vl_layer4.npy` | brief 0.48–0.65 is \|cos\| |
| Steering n, site | 36 trials, L4 last token, 5 samples | read-only | `scripts/steer_eu_confusion_axes.py` | `results/mech/steer_summary_qwen3vl_layer4.json` | |
| Ablation Δacc / Δconfusion / 5/180 | 0 / 0 / 5 flips of 180 | verified | `scripts/ablate_eu_confusion_axes.py` | `results/mech/pilot_n36/ablate_summary_qwen3vl_layer4.csv` and `ablate_trials_*.csv` | n=36; entropy 0, bored 1, disappointed 1, interested 2, random 1 |
| Human norms modality | Face | verified | `scripts/human_entropy.py` | `data/eu_emotions_human_entropy.json` `"modality": "Face"` | |
| n actors | 17 letters from filenames | verified | regex in reproduce script | same JSON keys | grouped CV possible |

| Target corr(1−p_target, H_human) | 0.954 | verified | `paper_v2.entanglement` | `research/reproduction/phase5_entanglement.json` | n = 243 |
| Probe cosine entropy vs 1−p (Qwen L4) | obs 0.808 vs null 0.955 | verified | same | same | n_sim = 100; obs below Gaussian null |
| Partial RSA vs entropy RDM (Qwen L4) | raw −0.079, partial −0.061 | verified | same | same | still near zero |

| Phase 2 Qwen L26 grouped T1 ρ | 0.440, boot CI [0.317, 0.540], p_Holm = 0.003 | verified | `paper_v2.probe` | `research/reproduction/phase2_gate.json` | exploratory 3-layer cache; passes gate |
| Phase 2 Gemma L15 grouped T1 ρ | 0.328, boot CI [0.200, 0.442], p_Holm = 0.003 | verified | same | same | exploratory; passes gate |
| Phase 2 stimulus baseline T1 ρ | 0.138 | verified | same | same | actor + duration/luminance/motion |
| Phase 2 would-be branch (3-layer) | A (not locked) | verified | same | same | superseded by all-layer lock |
| H2 output entropy CIs include 0 | Qwen [−0.121, 0.120]; Gemma [−0.013, 0.235]; Molmo [−0.154, 0.097] | verified | `paper_v2.behaviour` | `research/reproduction/behaviour_summary.json` | confirmatory H2 |
| Official branch | **B** | verified | `paper_v2.probe` all-layer per model; merge by `research/scripts/merge_phase2_all_layers.py` | `research/reproduction/phase2_gate_all_layers.json` | locked 2026-11-02; Holm floor. Merged by hand in `f6dd3cf` (no script then); the merge script, written 2026-11-08, rebuilds the file byte-identical from the three committed `phase2_gate_all_layers_{model}.json` |
| All-layer Qwen peak grouped T1 ρ | L34 0.471, p_perm = 0.001, p_Holm = 0.036 | verified | jobs 39709580 / 39712190 | same | 36 layers; 20 fail only Holm |
| All-layer Gemma peak grouped T1 ρ | L2 0.492, p_perm = 0.001, p_Holm = 0.042 | verified | jobs 39709581 / 39712191 | same | 42 layers |
| All-layer Molmo peak grouped T1 ρ | L22 0.518, p_perm = 0.001, p_Holm = 0.032 | verified | jobs 39709582 / 39712192 | same | at chance; completeness |
| Extract completeness | Qwen 36/36, Gemma 42/42, Molmo 32/32 npy; 0 errors | verified | extract_meta on CSD3 | HPC `baseline_*_6afc_all_layers_f4` | last_token, max_frames=4 |
| Qwen layers failing only Holm | 20 / 36 | verified | `paper_v2.probe` gate flags; CSV by `research/scripts/merge_phase2_all_layers.py` | `research/reproduction/phase2_all_layers_summary.csv` `holm_only_fail` | other four checks pass. CSV made by hand in `f6dd3cf`; the merge script rebuilds it byte-identical |
| Phase 2 grouped T1 vs layer figure | `research/figures/phase2_grouped_t1_rho.png` | verified | `research/scripts/plot_phase2_all_layers.py` | same PNG | filled markers = Holm-only fail |
| LOO vs grouped T1 figure | `research/figures/phase2_loo_vs_grouped.png` | verified | `research/scripts/plot_phase2_loo_vs_grouped.py` | same PNG | LOO > grouped at every layer |
| Pilot n=36 ablation flips recomputed | 5 / 180 | verified | `paper_v2.causal` | `research/reproduction/phase3_pilot_ablation.json` | entropy 0, bored 1, disappointed 1, interested 2, random 1 |

Dump of the Phase 0 recompute: `research/reproduction/phase0_numbers.json`.
Phase 2 exploratory dump: `research/reproduction/phase2_gate.json`.
Official Phase 2 dump: `research/reproduction/phase2_gate_all_layers.json`.
Phase 5 dump: `research/reproduction/phase5_entanglement.json`.
Stimulus features (HPC job 39709870): `research/reproduction/stimulus_features.json` (243/243 ok).

| Target position in 6AFC prompt | option 1 on 243/243 | verified | `research/scripts/audit_prompt_confound.py` | `research/reproduction/prompt_confound_audit.json` | accuracy = P(chose option 1) |
| Option sets per target label | 21 sets, 21 labels, 1 each | verified | same | same | prompt identifies category |
| Category-only grouped T1 rho | 0.503 | verified | same | same | no activations used; equals peak probes |
| Category decode, last token, grouped | 0.959 to 1.000 (9 sites) | verified | same | same | exploratory; text supplied |
| Within-category T1 probe | r = -0.063 to 0.177 | verified | same | same | only Qwen L13 has p < 0.05, uncorrected |
| Within-category p_target vs human | Qwen 0.162, Gemma 0.097, Molmo 0.209 | verified | same | same | exploratory, uncorrected |
| Category-only LOO, human entropy | 0.501 | verified | same | same | pilot probe range 0.441 to 0.532 |
| Category-only LOO, model 1-p_target | Qwen 0.645, Gemma 0.703, Molmo 0.526 | verified | same | same | pilot probe range 0.467 to 0.611 |
| Named pair option access | Bored offered on Unfriendly items, not the reverse; Kind and Interested likewise one way; Disappointed and Worried both | verified | same | same | humans could only confuse what was offered |

Accuracy and probe rows above this table are confounded by the prompt; see DECISIONS 2026-11-04.


## Family recovery and second AU detector (2026-11-04)

Criteria locked 15:12:57 UTC before any measured-cue recovery (`same_clip_family_criteria.json`, hash 4b0a8c3c). No model data involved.

- Families recover planted weights of 0.5 and above: r .80 to .85, sign .92 to .96. Sparse 0.25 and dense 0.3 fail.
- Families plus singles fail recovery in every cue condition. Single family cells: sign does not recover at human effect size.
- 8 of 20 conditions pass for families, 3 of 20 for families plus singles.
- Pattern score separates human-like from prior-only at 0.5x, 1x and 2x (AUC .999 to 1.00). The power of 1.0 is a best case; half-shared power is .545, .88, .965.
- Family-wise calibration fails for actor-prior observers (.115 against limit .081). Human cells stay provisional.
- Human, families, all frames: 5 of 88 cells flagged by two detectors, 2 family-wise, held-out gain 0.61%, split-half r .31.
- ME-GraphAU against LibreFace, within category and actor: r .33 to .57. Not independent data: both BP4D. AU20 uncovered, rule 8 unmet.

## Family calibration, second attempt (2026-11-04)

Post-hoc second attempt, locked in `same_clip_family_a2_criteria.json` before its runs. Failed. First attempt outputs are unchanged and remain the null of record for the primary spec.

- Within-actor shuffle: 7 of 20 conditions pass, no null condition passes. Actor-prior family-wise .17 (limit .081).
- Pattern score, families: power .995 to 1.0, size .035 to .05 on both nulls. Half-shared power .63, .925, .935.
- Different-cues observer exceeds .081 at 1x and 2x (.105, .115): a model using other cues could look aligned too often.
- Model-frames route, human lambda 10: prior-only family-wise .085 fails; actor-prior fails; sparse 0.5 fails narrowly (r .752, sign .898).
- Only Surprised with mouth open is family-wise under both nulls on all frames. All human cells are exploratory.

## Pattern score, model-frames route (2026-11-04)

Attempt 1 null, frozen lambda 10, families: size passes at 0.5x, 1x and 2x; power .775, 1.0, 1.0 against .80. Different-cues size .03 to .055. Real effect sits near 0.75x by reliability, 1x by gain. The strength rule later required both 0.75x and 1x: 0.75x passes (power .995, size .035 and .06, different-cues .055, half-shared .60), so the score is confirmatory per `DECISIONS.md`.


## A3 face input validity, `single_face` (2026-11-05)

Frozen analysis, run once after all three models finished (job 40023523). Scripts matched the hashes in A3. 243 of 243 items on every model, no failures. File: `research/reproduction/face_input_validity_summary.json`.

| Model | Accuracy real | Accuracy mismatch | delta (actor bootstrap 95% CI) | Holm p | Spearman with human p_target | Decision |
|---|---|---|---|---|---|---|
| Gemma 4 | .371 | .156 | .216 (.139 to .292) | .00014 | .376 | Pass |
| Qwen3-VL | .282 | .038 | .244 (.212 to .276) | .00009 | .392 | Pass |
| Molmo2 | .072 | .010 | .063 (.040 to .087) | .00010 | .342 | Pass |

The A3 rule tests vision dependence only, not competence. A pass cannot be read as face use until A1 is in (A1/A3 table in `PREREG_A3_DRAFT.md`).

Descriptive, outside the frozen summary and computed from the cells as the argmax of joint logprob; share answering "None of the above" (real, mismatch): Gemma .206, .305; Qwen .534, .688; Molmo .863, .886. Molmo is below 1/6 and its accuracy varies with target position (.140 at position 0, .025 at position 1), so its delta rests on a small minority of cells. The accuracy-against-1/6 p is descriptive by A3.

Trace (2026-11-08): **the "None of the above" shares are now traced.** Script `research/scripts/a3_nota_shares.py` (standard library only), run on 2026-11-08 on the cluster login node from `~/rds/hpc-work/study3` over the frozen cells `results/face_input_validity/cells_{gemma4,qwen3vl,molmo2}.jsonl` (2,430 rows each, all ok, no partial lines; sha256 `05eec2d4...f685`, `a3f302b3...c7e6`, `cd80b5c5...a6be`, full hashes in the output). Output: `research/reproduction/a3_nota_shares.json`, `models.<model>.<condition>.{count,n,share}`. The two numbers are the share of cells, over 243 items by 5 rotations (n = 1,215 per condition), whose unique argmax of joint logprob is "None of the above", in the real condition (the item's own face) and the mismatch condition (the partner item's face). Counts, real then mismatch: Gemma 250 and 370 (.20576, .30453); Qwen 649 and 836 (.53416, .68807); Molmo 1,049 and 1,077 (.86337, .88642). **Status: matched after rounding** to three decimals on all six values (.206, .305; .534, .688; .863, .886); none differs. Ties at the maximum (Gemma 13 and 19, Qwen 10 and 11, Molmo 0) never involve "None of the above": the first-index argmax gives the same counts. They are not cited in the paper text (`research/paper/v2/`); `PREREG_A4_DRAFT.md` repeats .534 and .863, which match. Accuracy by target position **is** traced: `face_input_validity_summary.json` (`7b8b1d0`), `<model>.secondary.accuracy_by_target_position`, made by `scripts/face_input_validity.py --summarise`; Molmo .13992 at position 0 and .02469 at position 1 (Gemma .342 to .387 across the five positions; Qwen .325 at position 0, .259 to .280 elsewhere).


## Registry gaps closed (2026-11-08)

Items the reproduction audit (`research/reproduction/REPRODUCE.md`, Registry coverage) found with no row. Each value was read from the named file on 2026-11-08 and checked against `research/paper/v2/methods.md`, `supplementary_note1.md`, `outline.md` and `introduction.md` where the paper cites it. "Commit" is the last commit that changed the output (`git log -1 -- <file>`). "Quick" means `scripts/reproduce.py --quick` reproduced the file in the clean-checkout test of 2026-11-08 (REPRODUCE.md); "cluster" means it cannot be rerun from public inputs. The sha256 of `human_au_effects.json`, `a4_human_ceiling.json`, `a4_v1_pilot.csv` and `a4_recovery.json` were checked on 2026-11-08 and equal the frozen pins in `PREREG_A4_DRAFT.md` line 9.

### A4 human target, `human_au_effects.json` (commit `c5bdb3a`; quick, CSV exact, JSON equal)

| claim | value | status | script | output file | notes |
|---|---|---|---|---|---|
| Snoek subjects per dataset | 60, 60, 20, 20 (basic WC, basic EA, conversational WC, conversational EA) | verified | `research/scripts/human_au_effects.py` | `research/reproduction/human_au_effects.json` `datasets.*.structure.n_subjects` | Methods, Human data |
| Snoek trials, "other" kept | 140,910; 139,728; 47,300; 47,460 | verified | same | same, `structure.n_trials` | Methods |
| Face codes in the public files | 967, 8, 4, 4 | verified | same | same, `structure.n_distinct_face_ids` | Methods; basic WC subjects 01-15 saw blended identities (`data_caveats`) |
| AU columns | 33 | verified | same | same, `structure.n_au_columns` | Methods |
| Unique AU configurations | 4,659; 2,323; 2,318; 2,325 | verified | same | same, `structure.n_unique_au_configurations` | Methods cites these from the A4 manifest; equal here |
| Selected ridge | 0.1, 0.1, 1.0, 1.0 | verified | same | same, `ridge_lambda_selected` | Methods ceiling table; Supp. Note 1 section 1 |
| Pooled split-half, subjects and stimuli, log odds | .960, .959, .908, .882 | verified | same | same, `reliability.beta_family_25_subjects_and_stimuli_split_pooled_fit.split_half_r` | outline.md audit row 3; not used for decisions |
| Leave-one-subject-out mean r | log odds .762, .781, .752, .718 (mean .753); do effects .740, .729, .727, .719 | verified | same | same, `reliability.{beta,do_effect}_family_25.leave_one_subject_out_mean_r` | outline.md audit row 3; not used |

### A4 human ceiling and energy shares, `a4_human_ceiling.json` (commit `4b5271c`; quick, exact)

| claim | value | status | script | output file | notes |
|---|---|---|---|---|---|
| Ceiling, raw (sqrt R_H) | .990, .989, .978, .974 | verified | `python -m scripts.a4_analyse --write_ceiling` | `research/reproduction/a4_human_ceiling.json` `datasets.*.ceiling` | 200 splits, seed 20261107; Methods table, Supp. Note 1 section 4 |
| Ceiling, residual | .970, .983, .974, .960 | verified | same | same, `ceiling_residual` | same |
| WC to EA reference, raw | basic .877, conversational .767 | verified | same | same, `wc_to_ea_cfa.basic`, `.conversational` | Methods table |
| WC to EA reference, residual | basic .780, conversational .732 | verified | same | same, `wc_to_ea_cfa.*_residual` | Methods table; Supp. Note 1 sections 4 and 5; outline "human-level" bar |
| Common-profile share | .2764, .2634, .3242, .3535 | verified | same | same, `human_energy.*.common_profile_share` | Supp. Note 1 S2 reading; Methods ".26 to .35". sqrt: .526, .513, .569, .595 (arithmetic in Supp. Note 1, checked) |
| AU22 and AU27 share of human squared effects | .109, .119, .183, .103 | verified | same | same, `human_energy.*.AU22_AU27_share` | Methods, Disclosures and limits |
| Detector-confirmed share of residual energy | .14, .13, .09, .20 (.1390, .1334, .0894, .1958) | verified | same | same, `human_energy.*.confirmed_share_of_residual` | Methods, Family flag |

### A4 recovery (quick: no; full mode)

| claim | value | status | script | output file | notes |
|---|---|---|---|---|---|
| Run 4 (frozen) | residual 66 of 66, raw 62 of 66; `confirmatory_statistic` residual; design `12c454da` | verified | `python -m scripts.a4_recovery` | `research/reproduction/a4_recovery.json` `checks`, `pass` | commit `00847f4`; tables and checks identical to run 3 (compared 2026-11-08) |
| Run 3 | raw 62 of 66 (false Aligned conversational WC 0.25, 0.5, 1 and conversational EA 0.25; highest .40); residual 66 of 66 | verified | same, at `568b6f5` | `a4_recovery_run3.json` | commit `568b6f5` |
| Run 2 | raw 63 of 66, false Aligned conversational WC up to .33; residual 62 of 66, pair-swapped within-limit share .76, .88, .80, .82 (basic WC 0.25 and 0.5, basic EA 0.25, conversational EA 0.5) | verified | same, at `6176b61` | `a4_recovery_second_run.json` | commit `6176b61`; Methods "76 to 88%" |
| Run 1 | 27 of 28; pair-swapped within-limit share .66 at conversational EA 0.25 | verified | same, at `17dc0cf` | `a4_recovery_first_run.json` | commit `17dc0cf` |
| Run 3 rates, raw / residual | human-like Aligned 1.0 / 1.0; half-shared Aligned at 1x lowest .87 / .79; different-cues Not aligned at 1x lowest .95 / .95; false Aligned highest .40 / .03; prior-only signal highest .08 / .08 | verified | same | `a4_recovery_run3.json` `tables` | false Aligned is the max over different-cues, prior-only, generic and salience-profile (`FALSE_ALIGNED`) |
| Run 3, Not aligned with true CFA at or above .30, highest | raw .01, residual .03 (boundary observer: .0132, .0263) | verified | same | same, `boundary.not_aligned_when_truth_at_bound` | the value is for the boundary observer, which the check uses. Over all observers raw reaches .10 (salience-profile, conversational EA 0.5, 10 replicates at the bound); see mismatch list |
| Exempt cells, residual pair-swapped within-limit share | conversational WC .85, conversational EA .62 at 0.25x; 66 of 68 if counted | verified | same | `a4_recovery.json` `tables.residual` | Methods, Exempt cells |
| Residual CFA on weak effects | human-like at conversational EA 0.25x reads .738 against truth .866 | verified | same | same, `cfa_mean`, `truth_mean` | Methods ".74 against .87"; Supp. Note 1 section 5 |
| Pair-swapped, run 2 | natural-output r .834 to .945; raw CFA .418 to .562 | verified | same | `a4_recovery_second_run.json` `tables.raw.*.pair_swapped` | Methods ".83 to .95", ".42 to .56" |
| Salience-profile true CFA, run 3 | residual -.129 to .059; raw .043 to .383 | verified | same | `a4_recovery_run3.json` `truth_mean` | Supp. Note 1 notes on S1 |

### A4 V1 pilot, `a4_v1_pilot.csv` (commit `6176b61`; cluster, job 40260039)

There is no summary file. The headline is the frozen rule `detector_gate` in `scripts/a4_analyse.py` applied to the CSV; the breakdown is printed by `research/scripts/registry_derived_counts.py` (read-only, written 2026-11-08).

| claim | value | status | script | output file | notes |
|---|---|---|---|---|---|
| Faces found | 432 of 432 | verified | `research/scripts/a4_detector.py`; count by `registry_derived_counts.py` | `research/reproduction/a4_v1_pilot.csv` `face_found` | Methods |
| V1 share rising | 113 of 144 = .7847 (.785), passes .70 | verified | `scripts/a4_analyse.py` `detector_gate` | same | Methods, Methods Amendment table |
| Intensity and presence pairs | 96 and 48 | verified | `registry_derived_counts.py` | same | |
| Intensity pairs with reading at amplitude 1 minus .25 of at least .5 | 22, all in AU01, 05, 25 and 26; 20 of the 22 also rise monotonically | verified | same | same | Methods "22 rise by at least .5"; see mismatch list |
| AU12 at amplitude 1 | .014 to .031 | verified | same | same | Methods "about .02" |
| Rising presence pairs | 43, of which 35 have a presence score below 0 at amplitude 1 | verified | same | same | Methods |

### Cue spectra, `cfa_identifiability.json` (commit `3fddfc5`; quick, exact)

| claim | value | status | script | output file | notes |
|---|---|---|---|---|---|
| EU-Emotion LibreFace | 243 clips, 15 cues, lambda_min .05435, condition 76.1, 1/lambda_min 18.4 | verified | `research/scripts/cfa_identifiability.py` | `research/reproduction/cfa_identifiability.json` `eu_emotion_natural_clips.libreface` | Methods table; Supp. Note 1 ".054" |
| EU-Emotion OpenFace 3.0 | 243, 10, .12505, 22.4, 8.0 | verified | same | same, `.openface3` | Methods table |
| Snoek as coded, 33 cues | basic WC 4,659 rows, .17776, 13.72, 5.63; basic EA 2,322, .19921, 11.3, 5.02; conversational WC 2,314, .20902, 11.02, 4.78; conversational EA 2,322, .19921, 11.3, 5.02 | verified | same | same, `snoek_randomised_design.*.as_coded` | Methods table, rounded |
| Snoek left and right merged, 25 cues | lambda_min .43936 to .46878; condition 3.36 to 3.51; 1/lambda_min 2.13 to 2.28 | verified | same | same, `.left_right_merged` | Methods ".44 to .47, 3.4 to 3.5, 2.1 to 2.3"; Supp. Note 1 |

### A2 recovery, `recovery_stimulus_interventions.json` (commit `b3af2cc`; quick, exact)

| claim | value | status | script | output file | notes |
|---|---|---|---|---|---|
| A2 recovery | 5 observers, 100 replicates, all pass; expected contrasts recovered at 1.0; share of replicates with an unexpected contrast: prior-only .06, frame-bag .07, mouth-only .08, eyes-only .08, human-like 0 | verified | `python -m scripts.recovery_stimulus_interventions --replicates 100` | `research/reproduction/recovery_stimulus_interventions.json` | 16 contrasts; Methods, A2 Recovery |

### Vision gates G2 to G4 (cluster)

| claim | value | status | script | output file | notes |
|---|---|---|---|---|---|
| Gemma 4 | G2, G3, G4 pass at bfloat16 and float32; G4 at 100%: RMS 4.87 nats, flip rate .675 | verified | `python -m scripts.vision_gates` | `research/reproduction/vision_gates_gemma4.json` `verdict`, `stages.bfloat16.g4.verdict` | commit `61123d4`; 40 clips, 17 actors |
| Qwen3-VL | G2, G3, G4 pass at bfloat16 only; G4 at 100%: RMS 9.30 nats, flip rate .55 | verified | same | `vision_gates_qwen3vl.json` | commit `e131dae`; float32 not in `verdict` (out of memory, Methods Disclosure 7) |
| Molmo2 | G2 fails at bfloat16 (worst relative L2 .0406 against .01), passes at float32; G3 passes; G4 fails (RMS 2.28 nats but flip rate 0 against .25) | verified | same | `vision_gates_molmo2.json` | commit `61123d4`; Methods "failed the option-independence and token-swap gates" |
| Eligibility | Gemma 4 and Qwen3-VL eligible for A2 and A4, Molmo2 not | verified | rule in `PREREG_A2_DRAFT.md` | the three files above | STATE item 5; Methods |

### Input delivery (cluster)

| claim | value | status | script | output file | notes |
|---|---|---|---|---|---|
| Gemma 4 tokens across the face, median | single_face 9.01 (pass), legacy grid 1.08 (fail) | verified | `python -m scripts.run_stimulus_interventions --step delivery` | `research/reproduction/input_delivery_gemma4.json` `verdicts` | commit `20077be`; Methods "about 1 ... about 9" |
| Qwen3-VL | single_face 13.52, legacy 11.54; all four routes pass | verified | same | `input_delivery_qwen3vl.json` | commit `20077be` |
| Molmo2 | single_face 12.96 (pass), video_face 5.07 (fail), legacy 2.01 (fail) | verified | same | `input_delivery_molmo2.json` | commit `e4c05de` |
| Clips losing frames to the 8-cell grid | 68 of 243 | verified | `research/scripts/registry_derived_counts.py` (rule of `scripts/frame_sampling.py`) | `research/reproduction/stimulus_features.json` `duration_s` (commit `d37caea`) | Methods, Delivery audit; DECISIONS 2026-11-04. Three clips have `duration_s` 9.00, at the boundary; stored to two decimals |

### Blind readout choice (cluster)

| claim | value | status | script | output file | notes |
|---|---|---|---|---|---|
| Chosen readout | "In one plain word, with no formatting or punctuation, what is this person feeling?", prefill "They feel" | verified | `python -m scripts.readout_ladder --step ladder`; `choose_shared` rerun 2026-11-08 on the three files returns it | `research/reproduction/readout_ladder_{gemma4,qwen3vl,molmo2}.json` `rung_1` | commits `20077be`, `a343843`, `d22026b`; Methods |
| Rule values | three candidates pass on every model; lowest median lexicon mass .640, .680, .660; chosen .680 (Gemma .803, Qwen .977, Molmo .680) | verified | same | same, `median_lexicon_mass`, `pass` | |
| Calibration clips | 24, 24 usable, labels or votes read: false | verified | same | same, `n_calibration_clips`, `n_usable`, `labels_or_votes_read`; ids in `crema_calibration_clips.csv` (quick, exact) | Methods "24 CREMA-D clips" |
| Lexicon | 106 words in 21 families | verified | `LEXICON` in `scripts/run_stimulus_interventions.py` | code constant, hashed in A2 | Methods |

### Human response predictor, `human_response_predictor.json` (commit `58bada3`; quick, exact)

| claim | value | status | script | output file | notes |
|---|---|---|---|---|---|
| Held-out log loss (base rate), r by stimulus, within group | basic WC 1.618 (1.935), .691; basic EA 1.567 (1.906), .815; conversational WC 1.494 (1.613), .593; conversational EA 1.491 (1.591), .636 | verified | `python -m scripts.human_response_predictor` | `research/reproduction/human_response_predictor.json` `within_group_unseen_subjects_and_stimuli.mean_over_splits` | not cited in the paper; PREREG_A2 appendix |
| Across group | basic WC 1.658 (1.928), .662; basic EA 1.624 (1.903), .768; conversational WC 1.479 (1.590), .630; conversational EA 1.481 (1.555), .647 | verified | same | same, `across_group` | a model of humans, never human data |

### CREMA-D reliability and selection (commit `e969c05`; quick, exact)

| claim | value | status | script | output file | notes |
|---|---|---|---|---|---|
| Selection | 819 of 7,442 clips | verified | `research/scripts/crema_select.py` | `research/reproduction/crema_d_selection.csv` (819 rows); `crema_d_reliability.json` `n_selected`, `n_clips` | not cited in the paper |
| Votes and raters | 71,654 face votes, 2,442 raters | verified | `research/scripts/crema_reliability.py` | `research/reproduction/crema_d_reliability.json` | |
| Krippendorff alpha, nominal | all clips .4582, selected .4598 | verified | same | same | |
| Split-half, selected clips, Spearman-Brown | p_target pooled .828; confusion off-diagonal .994 | verified | same | same, `selected_clips` | 2,000 splits |

### A6 human target, `a6_human_target.json` (commit `b056a7e`; quick, exact)

| claim | value | status | script | output file | notes |
|---|---|---|---|---|---|
| Sample | 819 clips, 91 actors | verified | `research/scripts/a6_human_target.py` | `research/reproduction/a6_human_target.json` | A6 is an unfrozen draft; Methods gives it at outline level only, with no number |
| Target pair | Anger and Disgust; pair excess X .166, 95% CI .132 to .199 | verified | same | same, `rule.target_pair`, `pairs[0]` | |
| Split-half rank ceiling | .984 (half rank r .938) | verified | same | same, `split_half_rank_ceiling` | |

### Paper text against these files (2026-11-08)

Every number above that the paper cites matches its file at the stated rounding. Two places where the wording claims slightly more than the file shows were left unchanged for the authors:

1. Methods, A4 recovery rate table, "Not aligned with true CFA at or above .30, highest": raw .01 and residual .03 are the boundary observer's rates, the observer the check uses. Over all observers, raw reaches .10 (salience-profile at conversational EA 0.5, on 10 replicates at the bound). The row does not say which observer it covers. The same row is in `PREREG_A4_DRAFT.md` line 210, which is frozen.
2. Methods, V1 pilot, "Of 96 intensity pairs, 22 rise by at least .5": 22 pairs gain at least .5 from amplitude .25 to 1, but 2 of them (AU01 on `ea_female`, AU26 on `ea_male`) dip at .5, so they are not among the 113 that rise under the V1 rule. Same wording in `PREREG_A4_DRAFT.md` line 154.
