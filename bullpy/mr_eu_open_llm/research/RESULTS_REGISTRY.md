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

## Rows added 2026-11-08 (from the paper scaffold review)

| claim | value | status | script | output file | notes |
|---|---|---|---|---|---|
| Weak labels, face-only human p_target (item mean) | Kind .2290 (n = 9), Unfriendly .2408 (n = 10), Jealous .2571 (n = 7); 26 items; chance 1/6 | verified | `research/scripts/make_figures.py` panel 1d (mean of `trials.*.p_target` by `human_target_label`) | `data/eu_emotions_human_entropy.json` | Matches DECISIONS 2026-11-04 (.229, .241, .257). Weak list from `WEAK_LABELS` in `scripts/run_stimulus_interventions.py` |
| A2 sign-flip FWER under skewed gain nulls | Per route, long tail opposite the prediction, strong skew: .091 / .098 (gain SD .3), .137 / .130 (SD .5); weak skew .054 / .075, .082 / .093; at step-down P < .01, worst .046 / .044; sharp and hetero .043 to .055 | verified | `research/scripts/a2_gain_skew_sim.py` (1000 reps per cell, frozen `test_family`, exact flips) | `research/reproduction/a2_gain_skew_sim.json` (`results`, `per_route_fwer_by_threshold`) | `single_face` / `video_face`. DECISIONS 2026-11-08 |

## A1 rotated 6AFC, vision mismatch (2026-11-09)

Frozen `research/scripts/analyse_rotation.py` (sha256 `98890c5b...`, unchanged since `410da27`), run once through `analyse_rotation_run.py` (wrapper sha256 `7f52f9e7...`, DECISIONS 2026-11-08, "A1 errored trials") by `research/scripts/post_a1.sh` at 16:55 UTC. Arrays 39982137 and 39982138 COMPLETED. 30 files pulled unread, remote and local sha256 equal (`a1_pull_manifest.json`, sha256 prefix `c7697012`). Output `research/reproduction/rotation_summary.json` (prefix `81500bdd`). A first direct call of the frozen script, before the runbook, stopped on the errored trial with a KeyError before writing or printing anything; the runbook run is the analysis of record.

**Errored trials:** 1 of 7,290 (`rotation_errored_trials.json`, prefix `ec876ebf`): Qwen3-VL, real, rotation 4, `emotions_17/EDITED/Afraid/MF3_cut.mp4`, "Failed to open video" on the cluster (I/O; the clip opened in the other nine Qwen files). Scored wrong in the primary, as frozen, and given 0 target mass in the secondary. Conservative for Qwen's delta.

| Model | Accuracy real | Accuracy mismatch | delta (actor bootstrap 95% CI) | Holm p | Spearman with human p_target | Within-category r | Decision |
|---|---|---|---|---|---|---|---|
| Qwen3-VL | .469 | .067 | .402 (.350 to .461) | .00009 | .409 | .288 | Pass |
| Gemma 4 | .402 | .174 | .228 (.172 to .283) | .00009 | .444 | .138 | Pass |
| Molmo2 | .163 | .092 | .071 (.049 to .094) | .00009 | .380 | .233 | Pass |

Holm p is .00009 for all three (100,000 draws): raw p .00003 for Qwen and Gemma (2 exceedances) and .00006 for Molmo (5), against a smallest possible raw p of .00001, so near the resolution, not at it (wording corrected 2026-11-12 after the draft review, finding 21; values unchanged). Accuracy by target position, real: Qwen .457, .510, .486, .449, .444; Gemma .379, .354, .407, .444, .424; Molmo .198, .132, .140, .169, .177. Accuracy against 1/6, pooled and descriptive: Qwen and Gemma far above; Molmo .163, p .65, not above chance. Molmo's Pass rests on mismatch accuracy falling below chance (.092), so it shows that the face moves Molmo's answers, not that Molmo reads the target; as with A3, a Pass tests vision dependence, not competence.

**Reading against A3** (`PREREG_A3_DRAFT.md`, "Reading against A1"): A1 Pass and A3 Pass on all three models, so "face use on both inputs". RQ0 is settled as R0-a in the Discussion outline, with Molmo2 still out of A2 and of A4's family on its gate failures.

## Natural-clip RQ1, family pattern score, model-frames route (2026-11-09)

`research/scripts/same_clip_family_analysis.py --model-route`, arguments as in STATE item 1, run once per model by the same runbook (exit 0; Qwen 2,230 s, Gemma 341 s, Molmo 296 s). Lambda 10 on all inputs, attempt 1 null of record, 4,999 permutations. Status from the files: "confirmatory: strength rule passed at 0.75x and 1x, at most 5 percent of clips lost" for all three, and the A1/A3 row Pass/Pass makes it confirmatory (DECISIONS 2026-11-07).

| Model | Clips kept | Composite score (p) | First detector (p) | Second detector (p) | Unparsed share | File (sha256 prefix) |
|---|---|---|---|---|---|---|
| Qwen3-VL | 242 of 243 | .549 (.0002) | .548 (.0002) | .520 (.0002) | 0 | `same_clip_family_model_qwen3vl.json` (`bc0d6795`) |
| Gemma 4 | 243 | .418 (.0006) | .411 (.001) | .350 (.0084) | .049 | `same_clip_family_model_gemma4.json` (`0f9c0a26`) |
| Molmo2 | 243 | .543 (.0002) | .522 (.0002) | .469 (.0004) | .0024 | `same_clip_family_model_molmo2.json` (`9ad3d59d`) |

The composite input is the confirmatory one; the two detectors alone are sensitivities. Qwen drops the errored clip (4 of 5 rotations usable; partial clips leave both matrices, DECISIONS 2026-11-04). .0002 is the floor at 4,999 permutations. Reading, as fixed in advance: each model's responses are **aligned with the human pattern** of AU families on these clips. This does not show that they use the same cues (Proposition 4; different-cues size .055 at 0.75x). Single and family-wise cells are exploratory.

## A7 Stage 0, Gemma 4 (2026-11-10)

Job 40468955 (intr QOS, 13 min), frozen A7 code (tag `a7-freeze`, design `9e40cfcf`; the table's `code_sha256` equals the pins). 24 CREMA-D calibration clips only, `single_face`, bfloat16. Records from `research/scripts/a7_stage0_records.py` (commit `dea8734`), records file sha256 `5b10a4d5cba6601918ddee513104a85885c8cb10323e2ad882ed8870f6bf6e37`. Output `research/reproduction/a7_stage0_gemma4.json`, sha256 `fff021f8...12b4`; arms applied by code, nothing reviewed by eye.

| Check | Worst abs nats | Over 0.001 | Result |
|---|---|---|---|
| C1 to C4 | 0.0 each | 0 | Pass |
| C5 blocks A, B | ratios .230, .885 of denominator 5.54 (floor .10) | | both live |

Exit complete; complete cuts: `exit` only; blocks exhaust the exit. Region test: **fail**, reason `leak_over_limit` (leak point .139 and .088, upper .181 and .113 at buffers 0 and 1; token map complete, hit rate 1.0; edit RMS 4.09 against floor .55). **Arms: route yes, token no.**

Reading, from the frozen table (`PREREG_A7_DRAFT.md`, reporting rows): the only complete cut is the whole path, so RQ3 is not testable beyond the interface on Gemma 4. The route arm may run in Stage 1 as a secondary decomposition, only with a DECISIONS entry made before any A2 analysis output is opened. Region test failed: tokens at the cue's location do not contain the cue's effect on calibration clips at this edit, so cue-level mediation is not testable by a token cut on this model.

## A7 Stage 0, Qwen3-VL (2026-11-10)

Job 40468956 (intr QOS, 14 min), same frozen code, records file and clips as Gemma 4. Output `research/reproduction/a7_stage0_qwen3vl.json`, sha256 `3762c7bd...a215`. C1 to C4: worst 0.0 nats, none over 0.001, all pass. C5 blocks A, B: ratios .802, .273 of denominator 4.30, both live. Exit complete; complete cuts `exit` only; blocks exhaust the exit. Region test: **fail**, reason `token_map_failed` (token map hit rate .760 over 96 trials, below its rule), so no leak was computed. **Arms: route yes, token no.** Reading as for Gemma 4: RQ3 not testable beyond the interface; cue-level mediation not testable by a token cut on this model, here because the token map did not locate the cue reliably.

## A6 pre-freeze cluster checks: first-frame AU check blocks the freeze (2026-11-10)

No model output. Steps from `slurm_jobs/study3_a6.sh`: decoder preflight (40595458) ok on `.flv` under cv2 4.11; landmarks (40595459) on 819 clips, 0 failures, `results/a6/geometry.json` sha256 `4fac9d90...7089`; AU list (40595460) `results/a6/au_manifest.json` sha256 `22006880...9d38`. `research/scripts/extract_au_measures.py` unchanged on that list: LibreFace and OpenFace (40597352, 8 CPU shards) and ME-GraphAU (40598906, 8 CPU shards), per-frame CSVs in `results/a6/au_frames*` on the cluster. Then `research/scripts/a6_human_target.py --cue_frames` with all three detectors: `research/reproduction/a6_human_cue_check.json`, sha256 `f42a51ef...aeec`. (A first run before GraphAU had finished, sha256 `fb595954...1273`, is kept on the cluster as `a6_human_cue_check_lf_of_only.json`; it reported GraphAU as missing and is superseded.)

First-frame check (mean first-frame value over mean apex value, clips of the Anger-Disgust pair, limit `first_frame_au_ratio_max` .50 for every detector):

| Detector column | Mean apex | Mean first | Ratio | Over limit |
|---|---|---|---|---|
| `lf_AU09` | .411 | .120 | .291 | no |
| `of_AU09` | .245 | .021 | .085 | no |
| `lfocc_AU10` | 3.578 | -13.333 | -3.727 | no (scale is not a ratio scale; see DECISIONS) |
| `ge_AU10` | .631 | .505 | .801 | **yes** |

`first_frame_blocks_freeze`: **true**. Under the rule fixed before any model output (A6 "Fixed before freezing", item 6), A6 does not freeze. The human cue slopes are descriptive and gate nothing (in the file).

## Rows added 2026-11-12 (paper slots that locked files already answer)

| claim | value | status | script | output file | notes |
|---|---|---|---|---|---|
| Half-shared pattern power, model-frames route | .34 at 0.5x, .83 at 1x, .97 at 2x; AUC against prior-only .824, .974, .995 | verified | `research/scripts/same_clip_family_analysis.py` (route pattern run, DECISIONS 2026-11-04, "Pattern score on the model-frames route, result") | `research/reproduction/same_clip_family_route_pattern_within_category.json` `pattern[*].power_half_shared`, `auc_half_shared_vs_prior_only` (sha256 `2bed8ba2...`) | commit `fae0ebc`; .60 at 0.75x is in the 0.75x file (section "Pattern score, model-frames route"); results.md section 4 |
| Literature table, eight papers coded | Target position controlled: yes 1, partly 1, no 2, not stated 1, does not arise 3. One option set across items: yes 3, no 1, not stated 2, does not arise 2. Delivery to the model checked: not stated 8. Face dependence tested: no 5, partly 3. Rows with an inferred cell: 1 | verified | `research/scripts/registry_derived_counts.py` block 3 (codes as in the CSV; labels as in `make_figures.py` `LIT_CODES`) | `research/reproduction/literature_table.csv` (sha256 `b855a64a...`) | Figure 1e; results.md section 1 (summary); supplementary_methods.md S2, "Literature check" (full coding, moved there from results.md on 2026-11-13, for length). Codes record what our reading of each paper found, at the verification level in each row's `verification` cell; "not stated" means not stated in what we read. 2026-11-13: [1] recoded from partly to no on option sets, on the user's decision (NEEDS_INPUT 11, review finding 36); its alternatives are set per item, and on the MRMET a repeated label always came with the same three alternatives. Before the recode: yes 3, partly 1, not stated 2, does not arise 2 (sha256 `48a7005d...`, commit `2422a6c`) |
| Proposition 2 check, Figure 3b | At cue correlation .5, .9 and .99: output correlation .5, .9, .99; mean squared output difference 1.0, .2, .02; effect cosine (CEA) 0 at every value | verified | `research/scripts/cfa_identifiability.py` (`two_cue_case`) | `research/reproduction/cfa_identifiability.json` `checks.proposition_2` (sha256 `31bd4b40...`) | commit `3fddfc5`; quick, exact (section "Cue spectra"); results.md section 3 |
| Residual CEA with estimated projectors, Proposition 5 check (G9) | 192 settings of 1,000 replicates (options 7 and 5 by 25 cue families; model common-profile share .1, .3, .6; true residual cosine 0 to .9; model raw reliability .5 to .95; profile cosine with the human 1 and .7). Corrected for model and human split-half reliability: within -.008 to +.019 of the truth. Corrected for model reliability only (the correction A4's Not aligned rule uses, applied here to the point estimate; the rule applies it to a bootstrap upper bound, clarified 2026-11-12 after the A4 audit): within -.027 to +.010, and -.016 to +.007 at a set cosine of .3. Estimating the projector lowers the observed residual cosine by up to .044 against a fixed projector; the split-half reliability absorbs it | verified | `research/scripts/residual_cea_attenuation.py` (seed 20261112) | `research/reproduction/residual_cea_attenuation.json` `summary` (sha256 `163f6b56...`) | synthetic matrices only; ridge shrinkage and per-half refits not modelled; Supp. Note 1 section 4; not a proof |

## Rows added 2026-11-12, late (independent draft review, `research/reviews/draft_review_2026-11-13.md`)

| claim | value | status | script | output file | notes |
|---|---|---|---|---|---|
| A1 share of answers "None of the above" (descriptive) | Real cells: Gemma 4 251 of 1,215 (.207), Qwen3-VL 305 (.251), Molmo2 703 (.579). Mismatch: Gemma 4 362 (.298), Qwen3-VL 549 (.452), Molmo2 742 (.611). No errored trial counted as an answer (the one errored Qwen trial is in the denominator) | verified | `research/scripts/a1_nota_shares.py` (answer field `stage2.prediction`, all 30 locked A1 files, inputs hashed) | `research/reproduction/a1_nota_shares.json` (sha256 `eca69704...`) | Review finding 5, which computed the same real-cell counts independently. Counterpart of the A3 line ("Trace (2026-11-08)"). Outside the frozen A1 summary; no decision uses it. results.md sections 2 and 4 |
| Snoek AU configurations shared between the basic and conversational experiments (descriptive) | Unique configurations (33 AU values): basic WC 4,659, basic EA 2,322, conversational WC 2,314, conversational EA 2,322. Basic EA and conversational EA are the same 2,322; all 2,314 conversational WC configurations are among basic WC's. Across cultures: 507 shared between the basic sets, 502 between the conversational sets | verified | `research/scripts/snoek_config_overlap.py` (frozen loader `human_au_effects.load_zip`) | `research/reproduction/snoek_config_overlap.json` (sha256 `0c626d76...`) | Review finding 7, which found the same counts independently. Counted on the 33 AU amplitudes the analysis uses. By configuration identifier only 56 are shared between basic and conversational WC (and 2,322 on EA), which is what `human_au_effects.json` `shared_au_configurations_between_datasets` and its caveat "Western basic and conversational do not" refer to; checked 2026-11-12 from the archive. Explains the identical basic EA and conversational EA rows of the cue-spectra tables. A4's basic and conversational claims for a model rest on largely the same configurations; they were already declared separate claims, not independent tests |
| A9 human target, Liu et al. 2022 Experiment II (pre-data, human only) | 20 subjects, 1,200 stimuli, 24,000 trials. Ridge selected 1.0 (held-out log loss .338; .349 at 0.1, .344 at 3, intercept only .396). Common profile share .150; AU22 and AU27 share .123; detector-confirmed families' share of residual energy .143 | verified | `research/scripts/a9_human_target.py` | `research/reproduction/a9_human_target.json` `datasets.liu_WC.structure`, `heldout_log_loss_by_lambda`, `human_energy` | A9 frozen 2026-11-09 (tag `a9-freeze`). Values rechecked from the file 2026-11-12; review finding 12. Supplementary Methods S7, Methods "Replication on a second human sample (A9)" |
| A9 human ceiling and reference (pre-data, human only) | Split-half r, subjects and stimuli split: raw .905 (SD .015), residual .862 (SD .034); ceiling raw .975, residual .962; with stimuli shared, for comparison only, .987 and .983. Liu against Snoek basic WC raw .861, residual .690; against basic EA .815 and .709; Snoek WC against EA .877 and .780 (descriptive yardstick) | verified | `research/scripts/a9_human_target.py` (ceiling as `a4_analyse.human_do_ceiling`, 200 splits, ridge 1.0) | `research/reproduction/a9_human_ceiling.json` `datasets.liu_WC`, `subjects_only_same_stimuli`, `human_reference_cfa` | Rechecked 2026-11-12; fills the slot in results.md 6b that read "no registry row yet". Liu against Snoek basic WC residual (.690) is the human-level reference for A9, descriptive |
| A9 recovery, run 4 on the final code (pre-data, synthetic) | Design A: 16 of 17 checks on residual and raw (fails pair-swapped estimate within .10 at 0.25x, .64). Design B: 17 of 17 on both. B at rho .25 and .5: residual 16 of 17 (same check, .89 and .87, rule .90), raw 17 of 17. Rates as stated in Supplementary Methods S7 (human-like .823 against true .859 at 0.25x under B; design A .346 against .411; B at rho .5, .359 against .396; design A different-cues Not aligned .98 and half-shared Aligned .84 at 1x) | verified | `scripts/a9_recovery.py` | `research/reproduction/a9_recovery.json` (sha256 `0ca7bd24...`) `runs.<run>.n_passed`, `failed`, `tables` | Pass counts and the three failing-check rates rechecked by the coordinator 2026-11-12; the rates were recomputed by the independent review (finding 12, "Checked and matching"). Runs 2 and 3 identical; run 1 draft code |
| A9 label-order corroboration (pre-data, human only) | Each Liu label's do-effect row (25 families) is closest to the same label's Snoek basic WC row for all seven: anger .648 (next disgust .535), disgust .882, fear .892, happy .926, sadness .805, surprise .947, other .536. Log-odds weights, anger against disgust: AU04 .60 and .37, AU07 .71 and .41, AU23 .60 and -.13; AU09 1.06 and 2.74, AU10 .77 and 2.67 | verified | `research/scripts/a9_label_corroboration.py` (family positions by the frozen `human_au_effects.family_positions`) | `research/reproduction/a9_label_corroboration.json` (sha256 `0bf8fee1...`) | Written 2026-11-12 so that values already in `PREREG_A9_DRAFT.md` ("Response codes") and Supplementary Methods S7 come from an output file; review finding 12 recomputed the same values. Corroborates the response-code order; did not set it |
| Chen et al. 2024 Western file against Snoek basic WC (pre-data, human only) | Snoek subjects 16 to 60: 2,342 of 2,342 rows per subject have a unique exact match to a Chen trial; agreement 1.000 for all 45. Subjects 1 to 15: 2,211 of 2,368 rows match exactly; per-subject agreement .9645 to .9996, median .998 | verified | `research/scripts/a9_human_target.py` (Chen check) | `research/reproduction/a9_chen_overlap.json` `cohorts` | Rechecked 2026-11-12. Verdict "same data": Chen et al. 2024 is the Snoek basic WC experiment and is not used |
| A1 frame delivery per trial (descriptive) | Every A1 trial recorded `composite_grid` (Gemma 4, Molmo2) or `native_list` (Qwen3-VL), except one clip, `emotions_30/EDITED/Surprised/PF19v2_cut.mp4` (1.88 s, `stimulus_features.json`), delivered as its first frame only (`single_first_frame`) in all 30 files, 5 times per model and condition, as its own item and as a mismatch partner; and the one errored Qwen trial, which has no strategy | verified | `research/scripts/a1_frame_strategies.py` | `research/reproduction/a1_frame_strategies.json` (sha256 `164adee7...`) | Review finding 24, which found the same clip. Supplementary Methods, Delivery audit |
| A3 on clips with at least 8 tokens across the face (declared sensitivity) | Gemma 4: 25 of 243 clips under the floor (share .103), delta .225 (actor 95% CI .150 to .294), Holm P .00010, still vision-dependent. Qwen3-VL and Molmo2: no clip under the floor, so identical to the primary (.244 and .063). Holm over the three models as frozen | verified | `research/scripts/a3_token_floor_sensitivity.py` (frozen `face_input_validity.summarise_model` and `decide`; A2's frozen `below_token_floor`; gate 1's formula, reproducing each delivery file's `gate_1_share_clips_at_least_8`; the locked cells reproduce the committed summary first) | `research/reproduction/a3_token_floor_sensitivity.json` (sha256 `672d2b64...`) | Declared in DECISIONS 2026-11-04 ("Gate 1 is read as the median"), before A3 ran; computed 2026-11-12, after the A3 result was read, because the frozen A3 summary omitted it (found by the deviations table). Cannot change the A3 verdicts |
| Human natural-clip family pattern: held-out gain over category alone | Model-frames route (confirmatory route): held-out cross-entropy .25860 with the four families against .25958 with category alone, a relative reduction of .00377 (0.38%); all frames: .00613 (0.61%, as in "Family calibration") | verified | `research/scripts/same_clip_family_analysis.py` (human side, no model output) | `research/reproduction/same_clip_family.json` `cue_sets.families.model_1fps.composite.heldout_ce_cues`, `heldout_ce_category_only`, `heldout_gain`; recovery's `human_scale_calibration.target_human_gain` in `same_clip_family_route_pattern_within_category.json` is the same .00377 | Gain = (category-only CE - cue CE) / category-only CE, checked 2026-11-12 from the two CE values. Raised by the deviations table (DECISIONS entries of 2026-11-04 cite 0.49% and .0038). Supplementary Methods S5 |

## A4 confirmatory analysis (2026-11-14)

Frozen `scripts/a4_analyse.py`, run once by `research/scripts/post_phase3.py analyse a4` at commit `dfb1782`, single-threaded, 01:07 to 01:26 UTC. Run note: `research/reproduction/a4_run_note.json`. Pre-run record: `research/reproduction/a4_prerun.json` (12 complete shards per model, no failures, no rows with non-finite log probabilities, so `cells_dir_used` is the pulled `results/a4`; design `12c454da`). Cells: 24 shard files, 157,892 cells, all status ok, written by nine one-hour `intr` slices (DECISIONS 2026-11-13) and pulled with their sha256 (`research/reproduction/a4_pull_manifest.json`; the 24 cancelled `gpu2` tasks are waived with that entry as the reason; sacct names task 11 of each array `_[11]`). Output: `results/a4/summary.json`, sha256 `08148303f2eb6bc98b694fdc4776da2fd0f800b634cfcc1d5584c66714ee36af`.

Source `results/a4/summary.json` (sha256 `08148303...`), printed by `research/scripts/registry_tables.py`.

Statistic: residual.

| Dataset | Model | Signal P | Residual CEA | 95% CI | R_S | Alignment Holm P | Raw CEA (secondary) | Decision |
|---|---|---|---|---|---|---|---|---|
| basic_WC | Gemma 4 | .001 | .386 | .372 to .397 | .994 | .00020 | .382 | Aligned |
| basic_WC | Qwen3-VL | .001 | .036 | .010 to .065 | .975 | .281 | .112 | Inconclusive |
| basic_EA | Gemma 4 | .001 | .516 | .497 to .535 | .994 | .00020 | .369 | Aligned |
| basic_EA | Qwen3-VL | .001 | .264 | .219 to .299 | .952 | .00080 | .182 | Inconclusive |
| conversational_WC | Gemma 4 | .001 | .025 | -.007 to .052 | .980 | .789 | -.077 | Not aligned |
| conversational_WC | Qwen3-VL | .001 | -.216 | -.279 to -.097 | .544 | .951 | -.097 | Inconclusive |
| conversational_EA | Gemma 4 | .001 | .369 | .348 to .385 | .989 | .00020 | .241 | Aligned |
| conversational_EA | Qwen3-VL | .001 | .063 | -.113 to .198 | .030 | .278 | -.053 | Inconclusive |

| Model | Basic headline | Conversational headline |
|---|---|---|
| Gemma 4 | Aligned | Mixed |
| Qwen3-VL | Inconclusive | Inconclusive |

Other values from the same file (formatted with `registry_tables.py`'s helpers):

| claim | value | status | script | output file | notes |
|---|---|---|---|---|---|
| A4 gates, Gemma 4 | basic_WC all passed; basic_EA all passed; conversational_WC all passed; conversational_EA all passed | verified | `python -m scripts.a4_analyse` (frozen; run once by `post_phase3.py analyse a4`) | `results/a4/summary.json` `datasets.*.<model>.gates` | missing means not passed (PREREG_A4 :141-148) |
| A4 gates, Qwen3-VL | basic_WC failed: v4_domain; basic_EA all passed; conversational_WC failed: v3_config_carries, v4_domain; conversational_EA failed: v4_domain | verified | `python -m scripts.a4_analyse` (frozen; run once by `post_phase3.py analyse a4`) | `results/a4/summary.json` `datasets.*.<model>.gates` | missing means not passed (PREREG_A4 :141-148) |
| A4 V3 replicate configurations, Gemma 4 | basic_WC delta_js .135, P .00010 (n 200); basic_EA delta_js .164, P .00010 (n 200); conversational_WC delta_js .100, P .00010 (n 200); conversational_EA delta_js .100, P .00010 (n 200) | verified | `python -m scripts.a4_analyse` (frozen; run once by `post_phase3.py analyse a4`) | `results/a4/summary.json` `datasets.*.<model>.replicate` | V3 passes at P < .01 |
| A4 V3 replicate configurations, Qwen3-VL | basic_WC delta_js .050, P .00010 (n 200); basic_EA delta_js .086, P .00010 (n 200); conversational_WC delta_js .000, P .185 (n 200); conversational_EA delta_js .002, P .00010 (n 200) | verified | `python -m scripts.a4_analyse` (frozen; run once by `post_phase3.py analyse a4`) | `results/a4/summary.json` `datasets.*.<model>.replicate` | V3 passes at P < .01 |
| A4 share of main stimuli lost, Gemma 4 | basic_WC .000; basic_EA .000; conversational_WC .000; conversational_EA .000 | verified | `python -m scripts.a4_analyse` (frozen; run once by `post_phase3.py analyse a4`) | `results/a4/summary.json` `datasets.*.<model>.share_lost` | limit .05 |
| A4 share of main stimuli lost, Qwen3-VL | basic_WC .000; basic_EA .000; conversational_WC .000; conversational_EA .000 | verified | `python -m scripts.a4_analyse` (frozen; run once by `post_phase3.py analyse a4`) | `results/a4/summary.json` `datasets.*.<model>.share_lost` | limit .05 |
| A4 residual CEA over ceiling, descriptive, Gemma 4 | basic_WC .399; basic_EA .527; conversational_WC .026; conversational_EA .387 | verified | `python -m scripts.a4_analyse` (frozen; run once by `post_phase3.py analyse a4`) | `results/a4/summary.json` `datasets.*.<model>.residual.over_ceiling, lower_bound_only` |  |
| A4 residual CEA over ceiling, descriptive, Qwen3-VL | basic_WC .038; basic_EA .276; conversational_WC -.301 (lower bound only); conversational_EA .381 (lower bound only) | verified | `python -m scripts.a4_analyse` (frozen; run once by `post_phase3.py analyse a4`) | `results/a4/summary.json` `datasets.*.<model>.residual.over_ceiling, lower_bound_only` |  |
| A4 raw CEA with 95% CI, secondary, Gemma 4 | basic_WC .382 (.371 to .389); basic_EA .369 (.357 to .379); conversational_WC -.077 (-.096 to -.060); conversational_EA .241 (.228 to .253) | verified | `python -m scripts.a4_analyse` (frozen; run once by `post_phase3.py analyse a4`) | `results/a4/summary.json` `datasets.*.<model>.raw.cfa, .ci` |  |
| A4 raw CEA with 95% CI, secondary, Qwen3-VL | basic_WC .112 (.095 to .127); basic_EA .182 (.156 to .200); conversational_WC -.097 (-.142 to -.048); conversational_EA -.053 (-.101 to .013) | verified | `python -m scripts.a4_analyse` (frozen; run once by `post_phase3.py analyse a4`) | `results/a4/summary.json` `datasets.*.<model>.raw.cfa, .ci` |  |
| A4 sensitivities, residual CEA, Gemma 4 | basic_WC: without_AU22_AU27 .353, without_other .403, flagged_families .376, confirmed_families .440; basic_EA: without_AU22_AU27 .458, without_other .523, flagged_families .483, confirmed_families .701; conversational_WC: without_AU22_AU27 .102, without_other .074, flagged_families .053, confirmed_families -.172; conversational_EA: without_AU22_AU27 .388, without_other .417, flagged_families .422, confirmed_families .158 | verified | `python -m scripts.a4_analyse` (frozen; run once by `post_phase3.py analyse a4`) | `results/a4/summary.json` `datasets.*.<model>.sensitivities.*.residual` | beside the primary, never replacing it (PREREG_A4 :173); subsets take the common profile from the full matrix |
| A4 sensitivities, residual CEA, Qwen3-VL | basic_WC: without_AU22_AU27 .049, without_other .034, flagged_families .055, confirmed_families -.052; basic_EA: without_AU22_AU27 .244, without_other .268, flagged_families .338, confirmed_families .040; conversational_WC: without_AU22_AU27 -.221, without_other -.206, flagged_families -.237, confirmed_families .048; conversational_EA: without_AU22_AU27 .120, without_other .039, flagged_families .027, confirmed_families .335 | verified | `python -m scripts.a4_analyse` (frozen; run once by `post_phase3.py analyse a4`) | `results/a4/summary.json` `datasets.*.<model>.sensitivities.*.residual` | beside the primary, never replacing it (PREREG_A4 :173); subsets take the common profile from the full matrix |
| A4 amplitude sets, residual CEA, Gemma 4 | basic_EA_0.75x .442; basic_EA_1x_same_configs .483; basic_EA_1.25x .519; conversational_EA_0.75x .170; conversational_EA_1x_same_configs .297; conversational_EA_1.25x .364 | verified | `python -m scripts.a4_analyse` (frozen; run once by `post_phase3.py analyse a4`) | `results/a4/summary.json` `models.<model>.scale.*_residual` | 0.75x and 1.25x against the 1x fit on the same configurations; 0.75x extrapolates past the top amplitude (PREREG_A4 :298) |
| A4 amplitude sets, residual CEA, Qwen3-VL | basic_EA_0.75x .027; basic_EA_1x_same_configs .200; basic_EA_1.25x .217; conversational_EA_0.75x .148; conversational_EA_1x_same_configs .179; conversational_EA_1.25x .273 | verified | `python -m scripts.a4_analyse` (frozen; run once by `post_phase3.py analyse a4`) | `results/a4/summary.json` `models.<model>.scale.*_residual` | 0.75x and 1.25x against the 1x fit on the same configurations; 0.75x extrapolates past the top amplitude (PREREG_A4 :298) |
| A4 direct contrast (one-AU set), residual CEA, Gemma 4 | basic_WC .279; basic_EA .421; conversational_WC .063; conversational_EA .210 | verified | `python -m scripts.a4_analyse` (frozen; run once by `post_phase3.py analyse a4`) | `results/a4/summary.json` `datasets.*.<model>.direct_cfa_residual` | descriptive, at amplitude 1 |
| A4 direct contrast (one-AU set), residual CEA, Qwen3-VL | basic_WC .118; basic_EA .167; conversational_WC -.196; conversational_EA -.074 | verified | `python -m scripts.a4_analyse` (frozen; run once by `post_phase3.py analyse a4`) | `results/a4/summary.json` `datasets.*.<model>.direct_cfa_residual` | descriptive, at amplitude 1 |
| A4 blind readout, secondary, Gemma 4 | gate passed (b1_repeat pass, b2_coverage pass, b3_image_dependence pass, b4_config_carries pass); raw CEA on six families .445; median lexicon mass .989 | verified | `python -m scripts.a4_analyse` (frozen; run once by `post_phase3.py analyse a4`) | `results/a4/summary.json` `models.<model>.blind` | basic_EA blind set; different units from MCQ CEA, not compared (PREREG_A4 :86, :232) |
| A4 blind readout, secondary, Qwen3-VL | gate passed (b1_repeat pass, b2_coverage pass, b3_image_dependence pass, b4_config_carries pass); raw CEA on six families .542; median lexicon mass .967 | verified | `python -m scripts.a4_analyse` (frozen; run once by `post_phase3.py analyse a4`) | `results/a4/summary.json` `models.<model>.blind` | basic_EA blind set; different units from MCQ CEA, not compared (PREREG_A4 :86, :232) |
| A4 behaviour on the same renders, descriptive, Gemma 4 | basic_WC r .140, modal agreement .152, kappa .063; basic_EA r .229, modal agreement .146, kappa .096; conversational_WC r -.030, modal agreement .096, kappa -.021; conversational_EA r .138, modal agreement .110, kappa .055 | verified | `python -m scripts.a4_analyse` (frozen; run once by `post_phase3.py analyse a4`) | `results/a4/summary.json` `datasets.*.<model>.behaviour` | Results 6; option means removed before r (PREREG_A4 :162-169) |
| A4 behaviour on the same renders, descriptive, Qwen3-VL | basic_WC r .044, modal agreement .091, kappa .012; basic_EA r .084, modal agreement .050, kappa .021; conversational_WC r -.025, modal agreement .093, kappa .000; conversational_EA r -.008, modal agreement .019, kappa .000 | verified | `python -m scripts.a4_analyse` (frozen; run once by `post_phase3.py analyse a4`) | `results/a4/summary.json` `datasets.*.<model>.behaviour` | Results 6; option means removed before r (PREREG_A4 :162-169) |
| A4 human WC to EA reference (as used by the analysis) | residual basic .780, conversational .732; raw basic .877, conversational .767 | verified | `python -m scripts.a4_analyse` (frozen; run once by `post_phase3.py analyse a4`) | `results/a4/summary.json` `human_reference` | equal to the pre-data reference rows above |
| A4 V1 detector (as used by the analysis) | share rising .785, pass True | verified | `python -m scripts.a4_analyse` (frozen; run once by `post_phase3.py analyse a4`) | `results/a4/summary.json` `detector` | equal to the A4 V1 pilot rows above |
| A4 residual CI upper bound over the square root of R_S (the Not aligned test), Gemma 4 | basic_WC .398 (not below .30); basic_EA .537 (not below .30); conversational_WC .053 (below .30); conversational_EA .387 (not below .30) | verified (derived) | ci[1] / sqrt(r_s) from the summary fields; `PREREG_A4` Not aligned rule | `results/a4/summary.json` `datasets.*.<model>.residual.ci`, `.r_s` | applies only where every gate passed and signal P < .05 |
| A4 residual CI upper bound over the square root of R_S (the Not aligned test), Qwen3-VL | basic_WC .066 (below .30); basic_EA .306 (not below .30); conversational_WC -.131 (below .30); conversational_EA 1.148 (not below .30) | verified (derived) | ci[1] / sqrt(r_s) from the summary fields; `PREREG_A4` Not aligned rule | `results/a4/summary.json` `datasets.*.<model>.residual.ci`, `.r_s` | applies only where every gate passed and signal P < .05 |

## A9 confirmatory analysis (2026-11-14)

Frozen `scripts/a9_analyse.py`, run once by `research/scripts/post_phase3.py analyse a9` at commit `b9f0241`, single-threaded, 06:33 to 06:34 UTC. Run note: `research/reproduction/a9_run_note.json`. Pre-run record: `research/reproduction/a9_prerun.json` (12 complete shards per model, no failures, no rows with non-finite log probabilities; A4's run note and summary sha256 checked, replication rule (ii)). Cells: 24 shard files, 72,004 cells, all status ok, written by five one-hour `intr` slices beside A2's (DECISIONS 2026-11-13, "A9's smoke and scored shards may run in one-hour slices beside A2's"), after both timing smokes (40 of 40 cells ok each), and pulled with their sha256 (`research/reproduction/a9_pull_manifest.json`; the 24 cancelled `gpu2` tasks are waived with that entry as the reason). The local pre-freeze copy of the manifest, byte-identical to the pin (`4b2f04ae...`), was moved to `results/a9_local_premanifest/` so that the pull could write the cluster's copy. Output: `results/a9/summary.json`, sha256 `1d1a99959e79f58c52a36284819cce13c317cb54d1a27184273bce07cb3e4a7f`.

Source `results/a9/summary.json` (sha256 `1d1a9995...`), printed by `research/scripts/registry_tables.py`.

Statistic: residual; design B.

| Model | Signal P | Residual CEA | 95% CI | R_S | Alignment Holm P (within A9) | Raw CEA (secondary) | Configurations (share lost) | Decision |
|---|---|---|---|---|---|---|---|---|
| Gemma 4 | .001 | .483 | .469 to .496 | .993 | .00020 | .450 | 1200 (.000) | Aligned |
| Qwen3-VL | .001 | .230 | .189 to .269 | .973 | .037 | .102 | 1200 (.000) | Not aligned |

| Model | Gates not passed | Replication reading | A9 decision | A4 basic WC decision (comparator) | A4 basic headline (beside) | Reason |
|---|---|---|---|---|---|---|
| Gemma 4 | none | replicated on a second human sample (same lab and generator) | Aligned | Aligned | Aligned | - |
| Qwen3-VL | none | replication not testable | Not aligned | Inconclusive | Inconclusive | A4 basic_WC Inconclusive: failed or missing gate: v4_domain |

| Model | Over ceiling (descriptive) | Residual correlation across faces (descriptive) | Earlier operationalisation of lead decision 2 (descriptive) |
|---|---|---|---|
| Gemma 4 | .504 | .192 | outside the recovery-validated conditions |
| Qwen3-VL | .242 | .338 | outside the recovery-validated conditions |

Other values from the same file (formatted with `registry_tables.py`'s helpers):

| claim | value | status | script | output file | notes |
|---|---|---|---|---|---|
| A9 gates, Gemma 4 | all passed | verified | `python -m scripts.a9_analyse` (frozen; run once by `post_phase3.py analyse a9`) | `results/a9/summary.json` `models.<model>.gates` | RQ0 and G2 to G4 as in A4; V1 from A4's frozen pilot; V2 to V4 recomputed on Liu renders |
| A9 gates, Qwen3-VL | all passed | verified | `python -m scripts.a9_analyse` (frozen; run once by `post_phase3.py analyse a9`) | `results/a9/summary.json` `models.<model>.gates` | RQ0 and G2 to G4 as in A4; V1 from A4's frozen pilot; V2 to V4 recomputed on Liu renders |
| A9 V3 replicate configurations, Gemma 4 | delta_js .165, P .00010 (n 200) | verified | `python -m scripts.a9_analyse` (frozen; run once by `post_phase3.py analyse a9`) | `results/a9/summary.json` `models.<model>.replicate` | V3 passes at P < .01 |
| A9 V3 replicate configurations, Qwen3-VL | delta_js .064, P .00010 (n 200) | verified | `python -m scripts.a9_analyse` (frozen; run once by `post_phase3.py analyse a9`) | `results/a9/summary.json` `models.<model>.replicate` | V3 passes at P < .01 |
| A9 configurations and share lost, Gemma 4 | 1200 configurations, design B, share lost .000 | verified | `python -m scripts.a9_analyse` (frozen; run once by `post_phase3.py analyse a9`) | `results/a9/summary.json` `models.<model>.n_configurations, .design, .share_lost` | limit .05 |
| A9 configurations and share lost, Qwen3-VL | 1200 configurations, design B, share lost .000 | verified | `python -m scripts.a9_analyse` (frozen; run once by `post_phase3.py analyse a9`) | `results/a9/summary.json` `models.<model>.n_configurations, .design, .share_lost` | limit .05 |
| A9 residual CEA over ceiling, descriptive, Gemma 4 | .504 | verified | `python -m scripts.a9_analyse` (frozen; run once by `post_phase3.py analyse a9`) | `results/a9/summary.json` `models.<model>.residual.over_ceiling` |  |
| A9 residual CEA over ceiling, descriptive, Qwen3-VL | .242 | verified | `python -m scripts.a9_analyse` (frozen; run once by `post_phase3.py analyse a9`) | `results/a9/summary.json` `models.<model>.residual.over_ceiling` |  |
| A9 raw CEA with 95% CI, secondary, Gemma 4 | .450 (.438 to .458), R_S .993 | verified | `python -m scripts.a9_analyse` (frozen; run once by `post_phase3.py analyse a9`) | `results/a9/summary.json` `models.<model>.raw` |  |
| A9 raw CEA with 95% CI, secondary, Qwen3-VL | .102 (.078 to .124), R_S .979 | verified | `python -m scripts.a9_analyse` (frozen; run once by `post_phase3.py analyse a9`) | `results/a9/summary.json` `models.<model>.raw` |  |
| A9 residual CI upper bound over the square root of R_S (the Not aligned test), Gemma 4 | .497 (not below .30) | verified (derived) | from the summary fields; A9 decision rule (unchanged from A4) | `results/a9/summary.json` `models.<model>.residual.ci, .r_s` | derived: ci[1] / sqrt(r_s); applies only where every gate passed and signal P < .05 |
| A9 residual CI upper bound over the square root of R_S (the Not aligned test), Qwen3-VL | .273 (below .30) | verified (derived) | from the summary fields; A9 decision rule (unchanged from A4) | `results/a9/summary.json` `models.<model>.residual.ci, .r_s` | derived: ci[1] / sqrt(r_s); applies only where every gate passed and signal P < .05 |
| A9 sensitivities, residual CEA, Gemma 4 | without_AU22_AU27 .445; without_other .490; flagged_families .477; confirmed_families .523; presence .468; frame_max .502 | verified | `python -m scripts.a9_analyse` (frozen; run once by `post_phase3.py analyse a9`) | `results/a9/summary.json` `models.<model>.sensitivities.*.residual` | beside the primary, never replacing it |
| A9 sensitivities, residual CEA, Qwen3-VL | without_AU22_AU27 .267; without_other .231; flagged_families .321; confirmed_families -.224; presence .253; frame_max .243 | verified | `python -m scripts.a9_analyse` (frozen; run once by `post_phase3.py analyse a9`) | `results/a9/summary.json` `models.<model>.sensitivities.*.residual` | beside the primary, never replacing it |
| A9 behaviour on the same renders, descriptive, Gemma 4 | r .290, modal agreement .182, kappa .122 | verified | `python -m scripts.a9_analyse` (frozen; run once by `post_phase3.py analyse a9`) | `results/a9/summary.json` `models.<model>.behaviour` | per configuration; option means removed before r |
| A9 behaviour on the same renders, descriptive, Qwen3-VL | r .083, modal agreement .043, kappa .009 | verified | `python -m scripts.a9_analyse` (frozen; run once by `post_phase3.py analyse a9`) | `results/a9/summary.json` `models.<model>.behaviour` | per configuration; option means removed before r |
| A9 descriptive fields, Gemma 4 | residual correlation across faces .192; earlier operationalisation of lead decision 2: outside the recovery-validated conditions | verified | `python -m scripts.a9_analyse` (frozen; run once by `post_phase3.py analyse a9`) | `results/a9/summary.json` `models.<model>.face_shared_residual_correlation, .earlier_rule_reading_descriptive` | descriptive, no label on the decision |
| A9 descriptive fields, Qwen3-VL | residual correlation across faces .338; earlier operationalisation of lead decision 2: outside the recovery-validated conditions | verified | `python -m scripts.a9_analyse` (frozen; run once by `post_phase3.py analyse a9`) | `results/a9/summary.json` `models.<model>.face_shared_residual_correlation, .earlier_rule_reading_descriptive` | descriptive, no label on the decision |
| A9 human references (as used by the analysis) | Liu against Snoek basic WC residual .690, raw .861; against basic EA residual .709, raw .815; Snoek WC to EA residual .780 | verified | `python -m scripts.a9_analyse` (frozen; run once by `post_phase3.py analyse a9`) | `results/a9/summary.json` `human_reference` | human-level only if the interval reaches the residual Liu against Snoek basic WC reference (descriptive) |
| A9 withdrawn claim and design | withdrawn under correlated face noise: liu_WC|0.25|pair_error; design B; behaviour ok | verified | `python -m scripts.a9_analyse` (frozen; run once by `post_phase3.py analyse a9`) | `results/a9/summary.json` `withdrawn_under_correlated_face_noise, design, behaviour_status` | PREREG_A9 lead decisions 2 and 8 |

## A2 confirmatory analysis (2026-11-14)

Frozen `scripts/analyse_stimulus_interventions.py` (with `--a2_frozen`) and the declared extras (`research/scripts/a2_declared_extras.py`), run once per model by `research/scripts/post_phase3.py analyse a2` at commit `59641b0`, 12:26 to 12:28 UTC. Run note: `research/reproduction/a2_run_note.json`. Pre-run records: `research/reproduction/a2_prerun_{gemma4,qwen3vl}.json` (three complete shards per model, no failures, no unparseable lines). Cells: six shard files of 2,349 cells each, written by 13 one-hour `intr` slices (DECISIONS 2026-11-13), with no A2-2 repair needed, and pulled with their sha256 and the run reports and geometry (`research/reproduction/a2_pull_manifest.json`; the six cancelled `gpu2` tasks are waived with that entry as the reason).

### A2, Gemma 4

Source `research/reproduction/stimulus_interventions_gemma4.json` (sha256 `4ceb8840...`), printed by `research/scripts/registry_tables.py`.
Source `research/reproduction/a2_extras_gemma4.json` (sha256 `792ee9ad...`), printed by `research/scripts/registry_tables.py`.

| Route | RQ0 | G2 to G4 | Gate B | Delivery | Sham |
|---|---|---|---|---|---|
| single_face | True | True | True | True | True |
| video_face | True | True | True | True | True |

| Route | Contrast | Actors | Change (predicted sign) | 95% CI | Step-down P, one-sided | Two-sided | Frozen verdict | After fill rule | At .01 | 90% equivalence interval |
|---|---|---|---|---|---|---|---|---|---|---|
| single_face | P1 | 8 | 9.983 | 7.660 to 12.326 | .004 | .008 | Matches | Matches | holds | -11.976 to -8.000 |
| single_face | P2 | 8 | 1.070 | -.142 to 2.263 | .234 | .391 | Inconclusive | Inconclusive | - | -2.068 to -.047 |
| single_face | P3 | 8 | .451 | -.372 to 1.291 | .309 | .602 | Inconclusive | Inconclusive | - | -1.160 to .255 |
| single_face | P4 | 8 | 2.282 | .441 to 4.239 | .152 | .266 | Inconclusive | Inconclusive | - | -3.863 to -.717 |
| single_face | P5 | 9 | 2.935 | 1.227 to 4.748 | .042 | .076 | Matches | Matches | not at .01 | -4.448 to -1.473 |
| single_face | P6 | 8 | 4.200 | 3.021 to 5.188 | .004 | .008 | Matches | Matches | holds | -5.049 to -3.221 |
| single_face | P7 | 17 | 3.654 | 1.263 to 6.222 | .056 | .096 | Inconclusive | Inconclusive | - | 1.642 to 5.768 |
| single_face | P8 | 8 | -.577 | -1.660 to .566 | .809 | .602 | Inconclusive | Inconclusive | - | -1.488 to .413 |
| single_face | P9 | 8 | 2.554 | 1.092 to 4.159 | .042 | .076 | Matches | Matches | not at .01 | 1.294 to 3.887 |
| single_face | P10 | 8 | 4.534 | 2.665 to 6.291 | .008 | .016 | Matches | Matches | holds | -6.078 to -2.970 |
| video_face | P12 | 17 | -1.932 | -2.960 to -1.043 | 1.000 | .024 | Opposes | Opposes | not at .01 | 1.179 to 2.777 |
| video_face | P13 | 17 | -.691 | -1.125 to -.317 | 1.000 | .071 | Inconclusive | Inconclusive | - | .366 to 1.051 |
| video_face | P14a | 8 | -1.077 | -2.111 to -.001 | 1.000 | .295 | Inconclusive | Inconclusive | - | -1.954 to -.186 |
| video_face | P14b | 9 | 2.092 | .819 to 3.421 | .052 | .095 | Inconclusive | Inconclusive | - | 1.029 to 3.202 |
| video_face | P15a | 9 | -1.520 | -3.101 to .035 | 1.000 | .295 | Inconclusive | Inconclusive | - | -2.851 to -.208 |
| video_face | P15b | 8 | -.646 | -1.783 to .586 | .999 | .336 | Inconclusive | Inconclusive | - | -1.632 to .380 |

### A2, Qwen3-VL

Source `research/reproduction/stimulus_interventions_qwen3vl.json` (sha256 `ff6a7741...`), printed by `research/scripts/registry_tables.py`.
Source `research/reproduction/a2_extras_qwen3vl.json` (sha256 `0d2e8d0d...`), printed by `research/scripts/registry_tables.py`.

| Route | RQ0 | G2 to G4 | Gate B | Delivery | Sham |
|---|---|---|---|---|---|
| single_face | True | True | True | True | True |
| video_face | True | True | True | True | True |

| Route | Contrast | Actors | Change (predicted sign) | 95% CI | Step-down P, one-sided | Two-sided | Frozen verdict | After fill rule | At .01 | 90% equivalence interval |
|---|---|---|---|---|---|---|---|---|---|---|
| single_face | P1 | 8 | 9.376 | 6.717 to 12.113 | .004 | .008 | Matches | Matches | holds | -11.674 to -7.143 |
| single_face | P2 | 8 | .769 | -.633 to 2.000 | .448 | .492 | Inconclusive | Inconclusive | - | -1.816 to .407 |
| single_face | P3 | 8 | 1.635 | .550 to 2.730 | .062 | .122 | Inconclusive | Inconclusive | - | -2.559 to -.704 |
| single_face | P4 | 8 | 3.941 | 2.058 to 6.160 | .017 | .034 | Matches | Matches | not at .01 | -5.757 to -2.312 |
| single_face | P5 | 9 | 3.336 | 1.991 to 4.896 | .004 | .008 | Matches | Matches | holds | -4.657 to -2.174 |
| single_face | P6 | 8 | 4.722 | 3.073 to 6.686 | .004 | .008 | Matches | Matches | holds | -6.321 to -3.288 |
| single_face | P7 | 17 | 1.671 | .230 to 3.011 | .076 | .142 | Inconclusive | Inconclusive | - | .480 to 2.789 |
| single_face | P8 | 8 | .061 | -.911 to 1.101 | .709 | .922 | Inconclusive | Inconclusive | - | -.765 to .933 |
| single_face | P9 | 8 | 1.060 | .406 to 1.745 | .061 | .121 | Inconclusive | Inconclusive | - | .477 to 1.638 |
| single_face | P10 | 8 | -1.574 | -3.603 to .241 | .902 | .458 | Inconclusive | Inconclusive | - | -.017 to 3.228 |
| video_face | P12 | 17 | .484 | .149 to .847 | .095 | .186 | Inconclusive | Inconclusive | - | -.785 to -.195 |
| video_face | P13 | 17 | .310 | -.051 to .673 | .301 | .554 | Inconclusive | Inconclusive | - | -.620 to -.008 |
| video_face | P14a | 8 | -.154 | -1.163 to .853 | .943 | .789 | Inconclusive | Inconclusive | - | -1.017 to .697 |
| video_face | P14b | 9 | .793 | -.796 to 2.240 | .549 | .759 | Inconclusive | Inconclusive | - | -.555 to 2.018 |
| video_face | P15a | 9 | -.691 | -1.687 to .309 | .958 | .686 | Inconclusive | Inconclusive | - | -1.543 to .149 |
| video_face | P15b | 8 | -.280 | -.857 to .366 | .958 | .759 | Inconclusive | Inconclusive | - | -.780 to .249 |


Other values from the same files (formatted with `registry_tables.py`'s helpers):

| claim | value | status | script | output file | notes |
|---|---|---|---|---|---|
| A2 items and failed cells, Gemma 4 | 243 items; 90 cells written as failed with a reason | verified | `python -m scripts.analyse_stimulus_interventions --a2_frozen` and `research/scripts/a2_declared_extras.py` (frozen; run once by `post_phase3.py analyse a2`) | `research/reproduction/stimulus_interventions_gemma4.json` `n_items, failed_cells, failures_by_route_condition_label_reason` | A2 writes a cell it cannot edit as failed; the analysis needs them present |
| A2 gates, Gemma 4, single_face | RQ0, G2 to G4, Gate B, delivery and sham all passed; Gate B: B1 max change .000, B2 median lexicon mass .621, B3 5.529 nats, B4 5.658 (P .00003), B5 5.999 (P .00003) | verified | `python -m scripts.analyse_stimulus_interventions --a2_frozen` and `research/scripts/a2_declared_extras.py` (frozen; run once by `post_phase3.py analyse a2`) | `research/reproduction/stimulus_interventions_gemma4.json` `single_face.gates, .gate_b` | Gate B is B1 to B5 (PREREG_A2 :109-117) |
| A2 classes after the fill rule, Gemma 4, single_face | Matches 5, Opposes 0, Null 0, Inconclusive 5 of 10; Matches holding at step-down P < .01: P1, P6, P10; the fill rule changed 0 verdicts | verified | `python -m scripts.analyse_stimulus_interventions --a2_frozen` and `research/scripts/a2_declared_extras.py` (frozen; run once by `post_phase3.py analyse a2`) | `research/reproduction/stimulus_interventions_gemma4.json` `single_face.results.primary_blind_acted.*.verdict; extras single_face.contrasts.*.verdict_after_fill_rule` |  |
| A2 sensitivities, verdict changes against the primary, Gemma 4, single_face | without_smoke_items: no change / without_weak_labels: no change / without_mode_not_target: P5 Matches to Inconclusive / clips_at_8_tokens_or_more: no change / without_align_flagged: P5 Matches to Inconclusive; P6 Matches to Inconclusive; P9 Matches to Inconclusive; P10 Matches to Inconclusive / without_donor_is_sent_frame: no change / without_own_neutral_donor: no change / canonical_word_only: P4 Inconclusive to Matches | verified | `python -m scripts.analyse_stimulus_interventions --a2_frozen` and `research/scripts/a2_declared_extras.py` (frozen; run once by `post_phase3.py analyse a2`) | `research/reproduction/stimulus_interventions_gemma4.json` `single_face.sensitivity.*.primary_blind_acted.*.verdict` | beside the primary, never replacing it |
| A2 capped contrasts, 90% equivalence intervals, Gemma 4, single_face | P5 -4.448 to -1.473; P6 -5.049 to -3.221; P9 1.294 to 3.887; P10 -6.078 to -2.970 | verified | `python -m scripts.analyse_stimulus_interventions --a2_frozen` and `research/scripts/a2_declared_extras.py` (frozen; run once by `post_phase3.py analyse a2`) | `research/reproduction/a2_extras_gemma4.json` `extras single_face.contrasts.*.equivalence_interval_90` | these contrasts cannot reach Null by design (PREREG_A2 :199); reported as intervals, not nulls |
| A2 gates, Gemma 4, video_face | RQ0, G2 to G4, Gate B, delivery and sham all passed; Gate B: B1 max change .000, B2 median lexicon mass .520, B3 4.971 nats, B4 4.947 (P .00003), B5 5.006 (P .00003) | verified | `python -m scripts.analyse_stimulus_interventions --a2_frozen` and `research/scripts/a2_declared_extras.py` (frozen; run once by `post_phase3.py analyse a2`) | `research/reproduction/stimulus_interventions_gemma4.json` `video_face.gates, .gate_b` | Gate B is B1 to B5 (PREREG_A2 :109-117) |
| A2 classes after the fill rule, Gemma 4, video_face | Matches 0, Opposes 1, Null 0, Inconclusive 5 of 6; Matches holding at step-down P < .01: none; the fill rule changed 0 verdicts | verified | `python -m scripts.analyse_stimulus_interventions --a2_frozen` and `research/scripts/a2_declared_extras.py` (frozen; run once by `post_phase3.py analyse a2`) | `research/reproduction/stimulus_interventions_gemma4.json` `video_face.results.primary_blind_acted.*.verdict; extras video_face.contrasts.*.verdict_after_fill_rule` |  |
| A2 sensitivities, verdict changes against the primary, Gemma 4, video_face | without_smoke_items: no change / without_weak_labels: no change / without_mode_not_target: P13 Inconclusive to Opposes; P14b Inconclusive to Matches / clips_at_8_tokens_or_more: no change / without_align_flagged: no change / without_donor_is_sent_frame: no change / without_own_neutral_donor: no change / canonical_word_only: P14b Inconclusive to Matches | verified | `python -m scripts.analyse_stimulus_interventions --a2_frozen` and `research/scripts/a2_declared_extras.py` (frozen; run once by `post_phase3.py analyse a2`) | `research/reproduction/stimulus_interventions_gemma4.json` `video_face.sensitivity.*.primary_blind_acted.*.verdict` | beside the primary, never replacing it |
| A2 capped contrasts, 90% equivalence intervals, Gemma 4, video_face | P14a -1.954 to -.186; P14b 1.029 to 3.202; P15a -2.851 to -.208; P15b -1.632 to .380 | verified | `python -m scripts.analyse_stimulus_interventions --a2_frozen` and `research/scripts/a2_declared_extras.py` (frozen; run once by `post_phase3.py analyse a2`) | `research/reproduction/a2_extras_gemma4.json` `extras video_face.contrasts.*.equivalence_interval_90` | these contrasts cannot reach Null by design (PREREG_A2 :199); reported as intervals, not nulls |
| A2 frame shuffle, descriptive, Gemma 4 | per-label mean change -.392 to 2.038 over 23 labels; 9 labels with a 95% interval excluding 0 | verified | `python -m scripts.analyse_stimulus_interventions --a2_frozen` and `research/scripts/a2_declared_extras.py` (frozen; run once by `post_phase3.py analyse a2`) | `research/reproduction/a2_extras_gemma4.json` `extras video_face.shuffles_overall; beside descriptive_changes.static_apex` | descriptive |
| A2 items and failed cells, Qwen3-VL | 243 items; 90 cells written as failed with a reason | verified | `python -m scripts.analyse_stimulus_interventions --a2_frozen` and `research/scripts/a2_declared_extras.py` (frozen; run once by `post_phase3.py analyse a2`) | `research/reproduction/stimulus_interventions_qwen3vl.json` `n_items, failed_cells, failures_by_route_condition_label_reason` | A2 writes a cell it cannot edit as failed; the analysis needs them present |
| A2 gates, Qwen3-VL, single_face | RQ0, G2 to G4, Gate B, delivery and sham all passed; Gate B: B1 max change .000, B2 median lexicon mass .943, B3 5.067 nats, B4 5.413 (P .00003), B5 5.630 (P .00003) | verified | `python -m scripts.analyse_stimulus_interventions --a2_frozen` and `research/scripts/a2_declared_extras.py` (frozen; run once by `post_phase3.py analyse a2`) | `research/reproduction/stimulus_interventions_qwen3vl.json` `single_face.gates, .gate_b` | Gate B is B1 to B5 (PREREG_A2 :109-117) |
| A2 classes after the fill rule, Qwen3-VL, single_face | Matches 4, Opposes 0, Null 0, Inconclusive 6 of 10; Matches holding at step-down P < .01: P1, P5, P6; the fill rule changed 0 verdicts | verified | `python -m scripts.analyse_stimulus_interventions --a2_frozen` and `research/scripts/a2_declared_extras.py` (frozen; run once by `post_phase3.py analyse a2`) | `research/reproduction/stimulus_interventions_qwen3vl.json` `single_face.results.primary_blind_acted.*.verdict; extras single_face.contrasts.*.verdict_after_fill_rule` |  |
| A2 sensitivities, verdict changes against the primary, Qwen3-VL, single_face | without_smoke_items: no change / without_weak_labels: no change / without_mode_not_target: no change / clips_at_8_tokens_or_more: no change / without_align_flagged: P4 Matches to Inconclusive; P6 Matches to Inconclusive; P9 Inconclusive to Matches / without_donor_is_sent_frame: no change / without_own_neutral_donor: no change / canonical_word_only: P3 Inconclusive to Matches; P7 Inconclusive to Matches | verified | `python -m scripts.analyse_stimulus_interventions --a2_frozen` and `research/scripts/a2_declared_extras.py` (frozen; run once by `post_phase3.py analyse a2`) | `research/reproduction/stimulus_interventions_qwen3vl.json` `single_face.sensitivity.*.primary_blind_acted.*.verdict` | beside the primary, never replacing it |
| A2 capped contrasts, 90% equivalence intervals, Qwen3-VL, single_face | P5 -4.657 to -2.174; P6 -6.321 to -3.288; P9 .477 to 1.638; P10 -.017 to 3.228 | verified | `python -m scripts.analyse_stimulus_interventions --a2_frozen` and `research/scripts/a2_declared_extras.py` (frozen; run once by `post_phase3.py analyse a2`) | `research/reproduction/a2_extras_qwen3vl.json` `extras single_face.contrasts.*.equivalence_interval_90` | these contrasts cannot reach Null by design (PREREG_A2 :199); reported as intervals, not nulls |
| A2 gates, Qwen3-VL, video_face | RQ0, G2 to G4, Gate B, delivery and sham all passed; Gate B: B1 max change .000, B2 median lexicon mass .913, B3 3.981 nats, B4 5.131 (P .00003), B5 5.327 (P .00003) | verified | `python -m scripts.analyse_stimulus_interventions --a2_frozen` and `research/scripts/a2_declared_extras.py` (frozen; run once by `post_phase3.py analyse a2`) | `research/reproduction/stimulus_interventions_qwen3vl.json` `video_face.gates, .gate_b` | Gate B is B1 to B5 (PREREG_A2 :109-117) |
| A2 classes after the fill rule, Qwen3-VL, video_face | Matches 0, Opposes 0, Null 0, Inconclusive 6 of 6; Matches holding at step-down P < .01: none; the fill rule changed 0 verdicts | verified | `python -m scripts.analyse_stimulus_interventions --a2_frozen` and `research/scripts/a2_declared_extras.py` (frozen; run once by `post_phase3.py analyse a2`) | `research/reproduction/stimulus_interventions_qwen3vl.json` `video_face.results.primary_blind_acted.*.verdict; extras video_face.contrasts.*.verdict_after_fill_rule` |  |
| A2 sensitivities, verdict changes against the primary, Qwen3-VL, video_face | without_smoke_items: no change / without_weak_labels: no change / without_mode_not_target: no change / clips_at_8_tokens_or_more: no change / without_align_flagged: no change / without_donor_is_sent_frame: no change / without_own_neutral_donor: no change / canonical_word_only: no change | verified | `python -m scripts.analyse_stimulus_interventions --a2_frozen` and `research/scripts/a2_declared_extras.py` (frozen; run once by `post_phase3.py analyse a2`) | `research/reproduction/stimulus_interventions_qwen3vl.json` `video_face.sensitivity.*.primary_blind_acted.*.verdict` | beside the primary, never replacing it |
| A2 capped contrasts, 90% equivalence intervals, Qwen3-VL, video_face | P12 -.785 to -.195; P13 -.620 to -.008; P14a -1.017 to .697; P14b -.555 to 2.018; P15a -1.543 to .149; P15b -.780 to .249 | verified | `python -m scripts.analyse_stimulus_interventions --a2_frozen` and `research/scripts/a2_declared_extras.py` (frozen; run once by `post_phase3.py analyse a2`) | `research/reproduction/a2_extras_qwen3vl.json` `extras video_face.contrasts.*.equivalence_interval_90` | these contrasts cannot reach Null by design (PREREG_A2 :199); reported as intervals, not nulls |
| A2 frame shuffle, descriptive, Qwen3-VL | per-label mean change -.661 to .337 over 23 labels; 5 labels with a 95% interval excluding 0 | verified | `python -m scripts.analyse_stimulus_interventions --a2_frozen` and `research/scripts/a2_declared_extras.py` (frozen; run once by `post_phase3.py analyse a2`) | `research/reproduction/a2_extras_qwen3vl.json` `extras video_face.shuffles_overall; beside descriptive_changes.static_apex` | descriptive |
| A2 frozen-code warning, both models | the fill check (`analyse_stimulus_interventions.py` line 179) printed 'Mean of empty slice' twice per model: in a secondary block an other fill had no actor effects and read as 'fills do not agree'. Under the primary readout every other fill has actor effects (checked with the frozen functions), and no contrast in any block that is significant in the predicted direction has fills that do not agree, so no verdict depends on it | verified (diagnostic) | scratch diagnostic using `actor_effects` and `CONFIRMATORY` | A2 outputs `results.*.*.fills_agree` | diagnostic with the frozen functions, 2026-11-14; counts only |
