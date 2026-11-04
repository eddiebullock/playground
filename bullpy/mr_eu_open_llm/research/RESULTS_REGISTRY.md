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
| Qwen acc high/mid/low consensus | 0.543 / 0.383 / 0.210; p = 1.20e-05 | read-only | `scripts/human_calibration.py` (historical) | `results/stats/human_calibration.json` | n = 81 tercile |
| Gemma acc high/mid/low | 0.506 / 0.309 / 0.123; p = 1.57e-07 | read-only | same | same | |
| Molmo acc high/mid/low | 0.346 / 0.148 / 0.099; p = 1.57e-04 | read-only | same | same | not < 1e-4 |
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
| Official branch | **B** | verified | `paper_v2.probe` all-layer | `research/reproduction/phase2_gate_all_layers.json` | locked 2026-11-02; Holm floor |
| All-layer Qwen peak grouped T1 ρ | L34 0.471, p_perm = 0.001, p_Holm = 0.036 | verified | jobs 39709580 / 39712190 | same | 36 layers; 20 fail only Holm |
| All-layer Gemma peak grouped T1 ρ | L2 0.492, p_perm = 0.001, p_Holm = 0.042 | verified | jobs 39709581 / 39712191 | same | 42 layers |
| All-layer Molmo peak grouped T1 ρ | L22 0.518, p_perm = 0.001, p_Holm = 0.032 | verified | jobs 39709582 / 39712192 | same | at chance; completeness |
| Extract completeness | Qwen 36/36, Gemma 42/42, Molmo 32/32 npy; 0 errors | verified | extract_meta on CSD3 | HPC `baseline_*_6afc_all_layers_f4` | last_token, max_frames=4 |
| Qwen layers failing only Holm | 20 / 36 | verified | `paper_v2.probe` gate flags | `research/reproduction/phase2_all_layers_summary.csv` `holm_only_fail` | other four checks pass |
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

