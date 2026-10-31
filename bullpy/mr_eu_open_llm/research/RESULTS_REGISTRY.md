# RESULTS_REGISTRY

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
| Phase 2 would-be branch | A (not locked) | verified | same | same | Molmo excluded from H1 |
| H2 output entropy CIs include 0 | Qwen [−0.121, 0.120]; Gemma [−0.013, 0.235]; Molmo [−0.154, 0.097] | verified | `paper_v2.behaviour` | `research/reproduction/behaviour_summary.json` | confirmatory H2 |

Dump of the Phase 0 recompute: `research/reproduction/phase0_numbers.json`.
Phase 2 dump: `research/reproduction/phase2_gate.json`.
Phase 5 dump: `research/reproduction/phase5_entanglement.json`.
Stimulus features (HPC job 39709870): `research/reproduction/stimulus_features.json` (243/243 ok).
