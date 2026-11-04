# DECISIONS

Non-trivial decisions. Date, decision, alternatives, why. Do not rewrite history; append.

## 2026-10-30 — Git remote and integration branch

- **Decision:** Treat `origin` (`git@github-personal:eddiebullock/playground.git`) as the personal remote. Create long-lived `paper-v2` from `origin/main`, not from the stale local `main`.
- **Alternatives:** (a) Use local `main` (11 commits behind); (b) add a second remote named `personal`.
- **Why:** `origin` already points at `eddiebullock/playground`. The 11 missing commits are unrelated Study A1/A2 work and should be in the default-branch history before we branch. Local `main` also has unrelated dirty files (`notes/notes.txt`, submodules) that must not enter this paper's commits.

## 2026-10-30 — Public repo + gitignore exceptions

- **Decision:** Keep working on the public repo. Extend `bullpy/mr_eu_open_llm/.gitignore` with targeted `!` exceptions so small result JSON/CSV, study3 figures, human item-level distributions, and `research/` are committable. Do not commit videos, activation tensors, model weights, `.env`, or finetune `tokenizer.json`.
- **Alternatives:** (a) Stop until the user makes the repo private; (b) commit nothing until gitignore is perfect.
- **Why:** Brief forbids sitting idle. Stimuli stay out. Aggregated per-item human response distributions (not videos) are required to reproduce entropy/RSA and are already on disk. Flagged in `NEEDS_INPUT.md` so the user can privatise if they want.

## 2026-10-30 — Molmo is at chance; do not claim otherwise

- **Decision:** Treat only Qwen3-VL and Gemma 4 as above-chance recognisers. Molmo2 48/243 = 19.8%, one-sided binomial p = 0.1155 vs chance 1/6. Mechanistic Molmo results may be reported as a completeness check, never as evidence of mental-state recognition.
- **Alternatives:** Keep the three-model "all above chance" wording from the study3 agent brief table.
- **Why:** Independent `scipy.stats.binomtest` matches the stored `p_binom_gt_chance` in the eval JSON. The brief's instruction to fix this is confirmed, not assumed.

## 2026-10-30 — Ablation headline in the user brief is unverified

- **Decision:** Do not cite `Δacc = 0`, `Δconfusion = 0`, or `5/180 flips` until a result file exists. Local docs (`docs/study3_brief_for_claude_agent.md`) mark the v2 ablation as "code ready — not run". No `qwen3vl_eu_ablation_layer4.json` on disk. HPC is unreachable.
- **Alternatives:** Copy the brief's numbers into the paper.
- **Why:** Integrity rule: never fill in a number. Steering (n = 36, layer 4, last token) **is** on disk and is the causal artifact we can currently discuss.

## 2026-10-30 — Difficulty-matched RSA numbers in the user brief are unverified

- **Decision:** Treat brief values (matched RSA 0.03–0.19; Bored–Unfriendly +0.27/+0.29/+0.47) as **not reproduced**. Probe JSON has only raw RSA. Code for `rsa_difficulty_matched` exists in `scripts/confusability_probe_rsa.py` but was not applied to the saved JSON (or the output was not pulled).
- **Why:** Cannot re-run without activation `.npy`. Will run as soon as caches are present, after preregistration.

## 2026-10-30 — Pair–entropy cosine: report signed values, not only |cos|

- **Decision:** Primary report is signed cosine from the saved unit axes. Qwen L4: Bored–Unfriendly +0.477, Interested–Kind +0.652, Disappointed–Worried **−0.527**. The brief's 0.48–0.65 range matches **absolute** cosine, which hides the sign flip.
- **Why:** Sign is scientifically meaningful (anti-aligned pair axis). Absolute-value summaries stay in a footnote/appendix.

## 2026-10-30 — Human EU-Emotion norms are Face (video-only)

- **Decision:** Phase 6.3 modality check is answered from the builder, not from memory: `scripts/human_entropy.py` filters the O'Reilly workbook to `FACE_MODALITY = "Face"`, and `data/eu_emotions_human_entropy.json` records `"modality": "Face"`. Model evals are `video_only`. Treat as modality-matched for the Face column of EU-Emotion, and still cite the paper for what "Face" meant in the validation study.
- **Why:** This is the documented filter, not an inference from filenames.

## 2026-10-30 — Actor IDs exist; grouped CV is possible

- **Decision:** Pre-register grouped-by-actor CV. Filenames parse to 17 actor letters (B–S, no Q) via the same regex as `parse_stimulus_code`.
- **Why:** Brief flags actor leakage as critical. Metadata does not store an `actor_id` field, but the stimulus code does.

## 2026-10-30 — Ablation n=36 and difficulty-matched RSA are now on disk

- **Decision:** Cite Δacc = 0, Δconfusion = 0, 5/180 flips from `results/mech/pilot_n36/`. Difficulty-matched RSA from the pulled `*_confusability_probe_rsa.json` files. Supercedes the earlier "unverified / missing" entries.
- **Why:** Files were on CSD3 and have been pulled. Counts recomputed from `ablate_trials_qwen3vl_layer4.csv` (5 prediction changes vs baseline: entropy 0, bored 1, disappointed 1, interested 2, random 1).

## 2026-10-30 — Phase 2 gate statistic is grouped-by-actor CV; 3-layer cache is exploratory

- **Decision:** The confirmatory ρ for Holm / permutation / Δρ vs stimulus is **Leave-One-Actor-Out**. LOO is still computed and reported for continuity with the pilot. Inner α is RidgeCV GCV on the training fold. Bootstrap CIs resample held-out (y, pred) pairs (2,000), not full nested CV. The 3-layer last-token run applies the full rule but **does not lock Branch A/B** (prereg: pilot 3-layer probes are exploratory). Official lock waits for `layer_set=all_layers`.
- **Alternatives:** Use LOO as the gate (matches the pilot script); re-run nested CV on every bootstrap sample.
- **Why:** Actor leakage was flagged as the critical confound. GCV and prediction-pair bootstrap keep ≥1,000 permutations tractable on CPU. Permutation refits use dual Ridge with α frozen from full-sample GCV (the observed ρ still uses nested GCV per fold).

## 2026-10-30 — Phase 2 3-layer run: would-be A, not locked

- **Decision:** Do not lock Branch A. Record an exploratory would-be A from Qwen L26 and Gemma L15. Re-apply the identical rule on the all-layer extract before the paper commits to a branch. Molmo’s passing layers are completeness only (not above chance).
- **Numbers (grouped-by-actor T1, 1000 perms, 2000 bootstrap, stimulus = actor one-hot + n_frames + last_frame + luminance + motion + duration):** Qwen L26 ρ = 0.440, p_Holm = 0.003, Δρ vs stim = 0.303 (CI 0.148–0.449). Gemma L15 ρ = 0.328, p_Holm = 0.003, Δρ = 0.190 (CI 0.021–0.360). Qwen L4/L13 fail Holm or Δρ. Gemma L31 fails the category-residual CI. File: `research/reproduction/phase2_gate.json`.
- **Category residual is often negative.** The pre-registered check is “CI excludes 0”, two-sided. Passing layers satisfy it with ρ ≈ −0.13 to −0.15. That means the positive T1 signal is largely between emotion categories. Do not rewrite the rule after seeing the sign.
- **LOO > grouped** (Qwen L4 0.39 vs 0.19): actor leakage is real; grouped CV is the right gate statistic.
- **Why not lock:** Preregistration: “Pilot 3-layer last-token probes are exploratory and do not decide the branch.” Holm over 3 layers is more lenient than Holm over all layers.

## 2026-11-02 — Official Phase 2 gate locks Branch B

- **Decision:** Lock **Branch B**. Do not raise n_perm, switch to BH-FDR, or shrink the layer family to recover Branch A.
- **Numbers:** All-layer last-token probes, jobs 39712190–92 after extracts 39709580–82. Qwen 36 layers, peak grouped T1 ρ = 0.471 at L34, raw p_perm = 0.001, p_Holm = 0.036. Gemma 42 layers, peak ρ = 0.492 at L2, p_Holm = 0.042. Zero layers with p_Holm < 0.01. Twenty Qwen layers fail *only* the Holm check (the other four gate checks pass). Molmo remains below chance. File: `research/reproduction/phase2_gate_all_layers.json`.
- **Why:** Preregistration forbids retuning after seeing the all-layer result. The 3-layer would-be A was exploratory (Holm over three tests). Paper is: above-chance, non-human-like output entropy, a mechanistic map, entanglement methods, generalisation. Phase 4 (readout/steering) is off.

## 2026-10-30 — Phase 5: 1−p_target and human entropy are almost the same target

- **Decision:** Treat the two scalar targets as entangled by construction. Spearman ρ(1−p_target, human entropy) = 0.954 on the 243 EU items. Simulated probes with that correlation have cosine ≈ 0.95; observed entropy vs confusability probe cosines are 0.78–0.83, i.e. *below* the Gaussian null, not above it. Partial RSA controlling an entropy-difference RDM stays near the raw RSA (still ~0).
- **Why:** Numbers from `research/reproduction/phase5_entanglement.json` (n_sim = 100, Ridge α = 10). This is H4, not a reason to drop T1; it is a reason T2 cannot be an independent discovery.

## 2026-11-02 — All-layer RSA on CPU; do not overwrite the 3-layer file

- **Decision:** Queue icelake RSA/probe jobs on `baseline_*_6afc_all_layers_f4` writing `results/mech/{model}_confusability_probe_rsa_all_layers.json`. Leave the 3-layer `*_confusability_probe_rsa.json` untouched. Exploratory (prereg).
- **Why:** All-layer last-token caches are on disk; GPU ablation is still Priority-bound. Difficulty-matched RSA is in the Branch B methods list and can run on CPU without delaying A1/A2/CAS.

## 2026-10-30 — Do not queue extra GPU option-order jobs yet

- **Decision:** Land the option-order interface and slurm script; do not `sbatch` them while 39709579–39709584 are pending.
- **Why:** Ampere is Priority-bound. Option-order is a full 243-trial eval and would delay all-layer extracts needed for the official gate.

## 2026-10-30 — Continue without CSD3; do not download 8B weights onto this Mac

- **Decision:** Sequence CPU work (Phase 1 refactor, tests, prereg, paper methods, Phase 5 entanglement null once activations exist) while HPC auth is broken. Do not pull Hugging Face 8B checkpoints onto a 16 GB M3 Air with 29 GB free.
- **Why:** Disk and unified memory cannot host the GPU phases. Gemma 4 E4B on MPS is a maybe for smoke tests only, after Phase 1.

## 2026-11-04 — Prompt confound; Phase 2 positive probe claim withdrawn pending rerun

- **Decision:** Stop reporting 6AFC accuracy and last-token T1 probes as evidence of mental-state recognition or latent human-ambiguity knowledge. Run prereg amendment A1 (5 rotations, real and vision-mismatch, 3 models). Do not requeue last-token ablation.
- **Numbers:** target in option 1 on 243/243 items; modal option-1 choices 92, 76, 48 equal the correct counts by construction. 21 option sets for 21 labels. Category-only grouped T1 rho = 0.503 vs peaks 0.471 / 0.492 / 0.518. Category decode from last-token activations 0.96 to 1.00, against 6AFC accuracy of 0.20 to 0.38. Within-category probes r = -0.06 to 0.18, none significant after correction across 9 site tests. File: `research/reproduction/prompt_confound_audit.json`.
- **Partial good news:** shuffled-order 4AFC runs (n = 118) are above chance for Qwen2-VL (0.415, video-only) and Gemma 4 (0.432, multimodal); LLaVA-NeXT is at chance (0.254). Within-category model p_target tracks human p_target for Qwen (r = 0.16) and Molmo (r = 0.21), and Qwen and Gemma track human abstention (r = 0.19, 0.20). Those are small, uncorrected, exploratory.
- **Alternatives:** Keep the Branch B narrative and add a caveat. Rejected: the confound touches the headline, and the option-order check was already listed as a caveat trigger.
- **Why:** A paper resting on these results would fail review. The RMET arm is unaffected on position (targets at 10/11/9/6).

## 2026-11-04 — Ablation jobs were cancelled, not pending

- **Decision:** Correct `STATE.md`. Jobs 39709579, 39709583, 39709584 show CANCELLED on 2026-11-02 18:21 with 0 elapsed. Cause not recorded here.

## 2026-11-04 — The 3 Nov study note matches this repo; its main claim is not supported as stated

- **Decision:** Do not use "models know where humans are uncertain, but don't show it in their answers" as the main claim. The category label alone reproduces the pilot LOO probes (0.501 for human entropy; 0.526 to 0.703 for model 1-p_target).
- **Bored-Unfriendly:** Not resolved. Its option overlap (Jaccard 0.43) ties for the highest of 210 label pairs (18 pairs share that value), so shared prompt text is a live explanation and has to be ruled out. Overlap alone cannot prove it, since other pairs tie. Human confusion for this pair could only run one way, because Unfriendly was not offered on Bored items.

## 2026-11-04 — Keep acted labels, add human-referenced scoring

- **Decision:** Report results against the human response distribution as well as the acted label, and flag Kind, Unfriendly and Jealous (26 items; human p_target 0.229, 0.241, 0.257 vs chance 0.167).
- **Why:** These labels are barely recognisable from the face alone, so acted-label accuracy is a poor ground truth there. Amendment A1 is unchanged; this is a report-level addition.

## 2026-11-04 — Input limits on A1, recorded before any rotated result was read

- **Decision:** Keep A1 as frozen, with the same frame policy as the original runs. Interpret it with these limits, written before the analysis.
- **Limits:** On composite-grid models, `make_frame_grid` keeps the first 8 frames while metadata reports all frames as used; by clip duration, 68 of 243 clips lose frames at 1 fps, cap 16. The grid also squashes 16:9 frames into square cells. On Gemma 4 the grid gives about 1 token across the face, against about 9 for a single face crop.
- **Consequence:** A model that passes A1 passes despite this input. A model that fails A1 on a composite grid has not shown that it cannot use faces, only that it did not on this format. A failure stops its claims under A1; it does not rule out a retest on a face-crop input, which needs its own preregistration.
- **Why now:** Written before results so that the reading cannot be fitted to them. Original metadata overstated frames used on these models; this does not change accuracy values.

## 2026-11-04 — A2 is a draft and is not frozen

- **Decision:** `research/PREREG_A2_DRAFT.md` stays a draft. The blind prompt failed coverage on Gemma 4 (lexicon mass 2e-5 to 0.004 against a proposed 0.05), so the prompt must change. Gate B2 is not lowered to fit.
- **Known deviation from Rule 2:** the 3-item smoke saw edited stimuli before predictions were committed. The draft discloses this and repeats the analysis without those items.

## 2026-11-04 — RQ1 same-clip analysis moves to AU families and matrix pattern

- **Decision:** Compare human and model cue use at AU-family level (brow raise AU01-02, brow lower AU04-09, smile AU06-12, mouth open AU25-26, others single) and by whole-matrix pattern. Single AU cells are exploratory. Written before any model matrix exists. Criteria were not loosened; the estimand changed because the old one is not identifiable.
- **Why:** Recovery failed the stated criteria in every condition. Planted observers reached r = .39 to .64 with the oracle, .64 even with perfect detectors, because AU pairs are collinear within cells (r .68 to .74) and estimates shrink strongly. Prior-only observers stayed essentially null but narrowly missed both false-positive limits (.060 and .014). Pooled analysis without a category control is invalid (.257).
- **Status of human results:** Provisional. 8 of 176 cells passed on both detectors, against about 3 expected from the null rate. Held-out gain over category alone is small: 0.49%.
- **Consequence:** The family-level analysis needs its own recovery run before freezing. Rule 8 is unmet for AU15 and AU20, which have one detector. Saved outputs contain human results only.
- **Disclosure:** While testing its loader, the job opened partial rotation files, for frame indices and a loader check. No saved output holds model values, but this was not a clean separation. RQ0 is computed only by the frozen script once all runs finish.

## 2026-11-04 — Family recovery criteria and permutation tests, fixed before the full run

- **Decision:** `research/scripts/same_clip_family_analysis.py` holds the family analysis and `FAMILY_RECOVERY_CRITERIA`, and `research/reproduction/same_clip_family_criteria.json` records their hash. Cue observers need median r at least .70 with the oracle and sign accuracy at least .90. Null observers need false positives at most .05 pointwise and .01 two-detector, and median absolute r at most .10 with a fixed matrix. These are the earlier thresholds.
- **New limits, set here:** at most .05 of null observers may flag any cell family-wise, plus two binomial standard errors of Monte Carlo error at the run's size. The pattern score needs size at most .05, with the same allowance, and power at least .80 for a human-like observer. There is no allowance on the pointwise or two-detector limits.
- **Changed method:** All tests are permutation tests, shuffling cue rows within category. This replaces bootstrap max-t, which flagged a cell on about 28% of null observers. The two-detector flag now tests the smaller of the two |t| jointly at .01, with matched shuffles. The old rule, both detectors pointwise with the same sign, is still reported. The test level does not move after a result.
- **Pilot before locking:** 40 prior-only and 40 actor-prior observers on synthetic cues only, with 199 shuffles, to check the code and cost: pointwise .046 and .048, family-wise .025 and .05. No measured-cue recovery ran before the criteria were written.
- **Human-like observer:** Its weights are a multiple of the estimated human family matrix, and the multiple is chosen only by matching held-out gain. Pattern power is estimated on pairs of simulated observers, so noise in the human reference counts.
- **If it fails:** Report the failure. Groupings, estimator and conditions are not changed to pass.

## 2026-11-04 — Decoder fault: restart A1 and rerun gates

- **Decision:** Pin cv2 4.11, add a preflight, discard A1 runs unread, resubmit all 30, and rerun gates. See amendment A1.1.
- **Alternatives:** Keep the runs and exclude about 49 affected clips. Rejected: the frozen design has 243 items, unread reruns are cheap on the free allocation, and decoders also differ slightly on unaffected clips.
- **Reach:** cv2 4.13 dates from 2026-03-16, before all baselines (July and August 2026). Image-dependent numbers in `RESULTS_REGISTRY.md` are affected, including accuracies, within-category correlations and the older shuffled 4AFC runs if run in this environment. Clean: AU measures, landmarks, and the readout ladder and delivery jobs run under cv2 4.11. Prompt-only findings stand.
- **Why the old gates missed it:** Nothing tested decoding, and frames were never hashed. Gate 3 of the delivery check caught it.

## 2026-11-04 — Gate 1 is read as the median; clips under 8 are a sensitivity check

- **Decision:** Keep gate 1 as median over clips of at least 8 tokens across the face, as run, with no change after seeing results. In A2 and A3, repeat each confirmatory analysis on clips at 8 or more, and report the share excluded. Gemma `single_face` has 90% of clips at 8 or more.
- **Molmo:** `single_full` passes narrowly (median 8.05, 55% of clips) and is not used for causal cue tests; `single_face` (median 12.96, 100%) is its route. Its native video route fails gate 1, so temporal cues are not testable on Molmo, and that is reported as a limit.
- **Why:** An every-clip rule would remove Gemma's only face routes because of a few small faces, whereas the sensitivity check exposes any dependence on them.

## 2026-11-04 — Family recovery result: partial pass, actor-prior null fails

- **Result of record:** `research/reproduction/same_clip_family_recovery.json`, 200 observers per condition, on the locked criteria (hash 4b0a8c3c). No threshold, grouping or test was changed after the run. Eight of 20 conditions pass. Three of the 12 failures are specs other than the primary one, and pooled was expected to fail.
- **Passes:** Family matrices recover planted observers at weights of 0.5 and 1.0 (r .80 to .85, sign .92 to .95).
- **Fails, cue side:** Weak sparse weights (0.25), dense weights, shared detector error (sign .896), and the human-like observer (r .78 passes, sign .83 fails, and .86 even with perfect detectors). So the matrix pattern is usable at human effect size, but the sign of a single family cell is not.
- **Fails, null side:** The actor-prior observer fails all three false-positive limits in the primary spec (.066 pointwise, .014 two-detector, .115 family-wise). Family-wise calibration is fixed for prior-only (.065) and not for actor-prior. Prior-only passes in the primary spec, narrowly (.0498, .0096), and misses pointwise at .053 with empirical noise and .051 with frozen ridge strength, which is the route a model takes.
- **Likely cause, not yet tested by a fix:** Cues are orthogonal to actor over all clips but not on the clips offering one option. The shuffle moves cues across actors and lowers that leakage (between-actor share of cue variance .27 against .33 observed), so the null is too narrow when actors differ in response.
- **Pattern score:** Separates a human-like observer from prior-only with power 1.0, and AUC at least .999, at half, one and two times the human effect. Size is .025 to .08. An observer sharing half the weights has power .55 to .97. This rests on simulated observers sharing true weights, and on independent noise.
- **Families plus singles:** All 10 cue-observer conditions fail (r .34 to .65). Adding seven singles at 11 cues loses what the families gained.
- **Consequence:** Human family cells stay provisional. Any change to the shuffle needs a new lock and new seeds, and this result stays on record as the failure of the first attempt.

