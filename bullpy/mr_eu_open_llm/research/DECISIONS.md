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


## 2026-11-04 — Full forwards on all models; no cached scoring

- **Decision:** `CACHED_MODELS` is empty. Gemma 4, Qwen3-VL and Molmo2 all score by full forwards, so every model uses one scoring path. The 0.25 nat parity tolerance was not changed, and it no longer gates anything.
- **Evidence:** `research/reproduction/cache_parity_qwen3vl_bfloat16.json`, 3 clips on `single_face` and `video_face`. On the blind readout the worst gap per sequence was 0.127 to 0.249 nats, and the mean was 0.004 to 0.011. All of it sat at prefix length 1, on second tokens of multi-token lexicon words such as " sickened", " unem" and " melancholy"; length 2 was at most 0.0001. On MCQ the gap was at most 0.000001. Two repeats of a full forward differed by 0.0. Values cluster at one or two bfloat16 steps at this logit scale, which suggests rounding, but float32 was not awaited and that cause is unconfirmed.
- **Gemma 4:** the cached path raised a CUDA device-side index assert (job 39987786), which poisons the process, so a try and fallback cannot be used.
- **Molmo2:** the cached path was not tried.
- **Alternatives:** (a) Keep the cache on Qwen, which passed the tolerance and saves time; (b) wait for float32 and decide then.
- **Why:** Full forwards are exact where the cached path is not. The gap falls on lexicon terms, so it could differ between stimulus conditions and reach the contrasts under test. Identical scoring across models is worth more than the time saved.
- **Cost:** Qwen stays at about 0.37 s per forward. Cost figures in the A2 and A3 drafts use full-forward timings.

## 2026-11-04 — Cache parity under float32: the gap is rounding, decision stands

- **Result:** `research/reproduction/cache_parity_qwen3vl_float32.json`, same clips and routes as the bfloat16 run. Blind readout: worst gap per sequence 0.00005 to 0.00013 nats, mean 0.000003 at most. MCQ: 0.0. Repeated full forwards: 0.0.
- **Reading:** The Qwen position fix is correct. The bfloat16 gap of up to 0.25 nats was rounding between two compute paths, not a logic fault, which answers the cause left open in the entry above.
- **Decision:** Unchanged. Runs use bfloat16, where the gap is real, so all models keep full forwards.

## 2026-11-04 — Second attempt at family calibration, locked before it runs

- **Decision:** Run a post-hoc second attempt with a within-actor shuffle, prompted by the actor-prior failure of the first. It is not a replacement. First-attempt outputs stay unchanged and on record, and their hashes are in `research/reproduction/same_clip_family_a2_criteria.json`. New outputs carry the prefix `same_clip_family_a2`.
- **Fixed:** Thresholds (hash 4b0a8c3c), the family-wise limit of .081, families, pattern score, estimator, conditions and test levels all stay as they were. Seeds are new.
- **One change to a condition's route:** Frozen-lambda conditions now give each input the ridge strength fixed on the human matrix for that input, as a model would take it. The first attempt used the composite value on all three inputs. The same conditions also rerun on the frames models saw, where the human values differ.
- **Why within actor:** A pilot on synthetic cues only (60 observers per condition, 199 shuffles) compared three schemes. Within-actor gave actor-prior .041 pointwise, .005 two-detector and .083 family-wise, with power close to the first attempt. Within-cell was broken (.68 to .77 family-wise). Sign-flipping by actor was calibrated but lost power. No measured-cue recovery of this scheme ran before this lock.
- **Correction to the earlier stated cause:** Within-actor shuffling does not restore the between-actor share of cue variance (.27 against .33 observed), yet the pilot improved. That share alone therefore does not explain the first failure.
- **Order:** Human cells are read only after the recovery result is written. If actor-prior still fails, single family cells stay provisional, and the confirmatory RQ1 unit stays families and pattern score.


## 2026-11-04 — Shared option-free readout: candidate 3 on rung 1

- **Decision:** Prompt "In one plain word, with no formatting or punctuation, what is this person feeling?" with prefill "They feel", on all three models. It sets `BLIND_PROMPT` and `BLIND_PREFILL`.
- **Rule:** All three models pass rung 1, so rung 1 is the highest shared rung. Among candidates passing on every model, the ladder takes the one with the highest lowest median lexicon mass (`choose_shared`). A candidate is a (prompt, prefill) pair, so the rule picks one prefill and no deviation is needed.
- **Calibration:** 24 CREMA-D clips outside the analysis set, no labels or votes read. All candidates shown below pass coverage and image sensitivity where a number is given; "fails" is the model that failed coverage.

| Candidate | Gemma 4 | Qwen3-VL | Molmo2 | Lowest |
|---|---|---|---|---|
| 1 "in one word" / They feel | fails | .946 | .299 | Out |
| 2 "one plain word" / They feel | .819 | .975 | .640 | .640 |
| 3 "In one plain word..." / They feel | .803 | .977 | .680 | .680 |
| 4 "In one plain word..." / This person is feeling | .843 | .966 | .660 | .660 |
| 5 "Finish the sentence" / They feel | .405 | fails | .549 | Out |
| 6 "Finish the sentence" / They look | .615 | fails | .490 | Out |

- **Margin:** .02 over candidate 4, on 24 clips, so the choice between 3 and 4 is close. It reversed once Molmo reported; Gemma and Qwen alone gave candidate 4. Candidate 3 is also Qwen's and Molmo's own best; Gemma's own best is candidate 4.
- **Limit:** Molmo keeps .46 lexicon mass on a grey image against a median of .68 on clips, so a large part of its mass is not image-driven. Its image sensitivity still passes (100% of pairs, 100% against grey). Gate B's image checks remain the test of this.
- **Alternatives:** Per-model prompts, which the plan allows only if no rung is shared.

## 2026-11-04 — Second attempt fails: actor-prior still uncalibrated

- **Result:** `research/reproduction/same_clip_family_a2_recovery.json`, 200 observers per condition, on the lock in commit 6ec22f4. The within-actor shuffle did not fix the failure and made the null side worse. Seven of 20 conditions pass, against eight in the first attempt, and no null condition passes.
- **Null observers:** Actor-prior fails all three false-positive limits (.064 pointwise, .014 two-detector, .17 family-wise, against .066, .014 and .115 in the first attempt). Prior-only, which passed before, now fails at .053, .011 and .09. Its family-wise rate reaches .145 with perfect detectors.
- **Power:** Mostly unchanged. Sparse 0.5 keeps r .80 and sign .92; two-detector detection falls from .157 to .143 and family-wise from .053 to .039. Shared detector error now passes narrowly (r .78, sign .906). Human-like still fails on sign (.815). The change is not too conservative: it errs the other way.
- **Pattern score:** Still separates human-like from prior-only at all three strengths for families (power at least .995, AUC at least .999, sizes .035 to .05). At 1x and 2x a different-cues observer reaches .105 and .115, above .081. That observer is outside the pass rule, but it is a real weakness.
- **The pilot did not predict this.** Within-actor looked calibrated on 60 synthetic-cue observers (.017 and .083 family-wise) and is not calibrated on measured cues. A synthetic-cue pilot is not evidence of calibration here. The cause is not established.
- **Frozen ridge strength:** The route works as designed. `frozen_route` takes each input's value from the human matrix and never chooses on the observer, and tests confirm this. Calibration on that route does not pass. All frames, values 3, 3 and 10: prior-only misses pointwise at .0501 and actor-prior fails. Model frames, value 10: sparse 0.5 fails narrowly (r .75, sign .898), and prior-only fails family-wise at .085 and actor-prior fails.
- **Comparison of the two nulls:** With the primary spec, the first attempt has the lower family-wise rate on every null condition. Pointwise rates differ by little, and the second is lower on three (.048 against .053 with empirical noise, .0501 against .0509 frozen, .064 against .066 actor-prior). Where the spec has no actor control the second is much better (actor-prior family-wise .115 against .695).
- **Decision:** No third attempt is proposed. Single family cells stay provisional and are not confirmatory. The confirmatory RQ1 unit stays families and pattern score. The first attempt's within-category null stays the one of record for the primary spec.
- **Human cells, read after the recovery result:** The two nulls agree on five two-detector family cells on all frames: Excited, None of the above and Surprised with mouth open, Sad with brow raise, and Worried with brow lower. Family-wise, only Surprised with mouth open holds under both. Worried with brow lower holds only under the first, and Excited with mouth open only under the second. Coefficients are identical; only the null differs.


## 2026-11-04: RQ1 inference after two failed calibration attempts

Decided before any model matrix exists or any rotation result is read. Two attempts at family-wise calibration failed for actor-prior observers (.115, then .17, limit .081). There will be no third attempt: a further scheme chosen after two failures would be a search for a null that passes.

- Confirmatory RQ1 unit: the family pattern score, on the model-frames route with human lambda 10 (3 inputs frozen from the human matrix). Null of record for the primary spec: attempt 1.
- No model or human claim rests on a family-wise cell or a single family cell. Those are descriptive, reported in full and labelled exploratory.
- The score is not specific to the human cues. The different-cues observer reaches size .055 to .085 under attempt 1 and up to .115 under attempt 2. A positive score therefore supports "aligned with the human pattern", not "uses the same cues". That claim needs the RQ2 stimulus tests.
- Model frames, frozen lambda, sparse 0.5 narrowly fails recovery (r .752, sign .898). The cause of the calibration failure is not established.

## 2026-11-04: Correction to the RQ1 inference entry

The entry above names the pattern score on the model-frames route as confirmatory, but that route has no pattern recovery: `pattern` is empty in `same_clip_family_a2_recovery_model_frames.json`. All pattern sizes and power quoted so far are from all frames. The confirmatory status is conditional until that test exists.

Rule set before the test runs: use the existing criteria (size .05 plus 2 Monte Carlo SE, power .80), the attempt 1 null of record, human lambda 10, unchanged families. If the score passes on the model-frames route, it is confirmatory as stated. If it fails, RQ1 on natural clips is descriptive only, and the confirmatory test of cue use rests on RQ2. A failure does not license a new score, route or null.

## 2026-11-04: Pattern score on the model-frames route, result

The test the correction above asked for: `research/reproduction/same_clip_family_route_pattern_within_category.json`, 200 observers, 199 shuffles, families, run through `frozen_route` with lambda held at the human value of 10 on all three inputs. Criteria hash 4b0a8c3c. The pass rule is read on the composite input, as in earlier pattern runs.

- **Attempt 1 null of record:** size passes at all three strengths (prior-only .055, .03, .05; actor-prior .035 to limit .081). Power is .775 at 0.5x, which fails .80, and 1.0 at 1x and 2x. Different-cues size is .05, .03, .055, all within the limit. Half-shared power is .34, .83, .97.
- **Attempt 2 null, sensitivity only:** same pattern of results, and 0.5x passes at .815 with prior-only size .08. This does not change the verdict on the null of record.
- **Unresolved reading:** The rule above says "passes" without naming a strength. Read at the calibrated human effect (1x), the score passes. Read as all three strengths, it fails at 0.5x. Held-out gain selects 1x (simulated .0025 against .0038 real, next step .011). Split-half reliability puts the real effect near 0.75x (simulated .18 at 0.5x, .28 at 0.75x, .38 at 1x, real .277). Power at 0.75x was not run. Which reading applies is not settled here, and must be fixed before any model matrix exists.
- **Loss from all frames:** Power at 0.5x falls from 1.0 to .775, so all-frames figures overstated this route.
- **Limits:** Power of 1.0 assumes identical true weights and independent item noise. A positive score still supports "aligned with the human pattern", not "uses the same cues".
- **Loader:** `model_route` connects `frozen_route` to the rotation loader, for one clean complete run. It refuses partial runs, and stops if any clip has no usable response. No rotation or model result was opened.


## 2026-11-04: Strength rule for the RQ1 pattern score, and missing clips

Decided after the model-frames pattern results (`same_clip_family_route_pattern_within_category.json`: size passes at 0.5x, 1x and 2x; power .775, 1.0, 1.0) and before any model matrix exists or any rotation result is read. The earlier rule did not name a strength, so this closes a gap the coordinator left open. It is disclosed as written with the 0.5x, 1x and 2x results known.

- Reading only the calibrated 1x row would favour a pass, because real-effect estimates sit at or below it: held-out gain says 1x or a little above, split-half reliability says about 0.75x.
- Rule: run 0.75x once, under the same criteria, null of record, lambda and families. RQ1 is confirmatory only if size and power pass at both 0.75x and 1x. If 0.75x fails on either, RQ1 on natural clips is descriptive only. This can only downgrade the 1x reading.
- 0.5x is below both estimates. Its power of .775 stays reported as a limit: an effect that small would often be missed, so a null model result does not show absence of alignment.
- Half-shared power is reported with the result, since a model sharing only part of the human weights is much harder to detect.

Missing clips in the model run: a clip with no usable response is dropped from both the model and human matrices for that model, lambda stays frozen, and null draws are restricted to the kept clips. Every dropped clip is listed by label and actor. If more than 5 percent of clips are lost for a model, that model's RQ1 result is descriptive only.

## 2026-11-04: Strength rule result: the family pattern score is confirmatory for RQ1

The single 0.75x run the rule above set: `research/reproduction/same_clip_family_route_pattern_075x_within_category.json`, 200 observers, 199 shuffles, attempt 1 null of record, lambda 10 on all three inputs, families, criteria hash 4b0a8c3c, new seeds, scale fixed at the 1x value.

- **Verdict:** Pass. Size and power pass at both 0.75x and 1x, so the family pattern score on the model-frames route is the confirmatory RQ1 unit, subject to the 5 percent clip rule per model.
- **0.75x:** Power 0.995 against .80. Size .035 for prior-only and .06 for actor-prior against .081. AUC .999.
- **1x, earlier run:** Power 1.0. Size .03 and .035.
- **Different-cues size:** .055 at 0.75x, within the limit. The claim stays "aligned with the human pattern", not "uses the same cues".
- **Half-shared power:** .60 at 0.75x and .83 at 1x. A model sharing half the human weights is missed about 40 percent of the time at 0.75x. Power .775 at 0.5x also stands as a limit. A null model result does not show absence of alignment.
- **Attempt 2 sensitivity:** Same verdict on the composite input (power .995, size .04 and .06, different-cues .065, half-shared .63). On the second detector alone, actor-prior size is .085, above the limit. That input is not part of the pass rule.
- **Sequence disclosed:** The rule was tightened after 0.5x, 1x and 2x were known, and this pass is on 200 observers at one strength. Power here assumes identical true weights and independent item noise.
- **Still exploratory:** Single family cells and family-wise cells. Actor-prior calibration for cells is not fixed, and sparse 0.5 still narrowly fails cell-level recovery on this route.
- **Clip rule, as built:** `model_route` chooses lambda once on the human matrix over all clips. A clip is dropped only if no rotation gave a usable response. A clip usable in some rotations stays and is listed under `partial_rotation_clips`, because the rule covers only clips with no usable response, though such a clip keeps some position confound.

## 2026-11-04: Partial clips are dropped in the model route

Decided before any model run. A clip counts as usable in `model_route` only with a usable response on all 5 rotations, matching A3's rule and keeping position confound out. Partial clips leave both matrices, count towards the 5 percent limit, and stay listed with their rotation count. This replaces the last bullet of the strength rule result entry above.

