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


## 2026-11-04: Vision gates G2 to G4, Gemma and Molmo (Qwen pending)

Run under cv2 4.11 per A1.1. Files: `research/reproduction/vision_gates_{gemma4,molmo2}.json`. Qwen full job 39981392 is queued, so Qwen has no gate result. Its 2-clip smoke is not a result, and anything below marked provisional comes from that smoke.

- **Gemma 4:** G2, G3 and G4 pass.
- **Molmo2:** G3 passes. G2 fails at bfloat16 (worst relative L2 .0406 against .01; float32 is exactly zero). G4 fails: monotonic, 2.28 nats at 100 percent, but flip rate 0 against .25.
- **Both failures stand.** Tolerances and thresholds were fixed before any model ran and are not changed. G4's "large" thresholds were set by the implementer, not by the brief, and Molmo's G4 failure rests on the flip part alone. That is disclosed, not a reason to relax it.
- **Consequence:** A2 as drafted requires G2 to G4, so Molmo is not eligible for the stimulus suite. Any route back in needs a dated rule written before new data, and must be disclosed as such.
- **G2 explanation is untested.** Two of 80 comparisons are non-zero; the gates run attributes both to foreign option sets of different length (SF12, FF20). The stored JSON holds per-layer aggregates only, so this attribution cannot be rechecked from the file. Sequence-length-dependent bfloat16 rounding fits, since masks block image-to-text and float32 is zero, but it is an inference. Diagnostic if Molmo clears RQ0: rerun those clips at bfloat16 with padded lengths, saving per-comparison rows. Deferred because the result only matters if Molmo survives RQ0.
- **Molmo answers "None of the above" on 40 of 40 gate clips**, accuracy 0. Logprobs still move toward the donor (+3.41 nats), so image content reaches the output distribution but not the argmax. Read RQ0 and any Molmo argmax measure with this in mind.
- **Gemma layers 24 to 41 reuse keys and values stored at layers 22 and 23.** Image-position patches at L24 or later change logits by exactly 0, so they are guaranteed false nulls. Patch at L6 to L23 only. No registered result is affected: the only registered ablation is Qwen L4, and `scripts/path_patching.py` output appears in neither the registry nor the paper. The script has no guard, so its default of all saved layers must not be used for Gemma.
- **Vision cuts:** Gemma needs both routes cut at L0 (.000 remaining; 42 percent from L1). Molmo from L1 leaves a mean of 11.7 percent. Qwen after LM layer 2 leaves 31 percent at bfloat16 and 46 percent at float32 (provisional). No cut at a later layer is a complete cut on any model.
- **Pilot ablation caveat:** Qwen L4 (5 of 180 flips) patched after LM layer 2. If the Qwen gate confirms this, that null covers an incomplete path and says less about the axes than it appeared to.


## 2026-11-05: Vision gates G2 to G4, Qwen

Job 39981392, run under cv2 4.11 per A1.1, script sha256 6327b6f8, the same script and rules as Gemma and Molmo. File: `research/reproduction/vision_gates_qwen3vl.json`, sha256 2f38aafb. This replaces the provisional Qwen line in the Gemma and Molmo entry.

- **Qwen3-VL:** G2, G3 and G4 pass at bfloat16 on all 40 clips. G2 and every G3 check are exactly zero. G4 is monotonic, 9.30 nats at 100 percent (95% CI 7.62 to 10.86), flip rate .55 against .25. The 100 percent swap equals a native donor forward exactly.
- **float32 did not run.** The stage ran out of GPU memory (74 GB in use), even limited to 8 clips of at most 6 frames, so no float32 G2 or G3 result exists. The job still exited 0, and the file's `verdict` lists bfloat16 only, so `pass: true` here covers less than it does for Gemma and Molmo. The script treats float32 as a numerical check on G2 and G3 and keeps the bfloat16 result on memory failure, and A2 names no dtype, so the pass stands, with this disclosed.
- **No float32 rerun.** Its purpose was to tell rounding from a real leak in a non-zero bfloat16 result, as for Molmo. Qwen's bfloat16 result is exactly zero, so there is nothing to tell apart. The evaluation dtype is bfloat16.
- **Consequence:** Gemma and Qwen are eligible for A2 on the gates. A1 is still required for each model.
- **Vision cuts, 12 clips, bfloat16:** Cutting image positions after LM layer 2 leaves a mean of 27 percent of the image effect (median 24, range 18 to 65). The smoke figure was 31 percent. The smoke's float32 figure of 46 percent has no full-run counterpart and stays unconfirmed. Cutting either vision route alone leaves about 100 percent (main 1.03, deepstack 1.03; single deepstack levels .98 to .99), so the routes are redundant and a cut must take out both.
- **Pilot ablation:** This confirms the caveat. Qwen L4 (5 of 180 flips) patched a path that still carries about a quarter of the image effect, so that null says less about the axes than it appeared to. It remains a pilot.
- **Descriptive:** On the 34 clips with the donor label in the options, a full swap moves logprob +14.55 nats toward the donor label. Gate-sample accuracy falls from .275 on the own clip to .05 on a full swap. These 40 clips are a gate sample, not an RQ0 result.
- **Qwen A1 runtime:** Arrays 39982137 and 39982138 have a 12 h limit, and `scripts/evaluate.py` cannot resume, so a timeout loses the task. The only timing on this configuration (composite grid, 16 frames, 20 sequential samples, 243 items) is the August baseline, job 33520827, at 7 h 47 min, under the older decoder. A Slurm limit cannot be raised again once lowered, so it stays at 12 h. Qwen has not run this rotation path before; only Gemma and Molmo have.


## 2026-11-07: A2 frozen before A1 is read

The user accepted the four open choices and gave full autonomy. An independent audit found 14 faults before freezing, among them B5 duplicating B4, most of the null rule uncoded, and a shard resubmission that would have written all items to one file. A second audit found that the A3 path would have made every verdict Inconclusive, and that the dose check passed on noise. All are fixed; the appendix to the draft lists them.

- **Why freeze now:** A1 is waiting on the queue, and freezing before reading it is stricter than the draft required.
- **Consequence:** 8 of 16 contrasts cannot reach Null, or 10 of 16 on Qwen, and 6 of 14 predictions. Those cues read as a match, an opposition, or Inconclusive. A2 cannot show that a model ignores them.
- **Recovery** was rerun on the frozen code and is unchanged. It shows that contrasts are recovered at large effects, not power at realistic ones.


## 2026-11-07: Natural-clip RQ1 under each RQ0 outcome

Decided before any A1 result was read. The natural-clip RQ1 route uses A1's composite-grid rotations, and no rule covered a model that fails A1. This closes that gap.

| A1 | A3 | Natural-clip RQ1 |
|---|---|---|
| Pass | Pass | Confirmatory, subject to the 5 percent clip rule |
| Fail | Pass | Computed and reported as descriptive. The grid limited that input, so the pattern score cannot carry a claim |
| Pass | Fail | Computed and reported as descriptive, pending the inspection the A3 table requires |
| Fail | Fail | Computed for the record, with no claim. No RQ1 to RQ5 claims for that model |

`model_route` runs once per model in every case. Rows 2 to 4 change only how the score is read. A later analysis on `single_face` would be exploratory, or needs its own amendment before data.

RQ0 text says no later RQ runs on a failing model. The frozen A3 table is the later and more specific rule, so it governs.


## 2026-11-07: Source check after the A2 freeze

A full read of the manuscript's references found that P7, P12 and P13 cited sources that do not carry them, that no cited source tests shuffling, and that Gold et al. 2013 and a VLM study point the other way. A2 has not run, so this is logged as a pre-data deviation in A2.

- **Decision:** Predictions, family and code stay frozen. Changing a prediction after the freeze would be a worse fault than a weak one, and the evidence grades are corrected openly instead.
- **Why not drop P12 and P13:** They stay in the family, and removal would change the multiplicity correction after the freeze. Their reading is limited by the deviation.
- **Also:** The audit claim is softened to "several", since 3 of 8 is not most. A no-image control in Hallmen et al. was missed in earlier notes. Nature Human Behaviour cannot take this as a Registered Report, so it would be an Article with a reporting summary.

## 2026-11-07: Freeze A4 before A1 is read

The queue allowed it, so A4 was frozen ahead of A1, which keeps the choice of statistic and thresholds independent of RQ0.

- **Decision:** Residual CFA stays confirmatory and the V1 pilot's thin pass is disclosed as a family flag. Six dependent scripts are pinned by per-file hash in the amendment, not added to `DESIGN_FILES`, because that would change the design hash and need a further recovery run for no gain.
- **Why not count recovery as clean:** Two accuracy cells at conversational 0.25x are exempt, at .85 and .62, so the amendment makes no accuracy claim there.
- **Error caught at the freeze:** a commit gave run 3's file the hash of run 4's. It was corrected before the tag.

## 2026-11-07: G7 becomes a scoped limit, resting on Stage 0

Draft A7 found that RQ3's literal test cannot fail: under a complete cut the output is the same for every image, so a cue effect there is zero by construction. Depth inside the language model is not testable on either eligible model, and no site has yet been shown complete on `single_face` or on the blind readout, since G2 to G4 ran on the legacy input.

- **Decision:** RQ3 is reported in the main paper as a scoped limit, resting on A7's Stage 0 table (about .71 GPU-hours, upper bound). The route arm is an optional supplement. This uses the allowance in `GOALS.md`.
- **Why not on cost:** compute is not the reason. The reason is that the literal test is vacuous and a route split has no human reading.
- **What would change this:** a pass on Stage 0's region test would give a non-vacuous cue-level answer, and G7 would be reopened by a dated entry.
- **Open:** whether a block of a complete cut meets G7's wording goes to A7's audit. If it does not, A7 is Stage 0 plus "not testable". A7 is not frozen, and its recovery is specified, not run.
- **Note on an earlier reading:** the stored `fraction_of_image_effect_left` is not a share, so ".93 remains" is wrong on Gemma, and reading routes as redundant fits Qwen but is not proved by the stored rows.

## 2026-11-08: A6 keeps its named results and gives up its cue-absent nulls

A6 recovery run 4 (design `c1961048...`) fails only on the Stage 3 nulls and on adjacency over all pairs. The named results recover at .91 or more with no false positive over .05.

- **Decision:** the nulls Not this cue and No tested feature stay in the table, but a model that reaches one is reported as Inconclusive on the cue question. Adjacency on all pairs is descriptive. The partner check stays, with the audit's Happy rule reported as a sensitivity.
- **Why not loosen the margin:** that would move a threshold after seeing a failure. The nulls fail even at no actor-by-region noise (.62 and .70), so the shortfall is in the design's power, not the noise setting.
- **Why run A6 at all:** a shared error that carries or opposes a named cue is recovered at .99, which is the cue-targeted test G8 asks for.
- **Still before freeze:** the cluster items (geometry and weight pins, first-frame AU check, decoder preflight, a mechanics run) and the proposed values in the block.

## 2026-11-08: A5 withholds balance verdicts in small families and keeps its power failures

A5 recovery run 3 (design `228bdf48...`) failed five checks. Recalibrating the balance-zero observers on B itself, rather than on mean Phi, removed a small real offset but left the B false positives in one- and two-cell families at .08 to .12 under portrayal-gain SD .5. An oracle default changed nothing, so the cause is skew from gain heterogeneity in small families, not dependence between actor rows.

- **Decision:** `min_balance_cells` = 4. Below it, Face, Default and Even are withheld and the verdict is Inconclusive (`balance_small_family`). Four is the only size tested at which every true-null rate was at or under .054; three was not tested, so it is not allowed.
- **Run 4** (design `df4583f0...`): every false-positive check passes. Two power checks fail at signal 2, where Inconclusive reaches .226 at Phi .5 and .398 at Phi .7, against .20. Both fail only into Inconclusive (`balance_unresolved`), with no wrong named verdict over .10. They stay recorded as failures, with the power table.
- **Why not move a limit:** .07 and .20 are unchanged. The scope rule changes what A5 may claim, not any threshold, and it was set from evidence written down before any A2 cell was read.
- **Cost:** if RQ0 or a gate removes a cell from a family, that model gets no balance verdict from A5.

## 2026-11-08: A4 human ceiling uses a related estimator, disclosed not changed

Writing Supplementary Note 1 found that A4's human noise ceiling uses pooled fits at ridge 1.0 for every dataset (`a4_analyse.py:360`), while the human target is the mean of per-subject fits at the selected ridge (0.1 for basic). So R_H is the reliability of a related estimator, not of the target itself. A4 is frozen, so the code stays. The corrected values are reported as approximate, with this stated in the methods and in Supplementary Note 1, beside the uncorrected values that the decision rule uses.

## 2026-11-08: EU-Emotion permissions settled

The user confirmed that the team controls EU-Emotion and its creators are collaborators, with full permission for public item-level norms and for in-memory edits of clips. NEEDS_INPUT item 7 is closed. The norms file stays public. Item IDs may now appear in public files. Videos, frames and edited clips stay out of the public repo anyway: they are large, and keeping the rule simple avoids mistakes with the other stimulus sets (RMET, FACSHuman renders), whose licences differ.

## 2026-11-08: A2's frozen sign-flip test is liberal under one kind of skewed null; disclosed, with a fixed stricter-threshold sensitivity

After A5's small-family finding, the frozen A2 test (actor sign-flip, Westfall-Young step-down maxT within model and route, exact 2^17 flips, code at tag `a2-freeze`) was checked under the same kind of null. The check ran synthetic cells in A2's format over the real 243 items and 17 actors, 1000 replicates per cell (`research/scripts/a2_gain_skew_sim.py`, output `research/reproduction/a2_gain_skew_sim.json`, single-contrast check `research/scripts/a2_gain_skew_single_column.py`). No A2 cell had been run or read.

- **Exact where it should be:** with no per-actor gain, family-wise error per route is .040 to .055. Gain alone (`sharp`) and symmetric gain-scaled noise (`hetero`) stay at .043 to .055.
- **Liberal under a skewed zero-mean null with its long tail opposite the prediction:** each edit removes evidence in proportion to the actor's log-normal gain, offset so the mean actor effect is zero. Per-route family-wise error is then .091 (`single_face`) and .098 (`video_face`) at gain SD .3, and .137 and .130 at gain SD .5, for the stronger skew (k = 2). For the weaker skew (k = .5) it is .054 and .075 at gain SD .3, and .082 and .093 at .5. With the long tail on the predicted side the test is conservative (.016 to .040).
- **Why maxT does not help:** every contrast shares each actor's gain, so the columns are strongly correlated and the step-down adds little protection. A2's families are fixed at 10 and 6 contrasts, so A5's small-family scope rule does not transfer.
- **Decision:** A2 stays frozen and its verdicts stay primary, as written. This is disclosed as a limit in the methods. The sensitivity below is fixed now, before any A2 cell exists: for every Supports verdict, the paper also says whether the contrast's step-down p is under .01. In the worst simulated case (k = 2, gain SD .5), per-route family-wise error at .01 is .046 and .044, while .025 and .02 do not hold (.095 and .080). A verdict that holds only at .05 is reported as not robust to this null. The two-sided Opposes test was not simulated; it takes absolute t, so it always sees the liberal tail, and the same .01 reading is applied to it.
- **Not adopted:** a studentized bootstrap-t brought the single-contrast rate from .106 to .074 at 8 actors, which is still over .07. It is not used.
- **Unknown:** whether real portrayal gain is skewed this way, or this large, is not known. The simulation sets bounds; it does not estimate the size.

## 2026-11-08: A5 balance verdicts become descriptive; evidence classes carry the confirmatory claims

Run 6 (design `fd20495b...`) tested a per-actor gain-heterogeneity guard. The estimator tracks the true spread (mean .34 at gain SD .3, .55 at .5), and a guard at .4 would have brought the gain .5 shared-family false-positive rate to .022. But it fired on 12% to 66% of cells at the validated gain .3, mostly because 17 actors give a noisy SD even for a perfect estimator. So it was dropped, as the stated rule required, and the estimate is reported only. Run 6 otherwise matches run 5: `shared_family_gain0.5` fails (.094 any column, .072 own column), and three power checks fail into Inconclusive.

- **Decision:** option (b) of the draft. Face, Default and Even leave A5's confirmatory claims and are reported descriptively, beside B, Phi and the gain estimate, labelled as not error-controlled under skewed portrayal gain. The evidence classes carry the confirmatory claims. Every failure stays on record.
- **Why not freeze with an exception:** the draft says a false-positive failure blocks the freeze, so freezing anyway would make the rule mean nothing. Calling the failed observer unjudged now, after it failed, would be post hoc.
- **Why not more units:** A2's data has 17 actors.
- **Order:** decided before any A2 cell exists.

## 2026-11-08: A1 errored trials, settled before any A1 file is read

A dry run of the post-A1 runbook on synthetic files found that the frozen `research/scripts/analyse_rotation.py` raises `KeyError` when a trial errored, because `scripts/evaluate.py` writes such trials with `prediction` and `correct` set to None and no `response_distribution`. Whether any real trial errored is not known, and nobody has looked.

- **Primary estimand:** unchanged. `bool(None)` is False, so an errored trial already scores as wrong, as A3's text says A1 does.
- **Secondary p_target:** an errored trial counts as 0 mass on the target, which is the same no-vote treatment. This is done by a wrapper, `research/scripts/analyse_rotation_run.py`, which checks the frozen script's hash, supplies an empty distribution, and counts the errored trials per model, condition and rotation in `rotation_errored_trials.json`. With no errored trials, its output is byte-identical to the frozen script's.
- **A trial missing between files** stays a hard stop. It means a job problem, to be fixed by a rerun, not by analysis.

## 2026-11-08: A7 audit closes block admissibility (pre-data)

The A7 audit (ROADMAP 0.12) took the stricter reading of the question left open on 2026-11-07. A block of a complete cut, or a token set, is not itself a cut and does not meet G7's wording. So RQ3 rests on the complete exit cut alone. The route and token arms survive only as an optional secondary decomposition of a complete exit cut and are never reported as an RQ3 result.

- **Complete cut, numeric:** all 24 calibration clips and all 106 unrounded word logprobs within 1e-3 nats of the image-free reference. One word on one clip over the limit fails the site.
- **Not testable:** any site not run, failed, or without output counts as not shown, and RQ3 is reported as "not testable" there.
- **No look at A2:** any Stage 1 scope decision must be in this file before any A2 analysis output is opened, or Stage 1 does not run. Extra runs conditioned on A2 verdicts are exploratory only.
- **Endpoints:** every cell of an item comes from the same A7 job, unrounded; A2's stored cells are a descriptive cross-check.
- **Timing:** A7 must be frozen (tag `a7-freeze`) before any full A2 cell is read, or A7 becomes exploratory. Its code, tests and CPU recovery are local work to finish before then.
- Design hash after the audit: `9e40cfcf...50bb09`, checked independently by the coordinator.

## 2026-11-08: A6 local items closed (ROADMAP 0.11, pre-data)

- **Recovery run 5** reran on the new binding hash (`41aa6ee5...`) after the code change and matches run 4 in every table. It still fails on Other cue, One-way other cue, Spread and Global face, as before. The decision already taken on 2026-11-08 stands: the design runs with those observers named as unrecoverable, and their verdict rows carry no claim. No threshold moved.
- **Stage 3 nulls** are reported as Inconclusive (`stage3_null_no_claim`); `verdict()`, which recovery judges, is unchanged.
- **Adjacency on all pairs** is descriptive; its failure (.148 and .152 against .10) stays on record under `checks_descriptive_not_judged`.
- **[proposed] values** confirmed as written, no number changed: `shared_margin`, `cue_margin_fraction`, `outcome_reference`, `reference_check`, `text_neighbour_rank`, `first_frame_au_ratio_max`. Confirmed by the coordinator before any A6 model output exists. The mid-face landmark list stays proposed until the cluster check.
- **Mechanics run** goes before the freeze, the conservative order.
- Design hash `1554afc0...60847`. Open items are cluster only, listed in the draft's Freeze section.

## 2026-11-08: A7 Level 2 recovery failures ruled (pre-data, before freeze)

A7 recovery Level 1 passed; Level 2 failed 5 of 18 judged observers (numbers in `PREREG_A7_DRAFT.md`, Recovery). Ruling, following the A5 precedent of narrowing a claim rather than moving a threshold:
- **Riders (Additive, Synergic, Opposed) become descriptive.** The carrying verdicts recovered at 1.0 and stay confirmatory. The rider failures come partly from recovery drawing independent noise for TE, PE_A and PE_B; that construction is not changed to make them pass.
- **No effect to carry stays confirmatory with the TE margin unchanged.** It fails safe (Inconclusive, never a wrong named verdict), so its recovery reach is reported beside it. Widening the equivalence margin would make a null claim easier and was rejected.
- Freeze still needs: an independent code-against-text audit, a `## Code hashes` section, the design hash restated, then tag `a7-freeze`, all before any full A2 cell is read. Ladder field names used by the region test are cluster items.

## 2026-11-08: A7 frozen (tag `a7-freeze`)

A7 is frozen before any full A2 cell is read. Recovery was rerun on the pinned code: Level 2 equals the earlier run in every value, Level 1 differs only in timing. The Level 2 rulings above apply unchanged. Code hashes are in the draft's "Code hashes" section and are checked by `a7_design.check_code_pins` before any A7 analysis runs. The recovery outputs are pinned in `scripts/reproduce.py`. Open items are cluster only: Stage 0, ladder field names, image-token counts, code sync.

## 2026-11-09: CFA renamed Cue Effect Alignment (CEA)

"Causal Feature Alignment (CFA)" is the name of an unrelated WACV 2024 training method (Venkataramani et al.). On the user's instruction (NEEDS_INPUT 9, option b), the paper, the theory note, the plans and the figure labels now say Cue Effect Alignment (CEA). The quantity, its code and every threshold are unchanged. Frozen files are not edited: PREREG_A2, A4 and A7, the pinned scripts (`scripts/a4_*.py`, `scripts/a7_*.py`), code identifiers (`SESOI_CFA`, `cfa_identifiability.py`) and JSON keys (`cfa`, `cfa_mean`) keep CFA, and the Methods says so where the estimand is defined. Dated records (DECISIONS, LIT_CHECK, citation_check, RESULTS_REGISTRY) keep the name they were written with. The file `research/theory/causal_feature_alignment.md` keeps its name so frozen references resolve.

A search for the new name (2026-11-09, web, standard) found no prior use of "Cue Effect Alignment". It found a close relative to read before submission: CALM (arXiv 2605.25256), which scores LLM–human process alignment as the cosine between cue-weighting coefficients estimated from observed decisions. That is the observational counterpart of CEA, and Proposition 4 and Theorem 1 of the theory note are the reason CEA uses randomised cues instead. Snippet level only; not cited until read (LIT_CHECK recheck 2026-11-09).

## 2026-11-09: Qwen A1 array throttle raised from 2 to 5

The user reported the `cas_*` jobs finished and asked for the free capacity to be used. `scontrol update ArrayTaskThrottle=5` on 39982137 and 39982138 (our own jobs; no resubmission, so queue age is kept). Scheduling only: no file, setting or analysis changes. The Phase 2 sync still waits for both arrays to finish, because pending tasks read the project tree at start and the frozen job scripts hard-code that tree; running Phase 2 from a separate copy would mean running edited job scripts, so it was not done. Interactive QOS (`intr`: 1 h, 3 nodes, 1 job) is used for short jobs that need no synced code.

## 2026-11-09: second randomised human dataset

`research/SECOND_DATASET_CHECK.md`. Liu et al. 2022 Experiment II (Mendeley Data 10.17632/dr853shk56.2, CC BY 4.0) is public: 20 Western observers, the same 1,200 random GFG animations, 7AFC on Snoek's labels. Both files downloaded to `data/external/liu2022/` (gitignored), sha256 equal to the repository's (parameters 79e24df6..., responses 52602f37...). Structure only was inspected: 42 binary AUs per trial (1 to 6 present) with peak amplitudes and 75-frame sequences; responses coded 1 to 7. No response distribution was examined. A replication amendment A9 is being drafted; it must be frozen before any model sees a Liu render. Chen et al. 2024 (OSF 3m95w) is probably Snoek basic again; being checked. Skog et al. 2023 is a pixel-space supplement only.

## 2026-11-09: benchmark leak audit is a methods note, not a headline

`research/BENCHMARK_LEAK_AUDIT.md` (script, JSON, 19 tests). Of 11 public affect benchmarks checked from annotation text, none has this study's confound (constant answer position plus one option set per label) in the order its own evaluation code presents. HumanVBench's released files put the answer last in every 5-option item (corrected p .011 to .039; run-time shuffling unknown, code not found); Face-Human-Bench stores the answer first but its evaluation script rotates options. Text-only baselines (answer prior, majority class, option set) beat nominal 1/k chance by up to 32 points in several benchmarks. The paper reports this as a methods recommendation (report text-only baselines beside 1/k) in the Discussion, not as a field-level confound. Any contact with benchmark authors is the user's.

## 2026-11-09: A8 frozen (tag `a8-freeze`, design text)

A8 adds Qwen3-VL-32B and Gemma 4 31B through the frozen pipelines, with its own Holm families; it cannot change any frozen verdict. Frozen on the design text (hash 95dd41b5) before the Qwen3-VL-8B A1 result was read, after an independent audit (21 findings), a re-check (R1 to R6) and a final check. The A3 wrapper reproduces the frozen summary hash exactly on the cluster; both weight copies match Hugging Face file by file. Code pins, the GPU A1 equivalence test and timing smokes are a dated pre-data step before any A8 output. Gemma 4 31B is on the lab project store with the user's approval (2026-11-09).

## 2026-11-09: A9 lead decisions (draft entry, before the freeze)

A9 (`research/PREREG_A9_DRAFT.md`) repeats A4's confirmatory test on Liu et al. 2022 Experiment II. Lead decisions, recorded before any model saw a Liu render:

1. Design B (each configuration on all four Western faces, averaged), with the `frame_max` set as secondary and descriptive.
2. Face independence is tested, not assumed: recovery adds design B with face-correlated noise at rho .25 and .5. General rule, stated before recovery run 2: if B fails under correlation, the claim is narrowed to the conditions that pass, and thresholds never move.
3. Response-code order from Liu's `PMI_exp2.m`, Chen et al.'s identical order and the AU signatures; corroborated by Liu-to-Snoek row matches (largest on the diagonal for all seven labels) and the Chen trial-level match (1.000). No further fetch.
4. The "Open" and "i" AU suffixes are disclosed as unverified.
5. Gender asymmetry kept (no gender in the human fit; rendered gender on the model side, as A4's frozen code), disclosed.
6. Liu against Snoek basic WC is a descriptive yardstick only.
7. The Liu file is used as released (323 stimuli with 5 or 6 AUs), disclosed.
8. Narrowing by check, **taken after recovery run 2's result was seen.** Under face-correlated noise B passed every decision check and failed only pair-swapped estimation accuracy at 0.25x. Decision 2's first operationalisation (a scope label by r_star; with r_star = 0 it would have placed every model outside the validated conditions) was replaced by: all decisions stay confirmatory; the 0.25x accuracy claim is withdrawn under face-correlated noise; each model's face-shared correlation and the reading the replaced rule would have given are reported beside its decision. The replacement is more favourable to confirmatory decisions. The replaced text (block `b548bc82`) was never committed; `a9_recovery_run2.json` carries its hash.

After the independent pre-freeze audit (`research/reviews/a9_audit.md`): the replication rule was fixed before data (comparator: the same model's A4 basic_WC decision; fixed wording per outcome pair; Inconclusive on either side reads "replication not testable"; A9 is never called independent of the GFG platform), and `a9_analyse.py` now hashes every input and refuses unpinned target, ceiling and detector files. Design hash and code pins are in the draft; recovery was rerun as run 4 on the final code. A9 cannot change any A4 verdict.

## 2026-11-09: RDS file quota filled by another project's job; this project's jobs held

At 18:21 the file count was 1,033,846 and rising about 3,700 a minute (997,099 at 17:50). The cause was not this project: `hel-c3-prefetch` (job 40470212, `~/rds/hpc-work/helsinki_cosmos`) was unpacking a container image into `tmp/` on RDS. All of this project's pending jobs were held (`scontrol hold`). With the user's approval the prefetch job was cancelled at about 18:25, and the user ran the removal of its leftover build directories (about 51,000 files), which its own script deletes on success. The count reached 1,048,542 of 1,048,576 before the cleanup and 997,449 after it (19:20). A9 render tasks 6 and 7 hit their time limit while writes failed; their `.partial` files (this project's own) were removed and the two tasks rerun as 40473504. All held jobs were released. No output of this project was lost or read. Note for that project: point `APPTAINER_TMPDIR` at node-local scratch, not RDS.

## 2026-11-10: A7 Stage 1 scope, recorded after Stage 0 and before any A2 analysis output exists

Taken under PREREG_A7 "When Stage 1 runs at all". No A2 cell has been written yet (A2 jobs 40468625 and 40468626 are still pending), so nothing in A2 has been opened. The decision rests only on the two Stage 0 tables (RESULTS_REGISTRY, A7 Stage 0) and compute.

- **Route arm: run, on Gemma 4 and Qwen3-VL**, conditional on the preconditions applied by code: RQ0 passed (both did, A1 and A3), gates G2 to G4 and delivery (both passed), Gate B and the sham for that model and route. If any fails for a model, its route arm is not run and is reported as not run with this entry's date. Cost about 1.06 GPU-hours per model, endpoints in the same job. It is a secondary decomposition of the complete exit cut, for the supplement; RQ3 stays "not testable beyond the interface" either way.
- **Token arm: not run** on either model; Stage 0 did not admit it (region test failed: Gemma `leak_over_limit`, Qwen `token_map_failed`).
- `DECISION_DATE=2026-11-10` is the value passed to `slurm_jobs/study3_a7.sh` for Stage 1.

## 2026-11-10: A6 closed as a limit; its first-frame check blocked the freeze

A6's own pre-freeze rule (`PREREG_A6_DRAFT.md`, "Fixed before freezing", item 6; `first_frame_au_ratio_max` .50, confirmed 2026-11-08 before any model output) requires the first-frame to apex ratio of AU9 and AU10 to be at or under .50 for every detector. ME-GraphAU's AU10 gave .801 (RESULTS_REGISTRY, A6 pre-freeze cluster checks), so `first_frame_blocks_freeze` is true. A6 is therefore not frozen and not run, and ROADMAP 2.5 is closed as a limit by its own rule.

- **No threshold moved, no detector dropped.** GraphAU outputs probabilities with a high resting level (first frame .505, apex .631), so its ratio may say more about that detector's scale than about expressive first frames, and LibreFace's AU10 occurrence column is not on a ratio scale at all. Both points are recorded, not used: the rule names every detector, and arguing a detector out after seeing the value would be choosing the rule by its result. A6's recovery showed what the check guards against: with a first frame keeping .8 of what its fill should hide, Same cue fell to .47 (`PREREG_A6_DRAFT.md`, recovery table).
- **Consequence for the paper.** RQ5 is named as future work, as the Discussion outline already provides ("Only if A6 freezes before its data and finishes. Otherwise named as future work"). No CREMA-D model output exists; none was produced.
- **Jobs.** The A6 weight-manifest jobs 40595461 and 40595462, no longer needed, were cancelled before they ran. The A6 outputs above stay on the cluster as this project's own records.
- **A redesign** (for example a fill taken from a neutral clip of the same actor rather than the first frame) would be a new amendment written and frozen before any data, not an edit of A6.

## 2026-11-12: EU-Emotion working set; the seven unmatched items are closed

The user confirmed that EU-Emotion's creators are collaborators and that the copy we hold is the set supplied for this work. So the seven workbook items with no clip are settled for the paper: they exist in that set only as unedited originals, and the clip list, which takes edited cuts only, leaves them out. Nothing remains to ask the authors (outline item 16). The frozen designs keep the 243 edited clips and do not substitute the originals, which are trimmed differently. With public item IDs permitted since 2026-11-08, the seven are now listed by ID in `research/reproduction/eu_missing_items_check.json`.

## 2026-11-12: A2 rulings after an independent code audit (pre-data)

An independent audit of A2's frozen code against its text found no change to the code or hashes needed, one ambiguity that could change a verdict, two ways the single run could fail or mislead, and several reporting gaps (`research/reviews/a2_code_audit.md`). No A2 cell exists. The rulings are appended to `PREREG_A2_DRAFT.md` as a dated pre-data deviation, which changes no hash.

- **The verdict-level one:** the fill rule now applies to Opposes as well as Matches, the stricter of the two readings the text allowed. An Opposes verdict on a masked cue stands only if the other two fills share the neutral fill's sign. This can only turn an Opposes into Inconclusive.
- **Run safety:** a torn cells line is repaired by removing only unparseable lines, logged, and recomputing them; cells are checked for model identity by pixel shape; a run with a missing gate input is void. `research/scripts/a2_prerun_check.py` does the checks and records every input's sha256 with the run. The cluster's six gate files were checked equal to the committed copies today.
- **Reporting:** 90 percent equivalence intervals and the shuffles-overall table come from `research/scripts/a2_declared_extras.py`, which uses the frozen functions and refuses if an interval does not reproduce the frozen equivalence result. Failed cells are not rerun. The remaining gaps are disclosed as limits.
- **Why not change the code:** it is frozen and correct for what it computes; every ruling either reads its output more strictly or adds a declared calculation beside it.

## 2026-11-12: A5 rulings after an independent code audit (pre-data)

The audit (`research/reviews/a5_code_audit.md`) found that the code which will run implements A5's final rules and evidence-class definitions on the real A2 cell format, with every pin intact. It found three input mistakes that would crash the single run or silently change it, and some reporting gaps. No A2 cell exists. Rulings are appended to `PREREG_A5_DRAFT.md` as a dated pre-data deviation, outside the hashed block and the code-hash section, so no hash changes.

- **Input errors void the run, they are not results:** a model with no pulled cells (which would also shrink the other model's step-down families), A1 or A3 not reading True, and a route with no ok original cell. `research/scripts/a5_prerun_check.py` checks all three, and the human norms hash, before the run.
- **Reporting:** word share, null-evidence reasons, flags, chain without intervals, lexicon mass from A2's output, Absent with E*'s interval ("below the margin" where it excludes 0), and survival beside every Shown cell.
- **Why not change code:** it is frozen and computes what the text says; the rulings fix inputs and readings before data.

## 2026-11-12: A4 rulings after an independent code audit (pre-data)

The audit (`research/reviews/a4_code_audit.md`) found the frozen code byte-identical to the tag and matching the text on the estimand, inference, thresholds, decision rule and Holm. It found two inputs that could change a verdict in an unusual case (a missing family model shrinks Holm; a non-finite log probability gets the smallest p-value), several ways the single run could fail or silently lose stimuli, and reporting gaps. No A4 output exists. The rulings are appended to `PREREG_A4_DRAFT.md` as dated pre-data deviations; the design hash is unchanged (`12c454da`).

- **Verdict-level:** the Holm family is fixed at the two named models, enforced by requiring all 12 complete shards of both before the run; rows with non-finite log probabilities are not usable and are removed in a filtered copy before the run, so their stimuli leave the analysis and count toward the 5 percent limit.
- **Conservative clarifications:** where Aligned and Not aligned both hold, which needs the point estimate above its bootstrap upper bound, the result is reported Inconclusive; R_S is stated as the Spearman-Brown stepped value.
- **Run safety:** shard completeness and uniqueness, inputs at pinned bytes, a gates file built by script from the frozen outputs (`research/reproduction/a4_gates.json`), and single-threaded BLAS, all checked by `research/scripts/a4_prerun_check.py`, which records every input's sha256.
- **Also appended:** the ceiling-estimator disclosure of 2026-11-08, quantified by the audit at .0003 or less on the basic ceilings.
- **Correction to my own text, same day:** the residual noise-ceiling simulation added to Supplementary Note 1 this morning corrects the point estimate, not the bootstrap upper bound that the Not aligned rule corrects. The note and the registry row now say so, and cite recovery's boundary check (at most .026) as the evidence for the rule as applied.
