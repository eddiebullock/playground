# DRAFT amendment A4: VLMs against randomised human cue effects

**Status: DRAFT. Not frozen, not preregistered.** Nothing here binds until it is audited, committed and tagged. No model has seen a render, and `scripts/a4_run.py` refuses to score until the amendment is frozen.

**Written:** 2026-11-07, before any A1 result was read and before any model saw a render. A3 was known, as were the gates, and the human Snoek effects (`research/reproduction/human_au_effects.json`).
**Depends on:** RQ0 and gates G2 to G4, per model. RQ0 is read through the frozen A1 and A3 table by `rq0_status`, the rule A2 uses, at `single_face`. So Pass on A3 with Pass or Fail on A1 admits a model. Any other outcome, or a missing result, leaves it Inconclusive.
**Models.** Gemma 4 and Qwen3-VL are the Holm family. Molmo2 fails G2 and G4, so it is descriptive if run at all.
**Code:** `scripts/a4_design.py` (fixed constants, hashed), `research/scripts/a4_stimuli.py`, `scripts/a4_run.py`, `scripts/a4_analyse.py`, `scripts/a4_recovery.py`, `slurm_jobs/study3_a4.sh`, `slurm_jobs/study3_a4_render.sh`, `research/scripts/render_au_faces.py`. Also `research/scripts/a4_detector.py`, `slurm_jobs/study3_a4_detector.sh`. Each cell and output carries `design_sha256`, which covers every constant in `a4_design.py` and the digests of the human matrix, prompts, A2's runner and the renderer. The manifest is pinned, and checked at run time. Theory: `research/theory/causal_feature_alignment.md`.

## Question

RQ1, on randomised cues. When a face's action units (AUs) are set at random, as they were for the human observers, does a model's judgement change with each AU the way a human's does?

**Why this design.** Natural clips cannot answer this: the theory note shows that behaviour on natural items does not constrain cue effects, and EU-Emotion's cues are far more correlated than the Snoek design: condition number 76.1 on 15 LibreFace cues and 22.4 on 10 OpenFace 3 cues, against 3.4 to 3.5 on Snoek's 25 merged cues. Cue counts differ, so this is rough (table in the theory note). The same-clip RQ1 result supports "aligned with the human pattern", not "uses the same cues". Here the cues are randomised for both.

**Novelty, as far as it is known.** We located no study comparing a vision-language model's response to named facial cues with effects estimated from human observers under randomised cue presentation. Coarse image-level counterfactuals on VLMs exist (Zhang et al., ICML 2026, whose stimuli are only partly faces, with no named cue and no human data), as do latent AU interventions on an affective model (Huynh et al., arXiv 2606.15779, which act on a learned latent rather than images and use no randomised human data). Both were read in part, and the Huynh wording came through a summarising tool, so quotes need rechecking (`research/LIT_CHECK_2026-11-07.md`). The claim is cue-level, keeps its search date, and notes that Semantic Scholar was down and CCN could not be fully searched.

**Scope.** Causal claims are about the model. The human side is a fixed target from another stimulus set. Results concern these avatars.

## Human target

Snoek et al. 2023, CC BY 4.0, four datasets: basic emotions 7AFC and conversational states 5AFC, each from Western (WC) and East Asian (EA) observers. Humans saw 1.25 s animations of random AU combinations on GFG faces, built on eight base faces per group. Models see the same AU configurations, not the same stimuli. The effect matrix is the existing output: `do_effect_probability_au_full_vs_none`, options by AUs, on the `family_25` basis. It is not refitted here.

| Dataset | Subjects | Ridge lambda | Ceiling, subjects and stimuli split, do effects | Fair comparison |
|---|---|---|---|---|
| basic_WC | 60 | 0.1 | .990 | basic WC to EA: .877 |
| basic_EA | 60 | 0.1 | .989 | as above |
| conversational_WC | 20 | 1.0 | .978 | conversational WC to EA: .767 |
| conversational_EA | 20 | 1.0 | .974 | as above |

The ceilings come from `research/reproduction/a4_human_ceiling.json`, 200 splits. The earlier ceiling was on log odds, so it is recomputed on do effects with the same splitting. Human data only.

**Reference.** One human group against the other, on the same basis, is the fair yardstick: a model cannot be asked to agree with Western observers better than East Asian observers do. The ceiling only says how far noise limits the score.

## Stimuli

Rendered by `research/scripts/render_au_faces.py`: headless, deterministic, 768 by 768, frontal, flat lighting, black background, as Snoek describes.

| Set | Content | Images |
|---|---|---|
| `main` | Every available configuration in each dataset, each on one identity: 4,659, 2,323, 2,318 and 2,325 | 11,625 rows |
| `replicate` | First 200 of those on a second identity | 800 rows |
| `one_au` and `neutral` | Each of 33 columns alone at .25, .5 and 1, on 6 identities, plus their neutral faces | 600 rows |
| `scale_0.75`, `scale_1.25` | 400 configurations, basic_EA and conversational_EA | 1,600 rows |
| `blind` | First 600 basic_EA configurations, under the blind readout | 600 rows |

That is 15,225 rows and 11,761 unique images. Manifest sha256 `2ef1161e`.

**Selection.** All configurations are used, because 1,200 per dataset failed recovery (see Recovery). They are ranked by a seeded hash of the configuration string, by `a4_stimuli.py`, with no use of responses, and that order picks the replicate, scale and blind subsets. Face gender comes from those shown to humans with that configuration, by hash.

**Identity.** Own-group faces, as humans saw: four Western faces (`female_a`, `female_b`, `male_a`, `male_b`) for WC sets, and four East Asian faces (`ea_female` and `ea_male`, each at seeds 0 and 1) for EA sets. Humans saw eight base faces per group, so identity coverage is narrower here, and it is not modelled beyond gender, as in the human fit. The East Asian faces are MakeHuman's macro sliders set to Asian, plus a seeded jitter for the second face. They are not based on East Asian people, and they are not independent of the Western meshes.

**Storage.** The cluster is near its file quota, at 995,969 of 1,048,576 files on 2026-11-07. So images are never loose files. They are rendered into 40 tar files, built aside and moved into place, about 2.3 GB in all, and read from the tars at run time. Output is one cells file per model and shard. Assets travel as one archive, `data/au_render_assets.tgz`, unpacked to node-local storage in each task and never onto RDS, since they hold 2,590 files. Nothing rendered enters git: FACSHuman's targets are LGPL-3.0 with no output clause, so no image is published.

**Approximations.** AU22 and AU27 are approximated in our renders (`research/notes_au_render.md`). That is our local finding: FACSHuman's own Table 9 lists AU22 as usable. The main analysis keeps them, and a sensitivity removes both families. Amplitude maps linearly to target weight, which is an assumption, so the 0.75x and 1.25x sets repeat the fit. At 1.25x amplitudes stop at 1, as the renderer's range requires, so that set compresses large amplitudes.

**Laterality.** `AU..L` moves the avatar's left side. Whether GFG's L means the same is unverified, so the confirmatory basis is `family_25`, which uses the bilateral composite.

## Task and route

**Route.** One image, sent as on `single_face` and scored as in A3. A static route is what can be tested: the renderer gives a peak frame, and the time-course parameters of each human stimulus are not in the public data. Humans saw 1.25 s of movement, and their effects average over time course. So this compares a model's reading of a peak frame with human effects averaged over motion.

**Prompt.** `build_4afc_prompt(options, condition="video_only")`, unchanged from A1 and A3.

**Options.** One fixed set per task, so options carry no information about the stimulus, unlike EU-Emotion.

- Basic: Anger, Disgust, Fear, Happy, Sadness, Surprise, None of the above.
- Conversational: Bored, Confused, Interested, Thinking, None of the above.

**"Other".** Human "other" maps to "None of the above", the abstain option of A1 to A3, and it stays last. This was decided from the house prompt and not from any model behaviour. The two are not the same word, so a sensitivity drops that row.

**Rotations.** All cyclic rotations of the named labels: 6 for basic, 4 for conversational. A1's five were a complete set for five named labels. Five here would miss a position on basic and repeat one on conversational. Human option order is not in the data.

**Scoring.** Exact option probabilities: each option's joint logprob after "EMOTION:", renormalised over the options, then averaged over a stimulus's complete rotation set. A stimulus without a usable response on every rotation leaves the analysis, and losing over 5 percent of a dataset's stimuli fails that dataset.

**Blind readout, secondary.** A2's frozen prompt and lexicon, on basic labels only: Confused and Thinking have no lexicon family. Six families are renormalised, with no "other" row. It never replaces or rescues the MCQ result. Its CFA is not in the same units as the MCQ CFA: the matrix has six rows, not seven, the human rows no longer sum to zero, and the scores are family sums. The two are not compared.

## Estimand

From `research/theory/causal_feature_alignment.md`: tau(j, k) is the average effect of setting cue j on option k. The fit is the human pipeline, `human_au_effects`: a ridge logistic per option, one-vs-rest, on 33 AU amplitudes plus centred face gender, with intercept and gender unpenalised, then G-computation of do(AU=1) minus do(AU=0), with probabilities renormalised over options. The model counts as one subject, with fractional responses. Ridge strength is the human value for that dataset, fixed in advance.

**Statistics.** For each model and dataset, two, each with its own null, interval and reliability:

    raw CFA      = cos(tau_model, tau_human)
    residual CFA = cos(R(tau_model), R(tau_human))

over options by 25 families. R takes out each cue's component along that matrix's own mean profile across cues. A system that answers every cue with one profile, scaled by salience, can score on raw CFA without any cue-specific agreement, and it has nothing left after R.

**Which is confirmatory, stated before data.** Residual CFA is confirmatory, and raw CFA is secondary. Raw CFA was tried first and failed recovery on a decision: a false Aligned of up to .33 at 1,200 configurations, and .40 with all of them, on conversational_WC. Residual CFA's failures at 1,200 were of estimation only. If residual CFA fails recovery, the analysis refuses to run. The choice is written into `a4_recovery.json`, and the analysis reads it from there.

For raw CFA, each cue's effects sum to zero across options, so this equals Pearson r over entries. It is reported raw, and over the ceiling times the square root of the model's split-half reliability across stimuli, 20 splits. Corrected values are not used for decisions.

**Dose.** The note defines tau at a dose delta from z. A4 uses the set-to form the human matrix uses: do(AU=1) minus do(AU=0), with other AUs as observed. So the dose is the full amplitude range, on both sides. The note gives the relation.

**Model reliability.** A model has no response noise, so its reliability here reflects which stimuli were sampled. It is taken by split-half across stimuli. The 200 replicates vary identity alone, and they serve V3, not the correction.

**Direct contrast.** The one-AU set gives the do() contrast itself: an AU alone against that identity's neutral face. At amplitude 1, bilateral sides summed. Secondary, because it tests an AU at a neutral baseline, whereas the human effect is averaged over other active AUs, and it is the only place dose can be read.

## Inference

**Signal.** Do AUs move the model at all? Stimuli are shuffled against responses, 999 draws, on the summed squared OLS AU coefficients, which needs no refit. p < .05 is required. A prior-only system fails here.

**Alignment.** AU labels are shuffled across the 25 columns of the model's matrix, 9,999 draws, one-sided on CFA. This keeps each cue's profile, and any effect common to all cues, so a model answering "None of the above" less whenever anything moves gains nothing. Holm across the two eligible models, within a dataset.

**Interval.** Stimulus bootstrap, 2,000 draws, 95 percent percentile. It treats stimuli as sampled, and does not include error in the human matrix, whose reliability is near 1 here.

**Smallest effect of interest.** CFA of .30 [proposed], used as a floor, not a target: it is the least we would call alignment, and clearing it says little. It sits under the half-shared observer, and is about 9 percent of shared variance. There is no external source for this value.

## Decision rule, per model and dataset

| Outcome | Requires |
|---|---|
| **Aligned** | All gates; signal p < .05; alignment Holm p < .05; CFA at least .30; CI lower bound above 0 |
| **Not aligned** | All gates; signal p < .05; CI upper bound divided by the square root of R_S below .30, where R_S is the model's split-half reliability across stimuli on that statistic |
| **Inconclusive** | Anything else, including any failed or missing gate |

**Null rule.** Not aligned is the null, and it requires the model to be shown to respond to AUs (signal), the gates, and the equivalence bound after disattenuation. A model with no signal cannot be Not aligned: it is Inconclusive, since nothing shows that the cues were delivered. A missing or non-positive R_S cannot support Not aligned.

**Asymmetric rule.** Aligned uses raw values and Not aligned uses disattenuated ones. That is deliberate: noise can only lower a cosine, so it cannot create Aligned, whereas it could create a false null. The rule therefore favours neither claim by the same margin, and Aligned can be reached at a CFA where the interval is wide.

**Reliability, stated before data.** Below R_S of .85, raw CFA reads as a lower bound. Aligned still stands, and the corrected rule decides Not aligned.

**Reading Aligned.** It means a reliable shared pattern, not the same strategy. In recovery, an observer with 16 of 25 cue families swapped in 8 pairs can still clear .30. So CFA is always reported against the WC to EA reference, and a model is called human-level only if its CI reaches that reference, which is descriptive.

**Headline across the four datasets.** Each dataset is reported alone, then one headline per model per task, by `headline`: Aligned or Not aligned only if WC and EA agree; Mixed if one of each; Inconclusive otherwise. Basic and conversational are separate claims and are never pooled. WC and EA share neither observers nor most configurations, but they are not independent tests of a model.

## Validity gate, before any human comparison

None of these uses a human target. All run per model and task, and missing means not passed.

| Check | Rule [proposed] |
|---|---|
| V1 Detector | Across pairs of covered column and identity in the one-AU set, at least 70% have detected output rising across .25, .5 and 1 |
| V2 Image dependence | Mean RMS change in option log probability, render against pixels removed, at least 0.25 nats, on 50 stimuli |
| V3 Configuration carries | Output closer for one configuration on two identities than for a mismatch, on the JS divergence, sign-flip over 200 configurations, one-sided p < .01. Mismatches are drawn within gender and identity pair, so face differences cannot pass for configuration. This is the real against mismatch delta, with no human target |
| V4 Domain limit | Mean mass on None of the above at most .90, and on any option at most .95 |

**V1 coverage.** LibreFace covers 17 AUs: 12 with intensity (1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26), and presence alone for 7, 10, 14, 23 and 24. So at least 8 of 25 families have no readout: AU11, 13, 16, 22, 27, 38, 39 and 43. The denominator is covered pairs only, and those columns are listed as unconfirmed, not as failures. LibreFace is bilateral, so L and R columns read the same output, and laterality is not tested. It was trained on natural faces, so a failure may reflect the detector and not the render; V1 does not separate them. Failing pairs are listed. The earlier MediaPipe check (`research/notes_au_render.md`) is not a substitute, as it reports blendshapes. Missing detector output stops the analysis: it is not read as an Inconclusive.

**Pilot, before the freeze: passes the rule, weakly.** Job 40260039 on cclake, with no model, read 2026-11-07, `research/reproduction/a4_v1_pilot.csv`. The first job, 40255068, failed because the archive's Mac `._` files were read as meshes, which is fixed. A face was found on all 432 images, and 339 of 432 covered pairs rise, .785 against .70.

**That pass is thin, and it is disclosed.** The rule counts any rise, and most are tiny. Of 96 intensity pairs, 22 rise by at least .5 on LibreFace's 0 to 5 scale, all in AU01, 05, 25 and 26. AU12 at full amplitude reads about .02, so LibreFace does not see the rendered smile, and the same holds for AU04, 06, 09, 15 and 20. V1 therefore shows the renders are monotonic where the detector responds. It does not show that a natural-face detector reads them as those AUs. The threshold was not changed after seeing this.

**V4 limits a known risk.** On A3, Qwen and Molmo chose None of the above on .534 and .863 of cells. If that recurs on avatars, V4 stops a comparison resting on a sliver of probability. Its .90 limit sits only just above Molmo's .863, and it is on mean mass, not modal choice, so V4 is a weak guard: a model near .86 on None passes it.

**Blind Gate B analogue.** B1 repeat change at most 1e-3; B2 median lexicon mass at least .05; B3 mean RMS change in family CLR, render against pixels removed, at least 0.25 nats, on 50 stimuli; and B4 as V3 at p < .01, on basic_EA's 200 replicate configurations. The blind result is reported only if these pass.

## Descriptive: does CFA say something that behaviour does not?

The theory note needs a demonstration, and this is where one can occur. On the same stimuli, per model and dataset, beside CFA:

- Output similarity: Pearson r between model and mean human response over stimuli and options, with option means removed.
- Modal agreement and chance-corrected kappa. Snoek has no correct answer, so error consistency proper is undefined, and this is its closest analogue.

Whether these disagree with CFA is an empirical result. Recovery's pair-swapped observer shows they can: see below.

## Sensitivity

Reported beside the primary, never replacing it: without AU22 and AU27; without the abstain row; amplitudes at 0.75x and 1.25x, against the 1x fit on the same configurations; and the direct contrast, by dose.

## Recovery

Before any run, synthetic observers answer the real 1,200-configuration design through `scripts/a4_analyse.py`, on all four datasets, with `scripts/a4_recovery.py`. Effect sizes are multiples of the human log odds at 0.25, 0.5 and 1. Realism: logit noise of SD 1 per stimulus and option, and a lognormal gain error of log SD 0.5 per AU, standing in for the unknown amplitude mapping. So the observers do not see the cues exactly as humans did.

**Pass rule, not changed between the second and third runs** (`RECOVERY` in `scripts/a4_design.py`), applied to each statistic:

- Human-like Aligned in at least 80% of replicates, at 0.5x and 1x.
- At 1x: half-shared alignment p in at least 80%, Aligned in at least 60%, and different-cues Not aligned in at least 50%.
- At every strength: false Aligned at most 10% for different-cues, prior-only, generic and salience-profile; prior-only signal at most 10%; Not aligned at most 10% where true CFA is at or above .30, on the boundary observer; and pair-swapped CFA within .10 of truth in at least 90%, except where withdrawn below.

New in the second run: a boundary observer, mixing human weights at .35, and a salience-profile observer, one human mean profile for every cue, scaled by each human cue's size and as strong as the human weights.

**Every attempt, in order. Failed runs are kept.**

| Run | Configurations | Datasets | File, sha256 | Statistic | Checks | Result |
|---|---|---|---|---|---|---|
| 1 | 1,200 per dataset | basic_EA, conversational_EA | `a4_recovery_first_run.json`, `a0fe98dc` | Raw | 27 of 28 | FAIL: pair-swapped accuracy on conversational_EA at 0.25x, 66% |
| 2 | 1,200 per dataset | All four | `a4_recovery_second_run.json`, `64d7d1c9` | Raw | 63 of 66 | FAIL: false Aligned on conversational_WC, up to .33 |
| 2 | 1,200 per dataset | All four | as above | Residual | 62 of 66 | FAIL: pair-swapped accuracy in 4 cells, 76 to 88% |
| 3 | All available | All four | `a4_recovery.json`, `69efb5de` | Raw | 62 of 66 | FAIL: false Aligned on conversational_WC and at conversational_EA 0.25x, up to .40 |
| 3 | All available | All four | as above | Residual | 66 of 66 | **PASS** |

**Stated before data, after run 1 and before any model.** Raw accuracy at conversational 0.25x is withdrawn as a claim, in both conversational datasets. Below R_S of .85, raw CFA is a lower bound. Aligned stands, and the corrected rule decides Not aligned.

**That exemption matters to the pass.** It was written for raw CFA after run 1, and it also covers residual CFA in run 3, where accuracy at those two cells is 85% on conversational_WC and 62% on conversational_EA. Without it, residual CFA would pass 64 of 66. So A4 makes no accuracy claim for residual CFA on weak conversational effects, and a decision there rests on the corrected rule.

| Rate, run 3, all cells unless stated | Raw CFA | Residual CFA |
|---|---|---|
| Human-like Aligned | 1.0 | 1.0 |
| Half-shared Aligned at 1x, lowest | .87 | .79 |
| Different-cues Not aligned at 1x, lowest | .95 | .95 |
| False Aligned, highest | .40 | .03 |
| Prior-only signal, highest | .08 | .08 |
| Not aligned with true CFA at or above .30, highest | .01 | .03 |

**Reading.**

- Raw CFA can be fooled. On conversational_WC, a system with no cue-specific agreement reaches Aligned in up to .40 of replicates, because about a third of human energy there lies along the common profile. More stimuli do not help, since the error is one of the statistic. So residual CFA is the right primary.
- Residual CFA makes no false Aligned above .03.
- Residual CFA reads low on weak effects: at 0.25x, human-like reads .74 against a true .87 on conversational_EA.
- Recovery assumes additive log-odds observers, independent stimulus noise and the gain error above. It does not model domain shift, so it shows how the analysis behaves, not that a real model will respond this way.

**Demonstration.** Pair-swapped observers have 16 of 25 families swapped in 8 pairs. On natural-like stimuli with paired cues at a constructed r of .9, their outputs correlate .83 to .95 with the human-weighted observer, whereas raw CFA on the randomised design was .42 to .56 in run 2. So on synthetic data behaviour stays high while CFA falls, which is the case the theory note predicts. The r of .9 is our construction, not a measurement: EU-Emotion's largest cue correlation is .86 on LibreFace, and most are far lower.

## Disclosures

These do not change the rule. They are stated so that a reader can weigh it.

- **SESOI.** .30 is a floor set by the implementer, with no external source. A model at .31 is Aligned and far from human-level.
- **Asymmetric rule.** Aligned is raw and Not aligned is disattenuated, so the two claims face different bars.
- **V4.** Its .90 limit sits just above Molmo's .863 on None of the above, and it is on mean mass, so a heavily abstaining model can pass.
- **AU22 and AU27.** They hold 10 to 18 percent of human squared effects: .109, .119, .183 and .103 on basic_WC, basic_EA, conversational_WC and conversational_EA. That is a large share for two approximated families (`a4_human_ceiling.json`).
- **Common profile.** It holds .26 to .35 of human squared effects. Removing it lowers the human WC to EA reference to .780 on basic and .732 on conversational, and each statistic must be read against its own reference. Residual ceilings are .960 to .983.
- **East Asian faces.** Macro sliders plus jitter, so "own-group faces" is approximate for EA sets.
- **Constructed r of .9** in the demonstration is ours.
- **Blind CFA** is in different units, with six rows, and is not compared with MCQ CFA.
- **Recovery** uses 100 bootstrap draws and 6 reliability splits per replicate; the real analysis uses 2,000 and 20. The number of configurations was raised after failures, so the design was tuned on recovery, on synthetic data only.
- **Bootstrap.** Stimuli only, conditional on the human matrix and on these identities.
- **Gate B analogue** is on basic_EA only, on 200 configurations for B4.
- **Citation gaps.** Huynh et al. and Zhang et al. were read only in part, the Huynh wording came through a summarising tool, and Semantic Scholar was down for the check. FACSHuman's Table 9 lists AU22 as usable, so "approximated" is local to our renders. Humans saw eight base faces per group, and models see the same AU configurations, not the same stimuli.
- **V1** has a denominator of covered pairs, so a pass says nothing about the 8 families without readout, or about laterality, and most of its rises are tiny.

## Limits

- **Avatar domain.** Bald, static, computer-generated faces, unlike those humans saw. A difference from humans may reflect the domain, and V1 to V4 only partly address it. This is a new stimulus set of the same AU recipes, not a reproduction.
- **Static route.** No motion. Humans saw 1.25 s.
- **Amplitude.** The mapping from GFG amplitude to target weight is unknown. Patterns are compared, not sizes.
- **AU22, AU27 and AU11.** Approximated or barely visible.
- **Observers.** Western and East Asian samples, with 20 subjects per group on conversational data. No claim about other populations.
- **Unmeasured cues.** Only these 33 columns. Effects are additive by assumption, on both sides.
- **Reach.** Six basic emotions and four conversational states, about 8 of EU-Emotion's 21 labels. None of Worried, Disappointed, Kind or Unfriendly.
- **Task.** Models read a text prompt with listed options, which is not how humans were asked. The prompt is the house prompt, which mentions "video frames" although one image is sent, as in A3.
- **Sample of faces.** Four per set.
- **Thresholds.** Marked **[proposed]** values were set by the implementer, with no external source.

## Cost

Figures are upper bounds. No model has run this task, so A3's 1:45 job limit over 2,430 cells gives 2.6 s per cell. Scaled by option count, that is 3.0 s on basic and 2.2 s on conversational. Blind cells use `smoke_v2` per-cell times.

| Part | Cells | Gemma 4, hours | Qwen3-VL, hours |
|---|---|---|---|
| Basic MCQ | 52,428 | 43.7 | 43.7 |
| Conversational MCQ | 25,568 | 15.6 | 15.6 |
| Blind | 850 | 0.7 | 1.2 |
| Image-free | 100 | under 0.1 | under 0.1 |
| **Total per model** | 78,946 | about 60 | about 61 |

Twelve shards per model at about 5 h, against a 6 h limit, resumable, about 121 GPU-hours for both of 2,765 available. A timing smoke after the freeze replaces these figures. Rendering: 11,761 images at 3.1 s is about 10 core-hours, on 40 CPU tasks. That timing comes from the Mac; V1's cluster job took under 19 minutes for 438 renders with detection.

## Order

1. Pilot V1, with no model.
2. Audit this amendment, then freeze with hashes and tag `a4-freeze`, before any A1 result is read if the queue allows.
3. Render into tars. No model.
4. After RQ0, run a timing smoke, then the full run per eligible model. Smoke files stay out of the analysis.
5. Analyse once, after all shards finish.
