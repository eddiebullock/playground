# DRAFT amendment A4: VLMs against randomised human cue effects

**Status: DRAFT. Not frozen, not preregistered.** Nothing here binds until it is audited, committed and tagged. No model has seen a render, and `scripts/a4_run.py` refuses to score until the amendment is frozen.

**Written:** 2026-11-07, before any A1 result was read and before any model saw a render. A3 was known, as were the gates, and the human Snoek effects (`research/reproduction/human_au_effects.json`).
**Depends on:** RQ0 and gates G2 to G4, per model, read through the frozen A1 and A3 table. A4 uses one image, so Pass on A3 with Pass or Fail on A1 admits the model. Any other outcome, or a missing result, leaves it Inconclusive.
**Models.** Gemma 4 and Qwen3-VL are the Holm family. Molmo2 fails G2 and G4, so it is descriptive if run at all.
**Code:** `scripts/a4_design.py` (fixed constants, hashed), `research/scripts/a4_stimuli.py`, `scripts/a4_run.py`, `scripts/a4_analyse.py`, `scripts/a4_recovery.py`, `slurm_jobs/study3_a4.sh`, `slurm_jobs/study3_a4_render.sh`, `research/scripts/render_au_faces.py`. Each cell and output carries `design_sha256`, currently `b84bd471` in draft. Theory: `research/theory/causal_feature_alignment.md`.

## Question

RQ1, on randomised cues. When a face's action units (AUs) are set at random, as they were for the human observers, does a model's judgement change with each AU the way a human's does?

**Why this design.** Natural clips cannot answer this: the theory note shows that behaviour on natural items does not constrain cue effects, and EU-Emotion's cues are far more correlated than the Snoek design (condition number 76 against about 3.5). The same-clip RQ1 result supports "aligned with the human pattern", not "uses the same cues". Here the cues are randomised for both.

**Novelty, as far as it is known.** We located no study comparing a vision-language model's response to named facial cues with effects estimated from human observers under randomised cue presentation. Coarse image-level counterfactuals on VLMs exist (Zhang et al., ICML 2026, whose stimuli are only partly faces, with no named cue and no human data), as do latent AU interventions on an affective model (Huynh et al., arXiv 2606.15779, which act on a learned latent rather than images and use no randomised human data). Both were read in part, and the Huynh wording came through a summarising tool, so quotes need rechecking (`research/LIT_CHECK_2026-11-07.md`). The claim is cue-level, keeps its search date, and notes that Semantic Scholar was down and CCN could not be fully searched.

**Scope.** Causal claims are about the model. The human side is a fixed target from another stimulus set. Results concern these avatars.

## Human target

Snoek et al. 2023, CC BY 4.0, four datasets: basic emotions 7AFC and conversational states 5AFC, each from Western (WC) and East Asian (EA) observers. Humans saw 1.25 s animations of random AU combinations on GFG faces. The effect matrix is the existing output: `do_effect_probability_au_full_vs_none`, options by AUs, on the `family_25` basis. It is not refitted here.

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
| `main` | 1,200 configurations per dataset, each on one identity | 4,800 rows |
| `replicate` | First 200 of those on a second identity | 800 rows |
| `one_au` and `neutral` | Each of 33 columns alone at .25, .5 and 1, on 6 identities, plus their neutral faces | 600 rows |
| `scale_0.75`, `scale_1.25` | 400 configurations, basic_EA and conversational_EA | 1,600 rows |
| `blind` | First 600 basic_EA configurations, under the blind readout | 600 rows |

That is 8,400 rows and 7,102 unique images. Manifest sha256 `0ae87d06`.

**Selection.** Configurations are ranked by a seeded hash of the configuration string, by `a4_stimuli.py`, with no use of responses. The cap of 1,200 per dataset, out of 2,314 to 4,659, is set by cost. Models answer without response noise, and recovery tests whether 1,200 is enough. Face gender comes from those shown to humans with that configuration, by hash.

**Identity.** Own-group faces, as humans saw: four Western faces (`female_a`, `female_b`, `male_a`, `male_b`) for WC sets, and four East Asian faces (`ea_female` and `ea_male`, each at seeds 0 and 1) for EA sets. Humans saw eight base faces or more, so identity coverage is narrower here, and it is not modelled beyond gender, as in the human fit.

**Storage.** The cluster is near its file quota, at 995,969 of 1,048,576 files on 2026-11-07. So images are never loose files. They are rendered into 40 tar files, built aside and moved into place, about 1.4 GB in all, and read from the tars at run time. Output is one cells file per model and shard. Assets travel as one archive, `data/au_render_assets.tgz`, unpacked to node-local storage in each task and never onto RDS, since they hold 2,590 files. Nothing rendered enters git: FACSHuman's targets are LGPL-3.0 with no output clause, so no image is published.

**Approximations.** AU22 and AU27 are approximated (`research/notes_au_render.md`). The main analysis keeps them, and a sensitivity removes both families. Amplitude maps linearly to target weight, which is an assumption, so the 0.75x and 1.25x sets repeat the fit. At 1.25x amplitudes stop at 1, as the renderer's range requires, so that set compresses large amplitudes.

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

**Blind readout, secondary.** A2's frozen prompt and lexicon, on basic labels only: Confused and Thinking have no lexicon family. Six families are renormalised, with no "other" row. It never replaces or rescues the MCQ result.

## Estimand

From `research/theory/causal_feature_alignment.md`: tau(j, k) is the average effect of setting cue j on option k. The fit is the human pipeline, `human_au_effects`: a ridge logistic per option, one-vs-rest, on 33 AU amplitudes plus centred face gender, with intercept and gender unpenalised, then G-computation of do(AU=1) minus do(AU=0), with probabilities renormalised over options. The model counts as one subject, with fractional responses. Ridge strength is the human value for that dataset, fixed in advance.

**Confirmatory statistic.** For each model and dataset:

    CFA = cos(tau_model, tau_human)

over options by 25 families. Rows sum to zero, so this equals Pearson r over entries. It is reported raw, and over the ceiling times the square root of the model's split-half reliability across stimuli. Corrected values are not used for decisions.

**Direct contrast.** The one-AU set gives the do() contrast itself: an AU alone against that identity's neutral face. At amplitude 1, bilateral sides summed. Secondary, because it tests an AU at a neutral baseline, whereas the human effect is averaged over other active AUs, and it is the only place dose can be read.

## Inference

**Signal.** Do AUs move the model at all? Stimuli are shuffled against responses, 999 draws, on the summed squared OLS AU coefficients, which needs no refit. p < .05 is required. A prior-only system fails here.

**Alignment.** AU labels are shuffled across the 25 columns of the model's matrix, 9,999 draws, one-sided on CFA. This keeps each cue's profile, and any effect common to all cues, so a model answering "None of the above" less whenever anything moves gains nothing. Holm across the two eligible models, within a dataset.

**Interval.** Stimulus bootstrap, 2,000 draws, 95 percent percentile. It treats stimuli as sampled, and does not include error in the human matrix, whose reliability is near 1 here.

**Smallest effect of interest.** CFA of .30 [proposed]. The half-shared observer sits near .50, so .30 is under the weakest alignment worth reporting, and it is about 9 percent of shared variance. There is no external source for this value.

## Decision rule, per model and dataset

| Outcome | Requires |
|---|---|
| **Aligned** | All gates; signal p < .05; alignment Holm p < .05; CFA at least .30; CI lower bound above 0 |
| **Not aligned** | All gates; signal p < .05; CI upper bound below .30 |
| **Inconclusive** | Anything else, including any failed or missing gate |

**Null rule.** Not aligned is the null, and it requires the model to be shown to respond to AUs (signal), the gates, and the equivalence bound. A model with no signal cannot be Not aligned: it is Inconclusive, since nothing shows that the cues were delivered.

**Reading Aligned.** It means a reliable shared pattern, not the same strategy. In recovery, an observer with 16 of 25 cue families swapped in 8 pairs can still clear .30. So CFA is always reported against the WC to EA reference, and a model is called human-level only if its CI reaches that reference, which is descriptive.

**Across datasets.** Each dataset is reported alone. Basic and conversational are separate claims. WC and EA share neither observers nor most configurations, but are not independent tests of a model.

## Validity gate, before any human comparison

None of these uses a human target. All run per model and task, and missing means not passed.

| Check | Rule [proposed] |
|---|---|
| V1 Detector | Across AU and identity pairs in the one-AU set, at least 70% have detected intensity rising across .25, .5 and 1 |
| V2 Image dependence | Mean RMS change in option log probability, render against pixels removed, at least 0.25 nats, on 50 stimuli |
| V3 Configuration carries | Output closer for one configuration on two identities than for mismatched configurations, on the JS divergence, sign-flip over 200 configurations, one-sided p < .01. This is the real against mismatch delta, with no human target |
| V4 Domain limit | Mean mass on None of the above at most .90, and on any option at most .95 |

**V1 coverage.** LibreFace reports intensity for AUs 1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25 and 26, and presence for 7, 10, 14, 23 and 24, so AU22 and AU27 are not confirmed by it. Unconfirmed columns are listed, not counted as failures. LibreFace was trained on natural faces, so a failure may reflect the detector and not the render; V1's result does not separate them. Its failing pairs are listed. The earlier MediaPipe check (`research/notes_au_render.md`) is not a substitute, as it reports blendshapes.

**V4 limits a known risk.** On A3, Qwen and Molmo chose None of the above on .534 and .863 of cells. If that recurs on avatars, V4 stops a comparison resting on a sliver of probability.

**Blind Gate B analogue.** B1 repeat change at most 1e-3, B2 median lexicon mass at least .05, B3 as V2 on the blind route where baselines are available, and B4 as V3 at p < .01. The blind result is reported only if these pass.

## Descriptive: does CFA say something that behaviour does not?

The theory note needs a demonstration, and this is where one can occur. On the same stimuli, per model and dataset, beside CFA:

- Output similarity: Pearson r between model and mean human response over stimuli and options, with option means removed.
- Modal agreement and chance-corrected kappa. Snoek has no correct answer, so error consistency proper is undefined, and this is its closest analogue.

Whether these disagree with CFA is an empirical result. Recovery's pair-swapped observer shows they can: see below.

## Sensitivity

Reported beside the primary, never replacing it: without AU22 and AU27; without the abstain row; amplitudes at 0.75x and 1.25x, against the 1x fit on the same configurations; and the direct contrast, by dose.

## Recovery

Before any run, synthetic observers answer the real 1,200-configuration design through `scripts/a4_analyse.py`, on basic_EA and conversational_EA, with `scripts/a4_recovery.py`. Effect sizes are multiples of the human log odds at 0.25, 0.5 and 1. Realism: logit noise of SD 1 per stimulus and option, and a lognormal gain error of log SD 0.5 per AU, standing in for the unknown amplitude mapping. So the observers do not see the cues exactly as humans did.

RECOVERY_TABLE

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
| Basic MCQ | 25,122 | 20.9 | 20.9 |
| Conversational MCQ | 16,684 | 10.2 | 10.2 |
| Blind | 800 | 0.7 | 1.2 |
| Image-free | 100 | under 0.1 | under 0.1 |
| **Total per model** | 42,706 | about 32 | about 32 |

Eight shards per model at about 4 h, against a 6 h limit, resumable, about 64 GPU-hours for both of 2,765 available. A timing smoke after the freeze replaces these figures. Rendering: 7,102 images at 3.1 s is about 6.1 core-hours, on 40 CPU tasks. This timing comes from the Mac, not the cluster.

## Order

1. Audit this amendment, then freeze with hashes and tag `a4-freeze`, before any A1 result is read if the queue allows.
2. Copy the assets archive, build the manifest, render, and run V1. No model.
3. After RQ0, run a timing smoke, then the full run per eligible model.
4. Analyse once, after all shards finish.
