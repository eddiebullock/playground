# Supplementary Methods

This file holds the full text moved from Methods on 2026-11-09, with no change of content. Paragraphs, tables and sentences appear here verbatim and in their original order. Where a paragraph was split, the sentences kept in the main Methods are not repeated here, and the remainder carries a bold label marked continued, added for orientation. Sections S1 to S11 follow the order of the main Methods headings; only sections that lost material appear. Citation numbers follow the main text and references.md, and frozen tags, commits and disclosures are in the main Methods.

### S1. Human data

**EU-Emotion (continued).** The item-level workbook we hold has all 249 (`eu_emotions_oreilly_item_level.xlsx`; `n_human_items_available` in `data/eu_emotions_human_entropy.json`). Our clip list is a scan of the edited face folders in the copy we hold (`scripts/build_eu_full_manifest.py`), with no exclusion by content or result. It has no clip for seven workbook items, six of High intensity and one of Low intensity. It has two files for one workbook item, one in each of two packs, and both take that item's norms. The 243 clips therefore map to 242 norm rows, and we have 48 Low intensity items against the published 49. None of the seven carries a weak label (below). The two files are one recording encoded twice: they differ in bytes and bitrate, but have the same frame count, rate and size, and every decoded frame differs by under one grey level on average (`research/reproduction/eu_duplicate_clip_check.json`). One video therefore counts twice among the 243 clips; the frozen designs keep it. The seven are in our copy, but only as unedited originals; none has an edited version, which is what the clip list takes, and for one an edited version exists only for a different take of the same item (`research/reproduction/eu_missing_items_check.json`). We did not substitute the originals, which are trimmed differently. Whether the distributed set has edited versions of the seven cannot be told from our copy. Actor and label are only partly crossed: each actor portrayed 10 emotions plus neutral. Human response distributions are per-option face-only norms from that workbook (`data/eu_emotions_human_entropy.json`); the published paper reports recognition scores [14]. Response options were described as counterbalanced [14], but the scheme is not given and no results by position are reported, so human position effects are unknown. In our item metadata, Happy, Afraid and Disgusted came from one set of 8 actors, and Sad, Angry and Surprised from a separate set of 8 or 9. A contrast across those sets is also a contrast between actors, so it is reported only descriptively.

**Weak labels (continued).** In our item norms, face-only humans chose the target on 23 to 26% of these responses, against chance at 17%. Every result is reported against the acted label and against the human response distribution. Model accuracy far above human on these items is treated as possible non-facial leakage, not as skill.

**Snoek AU data (continued).** Observers judged 1.25 s computer-generated animations of random AU combinations on eight base faces per group, matched to observer ethnicity, with a frontal view, flat lighting and a black background [19]. The public files do not show eight codes in every set: they hold 967 for Western basic, where 15 observers saw blended identities, 8 for East Asian basic, and 4 for each conversational set (`human_au_effects.json`). Within a stimulus set, every observer saw the same fixed stimuli. Basic emotions used seven options, including "other", and conversational states used five. The public files hold 33 AU amplitude columns, recoded from 42 raw AUs. They hold no timing parameters, pose or gaze, and no stimuli. Six basic emotions and four conversational states overlap about 8 of the 21 EU-Emotion labels, and include none of Worried, Disappointed, Kind or Unfriendly.

### S2. Validity audit

**Literature check.** We tabulated option handling and image-free or mismatch controls in eight emotion papers [1,2,8-13]. Reading status is in references; some supplements stayed out of reach. Search dates and coverage are in the Introduction and references.

**Delivery audit (continued).** Rule: at least 8 tokens across the face (median over clips), a change in output when any sent frame is blanked, logged frame indices equal to the frames sent, and input ids that do not change with frame content. On composite-grid models the grid keeps the first 8 frames, so 68 of 243 clips lose frames at 1 frame per second with a cap of 16, and 16:9 frames are squashed into square cells.

**Vision gates (continued).** Option independence: activations at image positions do not change with the option list. Replay and cut: replaying a captured value changes logits by zero, and a full cut reproduces the image-free output. Token swap: swapping image tokens with a different-label clip at 10, 25, 50 and 100% has a monotonic effect, of at least 1.0 nat at 100% with an answer flip rate of at least .25.

### S3. RQ0: does the answer depend on the face?

**Design (A1), continued.** Each rotation ran twice, on the prompt below. Prompts were:

> You are performing a 6-alternative forced-choice mental state recognition task. Analyze video frames only. Choose exactly one label. OPTIONS: [numbered options]. Respond with: EMOTION: <one of the option labels exactly>, REASONING: <brief justification>

**Face-input retest (A3), continued.** An item is correct on a rotation if the target alone has the highest probability. This is the limit of A1's modal answer as draws increase, so a difference between A1 and A3 could reflect scoring as well as input. The prompt still says "video frames" although one image is sent; this was kept identical on purpose.

**Secondary outcomes.** Both tests measure dependence on the image. Neither measures competence. Secondary outcomes are accuracy by target position, correlation with human target probability overall and within category, and A3 repeated without the three weak labels and against the human modal label. The accuracy-against-1/6 *P* value pools cells as independent and is descriptive only.

### S4. Blind readout and Gate B

**Gate B criteria.** B1: a repeat forward changes log probabilities by at most 1e-3. B2: median lexicon mass is at least .05. B3: mean root-mean-square change in family CLR, real against image-free, is at least 0.25 nats. B4: the acted family's CLR is higher on the item's own face than on a mismatched face (one-sided actor-level sign-flip *P* < .01, and a 95% interval excluding 0 from above). B5: the same test for the acted family's CLR on the item against that family's mean over clips of other labels.

### S5. RQ1 on natural clips

**Cue scores (continued).** AU intensities came from LibreFace [27], and per-AU outputs recorded as intensity from OpenFace 3.0 [28], on the frames models saw. A family score is the mean of its members' z-scores, rescaled to unit variance.

**Model fit (continued).** Ridge strength is frozen at the human value, 10. A category intercept absorbs the label prior and the foils offered.

**Model responses (continued).** A clip needs a usable response on all five rotations. This route uses A1's composite-grid rotations, so its reading depends on RQ0.

A later analysis on `single_face` would be exploratory, or would need its own amendment before data.

**Recovery and detectors (continued).** Power was .775 at 0.5 times. For an observer sharing half the human weights, power was .60 at 0.75 times and .83 at 1 times. Size for an observer using different cues was .055 at 0.75 times. Proposition 4 (Section S11) gives the reason. Single cells are descriptive. Intensity estimation is weak for AU15 and AU20 in LibreFace (PCC .36 and .34 on DISFA [27]), which are outside the four families. LibreFace's intensity outputs were fine-tuned on DISFA after pre-training on EmotioNet, AffectNet and FFHQ [27]. OpenFace 3.0's AU head was trained on DISFA and BP4D, with AffectNet, gaze and landmark sets in a shared multitask model [28]. The two detectors therefore share AU training data, so their agreement is not fully independent evidence, and neither reports training on EU-Emotion.

### S6. RQ1 on randomised cues (A4)

**Stimuli (continued).** Avatars are bald and static. Of the 33 columns, 31 map directly to targets, and AU22 and AU27 are approximated in our renders. That is a local finding, since FACSHuman's own Table 9 lists AU22 as usable [29]. Amplitude maps linearly to target weight, which is an assumption, so sets at 0.75 and 1.25 times amplitude repeat the fit. At 1.25 times, amplitudes stop at 1, as the renderer's range requires, so that set compresses large amplitudes. FACSHuman's targets are LGPL-3.0 with no output clause, so no image is published.

| Set | Content | Rows |
|---|---|---|
| `main` | Every available configuration in each dataset, each on one identity: 4,659 basic WC, 2,323 basic EA, 2,318 conversational WC, 2,325 conversational EA | 11,625 |
| `replicate` | The first 200 of those on a second identity | 800 |
| `one_au`, `neutral` | Each of 33 columns alone at .25, .5 and 1, on 6 identities, plus their neutral faces | 600 |
| `scale_0.75`, `scale_1.25` | 400 configurations, basic EA and conversational EA | 1,600 |
| `blind` | The first 600 basic EA configurations, under the blind readout | 600 |

All available configurations are used, because 1,200 per dataset failed recovery. They are ranked by a seeded hash of the configuration string, with no use of responses, and that order picks the replicate, scale and blind subsets. Face gender comes from those shown to humans with that configuration, by hash.

Identities are own-group faces, as humans saw: four Western faces for WC sets and four East Asian faces for EA sets. Identity coverage is narrower than for humans, and it is not modelled beyond gender, as in the human fit. The East Asian faces are MakeHuman's macro sliders set to Asian, plus a seeded jitter for the second face. They are not based on East Asian people, and they are not independent of the Western meshes, so "own-group" is approximate for EA sets. Whether Snoek's left and right labels mean the same as the avatar's is unverified, so the confirmatory basis is the 25 families, which use the bilateral composite.

**Rendering and detector checks.** Rendering was checked with no vision-language model (`research/notes_au_render.md`): 231 checks on 33 columns by 7 identities at .25, .5 and 1 found mesh and pixel change monotonic and laterality correct. The detector check is gate V1, below, and its pilot result is disclosed there.

**Task (continued).** That prompt mentions video frames although one image is sent. There is one fixed option set per task, so options carry no information about the stimulus: Anger, Disgust, Fear, Happy, Sadness, Surprise and None of the above for basic labels; Bored, Confused, Interested, Thinking and None of the above for conversational labels. Human "other" maps to None of the above, the abstain option of A1 to A3, which stays last. The two are not the same word, so a sensitivity analysis drops that row. All cyclic rotations of the named labels are used, six for basic and four for conversational. Responses are exact option probabilities averaged over a stimulus's complete rotation set.

**Blind readout on renders.** Six families are renormalised, with no "other" row, so its CEA is in different units from the option-based CEA and the two are not compared. It is reported only if a blind gate passes: repeat change of at most 1e-3, median lexicon mass of at least .05, mean root-mean-square change in family CLR, render against pixels removed, of at least 0.25 nats on 50 stimuli, and V3's test at *P* < .01 on 200 basic EA replicate configurations.

**Human target (continued).** One ridge logistic per subject and label, one-vs-rest, was fitted on the 33 AU amplitudes plus centred face gender, with ridge strength chosen by held-out log loss.

| Dataset | Subjects | Ridge | Ceiling, raw | Ceiling, residual | WC to EA reference, raw | WC to EA reference, residual |
|---|---|---|---|---|---|---|
| Basic WC | 60 | 0.1 | .990 | .970 | .877 | .780 |
| Basic EA | 60 | 0.1 | .989 | .983 | .877 | .780 |
| Conversational WC | 20 | 1.0 | .978 | .974 | .767 | .732 |
| Conversational EA | 20 | 1.0 | .974 | .960 | .767 | .732 |

Ceilings use 200 splits across subjects and stimuli, on do effects, because models see other stimuli (`research/reproduction/a4_human_ceiling.json`). One human group against the other is the yardstick: a model cannot be asked to agree with Western observers better than East Asian observers do. The ceiling only says how far noise limits the score. It is computed from pooled fits at ridge 1.0 for every dataset, while the human target averages per-subject fits at the selected ridge (0.1 for basic), so it is the reliability of a closely related estimator, not of the target itself, and corrected values are approximate (Supplementary Note 1). The two groups saw faces of their own ethnicity, so the yardstick mixes observer culture with stimulus faces [19].

**Estimand (continued).** The model counts as one subject with fractional responses, through the same fit at the human ridge strength for that dataset, fixed in advance.

**Residual projection (continued).** A system that answers every cue with one profile, scaled by salience, can score on raw CEA without any cue-specific agreement, and it has nothing left after R.

**Raw CEA, ratios and reliability.** For raw CEA, each cue's effects sum to zero across options, so it equals Pearson *r* over entries. Each statistic is also reported over its ceiling times the square root of the model's split-half reliability across stimuli (20 splits), and that ratio is descriptive and decides nothing. The dose is the full amplitude range, on both sides. A model has no response noise, so its reliability *R_S* reflects which stimuli were sampled. The 200 replicates vary identity alone, and they serve V3, not reliability.

The one-AU set gives a direct contrast: an AU alone against that identity's neutral face. It is secondary, because it tests an AU at a neutral baseline, whereas the human effect is averaged over other active AUs. It is the only place dose can be read.

**Inference (continued).** Holm correction runs across the two eligible models within a dataset. Interval: stimulus bootstrap, 2,000 draws, 95% percentile. It treats stimuli as sampled and does not include error in the human matrix. It has no external source, and it is about 9% of shared variance.

**Not aligned (continued).** It is the null, and it requires the signal test, the gates and the equivalence bound after disattenuation. A model with no signal cannot be Not aligned: it is Inconclusive, because nothing shows the cues were delivered. A missing or non-positive *R_S* cannot support Not aligned.

The rule is asymmetric on purpose. Aligned uses uncorrected values of the confirmatory statistic, and Not aligned uses disattenuated ones. Noise can only lower a cosine, so it cannot create Aligned, whereas it could create a false null. The two claims therefore face different bars, and Aligned can be reached at a CEA where the interval is wide. Below *R_S* of .85, uncorrected CEA reads as a lower bound: Aligned still stands, and the corrected rule decides Not aligned.

**Aligned (continued).** Aligned means a reliable shared pattern, not the same strategy. A model at .31 is Aligned and far from human-level. In recovery, an observer with 16 of 25 cue families swapped in 8 pairs could still clear .30. CEA is therefore always reported against the WC to EA reference for its own statistic, and a model is called human-level only if its interval reaches that reference, which is descriptive. WC and EA share neither observers nor most configurations, but they are not independent tests of a model.

**Validity gate rules.**

| Check | Rule |
|---|---|
| V1 Detector | Across pairs of covered column and identity in the one-AU set, at least 70% have detected output rising across .25, .5 and 1 |
| V2 Image dependence | Mean root-mean-square change in option log probability, render against pixels removed, of at least 0.25 nats, on 50 stimuli |
| V3 Configuration carries | Output closer for one configuration on two identities than for a mismatch, on the Jensen-Shannon divergence [57], sign-flip over 200 configurations, one-sided *P* < .01. Mismatches are drawn within gender and ordered identity pair, so each compares the same two faces |
| V4 Domain limit | Mean mass on None of the above of at most .90, and on any option of at most .95 |

*V1 coverage.* LibreFace outputs 17 AUs, 12 with intensity (1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26) and presence alone for 7, 10, 14, 23 and 24 [27]. At least 8 of the 25 families therefore have no readout (AU11, 13, 16, 22, 27, 38, 39 and 43). The denominator is covered pairs only, and those columns are listed as unconfirmed, not as failures. LibreFace is bilateral, so laterality is not tested. It was trained on natural faces, so a failure may reflect the detector and not the render, and V1 does not separate them. Missing detector output stops the analysis and is not read as Inconclusive.

*V1 pilot (continued).* The rule counts any rise, and most are tiny. Of 96 intensity pairs, 22 gain at least .5 on LibreFace's 0 to 5 scale from amplitude .25 to 1, all in AU01, 05, 25 and 26, and 20 of these rise at every step, as the V1 rule requires (`research/scripts/registry_derived_counts.py`). AU12 at full amplitude reads about .02, so LibreFace does not see the rendered smile, and the same holds for AU04, 06, 09, 15 and 20. Of the 43 rising presence pairs, 35 stay at negative logits, so they rise without reaching presence.

*Family flag, not an exclusion.* Only AU01, 05, 25 and 26 are read by the detector with a real rise, and every other family is flagged. These four hold .14, .13, .09 and .20 of residual human energy on basic WC, basic EA, conversational WC and conversational EA (`a4_human_ceiling.json`), so most of what A4 compares is unconfirmed by a detector. All families stay in the confirmatory statistic, and a sensitivity reports the flagged and confirmed families apart.

*V4 is a weak guard.* Its .90 limit is on mean mass, not modal choice, so a heavily abstaining model can pass it.

**Behaviour beside CEA.** On the same stimuli, per model and dataset, we report two descriptive measures: output similarity (Pearson *r* between model and mean human response over stimuli and options, with option means removed), and modal agreement with chance-corrected kappa. Snoek data have no correct answer, so error consistency proper is undefined, and modal agreement is its closest analogue. Whether these measures disagree with CEA is an empirical result.

**Eligibility (continued).** Any other outcome, or a missing result, leaves it Inconclusive. Molmo2 fails the vision gates, so it is descriptive if run at all.

**Recovery (continued).** Noise was logit SD 1 per stimulus and option, with a lognormal gain error of log SD 0.5 per AU that stands in for the unknown amplitude mapping. Observers were human-like, half-shared, boundary (human weights mixed at .35), different-cues, prior-only, generic, salience-profile and pair-swapped.

| Run | Configurations | Datasets | Raw CEA | Residual CEA |
|---|---|---|---|---|
| 1 | 1,200 per dataset | Basic EA, conversational EA | Fail, 27 of 28 checks: pair-swapped accuracy at conversational EA 0.25 times, 66% | Not run |
| 2 | 1,200 per dataset | All four | Fail, 63 of 66: false Aligned on conversational WC, up to .33 | Fail, 62 of 66: pair-swapped accuracy in 4 cells, 76 to 88% |
| 3 | All available | All four | Fail, 62 of 66: false Aligned on conversational WC and at conversational EA 0.25 times, up to .40 | Pass, 66 of 66 |
| 4 | All available | All four | Fail, as run 3 | Pass, 66 of 66. Same tables and checks as run 3; only the design hash differs |

| Rate, run 3, all cells unless stated | Raw CEA | Residual CEA |
|---|---|---|
| Human-like Aligned | 1.0 | 1.0 |
| Half-shared Aligned at 1 times, lowest | .87 | .79 |
| Different-cues Not aligned at 1 times, lowest | .95 | .95 |
| False Aligned, highest | .40 | .03 |
| Prior-only signal, highest | .08 | .08 |
| Not aligned with true CEA at or above .30, highest, on the boundary observer the check uses (across all observers raw reaches .10) | .01 | .03 |

*Exempt cells (continued).* That exemption matters to the pass, because it also covers residual CEA, where pair-swapped accuracy in those two cells is 85% on conversational WC and 62% on conversational EA (`a4_recovery.json`). Counting those two cells, residual CEA would pass 66 of 68. Residual CEA also reads low on weak effects: at 0.25 times, the human-like observer reads .74 against a true .87 on conversational EA.

**Recovery limits (continued).** On conversational WC, a system with no cue-specific agreement reached Aligned in up to .40 of replicates, because about a third of human energy there lies along the common profile, and more stimuli did not help. Recovery assumes additive log-odds observers, independent stimulus noise and the gain error above. It does not model domain shift, so it shows how the analysis behaves, not how a real model will respond. It used 100 bootstrap draws and 6 reliability splits per replicate, against 2,000 and 20 in the real analysis.

On synthetic stimuli with paired cues at a constructed correlation of .9, pair-swapped observers' outputs correlated .83 to .95 with the human-weighted observer, whereas raw CEA on the randomised design was .42 to .56 (run 2). The correlation of .9 is ours, not a measurement. This shows only that the analysis can register the gap on synthetic data.

**Sensitivity.** Reported beside the primary analysis and never replacing it, each on residual and raw CEA: without AU22 and AU27; without the abstain row; flagged and confirmed families apart; amplitudes at 0.75 and 1.25 times, against the 1 times fit on the same configurations; and the direct contrast, by dose. Subsets take the common profile from the full matrix.

### S7. Replication on a second human sample (A9)

**Human data (continued).** Each animation lasted 1.25 s and was shown on a randomly generated white identity, 600 female and 600 male [60]. The files hold 24,000 responses with none missing, binary presence and continuous peak amplitude for 42 AUs, and the per-frame amplitude of each AU over 75 frames (`PREREG_A9_DRAFT.md`, Human data; trial count in `a9_human_target.json`, `structure.n_trials`). They hold no face gender or identity per stimulus and no trial order. Stimuli carry 1 to 6 active AUs, mean 3.83 (`a9_human_target.json`). The article describes 1 to 4 per animation [60], and 323 stimuli carry 5 or 6 (`PREREG_A9_DRAFT.md`, Human data). We could not explain this, and the file is used as released. Response codes 1 to 7 were read as happy, surprise, fear, disgust, anger, sad and other from the order declared in the authors' released analysis script. That order is corroborated by the same lab's later deposit [61] and by the top AU family per code in our fit: AU12 for happy, AU27 for surprise, AU20 for fear, AU09 for disgust, AU17 for sad (`a9_human_target.json`, `label_corroboration_top3_families_by_beta`). Read in this order, each Liu label's row of do effects is closest to the same label's row in Snoek basic WC, for all seven labels (cosine .536 to .947; anger .648 against .535 for disgust), and anger carries the larger AU04, AU07 and AU23 weights while disgust carries the larger AU09 and AU10 weights, which excludes the one swap the top-family check leaves open. The same lab's coding convention, applied in Chen et al.'s file, matches Snoek's text labels trial by trial with agreement 1.000 in subjects 16 to 60 (`a9_chen_overlap.json`). These corroborate the order; they did not set it.

A9 differs from Snoek et al. in observers and stimuli only. It shares the lab, the generator and its AU library, the response protocol and Western culture [19,60]. A shared artefact of that platform would therefore replicate. On the model side A9 uses A4's renderer, faces, prompt and scoring, so it is a second sample, not a second method, and we never call it independent of the GFG platform.

**Why not Chen et al. 2024.** Chen et al. [61] also released randomised AU data from the same lab. We matched its Western file to Snoek's basic WC files trial by trial (`a9_chen_overlap.json`). For Snoek's subjects 16 to 60, all 2,342 rows per subject had a unique exact match to a Chen trial, and agreement after alignment was 1.000 for all 45 subjects. For subjects 1 to 15, 2,211 of 2,368 rows matched exactly, with per-subject agreement of .964 to .9996 (median .998). Chen et al. 2024 is therefore the Snoek basic WC experiment, not a second sample, and it is not used. Its East Asian file was not downloaded, so a match with Snoek basic EA is unverified.

**AU map.** Each of Liu's 42 AU labels becomes one of Snoek's 42 raw AU names by dropping the suffixes "Open" and "i". The stripped names equal Snoek's names one for one. Each name then sets A4's 33 columns by Snoek's own recoding: a bilateral AU sets both sides, a one-sided AU its own side, a compound each part, and where two AUs set one column it takes the maximum. Applied to the raw names kept in Snoek's row labels, this rule rebuilt Snoek's 33 columns for all 6,966 configurations with no mismatch, resolving 55 collisions by the maximum (`a9_human_target.json`, `au_map_check_against_snoek`). What the suffixes mean is inferred from the names and was not checked against the generator's documentation, so the claim that "Open" adds lip parting is unverified. The Open variants are drawn with the target of their column, so any added lip parting is not drawn. All 42 AUs reach A4's columns, and all 25 families are covered. AU22 and AU27 are approximated in renders, as in A4. Peak amplitude is used, rounded to 4 decimals on both sides; one amplitude rounds to zero and is dropped (`a9_human_target.json`, `raw_amplitudes_rounding_to_zero`). Presence coding is a sensitivity.

**Human target (continued).** Subject estimates were averaged, and G-computation [56] gave do(AU = 1) minus do(AU = 0) on 25 families. The ridge was chosen by A4's rule, held-out log loss on unseen stimuli, and is fixed for the model-side fit. The model-side fit keeps A4's rendered-gender covariate. Gender is randomised against AUs on both sides, so the marginal effect estimated is the same.

| Quantity | Value | Source |
|---|---|---|
| Ridge selected | 1.0 (held-out log loss .338, against .349 at 0.1 and .344 at 3; intercept only .396) | `a9_human_target.json`, `heldout_log_loss_by_lambda` |
| Split-half *r*, subjects and stimuli split, raw | .905 (SD .015) | `a9_human_ceiling.json`, `liu_WC` |
| Ceiling, raw | .975 | as above, `ceiling` |
| Split-half *r*, residual | .862 (SD .034) | as above |
| Ceiling, residual | .962 | as above, `ceiling_residual` |
| Ceiling with stimuli shared, raw and residual, for comparison only | .987 and .983 | `a9_human_ceiling.json`, `subjects_only_same_stimuli` |
| Liu against Snoek basic WC, raw and residual | .861 and .690 | `a9_human_ceiling.json`, `human_reference_cfa.basic_WC` |
| Liu against Snoek basic EA, raw and residual | .815 and .709 | as above, `basic_EA` |
| Snoek WC against EA, for scale | .877 and .780 | as above, `snoek_basic_WC_to_EA` |
| Common profile, share of squared effects | .150 (Snoek basic .26 to .35) | `a9_human_target.json`, `human_energy` |
| AU22 and AU27, share of squared effects | .123 | as above |
| Detector-confirmed families (AU01, 05, 25, 26), share of residual energy | .143 | as above |

**Ceiling (continued).** The ceiling uses 200 splits with halves of 10 subjects and 600 stimuli that share neither, on pooled fits at ridge 1.0, as for A4. Here the selected ridge is also 1.0, so A4's caveat that its ceiling belongs to a related estimator does not arise. That comparison mixes stimuli, faces, sample and amplitude coding with observers, and it is descriptive only.

**Stimuli and design (continued).** Humans saw 1,200 different faces, so identity coverage is far narrower for models. The statistic is never switched to raw. Recovery selected B (below). Under B a configuration is kept only if all four faces have a complete rotation set, and losing over 5% of configurations fails the dataset. `frame_max` renders each configuration at the one frame of largest summed activation, a frame that did occur in the animation, whereas the peak frame may never have occurred. It is secondary and descriptive.

**Task, statistic and decision (continued).** With one dataset there is no WC and EA headline, so the per-model decision is A9's result. Holm runs across Gemma 4 and Qwen3-VL within A9 only, so A9 is its own family. Gates are A4's per model: RQ0 through the A1 and A3 table at `single_face`, the vision gates, V1 inherited from A4's pilot because the renderer and faces are unchanged, and V2 to V4 recomputed on Liu renders. The blind readout is not run. A9 names A4's family, Gemma 4 E4B and Qwen3-VL-8B, with Molmo2 descriptive if run. It does not cover the A8 models.

**Recovery (continued).** A4's two exempt cells name A4 datasets and do not carry over. Under independent face noise no A9 cell is withdrawn; under correlated face noise one cell, pair-swapped accuracy at 0.25 times, is withdrawn (lead decision 8, taken after recovery run 2 was seen), the same kind of cell A4 withdrew. Each run has 17 checks per statistic. The table gives run 4, on the final code; runs 2 and 3 gave identical tables and checks, and run 1 (draft code) the same verdicts for A and B (`a9_recovery_run1_draft.json`, `a9_recovery_run2.json`, `a9_recovery_run3.json`).

| Run | Residual CEA | Raw CEA | Failing check |
|---|---|---|---|
| A | 16 of 17 | 16 of 17 | Pair-swapped estimate within .10 of truth at 0.25 times: .64 residual |
| B | 17 of 17 | 17 of 17 | None |
| B, rho .25 | 16 of 17 | 17 of 17 | Pair-swapped estimate within .10 of truth at 0.25 times: .89, against .90 |
| B, rho .5 | 16 of 17 | 17 of 17 | The same check: .87 |

<!-- Rates: research/reproduction/a9_recovery.json, runs.<run>.n_passed, failed, and tables.residual["liu_WC|0.25"].pair_swapped.share_cfa_error_within_limit; rule .90 from settings.pass.pair_within_error_share_min. -->

**Recovery rates (continued).** Under B, the human-like observer was Aligned in every replicate at every strength, with mean residual CEA of .823 against a true .859 at 0.25 times. False Aligned among the observers with no cue-specific agreement was at most .06, and the different-cues observer was Not aligned in at least .94 of replicates. Under correlated face noise, no decision check failed: false Aligned was at most .04, the prior-only observer reached signal in at most .07 of replicates, and the human-like observer stayed Aligned in every replicate. At 0.5 and 1 times, B passed every check at both correlations (`a9_recovery.json`, runs `B_rho0.25`, `B_rho0.5`).

**Beyond the tested correlations.** Design A, one image per configuration, approximates design B with fully shared noise across faces. In every run A passed every decision check: on residual CEA, human-like Aligned 1.00 at every strength, false Aligned at most .02, prior-only signal at most .07 (run 1; at most .05 in runs 2 to 4, all under the .10 rule), boundary Not aligned at most .024, different-cues Not aligned .98 and half-shared Aligned .84 at 1 times. It failed only pair-swapped accuracy at 0.25 times, where the estimate is biased low (.346 against a true .411; under B at rho .5, .359 against .396). This is an approximation to the fully correlated case, not a tested condition. Recovery does not model face-by-AU interactions, which design B averages over four faces.

Recovery used 100 bootstrap draws, 6 reliability splits, 1,999 cue shuffles and 199 signal shuffles per replicate, against 2,000, 20, 9,999 and 999 in the real analysis (`a9_recovery.json`, `settings`). Design B was written after A4's recovery history was known, and before any A9 recovery number existed.

### S8. RQ2: stimulus interventions (A2)

**What this tests.** Our research question RQ2 asks whether a human-diagnostic cue is necessary and sufficient to move a judgement, through graded edits of one cue with verified AUs, and whether the dose-response follows the human curve. It has no AU edit, no verified dose and no human dose-response curve, and its one graded cue, eye aperture, is outside the confirmatory family because the dose is unverified. Single rendered AUs appear only under A4, on avatars, as a secondary contrast.

**Eligibility.** RQ0 is read through the A1 and A3 table. Pass on both admits both routes. Fail on A1 with Pass on A3 admits `single_face` only, and any other outcome admits neither. The vision gates must also pass. Gemma 4 and Qwen3-VL are eligible, subject to RQ0.

**Routes.** Frames were decoded and edited in memory. `single_face` sends the apex frame as a face crop at 768 by 768 pixels, with one fixed square per clip at 1.5 times the largest landmark extent. `video_face` sends 8 uniformly spaced frames through the native video processor, on a synthetic 1 frame-per-second clock, so that timestamps depend only on position in the sent order. On Qwen3-VL, native video merges pairs of frames, so order within a pair is partly lost. Landmarks came from MediaPipe FaceMesh 0.10.14, and the apex frame is chosen from landmark movement alone. A route that fails delivery checks is not analysed. A sham that runs the mask pipeline with the frame's own pixels must equal the original exactly.

**Conditions.** Spatial conditions, on `single_face`: masks of mouth, eyes, brows, and eyes and mouth together, each with three fills (the same actor's neutral clip, the clip's first frame, and a heavy blur of the region); two eye-only views; and low-pass and high-pass filters (half amplitude at 8 and complement at 32 cycles per face width). Eye-aperture warps are run but are out of the confirmatory family, because the dose is unverified. Temporal conditions, on `video_face`: two fixed shuffles, reversal, truncation to the first or last third, and the apex frame repeated. No generator is used, and this suite contains no AU edit, gaze edit, head-angle edit or playback-speed test.

**Estimand (continued).** Predictions that another label rises use that family's CLR with the acted label left out of the reference mean, since otherwise a plain fall in the target would read as a predicted rise. A bridge readout using rotated options never replaces a blind result.

**Inference (continued).** With 8 actors a one-sided *P* value cannot go below .0039.

**Predictions (continued).** Almost none of the human baselines were measured on EU-Emotion, and none on these edits. Most sources were read at abstract level (references.md). Human-source agreement is recorded per prediction in A2.

| ID | Contrast, within item | Predicted direction | Sources |
|---|---|---|---|
| P1 | Mouth mask on Happy | Acted label falls | [30] (inferred); [31] pooled over expressions |
| P2 | Mouth mask on Disgusted | Acted label falls | [30] (inferred) |
| P3 | Eye mask on Afraid | Acted label falls | [30] (inferred) |
| P4 | Eye mask on Sad | Acted label falls | [30] (inferred; the Sad cue is a non-significant brow tendency) |
| P5 | Brow mask on Angry | Acted label falls | [32] (upper face, not brows alone) |
| P6 | Mouth minus eye mask, Happy and Disgusted against Afraid, same 8 actors | More negative on Happy and Disgusted | [30] (inferred) |
| P7 | Mouth mask on Happy, Sad and Angry, Neutral family | Neutral rises | [33], with a visible mask over nose and mouth |
| P8 | Mouth mask on Disgusted, Angry family | Angry rises | [33], descriptive, with the nose covered |
| P9 | Low-pass minus high-pass on Happy | Positive | [36] (abstract only). Task dependence on other labels [37] |
| P10 | Low-pass minus high-pass on Sad | Negative | [36] (abstract only) |
| P12 | Mean of shuffles on Happy and Surprised | Acted label falls | No direct shuffle source. Analogy from [39] (disrupted rhythm, subtle expressions, abstract only). Mixed: [50] (raw threshold cost of shuffling, no efficiency cost) |
| P13 | Mean of shuffles on 48 Low Intensity items | Acted label falls | No direct shuffle source. Analogy from [40,41] (dynamic against static, abstract only). Mixed: [50] (raw threshold cost of shuffling, no efficiency cost) |
| P14 | First minus last third, Surprised family on Afraid and Afraid family on Surprised | Confusable label higher on the early third | [42], ideal-observer classifier on Western observers' expectation models |
| P15 | First minus last third, Disgusted family on Angry and Angry family on Disgusted | Confusable label higher on the early third | [42], as P14 |

**Prediction sources.** P11, on eye aperture [43], is outside the family until its dose is verified. Directions for P1 to P4 and P6 are inferred, because [30] is a reveal study in which no region was masked with the rest visible. [33] used a visible community mask over nose and mouth on static photographs, whereas our mouth masks use neutral, first-frame or blur fills and may leave the nose visible. No cited source tests frame shuffling for P12 or P13, and the closest human test is mixed: shuffled order raised raw recognition thresholds, which the authors attribute to shuffled clips carrying less physical information, while efficiency relative to an ideal observer did not change and reversed order showed no cost [50]. A VLM study also found almost no change under frame shuffling [11]. P14 and P15 are analogues, not replications, of an ideal-observer analysis of expectation models on synthetic faces [42]. Those classifiers accumulate signals, so a stand-alone last third has no counterpart in that paper, and nothing shows that thirds of an EU-Emotion clip map onto its time bins. Eye-only views have no directional prediction, because sources differ by label type: [45] finds the whole face better than the eyes for basic labels and the two equal for complex ones, and [31,46] are read at abstract level; masks on complex labels are descriptive [46,47]. Reversal describes whether order changes the model's judgement, and we give no human direction for it. No category flip is predicted from head angle, here or later, because we located no human study showing one. A downward tilt raising perceived dominance is the located evidence [49], and this suite does not edit head angle.

**Null rule (continued).** Those contrasts read as a match, an opposition or Inconclusive, and A2 cannot show that a model ignores those cues.

**Recovery.** Synthetic observers (prior-only, frame-bag, mouth-only, eyes-only, human-like) ran through the analysis on the real 243 items and actors, 100 replicates each. All five passed the stated rule. Recovery used large effects and independent noise, so it shows that the pipeline works, not its power at realistic effects.

**Sensitivity.** Reported beside the primary analysis: without smoke items, without flagged alignment, on canonical words only, and on clips with at least 8 tokens across the face. Results are also shown for the human modal label and as Jensen-Shannon divergence from the human distribution on the item's options. Humans did not see these edits, so a change in divergence is descriptive and says nothing about what they would do.

### S9. Larger models (A8)

**Models (continued).** Each has the model class of its small sibling, so each runs through the frozen scripts under the existing code key, with a different weights directory and different output paths, and no code edit. Gemma 4 31B also differs from E4B in more than size. Both ran at bfloat16. They were chosen on licence, size, family and storage before either produced any output on a study stimulus. A third family is future work and would need its own amendment.

**Storage and integrity.** The Qwen3-VL-32B weights are read in place from a copy on the lab's project store that another project also uses. The Gemma 4 31B weights were downloaded once to the same store, with the user's approval. The size and sha256 of every weight file at the pinned revision are in the design block. Every file matched before any A8 job: 14 of 14 for Qwen3-VL-32B and 2 of 2 for Gemma 4 31B (`PREREG_A8_DRAFT.md`, Freeze). Each job rechecks names, sizes and the resolved path, and a full hash runs again after the last A8 job. All A8 output goes to separate folders, and the frozen outputs of the small siblings are hashed before and after each job, so any overwrite would fail the job.

**Which frozen pipelines.** Vision gates (G2 to G4), input delivery, the readout ladder (descriptive; the readout is not re-chosen), A3, A1, natural-clip RQ1 (descriptive only), A2 and A4, each through the frozen script with the frozen arguments, launched by one entry point that checks every frozen file against its pinned hash. A5, A6 and A7 do not run on A8 models (`PREREG_A8_DRAFT.md`, How every A8 job runs), and A9 names only A4's family.

**Sharded A1 (continued).** Splitting the item list is not allowed, because the frozen script builds mismatch partners over the items it loads. The shard runner therefore loads the full manifest in every shard, so partners are built over all 243 items as in the unsharded run, and restricts only the scoring loop. Draws are seeded per trial, so a record does not depend on its shard. Two equivalence tests check this. On the frozen code with model, media and generation stubbed, all 243 items in 4 shards, merged, were identical to the unsharded run. An out-of-memory error in any trial fails its job, which is rerun with more GPUs before any merge, so memory headroom cannot decide which trials score. All shards of a task run with the same GPU count.

**Eligibility (continued).** A model that fails either reports "A1 not run, by the A8 stop rule", whatever A1 might show. RQ0 is then read through the frozen A1 and A3 table: A3 Pass with A1 Pass admits both routes, A3 Pass with A1 Fail admits `single_face` only, and anything else, including a missing result, admits neither. A2 needs RQ0, the vision gates, delivery and Gate B by A2's rules. A4 needs RQ0, the vision gates and V1 to V4 by A4's rules. Gemma 4 31B's gates are strict: if the frozen gate adapter fails on it, it has no gate result and is ineligible. The launcher checks eligibility from the A8 files before A1, A2 and A4 runs, records the decision and refuses an ineligible model.

**Families (continued).** A2 keeps Westfall-Young step-down within model and route. With one model in a family Holm is the identity, and with two it is Holm over two. Holm controls familywise error at .05 at any size, but a smaller family is less strict per model than the frozen three-model family. A8 models are never pooled with the frozen models, with each other or with their small siblings.

**Reporting.** A8 rows are labelled A8 and placed under the frozen rows, whatever they show. A8 cannot rescue or weaken a frozen verdict: a frozen Inconclusive stays Inconclusive, a frozen null stays a null, and Molmo2 stays ineligible. Where an A8 model and its sibling disagree, both verdicts are reported and the difference is described, not tested. A8 may support a statement of scope, never a change to a frozen verdict.

### S10. Scoped and provisional questions

**RQ3 (continued).** Such a cut can only check that no route from pixels to the answer was missed. Holding image positions past the entry of the language model cannot be a complete cut, because text positions have by then read the image, and holding those too returns the image-free forward. The depth at which a cue is carried is therefore not testable on either eligible model. No site has yet been shown complete on `single_face` or on the blind readout, because the vision gates ran on the legacy input with the option prompt.

**A7 Stage 0 (continued).** It would establish whether the whole-path cut and its two named blocks are complete on that input, and whether the tokens at a cue's location can act as a cut for that cue. Stage 0 is costed at about .71 GPU-hours, an upper bound. Whether a block of a complete cut meets our rule that only complete cuts count is open to A7's audit. If it does not, the report is the Stage 0 table plus "not testable". A comparison of the two blocks would be an optional supplement, and it has no human reading. A pass on the region test would give a non-vacuous cue-level answer, and the question would be reopened by a dated entry.

**RQ4 and RQ5 (continued).** A5 addresses RQ4, how much of a judgement comes from face evidence and how much from label priors, by separating prior from evidence with image-free, grey and mismatch baselines on the blind readout. A6 addresses RQ5, whether a shared error has the same cause in humans and models, on CREMA-D, across its fixed option set, with a cue-targeted test of one human confusion.

### S11. Identifiability results

**Setup (continued).** Each row sums to zero. CEA(*S*, *T*) is the cosine between *tau_S* and *tau_T*, and it is undefined if either matrix is zero. A behavioural metric is any functional of the joint distribution of items and responses on *P*. This covers accuracy, confusion matrices, error consistency and representational similarity on outputs. A4 and the human matrix use the set-to form, do(AU = 1) minus do(AU = 0), with other AUs at their observed values. For a linear observer this equals the dose form at dose 1, and Propositions 2 and 3 hold unchanged. Theorem 1 is proved for the dose form. Its construction carries to the set-to form when neither intervened point is a natural item, and that case is not covered here. These results are stated for CEA as defined here. A4's confirmatory statistic, residual CEA, removes from each system's matrix the component along that system's own mean option profile, and is the cosine of what remains. Supplementary Note 1 defines it from the frozen code and states which results carry over: Corollary 1.1 carries when (K - 2)(p - 1) >= 2; Proposition 2 carries only in a three-option variant; the upper bound of Proposition 3 needs a shared mean direction and the lower bound fails; and Proposition 5 is approximate, because the projector is estimated from the data.

**Proposition 2.** With two standardised cues correlated at rho, a linear observer on each cue gives output correlation rho, yet CEA of 0 at equal doses. As rho approaches 1, behaviour converges while alignment stays at 0.

**Proposition 4.** If a system responds to an unmeasured cue *U* that is correlated with measured cues, a regression on measured cues gives non-zero slopes, while intervening on the measured cues alone has no effect. Regression on natural clips therefore cannot reach a claim about cue use.

**Proposition 5.** With independent zero-mean errors in both matrices, the observed cosine approximates CEA times the square root of the product of their reliabilities, which is Spearman's attenuation correction applied to a matrix [51]. A4 estimates human reliability by split-half across subjects and stimuli, and model reliability by split-half across stimuli, because a model has no response noise. When models see other stimuli than humans did, the subjects-and-stimuli ceiling applies. Corrected values above 1 can occur from sampling and are reported uncorrected beside them.

**Cue spectra.** These describe the designs and use no model output. Correlation matrices of each cue table gave the values below (`research/reproduction/cfa_identifiability.json`). Snoek rows count unique rows of the 33 amplitudes, whereas A4's manifest counts configuration strings, so basic EA, conversational WC and conversational EA have 2,322, 2,314 and 2,322 rows here against 2,323, 2,318 and 2,325 there:

| Cue table | Rows | Cues | lambda_min | Condition number | 1 / lambda_min |
|---|---|---|---|---|---|
| EU-Emotion, LibreFace | 243 clips | 15 | .054 | 76.1 | 18.4 |
| EU-Emotion, OpenFace 3.0 | 243 clips | 10 | .125 | 22.4 | 8.0 |
| Snoek basic WC | 4,659 configurations | 33 | .178 | 13.7 | 5.6 |
| Snoek basic EA | 2,322 | 33 | .199 | 11.3 | 5.0 |
| Snoek conversational WC | 2,314 | 33 | .209 | 11.0 | 4.8 |
| Snoek conversational EA | 2,322 | 33 | .199 | 11.3 | 5.0 |

With left and right merged, the Snoek designs have 25 cues, lambda_min of .44 to .47, condition numbers of 3.4 to 3.5 and 1 / lambda_min of 2.1 to 2.3. Most of the Snoek spread in the table therefore comes from bilateral pairs, which are correlated by design. Cue counts differ between rows. EU-Emotion cues are detector estimates, so measurement noise changes their spectrum, and clip-level means ignore within-clip variation.
