# Citation check, draft v2

Run 2026-11-07. Checks `introduction.md` and `methods.md` against the sources in `references.md`. Neither draft was changed: every correction is listed here. No A1 result was read.

**How it was read.** Open copies were saved as text and read in full by parallel reading passes, one reference at a time, with no summarising fetch tool. Every quote below was then checked by script against the saved text after normalising whitespace and case, so quotes are verbatim apart from PDF extraction damage. The "wrong" verdicts and the #2 option-order point were spot-checked again by the coordinator. Verdicts are: supported, needs rewording, wrong. "Not in paper" is filed under needs rewording, and means the number may be right but this source cannot carry it.

## Applied 2026-11-07

Every "wrong" row is fixed in `introduction.md`, `methods.md` and `references.md`, and every "needs rewording" verdict below is applied, with these exceptions and notes.

- **Left open for source access:** #18 is dropped from the reverse-correlation sentence and #42 cited in its place, so #18 no longer carries that claim. #2's order claim follows the suggested wording; OSF was not read.
- **Not changed, because the A2 and A4 texts govern them:** P6's "same 8 actors" and P13's 48 items stay, as local metadata facts. The V1 denominator is for A4. A2's frozen prediction rows stay as written, and the correction is in its Deviations section.
- **Item 9 resolved.** The Snoek counts are right. Observers within a stimulus set saw the same fixed stimuli, about 2,350 each, so 140,910 trials over 60 observers give 4,659 AU configurations across two sets (`human_au_effects.json`). The small gaps between 2,323 and 2,322 and so on come from counting configuration strings against unique amplitude rows after recoding, which is an inference.
- **Item 8.** Introduction now names the local norms for 23 to 26%, and the published scores as low. Methods names local metadata for the actor sets.
- **2026-11-13, for length (outline.md, "Length check", cut 4).** The paper-by-paper list in the `introduction.md` paragraph starting "We then asked" was removed, because Figure 1e codes the same eight designs one row per paper, from `research/reproduction/literature_table.csv`, with counts in Results, Section 1. The claims below that quote that list now record a check of text no longer in the draft. The search-coverage sentences moved to Supplementary Methods, Section S2, with the statement they support named; their wording is otherwise unchanged.
- **Gold et al. (2013) is [50].** Abstract read through the Europe PMC API. The journal page returned 403 and the PMC page a reCAPTCHA, so both were left. It reports equal efficiency for dynamic, reversed, randomised and static expressions, so the full text remains a priority.

## Result

| | Count |
|---|---|
| Status upgraded | 33 (31 to read in full, #24 to partial, #26 to abstract) |
| At read in full now | 33 of 49 (was 3, one of them #14 through a summarising tool only) |
| The 44 on the must-read list | 30 now read in full, 14 still below: #24, 31, 32, 36, 37, 39, 40, 41, 42, 43, 44, 46, 47, 49 |
| Still below read in full | 16: #18, 24, 26, 31, 32, 36, 37, 39, 40, 41, 42, 43, 44, 46, 47, 49 |
| Blocked by a refusal | 5: #18, 26, 31, 46, 47 (plus the published version of #1) |
| No open copy found, no refusal | 9: #32, 36, 37, 39, 40, 41, 43, 44, 49 |

## Claims that are wrong

| # | Where | Draft says | Problem | Fix |
|---|---|---|---|---|
| 1 | `methods.md`, Human data | "Human presentation order is not confirmed from the source" | [14] says "six counterbalanced response options" | "Response options were described as counterbalanced [14], but the scheme is not given and no results by position are reported, so human position effects are unknown." |
| 2 | `introduction.md`, second problem | "Behavioural metrics such as error consistency [15] compare outputs on natural items." | Three of [15]'s four stimulus sets are out-of-distribution, and it compares correct/incorrect patterns, not outputs | "Behavioural metrics such as error consistency [15] compare responses on a shared item set. On natural items, this comparison cannot settle cue use." |
| 3 | `introduction.md`, second problem | "limits of a related kind have been discussed elsewhere [16,17]" | [17] is a positive concept-testing framework and states no such limit. [16] holds, loosely | Cite [16] only. Sharper: "Empirical limits on identifying architecture from activation similarity have been shown for neural networks [16]" |
| 4 | `methods.md`, P7, and row 34 | Sources [33-35] for Neutral rising | [34] reports overall accuracy only: no Neutral response, confusion or per-emotion result | Remove [34] from P7. Row 34 used-for becomes overall mask accuracy loss |
| 5 | `methods.md`, P12, and row 38 | Source [38] for shuffles lowering Happy and Surprised | [38] tests dynamic against static under phase noise. There is no shuffled condition | Remove [38] as a shuffle source. It may stay only as a stated analogy from a dynamic advantage |
| 6 | `methods.md`, P13, and row 38 | Source [38] for shuffles lowering Low Intensity items | [38] varies phase signal, not expression intensity, and tests no shuffle | Remove [38] from P13 |

## Corrections that matter most, short of wrong

**1. #12 ran a no-image control. Every earlier note missed it.** Supplement Sec. S14 replaces all 2,052 FACES photographs with a gray rectangle under both scoring conditions, and all 22 runs answer neutral throughout. The main paper does not describe it, so a read of the main paper alone finds nothing (the supplement was fetched from arXiv's ancillary files for this reason). This changes the audit table, `LIT_CHECK_2026-11-07.md` section 2, and the Introduction:

- The literal claim survives, because a gray image is not a swapped face and the text arm kept a fixed option list. But it now survives narrowly.
- Replace "In the seven we could check, we found none that controlled option position and also swapped the face under a fixed prompt. The nearest is a design that crossed neutral text with paired images [13]." with: "Across the eight, we found none that estimated the accuracy lost when the face is swapped under a fixed prompt, with option position controlled or absent. The nearest are a design that held neutral text fixed across an emotional image and its generated neutral twin [13] and a no-image control under fixed prompts [12]."
- Replace "one did not control it [12]" with: "one scored its main result from a yes/no question per category, so no option position arises, and kept one fixed option list in its text arm [12]."

Other absence claims rest on the same risk where a supplement stayed out of reach: #1 (Tables S2 to S7), #2 (OSF materials), #3 (prompts), #34, #35 and #38. #11 was read with its supplement.

**2. "So most of the VLM emotion claims we located cannot separate perception from option order and prompt text."** This does not hold as written. By paper: #9 appears to avoid both faults; #10, #12's headline result and #13 have no option position and one prompt; #8 has the option-order issue but not the prompt-text fault; #1's inversion arm shows image dependence. That leaves #1, #2, #11 and #12's text arm as the clear cases. Suggested: "So several of the VLM emotion claims we located cannot rule out option order or, where option sets differ by item, prompt text."

**3. Opening claim [1-3].** "Such models match or beat people" is too general. [1] is GPT-4o on upright eye-region images. [2] is one of three models, against norms. [3] is two of three, and Claude fell below humans. [2] and [3] have no same-trial test against humans. [3] and [11] also report the mixed record that the opening omits. Suggested: "some such models".

**4. Audit sentences now stale.** After this read, change "mostly through targeted passages", "we did not check its baselines" (see #10), "In the seven we could check", "Most of the papers above were not read in full" and, in `methods.md`, "Most were read as targeted passages."

**5. #2 "fixed it".** The paper never mentions option order. Its sentence is about "the tests", which reads as item order, so `LIT_CHECK` over-read it. Suggested: "one fixed it [1], and one kept the original test order and reported no option randomisation [2]", or confirm from OSF.

**6. "The VLM emotion work ... sits almost entirely at the first degree."** Several papers compare no human data at all (#11, #13) and others go below behaviour (#9, #10, #22). "Compares scores, at most the first degree" is safer. CFA is also an instance of [4]'s second-degree benchmark, and [4] has the multiple-realizability argument, so cite [4] in the identifiability paragraph.

**7. Option effects sentence.** [6] is an oracle gap of about 13 to 75 points between best and worst orderings on two text models, not an arbitrary reorder, which moved accuracy by at most 8 points there. [7] is one model and one suffix: Qwen2.5-VL-7B, CoT-prefix, 93.54% against 51.71%, with first slot always labelled A.

**8. Numbers that [14] does not contain.** 23 to 26% (paper gives chance-corrected 9, 9 and 14%, and 13% in prose), the 8 or 9 actor-set split, and 48 Low items (paper has 49). Name the local norms as the source. "Face-only norms" as a response distribution also needs its source stated.

**9. Check `Snoek` configuration counts.** The Cue spectra rows (4,659, 2,322, 2,314, 2,322) nearly equal [19]'s counts of repeated trials (4,658, 2,322, 2,314), yet about 140,000 random trials should give far more unique configurations. Check whether the spectra, and so the A4 manifest, used a repeated subset only.

**10. Detector claims.** OpenFace 3.0 [28] describes AU detection, not intensity, and lists no AUs. "Detection is weak for AU15 and AU20" has support only from [27], as intensity PCC. The V1 gate may be unreachable: [27] outputs 17 AUs, 12 with intensity, against 25 families. The [PENDING: training-data lineage] can now be filled (see #27, #28), and the two detectors share DISFA, BP4D and AffectNet.

## The Predictions table and the A2 freeze

A2 is frozen, so the fixes below concern how sources are described. They do not license changing a prediction. Disclose in Methods that source checks after the freeze changed the stated support.

| Prediction | State after this check |
|---|---|
| P1 to P4, P6 | [30] is a reveal study, so directions are inferred, not measured. P4 is weakest: the Sad cue is a non-significant brow tendency. [31] is abstract only |
| P5 | [32] tested upper and lower face, not brows (abstract only). [30] puts anger at the eye, AU7 |
| P7 | Rests on [33] alone. [34] gives nothing, and [35] only restates [33] (its S1 Table is unread). Carbon's mask covers nose and mouth |
| P8 | [33] confirmed, nearly 38% against 2%, descriptive only, nose covered |
| P9, P10 | Direction supported by the abstract of [36] only. [37] concerns fear, disgust and anger, not happy or sad |
| P12 | No cited source tests shuffling. [39] disrupted rhythm on subtle expressions (abstract only). [38] is out |
| P13 | No cited source tests shuffling. [40] and [41] show dynamic against static on subtle expressions, abstract only. [38] is out |
| P14, P15 | Confusions come from Bayesian classifiers on models from 60 Western observers, with cumulative signals. They are not human responses over time, and a last-third clip has no counterpart. "Stated human predictions" needs softening |

**Evidence against P12 and P13 that the draft omits.** [38] reports Gold et al. (2013) as finding identical recognition for static, dynamic, shuffled and reversed expressions. [11] shuffles frames on VLMs including Qwen3-VL with almost no macro-F1 change. [38] also shows a static advantage for fear, a risk for P13 if Afraid is among the Low items. Gold et al. is not in the reference list and needs a first-hand read.

## Nature Human Behaviour rules

All read on 2026-11-07 from nature.com public pages.

**Article limits** ([content types](https://www.nature.com/nathumbehav/content)):

- "Main text – up to 5,000 words, excluding abstract, Methods, references and figure legends."
- "Abstract – up to 150 words, unreferenced."
- "Display items – up to 8 items (figures and/or tables)."
- "Results and Methods should be divided by topical subheadings; the Discussion does not contain subheadings."
- "References – we do not place limits on reference lists."
- Order is Introduction "(without heading)", Results, Discussion, Methods, so the note at the top of `introduction.md` is right. The Introduction is about 1,140 words of the 5,000.
- The page does not say whether Methods tables count as display items. `methods.md` has six tables. Ask the editors, or move design tables to the Supplement.

**Preregistration** ([policy](https://www.nature.com/nathumbehav/editorial-policies/preregistration-policy)):

- "For non-clinical confirmatory research, we strongly encourage study preregistation with an analysis plan." So an external registry is not required for an Article, and the [PENDING: external registration link if the journal requires one] can be settled.
- "If any of the reported studies was preregistered, authors must provide an active link to the preregistration in the Methods section and state the date of preregistration." So A1 and A3 need linkable commits and dates: their [PENDING] hashes are a condition of acceptance.
- "Authors must disclose all deviations from the preregistered protocol and explain the rationale for deviation (e.g., flaw, feasibility, suboptimality)."
- "In cases of deviation from the preregistered analysis plan for reasons other than fundamental flaw or feasibility, the originally planned analyses must also be reported." Check Disclosure 5: the unit change follows failed recovery, but the strength rule "tightened after power ... was known" reads as suboptimality, and the original rule may then need reporting.
- "Manuscripts reporting preregistered research will not be accepted for publication until they meet these requirements."
- Whether a tagged commit in a public repository counts as a preregistration is not stated. Ask the editors, or deposit the amendments with OSF.

**Registered Reports** ([guidelines](https://www.nature.com/nathumbehav/submission-guidelines/registeredreports)):

- "Please note that we are unable to consider projects where data collection has already started or is already complete, as one of the key aims of the format is to optimize research designs before the work is carried out."
- "For frequentist analysis plans, the a priori power must be 0.95 or higher for all proposed hypothesis tests."
- Secondary analyses need evidence "that they have had no prior access to the data in question". The Snoek data have already been analysed here.
- A pilot is required for novel paradigms, and protocols are registered after acceptance in principle.
- Reading: the work as drafted cannot go in as a Registered Report. It would be an Article.

**Reporting summary** ([reporting standards](https://www.nature.com/nathumbehav/editorial-policies/reporting-standards)): "Authors of research articles in the life sciences, behavioural & social sciences and ecology, evolution & environmental sciences are required to provide details about elements of experimental and analytical design that are frequently poorly reported in a reporting summary that will be made available to editors and reviewers during manuscript assessment." This settles the [PENDING] at the end of `methods.md`: plan on providing one.

## Access record

| # | Tried | Outcome |
|---|---|---|
| 1 (published) | ScienceDirect, through a URL guessed from the pattern, not the DOI | HTTP 403. Stopped. arXiv version read instead |
| 18 | Europe PMC PDF link; PMC page | HTTP 403; reCAPTCHA browser check. Stopped |
| 24 | Europe PMC PDF link; PMC page | HTTP 403; PMC page served normally |
| 26 | Europe PMC PDF link; PMC page | HTTP 403; reCAPTCHA browser check. Stopped |
| 31 | CORE download | HTTP 403. Stopped |
| 46 | CORE record | HTTP 403. Stopped |
| 47 | Journal of Vision through the DOI | HTTP 403. Stopped |

No spoofed user agent, cookie store, proxy or shadow library was used. Two things to review: after Europe PMC refused #18, #24 and #26, the separate NCBI PMC site was tried once as a different public host. And nature.com and link.springer.com redirect through an identity host, then serve the page to clients without cookies, with `error=cookies_not_supported` in the URL. Those pages were fetched by following plain redirects, with no cookie sent or stored. A university proxy mirror of #44 appeared in search results and was not used. `research/LIT_CHECK_2026-11-07.md` was read for context, as `references.md` depends on it. No other repo file was read.

## Leads, unread, not for citing yet

These are named by the papers above. None was read, so none is added to the reference list, and any detail here is as the citing paper gives it.

- Gold et al. (2013), shuffled and reversed expressions, via [38]. Highest priority.
- Daube et al. (2021), DNN against human causal features on generated faces, via [4]. A non-VLM precedent for CFA.
- Atabuzzaman et al. and Tian et al., VLM option bias, and Rosenthal et al., via [7]. Possibly stronger than [7] for the VLM half of the option sentence.
- Cunningham and Wallraven (2009), forward against reversed recognition, via [48].
- CausalAffect, AULLM++, EDCT and TAG, via [20], and DK-CLIP, PE-CLIP, ExpLLM and AU-DFER, via [11]. AU-informed models that the novelty search should have seen.
- `references.md` key and "Must read" section are now out of date. They were left unchanged because only the status column was in scope.

## Abstract-level checks

Status unchanged or abstract only. No full text was read, so "supported" here means only that the abstract agrees.

### #18 Jack et al. 2012
- Where read: Europe PMC abstract (PMID 22509011). Full text blocked.
- Claim: `introduction.md` "In reverse-correlation studies, observers judge faces whose facial action units (AUs) are set at random [18,19]."
  - Verdict: needs rewording
  - Quote: "Using a unique computer graphics platform that combines generative grammars [Chomsky N (1965) MIT Press, Cambridge, MA] with visual perception, we accessed the mind's eye of 30 Western and Eastern culture individuals"
  - Note: The abstract fits but never states randomised AUs or reverse correlation. Add [42], which does.

### #26 Cao et al.
- Where read: Europe PMC abstract (PMID 25653738). Full text blocked.
- Metadata: journal confirmed, IEEE Transactions on Affective Computing, 2014.
- Claim: `methods.md` "24 CREMA-D [26] clips"
  - Verdict: supported
  - Quote: "7,442 clips of 91 actors with diverse ethnic backgrounds were rated by multiple raters in three modalities: audio, visual, and audio-visual."

### #31 Blais et al.
- Where read: Europe PMC abstract (PMID 22974675). Full text blocked.
- Claim: `methods.md` P1, mouth mask on Happy, source [30,31]
  - Verdict: needs rewording
  - Quote: "We found that the mouth area is the most important cue for both static and dynamic facial expressions."
  - Note: Pooled over eight expressions in a Bubbles task. Nothing specific to Happy, and no mask.
- Claim: `methods.md` "accounts conflict [31,45,46]"
  - Verdict: supported
  - Quote: "However, we found an underutilization of the eye area by human participants in comparison to the ideal observer."
  - Note: Runs against the view of eyes as primary, so it does stand against [45]'s complex-state result.

### #32 Ponari et al.
- Where read: Europe PMC abstract (PMID 22642357). No open copy found.
- Claim: `methods.md` P5, brow mask on Angry
  - Verdict: needs rewording
  - Quote: "We found that the neutral lower half-face interfered with recognition of happiness and disgust, whereas the neutral upper half impaired recognition of anger"
  - Note: Upper half, not brows. The same sentence gives support to P1 and P2, and "recognition of fear and sadness was impaired by both manipulations", which complicates P3 and P4 as eye-only effects.

### #36 Kumar and Srinivasan
- Where read: Europe PMC abstract (PMID 21942699). No open copy found.
- Claim: `methods.md` P9, P10
  - Verdict: supported
  - Quote: "As expected we found that the identification of happy expression was dependent on low spatial frequencies and the identification of sad expression was dependent on high spatial frequencies."

### #37 Smith and Merlusca
- Where read: Europe PMC abstract (PMID 24708495). No open copy found.
- Claim: `methods.md` P9 "(use depends on task [37])"
  - Verdict: needs rewording
  - Quote: "we determined the facial features crucial for the categorization of three key facial expressions of emotion (fear, disgust, and anger)"
  - Note: Task dependence is supported, but not for Happy or Sad, the labels in P9 and P10.

### #39 Bould et al.
- Where read: Solent repository record abstract. No open copy found.
- Claim: `methods.md` P12, shuffles on Happy and Surprised, source [38,39]
  - Verdict: needs rewording
  - Quote: "Experiments 2a and 2b then changed the dynamics of the moving sequences by speeding up, slowing down or disrupting the rhythm of the motion sequences."
  - Note: The closest source to a shuffle, and it is now P12's only one, but it concerns subtle expressions. It suits P13 better than P12.

### #40 Ambadar et al.
- Where read: Europe PMC abstract (PMID 15869701). No open copy found.
- Claim: `methods.md` P13, shuffles on Low Intensity items
  - Verdict: needs rewording
  - Quote: "Participants viewed faces depicting subtle facial expressions in four modes (single-static, multi-static, dynamic, and first-last)."
  - Note: No shuffled mode. The account is perception of change, and a shuffle keeps change, so a fall is not implied.

### #41 Bould and Morris
- Where read: Europe PMC abstract (PMID 17535474). No open copy found.
- Claim: `methods.md` P13, shuffles on Low Intensity items
  - Verdict: needs rewording
  - Quote: "A second experiment demonstrated that the advantage of movement was reduced with expressions of a higher intensity"
  - Note: Intensity interaction supported. The contrast is dynamic against masked or single stills, not shuffled frames.

### #43 Lee and Anderson
- Where read: Europe PMC abstract (PMID 28406382). No open copy found.
- Claim: `methods.md` "P11, on eye aperture [43,44]"
  - Verdict: supported
  - Quote: "opposing features of eye widening versus eye narrowing that regulate sensitivity versus discrimination not only conveyed their associated basic emotions (e.g., fear vs. disgust, respectively)"

### #44 Lee et al.
- Where read: Europe PMC abstract (PMID 23620549). No open copy found.
- Metadata: authors Lee, Susskind and Anderson.
- Claim: `methods.md` "P11, on eye aperture [43,44]"
  - Verdict: needs rewording
  - Quote: "We present evidence that this benefit was driven by neither the perceived emotion nor attention but, rather, by an enhanced physical signal originating from greater exposure of the iris and sclera."
  - Note: The benefit is to gaze discrimination and locating targets, not fear recognition. Do not cite it for a label effect.

### #46 Cunningham et al.
- Where read: Crossref abstract. Full text blocked.
- Metadata: journal is ACM Transactions on Applied Perception, 2005.
- Claim: `methods.md` "accounts conflict [31,45,46]"; "masks on complex labels are descriptive [46,47]"
  - Verdict: needs rewording
  - Quote: "The results show that most expressions rely primarily on a single facial area to convey meaning with different expressions using different areas."
  - Note: Nine conversational expressions, with parts frozen in video, so it fits the second clause. It says nothing on eye-only views of basic labels.

### #47 Nusseck et al.
- Where read: Europe PMC abstract (PMID 18831624). Full text blocked.
- Claim: `methods.md` "masks on complex labels are descriptive [46,47]"
  - Verdict: supported
  - Quote: "Subsequent analyses of the facial movements and correlation with recognition performance show that, for some expressions, one individual facial region can represent the whole expression."

### #49 Witkower and Tracy
- Where read: Europe PMC abstract (PMID 31009583). No open copy found.
- Claim: `methods.md` "A downward tilt raising perceived dominance is the located evidence [49]"
  - Verdict: supported
  - Quote: "In five studies (total N = 1,517), we found that when eye gaze was directed forward, tilting one's head downward (compared with a neutral angle) increased perceptions of dominance"
  - Note: Neutral faces with forward gaze.

## Full-text checks

Order follows `references.md`.

### #1 Strachan
- Where read: https://arxiv.org/abs/2410.22309 (arXiv version; published version blocked)
- Coverage: read in full (references skimmed). This is the arXiv 2410.22309 version, NOT the published Computers in Human Behavior version, which was not compared. Every verdict below therefore applies to the arXiv text only. PDF extraction: ligature damage in parts (abstract, Methods), equations garbled, figures absent. Tables S2 to S7 are cited but not in the text, so the text gives no numeric accuracy for GPT-4o or humans.
- Metadata: fix: title and first author match. Venue, volume, article number and DOI cannot be checked from this text, because it is the preprint. The Status cell can move to "read in full (arXiv version); published version not compared". "Partial" is out of date for the arXiv text.
- Claim: (introduction.md para starting "Vision-language models (VLMs) can take") "such models match or beat people on tests of emotion and mental-state recognition from faces, including the ... RMET [1-3]"
  - Verdict: needs rewording
  - Quote: "Together with the analysis of the RMET, these findings demonstrate that GPT-4o outperformed humans when reading mental states from upright faces and exhibited a vulnerability to face inversion that was even more pronounced than in humans." (Results, end of Effect of stimulus sex)
  - Note: "Beat" holds for one model (gpt-4o-2024-08-06), on upright items only; on inverted items GPT-4o was worse than humans. Not RMET only: it used the RMET (N = 51 humans, 25 runs per orientation) and the MRMET (N = 400 humans, 200 runs per orientation). Stimuli are eye-region crops, not whole faces. Correction: name GPT-4o and say "upright eye-region images".
- Claim: (introduction.md para starting "We then asked") "Two RMET studies fixed it [1,2]"
  - Verdict: supported
  - Quote: "The order of images was fixed in line with the published standards of the two tests, as was the order of alternatives for each item." (Methods, Testing protocol)
  - Note: Literally true for humans and GPT-4o on both tests. The text does not state where the target sits. Humans and model shared the fixed order, so it is not a human-versus-model difference. No test of option order was run.
- Claim: (introduction.md para starting "We then asked") "we found none that controlled option position and also swapped the face under a fixed prompt"
  - Verdict: supported
  - Quote: "For GPT-4o, orientation was manipulated between-runs such that all images within a given test run would be presented upright or inverted." (Methods, Testing protocol)
  - Note: Absence claim. I looked for image-free, blank, mismatched, swapped and shuffled-image arms under any name and found none. Option position was not controlled and no face was swapped. Inversion is the only image manipulation. The other "random" or "control" items act on analysis, not stimuli: permuted alternatives for I_conc and the subsampled human control.
- Claim: (introduction.md para starting "We then asked") "So most of the VLM emotion claims we located cannot separate perception from option order and prompt text"
  - Verdict: needs rewording
  - Quote: "In contrast, GPT-4o’s accuracy for inverted faces dropped below chance level." (Results, Multiracial RMET, on the MRMET)
  - Note: Prompt and option order are unchanged while inversion takes accuracy from above human to below chance, so #1 does show that the score depends on image content. It is not a mismatch control, but "cannot separate" overstates the case for #1. Keep "most", and add that #1's inversion arm shows image dependence without showing which information is used.
- Claim: (methods.md, Literature check; introduction.md "eight recent papers on VLM emotion judgement"; references.md "Used for" and Why note) "RMET performance; fixed option order"; "'Fixed option order' and no mismatch control"
  - Verdict: supported
  - Quote: "Participants are instructed to choose the one word that best describes what the person in the photograph is thinking or feeling, and to answer as quickly and accurately as possible." (Methods, Tests)
  - Note: Fair as an emotion and mental-state paper. "Used for" should add MRMET and inversion. Both Why items are confirmed for the arXiv version only, and the published version remains open.
- Other findings: (1) The inversion arm is a stimulus edit with humans measured on the same edit, so it is close to the draft's step 3 and is a same-prompt, altered-image precedent. It belongs beside the Introduction's "Coarse image-level counterfactuals on VLMs exist [13]", and the "nearest" statement should be read against it. It concerns orientation, not named facial cues, so the core novelty claim stands. (2) The MRMET has the draft's second fault in part, supporting the prompt-text argument: "Some presented category labels repeated within the test but were always paired with the same three alternatives." (Methods, Error concentration). (3) The authors raise memorisation themselves: "This raises the concern that the faces (which were cut from magazines), and the response options may have been included in the training dataset for GPT-4o, potentially inflating its performance for upright faces." The MRMET was used because it postdates GPT-4o's training cutoff. That is a control for prior exposure, not for the prompt, and the draft does not mention it. (4) Error-pattern comparison with humans goes beyond accuracy but stays behavioural, so "first degree" holds.

### #2 Refoua
- Where read: https://pmc.ncbi.nlm.nih.gov/articles/PMC13022409/ (Europe PMC XML)
- Coverage: read in full. The file has references removed ("[[REFS OMITTED]]") and Fig. 1 is present as caption only. Tables are flattened but readable.
- Metadata: fix: first author, journal, year and DOI match. Add the title: "Evaluation of cross-ethnic emotion recognition capabilities in multimodal large language models using the reading the mind in the eyes test". The header tokens also show 16 and 9975, probably volume and article number; confirm from the DOI. Status can move to "read in full (PMC copy, references omitted)".
- Claim: (introduction.md para starting "Vision-language models (VLMs) can take") "such models match or beat people on tests of emotion and mental-state recognition from faces, including the ... RMET [1-3]"
  - Verdict: needs rewording
  - Quote: "In contrast, ChatGPT-4 performed near the human average, while Claude 3 Opus performed near chance level." (Abstract)
  - Note: True of one model in three. ChatGPT-4o reached the 85th, 94th and 90th percentiles. ChatGPT-4 reached 9.6, 43 and 17, and Claude 3 Opus .01, 4 and .04. Not RMET only: White RMET, B-RMET and K-RMET. The human comparison is descriptive: published norms, two administrations, and no test against humans (the only test is binomial against chance), despite the paper's phrase "significantly above the human average". Correction: "one of three models scored above average human norms".
- Claim: (introduction.md para starting "We then asked") "Two RMET studies fixed it [1,2]"
  - Verdict: needs rewording
  - Quote: "The tests were presented to the MLLMs in the same order as they appear in the original test materials and with the same prompts, and the correctness of the MLLMs’ responses (correct/incorrect) was recorded for later analysis." (Methods, Procedure)
  - Note: The text never mentions option order as such. "Fixed" is an inference from this sentence, which reads most naturally as item order, and no randomisation or rotation is reported. The paper says full prompts and materials are on OSF, which I did not read. Correction: "one fixed it [1], and one kept the original test order and reported no option randomisation [2]", or confirm from OSF first.
- Claim: (introduction.md para starting "We then asked") "we found none that controlled option position and also swapped the face under a fixed prompt"
  - Verdict: supported
  - Note: Absence claim, no quote. I looked for image-free, blank, mismatched, swapped, shuffled-image, inverted or any altered-image arm under any name and found none. The only comparisons are repeat administration, the three RMET versions, the three models, chance and published norms. Unlike #1, there is no same-prompt image manipulation.
- Claim: (introduction.md para starting "We then asked") "So most of the VLM emotion claims we located cannot separate perception from option order and prompt text"
  - Verdict: supported
  - Quote: "Higgins et al. 38 highlight evidence suggesting that performance on the RMET may rely less on direct emotion perception and more on other cognitive skills, such as vocabulary, abductive reasoning, and a process of elimination among the forced-choice options." (Limitations)
  - Note: The design has no arm that could separate them, and the authors state the concern about elimination among options themselves. They also say they cannot rule out stimulus overlap with training data.
- Claim: (methods.md, Literature check; introduction.md "eight recent papers on VLM emotion judgement"; references.md "Used for" and Why note) "RMET performance; fixed option order"; "'Fixed option order' and no mismatch control"
  - Verdict: needs rewording
  - Quote: "Each version of the RMET was presented in a separate and independent session, ensuring that the MLLMs had no memory of prior tests or responses." (Methods, Procedure)
  - Note: Paper type and "no mismatch control" hold. "Fixed option order" should carry the same caveat as above. "RMET performance" should say three RMET versions.
- Other findings: Access was manual through public chat interfaces (March to June 2024), so sampling settings are not reported. The authors use cross-version consistency as an argument against memorisation, but this is argument, not a control. They explicitly propose moving to video tasks, which the draft's design does; that does not threaten novelty.

### #3 Nelson
- Where read: https://pmc.ncbi.nlm.nih.gov/articles/PMC12533101/ (Europe PMC XML)
- Coverage: read in full (PMC XML text, reference list omitted from the file). Figures and Supplementary Tables 1 and 2, including the prompts, are not in the file.
- Metadata: fix: add the title, "Evaluating the performance of general purpose large language models in identifying human facial emotions", and volume and article number as shown in the file (8, 615). Authors shown: Nelson, Winbush, Siddals, Flathers, Allen, Torous. DOI 10.1038/s41746-025-01985-5 is confirmed by the text, so the "(U)" tag can go.
- Claim: (introduction.md para starting "Vision-language models (VLMs) can take") "such models match or beat people on tests of emotion ... sets of posed expressions [1-3]"
  - Verdict: needs rewording
  - Quote: "In contrast, Claude’s 95% CI did not overlap with that of humans, suggesting lower overall reliability." (Discussion)
  - Note: Two of the three models matched people overall, one fell short, and "beat" rests on three model-by-emotion comparisons only (GPT-4o on Surprise and Calm/Neutral, Gemini on Surprise). Correction: "some such models". The human side is published NimStim kappa compared by CI overlap, not a same-trial test, and the authors say the original kappa calculation was unspecified.
- Claim: (same sentence) GPT and Gemini match or exceed people
  - Verdict: supported
  - Quote: "GPT and Gemini matched or exceeded human performance, especially for calm/neutral and surprise." (Abstract)
  - Note: Direct support for those two models, which is the basis of the wording above.
- Claim: (same sentence) "sets of posed expressions"; (references.md row 3, Used for) "Performance on posed faces"
  - Verdict: supported
  - Quote: "The NimStim Set of Facial Expressions is a comprehensive collection of 672 images depicting facial expressions posed by 43 professional actors (18 female, 25 male) aged between 21 and 30 years." (Methods)
  - Note: Posed still images. #3 gives only this one set, so the plural "sets" depends on #1 and #2 supplying another or on rewording to "a set". #3 tests emotion only, not mental states.
- Claim: (same sentence) these are VLMs, as "such models" implies
  - Verdict: supported
  - Quote: "In contrast, zero-shot vision-language models without training, fine-tuning, or architectural customization may offer stronger generalization." (Discussion)
  - Note: Yes. The paper's main term is LLM, but the three models took uploaded images directly through their chat interfaces, and the paper itself uses "vision-language models". This is a VLM finding, not a text-model one.
- Other findings: The paper itself reports that the wider record is mixed, which the Introduction's opening omits: "Studies evaluating LLMs' visual emotion recognition have mixed results, with some models performing no better than chance 6 ." The work is purely behavioural, which fits the first-degree claim. It sits outside the eight-paper audit: prompts varied by model, testing was through the user interface, and no option-order or mismatch control is described in the main text (the prompts are in a supplement not in this file). "#3 ... no argument rests on them" is fair, since it is only opening context.

### #4 Schyns, Snoek and Daube
- Where read: https://eprints.gla.ac.uk/278852/ (University of Glasgow, version of record)
- Coverage: read in full. Version: version of record, not an accepted manuscript. The text carries journal typesetting, an "Opinion" header, page numbers 1090 to 1102 and the Elsevier CC BY line. Figure images are absent from the text; captions are intact. Some symbols and spacing are damaged (superscripts, ligatures).
- Metadata: ok. Authors, title, journal, year and DOI match. Optional additions the text shows: Vol. 26, No. 12, December 2022, pp. 1090-1102; article type Opinion. Change the Status column from "snippet" to read in full (and drop #4 from the "Must read" list; its "on a snippet" note is out of date).
- Claim: (introduction.md para starting "Vision-language models (VLMs) can take") "Schyns and colleagues split that question into three degrees of similarity: the same behaviour, the same stimulus features, and the same computations [4]."
  - Verdict: supported
  - Quote: "We have framed the question with three embedded degrees that progressively constrain algorithmic similarity evaluations: equivalence of (i) behavioral/brain responses, which is current practice, (ii) the stimulus features that are processed to produce these outcomes, which is more constraining, and (iii) the algorithms that process these shared features, the ultimate goal." (p. 1090, abstract)
  - Note: Extra check answer: it is a three-way split, but nested, not parallel. Each degree includes the one before ("embedded"). The paper's label for the third degree is algorithms, glossed as computations, so "computations" is fair. Degree 1 also covers brain or layer responses, and the paper is about DNN categorisation, not VLMs or face reading specifically. Recommended small tweak: "three embedded degrees of similarity: the same behaviour, then the same stimulus features, then the same computations on those features".
- Claim: (same para) "The VLM emotion work we located sits almost entirely at the first degree."
  - Verdict: needs rewording
  - Quote: "Current practices primarily target a ﬁrst degree of equivalent behavioral (or layer) outcomes" (p. 1091)
  - Note: Indirect dependency. The paper's first degree means similar responses to the same stimuli, and its worked case is a model that matches responses trial by trial. A report that a model matches or beats people on an accuracy score does not by itself reach that. Change to "at or below the first degree", or "compares scores, which is at most the first degree". Whether #1 to #3 compare item-level responses is for the batch that holds them.
- Other findings:
  - The draft's identifiability point has a direct conceptual precursor in this paper, the multiple realizability argument, yet the draft cites only [16,17] for related limits. Cite [4] in the identifiability paragraph too. The Theorem 1 formal result is not in this paper, which argues in words only.
    - Quote: "No, because we still face the classic argument of the multiple realizability [38] of responses." (p. 1092)
  - The claim that natural correlations must be broken is also made here: natural images lack causal structure, and generative control is the proposed remedy.
    - Quote: "Naturalistic images are observational datasets without a known causal structure [ 45]." (p. 1094; the space in "[ 45]" is in the file)
  - Novelty: Daube et al. 2021 (this paper's ref 18) is described as comparing DNN and human causal features on generated faces with reverse correlation. It is DNNs and face identity, not VLMs or emotion, so the Introduction's novelty sentence survives as worded. But CFA is an instance of the paper's second-degree benchmark, so "We call it Causal Feature Alignment" should acknowledge that lineage, and Daube et al. should be read and cited as a non-VLM precedent.
    - Quote: "We expand to the second degree by quantifying the similarity of the 3D F underlying human and DNN responses" (p. 1095, Box 2)
  - Tension: the framework warns against averaging over individuals, and A4's human target averages subject estimates ("Subject estimates were averaged"). A reviewer who knows the paper may raise this. Worth a line in Sensitivity and limits.
    - Quote: "Injunction 2. Never average over methods [information-processing algorithms]." (p. 1091, quoting Newell)
  - The paper's first two degrees assume the same stimuli for both systems. A4 uses new renders, which Methods discloses, so A4 is an approximation of the second-degree benchmark and not the benchmark as framed.
  - The Concluding remarks hedge on the need for the second degree, although Box 3 argues it is necessary. The draft relies on it only as framing, so no change is needed.
    - Quote: "However, it is possible that the second degre e may not be essential" (p. 1100; the space in "degre e" is in the file)

### #5 Zheng et al.
- Where read: https://arxiv.org/abs/2309.03882
- Coverage: read in full (references skimmed). Figures are extracted as axis tick fragments and carry no usable values; Tables 1 to 7 are readable. The file is arXiv v4, 22 Feb 2024.
- Metadata: fix: add the title, "Large Language Models Are Not Robust Multiple Choice Selectors"; authors are Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, Minlie Huang. The text shows "Published as a conference paper at ICLR 2024" but does not show "Spotlight", so that word rests on the repo note only. Update Status from partial to read in full.
- Claim: (introduction.md para starting "The first problem is one of validity") "selection bias has been traced mainly to the option identifiers [5]"
  - Verdict: supported
  - Quote: "we pinpoint that this behavioral bias primarily stems from LLMs’ token bias, where the model a priori assigns more probabilistic mass to specific option ID tokens (e.g., A/B/C/D) when predicting answers from the option IDs." (p. 1, abstract)
  - Note: "In text models" is correct: 20 text LLMs, three benchmarks. The paper elsewhere says "one primary intrinsic cause", and its Conclusion adds "and partially from position bias". Evidence is an ablation: removing IDs lowers recall imbalance in most cases. It is not uniform, and the ablation also changes the scoring (option likelihoods or generated text in place of ID tokens). The draft reader may not know the term, so a gloss helps: "selection bias, a preference for particular option IDs".
  - Quote: "The selection bias of llama-2-13/70B even slightly increases in MMLU and ARC after option IDs being removed while still decreasing in CSQA, implying the potential counteraction between token bias and position bias." (p. 5, s2.4)
- Other findings:
  - The sentence in the draft sets [6] and [5] side by side, but [5] explicitly disputes the position-bias account that [6] gives. This does not make the draft wrong, since [6] is cited only for effect size, but do not cite [6] for a cause.
    - Quote: "selection bias arises less from LLMs’ position bias" (p. 2)
  - Token bias is defined for scoring on option ID tokens. The draft's scorers read label text (A1 output, A3 joint probabilities), not ID tokens, even though options are numbered. So [5]'s main mechanism may not carry over to A1 and A3 unchanged. The draft does not claim that it does. Position bias, which [5] calls irregular, may be the more relevant channel.
    - Quote: "These results suggest that the model’s position bias is somewhat present but quite irregular, largely depending on models and tasks." (p. 5, s2.4)
  - A first-slot target does not always inflate accuracy in text models. Moving the gold answer to A gave llama-30B +15.2, but falcon-inst-40B -13.3 and falcon-40B -5.5 (Table 1). The draft's sentence that accuracy "could not be told apart from a preference for the first option" still holds, but the size and sign are model-specific.
    - Quote: "falcon-inst-40B 51.5 38.3 (-13.3) 38.9 (-12.7) 55.7 (+4.1) 69.1 (+17.6)" (p. 2, Table 1)
  - Design support the draft could cite: A1's rotation is close to [5]'s Cyclic Permutation baseline, which is uncited in Methods. A1 rotates five of six slots and keeps None of the above last, so it is a partial version.
    - Quote: "which intuitively cancels out both the model’s token bias and position bias." (p. 5, s3.1)
  - [5] dropped items with "none of the above" options, because such options do not move validly. The draft fixes that option last, which is a reasonable answer, but it means [5]'s results do not cover an abstain slot.
    - Quote: "where the options refer to each other like “A and B”, “none of the above”, in all the experiments." (p. 28, Table 7 note)
  - A3 scoring: when [5] scores option text it normalises likelihoods by length, and it reports that this format usually lowers accuracy. A3 as drafted is "joint log probability", renormalised over options, with no length normalisation stated. Option labels differ in token count ("None of the above" against one-word labels). State whether this was considered. A prior or length effect would enter the accuracy secondary outcomes, and the real-minus-mismatch contrast is less exposed.
    - Quote: "For open-source models, we compute the likelihoods of options, normalized by their lengths, and use the maximum one as the model prediction." (p. 4, s2.4)
  - [5]'s recall-by-position metric (RStd) is the same idea as the draft's "accuracy by target position"; citing it would give that secondary outcome a source.

### #6 Pezeshkpour and Hruschka
- Where read: https://arxiv.org/abs/2308.11483
- Coverage: read in full (references skimmed). Figures 2 and 3 are not available as text; Tables 1 to 8 are readable. The file is arXiv v1, 22 Aug 2023.
- Metadata: fix: add year 2023 and the title, "Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions". Authors are Pouya Pezeshkpour and Estevam Hruschka. Only v1 was checked, so whether a later version or venue exists cannot be told from this text. Update Status from abstract to read in full.
- Claim: (introduction.md para starting "The first problem is one of validity") "In text models, option order can shift accuracy by a wide margin [6]"
  - Verdict: needs rewording
  - Quote: "the sensitivity gap, which is the difference between the maximum and minimum LLMs’ performance when using an oracle ordering." (p. 3, s3)
  - Note: Extra check answer. Models: GPT-4 and InstructGPT (text-davinci-003) only, both text. Range: about 13 to 75 points, from the abstract's "13% to 75%". Table 1 gives zero-shot gaps of 13.1 to 53.0 for GPT-4 and 43.1 to 74.1 for InstructGPT, with both extremes on Logical Deduction, across five tasks. Against default order the moves run from -39.4 to +39.0 points. Few-shot gaps in Tables 6 and 7 run from 9.4 to 81.0, so the abstract's range is really the zero-shot one. The qualifier that matters: these are oracle gaps, best against worst ordering chosen with the answer known, not what an arbitrary reorder does. The number of orderings searched is not stated. They are percentage points, though the paper writes "%". Correction: "In text models, reordering options with the answers known opened gaps of about 13 to 75 percentage points between the best and worst accuracy [6]."
  - Quote: "a considerable performance gap of approximately 13% to 75% in LLMs" (p. 1, abstract)
- Other findings:
  - Non-oracle shifts are far smaller in this paper: majority vote over 10 random reorders changed accuracy by 0 to +8 points (Table 5). [5]'s Table 1 (-13.3 to +17.6 from moving the gold answer) is a useful realistic companion, and citing [5,6] together would give both scales.
    - Quote: "improvement of up to 8 percentage points." (p. 7, s5)
  - Only two closed models, no probabilities, and [5] criticises the study for not separating option IDs from position (p. 9 of [5]). Cause statements from [6] are conjecture by its own wording ("Conjecture 4.1"), and the draft does not use them.
  - Sensitivity is tied to uncertainty among the top choices, which suggests that option effects may be largest on EU-Emotion's ambiguous items. This is untested for faces or VLMs.
    - Quote: "this sensitivity arises when LLMs are uncertain about the prediction" (p. 1, abstract)
  - Design note: effects depend on where the top two choices sit relative to each other, with first and last placement amplifying bias. A1's cyclic rotation balances each target over named positions, but it keeps neighbouring options together in nearly all rotations, and the last slot is always None of the above. So A1 balances marginal position, not the relative placement of confusable options. Rotation is not claimed to do more, so this is a limit to note, not an error.
    - Quote: "for amplifying bias, it is preferable for the top-2 choices to be positioned in the first and last options." (p. 6, s4.2)

### #7 Li, Qiu and Xu
- Where read: https://arxiv.org/abs/2609.29278
- Coverage: read in full (references skimmed). Equations and table layouts are damaged but values are readable; figures are not available as images. The file is arXiv v1, 24 Sep 2026.
- Metadata: fix: add the title, "Reasoning Instructions Can Break Answer Decoding in Vision–Language Models"; authors are Zeyan Li, Siyuan Qiu, Jianfeng Xu; arXiv 2609.29278 and 2026 match. Update Status from abstract to read in full.
- Claim: (introduction.md para starting "The first problem is one of validity") "In one VLM preprint, most predictions fell on the first slot under one scoring method [7]."
  - Verdict: needs rewording
  - Quote: "On ScienceQA, Qwen2.5-VL-7B drops from 80.76% to 45.48%, and across five option-content permutations 93.54% of CoT-prefix predictions select the first slot." (p. 1, abstract)
  - Note: Extra check answer. Model: Qwen2.5-VL-7B. Task: ScienceQA, 4,241 items, including text-only items, with no faces. Scoring: "CoT-prefix", where a reasoning cue is appended and the answer-label logits are read at the next position. Rate: 93.54% over five content permutations (94.08% unpermuted), against 51.71% for Direct; gold is first on 40.0% of items. It is a position finding only in part, because slot and label are coupled (first slot is always A). "One scoring method" misleads: Direct and CoT-prefix use the same label readout and differ in the prompt suffix, so the cause is the cue-plus-immediate-readout pairing. Correction: "In one VLM preprint, one model picked the first slot, always labelled A, on 94% of items when a reasoning cue was appended and the answer label was read at once, against 52% without the cue [7]." The paper also says the answer survives this failure and is recovered by a probe or by generation, so it is an interface fault and not a general VLM preference for the first slot.
  - Quote: "content shuffling cannot fully separate position bias from label-token bias without independently counterbalanced labels" (p. 4, s5)
  - Quote: "A/first-slot rates of51.71±1.06% and93.54±0.43%" (p. 2, s3; missing spaces are in the file)
  - Quote: "differ only in the terminal text, so their contrast isolates the suffix." (p. 2, s2)
  - Quote: "all official ScienceQA test questions, including image-present and text-only items." (p. 2, s2)
- Other findings:
  - The result is not general. On Table 1 first-slot rates for other model and dataset pairs are 94.66, 66.71, 65.10 and 46.80, and on MMBench the cue raised accuracy by 5.80 points. Free generation dropped the rate to 42.09%. Do not lean on the sentence as evidence about VLMs overall.
    - Quote: "The collapse is severe but not universal." (p. 3, s3)
  - Prefill check, Methods A3 ("EMOTION:"). The paper does not test a prefill or any emotion task, and nothing in it addresses joint option-string probability after a prefill directly. Its rule favours the draft's design: the A1 prompt asks for the answer first and the reasoning after, so the "EMOTION:" field is the requested event, and it reads as a "dedicated answer field". A3 is not CoT-prefix scoring. The nearest evidence is that answer-eliciting endings work under immediate scoring. This is an analogy, not a test.
    - Quote: "or a dedicated answer field), rather than the immediate label logits at the first post-prefix position" (p. 4, s5)
    - Quote: "answer-eliciting endings such as “Answer:” perform well under immediate" (p. 3, s3)
    - Quote: "the prompt should avoid continuation-seeking instructions" (p. 4, s5)
  - Prefill cautions. (1) A3 scores full option strings, and the paper found this interface far from neutral on Direct: 52.32% against 80.76% for label scoring. That is in line with A3's stated "could reflect scoring as well as input", and it is a reason to read A1 against A3 accuracy with care. It concerns length-normalised strings, and A3 is not length-normalised (see #5). (2) Renormalising over options can hide a shift of mass away from the options, and the paper reports that renormalisation did not rescue the decision. A3 has no check of raw option mass before renormalising, unlike B2 for the blind readout. Consider reporting it. (3) The paper asks for the tokenisation of labels to be reported, which Methods does not yet give. These also apply to A4, which reuses A3's scoring.
    - Quote: "scoring length-normalized likelihoods of the full option strings reverses the ordering, 52.32% for Direct and 55.41% for CoT-prefix." (p. 3, s3)
    - Quote: "why simply renormalizing the valid choices does not" (p. 4, s4)
    - Quote: "Reporting should also specify the exact interface (labels vs. free-form generation), the label tokenization, and whether decoding is conditioned on an intermediate rationale" (p. 4, s5)
  - The paper's reference list names published VLM option-bias work (its refs 8 and 9: Atabuzzaman et al., EMNLP 2025 per the list, on selection bias; Tian et al., CVPR 2025 per the list, on position bias in multi-image input) and a flaws paper (its ref 12: Rosenthal et al., arXiv 2511.22341). These may be stronger sources than one preprint for the VLM half of the sentence. Unread here, so check them before citing. Ref 9 may bear on image-list input.
  - Small internal inconsistency in the paper: CoT-prefix native accuracy is 45.48 in the text and Table 1 but 45.56 in Table 2 (delta -35.28 against -35.20). If a number is quoted, use the abstract's.
  - No threat to the novelty claim; the paper has no face, emotion or human comparison content.

### #8 Bhattacharyya and Wang
- Where read: https://arxiv.org/abs/2502.05660
- Coverage: read in full, including appendices (references skimmed). PDF extraction: Figure 2, the only display of the option-order result, survives as axis labels with no bar values, so order effects cannot be checked numerically from this text. Prompt figures have lost spaces in places, and colour coding in tables is lost.
- Metadata: fix: authors match (Sree Bhattacharyya, James Z. Wang). The text shows arXiv:2502.05660v1, 8 Feb 2025, and does not show "NAACL Findings 2025", so confirm the venue elsewhere. Add the title: "Evaluating Vision-Language Models for Emotion Recognition". Status can move to "read in full".
- Claim: (introduction.md para starting "The first problem is one of validity") "Option order also moves VLM judgements of the emotion an image evokes [8]"
  - Verdict: supported
  - Quote: "Thus, overall, most open-source models deteriorate when negative emotions are presented first, while their positive bias is increased when emotions are grouped according to sentiment categories." (Section 6.1)
  - Note: What changed: three fixed list orders were compared (alphabetical, positive labels first, negative labels first). Negative-first lowered weighted F1 for all but GPT4-o, positive-first slightly raised it for LLaVA and LLaVA-Next, and sentiment bias shifted in all but GPT4-o. Optional tightening: "in most models tested". The effect is model-dependent, and results come from single runs with no significance test.
- Claim: (introduction.md para starting "We then asked") "One tested option order directly, on images rather than faces [8]"
  - Verdict: needs rewording
  - Quote: "Besides, though the datasets used do not contain any private identifiable information, a large number of images include humans, revealing their faces and gestures." (Section 10)
  - Note: Stimuli are not face stimuli, but they are not face-free. They are mixed affective images from EmoSet, FI, Abstract, ArtPhoto and Emotion6: people, nature, objects, art. The task concerns the viewer's evoked emotion, not an emotion expressed by a face. Also, "tested directly" means grouped list orders, despite the paper's label "shuffled"; there was no random shuffle, rotation, or manipulation of target position. Correction: "One compared fixed list orders, grouped by valence, on mixed affective images, judging evoked rather than expressed emotion [8]."
- Claim: (introduction.md para starting "We then asked") "we found none that controlled option position and also swapped the face under a fixed prompt"
  - Verdict: supported
  - Quote: "The experiments in the previous section present the emotion class labels in alphabetical order in the prompts." (Section 6.1)
  - Note: Absence claim, confirmed by full read, replacing the earlier keyword search. No image-free, blank, mismatch or shuffled-image arm exists. "Shuffled" refers to labels only, and the caption mechanism still supplies the image. All variations change the prompt. Baseline order was fixed, and position was tested, not controlled.
- Claim: (introduction.md para starting "We then asked") "So most of the VLM emotion claims we located cannot separate perception from option order and prompt text"
  - Verdict: needs rewording
  - Quote: "Your task is to answer what emotion is evoked in you when viewing the given image, choosing from the list: [emotion categories]." (Figure 16, persona prompt; the same wording appears in other prompt figures with spaces lost)
  - Note: #8 lacks the prompt-text fault. One label list is used for every image in a dataset, so the prompt does not identify the category. Option order is a real concern in its baseline, but prompt text is not. The sentence lumps two faults together, and #8 should not be counted as having both.
- Claim: (methods.md, Literature check; introduction.md "eight recent papers on VLM emotion judgement"; references.md "Used for" and Why note) "Option order on evoked emotion"; "Option order tested; 'none found' for mismatch came from a keyword search"
  - Verdict: supported
  - Quote: "We investigate the specific task of evoked emotion recognition" (Section 1)
  - Note: VLM paper with seven model variants across four families. "None found" now rests on a full read, so the Why note can be updated.
- Other findings: #8 also ran a no-label, open-vocabulary arm. All models did worse, and the sentiment bias reversed for all but GPT4-o. This is direct VLM evidence for the Introduction's "order and form of answer options", and it means #8 tested option form as well as order. The human study checks model errors against dataset labels, not human-model similarity. No threat to the novelty statement.

### #9 Zhao
- Where read: https://arxiv.org/abs/2608.17102
- Coverage: read in full (references skimmed). PDF extraction: heatmaps absent, equations and Tables I and II partly scrambled but readable.
- Metadata: fix: add the title and authors shown in the text: "Emotion Across Speech and Faces: Shared Affective Mechanisms in Multimodal Foundation Models", Xiutian Zhao, Luqi Sun, Björn Schuller, Berrak Sisman. arXiv:2608.17102v1, 17 Aug 2026, matches. Status can move from partial* to "read in full".
- Claim: (introduction.md para starting "We then asked") "One randomised it as a routine control [9]"
  - Verdict: supported
  - Quote: "To reduce position and label bias [59], the order of emotion options is randomized across examples, and the model is instructed to output only the option index (alphabetic letters) rather than the emotion word." (Section IV-A)
  - Note: Literally true. Order is randomised per example under fixed seeds. It is not a full rotation, and no accuracy by position is reported.
- Claim: (introduction.md para starting "We then asked") "eight recent papers on VLM emotion judgement"
  - Verdict: needs rewording
  - Quote: "Using speech emotion recognition and facial expression recognition as complementary probes, we identify acoustic and visual ESNs." (Abstract)
  - Note: Only partly a VLM emotion-judgement paper. It uses a facial-expression task on AffectNet in multimodal models, but that task is a probe in a neuron-level interpretability study spanning speech and faces, and its claims concern decoder units, not judgement accuracy. Correction: note that #9 is a mechanistic study that used a facial-expression task, or say "papers that test emotion judgement in VLMs".
- Claim: (introduction.md para starting "We then asked") "we found none that controlled option position and also swapped the face under a fixed prompt"
  - Verdict: supported
  - Quote: "As a control, we also construct random masks by selecting the same number of neurons uniformly from the decoder MLPs, without using activation statistics." (Section III-B)
  - Note: Absence claim. Option position was controlled, but no face was swapped. No image-free, blank, mismatch or shuffled-image arm exists. "Random" here means neuron masks, and "cross-modal" and "mismatched" refer to modality transfer and emotion pairs among neuron sets, not stimuli.
- Claim: (introduction.md para starting "We then asked") "So most of the VLM emotion claims we located cannot separate perception from option order and prompt text"
  - Verdict: needs rewording
  - Quote: "We focus on five emotion categories shared by the two tasks: anger, fear, happiness, neutral, and sadness." (Section IV-A)
  - Note: #9 appears to avoid both faults: options are randomised and the same five categories apply across items, so option position and prompt text should not predict the target. This inference assumes all five options appear on every item; the exact prompt is not printed. With one prompt across items, a mismatch arm is less needed to exclude prompt leakage. Do not count #9 among those that "cannot separate".
- Claim: (methods.md, Literature check; references.md "Used for" and Why note) "Randomised options as routine control"; "Summarising tool only. Recheck every statement and quote"
  - Verdict: supported
  - Quote: "All models are evaluated under the same multiple-choice question answering protocol." (Section IV-A)
  - Note: Rechecked against the full text. The "Used for" wording is right.
- Other findings: #9 is causal work on internal units, so it is a partial exception to "sits almost entirely at the first degree", though it makes no comparison with humans, and "almost" covers it. It uses no AUs, named facial cues or human cue effects, so the novelty statement stands. It could be named with the neighbours [13,20] as emotion-relevant activation interventions. Its models differ from the draft's: Gemma-4-12B-it is not Gemma 4 E4B, and Qwen2.5-Omni-7B is not Qwen3-VL-8B. Its authors caution that late-layer units may partly encode answer selection.

### #10 Xing et al.
- Where read: https://arxiv.org/abs/2609.34742
- Coverage: read in full to the end (60 pages, Appendices A to J, references skimmed). Tables and figures are partly garbled, Figure 9 and Figure 26 hold many CJK tokens, and many spaces are dropped. Numbers quoted here come from prose or legible tables.
- Metadata: fix. "CMES" is not a venue or the paper's name: it is the stimulus collection the paper builds, "CMES (Cross-Modal Emotion Stimuli)". As written in the citation column beside a venue column it can be read as a venue. Use "Xing et al." with the title "Do Emotion Concepts Generalize Across Sources, Modalities, and Architectures in Vision-Language Models?". The first author is Bohao Xing. arXiv:2609.34742v1, 28 Sep 2026, "Preprint. Under review." Year and arXiv ID are ok. Status should become "read in full (arXiv v1)": the old note "read to char 200,000" already exceeded the file (189,388 bytes), so that read reached the end anyway.
- Claim: (introduction.md para starting "We then asked how far") "We examined eight recent papers on VLM emotion judgement"
  - Verdict: supported
  - Quote: "We therefore intervene on activations while each VLM reads held-out face images." (Sec. 4)
  - Note: #10 fits, but it is an interpretability paper. It makes no accuracy claim against humans, and its emotion judgement is the readout for steering.
- Claim: (introduction.md same para) "The eighth avoids option lists [10]"
  - Verdict: needs rewording
  - Quote: "The model answersHow does this person feel?after the primeThey feel." (Appendix B.5)
  - Note: True for the main emotion readout, which scores seven label words from the next-token distribution with no list in the stated prompt. The prompts for the screening run, the "direct questioning" accuracy (47.6%), and the gender and safe-dangerous readouts are not given, so "avoids" is proved only for that readout. Say "avoids option lists in its main readout".
- Claim: (introduction.md same para) "and we did not check its baselines"
  - Verdict: needs rewording
  - Quote: "We also inject matched-norm random and emotion-orthogonal directions to test whether an arbitrary perturbation produces the same response." (Appendix E.1)
  - Note: Now stale. Baselines reported: the unsteered model on the same images; matched-norm random and emotion-orthogonal directions; for transfer, no mapping, a random orthogonal map and target-orthogonal directions; for removal, a matched-rank random direction or subspace; for the shared span, a random direction outside it and shuffled remainder labels; for correspondence, a 500-shuffle label permutation null and random or orthogonal replacement vectors; the gender readout as off-target check; chance references (12.5%, 1/6). All vary vectors, maps, labels or readouts. Searched for and not found: any image-free, blank, mismatched or swapped-image baseline. Replacement: "The eighth avoids option lists in its main readout, and its controls act on vectors and labels, with no image-free or mismatched-image baseline [10]."
- Claim: (introduction.md same para) "In the seven we could check, we found none that controlled option position and also swapped the face under a fixed prompt."
  - Verdict: needs rewording
  - Quote: "To test who the induced emotion is attributed to, we keep the image and intervention fixed and ask about the depicted person, the assistant, or a third party absent from the image." (Appendix E.6)
  - Note: The quote is the nearest design to a swap, and it holds the image fixed while varying the prompt, the reverse of the draft's test. The count becomes eight and the conclusion stands: #10 uses one fixed question, but it never swaps the face, so it adds no exception. Change to "Across the eight, we found none...". "Controlled option position" fits awkwardly now that #10 and #12 have no position to control, so a tighter form is "none that swapped the face under a fixed prompt with option position controlled or absent". #12's place in the eight rests on its main paper only. "The nearest" stays [13], since neither paper here is closer.
- Claim: (introduction.md same para) "So most of the VLM emotion claims we located cannot separate perception from option order and prompt text."
  - Verdict: needs rewording
  - Quote: "The model answersHow does this person feel?after the primeThey feel." (Appendix B.5)
  - Note: #10 has neither fault in its main readout, since it has no list and one question for all images. See the same claim under #12: 3 of 8 is not "most".
- Claim: (methods.md, Literature check) "image-free or mismatch controls in eight emotion papers [1,2,8-13]"
  - Verdict: supported
  - Quote: "This impairs recognition substantially more than removing a matched-rank random subspace (Table 21 in Appendix E.5)." (Sec. 4.2)
  - Note: #10 can now be tabulated as none on both counts. Its evaluation images were also pre-selected as correctly classified, which is worth a table note.
- Claim: (references.md row 10, Used for and Status; Must read row 10) "Avoids option lists"; "partial (read to char 200,000; baselines not checked)"; "Baselines never checked. The eighth paper in the count"
  - Verdict: needs rewording
  - Quote: "The model answersHow does this person feel?after the primeThey feel." (Appendix B.5)
  - Note: Used for: "No option list in main readout; vector-level controls only, no image-free or mismatch baseline". Status: "read in full (arXiv v1)". Remove #10 from Must read, and adjust the Priority 1 count (12) and the total (44) if the list keeps counting it. The stated reason no longer applies.
- Other findings:
  - Blind readout precedent: "How does this person feel?" with the prime "They feel", scored over label words from the next token, closely resembles the draft's blind readout (same prefill, same idea, label families instead of label words). The draft does not cite this. It claims no novelty there, but citing [10] in Methods would be fair.
  - First degree: "The VLM emotion work we located sits almost entirely at the first degree" still stands because of "almost", but #10 is not first-degree work: it intervenes on internal activations and removes a subspace. It compares with humans only through valence and arousal ratings of six emotions. Treat it as the counterexample that "almost" covers.
  - Cue-level novelty survives. #10 injects whole emotion vectors, not named cues, uses no AUs and no randomised human data. Its patch maps are qualitative by its own statement (Appendix C.2): "These maps provide a qualitative view of token similarity, not a test of which image regions cause the model’s prediction." It is a closer neighbour than the Introduction's list implies, because it is a latent causal intervention on VLM emotion, on Qwen3-VL-8B among others, so consider adding it beside [20].
  - For the #13 checker: #10 cites Zhang et al. (2026), Forty-third ICML, "Interpreting and enhancing emotional circuits in large vision-language models via cross-modal information flow". The text does not show whether this is reference #13.

### #11 Agarwal et al.
- Where read: https://arxiv.org/abs/2604.15280
- Coverage: read in full, including the supplement S1 to S6 (references skimmed). This is v2 of 21 Jul 2026. Figures are absent or garbled in extraction; Tables 1 to 3 and S1 to S6 are readable, though Table S4's diagonal digits run together.
- Metadata: fix: venue is out of date. Replace "arXiv preprint (2026)" with ECCV 2026 per the arXiv comment. The comment reads "European Conference on Computer Vision (ECCV) 2026", taken from the local abs.html, not paper.txt, which names no venue, and it gives no proceedings details. Title may be added: "Why Do Vision Language Models Struggle To Recognize Human Emotions?" Authors: Madhav Agarwal, Sotirios A. Tsaftaris, Laura Sevilla-Lara, Steven McDonagh. Status column can move to read in full.
- Claim: (introduction.md para starting "We then asked how far") "One did not report it [11]"; (references.md row 11) "Option order not reported"
  - Verdict: supported
  - Quote: "Emotion Labels choices: {List of Emotion Labels}" (Section 4.2)
  - Note: Truly unreported. This placeholder, in the MSCE prompt, is the only sign of a label list; the baseline prompt is not printed, and neither the order nor whether it varied is stated. All "order", "shuffle" and "random" passages concern video frames, so a keyword hit on "shuffle" would mislead.
- Claim: (introduction.md same para) "we found none that controlled option position and also swapped the face under a fixed prompt"
  - Verdict: supported
  - Quote: "we randomly shuffle the frame order." (Section 3.4; the file breaks the line after "we", which normalising whitespace closes)
  - Note: No mismatch, no face swap and no image-free control anywhere, main text or supplement. Manipulations are frame shuffle, frame rate, visual prompting, text enrichment and balanced fine-tuning, all on the item's own video.
- Claim: (introduction.md same para) "So most of the VLM emotion claims we located cannot separate perception from option order and prompt text."
  - Verdict: needs rewording
  - Quote: "sampling 495 videos (45 videos per class) for MAFW and 700 videos (100 videos per class) for DFEW" (Section 3.1)
  - Note: For #11 the sentence overreaches. If one label list serves a whole dataset (likely, not stated), the prompt cannot identify the category, and on balanced test splits neither a fixed slot preference nor the prompt alone can lift overall accuracy above chance; label position could still bias its per-class results, which carry its long-tail claim. Correction: "...cannot rule out option order or, where option sets differ by item, prompt text". Also, only #1, #2, #11 and #12 are option-order failures, so "most" holds only if lack of an image swap is counted too.
- Claim: (methods.md, Literature check) "We tabulated option handling and image-free or mismatch controls in eight emotion papers [1,2,8-13]"
  - Verdict: supported
  - Quote: none (absence).
  - Note: Searched for mismatch, image-free, text-only, blank and no-image conditions and read the whole file: #11 has none. The row should read option order not reported, no image-free control, no mismatch control.
- Other findings: (1) #11 is counter-evidence the draft's opening omits: on in-the-wild video it reports VLMs struggling and mostly below specialist vision classifiers, whereas the first sentence cites only match-or-beat reports [1-3]. A clause acknowledging this would be safer. (2) It already shuffles frames on VLMs including Qwen3-VL, with almost no change in macro-F1 (DFEW 0.5511 to 0.5538, MAFW 0.2738 to 0.2615). No human comparison is made, so the novelty claim stands, but this is direct prior work for our temporal step and P12 and P13, and should be cited there. (3) Table S1 tests a frame grid on Qwen2.5-VL: Video Montage lowers F1 from 0.2449 to 0.2012, and the paper blames the loss of per-frame resolution. That supports the Delivery audit and could be cited there. (4) Frame-rate results (peak near 5 FPS, decline above) bear on our input route. (5) Neighbours it cites, unread here: DK-CLIP, PE-CLIP, ExpLLM, AU-DFER, all using AU information as model input.

### #12 Hallmen et al.
- Where read: https://arxiv.org/abs/2610.08162
- Coverage: main paper read in full (arXiv v2, 10 pages, references skimmed). Supplement fetched afterwards from arXiv ancillary files: Sec. S5 to S7, S13 to S15 read, the rest keyword-scanned, so partial. The supplement overturns the absence statements below, as the block "Supplement" records. Extraction drops many spaces and garbles tables, but the prose is readable.
- Metadata: ok for authors, year and venue (arXiv:2610.08162v2, 31 Oct 2026; the footnote says submitted to the IEEE). Title if wanted: "Parsed Text Understates Vision–Language Models: From Below to Above Expert Agreement on Fine-Grained Facial Emotion". fix: status updated to main paper in full, supplement partial. Used for needs changing, see below.
- Claim: (introduction.md para starting "We then asked how far") "one did not control it [12]"
  - Verdict: needs rewording
  - Quote: "For each of the40categories we ask one question, “Does this face expressX?”." (Sec. VI)
  - Note: The headline result has no option list, so option position does not exist there. Text parsing (one JSON list of 40 categories, and a six-word FACES list) does not state option order anywhere in the main paper, and nothing rotates or randomises it, and the supplement shows one fixed list in the FACES text arm, so "not controlled" holds for that arm only. Correction: "one scored its main result from a yes/no question per category, so no option position arises, and kept one fixed option list in its text arm [12]."
- Claim: (introduction.md same para) "In the seven we could check, we found none that controlled option position and also swapped the face under a fixed prompt."
  - Verdict: supported
  - Quote: "A constant prediction scores0.000and the benchmark’s Random Baseline0.008(Tabs. I and S3)." (Sec. VII)
  - Note: Superseded in part: the main paper has no image-free run, but supplement S14 does (see the block below). No mismatched or swapped-image run was found. The nearest baselines are trivial predictors (constant, Random Baseline) and an expert passed through the mapping; Table IV's controls vary readout and scoring, never the image. For the count change see #10.
- Claim: (introduction.md same para) "So most of the VLM emotion claims we located cannot separate perception from option order and prompt text."
  - Verdict: needs rewording
  - Quote: "Instead of one written list of ratings, each model answers one yes/no question per category, and we read the probability it assigns to “yes”." (Sec. I)
  - Note: This does not hold for #12's headline claim. Every image gets the same 40 questions, so there is no option position and the prompt cannot identify the target, and an image-blind constant scores 0.000. The sentence fits only its text-parsing arms. With #12 and #10 out of that group, the draft's own tally leaves [1,2,11], 3 of 8, which is not "most" (other batches may change this). Suggested: "Several of the VLM emotion claims we located...".
- Claim: (methods.md, Literature check) "We tabulated option handling and image-free or mismatch controls in eight emotion papers [1,2,8-13]. Most were read as targeted passages."
  - Verdict: supported
  - Quote: "We test it on EmoNet-Face-HQ with eleven open-weight VLMs." (Sec. I)
  - Note: #12 belongs in the set, as a VLM emotion paper. Its main paper is now read in full, so recount "Most" once all batches are in.
- Claim: (references.md row 12, Used for) "Option order not controlled"; (Must read, row "9, 11, 12") "Summarising tool only. Recheck every statement and quote"
  - Verdict: needs rewording
  - Quote: "For text parsing on the single-label FACES task (Sec. IX), a reasoning model truncated while reciting the prompt’s option list still leaves a category word in its reply." (Sec. VI)
  - Note: This is the only mention of an option list in the paper's own design, and order is not stated. Used for should read "Yes/no per category, no option position; one fixed list in the text arm; no-image control in S14". The Must read row is now cleared for #12's main text only.
- Other findings:
  - Supplement: see the block below. It holds the prompts and a no-image control.
  - Novelty sentence: it stays literally true, since no face swap exists. But #12 keeps the prompt fixed across all faces, so its score cannot come from prompt text alone. That is image dependence by construction, without a mismatch control, and a reviewer may raise it. "None" should not imply that every paper is exposed to the prompt fault.
  - Authors' awareness: Sec. II cites option-order work and CircularEval, so the omission is not ignorance of the issue.
  - A3 relevance: the paper tests Qwen3-VL-8B and shows that the readout can reverse a verdict (mean 0.425 to 0.553 on graded ratings; only +0.030 accuracy on FACES). This supports the draft's caution that an A1 to A3 difference may reflect scoring.
  - Cue-level novelty: no AUs, no cue edits, no randomised human effects, so no threat.

### #12 Hallmen et al., Supplement
- Where read: https://arxiv.org/abs/2610.08162
- Coverage: partial. Sec. S5 to S7, S13 to S15 read directly, the rest keyword-scanned.
- Claim: (references.md row 12 and `LIT_CHECK_2026-11-07.md` section 2) "Image-free or mismatch: None"
  - Verdict: wrong (this is the repo note, not a draft sentence)
  - Quote: "re-ran both conditions on all2,052photographs with every image replaced by a uniform gray rectangle of the same size" (Sec. S14; missing spaces are in the file)
  - Note: All 22 runs answered neutral on every image. No mismatch arm, so this is image-free only. The authors add that it "does not rule out class biases on real inputs".
- Claim: (introduction.md para starting "We then asked how far") "one did not control it [12]"
  - Verdict: needs rewording
  - Quote: "with exactly one of: anger, disgust, fear, happiness, neutral, sadness." (Figure S4)
  - Note: The FACES text arm uses this one alphabetical list, with no rotation stated. The EmoNet text arm supplies the taxonomy as a list. The headline P(yes) arm has no option position. See the summary section for replacement text.

### #13 Zhang et al.
- Where read: https://arxiv.org/abs/2605.21980
- Coverage: read in full, including all appendices to the end (references skimmed). This is v2 of 24 Sep 2026. Figures and formulas are garbled in extraction (Figures 2 to 7 especially); prose, prompts and Tables 9 to 13 are readable.
- Metadata: fix: the text shows the venue, so replace "ICML 2026 (accepted per arXiv comment)" with Proceedings of the 43rd International Conference on Machine Learning, Seoul, PMLR 306, 2026. Title may be added: "Interpreting and Enhancing Emotional Circuits in Large Vision-Language Models via Cross-Modal Information Flow". Authors: Chengsheng Zhang, Chenghao Sun, Zhining Xie, Xinmei Tian. Status column can move to read in full; the file is 94,914 bytes, so the notes' "100,000 characters" figure cannot be right for this copy.
- Claim: (introduction.md para starting "We then asked how far") "in one the question did not apply [13]"
  - Verdict: needs rewording
  - Quote: "Is this emotion happy, sad, angry, fear, neutral or else?" (Appendix F)
  - Note: Option order does not apply to the main tasks, which are open-ended (continuation writing and hit rate on free text). But Appendix F lists labels inline in one fixed order, and the Appendix B.1 pre-screening "classification task" has an unreported format. Correction: "in one the main task was open-ended, so the question did not apply [13]".
- Claim: (introduction.md same para) "we found none that controlled option position and also swapped the face under a fixed prompt"
  - Verdict: needs rewording
  - Quote: "Crucially, we only include pairs where the model successfully exhibited the target emotion:" (Section 3.1)
  - Note: Literally true only because #13 has no option position to control; it does change the image under fixed text, so a reviewer can say it escapes both faults. What separates it from our audit is that it does not estimate image-dependent accuracy, its stimuli are mixed, filtered on the model, and the change is same-identity neutralisation, not a swap of faces. Correction: state these reasons, e.g. "none that estimated accuracy lost when the face is swapped, with option position controlled".
- Claim: (introduction.md same para) "The nearest is a design that crossed neutral text with paired images [13]"
  - Verdict: needs rewording
  - Quote: "These paragraphs are paired with both image variants to form contrastive sets" (Section 4.1)
  - Note: Right design, wrong verb: text is held fixed within a pair and one neutral event is randomly selected, so nothing is factorially crossed. Correction: "a design that held neutral text fixed across an emotional image and its generated neutral twin [13]".
- Claim: (introduction.md same para) "So most of the VLM emotion claims we located cannot separate perception from option order and prompt text."
  - Verdict: needs rewording
  - Quote: "X − = Concat(Ineu, Tneu)" (Section 3.1; extraction splits the line, see Note)
  - Note: #13 is an exception and must not be counted in "most": its mechanism analysis uses no option list and the text is identical across the pair. See #11 for the wider problem with this sentence.
- Claim: (introduction.md para starting "We located no study") "Coarse image-level counterfactuals on VLMs exist [13]"
  - Verdict: supported
  - Quote: "Modify the image to strictly neutralize ALL emotional indicators while preserving the original composition, identity, and style with" (Figure 12)
  - Note: "Coarse" is the draft's word, not the paper's, but it is fair: one edit resets mouth, eyes, brows, tears, flushing, symbols and props together, on Qwen3-VL and LLaVA-OneVision. No cue is manipulated alone.
- Claim: (introduction.md same para) "We located no study comparing a vision-language model's response to named facial cues with effects estimated from human observers"; "None of these, as far as we read them, compares a model with randomised human cue effects"
  - Verdict: supported
  - Quote: none (absence).
  - Note: Searched the whole text for action units, AU, observers, participants, human comparison and randomised cues. No AU appears, no human judgement data is used, and no effect is reported per named cue; mouth, eyes and brows appear only as joint instructions to the image generator. "Human-assisted" refers to dataset curation only.
- Claim: (methods.md, Literature check) "We tabulated option handling and image-free or mismatch controls in eight emotion papers [1,2,8-13]"
  - Verdict: supported
  - Quote: none (absence).
  - Note: #13 has no image-free run and no different-label mismatch; random pairs appear only as a baseline for similarity tables. Its neutral twin is a control of another kind and should be tabulated as such. The table itself is outside this check's file scope.
- Claim: (references.md row 13, Used for; Must read: "stimulus mix") "Image-level counterfactuals; partial analogue"
  - Verdict: supported
  - Quote: "The dataset spans a broad spectrum of domains, including human portraits, animal behaviors, anime characters, and abstract emojis." (Appendix B.1)
  - Note: Faces of people are only a subset, and the share is unreported. Scope is 4 basic emotions (Happy, Sad, Angry, Fear), 250 pairs for directions and another 250 for analysis, static images only.
- Claim: (references.md row 13, Must read: selection on the model; extra check) stimuli selected on the model
  - Verdict: supported
  - Quote: "Only samples where the model predicts the ground-truth emotion with high confidence (>95% accuracy) are retained." (Appendix B.1)
  - Note: Selection happens twice: this screen by the target LVLM, then the hit-rate filter at threshold 0.5 on the steering set. The draft could say so, since it is the strongest reason #13 is not a validity audit.
- Other findings: (1) Fairly close overlap in models: #13 reports Qwen3-VL-8B-Instruct, one of our three, though its primary analysis is on the 4B variant. (2) Figure 7 appears to give an Angry hit rate of about 18.67 on neutral inputs against 68.67 on emotional inputs, which is close to an image-dependence contrast on selected stimuli. Those values come from a garbled figure, so check the PDF before relying on this. (3) #13 is a within-model mechanism study with no human comparison, so it sits at no Schyns degree. The same holds for #11: neither compares with humans, so "sits almost entirely at the first degree" (introduction.md first para) needs a qualifier such as "compares behaviour at most, often with no human baseline". (4) Assets are promised on acceptance (Appendix B.6), so check whether the paired dataset is now public. No threat to the novelty claim.

### #14 O'Reilly et al.
- Where read: https://link.springer.com/content/pdf/10.3758/s13428-015-0601-4.pdf (publisher PDF)
- Coverage: read in full (reference list skimmed). PDF extraction: prose has some broken word spacing, Table 2 is intact, its 27 face rows sum to 249 items (49 Low). Supplementary files (Validation Data, Emotions Matrix) are not in the text file.
- Metadata: ok (text shows Behav Res (2016) 48:567-576, DOI matches, title "The EU-Emotion Stimulus Set: A validation study"). Status field can drop "through a summarising fetch tool": this pass read the paper directly and rechecked each number.
- Claim: (introduction.md para starting "An audit also has to ask") "face-only human observers chose the acted label on only 23 to 26% of responses"
  - Verdict: needs rewording (not in paper)
  - Quote: "Kind and unfriendly facial expressions had the lowest recognition rates, both at 9 %." (p. 574, Discussion)
  - Note: The paper reports only chance-corrected scores: Kind 9%, Unfriendly 9%, Jealous 14% (Table 2; the Results prose also gives 13%). Back-converting gives about 24, 24 and 28% (27.5 from 13%), though those means floor negative item scores at 0, so 23 to 26% may well be right for the local norms; cite that source for the raw figure, or give the published CCR values.
- Claim: (introduction.md same para; methods.md Human data) "against a chance level of 17%"
  - Verdict: supported
  - Quote: "Since we were using six response options, the raw scores were then adjusted for chance rates" (p. 571, Data treatment and analysis)
  - Note: The formula uses 1/6. "17%" is the rounded value and does not itself appear in the paper.
- Claim: (methods.md, Human data) "from the 249 face clips of the published set [14]"
  - Verdict: supported
  - Quote: "Face All All SUM: 249 SUM: 3,220 63 % 16 % 2.78 0.19 3.44 0.24 3.22 0.21" (Table 2)
  - Note: Also stated in the Method text. The count of 243 is not in paper; it is local.
- Claim: (methods.md, Human data) "243 silent face clips"
  - Verdict: supported
  - Quote: "No vocalizations were produced while the actors performed the facial expressions" (p. 569, Instructions)
  - Note: Supports "silent" only.
- Claim: (methods.md, Human data) "six response options per item (five named labels and None of the above)"
  - Verdict: supported
  - Quote: "The six response options consisted of the target emotion/mental state" (p. 570, Recognition task)
  - Note: The same sentence continues with four control labels and None of the above.
- Claim: (methods.md, Human data) "Clips show 21 labels (20 states plus neutral)"
  - Verdict: supported
  - Quote: "The set consists of the following 20 emotions/mental states, plus a neutral state" (p. 568)
  - Note: Table 2 also has 21 face labels.
- Claim: (methods.md, Human data; Statistics and sample size) "acted by 17 face actors"; "each actor portrayed 10 emotions plus neutral"
  - Verdict: supported
  - Quote: "For the facial expressions, each actor (N = 17) was asked to portray a subset of ten emotions/mental states, plus neutral." (p. 569)
  - Note: Table 1 agrees (19 actors, 2 without facial expressions). Basics in each set were portrayed twice (high and low), and two takes of an item could go forward, so items per actor and label can exceed 1. 17 actors times 14 portrayals is 238, against 249 items.
- Claim: (methods.md, Human data) "Actor and label are only partly crossed"
  - Verdict: supported
  - Quote: "The actors received facial expression scripts for either Set A or B." (p. 569)
  - Note: The paper also says this prevented comparison across actors.
- Claim: (methods.md, Human data; P6 "same 8 actors"; RQ2 Inference "With 8 actors") "Happy, Afraid and Disgusted came from one set of 8 actors, and Sad, Angry and Surprised from a separate set of 8 or 9"
  - Verdict: needs rewording (not in paper)
  - Quote: "The 20 emotions/mental states were divided into two sets of ten emotions/mental states each (three basic emotions and seven complex emotions/mental states)." (p. 569)
  - Note: The paper confirms two sets with three basics each, but never says which labels are in which set or how many actors had each. Table 2 counts items, not actors (Afraid Low 9 and Disgusted Low 10 exceed 8), so name the local metadata as the source, and resolve "8 or 9" from it.
- Claim: (methods.md, Human data) "Kind (9 items), Unfriendly (10) and Jealous (7)"
  - Verdict: supported
  - Quote: "Face Kind High 9 108 9 % 11 % 3.83 0.32 3.31 0.31 3.2 0.24" (Table 2)
  - Note: Unfriendly (10) and Jealous (7) match their rows too. These are counts for the 249-clip set; the paper cannot show that none of the 6 absent clips falls in these labels.
- Claim: (methods.md, Human data) "Face-only humans chose the target on 23 to 26% of these responses"
  - Verdict: needs rewording (not in paper)
  - Quote: "Jealous also achieved low recognition scores through facial expressions (13 %) and body gestures (3 %)" (p. 571)
  - Note: Same issue as the Introduction sentence. Table 2 gives Jealous 14%, so the paper disagrees with itself; if the CCR is quoted, use the table value and say so.
- Claim: (methods.md, Human data) "Human response distributions are the face-only norms [14]"
  - Verdict: needs rewording
  - Quote: "These data are also available on a stimulus item level" (p. 573, Table 2 note)
  - Note: "These data" are item counts, recognition (CCR), valence, intensity and arousal. The paper says raw rates were computed for all response options but does not state that per-option distributions are published, and the supplement is not here; check it, and if the distributions came from elsewhere, say where.
- Claim: (methods.md, Human data) "Human presentation order is not confirmed from the source, so human position effects are unknown."
  - Verdict: wrong
  - Quote: "For each stimulus, a video of the emotional stimuli was shown, together with six counterbalanced response options." (p. 570)
  - Note: The source does address option order, so the first clause is contradicted. Replace with: "Response options were described as counterbalanced [14], but the scheme is not given and no results by position are reported, so human position effects are unknown." Item order within surveys is not stated.
- Claim: (methods.md, RQ2 Predictions, P13) "48 Low Intensity items"
  - Verdict: needs rewording (not in paper)
  - Quote: "Face Selection** Low SUM: 49 SUM: 748 63 % 18 % 2.69 0.17 3.25 0.28 3.08 0.25" (Table 2)
  - Note: The paper has 49 Low items. 48 may be right for the 243 local clips, and no [14] citation is attached, but check that one of the 6 absent clips is Low.
- Other findings: (1) The human task also used one fixed option set per target label: "The four control response options varied depending on the target emotion/mental state portrayed." (p. 574). So the label-identifies-category defect comes from the validation design. The Introduction could say so, and human option effects cannot be separated from it either. (2) With counterbalanced human options, v1's fixed first position differed from the human task. (3) Norms pool three languages (UK, Sweden, Israel) with translated labels, while the draft's prompts appear to be English only. (4) Each actor having a neutral clip (Table 2: 17 Neutral items) fits the neutral-fill condition, though the paper does not map clips to actors. No threat to novelty claims.

### #15 Geirhos, Meding and Wichmann
- Where read: https://arxiv.org/abs/2006.16736
- Coverage: read in full. Main text and supplement S.1 to S.6 read line by line; references skimmed. Lines 1300 to 5369 are figure glyphs: only their text labels and captions (SF.5 to SF.12) were read. Equations are garbled in extraction, so definitions below come from the prose and the legible form of eqs 1 and 4.
- Metadata: ok on authors, venue and arXiv ID. Add the title, "Beyond accuracy: quantifying trial-by-trial behaviour of CNNs and humans by measuring error consistency". Geirhos and Meding are joint first authors. Status can move from snippet to read in full.
- Definition, for the record: for two observers on the same n trials, each trial is scored only correct or incorrect. c_obs is the share of trials where both are correct or both incorrect; c_exp = p_i p_j + (1 - p_i)(1 - p_j); kappa = (c_obs - c_exp) / (1 - c_exp). It is Cohen's kappa on correct/incorrect, computed per pair of observers. It requires a correct answer and identical stimuli for both observers.
- Claim: (introduction.md para starting "The second problem") "Behavioural metrics such as error consistency [15] compare outputs on natural items."
  - Verdict: wrong
  - Quote: "three different types of out-of-distribution (o.o.d.) images" (section 2.3)
  - Note: Three of the four stimulus sets in [15] are not natural (cue conflict from style transfer, edges, silhouettes), the fourth is a baseline, and the metric has no restriction to natural items; it also compares correct/incorrect patterns, not outputs. Fix: "Behavioural metrics such as error consistency [15] compare responses on a shared item set. On natural items, this comparison cannot settle cue use."
- Claim: (methods.md, Identifiability results, Setup) "A behavioural metric is any functional of the joint distribution of items and responses on P. This covers ... error consistency"
  - Verdict: supported
  - Quote: "we only analyse whether the decisions were correct/incorrect (irrespective of the number of choices)" (section 2.1)
  - Note: Holds if an "item" carries its correct answer, since kappa needs one, and if both systems' responses are taken jointly per item. The same applies to the Introduction's "every behavioural metric on those items gives the same value".
- Claim: (methods.md, A4, Behaviour beside CFA) "Snoek data have no correct answer, so error consistency proper is undefined, and modal agreement is its closest analogue."
  - Verdict: supported
  - Quote: "we only analyse whether the decisions were correct/incorrect (irrespective of the number of choices)" (section 2.1)
  - Note: Undefined is right without an imposed reference label; [15] itself imposed one on cue conflict items by scoring the shape category as correct (Table 1). Chance-corrected modal agreement is plain Cohen's kappa on labels, the statistic [15] says it repurposes, so "closest analogue" is fair, though its chance term is over K options, not binary.
- Claim: (methods.md, A4, Behaviour beside CFA) "On the same stimuli we report output similarity and chance-corrected modal agreement"
  - Verdict: needs rewording
  - Quote: "As a prerequisite, all decision makers need to evaluate the exact same stimuli." (section 2)
  - Note: A4 renders are "a new stimulus set of the same AU recipes" and do not reproduce what humans saw, so the comparison shares AU configurations, not stimuli. This is a second reason error consistency proper is unavailable. Fix: "On the same AU configurations".
- Claim: (references.md row 15 and Priority 2 note) "Error consistency"; "Definition of error consistency, which the identifiability claim addresses"
  - Verdict: supported
  - Quote: "Here we repurpose and extend Cohen’s" (Related work)
  - Note: The paper introduces and defines the metric. Definition is given above.
- Other findings:
  - [15] already states the non-sufficiency in words: same decision rule, different strategies (Quote: "Note that the same decision rule can result from different strategies."). Its example (Quicksort, Mergesort) is identical everywhere as a function, while Theorem 1 concerns systems equal only on the tested support. So it anticipates the slogan, not the result. The Introduction should credit this if it claims the point as new.
  - [15] claims the other direction: "For error consistency, zero error consistency implies that two decision makers are not using the same strategy." "This comparison cannot settle cue use" should therefore be read as "cannot establish", because [15] treats low consistency as falsifying. For linear observers, the lower bound in Proposition 3 agrees: unequal behaviour forces unequal effects.
  - [15] links the metric to cue use on cue conflict stimuli (S.6, four models): "Model shape bias predicts human-CNN error consistency". A reviewer could cite this against "cannot settle". The answer is that those stimuli break a natural cue pairing, and the link is correlational across four models. That fits the draft's conclusion, but it is one more reason not to say the metric lives on natural items.
  - Theorem 1 needs only a finite tested support, not natural items, so it applies to any fixed test set whose shifted points lie off that set. Only Propositions 2 and 3 use the natural covariance.

### #16 Han, Poggio and Cheung
- Where read: https://arxiv.org/abs/2302.06677
- Coverage: read in full, including Appendix A.1 to A.4; references skimmed. Figures are axis numbers only in extraction, so figure content is known from captions and text.
- Metadata: ok. Text shows the title, the three authors, and "Proceedings of the 40 th International Conference on Machine Learning, Honolulu, Hawaii, USA. PMLR 202, 2023." Status can move from snippet to read in full. Replace "Related limits. Relation unread" with the relation below.
- What it shows: an empirical simulation. Known ANNs replace brain recordings; six source architectures, then 12 CNNs and 14 ViTs, are scored against a target with ridge regression (Pearson r per unit) and linear CKA, both on internal layer activations. Findings: the correct architecture usually but not always ranks first; for several layers in VGG11, ResNet18 and CORnet-S a wrong architecture wins; identifiability depends on stimuli, target coverage and depth; convolution and attention scores overlap across many layers.
- Claim: (introduction.md para starting "The second problem") "limits of a related kind have been discussed elsewhere [16,17]"
  - Verdict: supported
  - Quote: "This fundamental limitation of present evaluation techniques, such as the linear encoding analysis used in isolation, emphasizes the need to develop new approaches beyond comparing functional similarities." (Discussion)
  - Note: True only at the level of "similarity scores do not determine mechanism"; the object differs (architecture, from internal activations), and the evidence is empirical, not a proof. Optional sharper text: "Empirical limits on identifying architecture from activation similarity have been shown for neural networks [16]". This check covers [16] only.
- Claim: (references.md row 16 and Priority 2 note) "Related limits"; "The theory note requires a full read before any novelty claim for these results"
  - Verdict: supported
  - Quote: "Such ambiguity is due to the many-to-one mapping of a model onto a leaderboard score." (section 1.1)
  - Note: The nearest statement to the draft's logic is this qualitative many-to-one remark. A full read finds no pre-emption, as set out below.
- Other findings: exact relation to the Methods results. [16] does not pre-empt Theorem 1 or Propositions 2 to 4. A case-insensitive search also found no "theorem", "unidentif", "causal" or "intervention" in the file, and "correlat" appears only for Pearson correlation and a cited title.
  - Theorem 1 (any effect matrix is compatible with identical behaviour on a finite support, by a constructive off-support change): [16] has no existence or impossibility result and never perturbs a system or measures off-stimulus behaviour. Its ideal-case finding is that the truth usually wins: "For most target layers, source networks with the highest median score are the correct network".
  - Proposition 2 (two correlated cues, output correlation rho, CFA 0): no cues, no cue correlation, no observers defined on cues.
  - Proposition 3 (eigenvalue bounds on effects from behaviour): no bound of any kind. Singular values appear only in describing what regression and CKA depend on, for representations, not for a cue covariance.
  - Proposition 4 (unmeasured cue, non-zero slopes, no interventional effect): no omitted-variable argument. Its regression maps source features to target units.
  - Differences that matter: [16] compares internal activations, while the draft's metrics are functionals of outputs; [16] targets architecture, while the draft targets cue effects; [16] runs a finite model set, while the draft quantifies over all systems. [16] also evaluates trained networks, which the draft's Theorem 1 says it does not address, so it is the complementary kind of evidence.
  - Tension to anticipate: its first recommendation is "Using stimuli images that are more natural." The draft says natural items cannot identify cue effects. There is no contradiction, because the target and the metric differ, and [16] also proposes "specially designed, critical stimuli", which is closer to the draft. A reviewer may raise it.
  - Name clash: its "Identifiability Index" is a normalised score difference, not statistical identifiability.

### #17 Teneggi and Sulam
- Where read: https://arxiv.org/abs/2405.19146
- Coverage: read in full for the main text (sections 1 to 6, arXiv v2). Appendices skimmed, as the prompt allowed: A, B and C (including proofs), D.1 prose read, D.2 to D.4 captions only; references skimmed. PDF extraction: page 1 figure, equations and figure labels are garbled, so very short fragment lines in sections 4 to 6 were filtered out; prose was not.
- Metadata: fix: (a) add the title, "I Bet You Did Not Mean That: Testing Semantic Importance via Betting" (shown in the text and in abs.html). (b) Authors and arXiv ID ok; the first version date in abs.html is 2024/05/29 and the text is v2 of 7 Oct 2024. (c) "NeurIPS 2024" cannot be confirmed: neither paper.txt nor abs.html names any venue, so check it elsewhere before submission. (d) Status "snippet" is stale: main text now read, appendices skimmed.
- Claim: (introduction.md para starting "The second problem is one of identifiability") "limits of a related kind have been discussed elsewhere [16,17]"
  - Verdict: wrong
  - Quote: "In this work, we formalize semantic importance at three distinct levels of statistical independence with null hypotheses of increasing granularity" (section 1)
  - Note: The paper is a positive testing framework (definitions of concept importance, SKIT, c-SKIT, x-SKIT, an FDR post-processor) and contains no limit on what behaviour or natural data can show about cue use; I searched for identifiability, indistinguishable systems, interventions and do-style effects and found none. Correction: drop 17 from this citation, and if wanted add "Concept-level tests of conditional independence exist for vision-language encoders [17]", as a neighbour and not as a limit (the decision on #16 is outside this batch).
- Claim: (references.md row 17, "Used for") "Related concept tests. Relation unread"
  - Verdict: needs rewording
  - Quote: "We use recent ideas of sequential kernelized independence testing (SKIT) to induce a rank of importance across concepts" (abstract)
  - Note: "Related concept tests" is fair; "Relation unread" no longer holds. Replace with: "Statistical tests of concept importance on natural data, by conditional independence; not an identifiability result."
- Claim: (methods.md, Identifiability results, Theorem 1 and Propositions 2 to 4; introduction.md "We show that this comparison cannot settle cue use") implied to be this paper's own results
  - Verdict: supported
  - Quote: "We show in the short lemma below (whose proof is in Appendix C.1) that this is false, and that, arguably unsurprisingly, statistical independence is conceptually different from orthogonality between vectors, which motivates the need for our testing procedures." (section 3.1)
  - Note: Not pre-empted: none of Theorem 1 or Propositions 2 to 4 appears, in statement or in substance, and Lemma 1, quoted here, is the nearest technical item. See the item-by-item comparison under Other findings; the same holds for Proposition 5 (no attenuation or reliability result).
- Other findings:
  - What it establishes. For a fixed model and user-chosen concepts: global importance (Y-hat independent of Z_j), global conditional importance (Y-hat independent of Z_j given Z_-j) and local conditional importance on one sample, each with a sequential kernel test with Type I error control, plus greedy FDR control and a rank from rejection times. Tested on Gaussian data, a ResNet18 counting MNIST digits, and zero-shot classification on AwA2, CUB and Imagenette across 8 encoders.
  - Item by item against Methods (this comparison is ours; the paper does not make it). Theorem 1: no counterpart; no pair of systems equal on natural items, no cue-effect matrix. Proposition 2: nearest neighbour is Lemma 1 (a linear classifier orthogonal to a concept can still be statistically dependent on it, and the reverse), which is about one classifier against a concept vector under an arbitrary H, not two observers, not correlated cues and not CFA. Proposition 3: no counterpart; no eigenvalue or covariance bound. Proposition 4: nearest neighbour is the remark that marginal importance ignores other concepts, quoted next, which motivates conditioning among measured concepts and says nothing on an unmeasured correlated cue.
  - Quote for that remark: "but it ignores the information contained in the rest of them, and concepts may be correlated" (section 3.1). The only related limitation stated is about concept choice: "there is no guarantee that these are well-represented in the feature space of the model, nor that they are the most informative ones for a specific task" (section 6). The other stated limits are samplers and characteristic kernels.
  - It works the other way round from a precedent: its importance is observational dependence, so it is the kind of analysis Proposition 4 warns about. c-SKIT swaps Z_j in the test data and does not re-query the model, and x-SKIT draws embeddings from real images: "Step 2. above makes sure that samples are coming from real images" (appendix D.1). Where Z is stimulus-defined (CUB annotations, synthetic digits), the tests read only natural-support behaviour, so Theorem 1 applies to them (my inference). Elsewhere Z is not a stimulus cue at all: "Note that both ˆY and Zj are fixed functions of the same random variable H" (section 3.1).
  - Risk to the draft's reading: "to learn which cues a system uses, one has to break the natural correlations between cues" (Introduction, not cited to 17) may draw a reply from this literature, because c-SKIT tests dependence on a concept given the others using residual natural variation without breaking correlations. The draft's position survives only because of unmeasured cues and non-linear systems (Theorem 1, Proposition 4); under Proposition 3, linear observers with lambda_min above 0 are constrained by natural data. Consider "one has to break the natural correlations, or assume that no relevant cue is unmeasured".
  - Novelty claim: no threat. No faces, emotions, human observers or randomised human effects. "vision-language models" here means CLIP-style image and text encoders (CLIP, OpenClip, FLAVA, ALIGN, BLIP) used for zero-shot classification, not models that answer in words, so it should not be counted as VLM cue work in the Introduction's sense. One scoop signal: "we plan to design user-studies to quantitatively evaluate the alignment of the produced ranks with human intuition" (section 5.3), future work only and about intuition ratings, not randomised cue effects.

### #19 Snoek et al. 2023
- Where read: https://pmc.ncbi.nlm.nih.gov/articles/PMC9916981/ (Europe PMC XML)
- Coverage: read in full (whole main text including Materials and Methods). The reference list is omitted from the file, so numbered citations such as ( 11 ), ( 67 ) and ( 73 ) could not be resolved. The Supplementary Materials are not in the file. I did not open the sibling full.xml, because the brief limits work to the named text file.
- Metadata: ok. Snoek et al., Science Advances 9(6), eabq8421 (2023), DOI matches. Optional: add the title "Testing, explaining, and exploring models of facial expressions of emotions". The "read in full" status holds for the main text, not the supplement.
- Claim: (introduction.md para starting "Human data of that kind exist") "In reverse-correlation studies, observers judge faces whose facial action units (AUs) are set at random [18,19]."
  - Verdict: needs rewording
  - Quote: "Each trial comprised an agnostically generated facial animation composed of a random combination of dynamic AUs." (Prediction)
  - Note: The random design is plainly in #19, but the phrase "reverse correlation" appears nowhere in it; I searched "reverse" with no hits. #19 tests AU models on such data and does not present itself as a reverse-correlation study. Correction: keep the sentence and cite #42, which uses the term and the design, as in "[19,42]" or "[18,19,42]".
- Claim: (introduction.md, same sentence) the pairing [18,19], with #18 unopened
  - Verdict: needs rewording
  - Quote: "The WE data have previously been used and described in ( 11 , 73 ) and included 59 European participants and 1 North American participant (31 female; mean age = 22 years, SD = 1.7 years)." (Participants)
  - Note: #19 is a secondary analysis, and its text never names a 2012 paper, so it gives no independent support for #18. As read here, the sentence rests on #19 and #42, with #18 unverified. #42's ref 4 shows the 2012 paper as "Facial expressions of emotion are not culturally universal", PNAS 109, 7241-7244, cited there for perceptual expectations; that is the only trace of #18 in this batch.
- Claim: (introduction.md, next sentence) "Randomisation identifies the effect of each AU on human judgement."
  - Verdict: needs rewording
  - Quote: "Although the AUs and their parameters were randomly sampled, the resulting facial expressions all displayed morphologically plausible facial movements, because the generative facial expression model prohibits impossible movements with morphological constraints [see ( 69 ) for details]." (Materials)
  - Note: Sampling is constrained, and left/right and compound AUs are tied by the recoding. #19's own causal language is about AU ablation in models, not effects on humans. Correction: "Randomisation, within the generator's morphological constraints, identifies..."; this is the draft's inference and not a #19 finding.
- Claim: (introduction.md para starting "Here we ask") "faces rendered from the AU recipes that human observers judged"
  - Verdict: supported
  - Quote: "Our analyses only used the peak amplitude parameter and ignore the six other (temporal) parameters." (Materials)
  - Note: Recipes here means AU amplitudes only, since humans also saw timing. Methods states this limit. #19's "six other" is its own slip, because peak amplitude is one of the six.
- Claim: (methods.md Snoek AU data; Data availability) "public trial-level data [19,25]" and "Snoek data are public [25]"
  - Verdict: supported
  - Quote: "Data from both datasets are deposited on figshare ( https://doi.org/10.21942/uva.21261885 )." (Data and materials availability)
  - Note: "Trial-level" is not stated in the paper and rests on #25.
- Claim: (methods.md Snoek AU data) "(CC BY 4.0)"
  - Verdict: needs rewording
  - Quote: "Distributed under a Creative Commons Attribution License 4.0 (CC BY)." (copyright statement)
  - Note: This licence covers the article. The paper states no licence for the data deposit, so the data licence is not in the paper. Confirm it from the #25 record, or attach CC BY to [19] only.
- Claim: (methods.md Snoek AU data; A4 limits) "1.25 s animations"; "humans saw 1.25 s of movement"
  - Verdict: supported
  - Quote: "All animations had a duration of 1.25 s (30 frames, presented at 24 frames/s)." (Materials)
  - Note: Exact match.
- Claim: (methods.md Snoek AU data) "random AU combinations"
  - Verdict: supported
  - Quote: "Each face was animated with a subset of randomly selected AUs from a set of 42 possible AUs (with the number of AUs drawn from a binomial distribution with n = 5 and P = 0.6)." (Materials)
  - Note: Each stimulus carries few AUs, three on average by that binomial.
- Claim: (methods.md Snoek AU data; A4 Stimuli) "on generated faces"; "the original generated faces"
  - Verdict: needs rewording
  - Quote: "Each dynamic facial expression stimulus comprised one of eight “base faces” and was of the same ethnicity as the participant (WE base faces: four males, four females, mean age = 23 years, SD = 4.1 years; EA base faces: four males, four females, mean age = 22.1 years, SD = 1.0 years)." (Materials)
  - Note: #19 calls the animation generated. The faces are eight base faces per group, with reported ages, so "generated faces" goes beyond the text. Correction: "computer-generated animations on eight base faces per group, matched to observer ethnicity".
- Claim: (methods.md Snoek AU data) "with a frontal view, flat lighting"
  - Verdict: supported
  - Quote: "The facial animations were rendered from frontal view using flat lighting, which avoids shadowing [see supplementary movie S1 from ( 73 ) for an example]." (Materials)
  - Note: Exact match.
- Claim: (methods.md Snoek AU data) "and a black background"
  - Verdict: supported
  - Quote: "Participants viewed stimuli on a black background displayed on a monitor with a diagonal of 48.26 cm, a refresh rate of 60 Hz, and resolution of 1024 × 1280." (Materials)
  - Note: Exact match.
- Claim: (methods.md Snoek AU data) "Basic emotions used seven options, including "other", and conversational states used five."
  - Verdict: supported
  - Quote: "In each experimental session, participants completed a seven-alternative forced-choice (basic emotion dataset) or five-alternative forced-choice (conversational signal dataset) emotion categorization task of 200 dynamic facial expression stimuli." (Procedure)
  - Note: The Procedure section says "other", while the earlier Prediction section says "do not know". The draft follows the Procedure.
- Claim: (methods.md A4 Task) "Human "other" maps to None of the above"
  - Verdict: supported
  - Quote: "We excluded the trials categorized as other from our analyses because there are no specific hypotheses about this category." (Preprocessing)
  - Note: The mapping is the draft's own choice, and it fits the "only if the expression matched" instruction. It differs from #19, which drops those trials, so the draft's effect matrices are not #19's sample.
- Claim: (methods.md Snoek AU data) "Western (WC) and East Asian (EA) groups took part: 60 observers per group for basic emotions"
  - Verdict: supported
  - Quote: "The basic emotion dataset contains data from 60 WE and 60 EA participants." (Participants)
  - Note: #19 abbreviates Western as WE, not WC, which is a harmless label difference. EA participants were Chinese, and the basic EA group was tested in China.
- Claim: (methods.md Snoek AU data) "20 per group for conversational states"
  - Verdict: supported
  - Quote: "The conversational signal dataset contains data from 20 WE and 20 EA participants and has previously been used and described in ( 15 , 74 )." (Participants)
  - Note: The conversational EA group was tested in Glasgow, unlike the basic EA group.
- Claim: (methods.md Snoek AU data) "(140,910 and 139,728 trials)"; "(47,300 and 47,460 trials)"
  - Verdict: needs rewording (not in paper)
  - Quote: "For the basic emotion dataset, this leaves a grand total of 247,782 trials (total WE: 119,382, total EA: 128,400) with an average of 2065 trials per participant (average WE: 1990, average EA: 2140)." (Preprocessing)
  - Note: The paper gives counts after dropping "other" (conversational: 40,540 and 43,000). The draft's counts are larger and at or below 2,400 per observer, which fits data that keep "other" (the gaps are about 15, 8, 14 and 9% of the draft's counts), but that needs confirming from #25. No conflict.
- Claim: (methods.md Snoek AU data) "33 AU amplitude columns, recoded from 42 raw AUs"
  - Verdict: supported
  - Quote: "The original set of 42 AUs comprised 3 compound AUs (AU12 + 25, AU1 + 2, and AU6 + 12), 15 unilateral AUs (left or right, e.g., AU12L and AU12R), and 24 bilateral AUs (such as AU12)." (Preprocessing)
  - Note: The printed list holds 33 AUs. Column layout in the public files is not in the paper.
- Claim: (methods.md Snoek AU data) "They hold no timing parameters, pose or gaze, and no stimuli."
  - Verdict: needs rewording (not in paper)
  - Quote: none. I looked for any description of the deposit's contents and found only the deposit statement.
  - Note: This rests on #25. Likewise "the original generated faces and timing parameters are not public" (A4 Stimuli) is not in the paper.
- Claim: (methods.md Snoek AU data) "Six basic emotions and four conversational states overlap about 8 of the 21 EU-Emotion labels, and include none of Worried, Disappointed, Kind or Unfriendly."
  - Verdict: supported
  - Quote: "Participants were instructed to label the stimuli with one of the six basic emotions (anger, disgust, fear, happy, sadness, and surprise; basic emotion study) or one of the four conversational signals (bored, confused, interested, and thinking; conversational signal study)—but only if the facial expression matched one of the emotion categories." (Procedure)
  - Note: Supported for the Snoek side, including the "none of" list. The "about 8 of 21" figure also needs the #14 label list, which is outside this batch.
- Claim: (methods.md A4 Stimuli) "each on one own-group identity"; "identity coverage is narrower than for humans"
  - Verdict: supported
  - Quote: "was of the same ethnicity as the participant" (Materials)
  - Note: Humans saw eight base faces per group, so narrower holds, given A4's mostly one or two identities per configuration.
- Claim: (methods.md A4 Sensitivity and limits) "their effects average over time course"
  - Verdict: supported
  - Quote: "Given that AU onset ( 11 , 23 ) and speed ( 24 ) influence the emotional interpretation of facial expressions, additional temporal information about AUs is likely to improve model performance." (Discussion)
  - Note: #19 ignores timing in its analyses and names this as a limit.
- Claim: (methods.md A4 Behaviour beside CFA) "Snoek data have no correct answer"
  - Verdict: supported
  - Quote: "Each model’s predictive performance depends on the correspondence between its predictions and the actual participants’ emotion labels (see step 5 in Fig. 7 )." (Hypothesis kernel analysis)
  - Note: Stimuli are random, with no acted target, and #19 scores models against human labels only.
- Claim: (methods.md Cue spectra, Rows column) "Snoek basic WC | 4,659 configurations"; "Snoek basic EA | 2,322"; "Snoek conversational WC | 2,314"; "Snoek conversational EA | 2,322"
  - Verdict: needs rewording (not in paper; check before use)
  - Quote: "This grand total contains 6473 repeated trials (total WE: 4658, total EA: 2322), i.e., stimuli with the same AUs and amplitudes, with an average of 38 repetitions per participant (average WE: 26, average EA: 55)." (Preprocessing)
  - Note: The draft's "configurations" nearly equal #19's "repeated trials" (4,658 against 4,659; 2,322; and 2,314 for conversational WE). With about 140,000 random trials, all configurations should far outnumber 4,659, so check in #25 whether the spectra used only a repeated subset; the same check applies to the A4 manifest. #19's subtotals also fail to sum (4658 + 2322 is not 6473; 2314 + 2322 is not 4134).
- Claim: (methods.md Cue spectra) "the Snoek designs have 25 cues"; "bilateral pairs, which are correlated by design"
  - Verdict: supported
  - Quote: "To encode each AU as an independent variable, we recoded compound AUs (e.g., AU1 + 2 as activation of both AU1 and AU2) and bilateral AUs (e.g., AU12 as activation of both AU12L and AU12R), yielding a total of 33 AUs" (Preprocessing)
  - Note: The list has eight left/right pairs, so 33 minus 8 gives 25. The spectral values themselves are not in the paper.
- Other findings: (1) #19 does not threaten novelty: it compares AU-based predictive models with humans, with no vision model or VLM, and its ablation acts on those models. It is the nearest conceptual relative of the cue-effect idea. (2) Face ethnicity is matched to observer group, so the WC-against-EA yardstick mixes observer culture with stimulus faces. The basic EA group was also tested in China and the conversational EA group in Glasgow. (3) #19 reports its own categorical noise ceiling (average AUROC 0.88) and large individual differences, which are worth noting beside the draft's ceilings. (4) #19's "Jack et al. 2014" model gives anger 9 + 10 + 16 + 22 and disgust 9 + 10 + 11 + 43, consistent with shared AU9 in P15.

### #20 Huynh et al.
- Where read: https://arxiv.org/abs/2606.15779
- Coverage: read in full to the end (references skimmed). This is v1 of 14 Jun 2026. Text is clean; tables are readable, and Figure 1 is partly garbled.
- Metadata: ok. Optional additions: title "Faithful Action-unit Causal Reasoning for Counterfactually Faithful Emotion Explanations"; authors Van Thong Huynh, Hong Hai Nguyen, Thuy Pham, Trong Nghia Nguyen, Soo-Hyung Kim. The text names no venue. Status column can move to read in full.
- Claim: (introduction.md para starting "We located no study") "latent AU interventions on an affective model [20]"
  - Verdict: supported
  - Quote: "applies do-interventions in the AU latent space" (Section 1)
  - Note: Confirmed: the intervention is do(z_k = 0) on a sigmoid AU latent, not on pixels. The model is a ResNet-18 encoder with a class head and a small templated verbalizer, so "affective model" is right and "VLM" would be wrong.
- Claim: (references.md, Must read, #20) "Confirm 'acts on a latent' and the pixel-level future work statement"
  - Verdict: supported
  - Quote: "a pixel-level counterfac- tual is left to future work." (Section 6; the hyphen is the line break in the file)
  - Note: Stated in the fourth limitation. Rendered stimuli are not used anywhere, and this is the plainest sign of where the group may go next.
- Claim: (introduction.md same para) "None of these, as far as we read them, compares a model with randomised human cue effects."
  - Verdict: supported
  - Quote: "metric, which we evaluate against a known causal structure, the PSPI pain-AU composition, as no existing" (Abstract)
  - Note: Searched for human, observer, participant, rater and randomised: none used. Targets are a fixed formula (PSPI), EMFACS prototypes and frame-level AU labels, and the only "randomness" is a permuted-graph control. Humans appear only as filmed subjects, and the AU annotation source is largely automated or not described (FABA labels are GPT-4V derived).
- Claim: (introduction.md same para) "We located no study comparing a vision-language model's response to named facial cues with effects estimated from human observers under randomised cue presentation."
  - Verdict: supported
  - Quote: "so this static naming agreement is the only faithfulness measure available for it" (Section 4.4)
  - Note: The one VLM here, EmoLA, is scored on which AUs it names against labels, with no cue manipulated and no human effect. The paper also says affective MLLMs expose no AU latent, so its intervention cannot be applied to them.
- Other findings: Neighbours the draft does not name, cited by #20 and not read here: CausalAffect [its ref 4] uses a feature-level counterfactual in a disentangled AU latent, AULLM++ [12] pairs an LLM with an AU graph and a counterfactual regulariser, EDCT [27] audits VLM explanations by counterfactual image edits, and TAG [8] ties reasoning to AU regions. As #20 describes them, none uses randomised human effects, but the "located no study" sentence should have seen them. Section 4.7 also removes an AU from text evidence given to a frozen language model, which is a text intervention, not a facial one. No direct scoop in this paper; it does show AU-level causal audits are active.

### #21 Chen FOCUS
- Where read: https://arxiv.org/abs/2601.06931
- Coverage: read in full, including appendices. PDF extraction: figures and heatmaps are reduced to axis labels, tables are flattened, and some spaces are lost.
- Metadata: fix: add the title, "Measuring Social Bias in Vision-Language Models with Face-Only Counterfactuals from Real Photos". FOCUS is the dataset and REFLECT the benchmark. Authors shown: Chen, Huang, Zhao, Jiang, Chang, Yu. arXiv 2601.06931 matches (v2, 20 Apr 2026). "ACL 2026" does not appear in the text file, so the venue is unconfirmed from this copy.
- Claim: (introduction.md para starting "We located no study") "Related work edits race and gender in face images [21]"
  - Verdict: supported
  - Quote: "Starting from real photographs, we generate counterfactual variants by editing only facial attributes related to race and gender, keeping all other visual factors fixed." (Abstract)
  - Note: Plainly backed. Optional tightening: these are real photographs of people in occupational scenes with the face region edited, not face crops, so "edits race and gender on faces in real photographs" is more exact.
- Claim: (introduction.md para starting "We located no study") "None of these, as far as we read them, compares a model with randomised human cue effects"; also "named facial cues" in the first sentence of that para
  - Verdict: supported
  - Quote: "Your role is to approximate how people in society might intuitively judge others based on appearance, even if these judgments may reflect stereotypes or be socially undesirable." (Figure 8, 2AFC prompts; nearest statement to a human target)
  - Note: Looked for human observers, ratings, baselines, reverse correlation and randomised cues, and found none. There is no human perceptual target: the human element is only a persona instruction to the model, demographic labels were checked with GPT-4o, and expression drift was labelled by VLM prompts. Edits are bundled race and gender presentation, not named cues, and the authors disclaim a strict causal decomposition.
- Claim: (references.md row 21, Used for) "Race and gender face counterfactuals"; (Must read, Why) "confirm that none uses randomised human cue effects"
  - Verdict: supported
  - Quote: "Adjust skin tone and subtle facial features so the person is perceived as a {TARGET_GROUP}." (Figure 7, editing prompt)
  - Note: Confirmed. The row status can move from abstract once the author accepts this read. #21 is cited nowhere else in introduction.md or methods.md, and no range includes it.
- Other findings: Does not break novelty, but a reviewer may raise it. On its 2AFC task, Chen queries each pair in AB and BA order and edits faces under a fixed prompt, so it does control position and swap the face under a fixed prompt, on social judgements (income, education, perceived safety), not emotion. The Introduction's "In the seven we could check, we found none that controlled option position and also swapped the face under a fixed prompt" is safe only because it is scoped to the seven emotion papers. Keep that scope explicit. Chen's MCQ tasks report no option rotation.

### #22 Kataria VideoFace-X-LLM
- Where read: https://arxiv.org/abs/2610.17791
- Coverage: read in full. PDF extraction: equations are garbled, tables are flattened, and many spaces are lost (for example "top10%highest-attention"), which is reproduced in the quotes.
- Metadata: fix: drop "et al." from the citation. The text shows exactly four authors: Anubhav Kataria, Shreya Ghosh, Jingwen Ye, Abhinav Dhall. Title and arXiv 2610.17791 match (v1, 13 Oct 2026).
- Claim: (introduction.md para starting "We located no study") "tests whether facial regions are necessary for fine-tuned models [22]"
  - Verdict: needs rewording
  - Quote: "blur the top10%highest-attention patches in the peak frame and check whether the attribute prediction changes" (Section 5, Phase 3, Spatial validation)
  - Note: Fine-tuned is true, but it is one model, not "models" (next claim). The "regions" are the model's own top-attention patches, not preset facial regions, and they need not be facial: the worked example targets black hair. The test also covers peak frames. Suggested wording: "uses targeted occlusion to test whether the frames and image regions a fine-tuned video VLM attends to are necessary for its predictions [22]".
- Claim: (same sentence) are the models fine-tuned?
  - Verdict: supported
  - Quote: "use VideoFace-X-LLM itself to check whether the target attribute is still detected after masking" (Section 5, Phase 3)
  - Note: Yes. VideoFace-X-LLM is InternVideo2.5 fine-tuned on the ExplainFace corpus, and it is the only model named for the necessity test. Qwen3-VL was also fine-tuned on ExplainFace, but only for the task benchmark, with no necessity result reported for it.
- Claim: (introduction.md para starting "We located no study") "None of these, as far as we read them, compares a model with randomised human cue effects"
  - Verdict: supported
  - Quote: "We use the termcausalin the sense of a controlled intervention test, blur the evidence and measure the effect which establishes that the identified evidence isneededfor the prediction; it confirms necessity rather than recovering the model’s full internal reasoning." (Section 5, end of Phase 3)
  - Note: Looked for human perceptual effects, randomised cues and reverse correlation, and found none. The two human studies only validate corpus text (description matching and video retrieval). AUs appear only as a DISFA detection task scored by F1 and mean accuracy, and the intervention is compared with random blurs, not with people.
- Claim: (references.md row 22, Used for) "Region necessity tests"
  - Verdict: needs rewording
  - Quote: "We validate the Phase 2 attention regions the same way." (Section 5, Phase 3, Spatial validation)
  - Note: Change to "Attention-selected region and frame occlusion tests". Same reason as the first claim.
- Other findings: This is the nearest neighbour to A2's interventions: blurring frames and regions on a video VLM, including emotion, with explicit causal language. It differs in that regions are chosen by the model's attention, the outcome is a prediction flip against random blur, and no human prediction is stated, so A2's stated-human-prediction test stays distinct. It reaches beyond behaviour, so it is one of the exceptions to "sits almost entirely at the first degree"; "almost" covers it. No option handling is reported.

### #23 Tsangko
- Where read: https://arxiv.org/abs/2506.19079
- Coverage: read in full. PDF extraction: Figures 1 and 3 to 7 are absent as images, so per-model teeth results in Fig. 5 could be read only through the text; tables are legible.
- Metadata: fix: add year 2025 (arXiv 2506.19079v1, 23 Jun 2025) and the title, "Reading Smiles: Proxy Bias in Foundation Models for Facial Emotion Recognition". Authors shown: Tsangko, Triantafyllopoulos, Abdelmoula, Mallol-Ragolta, Schuller.
- Claim: (introduction.md para starting "We located no study") "relates annotated teeth visibility to VLM performance [23]"
  - Verdict: supported
  - Quote: "We benchmark varying scale VLMs on a teeth-annotated subset of AffectNet dataset and find consistent performance shifts depending on the presence of visible teeth." (Abstract)
  - Note: Exact. Annotation is manual, binary, by one annotator, on 3,500 AffectNet images, and it is observational stratification. Teeth visibility is confounded with emotion category in that subset (Table II), which the paper notes.
- Claim: (introduction.md para starting "We located no study") "None of these, as far as we read them, compares a model with randomised human cue effects"
  - Verdict: supported
  - Quote: "While these models often outperform a supervised baseline like ViT-FER, their performance is partly driven by perceptual shortcuts aligned with human heuristics." (Section V)
  - Note: Looked for any human effect estimate, randomised cue or reverse correlation, and found none. Human work appears only as cited background, and "aligned with human heuristics" is a narrative comparison with no human data. The cue regression uses GPT-4o's own self-reported features and its own valence and arousal outputs.
- Claim: (references.md row 23, Used for) "Teeth visibility and VLM performance"
  - Verdict: supported
  - Quote: "We use these annotations to uncover the impact of that feature on model performance for all models and the generated features of GPT-4o." (Section I)
  - Note: Confirmed. No range in the drafts includes #23.
- Other findings: This is the closest paper in the batch in spirit: it asks which visual cues VLMs rely on, and names cues. So it is a second exception to "sits almost entirely at the first degree", and a reviewer may cite it against the novelty sentence. The sentence still holds, because no human effects are estimated. Reliability caveats if cited further: the GPT-4o prompt's example output pairs "Happiness" with teeth visible "Yes", which could itself seed the teeth-to-happiness pattern; "GPT-4o" is listed as GPT-4o-mini in Table IV; and the model got a different prompt from the others.

### #24 Barrett et al.
- Where read: https://pmc.ncbi.nlm.nih.gov/articles/PMC6640856/ (author manuscript)
- Coverage: partial (first 40,000 characters, through Table 2; plus passages on posed movements in production, posed stimuli and reverse correlation in perception, the perception summary, "Summary and Recommendations" with Tables 8 and 9 in part, and the closing paragraphs). Located by grep. Most of the evidence review and the SOM were not read.
- Metadata: fix: journal confirmed: the page reads "Psychol Sci Public Interest" with this DOI, which expands to the name in the row, so drop "journal to confirm from DOI". Add title "Emotional Expressions Reconsidered: Challenges to Inferring Emotion From Human Facial Movements", and volume and pages 20(1), 1-68 (2019 Jul). The PMC copy is an author manuscript and carries a correction notice (Psychol Sci Public Interest. 2019 Nov 15:1529100619889954): the correction was not seen, so check it. Authors listed on the page include Barrett, Adolphs, Marsella, Martinez and Pollak. Status can move from abstract to partial.
- Claim: (introduction.md para starting "All confirmatory tests were fixed in advance") "Results concern the recognition of acted conventions in these stimuli, not the reading of felt states [24]."
  - Verdict: needs rewording
  - Quote: "The majority of the experiments that study emotion perception ask participants to infer emotion in facial configurations that are posed by actors who are not in an emotional state when the photos were taken or by computer-generated humans who have no actual emotional state." (Perceiving Emotions, posed configurations)
  - Note: The core point is backed: posed stimuli cannot show accuracy, and responses are better read as agreement with common beliefs. Two fixes: "acted" leaves out the rendered faces of A4, which [24] covers separately, and [24] tells readers to replace "emotion recognition" with "perception" or "inference" (Table 8); suggested: "Results concern agreement with conventional labels for acted and rendered faces in these stimuli, not the reading of felt states [24]."
- Claim: (same sentence, the inference that follows from posed stimuli) "not the reading of felt states"
  - Verdict: supported
  - Quote: "As a consequence, it is not possible to assess the accuracy (i.e., validity) of perceivers’ emotional inferences and, correspondingly, data from emotion perception studies cannot be interpreted as support for the validity of common beliefs about emotional expressions." (Perceiving Emotions, posed configurations)
  - Note: Direct support, stated for human perceivers; the draft applies it to models, which is a fair extension since the limit comes from the stimuli. Whether EU-Emotion actors were in an emotional state is not something [24] addresses (a matter for #14).
- Claim: (references.md row 24, "Used for") "Felt-state inference is weak"
  - Verdict: supported
  - Quote: "It is not possible to confidently infer happiness from just a smile, anger from a scowl, or sadness from a frown, as numerous technology tries to do when applying what they mistakenly believe to be the scientific facts." (Summary and Recommendations)
  - Note: Keep in mind the paper's qualifier: expression occurs above chance, but reliability is weak and specificity is lacking. The review centres on six categories, and the authors extend it: "our observations and conclusions generalize to studies of other emotion categories that use similar methods", which covers the EU-Emotion labels beyond the six.
- Other findings:
  - Reverse-correlation data have the same status, which matters for the human target of CFA: "Using a choice-from-array response method with the reverse correlation method is an inductive way to learn people’s beliefs about which facial configurations express an emotion category" (reverse correlation section). So Snoek effects are best described as judgement or belief effects, which fits the Introduction's "effect of each AU on human judgement", and they should not be glossed as cues to felt emotion anywhere in Results.
  - Choice-from-array support for human norms: the paper notes such tasks give support "even when it is applied to completely novel emotion categories with made up expressive cues" (Summary of Scientific Evidence on the Perception of Emotion in Faces), and points to the emotion words and the foils. This is a human-side analogue of the draft's prompt audit and is unused; it also means the EU-Emotion norms, if collected by forced choice, are not a method-free baseline.
  - Terminology: the paper uses "stereotypes", "consensus" and "common beliefs", not "conventions", and asks for "agreement" over "accuracy" (Table 8). The Methods' "acted label" wording is fine. No threat to novelty: the paper predates VLMs and does not test them.

### #27 LibreFace
- Where read: https://arxiv.org/abs/2308.10713
- Coverage: read in full; references skimmed. Tables legible; Figure 4 content known from its caption.
- Metadata: fix. Authors: Di Chang, Yufeng Yin, Zongjian Li, Minh Tran, Mohammad Soleymani. Title: "LibreFace: An Open-Source Toolkit for Deep Facial Expression Analysis". Venue: WACV 2024, not arXiv alone. The paper text shows only arXiv:2308.10713v2; the venue comes from the arXiv comment in r27/abs.html ("Accepted by WACV 2024 Round 1. (Application Track)") and from r28's reference [13] ("Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 8205–8215, January 2024"). Status can move from partial to read in full.
- AU outputs (Table 2). Intensity (R): AU1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26. Occurrence only (D): AU7, 10, 14, 23, 24. All eight draft AUs (1, 2, 4, 9, 6, 12, 25, 26) have intensity output, on a 0 to 5 scale. PCC on DISFA, five-fold: AU1 .63, AU2 .72, AU4 .78, AU9 .62, AU6 .59, AU12 .85, AU25 .94, AU26 .68.
- Claim: (methods.md, RQ1 on natural clips) "AU intensities came from LibreFace [27] ... on the frames models saw"; cue set AU1, 2, 4, 9, 6, 12, 25, 26
  - Verdict: supported
  - Quote: "In Table 2, we provide the AUs that LibreFace can predict." (section 3.1)
  - Note: All eight are marked R in Table 2. Validation is within-dataset cross-validation on spontaneous DISFA video, not acted clips such as EU-Emotion.
- Claim: (methods.md, RQ1 on natural clips) "brow lower (AU4, 9)"
  - Verdict: needs rewording
  - Quote: "AU9 Nose wrinkler R" (Table 2)
  - Note: The paper names AU9 nose wrinkler, not a brow action, so the family label misdescribes one member. Fix: rename it, for example "brow lower and nose wrinkle (AU4, 9)", or state why AU9 is grouped there.
- Claim: (methods.md, RQ1 on natural clips) "Detection is weak for AU15 and AU20, which are outside the four families."
  - Verdict: needs rewording
  - Quote: "LibreFace (Ours) 0.63 0.72 0.78 0.59 0.59 0.62 0.85 0.36 0.50 0.34 0.94 0.68 0.63" (Table 3)
  - Note: Backed for LibreFace intensity on DISFA (AU15 .36, AU20 .34, the two lowest, against mean .63), but these are intensity outputs, not detection. Fix: "Intensity estimation is weak for AU15 and AU20 in LibreFace (PCC .36 and .34 on DISFA [27])". Occurrence F1 on BP4D is not specific to these AUs: AU15 46.0, but AU1 49.9, AU2 47.8, AU23 43.2, and AU20 was not tested.
- Claim: (methods.md, RQ1 on natural clips) "[PENDING: training-data lineage of the two detectors.]"
  - Verdict: needs rewording
  - Quote: "After pre-training on these two datasets, we fine-tune the model on DISFA [31] for AU intensity estimation." (section 4.3)
  - Note: Lineage to fill in: MAE ViT-base teacher pre-trained on EmotioNet (self-supervised, weights from another group), then AffectNet and FFHQ, then fine-tuned on DISFA. Deployed ResNet-18 student pre-trained on AffectNet and FFHQ, fine-tuned on DISFA by feature-wise distillation from the MAE; AU intensity labels come from DISFA only, and the occurrence-only AUs from BP4D. See #28 for suggested text.
- Claim: (methods.md, A4, Validity gate) "V1: at least 70% of AU and identity pairs show detected intensity rising across the three doses (AU22 and AU27 cannot be confirmed by the detector)"
  - Verdict: needs rewording
  - Quote: "We apply the same model we used for AU intensity estimation to BP4D [51] dataset to provide AU detection results for those AUs that are not provided with labels in DISFA [31]." (section 5.1)
  - Note: The detector is not named here, but Table 2 lists only 17 AUs, 12 with intensity, against 25 merged families and 33 columns, so more than AU22 and AU27 lack a readout. At least 8 of 25 families have none, and the 70% rule may be unreachable if the denominator is all columns. Fix: restrict the gate to named detectable AUs and state how many. No validation on rendered faces is reported.
- Other findings:
  - Cue spectra table says LibreFace has 15 cues. Table 2 gives 17 AUs (12 intensity plus 5 occurrence), so 15 is not explained by the paper. The difference should be stated from the repo. I did not open any repo file beyond the three allowed.
  - Which model: "Res-18+FM distill is the final model we deployed in LibreFace". Its mean PCC is .628; the .674 figure belongs to the heavier MAE, which should not be quoted for this tool.
  - No threat to novelty.

### #28 OpenFace 3.0
- Where read: https://arxiv.org/abs/2506.02891
- Coverage: read in full, including appendix; references skimmed. Tables legible.
- Metadata: fix. Authors: Jiewen Hu, Leena Mathur, Paul Pu Liang, Louis-Philippe Morency. Title: "OpenFace 3.0: A Lightweight Multitask System for Comprehensive Facial Behavior Analysis". Year 2025 (arXiv:2506.02891v1, 3 Jun 2025). Venue: arXiv preprint; neither the text nor r28/abs.html shows a peer-reviewed venue, so check before submission. Status can move from partial to read in full.
- AU outputs: the paper never lists which AUs the system outputs. It says only "N action units", trained on DISFA (12 AUs, per its Table I) and BP4D (27 AUs annotated, per Table I). So AU1, 2, 4, 9, 6, 12, 25, 26 cannot be confirmed from the paper. Only AU12 and AU26 are named as measurable, both in the user study. No per-AU result is reported, only overall F1.
- Claim: (methods.md, RQ1 on natural clips) "AU intensities came from ... OpenFace 3.0 [28] on the frames models saw"
  - Verdict: needs rewording
  - Quote: "and pi is the predicted probability of its occurrence" (section III-D)
  - Note: The paper describes AU detection: an occurrence loss and F1 evaluation, with no intensity metric. The word intensity for its output appears only in two user-study tasks ("Measure intensity of AU 26 (mouth drop)."). Fix: "AU intensities came from LibreFace [27], and per-AU detector outputs from OpenFace 3.0 [28]", after confirming from the code what the output is and that all eight AUs are present.
- Claim: (methods.md, RQ1 on natural clips) "Detection is weak for AU15 and AU20"
  - Verdict: needs rewording
  - Quote: "Metrics: The F1 score is used to evaluate the performance of AU detection" (section IV-A2)
  - Note: The paper gives no per-AU figures and never mentions AU15 or AU20, so it offers no support for this claim. Attribute it to [27] only, or to a local check if one exists.
- Claim: (methods.md, RQ1 on natural clips) "[PENDING: training-data lineage of the two detectors.]"
  - Verdict: needs rewording
  - Quote: "DISFA [49] 27 hours video Action Unit Recognition Intensity of 12 action units on a 6-point scale" (Table I)
  - Note: Lineage to fill in: EfficientNet backbone pre-trained on VGGFace2; landmarks 300W and WFLW; AUs DISFA and BP4D; emotion AffectNet; gaze MPII Gaze and Gaze360; face detection by RetinaFace. Suggested text: "LibreFace's intensity outputs were fine-tuned on DISFA after pre-training on EmotioNet, AffectNet and FFHQ [27]. OpenFace 3.0's AU head was trained on DISFA and BP4D, with AffectNet, gaze and landmark sets in a shared multitask model [28]. The two detectors therefore share AU training data, and neither reports training on EU-Emotion."
- Other findings:
  - The two detectors are not independent: both trained on DISFA, BP4D and AffectNet (Table I here, section 5.1 in [27]). Agreement between them is not independent validation.
  - AU9, AU25 and AU26 are not among the 12 AUs that [27] evaluates on BP4D (Table 4), so for those AUs DISFA is the only AU evaluation data shown in [27]. This is an inference across the two papers.
  - Cue spectra table says OpenFace 3.0 has 10 cues. The paper gives no AU list, so 10 cannot be checked from it.
  - Three variants are reported. The prose headline of 60% and 62% F1 is the MTL row; the row with uncertainty weighting has 59 and 59. The paper does not say which is released, so the draft should not quote an F1 without confirming this.
  - No threat to novelty.

### #29 Gilbert et al., FACSHuman
- Where read: https://link.springer.com/content/pdf/10.3758/s13428-021-01559-9.pdf (publisher PDF)
- Coverage: read in full (reference list skimmed). PDF extraction: some broken word spacing, tables usable, figures absent.
- Metadata: ok (text shows Behavior Research Methods (2021) 53:2252-2272, DOI matches; authors Gilbert, Demarchi, Urdapilleta; title "FACSHuman, a software program for creating experimental material by modeling 3D facial expressions"). Status "abstract (validation effect sizes unread)" and the must-read "Why" note are now out of date.
- Claim: (methods.md, RQ1 on randomised cues, Stimuli; references row "Render targets") "built each AU configuration on MakeHuman meshes using FACSHuman targets [29]"
  - Verdict: supported
  - Quote: "All targets used in FACSHuman plugins were created with the Blender software and imported into Makehuman." (p. 2253)
  - Note: FACSHuman is three open-source MakeHuman plugins using blendshape targets. The paper mentions automated image creation, but the headless renderer, 768 pixel size and bald avatars are the draft's own and not in paper.
- Claim: (methods.md, same para, indirect) "Of the 33 columns, 31 map directly to targets, and AU22 and AU27 are approximated."
  - Verdict: needs rewording (not in paper)
  - Quote: "22 Lip Funneler X X(2)" (Table 9)
  - Note: Table 8 backs AU27 ("Use 26"), but Table 9 lists AU22 as usable with an alternative, so the paper gives no reason to call it approximated. If the shipped targets force this, say so as a local finding, and check that no Snoek column is AU8 or AU21, which Table 9 marks as not implemented.
- Other findings: Validation numbers. Exp 1 (43 lay raters, 42 static black-and-white images): 85.5% correct, fear lowest at 73.86%, happiness 79.92% with 18.18% read as neutral. Overall against POFA: p = .44, d = 0.47, power = .65, which is underpowered for showing equivalence. By category FACSHuman beat POFA on four emotions and lost on happiness and surprise (r = .88 to .94). Exp 2 (50 lay raters, single AU against FACS manual): 82.44%, A' = 0.90, d' = 1.97, AU24 38% and AU11 46% on congruent trials. Exp 3 (2 coders, 47 AUs): 63.83% and 85.11%, alpha = 0.59, called moderate; AU6, AU13 and AU25 were coded wrongly by both coders. AU6 wrinkles were corrected later, and the draft should confirm that its targets include the fix. Exp 4 (54 combinations): 68.08% and 81.56%. Limits that matter for A4: "we only used 100% intensity as described in the FACS manual." (p. 2263, Exp 3), whereas A4 samples amplitudes of .25, .5 and 1; AU tests used one Caucasian male model, whereas A4 uses several identities; and the authors say the software "does not dispense with performing an evaluation process for the sets of images or videos produced." (p. 2265). A4 has no human evaluation of its renders. On "linearly": Table 1's note says of FACS intensity "This quantification does not follow a linear curve (Ekman et al., 2002)", which is direct support for flagging the linear amplitude map as an assumption, and could be cited there. No threat to novelty claims.

### #30 Wegrzyn et al.
- Where read: https://pmc.ncbi.nlm.nih.gov/articles/PMC5426715/ (Europe PMC XML)
- Coverage: read in full (references omitted from the file). Figures absent, so Fig 5 and Fig 6 values are known only from the prose.
- Metadata: ok (PLOS ONE 2017, 12(5), e0177239, DOI matches; title "Mapping the emotional face. How individual face parts contribute to successful emotion recognition"). Status "abstract" is now out of date.
- Claim: (methods.md, Predictions table, all rows sourced to [30]) P1 to P4 and P6 "Acted label falls" under a mask
  - Verdict: needs rewording
  - Quote: "Each trial consisted of a face masked by a 6x8 grid of white tiles, starting with one randomly revealed tile and subsequently revealing one additional tile every second, in a fully randomized fashion." (Methods, Design)
  - Note: No region was ever masked out with the rest visible. Tile weights show what was visible on short correct trials, which is diagnosticity and not necessity, so a fall under a mask is an inference. Stimuli were 2 static NimStim identities with open mouths, 7AFC with feedback, 94 students. Add to the table note: "[30] is a reveal study; mask directions are inferred from it".
- Claim: (methods.md, Predictions table, P1) "Mouth mask on Happy: acted label falls"
  - Verdict: needs rewording
  - Quote: "For happiness, this significantly most diagnostic AUs are around the mouth (AU12 and 25, lip corner puller and parting of lips)." (Results, action units)
  - Note: The region is firmly backed, and the fall does not strictly follow. The paper notes correct happy responses after only one or two tiles, so other tiles may suffice. Keep P1, marked as inferred.
- Claim: (methods.md, Predictions table, P2) "Mouth mask on Disgusted: acted label falls"
  - Verdict: needs rewording
  - Quote: "For disgust, there is also a significant effect for the compound action units around the mouth (including raising the lip and plucking it)." (Results, action units)
  - Note: Region backed, fall inferred, as for P1.
- Claim: (methods.md, Predictions table, P3) "Eye mask on Afraid: acted label falls"
  - Verdict: needs rewording
  - Quote: "For fear, surprise, anger and sadness the regions around the eyes have the highest weights, with the lid raiser (exposing the sclera of the eyes; AU5) significantly most important for fear and the lid tightener (AU7) significantly most important for anger." (Results, action units)
  - Note: Region backed through AU5, fall inferred, and the eye mask must cover the upper lid. The paper's upper-half result holds only "for at least one of the face models".
- Claim: (methods.md, Predictions table, P4) "Eye mask on Sad: acted label falls"
  - Verdict: needs rewording
  - Quote: "There is a tendency to focus more on the brows in sadness (AU1+4) and both eyes and mouth contribute significantly to the recognition of surprise." (Results, action units)
  - Note: Weakest of the set. The specific Sad cue is a non-significant tendency at the brows, and the draft has a separate brow mask, so an eye mask that leaves brows visible does not test it. Sad stimuli were also flagged as poorly recognised; note the brow caveat in P4, or use an eyes-plus-brows mask.
- Claim: (methods.md, Predictions table, P6) "Mouth minus eye mask, Happy and Disgusted against Afraid: more negative on Happy and Disgusted"
  - Verdict: needs rewording
  - Quote: "However, it should be noted that a similar gradient (happy, disgusted, surprised, angry, fearful, sad) can be found in the global metrics of correct responses or number of revealed tiles." (Discussion)
  - Note: The eye-to-mouth grouping is the paper's clearest result, but the mask form is inferred and the authors warn the gradient tracks overall recognisability. "Same 8 actors" is not from this paper (see #14).
- Other findings: (1) The authors limit their own stimuli: "This certainly concerns the expressions of sadness and anger, which were poorly recognized, varied most strongly between the male and female models and whose action unit labelling deviated somewhat from the reference guide [ 7 , 21 ]." (2) P5 (brow mask on Angry) is sourced to [32], but here AU7 at the eye is most important for anger and the sentence names no brow effect, so this source pulls against P5 as a single-region test. (3) The paper suggests its paradigm as a benchmark for computer vision and comparing similarity spaces across modalities. This is a passing proposal with no model tested, so no threat to novelty.

### #33 Carbon
- Where read: https://pmc.ncbi.nlm.nih.gov/articles/PMC7545827/ (Europe PMC XML)
- Coverage: read in full (references omitted from the file). Figure 5, the confusion matrices, is absent, so only the percentages quoted in prose are available. No masked-to-Neutral percentage is in the text.
- Metadata: ok (Frontiers in Psychology 2020, 11, 566886, DOI matches; single author; title "Wearing Face Masks Strongly Confuses Counterparts in Reading Emotions"). Status "abstract" is now out of date.
- Claim: (methods.md, Predictions table, P7) "Mouth mask on Happy, Sad and Angry, Neutral family: Neutral rises"
  - Verdict: supported
  - Quote: "For instance, all emotional states with the exception of fearful were repeatedly confused with a neutral state." (Results)
  - Note: The Discussion and abstract name happy, sad and angry. This covers [33] only; [34] and [35] are outside this batch. Numbers are in Fig 5 only, and the confusions are descriptive, with no inferential test.
- Claim: (methods.md, Predictions table, P8) "Mouth mask on Disgusted, Angry family: Angry rises"
  - Verdict: supported
  - Quote: "Most drastically was the misinterpretation of disgusted as angry , which showed up in nearly 38% of the cases, although such a confusion did only happen in 2% of the cases where no face mask was used." (Results)
  - Note: Exact confusion confirmed: disgusted read as angry, nearly 38% masked against 2% unmasked. Also descriptive, with no inferential test.
- Claim: (methods.md, Predictions table, P7 and P8) the contrast is a "Mouth mask"
  - Verdict: needs rewording
  - Quote: "to shield the mouth and the nose, cover about 60–70% of the area of the face that is relevant for emotional expression" (Introduction)
  - Note: Carbon pasted a visible community mask over nose and mouth on static photos, with a 6AFC list that included neutral. The draft's mouth mask has neutral, first-frame or blur fills and may leave the nose visible, which matters most for P8 because a disgust nose wrinkle stays visible. A neutral fill also pastes a neutral mouth, a separate route to Neutral rising. Say in the table note that these are analogues of a nose-and-mouth occluder.
- Other findings: (1) Carbon calls the fear evidence contested: "For other emotional states than happy faces, however, there are quite contradictory results to be found in the literature, e.g., for fear detection". He lists Wegrzyn et al. 2017 under "but see" for eye dominance in fear, although #30's own text finds eyes for fear. This bears on P3. (2) Carbon's fear and neutral accuracy held up under the mask, with ceiling effects noted, which fits P3's eye account only indirectly. (3) Sad was already read as disgusted on 20.3% of unmasked trials, so Sad confusions have a high baseline. No threat to novelty claims.

### #34 Grundmann et al.
- Where read: https://pmc.ncbi.nlm.nih.gov/articles/PMC8064590/ (Europe PMC XML)
- Coverage: read in full, including the appended peer-review history. The reference list is omitted from the extraction. Figures (including the percentages shown under Fig 1) and the OSF supplement are not in the text.
- Metadata: ok. Row is right on venue, year and DOI (PLOS ONE 16(4), e0249792). Authors in file: Grundmann, Epstude, Scheibe. Title available if wanted: "Face masks reduce emotion-recognition accuracy and perceived closeness".
- Claim: (methods.md, RQ2 Predictions table, P7) "Mouth mask on Happy, Sad and Angry, Neutral family | Neutral rises | [33-35]"
  - Verdict: wrong
  - Quote: "Overall emotion-recognition accuracy declined from 69.9% for unmasked to 48.9% for masked target faces (see also Fig 2 )." (Results)
  - Note: The paper gives no Neutral-response or confusion result for any emotion. Happy: nothing. Sad: nothing. Angry: nothing. No analysis has emotion as a factor, so there is not even a per-emotion accuracy figure; emotions appear only collapsed into a negative versus non-negative valence code for social judgements. Remove [34] from P7. It supports only overall accuracy loss, which is at most general context for P1.
- Claim: (references.md, row 34, Used for) "P7"
  - Verdict: wrong
  - Quote: "We coded correct choices as 1 and incorrect choices as 0 ( n missing = 2)." (Materials and procedure)
  - Note: The analysed outcome is binary correctness, so which wrong label was chosen is never reported. Neutral was a response option and raw data are on OSF, so a confusion matrix could be computed from them, but the paper does not do so. Change Used for to overall mask accuracy loss, or drop the reference.
- Claim: (methods.md, RQ2 Predictions) "Almost none of the human baselines were measured on EU-Emotion, and none on these edits."
  - Verdict: supported
  - Quote: "Stimuli are portrait photographs of German adults belonging to three age segments (young: 19–31 years; middle-aged: 39–55 years; old: 69–80 years)." (Participants)
  - Note: Static FACES photographs with a prototypical face mask, between subjects, 191 adults, nine response options (six targets plus three distractors). Not EU-Emotion and not A2's fills.
- Other findings: No threat to novelty. Task caveat: the authors say option sets shape mask accuracy, attributing the gap from Carbon (69.9 and 48.9% against 89.5 and about 71.7%) possibly to their distractors, which bears on how far a confusion pattern from a six-option task transfers. Exploratory result: masks weakened the fall in trust, liking and closeness for negative expressions, which the authors explain by ambiguity and a positivity bias, not by faces reading as Neutral. It cannot stand in for P7. Design point for all P7 sources: humans saw an opaque mask, whereas A2's primary fill is the actor's own neutral clip, so a Neutral rise under that fill may reflect neutral pixels being shown rather than information being removed; the first-frame and blur fills in the decision rule partly guard against this.

### #35 Pazhoohi et al.
- Where read: https://pmc.ncbi.nlm.nih.gov/articles/PMC8483373/ (Europe PMC XML)
- Coverage: read in full. The reference list is omitted from the extraction. S1 File (S1-S3 Tables of response frequencies) and figures are not in the text.
- Metadata: ok. Row is right on venue, year and DOI (PLOS ONE 16(9), e0257740). Authors in file: Pazhoohi, Forby, Kingstone. Title available if wanted: "Facial masks affect emotion recognition in the general population and individuals with autistic traits".
- Claim: (methods.md, RQ2 Predictions table, P7) "Mouth mask on Happy, Sad and Angry, Neutral family | Neutral rises | [33-35]"
  - Verdict: needs rewording
  - Quote: "Facial masks also create confusion, in that disgust can be mistaken for anger, while happy, sad, and anger can be mistaken for neutral emotional expression [ 22 ]." (Introduction)
  - Note: This is the only Neutral statement in the text, and it is a citation of Carbon, which is #33, so it is not independent evidence. The paper's own text reports no Neutral-response result for Happy, Sad or Angry, and its accuracy models have no emotion factor. Cite [33] alone for P7, unless S1 Table is checked and shows the rise for all three emotions, in which case cite it as that table with the numbers.
- Claim: (references.md, row 35, Used for) "P7"
  - Verdict: needs rewording
  - Quote: "see S1 Table in S1 File for the frequency and percentage of responses for each facial expression" (Experiment 1, Results)
  - Note: The supplement is the only place this paper's own confusion data could be, and it is not in the text file. Until read, Used for should say "restates Carbon; own response frequencies in S1 Table unread". Its main-text contribution is accuracy, confidence and perceived intensity falling under masks.
- Claim: (methods.md, RQ2 Predictions) "Almost none of the human baselines were measured on EU-Emotion, and none on these edits."
  - Verdict: supported
  - Quote: "Another set of 96 stimuli were created by superimposing a facial mask on the original images (see Fig 1 for an example)." (Experiment 1, Stimuli and procedure)
  - Note: Static FACES photographs of young adults at peak intensity with an overlaid mask, within subjects, six response options including neutral, in three online experiments (420, 199 and 142 participants).
- Other findings: One finding complicates P7. The paper reports that masks disrupt recognition of neutral faces too, which departs from Carbon, so masks are not shown to act as a simple pull towards Neutral: "our studies showed that facial masks disrupt recognition of all investigated emotions." The text gives no per-emotion accuracy statistics behind that sentence (they would be in S1). Masked expressions were rated as less intense, which fits the spirit of P7 but is an intensity rating, not a Neutral response, and the intensity question was not asked after a Neutral choice. Face-gender and AQ-10 effects are outside the draft's claims. No threat to novelty.

### #38 Richoz et al.
- Where read: https://pmc.ncbi.nlm.nih.gov/articles/PMC10790674/ (Europe PMC XML)
- Coverage: read in full. The reference list is omitted from the extraction. Figures and Supplement 1 (confusion plots) are not in the text; Tables 1 and 2 are legible.
- Metadata: ok. Row is right on venue, year and DOI (Journal of Vision 24(1), 7). Title available if wanted: "Recognizing facial expressions of emotion amid noise: A dynamic advantage". The paper itself gives 70 observers in the abstract and 71 in the body, so quote neither without checking.
- Claim: (methods.md, RQ2 Predictions table, P12) "Mean of shuffles on Happy and Surprised | Acted label falls | [38,39]"
  - Verdict: wrong
  - Quote: "To this aim, we asked 70 healthy young adults to perform static and dynamic facial expression recognition of the six basic expressions while parametrically and randomly varying the low-level normalized phase and contrast signal (0%–100%) of the faces." (Abstract)
  - Note: There is no shuffled, scrambled or reversed condition. The contrasts are dynamic against static (the final frame repeated 30 times), crossed with phase coherence, in young adults only (M = 20.9 years) with central presentation; age and visual field are not manipulated. Remove [38] from P12 as a source for a shuffle effect, or reword the row to say it is an inference from a dynamic advantage, as in the next block.
- Claim: (methods.md, P12, implicit rationale) dynamic information helps Happy and Surprised at full signal
  - Verdict: needs rewording
  - Quote: "However, when examining each expression independently, our findings revealed a dynamic advantage with full signal strength for happiness and surprise." (Discussion)
  - Note: This is the only link to P12, and the labels do match. But the comparison is movie against repeated apex, not ordered against shuffled frames, and at 100% signal the gain is small: 2.23 [0.6, 4.39] for happiness and 4.09 [1.48, 6.72] for surprise (Table 2). A shuffle keeps all frames and frame-to-frame change, so a dynamic advantage over a still does not predict a shuffle cost.
- Claim: (methods.md, RQ2 Predictions table, P13) "Mean of shuffles on 48 Low Intensity items | Acted label falls | [38,40,41]"
  - Verdict: wrong
  - Quote: "Each stimulus was then characterized by its level of signal with 0% corresponding to completely de-phased images and 100% to natural images (100% of phase coherence)." (Introduction)
  - Note: Richoz varied phase noise on the image, not expression intensity, and tested no shuffle. Low signal is not low intensity: the stimuli are full expressions from Gold et al., and the paper mentions intensity only in discussing other work. Remove [38] from P13. Its dynamic-over-static advantage at low signal may be cited only as a stated analogy, and only for phase-degraded faces (overall peak at 18.6% signal, gone from 37.36%).
- Claim: (references.md, row 38, Used for) "P12, P13"
  - Verdict: wrong
  - Quote: "Seventy-one healthy young adults performed FER tasks of the six basic expressions (i.e., anger, disgust, fear, happiness, sadness, surprise) in static and dynamic conditions." (Introduction)
  - Note: Replace with "dynamic against static advantage under phase noise; background to P12 only". The row's abstract status explains the error: the abstract names happiness and surprise and "less signal", which can read like the predictions but is not.
- Claim: (introduction.md para starting "Here we ask") "changes to frame order, and ask whether each model's judgement moves as stated human predictions say it should"
  - Verdict: needs rewording
  - Quote: "Gold et al. (2013) further evidenced identical recognition scores for static, dynamic, and even shuffled and reversed expressions in which movie frames were randomly presented or temporally reversed." (Introduction)
  - Note: For this reference there is no frame-order prediction, and the one statement it makes about shuffling points against P12 and P13: no recognition cost in young adults. That is Richoz describing Gold et al. 2013, an overall result, so it needs a first-hand read. Gold et al. is not in references.md; read it and disclose it, since it is the closest human test of the edit. Whether the sentence survives depends on [39-41], which are outside this batch.
- Claim: (methods.md, RQ2 Predictions) "Almost none of the human baselines were measured on EU-Emotion, and none on these edits."
  - Verdict: supported
  - Quote: "We used a set of static and dynamic stimuli created by Gold et al. (2013) ." (Stimuli)
  - Note: Eight actors, six basic expressions, grayscale, 1 s at 30 frames per second. Not EU-Emotion and not a shuffle. A2 sends 8 frames per clip, far sparser than these 30-frame movies.
- Other findings: Fear runs the other way: a static advantage from 22.95% signal onward, -6.75 [-9.59, -3.81] at 100%. Anger, disgust and sadness show no full-signal difference. So if the 48 Low Intensity items include Afraid, this paper gives no reason to expect a uniform fall, and it may suggest the opposite for those items. methods.md does not say which labels the 48 items cover, while it says all predictions are BASIC ONLY; Richoz covers only the six basic expressions, so confirm the items fit that tag. Richoz names Ambadar et al. and Bould et al. for dynamic gains on subtle expressions, which is the proper route for P13, to be checked in #39 to #41. Its account of early confusion between fear and surprise and between anger and disgust, resolved by late signals, agrees with the use of #42 for P14 and P15, but it is a restatement of Jack et al., not a test. No threat to novelty.

### #42 Jack et al. 2014
- Where read: https://eprints.gla.ac.uk/88928/ (University of Glasgow accepted copy)
- Coverage: read in full (accepted main text; references skimmed). The supplement is not in the file. Figures are absent, and the Figure 3 labels are garbled, so per-cell confusion values could not be seen.
- Metadata: ok. The text shows Jack, Garrod and Schyns, Current Biology 24(2), 187-192 (2014), and the DOI (printed with a stray space in the text). Optional: add the title "Dynamic Facial Expressions of Emotion Transmit an Evolving Hierarchy of Signals over Time".
- Claim: (methods.md RQ2 Predictions, row P14) "First minus last third, Surprised family on Afraid and Afraid family on Surprised ... Confusable label higher on the early third" [42]
  - Verdict: needs rewording
  - Quote: "Surprise versus fear: early confusion arises from the expected common transmission of Upper Lid Raiser and Jaw Drop (t2), then Upper Lid Raiser (t3–t5)." (Figure 3 caption)
  - Note: The early/late direction matches, but the confusions belong to Bayesian classifiers, not human responses. Those classifiers see all AUs "up until that time point", so late means cumulative; a stand-alone last third has no counterpart in the paper. Correction: mark the source in the row as "[42], ideal-observer classifier on Western observers' expectation models; cumulative signal, not a truncated clip".
- Claim: (methods.md RQ2 Predictions, row P15) "First minus last third, Disgusted family on Angry and Angry family on Disgusted ... Confusable label higher on the early third" [42]
  - Verdict: needs rewording
  - Quote: "Disgust versus anger: here, early confusion arises on the basis of Nose Wrinkler (t2–t5) then Lip Funneler (t6)." (Figure 3 caption)
  - Note: Same limits and same correction as P14. Discrimination arrives at t7 here, against t6 for surprise/fear, and the ten bins are expected AU peak latencies, so nothing shows that a first or last third of an EU-Emotion clip maps onto them.
- Claim: (methods.md, under the Predictions table) "P14 and P15 are analogues of a study on synthetic faces, not replications."
  - Verdict: needs rewording
  - Quote: "The dynamic AUs are then combined to produce a 3D facial animation" (Figure 1 caption)
  - Note: Synthetic is correct: stimuli are random 3D animations from the Generative Face Grammar. "Analogues of a study" hides that the temporal confusions are computed by an "idealized observer" classifier, and the main text reports no human confusion over time (supplement unread). Correction: "P14 and P15 are analogues, not replications, of an ideal-observer result: Bayesian classifiers applied to AU models reverse-correlated from 60 Western observers' categorisations of synthetic faces."
- Claim: (introduction.md para starting "Here we ask") "ask whether each model's judgement moves as stated human predictions say it should"
  - Verdict: needs rewording
  - Quote: "we analyzed how this signaling supports emotion categorization for an idealized observer" (Results, Classifying the Signaling Dynamics)
  - Note: For P14 and P15 the prediction is derived from human expectation models through a classifier; it is not a measured human response. Correction: "as stated predictions from human studies say it should", or carry the Methods correction above.
- Claim: (methods.md RQ2 Predictions) "Almost none of the human baselines were measured on EU-Emotion, and none on these edits." (as applied to P14, P15; which observers)
  - Verdict: supported
  - Quote: "Using perceptual expectation modeling, we derived the dynamic signaling of the six classic facial expressions of emotion—happy, surprise, fear, disgust, anger, and sad—in 60 Western white Caucasian observers." (Discussion)
  - Note: Observers are Western only; I looked for any East Asian group in the main text and found none. The draft never states this for P14 and P15, and it should. Faces were "same-race" synthetic faces, with no EU-Emotion clips and no truncation.
- Claim: (references.md row 42) "partial (main text read 2026-11-07; supplement unread)", used for P14, P15
  - Verdict: supported
  - Quote: "two tables, and one movie and can be found with this article online" (Supplemental Information)
  - Note: Tables S1 and S2 and the Supplemental Experimental Procedures hold the task, classifier and latency details, and remain unread.
- Other findings: (1) #42 gives 41 AUs ("from a total of 41"), while #19 gives 42, and #19 states that its WE data were described in its ref 11, which its Table 1 labels Jack et al. 2014. The draft's 42 comes from #19, so it stands, but do not quote 42 for #42. (2) #42 is a verified source for the reverse-correlation sentence that the draft gives to [18,19], and its ref 4 is the 2012 PNAS paper (see #19). (3) #42 names Upper Lip Raiser Left as the late disgust/anger discriminator, whereas Yan et al. list Upper Lip Raiser as shared and confusing (see #- Yan et al.). (4) These are perceptual expectations, not recordings of produced expressions, so applying them to acted clips is itself an assumption. No threat to novelty.

### #45 Baron-Cohen et al. 1997
- Where read: https://docs.autismresearchcentre.com/papers/1997_BC_%20Is%20there%20a%20language_eyes_Vis%20Cog.pdf (authors' centre)
- Coverage: read in full (references skimmed). This is a PDF extraction with word-spacing damage. Table 2 is unreliable: the printed basic mouth mean (76.5) does not match its listed cells (they average 84.0), and the text says six basic and seven complex faces while the table lists seven and six. Experiment 2 cell values should be checked on the PDF before use.
- Metadata: ok on venue and year. The text shows Baron-Cohen, Wheelwright and Jolliffe, Visual Cognition 4(3), 311-331 (1997). The DOI is not printed in the text, so it is unchecked. Fix: the Status can change from "abstract" to read in full. Optional: add the title "Is There a "Language of the Eyes"? Evidence from Normal Adults, and Adults with Autism or Asperger Syndrome".
- Claim: (methods.md, under the Predictions table) "Eye-only views have no directional prediction, because accounts conflict [31,45,46]" (what #45 says on basic emotions)
  - Verdict: needs rewording
  - Quote: "This suggests that for the basic emotions, the whole face is most informative." (Experiment 1, Results)
  - Note: #45 does not itself conflict on basic emotions: the whole face beat the eyes, marginally (z = -1.94, p = .05), and eyes did not differ from mouth (p = .19). Alone it points to a fall under eye-only views, and that is the scope the BASIC ONLY tag covers. Whether [31,46] conflict with it must come from their own checks. Correction: "because sources differ by label type and stimuli" (one poser per experiment, static photographs, two-option choice).
- Claim: (methods.md, same sentence) "accounts conflict" (what #45 says on complex states and on earlier work)
  - Verdict: needs rewording
  - Quote: "(3) for the complex mental states, seeing the eyes alone produced significantly better performance than seeing the mouth alone, and was as informative as the whole face." (Abstract)
  - Note: The real contrast in #45 is by label type: basic favours the whole face, complex shows eyes equal to the whole face (p = .36) and better than the mouth (p = .02). Cite it for that interaction, not as one side of a general dispute.
- Claim: (methods.md, same sentence; earlier literature as #45 describes it) "accounts conflict"
  - Verdict: supported
  - Quote: "From previous work, this question remains inconclusive." (Introduction, before Experiment 1)
  - Note: #45 says this of eyes versus mouth before 1997 (Dunlap, Frois-Wittman, Coleman, Hanawalt), then offers its own result as a refinement. This supports "conflict" only as a description of earlier eyes-versus-mouth work.
- Claim: (references.md row 45) Used for "Eye-only views"
  - Verdict: supported
  - Quote: "Instead, the eyes and thewhole face were both equally informative" (Experiment 1, Results; spacing damage is in the file)
  - Note: The use is right. Stimuli were black and white photographs of one actress (Experiment 1, N = 50) or one actor (Experiment 2, N = 17), with two same-valence, same-class options.
- Other findings: (1) #45 directly tests masks on complex states, so it could also sit beside [46,47] in "masks on complex labels are descriptive". (2) It supports the BASIC ONLY split: the mouth did worse on complex than on basic states in Experiment 2. (3) Item-level results vary widely by emotion in Experiment 1 (Afraid vs Angry, mouth 24/50 and face 45/50; Disgust vs Sad, mouth 49/50 and eyes 41/50), but each rests on one posed image, so they do not bear strongly on P2 or P3. (4) Experiment 2 pools eyes above mouth for "all mental states", while Experiment 1 finds no eyes-mouth difference on basic emotions. No threat to novelty.

### #48 Reinl and Bartels
- Where read: https://pmc.ncbi.nlm.nih.gov/articles/PMC4523710/ (Europe PMC XML)
- Coverage: read in full (references omitted from the file). Figures absent.
- Metadata: ok (Frontiers in Psychology 2015, 6, 1107, DOI matches; title "Perception of temporal asymmetries in dynamic facial expressions"). Status "abstract" is now out of date.
- Claim: (methods.md, text after Predictions table) "reversal describes the model's temporal code [48]"
  - Verdict: needs rewording
  - Quote: "The results on the main effects of timeline show that temporally reversed facial dynamics appeared more artificial, less fearful and, in the case of increasing fear, less convincing to our subjects than the natural timeline counterparts." (Discussion, Effects of Timeline)
  - Note: The paper has no model and no "temporal code". It is a human rating study of fear only: 11 actors, about 600 ms clips of rise or fall, rated for intensity (28 raters) and for artificialness and convincingness (19), with no categorisation task. Read as an attribution the clause would be wrong, so reword it as a design choice: "Reversal has no confirmatory prediction: humans rate reversed fear clips as less intense and more artificial [48], but that is not a label prediction, so reversal is reported descriptively."
- Other findings: (1) Reversal changes emotion direction as well as timeline. The paper's 2-by-2 design exists to separate them, and it faults earlier work: "However, in contrast to us, they did not control for the change in emotion direction that results from timeline reversal." A plain reversal of a whole EU-Emotion clip does not separate them, so any model change is not a pure temporal-code reading. (2) A human direction does exist, at least for fear, and the paper reports second-hand from Cunningham and Wallraven (2009) that forward clips were recognised slightly better than reversed ones, so "no prediction" is a choice and not a gap in the literature. (3) Convincingness showed a timeline effect only for increasing fear. No threat to novelty claims.

### #- Yan et al.
- Where read: https://pmc.ncbi.nlm.nih.gov/articles/PMC12207432/ (Europe PMC XML). Not in the 49 references
- Coverage: read in full (whole main text). The reference list is omitted from the file, and the SI Appendix is not in it, including the synergy and per-emotion results.
- Metadata: not in the 49 references, so there is no row to check, and no number is assigned. The text shows Yan, Zhan, Garrod, Ince, Jack and Schyns, "The brain computes dynamic facial movements for emotion categorization using a third pathway", PNAS 122(25), e2423560122 (2025), DOI 10.1073/pnas.2423560122, licence CC BY-NC-ND. A linked commentary is listed: "Face (e)motion and the third visual pathway", 10.1073/pnas.2513881122.
- Claim: (methods.md row P15, bearing on) "Disgusted family on Angry and Angry family on Disgusted"
  - Verdict: supported
  - Quote: "Disgust and anger are behaviorally confused because they share Nose Wrinkler and Upper Lip Raiser AUs." (Fig. 5 caption)
  - Note: This gives the human behavioural confusion that #42 lacks, on synthetic faces, in 10 White Western participants viewing their own reverse-correlated models. It supports the confusion, not the early-versus-late claim.
- Claim: (methods.md rows P14, P15, bearing on) "Confusable label higher on the early third"
  - Verdict: needs rewording
  - Quote: "However, ambiguous emotions (e.g., the similarity between disgust and anger due to Nose Wrinkler, AU9) require later integration of supplementary AUs in bSTS/STG for disambiguation." (Discussion)
  - Note: "Later" here is neural latency, not when an AU occurs in the stimulus, so it cannot be cited for a first-third against last-third contrast. For P14 the main text reports no surprise/fear confusion at all; I looked and found none. It also disagrees with #42 on the cue: Upper Lip Raiser is shared and confusing here, but its left form is the late discriminator in #42.
- Claim: (introduction.md para starting "Human data of that kind exist") "In reverse-correlation studies, observers judge faces whose facial action units (AUs) are set at random"
  - Verdict: supported
  - Quote: "In each of 2,400 trials, a generative model synthesized a facial expression animation by randomly selecting AUs, each animated with randomly sampled temporal parameters–onset, acceleration, peak intensity, peak latency, deceleration, and offset." (Fig. 1 caption)
  - Note: It describes this as reverse correlating AUs against choices, and it is a recent verified example. #42 is the more direct citation; this one is not needed.
- Claim: (introduction.md para starting "The second problem") "Facial movements co-occur on natural faces"; "one has to break the natural correlations between cues"
  - Verdict: supported
  - Quote: "While real videos may offer greater ecological validity, they lack the experimental control necessary to disentangle overlapping AU dynamics." (Discussion)
  - Note: This is a statement of the same rationale for human work, not an identifiability result.
- Claim: (introduction.md para starting "We located no study") "We located no study comparing a vision-language model's response to named facial cues with effects estimated from human observers under randomised cue presentation."
  - Verdict: supported
  - Quote: "A central challenge for computational cognitive neuroimaging (and AI) lies in ensuring that brain processing and artificial models align—both should process the same features using similar computational algorithms to generate similar behaviors ( 74 , 75 )." (Discussion)
  - Note: No threat: this is a human MEG study with no model comparison, and AI appears only in this forward-looking remark. The remark shares the framing of Schyns's degrees of equivalence [4], though refs 74 and 75 cannot be resolved from this file.
- Other findings: Recommendation: the draft may cite it once but does not have to; no claim depends on it, and the novelty claim needs no mention of it. The best place is a sentence after "P14 and P15 are analogues...", as human evidence that disgust and anger are confused on synthetic AU faces. Keep it out of the P15 Sources column, because A2 is frozen and that column records the prediction's fixed sources. Do not use it for timing or for P14, and read the SI first. Its random-AU behavioural step and deposited data are a further human source, but with 10 participants and individual models it does not change the draft's choice of Snoek data.

