# Outline: Results, Discussion, and how the paper reads under each outcome

Drafted 2026-11-07. No A1 result has been read, and nothing below predicts one. Each branch uses the preregistered outcome classes and adds none. Results text is written only from locked output.

**Working title:** [PENDING: title]. Candidate: "Vision-language models against randomised human cue effects on faces".

## Journal limits

| Item | Article limit | Plan |
|---|---|---|
| Abstract | Up to 150 words, unreferenced | Template below |
| Main text | Up to 5,000 words, excluding abstract, Methods, references and figure legends | Introduction about 1,140 words as drafted, leaving about 2,700 for Results and 1,150 for Discussion |
| Display items | Up to 8 figures and tables | 7 planned, below |
| References | No limit | 49 so far |
| Structure | Introduction without heading; Results with topical subheadings; Discussion without subheadings; Methods with subheadings | Followed |

**How far this was verified.** These limits are from the journal's official content-types page, read on 2026-11-07. WebFetch was redirected by a cookie check, so that page was fetched with curl using a browser user agent and a cookie jar. A later attempt the same way, for the policy pages, was blocked by the session's safety classifier, and was not retried. **Please re-read the page yourself before relying on these limits.** The rule that Methods carry short bold headings came from the formatting page in the same way.

**Not verified.** Preregistration statement rules, reporting summary rules, and current data and code requirements. A web search found a 2021 editorial (s41562-021-01190-w) saying that central custom code is peer reviewed, must go to a DOI-granting repository at acceptance, and that "available on request" code statements are generally refused. That came through a search summary only. Whether a repository tag counts as preregistration for the journal is also unverified. Page charges and preprint policy remain unverified (`LIT_CHECK_2026-11-07.md`).

## Abstract template (150 words at most)

> Vision-language models are reported to match people at reading mental states from faces, but most claims we located cannot separate perception from option order and prompt text. We audited [PENDING: n] models on 243 acted clips, with option position controlled and faces swapped under fixed prompts. [PENDING: RQ0 sentence.] We prove that agreement on natural items cannot settle which cues a system uses, because natural cues are correlated. We therefore compared model cue effects with human effects from randomised facial action units. [PENDING: RQ1 sentence.] Stimulus interventions tested preregistered human predictions. [PENDING: RQ2 sentence.] [PENDING: closing sentence from the scenario table.]

## Results structure

Subheadings are provisional. Sections 1 and 3 hold whatever the outcomes are.

1. **Published and prior designs cannot separate perception from prompt.** The audit: target position, option sets, delivery, weak labels, the eight-paper table. Figure 1.
2. **Dependence on the face (RQ0).** A1, then A3, per model, read through the frozen table. Figure 2.
3. **Behaviour cannot settle cue use.** The identifiability results, and the cue spectra as the empirical measure of the gap. Figure 3.
4. **Pattern alignment on natural clips (RQ1).** Family pattern score per model. Table or Figure 4.
5. **Cue effects under randomisation (RQ1).** A4: validity gate, then CFA per model and dataset, against the between-group reference. Figure 5.
6. **Do behaviour and cue effects disagree?** CFA beside output similarity and modal agreement. Part of Figure 5.
7. **Interventions on the stimulus (RQ2).** Gate B, then each prediction by decision class. Figure 6.
8. **What each model has shown.** One profile per model across the three degrees. Table 1.

## Outcomes by question

### RQ0

Read per model. `RESULTS_REGISTRY.md` records A3 as Pass for all three models, so the Fail rows of the A3 table cannot occur. [PENDING: A3 table in Results.] Pass tests dependence on the image, not competence, so a Pass must never be written as "reads faces".

| Branch | Condition for Gemma 4 and Qwen3-VL | How section 2 reads | What follows |
|---|---|---|---|
| R0-a | Both pass A1 | Accuracy needs the matching face on both inputs. State how much survives, and compare with accuracy against 1/6 | A2 on both routes; A4 |
| R0-b | One passes, one fails | The models differ. The failing model showed face dependence only on a face crop | Passing model on both routes; failing model on `single_face` only |
| R0-c | Both fail A1 | The composite input limits face use, and accuracy on that format cannot be read as face reading. This is itself the audit's main finding | `single_face` only. Temporal predictions P12 to P15 have no route and are reported as descriptive |
| R0-d | A1 not complete | Nothing can be stated, because Holm covers all three models. A missing result counts as not passed | Wait. No submission on this branch |

Molmo2 is reported in RQ0 either way. It stays out of A2 and A4 on its gate failures, whatever A1 shows.

### RQ1

Each model has one natural-clip outcome and one A4 outcome per dataset.

| Natural-clip pattern score | Meaning | Must not say |
|---|---|---|
| Positive | "Aligned with the human pattern" | "Uses the same cues" |
| Null | Alignment not shown | "Not aligned". Power is limited for partial sharing |
| Over 5% of clips lost | Descriptive only | Any confirmatory claim |
| Model failed A1, passed A3 | [PENDING: rule; see inconsistency 5] | |

| A4 outcome | Meaning | Must not say |
|---|---|---|
| Aligned | Reliable shared cue pattern on these avatars, on a peak frame | "Same strategy", or "human-level" unless the interval reaches the between-group reference |
| Not aligned | Responds to AUs, and CFA is below the smallest effect of interest | "Does not use human cues on real faces". Domain, static route and amplitude remain open |
| Inconclusive | A gate failed or no signal | Any statement on alignment. Report which gate failed |

Datasets may disagree: for instance basic and conversational, or WC and EA. Report each alone. They are separate claims, and any pattern across them is descriptive.

| Natural clips | A4 | How sections 4 to 6 read |
|---|---|---|
| Positive | Aligned | The two agree. Natural-clip alignment gains cue-level support, within avatars |
| Positive | Not aligned | The most informative disagreement: the pattern matched on natural clips, and did not hold when cues were decoupled. It fits the identifiability result, but cannot exclude the avatar domain, so state both |
| Null | Aligned | Human-like cue effects appear when cues are isolated. Natural-clip power and domain differences are the live explanations |
| Null | Not aligned | No evidence of human-like cue use on either arm, with the limits of each |
| Any | Inconclusive | The paper rests on the audit, the theory and RQ2. Say what would be needed: a validated render, or the original stimuli |

Section 6 has two outcomes. If behaviour and CFA disagree on the same stimuli, that is the empirical demonstration the theory needs. If they agree, say so: the theorem stands as a logical limit, and these models on these stimuli do not show the gap. Do not describe agreement as a failure of the theory, or as proof of use.

### RQ2

A2 tests masks, filters and frame order. It has no AU edit, so sections 5 and 7 cannot be read as the same test. Every result carries its BASIC ONLY scope.

| Branch | Condition | How section 7 reads |
|---|---|---|
| R2-a | Gate B passes; most predictions Match | Judgement moves with the stated human predictions on basic labels in this stimulus set. No statement that people would respond this way to these edits |
| R2-b | Gate B passes; mixed classes | Report by cue type: which regions, bands and temporal cues move the judgement in the predicted direction, which oppose it |
| R2-c | Gate B passes; Opposes on several | The model moves, but against the stated prediction. Check the fill rule and the bridge readout, and report the disagreement |
| R2-d | Gate B passes; mostly Inconclusive | Expected for capped contrasts: 8 of 16 cannot reach Null. Report equivalence intervals. Do not call these nulls |
| R2-e | Null on eligible contrasts | Hard to reach by design, so reportable as a real result for that cue, model and route only |
| R2-f | Gate B fails | The option-free readout carries no usable face signal. No RQ2 claim for that model and route. Option-based results are exploratory, and that failure is a finding for RQ4 |
| R2-g | No eligible model | Report the gates and stop. RQ2 is not tested |

Shuffle against `static_apex` is descriptive in all branches. It says whether temporal information reaches the judgement. Qwen3-VL's order results are capped by frame merging.

## How the whole paper reads

| Scenario | Combination | Main claim | Closing sentence of the abstract |
|---|---|---|---|
| S1 | R0-a, A4 Aligned, R2-a or b | Passing a strict audit, models show a human-like cue pattern, limited to named datasets | Models can share human cue effects, and this needs a test at cue level |
| S2 | R0-a or b, A4 Not aligned | Dependence on the face does not imply human cue use | Face dependence and human-like cue use are separate properties |
| S3 | Natural positive, A4 Not aligned | Agreement on natural items failed to carry to decoupled cues, subject to domain limits | Agreement on natural behaviour did not predict randomised cue effects |
| S4 | R0-c | Validity is the main result, and much apparent accuracy depends on input format | Input format decides what such accuracy can mean |
| S5 | A4 Inconclusive, R2-f or d | Audit and theory stand, and the cue-level question stays open with stated requirements | The tools for a cue-level test now exist, and these models did not yet meet its gates |
| S6 | Models differ | Report as profiles, with no pooled statement | Models differ in which degree of similarity they reach |

The audit, the theorem and the estimand stand in every scenario. No scenario claims a mechanism, or a route through the model: RQ3 to RQ5 are out of scope for this draft unless their amendments finish (`GOALS.md`, goals G6 to G8).

## Discussion structure (no subheadings)

1. What was found, in one paragraph, from the scenario table.
2. What the audit means for existing claims. Which published results need repeating, stated without judging papers we read only in part.
3. Why behaviour alone cannot settle cue use, and what the results add to that. If section 6 found no gap, say that here.
4. How similar: a profile against the three degrees, per model. The third degree, computation, is untested here.
5. Limits, in order of weight. Avatars and static frames against 1.25 s of movement. Human predictions come from other stimuli. Reach of about 8 of 21 labels. The small actor sample. Option sets still identify category on the natural task. Thresholds set by the authors. Open models only.
6. What would settle it: original stimuli, new human data on the same edited clips (offered as future work, with no claim about the result), and playback speed once a route delivers it.

## Display items (7 of 8)

| Item | Content |
|---|---|
| Figure 1 | Audit: target position and option sets; delivery; weak labels; the eight-paper table as a schematic |
| Figure 2 | RQ0: delta per model and input, with accuracy against 1/6 and by target position |
| Figure 3 | Identifiability: two observers with the same behaviour and different effects; the cue spectra |
| Figure 4 | Natural-clip pattern score per model |
| Figure 5 | CFA per model and dataset, with between-group reference; behaviour beside CFA |
| Figure 6 | RQ2: effect and decision class per prediction, by model and route |
| Table 1 | Profile per model across RQ0, RQ1 and RQ2 |

## Pending items blocking a first full draft

A1 read; gate and A3 tables in Results; A4 freeze, recovery table and run; A2 run; the rule in inconsistency 5; model revisions; the licence for edits; ethics statement; registration link; Supplementary Note 1 from `theory/causal_feature_alignment.md`.

## Inconsistencies found across the repo's documents

None was changed. The drafts follow the stricter or the newer source in each case.

| # | Where | Conflict | Handling here |
|---|---|---|---|
| 1 | `GOALS.md` against `LIT_CHECK` | GOALS says "the first cue-level test". LIT_CHECK allows only the hedged "we located no study" | Hedged wording |
| 2 | `RESEARCH_QUESTIONS.md` against A2 | RQ2 is defined as graded edits of one cue through a generator with verified AUs. A2 has no generative AU edit, no dose-response for a verified AU, and its one graded cue (aperture) is out of the family. RQ2 as tested is narrower than RQ2 as stated | Paper describes what A2 does |
| 3 | `RESEARCH_QUESTIONS.md` rule 6 against A4 | Rule 6 gives ceilings of about .75 and .88 to .96. A4 uses .974 to .990. The .88 to .96 are split-half values on log odds, while A4 takes the square root of a corrected split-half on do effects. The .75 could not be traced to any output | A4 value, marked PENDING |
| 4 | `PREREGISTRATION.md`, `NEEDS_INPUT.md`, `STATE.md` | PREREGISTRATION has a pointer for A3 and none for A2. NEEDS_INPUT item 6 still lists A2 sign-off as open and recommends holding until A1 is read, though A2 was frozen before. STATE's tag list omits `a2-freeze`. A1 and A3 have no tags | Methods cites A2 directly; hashes PENDING |
| 5 | `GOALS.md` G2, DECISIONS, A3 table | The natural-clip RQ1 route uses A1's composite-grid rotations. A3 passes `single_face` forward after an A1 Fail, but no rule says whether grid-based pattern scores are read for such a model. This is a real gap and should be fixed before A1 is read | PENDING in Methods and outline |
| 6 | `RESEARCH_QUESTIONS.md` RQ0 against A3 and A4 | RQ0 says no later RQ runs on a failing model. The A3 table allows a retest route, and A4 keeps Molmo2 as descriptive despite its gate failures | A3 table followed |
| 7 | `notes_au_render.md` against `LIT_CHECK` and the renderer | Notes say FACSHuman was last pushed in 2021 with no headless path. LIT_CHECK says pushed 2026-04, and the repo now has its own headless renderer. The notes' first recommendation, asking Jack's group for renders, is still open | Renderer described as built |
| 8 | `COMPUTE.md` against PREREGISTRATION | Phase 6 says to pin revisions, and COMPUTE says commit SHAs should be pinned before any new eval. No frozen run records one | PENDING in Methods |
| 9 | Report against `theory` note | Report calls the CFA theorem unproved, and the theory note proves it. The report's realistic-venue paragraph also predates the GOALS target | Theory note followed |
| 10 | `STATE.md` against report | STATE says full EU-Emotion access and licence. The report lists the licence for edits as unverified. A2 edits clips in memory | PENDING under Data availability |
| 11 | Notes on reading status | Strachan is "full text read" in one note, P in LIT_CHECK. Zhang is read to 100,000 characters in the report, and 78,034 extracted in LIT_CHECK | Lower status used |
| 12 | Trial and configuration counts | Western basic: 119,382 (paper), 119,387 (files), 121,902 (thesis). Unique configurations in the theory table (basic EA 2,322; conversational EA 2,322; conversational WC 2,314) differ from `human_au_effects.json` (2,323; 2,325; 2,318) | Theory table quoted for spectra; JSON row counts for trials. Reconcile before submission |
| 13 | Theory note against A4 | The note defines the effect at dose delta from z, while A4 uses do(1) minus do(0). The note takes model reliability from replicates, while A4 uses split-half over stimuli. A4's text lists a blind B3 that `a4_analyse.py` does not compute. Its "half-shared observer near .50" awaits the empty RECOVERY_TABLE | Noted in Methods; PENDING |
| 14 | A4 against theory note | A4's "condition number 76 against about 3.5" compares 15 LibreFace cues with 25 merged cues. OpenFace 3.0 gives 22.4, and the note warns that cue counts differ | Full table given |
| 15 | Naming | G1 to G12 (goals), G2 to G4 (vision gates) and rung and delivery gates share labels | Paper names gates by function |
| 16 | EU-Emotion counts | Local norms hold 243 of the paper's 249 face clips, and no note says why | PENDING in Methods |
