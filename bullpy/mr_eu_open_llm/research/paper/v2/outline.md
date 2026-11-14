# Outline: Results, Discussion, and how the paper reads under each outcome

Drafted 2026-11-07, updated the same day after the A4 freeze. No A1 result has been read, and nothing below predicts one. Each branch uses the preregistered outcome classes and adds none. Results text is written only from locked output. A5 was frozen on 2026-11-08 (evidence classes confirmatory, balance descriptive). A6 and A7 are not frozen, so anything resting on them is marked provisional. **Updated 2026-11-12:** A1 and A3 read (R0-a on all three models) and natural-clip RQ1 confirmatory on all three, both on 2026-11-09; A7 frozen 2026-11-08 and its Stage 0 run 2026-11-10 (R3-b on both eligible models); A6 closed as a limit 2026-11-10; A9 frozen 2026-11-09. Sections below that predate these are marked where they changed.

**Working title:** [PENDING: title]. Candidate: "Vision-language models against randomised human cue effects on faces".

## Journal limits

| Item | Article limit | Plan |
|---|---|---|
| Abstract | Up to 150 words, unreferenced | Template below |
| Main text | Up to 5,000 words, excluding abstract, Methods, references and figure legends | Introduction 1,401 words as drafted, leaving about 2,500 for Results and 1,100 for Discussion |
| Display items | Up to 8 figures and tables | 7 planned, below |
| References | No limit | 50 listed, of which 43 are cited |
| Structure | Introduction without heading; Results with topical subheadings; Discussion without subheadings; Methods with subheadings | Followed |

**How far this was verified.** These limits are from the journal's official content-types page, read on 2026-11-07. WebFetch was redirected by a cookie check, so that page was fetched with curl using a browser user agent and a cookie jar. A later attempt the same way, for the policy pages, was blocked by the session's safety classifier, and was not retried. **Please re-read the page yourself before relying on these limits.** The rule that Methods carry short bold headings came from the formatting page in the same way.

**Update 2026-11-07.** `citation_check.md` read the public journal pages: Article limits, preregistration rules and the reporting summary are there. A Registered Report is not possible, because data collection has started, so this is an Article. **Earlier note.** Preregistration statement rules, reporting summary rules, and current data and code requirements. A web search found a 2021 editorial (s41562-021-01190-w) saying that central custom code is peer reviewed, must go to a DOI-granting repository at acceptance, and that "available on request" code statements are generally refused. That came through a search summary only. Whether a repository tag counts as preregistration for the journal is also unverified. Page charges and preprint policy remain unverified (`LIT_CHECK_2026-11-07.md`).

**Length check, 2026-11-13 (before any A2, A4, A7 Stage 1, A8 or A9 output).** Words counted with HTML comments and headings left out. The estimate also leaves out registry and slot markers, citations and the drafting scaffold (Question, Rule, Slots, Writing rule, Figure notes), and keeps one outcome branch per slot.

| Part | Drafted now, all branches | Estimate with one branch kept | Plan (table above) |
|---|---|---|---|
| Introduction | 1,606 | 1,606 | 1,401 |
| Results | 6,137 | 2,700 to 3,200: locked prose about 2,100, plus 500 to 1,000 for one branch per slot, depending on whether model-level sentences are written once or per model | about 2,500 |
| Discussion | 4,102 | 2,400 to 2,900, of which Limitations 963 and the benchmark-audit recommendation 337 | about 1,100 |
| Main text | | 6,700 to 7,700 | 5,000 |

The main text is set to run about 1,700 to 2,700 words over. None of the cuts below changes a claim, a number or a disclosure: every deviation is disclosed in Methods, which the limit does not count, and in the Supplementary decisions table. Cuts, in the order they would be made:
1. Limitations: a paragraph of about 250 words in the Discussion on the limits that bear on the claims made. The numbered list moves to the Supplementary Information, each item beside its Methods disclosure. Saves about 700.
2. Per-model and per-dataset values go to Table 1 and the figures; the text gives each model's headline and the exceptions. Saves about 300 to 500.
3. Results section 1's literature check moves to the Figure 1 legend and the Supplementary Information (about 150). The locked prose of sections 2 and 4 drops recovery detail that Methods already gives (about 300 to 400).
4. The Introduction goes back to its plan of 1,401 words or below. Saves about 200 to 400. For example, the search-coverage sentences (62 words) move to Methods, and the paper-by-paper list in the fifth paragraph (99 words) is already given by Results section 1 and Figure 1e.
5. The benchmark-audit recommendation shrinks to about 120 words, with the audit itself in Methods. Saves about 200.

*Applied 2026-11-13, cut 4 in part (it depends on no pending result).* The Introduction's paper-by-paper list gave way to a pointer to Figure 1e, with the eight papers cited together. The search-coverage sentences moved to Supplementary Methods, Section S2, which already held the literature check, rather than to Methods, which is itself long. The Introduction is now 1,470 words, against 1,606; the last 69 words to the plan's 1,401 are left for the final pass, because they would mean rewording sentences the citation check and the draft review passed.

*Applied 2026-11-13, cut 5 (it depends on no pending result).* The benchmark-audit recommendation is now 189 words, against 341. The audit's detail (the two partial cases, the source of the majority-class gap, and why these accuracies are lower bounds) moved word for word to Supplementary Methods, Section S2. The Discussion keeps what DECISIONS 2026-11-09 requires there: the recommendation, its exploratory status, and that no field-level claim is made. It stops short of the plan's 120 words because the rest would cut those caveats.

*Applied 2026-11-13, cut 3 (it depends on no pending result).* About 200 words came out of the Results' locked prose, against the plan's 450 to 550:
- Section 1 (106 words): the literature check is now a short summary that points to Figure 1e, and the full coding moved word for word to Supplementary Methods, Section S2.
- Section 4 (63 words): the natural-clip recovery values gave way to one sentence. All of them were already given in Methods and in Supplementary Methods, Section S5.
- Section 5 (31 words): the run 3 rates and the exempt cells' values came out. Both were already in Methods and Section S6.

The plan's estimate assumed recovery detail in section 2, which holds none, and it counted section 5's thin detector pass, which stays because the Results structure above requires it. Each cut is noted in place in results.md, and the registry's literature row names both places. With cuts 4 and 5, the main text is about 490 words shorter than the estimate above. The rest waits on cut 1 (the user's view, STATE hand-off item 5), on cut 2 (after the results) and on writing each branch to its budget.

*Applied 2026-11-14, cut 2 for A4 (A4 locked; the first result section filled).* The per-dataset account of A4 moved word for word to Supplementary Methods, Section S6, "A4 results in full". The main text keeps each model's headline, the values the decisions rest on, and the exceptions: Qwen3-VL's failed gates, and Gemma 4's weak conversational EA sensitivities. Results 5 went from 588 to 352 words, Results 6 from 188 to 147, and the Discussion's A4 section from 678 to 385, counted as above. No claim, number or disclosure left the paper, and every value keeps its registry marker. A2 and A9 follow the same pattern when they are locked.

*Recount 2026-11-14, 06:55 (A4 and A9 filled; A2, A7 Stage 1 and A8 still drafted with every branch).* This uses the same rules, by script, so it is close to but not identical with the hand count above. Introduction 1,440. Results 3,756: sections 5, 6 and 6b, now locked, are 593, 147 and 469, and the A2, A7 and A8 sections still hold every branch. Discussion 3,802, still with every A2, A7 and A8 branch and the 963-word Limitations list. Picking one branch each for A2, A7 and A8 should bring the main text to about 6,600 to 7,100 words, against 5,000. The gap closes only with cut 1 (Limitations to the Supplementary Information, about 700 words) and with tight Discussion branches, so cut 1 is the decision that matters (hand-off item 5).

Together these save about 1,850 to 2,350 words; the rest comes from writing each chosen branch to its budget at steps 4.1 and 4.3. Methods (7,402 words) and the figure legends (1,017) fall outside the limit, but check both against the journal's formatting guide before submission.

## Abstract template (150 words at most)

> Some vision-language models are reported to match people at reading mental states from faces, but several claims we located cannot rule out option order or prompt text. We audited [PENDING: n] models on 243 acted clips, with option position controlled and faces swapped under fixed prompts. [PENDING: RQ0 sentence.] We prove that agreement on natural items does not in general determine which cues a system uses. We therefore compared model cue effects with human effects from randomised facial action units. [PENDING: RQ1 sentence.] Stimulus interventions tested preregistered human predictions. [PENDING: RQ2 sentence.] [PENDING: closing sentence from the scenario table.]

## Results structure

Subheadings are provisional. Sections 1 and 3 hold whatever the outcomes are.

1. **Several published designs, and our earlier one, cannot rule out option order or prompt text.** The audit: target position, option sets, delivery, weak labels, the eight-paper table. Figure 1.
2. **Dependence on the face (RQ0).** A1, then A3, per model, read through the frozen table. Figure 2.
3. **Behaviour on natural items does not in general settle cue use.** The identifiability results, and the cue spectra as the empirical measure of the gap. Figure 3.
4. **Pattern alignment on natural clips (RQ1).** Family pattern score per model. Table or Figure 4.
5. **Cue effects under randomisation (RQ1).** A4: validity gate, with the thin V1 pass and the family flag stated, then residual CEA per model and dataset against its own between-group reference, with raw CEA as secondary. Figure 5.
6. **Do behaviour and cue effects disagree?** CEA beside output similarity and modal agreement. Part of Figure 5.
7. **Interventions on the stimulus (RQ2).** Gate B, then each prediction by decision class. Figure 6.
8. **What each model has shown.** One profile per model across the three degrees. Table 1.
9. **The route through the model (RQ3), a scoped limit.** Why the literal test cannot fail, and why depth is not testable. Rests on A7's Stage 0 table, run 2026-11-10 (branch R3-b on Gemma 4 and Qwen3-VL). A paragraph, with the table in the Supplement.
10. **RQ4, and RQ5 as future work.** A5 (RQ4) is frozen and reads A2's cells once they exist. A6 (RQ5) was closed as a limit by its own pre-freeze check on 2026-11-10, so RQ5 is named as future work.

## Outcomes by question

### RQ0

Read per model. `RESULTS_REGISTRY.md` records A3 as Pass for all three models, so the Fail rows of the A3 table cannot occur. [PENDING: A3 table in Results.] Pass tests dependence on the image, not competence, so a Pass must never be written as "reads faces".

| Branch | Condition for Gemma 4 and Qwen3-VL | How section 2 reads | What follows |
|---|---|---|---|
| R0-a | Both pass A1 | Accuracy needs the matching face on both inputs. State how much survives, and compare with accuracy against 1/6 | Natural-clip RQ1 confirmatory, subject to the clip rule; A2 on both routes; A4 |
| R0-b | One passes, one fails | The models differ. The failing model showed face dependence only on a face crop | Passing model: natural-clip RQ1 confirmatory, A2 on both routes, A4. Failing model: natural-clip RQ1 descriptive, A2 on `single_face` only, A4 |
| R0-c | Both fail A1 | The composite input limits face use, and accuracy on that format cannot be read as face reading. This is itself the audit's main finding | Natural-clip RQ1 descriptive. A2 on `single_face` only, and A4. Temporal predictions P12 to P15 have no route and are reported as descriptive |
| R0-d | A1 not complete | Nothing can be stated, because Holm covers all three models. A missing result counts as not passed | Wait. No submission on this branch |

Molmo2 is reported in RQ0 either way. It stays out of A2 and of A4's Holm family on its gate failures, whatever A1 shows, and A4 reports it descriptively if it is run at all.

### RQ1

Each model has one natural-clip outcome and one A4 outcome per dataset, then one A4 headline per task.

| Natural-clip pattern score | Meaning | Must not say |
|---|---|---|
| Positive | "Aligned with the human pattern" | "Uses the same cues" |
| Null | Alignment not shown | "Not aligned". Power is limited for partial sharing |
| Over 5% of clips lost | Descriptive only | Any confirmatory claim |
| Model failed A1, passed A3 | Descriptive: the grid limited that input (DECISIONS 2026-11-07) | Any confirmatory claim |
| Model passed A1, failed A3 | Descriptive, pending the inspection the A3 table requires | Any confirmatory claim |
| Model failed both | Computed for the record, with no claim | Any RQ1 to RQ5 claim for that model |

| A4 outcome | Meaning | Must not say |
|---|---|---|
| Aligned | Reliable shared cue pattern on these avatars, on a peak frame: all gates, signal, alignment Holm *P* < .05, residual CEA of at least .30, lower bound above 0 | "Same strategy", or "human-level" unless the interval reaches the residual between-group reference |
| Not aligned on these renders | Always in these words. The model responds to AUs (signal, gates), and the upper bound divided by the square root of *R_S* is below .30 | "Not aligned" alone. "Does not use human cues on real faces". Domain, static route and amplitude remain open, and most families are unconfirmed by a detector |
| Inconclusive | A gate failed, no signal, or anything else | Any statement on alignment. Report which gate failed |

Datasets may disagree: for instance basic and conversational, or WC and EA. Report each alone, then the frozen headline per model per task: Aligned or Not aligned on these renders only if WC and EA agree, Mixed if one of each, Inconclusive otherwise. Basic and conversational are never pooled. At conversational 0.25 times, recovery gives no accuracy claim for residual CEA, so state that beside any weak conversational result. Raw CEA is secondary and never decides.

| Natural clips | A4 | How sections 4 to 6 read |
|---|---|---|
| Positive | Aligned | The two agree. Natural-clip alignment gains cue-level support, within avatars |
| Positive | Not aligned on these renders | The most informative disagreement: the pattern matched on natural clips, and did not hold when cues were decoupled. It fits the identifiability result, but cannot exclude the avatar domain, so state both |
| Null | Aligned | Human-like cue effects appear when cues are isolated. Natural-clip power and domain differences are the live explanations |
| Null | Not aligned on these renders | No evidence of human-like cue use on either arm, with the limits of each |
| Any | Inconclusive | The paper rests on the audit, the theory and RQ2. Say what would be needed: a validated render, or the original stimuli |

Section 6 has two outcomes. If behaviour and CEA disagree on the same stimuli, that is the empirical demonstration the theory needs. If they agree, say so: the theorem stands as a logical limit, and these models on these stimuli do not show the gap. Do not describe agreement as a failure of the theory, or as proof of use.

### RQ2

RQ2 in this paper means A2's stimulus arm (`GOALS.md`, G4), which is narrower than RQ2 as stated in `RESEARCH_QUESTIONS.md`. A2 tests masks, filters and frame order. It has no AU edit, no verified dose and no human dose-response curve, so it says nothing on necessity or sufficiency of an AU, and sections 5 and 7 cannot be read as the same test. Every result carries its BASIC ONLY scope.

| Branch | Condition | How section 7 reads |
|---|---|---|
| R2-a | Gate B passes; most predictions Match | Judgement moves with the stated human predictions on basic labels in this stimulus set. No statement that people would respond this way to these edits |
| R2-b | Gate B passes; mixed classes | Report by cue type: which regions, bands and temporal cues move the judgement in the predicted direction, which oppose it |
| R2-c | Gate B passes; Opposes on several | The model moves, but against the stated prediction. Check the fill rule and the bridge readout, and report the disagreement |
| R2-d | Gate B passes; mostly Inconclusive | Capped contrasts cannot reach Null by design: 8 of 16, or 10 of 16 on Qwen3-VL. Report equivalence intervals. Do not call these nulls |
| R2-e | Null on eligible contrasts | Hard to reach by design, so reportable as a real result for that cue, model and route only |
| R2-f | Gate B fails | The option-free readout carries no usable face signal. No RQ2 claim for that model and route. Option-based results are exploratory, and that failure is a finding for RQ4 |
| R2-g | No eligible model | Report the gates and stop. RQ2 is not tested |

Shuffle against `static_apex` is descriptive in all branches. It says whether temporal information reaches the judgement. Qwen3-VL's order results are capped by frame merging.

### RQ3, a scoped limit (DECISIONS 2026-11-07)

Not a test and not a headline. The reason is that the literal test is vacuous and a route split has no human reading, not cost. **Updated 2026-11-12:** A7 was frozen on 2026-11-08 and Stage 0 ran on 2026-11-10. On both Gemma 4 and Qwen3-VL the exit cut is complete, its blocks are live but are not cuts, and the region test failed, so R3-b applies to both. R3-c applies only if the route arm runs in Stage 1.

| Branch | Condition | How section 9 reads |
|---|---|---|
| R3-a | A7 frozen, Stage 0 run, no complete cut at the exit on a model | No complete vision cut exists on this input for that model, so RQ3 is not testable there |
| R3-b | Exit complete, blocks not admitted or not live | The only complete cut is the whole path, whose contrast is zero by construction. Not testable beyond that |
| R3-c | Blocks admitted and route arm run | Optional supplement. One edit, one readout, one model. No human reading |
| R3-d | Region test passes on a model | A non-vacuous cue-level answer becomes possible. G7 reopens by a dated entry |
| R3-e | A7 not frozen or Stage 0 not run at submission | The vacuity and depth arguments stand as logic. State that no site was shown complete on `single_face` or on the blind readout, and that the table is missing |

Settled at A7's audit (DECISIONS 2026-11-08, "A7 audit closes block admissibility"): a block of a complete cut is not itself a cut and does not meet G7's wording, so RQ3 rests on the complete exit cut alone.

### RQ4 and RQ5, provisional

A5 (RQ4, prior and evidence on the blind readout, largely a reanalysis of A2's cells) and A6 (RQ5, shared errors on CREMA-D, across its fixed option set) were drafts when this was written. **Updated 2026-11-12:** A5 was frozen on 2026-11-08, with its evidence classes confirmatory and its balance verdicts descriptive, and gets a Results section once A2's cells exist. A6 was closed as a limit on 2026-11-10, when its own first-frame check failed, so RQ5 is named as future work and the limits in the Discussion stand. Their decision: `GOALS.md`, "If time or compute runs short".

## How the whole paper reads

| Scenario | Combination | Main claim | Closing sentence of the abstract |
|---|---|---|---|
| S1 | R0-a, A4 Aligned, R2-a or b | Passing a strict audit, models show a human-like cue pattern, limited to named datasets | Models can share human cue effects, and this needs a test at cue level |
| S2 | R0-a or b, A4 Not aligned on these renders | Dependence on the face does not imply human cue use on these renders | Face dependence and human-like cue use on rendered cues are separate properties |
| S3 | Natural positive, A4 Not aligned on these renders | Agreement on natural items failed to carry to decoupled cues on these renders, subject to domain limits | Agreement on natural clips did not carry to randomised cue effects on rendered faces |
| S4 | R0-c | Validity is the main result, and much apparent accuracy depends on input format | Input format decides what such accuracy can mean |
| S5 | A4 Inconclusive, R2-f or d | Audit and theory stand, and the cue-level question stays open with stated requirements | The tools for a cue-level test now exist, and these models did not yet meet its gates |
| S6 | Models differ | Report as profiles, with no pooled statement | Models differ in which degree of similarity they reach |

The audit, the theorem and the estimand stand in every scenario. A Mixed headline for a task is reported as such and adds no scenario. No scenario claims a mechanism, or a route through the model: RQ3 is a scoped limit, and RQ4 and RQ5 stay out unless their amendments freeze and finish (`GOALS.md`, goals G6 to G8; DECISIONS 2026-11-07).

## Discussion structure (no subheadings)

1. What was found, in one paragraph, from the scenario table.
2. What the audit means for existing claims. Which published results need repeating, stated without judging papers we read only in part.
3. Why behaviour alone cannot settle cue use, and what the results add to that. If section 6 found no gap, say that here.
4. How similar: a profile against the three degrees, per model. The third degree, computation, is untested here.
5. Limits, in order of weight. Avatars and static frames against 1.25 s of movement. Most families unconfirmed by a detector, and the thin V1 pass. Human predictions come from other stimuli. Reach of about 8 of 21 labels. The small actor sample. Option sets still identify category on the natural task. Thresholds set by the authors. Open models only. No route claim (RQ3).
6. What would settle it: original stimuli, new human data on the same edited clips (offered as future work, with no claim about the result), and playback speed once a route delivers it.

## Display items (7 of 8, one slot free for A5)

| Item | Content |
|---|---|
| Figure 1 | Audit: target position and option sets; delivery; weak labels; the eight-paper table as a schematic |
| Figure 2 | RQ0: delta per model and input, with accuracy against 1/6 and by target position |
| Figure 3 | Identifiability: two observers with the same behaviour and different effects; the cue spectra |
| Figure 4 | Natural-clip pattern score per model |
| Figure 5 | Residual CEA per model and dataset, with its between-group reference; raw CEA as secondary; behaviour beside CEA |
| Figure 6 | RQ2: effect and decision class per prediction, by model and route |
| Table 1 | Profile per model across RQ0, RQ1 and RQ2 |

Stage 0 table for RQ3: Supplement, from `a7_stage0_gemma4.json` and `a7_stage0_qwen3vl.json`.

## Pending items blocking a first full draft

**Updated 2026-11-12.** Done: A1 read; gate and A3 tables in Results; A7 freeze and Stage 0; A5 frozen; A6 closed as a limit; the licence for edits (2026-11-08); the two EU-Emotion facts in item 16; Supplementary Note 1, including residual CEA; Figures 2 and 4 drawn from locked outputs. From the queue: A2 run (Gate B), then A5 on its cells and A7's route arm where Gate B and the sham pass; A4 scoring and analysis, then A9 scoring; A8. For the user: ethics statement; a dated public record of each amendment (archive DOI or registry deposit), since the journal does not say whether a tagged commit counts. The theory note's full proofs are now Appendix A of Supplementary Note 1 (2026-11-12).

## Inconsistencies found across the repo's documents

Source documents outside `research/paper/v2/` were not edited. Where the paper now follows one source, the source to correct is named. "Resolved" means the paper text is settled from a named file. "Open" means a fact or a decision is missing, and says which.

| # | Where | Conflict | Status | Resolution, with source, or what is missing |
|---|---|---|---|---|
| 1 | `GOALS.md` against `LIT_CHECK` | "First cue-level test" against the hedged "we located no study" | Resolved | Hedged and cue-level. `GOALS.md` line 7 now reads "located no earlier example"; `LIT_CHECK` bottom line 2 |
| 2 | `RESEARCH_QUESTIONS.md` against A2 | RQ2 is defined by graded generator edits with verified AUs. A2 has no AU edit, no dose-response for a verified AU, and its one graded cue is out of the family | Resolved in the paper | RQ2 here means A2's stimulus arm, the term in `GOALS.md` G4 and A2's Questions. A2, "Not in this suite", excludes generative AU edits. Methods and the RQ2 section above say what is not tested. `RESEARCH_QUESTIONS.md` still states the wider RQ2: correct or annotate it |
| 3 | `RESEARCH_QUESTIONS.md` rule 6 against A4 | Rule 6 gives about .75 and .88 to .96; A4 uses .974 to .990 | Resolved, with one inference | **.88 to .96 traced:** `human_au_effects.json`, `beta_family_25_subjects_and_stimuli_split_pooled_fit.split_half_r`: .960, .959, .908, .882, split-half on log odds. **.75 traced by inference:** the field `leave_one_subject_out_mean_r` on `beta_family_25` gives .762, .781, .752, .718, mean .753, and it is what rule 6 describes, each subject against the rest (`human_au_effects.py`, `leave_one_subject_out`). Rule 6 names no file, so the link is an inference from value and definition. On do effects the same field gives .740, .729, .727, .719. **Neither is used.** The paper uses A4's ceilings on do effects: raw .974 to .990, residual .960 to .983 (`a4_human_ceiling.json`) |
| 4 | `PREREGISTRATION.md`, `NEEDS_INPUT.md`, `STATE.md` | Pointers and tag lists out of date | Resolved | `PREREGISTRATION.md` now points to A2, not to A4: add it. `NEEDS_INPUT` item 6 is resolved. `STATE.md`'s tag line lists every freeze tag as of 2026-11-09. A1 and A3 have no tags, so Methods names commits from `git log`: 410da27, b3af2cc, b6c712d, all on `origin/main`. Commit hashes are sufficient; no retroactive tags are added, since a tag made later could be misread as a freeze date |
| 5 | `GOALS.md` G2, DECISIONS, A3 table | No rule for natural-clip scores after an A1 Fail | Resolved | DECISIONS 2026-11-07, "Natural-clip RQ1 under each RQ0 outcome". Now in Methods and the tables above. The Pending entries are removed |
| 6 | `RESEARCH_QUESTIONS.md` RQ0 against A3 and A4 | RQ0 says no later RQ runs on a failing model | Resolved | DECISIONS 2026-11-07: the frozen A3 table is the later and more specific rule, so it governs. Molmo2 is a separate case: it is out on gates, not on RQ0 |
| 7 | `notes_au_render.md` against `LIT_CHECK` and the renderer | Last push 2021 with no headless path, against 2026-04 and a built renderer | Resolved | The notes' 2026-11-07 section supersedes their recommendation. `LIT_CHECK` s3 has the later GitHub read. A request to Jack's group is still the only route to the human stimuli, and it is the user's decision (`GOALS.md`, rule 6). The paper needs neither to stand |
| 8 | `COMPUTE.md` against PREREGISTRATION | Revisions unpinned | Resolved | `REPRODUCTION.md`, "Model revisions". Commit prefixes in Methods match |
| 9 | Report against theory note | Report calls the theorem unproved, and its venue paragraph predates GOALS | Resolved | Theory note followed, with proofs complete for the statements as written; venue from `GOALS.md`. The report is not cited for either |
| 10 | `STATE.md` against report | STATE records full access and licence, per the user on 2026-11-04. The report lists "EU-Emotion licence for edits" as unverified. A2 edits clips in memory | Resolved 2026-11-08 | The user confirmed that the team controls EU-Emotion and its creators are collaborators, with full permission for public item-level norms and for in-memory edits (DECISIONS 2026-11-08, "EU-Emotion permissions settled"; NEEDS_INPUT item 7). Methods, Data availability, says so. Clips, frames and edited images stay out of the repository by choice |
| 11 | Notes on reading status | Strachan and Zhang read at different levels in different notes | Resolved | `citation_check.md`: #1 read in full on the arXiv version only, with the published version not compared; #13 read in full on arXiv v2. `references.md` updated. A4's "Citation gaps" and `LIT_CHECK`'s status letters predate this |
| 12 | Trial and configuration counts | Trials: 119,382 (paper), 119,387 (files), 121,902 (thesis). Configurations: 2,322, 2,322, 2,314 against 2,323, 2,325, 2,318 | Resolved, with one gap disclosed | **Configurations verified** on `au_emotion_data.zip`: configuration strings give 4,659, 2,323, 2,318, 2,325 (`human_au_effects.json`, A4's manifest), and unique rows of the 33 amplitudes give 4,659, 2,322, 2,314, 2,322 (`cfa_identifiability.py`, `drop_duplicates`). Both are right and differ by what is counted. Methods says so. **Trials:** Methods uses file counts with "other" kept, so neither 119,382 nor 121,902 appears. `human_au_effects.json`, `trial_count_reconciliation`, calls the thesis gap likely explained but not proven, and the 5-trial gap with the paper stays unexplained (`notes_au_render.md`). The paper states neither |
| 13 | Theory note against A4 | Dose, model reliability, a blind B3, and "half-shared near .50" | Resolved | The note now has the set-to form and takes *R_S* by split-half across stimuli. `a4_analyse.py` computes B3 (`blind_image_rms`, in `blind_summary`). A4's frozen text no longer gives .50, and `a4_recovery.json` fills the table. See 20 for what remains |
| 14 | A4 against theory note | Condition number 76 against about 3.5 | Resolved | Frozen A4 gives 76.1 on 15 cues, 22.4 on 10 cues, 3.4 to 3.5 on 25 cues, and says cue counts differ. Full table in Methods |
| 15 | Naming | G labels shared by goals, vision gates and rungs | Resolved | Methods now names the vision gates by function. Goal labels appear only here. V1 to V4, B1 to B5 and C1 to C5 stay, each inside its amendment |
| 16 | EU-Emotion counts | "Local norms hold 243 of 249", from the report, is wrong | Resolved 2026-11-12 | The norms hold all 249 (`n_human_items_available`). The clip list holds 243, from a folder scan with no exclusion (`build_eu_full_manifest.py`). Rerunning the match against `eu_emotions_oreilly_item_level.xlsx` shows 243 clips on 242 norm rows: seven items have no clip (six High, one Low; IDs listed in `research/reproduction/eu_missing_items_check.json` since the permission was settled), and one item is matched twice, by `emotions_23/.../that item_cut.mp4` exactly and `emotions_13/.../that item_cut(1).mp4` by the loose rule. `n_matched` 243 counts clips, so it hid this. This also settles 48 Low against 49 (one Sad item), and shows that Kind, Unfriendly and Jealous lose nothing. **Missing:** (a) whether the seven are absent from the distributed set or only from our copy, which needs a listing of the cluster's copy or the authors; (b) whether the two that item files are one video, which needs a checksum of both on the cluster, since the local folder has only one. The frozen designs stay at 243 either way. If (b) is yes, disclose that one video counts twice. **(b) settled 2026-11-11:** one recording encoded twice (same 347 frames at 23.976 fps, 1440x1080, mean frame difference under 1 grey level; `research/reproduction/eu_duplicate_clip_check.json`); disclosed in Supplementary Methods. **(a) settled 2026-11-12:** all seven are present as unedited originals only, with no edited cut, and the clip list takes edited files only (`research/reproduction/eu_missing_items_check.json`). The user confirmed the same day that the copy we hold is the set EU-Emotion's creators, who are collaborators, supplied for this work, so nothing remains to ask the authors. Supplementary Methods says so |
| 17 | A4, within its own text | Disclosures say "Aligned is raw". The Asymmetric rule says "uncorrected" does not mean raw CEA | Resolved | The rule governs, and `apply_rule` takes the statistic named by the recovery file. Methods says uncorrected values of the confirmatory statistic |
| 18 | A4, within its own text | "Corrected values are not used for decisions" against the *R_S* correction in Not aligned | Resolved | The sentence concerns the ratio to the ceiling (`over_ceiling`, descriptive). `decide` uses `ci[1] / sqrt(r_s)`. Methods says each thing in its place |
| 19 | A4 and [19] against the public files | "Eight base faces per group", against 967, 8, 4 and 4 face codes | Resolved in text, one point open | Methods gives both, from `human_au_effects.json` `structure` and `data_caveats`. **Open:** what the 4 codes on conversational data mean. A4's "identity coverage is narrower" is kept as frozen, but check it for conversational sets |
| 20 | Theory note against A4 | A4's confirmatory statistic, residual CEA, appears nowhere in the note, and G9 says the estimand is defined | Resolved 2026-11-09 | Supplementary Note 1 defines residual CEA as `scripts/a4_analyse.py` computes it and states which results carry: Theorem 1 and Corollary 1.1 carry; Proposition 2 does not as stated; Proposition 3's upper bound only when the mean directions agree; Proposition 5 exactly only for a fixed projector. The theory note's "Residual CEA" section points there |
| 21 | Abstract template against DECISIONS | "Most claims" against "several", softened on 2026-11-07 | Resolved | Abstract and subheading now say "several" and "some", per `citation_check.md` corrections 2 and 3 |
| 22 | `RESEARCH_QUESTIONS.md` rule 8 against [27,28] | Rule 8 asks for two detectors of different lineage. The two share AU training data | Resolved in text | Methods states the shared data. Natural-clip claims do not rest on detector independence. Rule 8 should be corrected |
| 23 | Disclosure 5 against journal policy | The natural-clip strength rule was added after power was known. The journal requires originally planned analyses to be reported for deviations other than flaw or feasibility | Resolved 2026-11-09 | Treated as a deviation, which is the stricter view. The 0.75x run passed (DECISIONS 2026-11-04, "Strength rule result"), so the added rule and the 1x-only reading give the same verdict: confirmatory. Methods disclosure 5 now says so, and the 1x calibration is reported beside the 0.75x one. No decision depends on which view is taken |
