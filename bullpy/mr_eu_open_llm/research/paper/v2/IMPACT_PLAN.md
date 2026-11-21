# Plan: the paper's headline and impact

**Written:** 2026-11-21, at the author's request: "construct a comprehensive plan for how this paper's impact and headline can be improved and implement it". Steps 1 to 8 were implemented the same day (Section 5). Steps 9 to 16 are next (Section 6). Decisions: DECISIONS 2026-11-21 (two entries); supplementary decisions table rows 103 and 104.

Word counts in this file come from `research/scripts/manuscript_words.py`, which removes comments, registry markers, slots, citations, headings and table rows before counting.

---

## 1. Diagnosis: why the draft undersold itself

**The headline was a list, not a finding.** The abstract closed on "Models differ in which degree of human similarity they reach", the scenario sentence outline.md had fixed for "the models differ". Four of its nine sentences were about validity, led by our own withdrawn task. The cue-level result, the paper's citeable core, arrived in sentence seven, without its size or the human reference.

**It read as a preregistration log.** Results had twelve sections, each with Question, Rule, Locked text and Writing rule blocks, seven rule tables, file:line citations to amendments and about thirty internal labels (A1 to A9, RQ0 to RQ5, G2 to G4, V1 to V4, B1 to B5, P1 to P15). Cue Effect Alignment was first defined in paragraph 8 of 11 of the Introduction. The Discussion ran test by test and opened with process ("Three findings hold whatever the later tests showed").

**It was too long.** As a reader would see it, the main text was 9,420 words (7,080 of prose, without the scaffold), against Nature Human Behaviour's 5,000.

**Some statements said more than the locked results show.** An independent editor-style read of 2026-11-21, given the paper files only and no view of ours, found these. Each was checked against the locked text before being acted on (Section 5, step 5):
- "That is the case the theory needs, observed" (Discussion), and "The identifiability result allows this" (Results). The behaviour comparison on the renders has no noise ceiling. Each configuration's human response rests on few trials, while cue effects pool all of them. On a randomised design, Proposition 3 says behaviour does constrain cue effects for linear observers. So the gap may be noise, and it is not the theory's demonstration.
- The Discussion said several published designs "let option position or prompt text identify the answer". Results, Section 1 shows only that they could not rule it out.
- "One stayed at chance": Molmo2 was at chance on the composite input (.163) and below it on the single crop (.072).
- "On rendered faces": the models saw new renders of the AU recipes. People saw animations from the original generator (Methods: "the same AU configurations, not the same stimuli").
- "Models differ in which degree...": the randomised test approximates Schyns' second degree and "does not meet it as framed" (Methods), and an Inconclusive result is not a degree.
- "Mid-sized" (Discussion) against "small" (Methods).
- Cover letter: "five of ten contrasts" overstated the stimulus edits, since only two matches survive every sensitivity; "matched no prediction" read as a null; and "written to stand whatever the results are" read as a paper with no finding.

**The independent read's verdict.** As drafted, about a 15% chance of being sent out for review; about 30% once reframed and with A8 in hand. The strongest contribution is methodological: a validity-gated, preregistered test of model cue effects against randomised human ones. Nature Human Behaviour fits only if the paper is framed around the human comparison. Otherwise Nature Machine Intelligence fits better.

## 2. The headline

**One sentence.** Matching answers cannot show that a vision-language model reads faces as people do. When facial cues are randomised, one small open model shares people's cue effects for basic emotions, replicated in a second human sample from the same laboratory, though less closely than two human groups agree with each other; the other model was not shown to.

**Three supporting claims, in the order the paper now makes them:**
1. *Accuracy alone cannot show that a model used the face.* Our earlier design let option position and the option set identify the answer. None of eight published designs measured the accuracy lost when the face is swapped. With option order rotated and the prompt fixed, all three models' answers depended on the face (deltas .071 to .402), although one never exceeded chance. (Results 1 and 2)
2. *Agreement on natural faces cannot identify which cues a model uses.* Theorem 1: identical answers on every natural item are compatible with any cue effects. In the data, all three models aligned with the human pattern on natural clips, including the model at chance. (Results 3)
3. *Randomising the cues can, and it separated the models.* Gemma 4 was aligned for basic emotions in both observer groups (residual CEA .386 and .516) and on a second human sample (.483), below human agreement (.780; .690). For conversational signals it was mixed. Qwen3-VL was inconclusive on the first sample and not aligned on these renders on the second. (Results 4)

The supporting results are face evidence beyond word knowledge (Results 7) and stimulus edits that agree with human predictions for some cues (Results 6).

**What it does not claim.** That any model uses the same strategy as people, or is human-level. Anything about real faces, or about the second sample as independent. That the answers-against-cues gap (Results 5) is real. That the faults we found are common in published benchmarks.

**Why this headline.**
- It is true in every A8 outcome.
- It does not lean on the one result without a noise ceiling.
- It gives the journal's broad readership a general lesson (answers against cues), a method that delivers on it, and a positive, novel finding with its size.

## 3. Guardrails that did not move

- No class, number, threshold, rule, eligibility or disclosure changed. Every value in the main text keeps its registry marker.
- Text that left the main text moved word for word to Supplementary Methods, into the topical section, under a block headed "... in full (moved 2026-11-21, length cut 7)". Paragraphs were copied from the committed text by script, not retyped.
- An automated check found every number of the old Introduction, Results and Discussion still visible somewhere in the paper. The only numbers it flagged are file:line references from the drafting scaffold, which is kept in comments.
- The wording rules hold:
  - a face-dependence pass is never "reads faces";
  - natural-clip alignment is never "uses the same cues";
  - an aligned CEA is "a shared pattern of cue effects", not human-level;
  - "not aligned on these renders" is always written in full;
  - the second sample is "same lab and generator" and never independent;
  - a stimulus-edit match is agreement with a prediction from other stimuli, BASIC ONLY;
  - an inconclusive result is not a null.
- Section 5's behaviour comparison stays descriptive and out of the abstract.
- A8 is untouched. Its scaffold and its Discussion branches, drafted before data, are kept in comments for the fill.
- Nothing was read that the rules hold back. No A8 output was opened.

## 4. Structure now

| Part | Content | Words |
|---|---|---|
| Title | Working title "Beyond matching answers: testing whether vision-language models use facial cues as people do"; three alternatives in `abstract.md` | 13 |
| Abstract | Question, audit, face dependence, theorem, randomised result with its size against human agreement, take-home | 150 (limit 150) |
| Introduction | Claims and Schyns' three degrees; validity; our earlier task; the eight-paper check; identifiability; CEA; novelty; what we did and found; preregistration and scope | 1,223 |
| Results | 1 Accuracy alone cannot show face use. 2 Answers depend on the face. 3 Agreement on natural faces cannot identify cue use. 4 With cues randomised, Gemma 4 aligns for basic emotions. 5 Answers on the same faces track people's only weakly (descriptive). 6 Masks and filters, about half the single-face contrasts. 7 Face evidence, not word knowledge alone. 8 Larger models (pending). 9 What could not be tested | 2,405 |
| Discussion | Continuous prose: answers against cues; the cue-level result and its limits; two descriptive observations; edits and priors; A8 (pending); five limits; two lessons | 1,235 |
| Main text | | 4,863 (limit 5,000), leaving about 140 words for A8 |

## 5. Implemented on 2026-11-21

1. **Title and abstract** (`abstract.md`). A working title, three alternatives, and a 150-word abstract built on the headline above. "Preregistered" is left out until the editors answer the cover letter's question.
2. **Introduction** (`introduction.md`). Cited sentences are kept word for word. Four passages moved to S2 and one to S11. A new paragraph states what we did and found.
3. **Results** (`results.md`).
   - Reordered, merged and retitled as findings.
   - The scaffold is kept in comments beside each section.
   - Labels are replaced by plain names.
   - Section 8 ("What each model has shown") is replaced by a pointer to Table 1.
4. **Discussion** (`discussion.md`). Rewritten as continuous prose with no subheadings, as the journal asks. The benchmark recommendation is kept word for word, with one added recommendation: report the accuracy lost when the face is swapped.
5. **Corrections**, each narrower than before: every item in Section 1's last list (DECISIONS 2026-11-21, second entry).
6. **Supplementary Methods.**
   - 33 Results paragraphs moved to S2, S3, S5, S6, S7, S8, S10 and S11.
   - Five Introduction passages moved to S2 and S11.
   - 17 Discussion paragraphs moved to S12.
   - Pointers renumbered, with a note mapping old section numbers to new.
7. **Display items.**
   - Legend titles now state each finding, and the legends define the labels still printed in the figures.
   - Table 1's header and note use plain names. Changed in `research/scripts/make_table1.py` and rebuilt, with the cells unchanged.
   - Cover letter rewritten to the new frame (`submission/cover_letter_draft.md`).
8. **Tooling.** `research/scripts/manuscript_words.py` counts words against the limits and writes a clean reading copy (`--out`), with tests in `tests/test_manuscript_words.py`.

## 6. Next steps, in order

| # | Step | Who | Why it matters |
|---|---|---|---|
| 9 | Relabel the figures in `make_figures.py`: plain panel titles in place of A1, A3, RQ0, SESOI and do-effects. Then rebuild with `--real`, update `SOURCES.md` and the reproduction pins, and rerun the clean-checkout check | agent | Editors look at Figure 1 and the main figure first |
| 10 | Read the reframe (`git diff`) and choose the title | author | Authorship is the author's |
| 11 | Decide whether to compute a noise ceiling for the answer agreement of Results 5: split-half of each configuration's human responses, labelled exploratory and post hoc, CPU only. If the gap survives the ceiling, it becomes a second finding (answers can understate shared cue effects). If not, Results 5 says so, and the Discussion's second observation goes | author decides, agent runs | It is the most interesting secondary result and the one most open to attack |
| 12 | Fill A8 within a budget of 120 words in Results and 50 in the Discussion, with detail to S9, keeping the main text under 5,000 | agent, after A8 locks | Model size is the first scope question a reviewer will ask |
| 13 | Optionally, a schematic Figure 1, the eighth display item. One row per model: answers, face dependence, natural-pattern alignment, randomised cue effects and the human reference, from locked outputs, built by `make_figures.py` | agent, if the author wants it | It shows the paper's argument at a glance |
| 14 | Optionally, human accuracy beside model accuracy. Report the human item-norm accuracy on the 243 items as a descriptive design fact, with its caveat that the procedure differed | author decides | The independent read asked for it; the paper never sets model accuracy (at most .469) beside people's |
| 15 | At final formatting: name Results headings instead of "Section N"; give references cited only in the Supplement ([14], [16], [21], [23]) their own Supplementary reference list; check Methods (6,653 words) against the formatting guide | agent | The journal's format |
| 16 | Send a presubmission enquiry to Nature Human Behaviour with the new abstract and cover letter. If declined, go to Nature Communications or Nature Machine Intelligence, with TMLR as the planned fallback. Post a preprint at submission, after checking the journal's preprint policy, which the repository has not verified | author | A fast, cheap signal on fit before a full submission |

**For the why-this-journal sentence (step 16).** A candidate, for the author to edit: "How far AI systems share human social perception is a question for psychology, cognitive science and AI alike, and the cue-level test we propose speaks to all three." A precedent would help: we recall that Nature Human Behaviour published Strachan et al. (2024) on theory of mind in large language models and humans. That has not been checked here, so do not cite it until it is.

## 7. Risks and how the text meets them

- **"One small model, on avatars."** The limits paragraph states this first. Larger models (A8) bound the scope, and Gemma 4 31B's gates were preregistered as possibly untestable.
- **"The replication is from the same lab."** Every mention says "same lab and generator", and the limits paragraph calls it a second sample, not a second method.
- **"The behaviour gap is noise."** The text now says it may be, and step 11 can settle it.
- **"Is it preregistered?"** The cover letter asks the editors. The abstract avoids the word until they answer.
- **"The faults are your own."** The text says so, and it says that our confound is not shown to be common (Discussion, benchmark audit).
