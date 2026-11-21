<!--
Draft, 2026-11-21 (IMPACT_PLAN.md, step 16). A presubmission enquiry asks the editors whether they would consider the full manuscript, before it is submitted. Sending it, and everything in it, is the author's decision; nothing here has been sent.
Check the journal's current instructions for presubmission enquiries before sending: what they ask for (usually a short letter and the abstract, sometimes a summary of findings) was not verified in this repository.
Claims are limited to what is locked in RESULTS_REGISTRY or proved in Supplementary Note 1, as in cover_letter_draft.md. A8 (larger models) is pending; the enquiry states it as under way, with no result.
-->

[USER: date]

Dear Editors,

We would like to ask whether Nature Human Behaviour would consider an Article titled "[USER: title; working title: Beyond matching answers: testing whether vision-language models use facial cues as people do]".

**The question.** Several reports say that vision-language models match people at reading emotion and mental states from faces. In the work we located, such claims rest on matching answers. Matching answers cannot settle whether a model reads faces as people do, for two reasons that we document and one that we prove.

**What the manuscript shows.**
1. **Accuracy can miss the face entirely.** Our own earlier design let option position and option sets give the answer away, and we withdrew it. None of eight recent published designs measured the accuracy lost when the face is swapped. With option order rotated and the prompt fixed, swapping the face cut three open models' accuracy by 7 to 40 points, although one never exceeded chance.
2. **Agreement on natural faces cannot identify cue use.** We show formally that even identical answers on every natural face are compatible with any pattern of cue effects. In our data, all three models aligned with the human response pattern on natural clips, including the one at chance.
3. **A test that can.** We compare each model's effects of randomised facial action units with human effects from reverse-correlation data, against a noise ceiling and the agreement between two human observer groups. Gemma 4's cue effects aligned with people's for basic emotions in both groups and again in a second human sample from the same laboratory, though less closely than the human groups agree with each other. Qwen3-VL's were inconclusive, or below our bound. We located no earlier comparison of a vision-language model with randomised human cue effects.

**Why it matters for the journal's readers.** Whether AI systems perceive social signals as people do is a question about cues, not answers. The manuscript gives a test that can answer it, and a short list of checks that any answer-level comparison should report.

**Status.** Every confirmatory analysis was fixed in a dated amendment before its data were analysed. Inconclusive and negative results are reported, and code and derived outputs are public with a reproduction script. A preregistered extension to two larger open models is under way and will be complete before submission. The main text is about 4,900 words, with six figures and one table.

[USER: one or two sentences on fit with the journal, in your words.]

We would be grateful to know whether you would like to see the full manuscript. The abstract follows.

Yours sincerely,

[USER: corresponding author name, affiliation and email]

---

**Abstract**

[Paste the current abstract from `research/paper/v2/abstract.md` (150 words) when sending, so that the two cannot differ.]
