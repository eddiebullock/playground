<!--
Draft v2 skeleton, 2026-11-08 (ROADMAP step 0.8). No claim below depends on a result that is not locked.
Length. Nature Human Behaviour Articles allow "up to 150 words, unreferenced" (content-types page, read 2026-11-08; see submission/reporting_summary.md). Count with research/scripts/manuscript_words.py.

2026-11-21, reframe (the author's instruction of 2026-11-21; research/paper/v2/IMPACT_PLAN.md; DECISIONS 2026-11-21). The abstract now leads with the question the paper answers (answers against cues), gives the cue-level result its size relative to human agreement, and closes on the take-home. Changes against the version at bb3f90b, each checked against the locked text:
- Our earlier design and the published designs share one verb, "could not rule out", so the abstract no longer implies they share our defect (Discussion, benchmark audit: "our confound is not shown to be common").
- "one stayed at chance" became "one never exceeded chance": Molmo2 was at chance on the composite input (.163) and below it on the single crop (.072).
- "on rendered faces" became "faces rendered from randomised facial-action recipes people had judged": the models saw the same AU configurations, not the stimuli people saw (Methods, "Stimuli").
- Qwen3-VL's second-sample result, not aligned on these renders, is now stated beside its inconclusive first test.
- "We prove" became "We show formally": the results are elementary (Supplementary Note 1, Section 5).
- "preregistered" is left out until the editors say whether tagged commits count (cover letter; Methods, "Preregistration and order of work").
- The stimulus interventions leave the abstract for length; they are in Results, Section 6.
- The closing sentence no longer follows outline.md's scenario S6 ("Models differ in which degree of human similarity they reach"). That table was a drafting aid, not part of any amendment, and the editor-style read of 2026-11-21 found S6's sentence inaccurate: the randomised test approximates the second degree and does not meet it as framed (Methods), and an Inconclusive is not a degree.
Locked facts used: RESULTS_REGISTRY, "A1 rotated 6AFC" (deltas .071 to .402; Molmo2 .163), "A3 face input validity" (Molmo2 .072), A4 table headlines and "WC to EA reference, residual", A9 table and "A9 human references"; the identifiability statement rests on Supplementary Note 1 and the theory note, not on data.
-->

# Title

Beyond matching answers: testing whether vision-language models use facial cues as people do

<!-- Working title, 2026-11-21. The title is the author's choice. Alternatives, each true of the locked results:
1. "A vision-language model partly shares people's facial cue effects for basic emotions" (leads with the finding; one model, basic emotions, "partly").
2. "Matching people's answers cannot show that vision-language models read faces as people do" (leads with the thesis).
3. "Answer agreement is not cue agreement when vision-language models judge faces" (the editor-style read's suggestion; the descriptive behaviour gap in Results, Section 5 has no noise ceiling, so this one leans hardest on the theory).
-->

# Abstract

Some vision-language models reportedly match people at reading emotion from faces, but matching answers cannot show that they read faces as people do. Several published designs, and our earlier one, could not rule out that option order or prompt text supplied the answer. With option order rotated and the prompt fixed, swapping the face cut three small open models' accuracy by 7 to 40 points, although one never exceeded chance. We show formally that even perfect agreement on natural faces cannot, in general, identify which cues a system uses. We measured each model's responses to faces rendered from randomised facial-action recipes people had judged. Gemma 4's cue effects aligned with people's for basic emotions, in two observer groups and a second sample, but less closely than human groups agree (mixed for conversational signals); Qwen3-VL's were inconclusive or below our bound. Human-like face reading must be tested on cues, not answers.
