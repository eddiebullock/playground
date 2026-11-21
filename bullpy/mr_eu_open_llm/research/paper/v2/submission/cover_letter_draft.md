<!--
Draft, 2026-11-08 (ROADMAP step 0.9); results filled on 2026-11-14 from the locked outputs (A8 has not run). Claims are limited to what is locked in RESULTS_REGISTRY or proved in Supplementary Note 1.
Submission and authorship belong to the user. Nothing here has been sent.
2026-11-21, reframed with the paper (IMPACT_PLAN.md; DECISIONS 2026-11-21). Changes against bb3f90b, from the editor-style read of 2026-11-21: the stimulus edits are stated with their robust part, not as "five of ten"; "matched no prediction" now says the contrasts were inconclusive, one against its prediction; "written to stand whatever the results are" is gone, since it reads as a paper with no finding; "first" is not claimed (outline.md, inconsistency 1: "located no earlier example"). The why-this-journal sentence stays the author's; IMPACT_PLAN.md suggests one, with a precedent still to verify.
-->

[USER: date]

[USER: handling editor's name, if known], Editor
Nature Human Behaviour

Dear [USER: editor],

We submit "[USER: title; working title: Beyond matching answers: testing whether vision-language models use facial cues as people do]" for consideration as an Article in Nature Human Behaviour.

Some vision-language models are reported to match people at reading emotion and mental states from faces. In the work we located, such claims rest on matching answers. The manuscript shows why matching answers cannot settle whether a model reads faces as people do, and tests what can.

First, accuracy may not measure face reading at all. Our own earlier task placed the correct option first on all 243 items and gave each of 21 labels its own option set, so we withdrew it, and in a spot check of eight recent papers none estimated how much accuracy is lost when the face is swapped under a fixed prompt. With option order rotated and the prompt fixed, swapping the face cut three small open models' accuracy by 7 to 40 points, although one never exceeded chance.

Second, even perfect agreement on natural faces cannot identify which facial cues a system uses, because facial movements co-occur. We show this formally. On natural clips, all three models aligned with the human response pattern, including the one whose accuracy never exceeded chance.

Third, a cue-level test. We compare each model's effects of randomised facial action units with human effects estimated from public reverse-correlation data, against a noise ceiling and the agreement between two human observer groups. Gemma 4's cue effects aligned with people's for basic emotions in both observer groups and again in a second human sample from the same laboratory and face generator, though less closely than the two human groups agree with each other; for conversational signals the result was mixed. Qwen3-VL's test did not reach a decision, and on the second sample its cue effects were above chance but below our bound. We located no earlier comparison of a vision-language model with randomised human cue effects. Editing the clips moved the models' judgements as human studies predict for some cues: masking the mouth on happy faces did so for both models, and masking the brow on angry faces for Qwen3-VL, under every sensitivity analysis, while the other single-face agreements depended on which items were included, and the temporal edits were inconclusive, one moving against its prediction.

The result matters beyond these models. Whether an AI system perceives social signals as people do is a question about cues, and the manuscript gives a test that can answer it, with a short list of checks that any answer-level comparison should report.

Every confirmatory analysis was fixed in a dated amendment or decision-log entry before its data were analysed, with decision rules that can return a match, a mismatch or an inconclusive result. Amendments, frozen hashes and every dated deviation are listed in the manuscript, and inconclusive and negative results are reported in full. Code and derived outputs are public, and a reproduction script rebuilds every output that needs only public inputs and checks the rest against pinned hashes.

[USER: why this journal and its readership; one or two sentences.]

[USER: statement that the manuscript is not under consideration elsewhere, and any preprint.]

[USER: competing interests.]

[USER: suggested and excluded reviewers, if any.]

We would also ask the editors one question. The amendments are preregistered as tagged commits in a public repository. Please tell us whether this meets the journal's preregistration requirement, or whether we should deposit them on a registry.

Yours sincerely,

[USER: corresponding author name, affiliation and email]
[USER: co-author names and affiliations]
