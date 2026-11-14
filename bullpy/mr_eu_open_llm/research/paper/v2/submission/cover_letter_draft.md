<!--
Draft, 2026-11-08 (ROADMAP step 0.9); results filled on 2026-11-14 from the locked outputs (A8 has not run). Claims are limited to what is locked in RESULTS_REGISTRY or proved in Supplementary Note 1.
Submission and authorship belong to the user. Nothing here has been sent.
-->

[USER: date]

[USER: handling editor's name, if known], Editor
Nature Human Behaviour

Dear [USER: editor],

We submit "[USER: title; a working candidate is in outline.md]" for consideration as an Article in Nature Human Behaviour.

Some vision-language models are reported to match people at reading mental states from faces. We ask how similar these models are to humans in how they read faces, not only in the answers they give. The manuscript makes three contributions.

First, a validity audit. Our own earlier task placed the correct option first on all 243 items and gave each of 21 labels one option set, so its accuracy could not be told apart from position or prompt text. We withdrew it. In a spot check of eight recent papers, several of the claims we located cannot rule out option order or prompt text. We then measured how much accuracy each model loses when the face is swapped under a fixed prompt, with option position rotated. Swapping the face cost all three models accuracy, by 7 to 40 points, although one model stayed at chance on the right face.

Second, an identifiability result. We prove that agreement on natural items does not in general determine which facial cues a system uses: two systems can agree on every natural item and still differ in every cue effect. We define Cue Effect Alignment, a comparison of a model's cue effects with human effects estimated under randomised cue presentation.

Third, a cue-level test. We compare model cue effects on rendered faces with randomised human cue effects from public data, against a noise ceiling and a between-group reference. On these renders, Gemma 4's cue effects aligned with human effects on basic labels in both observer groups, and again on a second human sample from the same lab and generator, though below the agreement between the two human groups; on conversational states the result was mixed, and Qwen3-VL's test did not reach a decision, because its answers failed validity gates on three of four datasets. We also test preregistered human predictions under stimulus interventions. On basic labels, masks and frequency filters on single faces moved the models' option-free judgements as the preregistered predictions state on five of ten contrasts for Gemma 4 and four for Qwen3-VL, while shuffled frames and clips cut to a third matched no prediction.

Every confirmatory analysis was fixed in a dated amendment or decision-log entry before its data were analysed, and the rules of the cue-level and intervention tests can each return a match, a mismatch or an inconclusive result. Amendments, frozen hashes and every dated deviation are listed in the manuscript. The paper is written to stand whatever the results are, and nulls are reported as results. Code and derived outputs are public, and a reproduction script rebuilds every output that needs only public inputs and checks the rest against pinned hashes.

[USER: why this journal and its readership; one or two sentences.]

[USER: statement that the manuscript is not under consideration elsewhere, and any preprint.]

[USER: competing interests.]

[USER: suggested and excluded reviewers, if any.]

We would also ask the editors one question. The amendments are preregistered as tagged commits in a public repository. Please tell us whether this meets the journal's preregistration requirement, or whether we should deposit them on a registry.

Yours sincerely,

[USER: corresponding author name, affiliation and email]
[USER: co-author names and affiliations]
