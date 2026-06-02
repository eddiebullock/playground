## Manuscript change set (patch-ready)

This file contains patch-ready replacement text to update the manuscript based on reviewer feedback:

- rerun/position results using **audio+video** for EU-Emotions and Mindreading where applicable
- add the **new secondary research question** (basic vs complex; high vs low intensity)
- **remove autistic benchmark comparisons**
- **replace non-autistic benchmark** with the appropriate **EU human benchmarks** from:
  - O’Reilly et al., *The EU-Emotion Stimulus Set: A validation study*
  - Lassalle et al., *The EU-Emotion Voice Database*
- clarify foil replication limits
- address modality ablations (audio-only, video-only, audio+video) and spoken-text confounds

Where numeric benchmark values are required, placeholders are marked as `[[FILL]]`.

---

## Abstract (replacement)

Large language models (LLMs) demonstrate impressive performance on text-based theory-of-mind (ToM) tasks, but comparisons of LLM and human performance on dynamic, multimodal ToM batteries remain sparse. We evaluated five state-of-the-art LLMs (GPT-5, GPT-5 Mini, Gemini 3 Pro, Gemini 3 Flash, and Claude Opus 4.5) on two validated mental-state recognition batteries: EU-Emotions (27 mental states including intensity variants) and Mindreading (359 mental states). Trials used a four-alternative forced-choice design in which models selected the target mental-state label from four candidates for each stimulus. For models supporting audio input, we evaluated performance under video-only, audio-only, and combined audio+video conditions.

Across batteries, all models performed above chance. On EU-Emotions, Gemini models exhibited improved performance when audio was included relative to video-only, and modality ablations enabled direct comparison to human benchmark performance reported in validation work on the EU-Emotions video and voice stimulus sets (O’Reilly et al.; Lassalle et al.). On Mindreading, performance depended strongly on input modality; audio-inclusive conditions can be vulnerable to spoken-label confounds, which we quantify via modality ablations and interpret cautiously. Per-mental-state analyses further indicated that recognition accuracy varies systematically across mental states, with lower accuracy for subtle or low-intensity variants than for higher-intensity counterparts.

---

## Aims / research questions (additions and edits)

### Primary research question (retain)
Can LLMs classify complex mental states at accuracy levels approaching or even exceeding the performance of human participants on validated mental-state recognition batteries?

### Secondary research question (new; add)
Are specific mental states more reliably detected across models? Specifically: (i) are basic mental states more accurately identified than complex mental states, and (ii) are high-intensity mental states more accurately identified than their low-intensity counterparts?

Operationalisation: “low intensity” variants are those explicitly labelled as such in EU-Emotions; “high intensity” refers to the corresponding base label without the low-intensity qualifier. “Basic vs complex” mental states are defined by an explicit mapping (see Analysis section / Appendix `[[FILL: appendix label]]`).

---

## Methods: Materials (replacement paragraph(s))

### EU-Emotions stimulus set (update)
The EU-Emotions battery (O’Reilly et al.; Lassalle et al.) comprises professionally recorded stimuli portraying mental states via facial expression and body language (video) and via vocal prosody (voice recordings). For the present study, we evaluated model performance under three input conditions where supported: video-only, audio-only, and combined audio+video. This design enables direct comparison to modality-matched human benchmark performance reported in the EU-Emotions validation literature.

### Mindreading (update)
Mindreading stimuli consist of short face video clips accompanied by associated audio. For models supporting audio input, we evaluated video-only, audio-only, and combined audio+video conditions. Because audio recordings may contain spoken content that can act as a trivial cue to the label, we explicitly assessed the impact of modality by comparing performance across these conditions and interpret audio-inclusive results with caution where they approach ceiling performance.

---

## Methods: Trial structure / foils (replacement paragraph)

Following prior work using four-alternative forced-choice paradigms, each trial presented one stimulus and four candidate mental-state labels (the correct label and three foils). Foils were sampled from the full label inventory excluding the correct label, with deterministic seeding to ensure reproducibility. Exact replication of the item-level foil composition employed by Golan et al. (2006) was not possible because the original item-level foil composition was not publicly available. Additionally, corrupt or undecodable Mindreading files prevented identical replication of the original trial set; all exclusions and the resulting evaluation sample are reported.

---

## Results: Benchmarking (rewrite guidance)

1) Remove any sentences of the form:
- “significantly exceeded the autistic benchmark …”
- “met the autistic benchmark …”
- any z-tests/effect sizes vs autistic-group values

2) Replace “non-autistic benchmark” comparisons with modality-matched EU human benchmarks:
- EU video-only benchmark: `[[FILL: accuracy, n, citation details]]`
- EU audio-only benchmark: `[[FILL: accuracy, n, citation details]]`
- EU audio+video benchmark (if available in the cited studies): `[[FILL]]`

3) If Mindreading human benchmarks are retained, ensure they are justified by an appropriate Mindreading human study; otherwise remove those direct human comparisons and keep only within-model/per-state analyses plus chance tests.

---

## Results: Per-mental-state and intensity (add)

Neutral is not a mental state and was excluded from per-mental-state summaries and aggregate “mental state recognition” results. For EU-Emotions, we additionally compared accuracy on low-intensity variants (labels containing “low intensity”) to their corresponding base labels, reporting per-model mean accuracy for high-intensity and low-intensity states and the mean difference (high minus low).

---

## Discussion / limitations (additions)

### Modality dependence and spoken-label confounds (add)
Performance differed systematically across input modalities. Audio-only and audio+video conditions can provide additional prosodic information that improves recognition; however, audio recordings may also contain spoken content that is directly diagnostic of the target label. To evaluate this risk, we performed modality ablations (video-only vs audio-only vs audio+video) on identical trial sets. Where audio-inclusive performance approaches ceiling levels, results are interpreted as consistent with a spoken-label confound rather than as evidence of improved ToM ability.

---

## Checklist to apply

- Remove autistic benchmark comparisons throughout Abstract/Results/Figures.
- Replace benchmark framing with EU-Emotions benchmarks (O’Reilly; Lassalle), modality-matched.
- Add secondary question text and operational definitions.
- Add intensity analysis text; exclude Neutral.
- Insert foil replication statement above.

