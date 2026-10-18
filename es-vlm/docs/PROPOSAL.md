# Empathizing and Systemizing Inside Vision-Language Models

**Date:** 18 October 2026
**Status:** working proposal (implementation notes in `docs/DECISIONS.md`)

Do not paste licensed questionnaire items, human item-level data, or
site-specific credentials into this file.

Implementation already differs from the original hardware/panel sketch
in two places: Molmo2 is out of the confirmatory panel (processor
failure during setup; no E/S results), and InternVL3.5 is optional /
throwaway. Gemma 4 E4B is the non-Qwen VLM. See `docs/DECISIONS.md`.

## Summary

This project tests whether vision-language models (VLMs) contain
separable, causally functional representations of empathizing and
systemizing, and whether their relationship mirrors the structure that
empathizing–systemizing (E-S) theory proposes for humans.

We extract candidate empathizing (E) and systemizing (S) directions from
the activations of open-weight VLMs. We then steer along them while
measuring visual mental-state inference (Reading the Mind in the Eyes,
Mind Reading clips, social video) and physical or mechanical reasoning.
The core test is a double dissociation: E steering should move social
perception but not physical reasoning, and S steering the reverse.

Two further analyses use what is distinctive about the Autism Research
Centre. First, we test whether models' self-reported EQ/SQ shifts track
real changes in competence. Second, we compare model error patterns with
human error patterns stratified by EQ and SQ.

The quotients generate predictions. They are not scores to give the
model. That avoids the well-documented validity problems of
administering human questionnaires to LLMs. It also gives two
contributions: one to ML, on the geometry of persona and
social-reasoning representations, and one to psychology, on whether E-S
structure emerges from learning on human data.

## Background and gap

Two obvious studies already exist. The open gap is E-S structure inside
models.

- Questionnaires given to LLMs are not valid trait measures. Likert
  self-report profiles diverge substantially from generation-based
  profiles, and item consistency disappears in actual generations (Song
  et al.). Latent-variable analyses suggest the human constructs may not
  exist in LLMs at all (Peereboom et al.). EQ-60 and SQ-R scores for
  commercial models already circulate in grey literature.
- VLMs on the Eyes test are already measured. GPT-4o beats humans on
  upright RMET and MRMET faces, underperforms on inverted faces, and is
  more accurate for White faces. Its errors are consistent but
  structured differently from humans' (Strachan et al.). Block-scrambled
  RMET images barely hurt GPT-4V, which points to local-feature
  shortcuts (Murray and Boynton, VSS 2024).
- Mechanistic ToM work is text-only and focused on beliefs. Belief
  representations can be probed, improve with scale and fine-tuning, and
  steering improves ToM (Bortoletto et al.; CoSToM). Emotion directions
  in VLMs share structure across architectures.
- Persona vectors provide the method and an open question. Traits can be
  extracted as linear directions from a description. The authors
  explicitly leave open whether correlations between persona vectors
  predict co-expression of traits (Chen et al.). E–S is a natural test
  case.
- VLMs are weak at social interaction. Benchmarks find relation
  inference and multi-step mental-state reasoning are bottlenecks
  (SIV-Bench; R³-Bench; MoMentS).

**Gap.** We found no work that tests E-S theory's structural claims
inside a model. We found none that does mechanistic analysis of complex
mental-state reading, as opposed to basic emotions, in VLMs. And we
found none that uses EQ/SQ-stratified human item-level data as the
reference distribution.

## Research questions and hypotheses

**Main question:** Do VLMs contain separable, causally functional
empathizing and systemizing representations, and does their relationship
reproduce the structure E-S theory proposes for humans?

| RQ | Question | Hypothesis (H1) | What would falsify it |
| --- | --- | --- | --- |
| RQ1 Separability | Are E and S distinct linear directions, and distinct from generic emotion, valence and helpfulness directions? | E and S separate from each other beyond a matched-random null. Across simulated personas, their projections correlate weakly and negatively, as the EQ and SQ-R do in humans. | E is collinear with S, or with valence or helpfulness; or the persona-level correlation differs strongly from the human estimate. |
| RQ2 Double dissociation (core) | Does steering along E change visual mental-state inference but not physical reasoning, and S the reverse? | Direction x task-domain interaction. E has a larger effect on RMET, MRMET, Mind Reading and social video; S has a larger effect on physical and mechanical VQA. | No interaction (both directions are generic "try harder" signals), or effects only on text-answerable items. |
| RQ3 Self-report vs competence | Does steering E move EQ answers more than RMET accuracy? | The questionnaire slope is larger than the competence slope: models talk the talk more than they walk the walk. | The slopes are matched, which would imply the questionnaire tracks competence in models. |
| RQ4 Human alignment | Under E-down steering, do item-level errors resemble those of low-EQ humans? | Partial alignment on upright items and divergence on inverted or scrambled items, extending the Strachan et al. finding. | Errors are unrelated to human EQ strata at every steering level. |

Each outcome is informative: a trade-off, independence, or entanglement
all bear on whether E-S structure emerges from training on human data. A
secondary check compares each VLM with its text-only language backbone
where available. This tests whether E and S directions exist before
multimodal training.

Always say E-direction / S-direction. Never label a model or condition
"autistic" or "low-empathy".

## Models

Steering requires open weights. The intended panel covers open VLM
lineages with different vision encoders and language backbones, plus a
scale ladder within one family. Closed frontier models serve only as
behavioural reference points.

| Role | Model | Why | Where it runs |
| --- | --- | --- | --- |
| Primary, scale ladder | Qwen3-VL 2B / 8B / 32B (dense, instruct) | Strongest all-round open VLM family in 2026, Apache 2.0, good video support. Three sizes test whether E and S emerge with scale. | 2B locally; 8B and 32B on the cluster |
| Replication anchor | Qwen2.5-VL-7B-Instruct | Most-used model in published VLM steering work, so existing methods and code can be checked | Cluster |
| Older Qwen VLM | Qwen2-VL-7B-Instruct | Already on disk; not a primary lineage but a cheap extra Qwen-VL point | Cluster |
| Different lineage | Gemma 4 E4B instruct | Independent training pipeline. Tests whether directions survive a non-Qwen stack. | Cluster |
| Text-only control | Qwen3-8B (Instruct id was HTTP 401 at pin time) | Tests whether E and S directions exist before vision training | Cluster |
| Optional / excluded | InternVL3.5-8B; Molmo2-O-7B | InternVL is throwaway. Molmo2 dropped during setup (processor API), before any E/S result. | Not confirmatory |
| Behavioural reference only | GPT-4o and current frontier APIs | Replicates Strachan et al. and gives a ceiling. No steering. | API |

Where base checkpoints exist, base and instruct models are compared,
since belief representations change with fine-tuning (Bortoletto et
al.). Exact checkpoints are pinned by Hugging Face revision hash. See
`docs/DECISIONS.md`.

## Stimuli and tasks

Vectors are built from text. They are then tested on visual tasks whose
answers cannot be read off the prompt, which guards against an E vector
that only means "talk about feelings".

| Set | Purpose | Content | Notes |
| --- | --- | --- | --- |
| Vector construction | Extract E and S directions | Contrastive statement pairs derived from EQ-60 and SQ-R item content; persona-vector-style trait descriptions; held-out non-questionnaire narratives | Raw items stay in a private data folder |
| E-tasks (visual) | Social perception | RMET (36 items), Multiracial RMET, Mind Reading library clips, a social-video subset from SIV-Bench / R³-Bench / MoMentS | RMET and Mind Reading are licensed research materials |
| S-tasks (visual) | Systemizing | Intuitive-physics and mechanical-reasoning VQA | Default recommendation: PhysBench dynamics + CLEVRER MCQ; see `docs/OPEN_QUESTIONS.md` |
| Controls | Specificity and shortcut checks | Basic-emotion facial expression recognition; generic VQA (steering damage); inverted and block-scrambled RMET; newly photographed or generated eye-region items | Scrambling and inversion follow VSS 2024 and Strachan et al. |
| Self-report | RQ3 | EQ-60 and SQ-R scored by option log-probabilities, plus generation-based scenario responses | Follows Song et al.'s generation-based profiling |
| Human reference | RQ4 | Item-level RMET and related task responses with EQ/SQ-R scores | Stays on university storage, never in the repo |

Scoring uses log-probabilities over the answer options wherever
possible, not free generation. This is cheaper, deterministic and
comparable across steering strengths. Every task is run with three or
more prompt templates so that prompt sensitivity can be estimated.

## Methods

The pipeline runs in five stages, each with a checkpoint before the
next.

1. **Baseline behaviour.** Run every model, unsteered, on all E-tasks,
   S-tasks and controls. Replicate the GPT-4o RMET result (accuracy,
   inversion effect, White vs non-White faces) as a sanity check.
2. **Vector extraction (RQ1).** For each model and decoder layer, take
   the residual-stream activation at the final prompt token. Compute
   difference-of-means vectors for E+ vs E− and S+ vs S− contrast sets,
   and compare them with persona-vector-style extraction.
   - Validate each vector by projection: held-out E+ texts should
     project higher than E− texts. Pick candidate layers by held-out
     separation.
   - Separability: cosine similarity and principal angles between E and
     S per layer. Also compare against emotion, valence and "helpful
     assistant" vectors. The null is random contrast sets matched on
     length and topic.
   - Persona population: prompt 500 or more simulated respondents with
     varied backgrounds and project each onto E and S. Compare the E–S
     correlation with the human estimate (weak negative).
3. **Steering (RQ2).** Add `α · ‖h̄‖ · v̂` to the residual stream at
   the chosen layers. Sweep α over about seven values, both signs. Also
   run directional ablation (projecting `v` out) to test necessity, not
   just sufficiency.
   - Analysis: a mixed-effects logistic regression of item accuracy on
     direction × domain × α, with random intercepts for item and prompt
     template. The double dissociation is the interaction term.
   - Guardrails: report generic-VQA accuracy and output perplexity at
     every α so that "breaking the model" is not confused with
     specificity.
4. **Self-report vs competence (RQ3).** At each α, score EQ and SQ-R by
   log-probability and score RMET accuracy. Compare standardised slopes
   (questionnaire vs task) with a bootstrap confidence interval.
5. **Human alignment (RQ4).** Compute item-level accuracy and confusion
   over foil options for models at each α, and for human participants
   split into EQ and SQ-R tertiles. Compare them with representational
   similarity analysis and Jensen–Shannon divergence of the error
   distributions, following Strachan et al.

**Rigour.** Hypotheses, layer-selection rule and analysis plan are
preregistered on OSF before steering runs. Multiple prompt templates and
seeds are used. All checkpoints are pinned by revision hash. Results
across the model panel are reported in full, including nulls.

## Compute and infrastructure

Development happens on a Mac. Anything at 30B parameters or more, and
the full steering sweeps, run on the university cluster (A100 80 GB).

| Work | Where | Why |
| --- | --- | --- |
| Code, tests, small-model debugging (2B–8B, bf16, few items) | Mac (Apple Silicon, MPS) | Fast iteration with no queue |
| Vector extraction for 2B–8B models | Mac or one A100 | Forward passes only |
| 32B models, full steering sweeps, video tasks | Cluster GPU partition | A 32B model in bf16 fits on one 80 GB A100 |
| API reference models | Laptop | Lightweight HTTP calls |
| Human data and RQ4 analysis | University storage | Data-governance requirement; never synced to GitHub |

Rough budget. The panel has about six models. A sweep of 8 layers × 7 α
values × 2 directions × 2 signs, run over roughly 1,500 items scored by
log-probability, comes to a few hundred thousand short forward passes.
That is on the order of 100–300 A100 GPU-hours, plus the video tasks.
Check against Phase 1 timings.

Code stack. PyTorch, Hugging Face Transformers, and nnsight (or plain
forward hooks) for activation reading and steering. Hydra configs,
results in Parquet, and uv for environments. The same code runs on the
Mac and on the cluster, with device chosen by config.

Repository. `es-vlm/` inside `eddiebullock/playground`: `configs/`,
`src/esvlm/`, `scripts/`, `slurm/`, `tests/`, `docs/`, and a git-ignored
`data/` folder. Licensed stimuli and human data are never committed.

## Ethics, licensing and data governance

- **Framing.** The project studies cognitive dimensions in models. It
  never describes a model, or a steering condition, as "autistic" or
  "low-empathy".
- **Stimuli licences.** RMET, the Mind Reading library and EQ/SQ-R items
  are distributed for research use. They stay in the git-ignored `data/`
  folder, never committed or published, and figures show only items
  cleared for publication.
- **Repository visibility.** Only code, configs and aggregate results go
  to GitHub. Check before pushing that no stimuli, item text or human
  data are in any commit.
- **Human data.** Participant data stay on university storage under the
  existing ethics approval and consent. Only aggregate or appropriately
  anonymised item-level statistics leave it. Check whether secondary use
  for model comparison needs an amendment.
- **Fairness.** Report accuracy by face ethnicity and gender for every
  model and steering condition.
- **Dual use.** Steering social perception up or down could in principle
  be used to make manipulative agents. The paper reports methods and
  findings but does not release tuned high-empathy persuasion vectors
  for deployment.

## Timeline and milestones

The plan runs for 12 months from November 2026. Dates are a proposal to
agree with the supervisor.

The Phase 2 gate is the key decision. If no E or S direction beats the
null, the project pivots to the behavioural fallback: RQ4 on unsteered
models, as a standalone trait-stratified benchmark paper.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| The E vector just means "emotional words" or "be nice" | Compare it with valence and helpfulness vectors; test only on visual tasks scored by options; use held-out narratives that are not questionnaire-like |
| Steering breaks the model rather than shifting a skill | Report generic-VQA accuracy and perplexity at every α; use ablation alongside addition; preregister the α range |
| RMET contamination, or local-feature shortcuts | Inverted, scrambled and newly made eye items; prefer Mind Reading clips and video, which are less exposed |
| Prompt sensitivity | Three or more templates per task; template as a random effect |
| No clean S benchmark | PhysBench dynamics + CLEVRER MCQ as the default public pair; pilot ceiling/floor before Phase 1 lock |
| Cluster queue time or allocation | Develop on the Mac; batch log-probability scoring |
| Licence or ethics delay for human data | Phases 1–3 do not need human data; start the request in Phase 0 |
| Null result on RQ2 | Still publishable: it shows that E-S structure does not emerge in current VLMs. RQ4 is the fallback. |

## Expected contributions and outputs

- **ML:** the first test of whether two theory-derived cognitive
  dimensions are separable and causally specific inside VLMs. This
  addresses the open question about persona-vector correlations and
  co-expression.
- **Psychology:** a computational test of E-S theory's structure, asking
  whether it emerges from learning on human data without biology.
- **Methods:** a mechanistic demonstration of the gap between
  questionnaire scores and competence in models.
- **Resource:** an open evaluation harness for complex visual
  mental-state inference with trait-stratified human norms. Code and
  aggregate norms are released; licensed stimuli are not.
- **Outputs:** one main paper, an OSF preregistration, a thesis chapter,
  and a public code repository.

## Sources

- Strachan et al., GPT-4o reads the mind in the eyes
- Murray and Boynton, Reading minds in the eyes with GPT4-vision (VSS 2024)
- Song et al., Human Psychometric Questionnaires Mischaracterize LLM Behavior
- Peereboom et al., Cognitive phantoms in LLMs
- Chen et al., Persona Vectors
- Bortoletto et al., Benchmarking Mental State Representations in Language Models
- Li et al., CoSToM
- Emotion-concept generalisation across sources, modalities, and architectures in VLMs
- SIV-Bench, R³-Bench, MoMentS
- Empathizing–Systemizing theory overview (EQ and SQ-R correlation)
- Baron-Cohen, Wheelwright, Spong, Scahill and Lawson (2001), folk physics / folk psychology
- Baron-Cohen, Leslie and Frith (1986), mechanical vs intentional picture sequencing
- PhysBench (Chow et al., ICLR 2025); CLEVRER (Yi et al.); Physion
- CSD3 / cluster hardware notes
