# Decisions

Dated log. Conventional commits should point here when a choice is
non-obvious.

## 2026-10-18 — Layer sweep, not a single early layer

Prior pilots in this PhD steered at **one early layer** (Qwen3-VL layer 4)
along **emotion-confusion / diagnosticity axes**, not E/S contrasts:

- EU-Emotions pilot, job 33776847: null.
- RMET study, job 33536702: near-null / non-specific (~84% zero ΔJS;
  largest mean shift on the random axis).

Those results motivate this project. We extract E-direction and
S-direction from contrastive text, then **sweep many layers**. We do not
re-use emotion-confusion axes at one early layer, and we do not treat
those nulls as evidence that no E/S mechanism exists anywhere.

## 2026-10-18 — Two Transformers stacks

Gemma 4 needs Transformers 5.x. Molmo2's processor uses
`optional_attributes`, which 5.x dropped. InternVL2-8B failed to load on
Transformers 5.10 and is out of the panel. Profiles:

- `env-main`: Transformers 5.x
- `env-molmo`: Transformers 4.57

One model per process, one model per GPU job.

## 2026-10-18 — Pinned Hugging Face revisions

Verified via the Hugging Face models API on 2026-10-18 (not invented):

| Key | HF id | Revision | Env | Notes |
| --- | --- | --- | --- | --- |
| qwen3vl_8b | `Qwen/Qwen3-VL-8B-Instruct` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | main | On cluster; primary |
| qwen3vl_2b | `Qwen/Qwen3-VL-2B-Instruct` | `89644892e4d85e24eaac8bacfd4f463576704203` | main | Mac smoke / scale ladder |
| qwen3vl_32b | `Qwen/Qwen3-VL-32B-Instruct` | `0cfaf48183f594c314753d30a4c4974bc75f3ccb` | main | Fits one A100 in bf16 |
| gemma4_e4b | `google/gemma-4-E4B-it` | `ee0ef6023621cff504d758262d4e04895a5af4a2` | main | On cluster |
| molmo2 | `allenai/Molmo2-O-7B` | `784410650d12be9bc086118fdefa32d2c3bced86` | molmo | Open data; contamination check |
| qwen25vl_7b | `Qwen/Qwen2.5-VL-7B-Instruct` | `cc594898137f460bfe9f0759e9844b3ce807cfb5` | main | Published-steering replication |
| qwen3_text | `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` | main | Text-only control; matching 8B backbone |
| internvl35_8b | `OpenGVLab/InternVL3_5-8B-HF` | `741a7d03020411e666c6109218ab71e08151ef86` | throwaway | Optional; drop after two load failures |
| qwen2vl_7b | `Qwen/Qwen2-VL-7B-Instruct` | `eed13092ef92e448dd6875b2a00151bd3f7db0ac` | main | On cluster; not a primary panel model |

`Qwen/Qwen3-8B-Instruct` returned HTTP 401 from the public API on this
date, so the text-only control is pinned to the public `Qwen/Qwen3-8B`
checkpoint (same 8B backbone family). Re-check Instruct if access
appears later.

## 2026-10-18 — huggingface_hub pin relaxed

The brief asked for `huggingface_hub>=0.34,<1.0`. Current Transformers 5.x
requires `huggingface_hub>=1.3` (5.18+ wants >=1.31). The lock cannot
satisfy both. Pinned `huggingface_hub>=0.34,<2.0` so env-main can exist.
Revisit if a future 5.x wheel returns to hub 0.x.

## 2026-10-18 — Mac smoke (Milestone B)

Qwen3-VL-2B-Instruct (`89644892e4d85e24eaac8bacfd4f463576704203`) on
this Mac: MPS bf16, 12 synthetic items, layers 0 and 14, alphas
{-1, 0, 1}, 72 scored rows. Load 22.8 s (warm cache), score 109.9 s
(9.2 s/item), 27 option-tokens/item, peak allocated 4073 MB. Summary:
`results/aggregate/smoke_mac_qwen3vl_2b.json`.

## 2026-10-18 — Molmo2 dropped from the primary panel

Dropped **before Phase 1**, because of loader/processor incompatibility
(text-only residual extract). Not dropped because of an E/S finding:
there are no Molmo E/S results. `env-molmo` stays on disk.

## 2026-10-18 — S-domain defaults

Locked public S-sets (never mixed with licensed stimuli trees):

1. PhysBench dynamics / relationships, text A-D options only. Official
   `test.json` has no answers; labels from `lmms-lab-eval/PhysBench`.
   This is the primary confirmatory visual S-set. One S-set is enough
   for the direction x domain interaction; a benchmark battery is not.
2. CLEVRER validation explanatory / predictive / counterfactual MCQ is
   optional robustness / video S, not required. Do not run the full
   19804 or the full model panel unless PhysBench is unusable (floor).
3. PIQA validation is an optional text-only backbone control, not a
   confirmatory S-task.

Rejected as the main visual S-set: 2001 Folk Physics Test; ScienceQA /
AI2 ARC. Fetch via `slurm/fetch_sdomain.sbatch` into `$ESVLM_PUBLIC_DIR`.
CLEVRER videos use `$ESVLM_CLEVRER_DIR` when the user RDS file cap is tight.

8B unsteered PhysBench accuracy was 0.2635 vs 0.25 chance (path check,
not a finding). If later panel models are also at floor, use the
already-running CLEVRER shard as the fallback S-readout instead of
fetching more benchmarks.

## 2026-10-18 — E-task scoring wired

Public S path is confirmed, so `esvlm-score` loads licensed E-sets in
place: RMET (36, 4-AFC) and EU-Emotions full (243, 6-way from the
entropy option set; never the 118-trial subset). Mind Reading jsonl has
labels but no forced-choice options; it is not scored until a MCQ form
exists. No stimuli are copied into the repo.

Schema work that was not just “read the manifest”:

- RMET image fields are stale multi-part paths; files live as
  `item_NN.png` under the stimuli dir. Loader resolves by basename.
- RMET answer-key JSON is an `items` list, not an `answers` map. The
  manifest already has `correct_label`, so the key is unused for scoring.
- EU-Emotions trials have `stimulus_path` and no `options`. The 6-way
  set comes from `human_options` in the entropy JSON (`trials` map).
  48/243 labels were missing from that set; the manifest label is
  prepended (6 or 7 options).
- Mind Reading `val_full.jsonl` is open-set (one label, ~258 classes).
  No foil policy is locked, so it is not scored.

## 2026-10-18 — Overnight unsteered E-panel

After the 8B EU-Emotions job, submit one 59-minute `PACK=e_panel`
chain (intr is one job / one hour). Resume skips finished parquets.

Run: RMET on 2B, Gemma, Qwen2-VL, Qwen2.5-VL, 32B, text-only Qwen3;
then EU-Emotions on 2B and Gemma.

Do not run: more CLEVRER, full PhysBench, Mind Reading, Molmo,
confirmatory steering. PhysBench 64s were at chance on 8B / 2B / Gemma.

## 2026-10-18 — Framing

Questionnaires generate hypotheses. They are not valid trait measures of
models. Always say E-direction / S-direction.

## 2026-10-18 — OSF was a steering gate (superseded 2026-10-19)

OSF is not being posted. See 2026-10-19.

## 2026-10-19 — Confirmatory panel is four models, not a Qwen sweep

Paper robustness is 8B done properly, Gemma as an independent family,
2B as a scale check, and text-only Qwen3 as a vision-ablated control.
Qwen2-VL, Qwen2.5-VL, and 32B are the same family at nearby sizes.
They are not independent replications. Do not extract or score them
for the confirmatory panel. 32B only if 8B steering later works;
Qwen2 / 2.5 only if a reviewer asks.

## 2026-10-19 — E and S both vs a shared control

The first extract defined E = mean(E_narr) − mean(S_narr) and
S = mean(S_narr) − mean(E_narr). That makes S = −E, so cos(E, S) was
−1.0 on every layer of every model (qwen3vl_8b / 2b, gemma4_e4b,
qwen3_text). That is not a separability result.

E and S now each subtract the same CONTROL narratives. `E_minus_S`
keeps the bipolar pair as a named one-axis readout. Template E/S
pairs use shared CONTROL_TEMPLATES for the same reason.

Job 38787870 was cancelled mid-run because it was a FORCE re-extract
with the broken contrast. Re-extract the confirmatory four after this
fix. Existing `cos_E_S = -1` JSON is invalid for RQ1.

## 2026-10-19 — OSF dropped

Not posting an OSF preregistration. Remove the `ESVLM_PREREG_POSTED`
gate. `--allow-unpreregistered` is ignored. Steering may run when the
sweep runner exists. Results are still not a preregistered confirmatory
claim; say path check / planned analysis.

## 2026-10-19 — Next: RQ1 report, then personas, then steering

No extra Qwens, no more CLEVRER / PhysBench / Mind Reading as new
fetches. Steering is the next GPU work once the sweep runner is real.

1. Finish RQ1 from the existing confirmatory extracts (CPU). The
   headline that was missing is E-vs-S class separation
   (`E_minus_S` held-out AUROC), plus the unique residual
   `||E_⊥S|| / ||E||`. First look from the layer JSON: Qwen 8B / 2B /
   text ~0.90–0.91; Gemma ~0.48 (chance). That is a path check until
   `esvlm-report-rq1` writes the locked summary. Also pick a layer
   shortlist for a later sweep; do not steer it yet.
2. Hypothesis 2: 500 mundane persona texts (combinatorial, not
   questionnaire items, not E/S-loaded). Extract residuals on the
   confirmatory four, project onto E and S, and onto the residualized
   unique directions. Report `r(E, S)` and `r(E_⊥S, S_⊥E)` by layer.

`esvlm-report-rq1` on the confirmatory extracts (job 38788030), path
check only:

| Model | cos(E,S) | E vs ctrl | S vs ctrl | E vs S | unique E | layer |
| --- | --- | --- | --- | --- | --- | --- |
| qwen3vl_8b | 0.83 | 0.986 | 0.996 | 0.912 | 0.55 | 27 (0.99) |
| qwen3vl_2b | 0.85 | 0.985 | 0.996 | 0.896 | 0.53 | 26 (0.98) |
| qwen3_text | 0.82 | 0.984 | 0.994 | 0.913 | 0.57 | 32 (0.98) |
| gemma4_e4b | 0.88 | 0.803 | 0.803 | 0.475 | 0.47 | 0 (0.74) |

Qwen family: E and S texts are linearly separable even though the
mean-difference directions are aligned. Gemma: E vs S is at chance;
the layer-0 pick is not a usable unique axis.

## 2026-10-19 — First causal sweep: 8B L27 add, RMET + CLEVRER 128

OSF is not a gate. The sweep runner is real. First job is Qwen3-VL-8B
layer 27, mode=add, alphas {-2,-1,0,1,2}, vectors E / S / E_minus_S.
Tasks: RMET (36) then CLEVRER offset 0 limit 128 (overlaps the
unsteered 6000). Not PhysBench (floor). Not Gemma. Not extra Qwens.
Not the old emotion-axis layer 4.

Job `38798228` completed (19:51, failed=0). Alpha 0 RMET 0.3889
matches the unsteered 8B baseline. Every non-zero add **hurts** RMET
(down to 0.22–0.33) on E, S, and E_minus_S alike. CLEVRER alpha 0 is
0.2266 on this 128-slice; non-zero cells are 0.23–0.27 (noise; SE
~0.04). No direction x domain dissociation. Next: L27 ablate, then
add with position=all. Still no 2B / Gemma / extra layers unless one
of those two moves as predicted.

Job `38802498` completed (21:36, failed=0). **Ablate** last-token: RMET
stays 0.3889 on E / S / E_minus_S; CLEVRER stays ~0.23. Removing the
narrative direction does nothing. **Add position=all**: RMET still
does not rise with E (E−1 drops to 0.17). CLEVRER can move (up to
~0.34) but on every vector and both signs — a generic shove, not a
dissociation. Narrative difference-in-means is closed as a causal
target.

## 2026-10-19 — Construct change: task-conditioned social vs mechanical

Narrative E/S prose is retired for causal work. It measured writing
style, then we steered that on visual tasks.

New construct: last-token residual **while the model is asked the
task**. Social stack = RMET (question + face). Mechanical stack =
CLEVRER (question + video). Direction = mean(social) −
mean(mechanical). Held-out AUROC on items not used to build it. Then
steer that direction on held-out RMET and CLEVRER.

This is still an E/S-mechanism study if E–S theory is treated as a
prediction about separable social vs mechanistic computation, not as
a claim that our stories are EQ/SQ. Say social-inference direction /
mechanical-inference direction.

`esvlm-extract-task` writes that pair. The first RMET-vs-CLEVRER
extract was Design C (confound); the paper uses same-image and
matched-video designs instead. Do not go back to narrative
add-sweeps. SAEs are a later decomposition of this task direction if
it is real and causal, not a rescue of the prose axis.

Persona job `38790120` (500 mundane texts, confirmatory four) completed.
Raw `r(E, S)` is strongly positive, as the aligned directions predict.
Residualized `r(E_⊥S, S_⊥E)` is strongly negative (a trade-off, not
independence). Path check only; H2 as written is not supported.

| Model | r(E,S) mean [min, max] | r unique mean | at candidate layer |
| --- | --- | --- | --- |
| qwen3vl_8b | 0.90 [0.65, 0.98] | -0.72 | L27 r=0.89 / r_u=-0.76 |
| qwen3vl_2b | 0.86 [0.42, 0.95] | -0.84 | L26 r=0.64 / r_u=-0.89 |
| qwen3_text | 0.80 [0.37, 0.98] | -0.78 | L32 r=0.49 / r_u=-0.87 |
| gemma4_e4b | 0.90 [-0.30, 0.99] | -0.81 | L0 r=0.80 / r_u=-0.81 |

## 2026-10-19 — Design C is a confound, not a finding

Job `38807101` (RMET faces vs CLEVRER videos) completed. Hold-out
AUROC was 1.0 at every layer, including layer 0. That is
image-versus-video, not a social/mechanical mechanism. Keep it as
Design C (negative control). Do not steer it. Do not treat it as
evidence for an E/S direction.

## 2026-10-19 — One paper: matched-modality task directions

Not everything from the "what would be exciting" list is one paper.
In: Design A (same-image RMET vs appearance + text residualize),
Design B (EU video vs CLEVRER video), confirmatory steer on hold
items if `steer_ok`, then 2B / Gemma transfer only if 8B steering
moves as predicted. Out of this paper: SAEs, old RQ3/RQ4, extra
Qwens, Mind Reading, more narrative add-sweeps.

`esvlm-extract-task` now writes train/hold ids, projection AUROC on
hold, a visual leftover for Design A, and `steer_ok`. Candidate
layer is locked from that leftover (Design A) or from hold AUROC
with early layers deprioritised (Design B) before any steer.

`scripts/activate_env.sh` now falls back to `$ESVLM_VENV_ROOT/<profile>/.venv`
so a later rsync of `envs/main/` cannot hide the project-store venv.

`resolve_local_dir` now inverts `CLUSTER_KEY_ALIASES` (disk `qwen3vl`
-> panel `qwen3vl_8b`). Job `38815744` failed in 2m34s because it
looked for `qwen3vl_8b/config.json` and skipped the on-disk `qwen3vl`
folder. First fix; resubmit `PACK=paper8b` on 8B only.

Job `38817548` completed (16:01, failed=0). Gate 1 is closed.

Design A (same-image RMET, n=36/36, hold 11): raw and text-only
AUROC 1.0 at every layer. `unique_vs_text` is ~0. Visual leftover
AUROC is 0 or undefined except a last-layer 1.0 spike on 11 items.
`wording_only=true`. That spike is a collinear leftover, not a
visual mechanism. Do not steer. `steer_ok` must be false when
`wording_only` is true or unique fraction is ~0.

Design B (EU vs CLEVRER, 243/256): hold AUROC 1.0 including layer 0.
`modality_confound=true`, `steer_ok=false`. Faces-in-video vs
objects-in-video is still a content confound.

Cosine of the new 8B task directions with the closed narrative E
vector is near 0 (Design A mean -0.05; Design B mean 0.02). The
construct actually changed; it is still not a causal social axis.

No 8B causal job. No 2B / Gemma transfer. No SAEs. The paper is a
controlled null: narrative leftover is writing style; task residuals
are wording (A) or modality/content (B, C).

See `docs/PROPOSAL.md` and `docs/PLAN.md`.

## 2026-10-20 — Leave E–S. Two public-item papers in parallel

The licensed task-conditioned paper is a controlled null, not a
highly cited benchmark. Rebuild onto public items. Run both papers
on the same stack.

Paper 1: NExT-QA same-video pairs. Social = type CW only (not CH).
Physical = DO, then DC, then DL; lowest qid. Published questions.
Val lock: 461 pairs from 4996 trials (228 DO, 108 DC, 125 DL).
Videos `{video}.mp4` are not fetched in the annotation job. Missing
mp4 loads as `media_type=none` plus `meta.media_rel`. Design
`same_scene_nextqa` residualizes against text. `--allow-missing-media`
is the wording control and may run before videos exist.

Paper 2: MMToM-QA text-only (MIT, 600 items) versus PIQA. Official
`answer_list` is `["a","b"]`; parse `(a)` / `(b)` from the question.
Do not download the 257GB video zip. Design `mmtom_vs_piqa`.

RMET appearance is not this rebuild: licensed, n=36, author-written
foil, syntax-mismatched, already `wording_only`. CLEVRER has no
social demand.

Fetch via `esvlm-fetch-public` into `$ESVLM_PUBLIC_DIR`. Annotations
only. See `docs/PUBLIC_PAPERS.md`.
