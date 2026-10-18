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
| molmo2 | `allenai/Molmo2-O-7B` | `784410650d12be9bc086118fdefa32d2c3bced86` | molmo | Excluded 2026-10-18; not confirmatory |
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
(text-only residual extract: dummy image grid; `images=[]` then failed
in the processor). Not dropped because of an E/S finding: there are no
Molmo E/S results.

The confirmatory panel is still Qwen3-VL at 2B/8B/32B, Qwen2.5-VL-7B,
Qwen2-VL-7B, Gemma 4 E4B, and text-only Qwen3-8B. Gemma 4 remains the
non-Qwen VLM. `env-molmo` stays on disk if we ever reconsider.

## 2026-10-18 — Framing

Questionnaires generate hypotheses. They are not valid trait measures of
models. Always say E-direction / S-direction.

## 2026-10-18 — S-domain defaults

Locked public S-sets (not licensed stimuli; never mixed with RMET /
EU-Emotions / Mind Reading trees):

1. **PhysBench** dynamics / relationships / phenomena items with
   *textual* A-D options (Chow et al., ICLR 2025). The official HF
   `test.json` has no answers; labels come from
   `lmms-lab-eval/PhysBench`. Image-option items are dropped because
   sequential option log-probs need text labels.
2. **CLEVRER** validation explanatory / predictive / counterfactual
   MCQ (Yi et al.). Causal-video complement.
3. **PIQA** validation as the text-only physical-commonsense control
   for the Qwen3 backbone.

Rejected as the main visual S-set: 2001 Folk Physics Test (n=20,
public-PDF contamination, licence); ScienceQA / AI2 ARC (school-science
or text, not intuitive physics).

Fetch into `$ESVLM_PUBLIC_DIR` via `slurm/fetch_sdomain.sbatch`.
CLEVRER videos go under `$ESVLM_CLEVRER_DIR` on the project store
because the user RDS file cap is too tight for thousands of mp4s.

## 2026-10-18 — No steering until OSF preregistration

Milestone E is a go/no-go. Do not start Phase 3 sweeps until the user
confirms the OSF preregistration is posted.
