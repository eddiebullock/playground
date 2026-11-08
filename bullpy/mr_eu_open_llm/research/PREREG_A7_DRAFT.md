# DRAFT amendment A7: vision-path cuts (RQ3)

**Status: DRAFT. Not frozen, not tagged, not preregistered.** Nothing here binds until it is audited, committed and tagged. It must freeze before any full A2 cell is read. No code for it exists, no recovery has run, and no A7 forward has run. Audited locally 2026-11-08 (section "Audit 2026-11-08" at the end); the audit's rulings are in the text, marked "[audit 2026-11-08]".

**Next step.** The next step that produces evidence is **Stage 0**: mechanics on the 24 CREMA-D calibration clips, which are outside the analysis set, showing for each candidate site whether it is a complete cut on `single_face` and the blind readout. **Stage 0 needs the cluster** (GPU, about .45 GPU-hours for the exit and route checks, .71 with the region test, upper bounds). Before it can run, three local items must be done: the code, the CPU-only toy recovery of Level 1, and the freeze (see Order of work). Stage 0 runs only after the freeze, so that no Stage 0 number can move a threshold.

**The rule for RQ3, stated plainly [audit 2026-11-08].** A site is a **complete cut** on a model only if Stage 0 shows it, by the numeric criterion in "Complete cut: the numeric criterion" below. **Wherever no complete cut is shown, RQ3 is reported as "not testable" for that model and question.** A site that was never run, a run that failed, or a missing output counts as not shown.

Draft `design_sha256` of the block below: `9126e4160748a690a7bc702a6e7936d13d7a8b187f0a9b3accc9f9129184d799` before the audit; restated after the audit in "Audit 2026-11-08". It changes with any edit to the block and is restated at the freeze.

**Written:** 2026-11-07, before any A1 result was read, before any full A2 cell exists, and before any A7 quantity was computed.
**Goal:** G7 in `research/GOALS.md`.
**Depends on:** G1 and G4 for the contrast stage. RQ0 is read through the frozen A1 and A3 table by A2's `rq0_status`, on `single_face`. Gates G2 to G4, delivery, Gate B and the sham as in A2. Stage 0 depends on none of these, because it is mechanics on clips outside the analysis set.
**Models.** Gemma 4 and Qwen3-VL, on `single_face` only. Molmo2 is out under A2, so it has no cells here.
**Code:** none yet. See Order of work.

**The short answer, stated first.** The literal test in RQ3, "does blocking that path remove the effect", cannot fail. A cut that takes out the whole vision path gives the same output for every image (G3, `cut_clip_vs_cut_donor`, exactly zero), so any cue effect under it is zero by construction. Every narrower cut measured so far inside the language model leaks: the stored distance fraction (defined under Evidence, and not a share of the effect) is .274 on Qwen3-VL and .423 on Gemma 4 for the earliest image-position cuts tested, not zero, and a cut at Gemma layers 24 and later leaves the output exactly unchanged. So the depth at which a cue is carried is not testable on either model. What remains testable is narrower: which of two blocks of the complete cut carries a cue's effect, and whether tokens at a cue's location can be a real cue cut, which has no evidence yet and is expected to fail on architecture. A7 says this plainly and tests only that. Under the audit's ruling (see "Ruling on blocks"), neither of these narrower questions counts as an RQ3 cut result: they are a secondary decomposition of a complete cut, and RQ3 itself is reported from Stage 0 and the "not testable" rule.

**What was known when this was written.**

- No A1 result, and no full A2 cell. `rotation_summary.json` does not exist in `research/reproduction`.
- A3's result as stated in `STATE.md`: all three models pass on `single_face`. The A3 summary file was hashed for the design block and not opened.
- The vision gates, in full, including gate-sample accuracy on 40 clips (Gemma .425, Qwen .275 on the own clip). Those are gate figures, not an RQ0 result.
- The 3-item delivery checks and the readout ladder, on mechanics. The ladder names the 24 CREMA-D calibration clips, and their clip ids carry an emotion code. No label or vote was used.
- The smoke_v2 mechanics only as restated in A2's appendix. No smoke cell file was opened.
- The EU-Emotion item counts by label and actor, counted from `data/eu_emotions_human_entropy.json`. Response distributions were not used.
- The pilot ablation (5 of 180 flips, Qwen layer 4) and the Phase 2 probe gate file, both from before the prompt confound was found.

**Files read for this draft, and files not read.**

| Read | Why |
|---|---|
| `research/GOALS.md`, `RESEARCH_QUESTIONS.md`, `STATE.md`, `DECISIONS.md` (searched, and the entries of 2026-11-04 to 2026-11-07 read), `REPRODUCTION.md`, `COMPUTE.md`, `NEEDS_INPUT.md`, `PLAN_READOUT_AND_INPUT.md` | Rules and status |
| `research/PREREG_A2_DRAFT.md`, `PREREG_A5_DRAFT.md`, `PREREG_A6_DRAFT.md` (first 240 lines and section list) | Format, imports |
| `scripts/vision_gates.py` (whole), and parts of `scripts/run_stimulus_interventions.py`, `analyse_stimulus_interventions.py`, `stimulus_interventions.py`, `a5_analyse.py`, `path_patching.py`, `readout_ladder.py` | Hooks and functions to import |
| `research/reproduction/vision_gates_{gemma4,qwen3vl,molmo2}.json`, `input_delivery_{gemma4,qwen3vl}.json`, `readout_ladder_{gemma4,qwen3vl}.json`, `phase3_pilot_ablation.json`, `phase2_gate.json` (opening only). The Molmo2 delivery and ladder files were not opened | Evidence on sites |
| `research_notes/VLM versus human emotion mechanisms/causal_methods.md` (excerpts of sections 1 to 4 and 8) | Method notes |
| Local `transformers` 5.9.0 source for `gemma4` and `qwen3_vl`, by excerpt | Vision attention and route wiring. The gates ran on 5.10.1, so these readings are of a neighbouring version |

**Not read.** No A1 result, no file matching `eval_v2_*rot*.json`, no `research/reproduction/same_clip_family_model_*.json`, no A2 cell output. Directory listings were filtered so that those names were not displayed.

## Question

RQ3, stated causally and about the model only. Fix a model, the blind prompt and the prefill. Let X be the pixels sent on `single_face`, V the tensors that carry them out of the vision stack, and Y the blind readout. The forward is deterministic, so for one item every combination of inputs has one exact output.

    X  ->  V = (V_A, V_B)  ->  language model  ->  Y

Take an edit that A2 applies to pixels, x to x'. A2 measures its total effect on Y. RQ3 asks whether that effect travels through a named part of V. Because the system is deterministic and can be set to any state, cross-world quantities such as "V_A as under x' and V_B as under x" can be computed, and need no assumption about unmeasured confounders. That is a property of the model and has no human counterpart.

**Scope.** Causal language applies to the model. A7 makes no claim about humans, no match to a human prediction, and no statement that a route resembles any human pathway.

**What the second half of RQ3 asks.** "Or can it merely be read out" contrasts use with decodability. A2 already answers it at the level of a cue. An edit that changes pixels, and so tokens, with no change in the readout was readable and not used, and A2's Null verdict states that. A7 adds nothing to that sentence. It only divides an effect that exists.

## What a cut can and cannot show

**Relative to A2.**

- A2 changes pixels. Its contrast gives the total effect of an edit. A cut can only divide that effect. It cannot find a cue effect A2 did not find, and with no total effect there is nothing to trace.
- A pixel edit may add a neutral cue, as well as remove one (A2, Limits). A cut inherits that. It says how the edit's effect travels, not what the edit means.

**Relative to A5.**

- A5's `image_free` row is the same forward as a complete exit cut, if Stage 0 confirms this on `single_face`. So A5's W is the output under the complete cut, and a complete cut adds nothing to the split W + G + F + E + R.
- A cut at the exit acts on what reaches the language model. It says nothing about what the language model does next. If a face triggers stored word associations, that happens after the exit, so a cut cannot separate perception from word knowledge. RQ4 stays with A5.

**What a complete cut shows.** That no route from pixels to the answer has been missed. That is a validity result about the hooks and the architecture. It is worth stating once and is not evidence about any cue.

**What a block of a complete cut shows.** The size of one path-specific effect on one set of stimuli: which tensor carried this edit's effect in this model.

**What no cut here shows.** That a model uses a cue as a human does. That a route is necessary in general, since a route that seems unneeded here may matter on another stimulus (Mueller 2024; Grant et al. 2025, both at notes level). That a depth in the language model matters, since no cut there is complete. Any statement about the temporal route, since `video_face` is not tested.

## Design block

All fixed constants are here and nowhere else. The hash is sha256 of the bytes between the fence lines, excluding the fences, with a final newline.

```json
{
  "amendment": "A7",
  "a2_design_sha256": "0ae22118d3b2690c581595890b9bb8b904e1f7860f37be9ac909f925b7058de7",
  "a2_code_sha256": {
    "scripts/stimulus_design.py": "89d256c1072e0768fc747294553c5710adaf42184a175866d7bd75bcc6104978",
    "scripts/stimulus_interventions.py": "93c601eb42d6700d4f0fc8ba72317ed5c2e354105d57fcac611c59660125deb3",
    "scripts/run_stimulus_interventions.py": "f7880c94e9671cc294cd4eaeece2108d7cd1d7a4c7e6f9e30a1c495964e6cd82",
    "scripts/analyse_stimulus_interventions.py": "aba4f533daad6210e87f7bb4e68fc4627805cb4dfd9d9308df734fada4edc8b1"
  },
  "a3_summary_sha256": "c1c64e6975b32356a3314268c054f0dee483ddf74b11cd66d03b2faf972e64a4",
  "other_code_sha256": {
    "scripts/vision_gates.py": "6327b6f88e99409cba5c63d5aca0fe97348fbce4f3b08522158a8d9ba3ce8194",
    "scripts/readout_ladder.py": "738b2c5031e5f7f1c0337896b0e8e680b2f767d0f33c850588f743d6c5595c1c"
  },
  "evidence_sha256": {
    "research/reproduction/vision_gates_gemma4.json": "b9e5743f72c9a9271aa79d2fa0d2f446030793b4e0959a2acec7bbaa7057d764",
    "research/reproduction/vision_gates_qwen3vl.json": "2f38aafb41c93505b792bd355c72d2a2b30e841a0e29be067a1d7c5ba01ede8d",
    "research/reproduction/input_delivery_gemma4.json": "94f55b9fd96767e0c19c0bbeb34eebd5a47dbbbf5b14058d53d6f15a8b40a8ca",
    "research/reproduction/input_delivery_qwen3vl.json": "802f3231db29f2b27d67e8a6cfd70c78a9ea72c9e3fb4b1cadf7f8c966f1f801",
    "research/reproduction/crema_calibration_clips.csv": "7f442aec7d41ff421d9e5025c7c61dc99e9302e9d5dba4201e1897bdcd69d98d"
  },
  "models": ["gemma4", "qwen3vl"],
  "route": "single_face",
  "dtype": "bfloat16",
  "readout": "blind",
  "image_tokens": {"gemma4": 256, "qwen3vl": 576},
  "tokens_across_view": {"gemma4": 16, "qwen3vl": 24},
  "crop_px": 768,
  "n_word_logprobs": 106,
  "n_families": 21,
  "n_calibration_clips": 24,
  "n_items_primary": 200,
  "n_actors": 17,
  "excluded_labels": ["Kind", "Unfriendly", "Jealous", "Neutral"],
  "baseline_condition": "original",
  "stage1_edit": "mask_eyesmouth_neutral",
  "stage0_edit": "mask_eyesmouth_blur",
  "other_fills": ["mask_eyesmouth_first", "mask_eyesmouth_blur"],
  "route_blocks": {
    "gemma4": {"A": "hidden_at_image_positions_entering_L0", "B": "per_layer_input_at_image_positions_all_layers"},
    "qwen3vl": {"A": "main", "B": ["deepstack_0", "deepstack_1", "deepstack_2"]}
  },
  "token_buffers": [0, 1, 2],
  "max_region_token_share": 0.5,
  "token_map_min_hit_rate": 0.90,
  "token_map_tolerance_tokens": 1,
  "token_map_square_px": 96,
  "token_map_positions": 4,
  "token_map_centres_px": [[192, 192], [576, 192], [192, 576], [576, 576]],
  "region_edit_min_fraction": 0.10,
  "region_bootstrap_unit": "clip",
  "stage1_endpoints": "same_job_native",
  "sentinel_reference": "same_job_native_unrounded",
  "sentinel_every": 10,
  "min_item_share": 0.90,
  "seed": 42,
  "alpha": 0.05,
  "ci_decision": 0.95,
  "ci_equivalence": 0.90,
  "bootstrap_draws": 10000,
  "completeness_tolerance_nats": 0.001,
  "leak_limit": 0.05,
  "block_live_fraction": 0.10,
  "share_margin_fraction": 0.20,
  "recovery": {
    "replicates": 500,
    "expected_verdict_min": 0.80,
    "other_named_verdict_max": 0.10,
    "inconclusive_max": 0.20,
    "false_positive_max": 0.07,
    "seed": 7042,
    "effect_units": "nats_acted_family_clr",
    "effect_grid": [0.25, 0.5, 1.0, 2.0],
    "pass_effect": 2.0,
    "te_margin_nats": 0.2,
    "actor_sd": 0.3,
    "item_sd": 0.5,
    "false_carry_shares": [0.25, 0.30],
    "all_null_replicates": 2000,
    "toy_leak_nats": [0.0, 0.0005, 0.002, 0.05],
    "toy_region_leak": [0.0, 0.03, 0.08, 0.30]
  }
}
```

`completeness_tolerance_nats` is A2's B1 determinism rule (1e-3 on logprobs, `B_RULES["repeat"]` in the frozen analysis code), applied here to the 106 unrounded word logprobs. `leak_limit`, `block_live_fraction`, `share_margin_fraction`, `token_buffers`, `max_region_token_share`, `min_item_share` and the token-map values are **[proposed]**. The 2026-11-08 audit confirms them for the freeze unless recovery shows a row cannot be reached (see Recovery); none has been tried on any data. `tokens_across_view` and `crop_px` are descriptive of the `single_face` layout and decide nothing.

**Added by the 2026-11-08 audit, all [proposed].** `n_word_logprobs` and `n_families` restate counts of A2's frozen lexicon (106 words, 21 families, counted from `LEXICON` in `scripts/run_stimulus_interventions.py`). `token_map_centres_px` fixes the four positions of the token-map square, which the draft left undefined: the centres of the four quadrants of the 768 by 768 crop. `region_edit_min_fraction` adds a floor on the region test's denominator, which had none (see Stage 0). `region_bootstrap_unit` fixes the resampling unit; each of the 24 calibration clips has a different CREMA-D actor (counted from the actor prefix of the clip ids), so a clip bootstrap is an actor bootstrap. `stage1_endpoints` and `sentinel_reference` record the ruling that every Stage 1 cell of an item, including the two native endpoints, comes from one job (see Stage 1). `effect_units` and `te_margin_nats` fix two recovery quantities the draft left undefined (see Recovery).

## Evidence by model and site

**What the evidence is.** All figures come from `vision_gates_{model}.json`: stage `bfloat16`, 12 clips, field `route_tests` unless stated. Each is the mean of `fraction_of_image_effect_left`, with its range over clips.

**Three limits on all of it.**

1. **It is on the wrong input and the wrong readout.** G2 to G4 ran on the legacy media path (`prepare_trial_media`) with the MCQ prompt and the prefill "EMOTION:". Gemma had 253 to 256 image tokens and Qwen 6,120 to 22,440. A2's `single_face` route has 256 and 576 (3-item delivery rows). The blind prompt and prefill never ran under a cut. So **no site below has been shown complete on `single_face` or on the blind readout.** That is not a formality to assume away. It is what Stage 0 is for.
2. **It reads the last position of one forward.** A2's blind score needs one forward per internal word prefix. The hook applies to each, but this has not been run.
3. **The stored fraction is not a share.** It is the distance from the cut output to the image-free output, over the distance from the full output to the image-free output, on all vocabulary logits. A value near 0 does mean the cut output is near image-free. A value near 1 does not mean the effect survives: the output may have moved elsewhere at the same distance. On Gemma, the per-layer cut has a fraction of .93, and yet its median largest logit change from the full output (16.8) is as big as the full output's from image-free (14.2). So it is not "93% left". On Qwen the same measure does fit the reading in `DECISIONS.md`, since a route cut stays nearer the full output (median 10.6 for main, 4.6 for deepstack, against 36.8). The stored rows cannot settle it either way, because the distance to the full output is stored only as a maximum. A7 does not use this fraction for any decision.

**Gemma 4.** 42 language model layers, 16 ViT layers. One tensor leaves the vision stack (`embed_vision`). Inside the language model it enters as the hidden state at image positions and, separately, through a per-layer input at image positions in every layer. Layers 24 to 41 reuse keys and values stored at layers 22 and 23. Results at float32 are in brackets.

| Site | Remaining, mean (range) | Complete? | Evidence |
|---|---|---|---|
| Exit, `embed_vision` output, all tokens | Exactly 0 against image-free | **Yes, on the legacy MCQ input.** Not shown on `single_face` | G3 `all_vision_cut_vs_image_free` and `cut_clip_vs_cut_donor`, exact zero at both dtypes |
| Language model entry, hidden stream and per-layer stream together | .000 at every clip | **Yes, on the legacy MCQ input.** Not shown on `single_face`. The stored run clamped the hidden stream at all layers. Clamping it at L0 only, with the per-layer stream held, follows by logic and was not run | `main_stream_and_per_layer_cut` |
| Hidden stream at L0 only, per-layer live | .517 (.193 to .930) [.487] | No | `main_stream_cut_L0_per_layer_live` |
| Per-layer stream only, hidden live | .930 (.428 to 1.979) [.881] | No | `per_layer_cut_main_stream_live` |
| Image positions from L1, per-layer held | .423 (.077 to .797) [.443] | No. Text has already read the image at L0 | `late_cut_image_positions_from_L1` |
| Image positions, L6 to L23 window | .671 (.127 to 1.134) [.628] | No | `image_positions_window_before_kv_sharing` |
| Image positions from L24 | 1.000, exactly no change from the full output | **No. A guaranteed false null** | `image_positions_from_first_kv_shared_layer` |
| A subset of exit tokens | No completeness evidence | Not shown | G4 shows the swap works, not that a subset contains anything |
| ViT internal layers | Never tested | Not shown | None |
| Text-to-image attention, by layer window | Never tested, and no code exists | Not shown | None |

**Qwen3-VL.** 36 language model layers, 27 ViT layers. Four tensors leave the vision stack: main, and deepstack from ViT blocks 8, 16 and 24, injected after language model layers 0, 1 and 2 (measured, `injected_after_lm_layer`). Bfloat16 only: float32 ran out of memory, so G3 here covers less than it does on Gemma (`DECISIONS.md`, 2026-11-05).

| Site | Remaining, mean (range) | Complete? | Evidence |
|---|---|---|---|
| Exit, `visual`, all four routes | Exactly 0 against image-free | **Yes, on the legacy MCQ input, at bfloat16.** Not shown on `single_face` | G3, exact zero |
| Main only, deepstack live | 1.032 (.909 to 1.231) | No | `main_only_cut_deepstack_live` |
| All deepstack, main live | 1.034 (.964 to 1.185) | No | `deepstack_only_cut_main_live` |
| One deepstack level | .978, .992, .990 | No | `deepstack_{0,1,2}_cut` |
| Image positions from L3 | .274 (.182 to .653) | No. Text has already read the image | `late_cut_image_positions_from_L3` |
| Image positions from L1 or L2 | Never tested | Not shown. Deepstack would still enter there | None |
| A subset of exit tokens | No completeness evidence | Not shown | As Gemma |
| ViT internal layers | Never tested | Not shown. A whole-state hold above block 8 would leave deepstack live | None |
| Text-to-image attention | Never tested, and no code exists | Not shown | None |

**Molmo2.** Out under A2, listed for the record. G3 passes at both dtypes, G2 fails at bfloat16, G4 fails. Image positions from L1 leave .117 (.040 to .229). No verdict is given.

**Why the language model has no complete cut.** Image positions are replaced at layer k, but text positions after the image read image keys in every layer before k, and that information stays in them. A hold on image positions alone therefore cannot be complete at any k past the entry. Also holding the text positions would give the image-free forward again, which is the exit cut under another name.

**Why a token subset is unlikely to be a cue token set.** Both vision stacks use global self-attention: no window appears in the `qwen3_vl` vision attention, and `gemma4` builds a bidirectional mask over patches (5.9.0, by excerpt). So every exit token depends on every patch, and a pixel edit at the mouth changes tokens elsewhere. Neo et al. and "Where To Look?" point the same way on other models (notes level). This predicts a failure of the test below. It does not establish one, because a token may change with no effect on the answer.

### Testability, as it stands today

| Question | Gemma 4 | Qwen3-VL |
|---|---|---|
| Does blocking the whole vision path remove a cue effect? | **Vacuous.** Removal is guaranteed by construction, so it is reported as validity, not as a finding | **Vacuous**, same |
| At what depth of the language model is the effect carried? | **Not testable.** No complete cut exists past the entry | **Not testable**, same |
| Which of the two streams carries the effect? | Not an RQ3 cut result under the ruling on blocks. Estimable as a secondary decomposition if Stage 0 passes C1 to C4. No evidence on `single_face` yet | Same, main against deepstack |
| Do tokens at the cue's location carry it? | **Not testable** by a complete cut. A secondary decomposition only if the region test passes. No evidence yet, and failure is expected | Same |
| Any question on `video_face` | **Not testable here.** No site was tested on it, and A7 does not run it | **Not testable here**, same |

## Three kinds of cut

The wording of G7 and of the brief for this amendment is that a cut with leakage is not a cut. A7 keeps three things apart so that this rule can be applied without loss.

| Kind | Definition | May be called | Use |
|---|---|---|---|
| **Complete cut** | Every path from pixels to the answer crosses the site, and everything at the site is held. Must pass Stage 0 | A cut of the vision path | Validity only. Its contrast is zero by construction |
| **Block** | One named part of a complete cut, with the rest of that cut as its complement. Each block alone is not a complete cut, and is never called one | "Block", "route" or "stream". Never "the vision path was blocked" | The estimands below. What passes through the complement is the other block's share, and it is measured, not ignored |
| **Cue token set** (the draft's "cue cut") | A set of tokens such that holding it stops the edit's effect, up to `leak_limit` | "A token set that holds the edit's effect to within 5%". Never "a cut" [audit 2026-11-08] | Only if the region test passes, and then as a block of the complete exit cut. Otherwise cue-level mediation is reported as not testable |

**Hold, not removal.** A hold sets a tensor to its own value under another image of the same clip. G3's single-route cuts instead wrote in the image-free placeholder or zeros, which puts the model in a state no image produces. A hold keeps the model near its normal range, and the difference between x and x' is local. This is the interchange intervention of Geiger et al. and the "necessity" and "sufficiency" patches in `causal_methods.md`. Zero ablation is avoided, as the notes advise. Hold operations reuse `vision_intervention`, `LayerLab` and `swap_tokens`, the same code whose replay gave exactly zero in G3 and whose 100% swap equalled a native donor forward exactly in G4.

**Definitions of the operations [audit 2026-11-08].** The draft used these without defining them.

- **`image_free`** is A2's image-free forward, as built in `baseline_cells` of `run_stimulus_interventions.py`: the same encoded prompt and token layout, with the pixel tensors (`PIXEL_KEYS`) removed by `strip_pixels`. It is the same row A5 uses.
- **Exit cut** is a removal, used for validity only: every vision route at the exit is replaced by what the image-free forward has in its place, by `_cut_fn` of `vision_gates.py` with all routes (the image-free placeholder for Gemma's `embed_vision` output and Qwen's main route, zeros for Qwen's additive deepstack routes). It is the operation G3 tested.
- **Hold** sets a named tensor to its value under another image of the same layout, as described below. Blocks, the C4 check and every Stage 1 hybrid use holds, never removal.

**Complete cut: the numeric criterion [audit 2026-11-08].** A site is a complete cut on a model if and only if, in Stage 0 on that model, every one of the following holds on **all 24** calibration clips and on **every one of the 106 word logprobs** of the blind readout, unrounded, at bfloat16:

- C1: native and repeat differ by at most `completeness_tolerance_nats` = 1e-3 nats;
- for the exit: C2 (exit cut against `image_free`) and C3 (exit cut on i against exit cut on j), each at most 1e-3 nats;
- for a partition into blocks: C4 (all blocks held at i's values with pixels from j, against native on i) at most 1e-3 nats, in addition to the exit passing.

The worst absolute difference over all clips and logprobs is what is compared, so one clip or one word over 1e-3 fails the site. There is no partial credit and no rerun to get a pass; a rerun is allowed only to fix a code fault, with a dated note, and the rerun then replaces the whole Stage 0 output for that model.

**Ruling on blocks [audit 2026-11-08].** The draft left open whether a block of a complete cut is admissible under G7 (DECISIONS 2026-11-07, "Open"). The audit takes the **stricter reading, as the more conservative option**: a block leaves the other block live, so it is not a complete cut, and a block result is **not** an RQ3 cut result. Consequences:

1. RQ3, as G7 reports it, rests on Stage 0 only: the table of sites, the vacuous complete-cut contrast, and "not testable" for every narrower question on each model.
2. The route arm and the token arm stay preregistered here as a **secondary decomposition of a complete cut**, with the rules written below and no change to them. They are optional, as the cost table already says. Their results are reported in the supplement under the heading "decomposition of the complete exit cut", and are never stated as "a cut", as an answer to RQ3, or in the abstract.
3. A cue token set that passes the region test leaks up to 5% by its own rule, so under G7's wording ("a cut with leakage is not a cut") it is not a cut either. It enters only as a block of the complete exit cut.
4. The study lead may delete the secondary arms at the freeze; that would be a narrowing and needs only a dated note. Adding them back after the freeze is not allowed.

The draft's lenient reading was defensible (the site is complete and the blocks exhaust it, with the complement measured), and it is recorded here so that the choice is visible.

## Stage 0: completeness, before any contrast

Stage 0 runs on the 24 CREMA-D calibration clips (`crema_calibration_clips.csv`), on `single_face`, at bfloat16, with A2's blind prompt, prefill and lexicon. They are outside the 243 EU-Emotion items and outside A6's 819. All 24 were usable on both models in the readout ladder, and labels and votes stay unread. So no analysis clip is touched, and no contrast by cue, label or direction can be computed. Clips are paired in file order: clip i with clip i + 1, the last with the first.

**Why calibration clips.** Using analysis clips would mean reading model output on edited analysis items before the analysis. Completeness is a property of the wiring, not of the clip, once the token layout is fixed. On `single_face` the layout is fixed at 256 and 576 tokens by the 768 by 768 crop. The sentinels in Stage 1 then repeat the key check on analysis items at run time, on a rule that reads no contrast.

**Cells per clip, exit and route test, 7 cells.** Native on i; a repeat of it; `image_free` on i; exit cut on i; all blocks held at i's values with pixels from j; block A held at j's values on i; block B held at j's values on i.

| Check | Rule | On failure |
|---|---|---|
| C1 Determinism | Native and repeat differ by at most `completeness_tolerance_nats` on every one of the 106 word logprobs, on every clip | Stop. Stage 0 has no result for that model. A fault to fix and rerun, with a dated note |
| C2 Exit cut is image-free | Exit cut on i against `image_free` on i, same rule | Exit is not a complete cut on `single_face` |
| C3 Exit cut ignores the image | Exit cut on i against exit cut on j, same rule | Exit is not a complete cut on `single_face` |
| C4 Hold is complete | All blocks held at i's values, pixels from j, against native on i, same rule | The two blocks do not exhaust the site. A route was missed |
| C5 Both blocks live | For each block: mean RMS over 21 family CLRs of (hold from j on i, minus native on i), at least `block_live_fraction` of the mean RMS of (native j minus native i). Both means are over the 24 clips, the ratio is of the two means, and it is a point estimate with no interval [audit 2026-11-08] | That block is dead, see Null and Inconclusive. C5 is not part of the complete-cut criterion |

**The threshold.** C1 to C4 use one tolerance, 1e-3 nats, on unrounded word logprobs. This is A2's B1 rule, on the same scale. The tolerance is tight on purpose. When a hold is complete, every tensor past the site is bit-identical, so the difference is exactly zero, as it was at every site that passed G3. Any larger difference means information got through or the run is not deterministic, and C1 separates these. A failure at bfloat16 stands. No float32 result can rescue it, as with Molmo's G2. A leak of any size at C2 to C4 means no cut, and there is no partial credit.

**Region test, 4 further cells per clip.** Only needed for the token block. It asks whether holding a set of tokens stops an edit, on a readout scale, with no analysis clip.

- **Token map.** Each model's map from face crop pixels to exit token is written from the processor, then checked. On each clip a grey square of `token_map_square_px` is painted, centred in turn at each of the `token_map_positions` positions in `token_map_centres_px`, and only the vision stack runs (24 clips by 4 positions, 96 trials). A token's change is the L2 norm of its vector difference from the unpainted pass, summed over the exit routes (one route on Gemma, four on Qwen). The token with the largest change must fall within `token_map_tolerance_tokens` (Chebyshev distance on the token grid) of the expected token in at least `token_map_min_hit_rate` of the 96 trials. On failure the region test has not passed. [audit 2026-11-08: positions, change measure and distance were undefined]
- **Set S_b.** The tokens whose pixel footprint, by the token map, contains any pixel of `mask_eyesmouth` on that clip, dilated by b tokens on the grid, for b in `token_buffers`. S_b is built per clip. A value of b is admissible only if S_b holds at most `max_region_token_share` of the image tokens on **every** clip [audit 2026-11-08: the draft did not say per clip or overall]. A larger set is not a local claim. The cap also guarantees that a same-size placebo set Q outside S exists in Stage 1.
- **Cells.** Native on the edited frame x' (`stage0_edit`, a blur fill, since calibration clips have no pinned neutral donor). Then for each b, S_b held at x's values with all other tokens from x'.
- **Leak.** With v the 21-family CLR vector, and means taken over clips:

        leak(b) = mean RMS[ v(S_b from x, rest from x') - v(x) ] / mean RMS[ v(x') - v(x) ]

- **Floor on the denominator [audit 2026-11-08].** leak(b) is a ratio, and the draft gave no floor on the edit's own effect. If the blur edit barely moves the readout, the ratio is unstable and a pass would mean nothing. So the region test is run only if mean RMS[ v(x') - v(x) ] is at least `region_edit_min_fraction` of the mean RMS of (native j minus native i) from the C5 cells, the same scale C5 uses. Otherwise the region test has not passed, with reason `edit_too_weak`.
- **Rule.** The set is the smallest admissible b whose upper 95% clip-bootstrap limit of leak(b) is at or below `leak_limit`. The bootstrap resamples the 24 clips with replacement, recomputes the ratio of means, uses `bootstrap_draws` draws and `seed`, and takes the percentile limit [audit 2026-11-08: unit, draws, seed and method were unstated]. If none passes, **no cue token set exists, and cue-level mediation is reported as not testable on that model.** Stage 1's token arm is then not run.

**Why 5%.** It is a convention, not a derived value, and half of A2's 10% margin. For scale, every leaking site measured so far stands above it on the stored measure: .117, .274, .423. That comparison is rough, since the measures differ. It is fixed before any leak is computed.

**Output.** One table per model, with every check, its worst value and its result. It is written to `research/reproduction/a7_stage0_{model}.json`. It decides mechanically which arms exist in Stage 1, and nothing in it is reviewed by eye. The mapping is fixed [audit 2026-11-08]: exit complete (C1 to C3) is required for any Stage 1 arm; the route arm also needs C4; the token arm also needs the region test, including the token map and the denominator floor; C5 does not decide whether an arm runs, only the `block_dead` reading. If the exit is not complete, RQ3 is "not testable" on that model at every level, and A5's W gets the caveat in "What is reported".

**Stage 0 mechanics plan [audit 2026-11-08].** A short plan of the run, since no code exists yet.

1. Local, before the freeze: `a7_cuts.py` implements the exit cut through `_cut_fn` and holds through `vision_intervention` and `LayerLab`; `a7_stage0.py` writes the 7 cells per clip (and 4 more for the region test), with every word logprob unrounded; the Level 1 toy recovery runs on CPU and must give the expected row for every toy.
2. Local: a dry run of `a7_stage0.py` on the toy module end to end, writing a Stage 0 table in the final format, so the cluster run only swaps the model.
3. Cluster, after the freeze and after code sync with hash check (ROADMAP 2.1): one task per model, 59 minute limit, on the 24 calibration clips only. The job refuses any clip id not in `crema_calibration_clips.csv`, so no analysis clip can be read.
4. The table is pulled and the arm mapping above is applied by code. The Stage 0 table is a result of A7 and goes to `RESULTS_REGISTRY.md` with its script, commit and output hash. A fault in the A7 code may be fixed and Stage 0 rerun, with a dated note, because no contrast exists at that point. A threshold may not be changed.

## Stage 1: stimuli and cells

- **Contrast.** One edit, `stage1_edit`: `mask_eyesmouth_neutral` against `original`, A2's joint ablation, on `single_face`.
- **Why this edit.** It is the largest region dose in A2 and it is not itself one of A2's confirmatory contrasts, so A7 does not select on A2's verdicts. It covers both regions, so it has a clear token set. It has a unit of 17 actors, while P1 to P4 have 8.
- **What this costs in meaning.** "The cue" here is the information in eyes and mouth together, not one action unit.
- **Items.** 200 of the 243: all except the 26 flagged items and the 17 Neutral clips. No flagged label is a target of a confirmatory contrast, as in A2. A Neutral clip under a neutral fill has its own label filled in, and may be its own donor. All 17 actors remain, with 6 to 15 items each (counted from `data/eu_emotions_human_entropy.json`).
- **Endpoints, same job [audit 2026-11-08, design choice].** The draft took y(A, B) on `original` and y(A', B') on the edited frame from A2's stored cells, and only the hybrids from the A7 run. That mixes two runs, possibly on different nodes, into every estimand (PE_A = y(A', B) - y(A, B) would difference an A7 cell against an A2 cell), and A2 stores word logprobs rounded to 4 decimals. Cross-run equality at bfloat16 has not been shown, and the sentinels would check it on only 1 item in 10. The audit takes the more conservative option: **for each item, the two native endpoints and every hybrid are computed in the same process, unrounded**, and the endpoint forwards also supply the tensors that the hybrids hold, so no separate capture forward is needed. A2's stored cells for the same two conditions are compared descriptively (largest absolute difference per model is reported) and decide nothing. The cost is 2 more cells per item, shown in Cost.
- **One Stage 1 run per model.** Every admitted arm runs in the same item loop, so all arms share the same endpoints and one TE per model.
- **Readout.** Blind only, by A2's `score_blind`, with the hold active on every forward. A hybrid has no MCQ. Its cell holds all 21 family scores, so the acted label, the human modal label and any other family are read from one cell at no extra cost.
- **No storage of tensors or frames.** Held tensors live in memory for one item. Frames are edited in memory, as in A2.

**Hybrid cells per item.**

| Arm | Cells | Exists if |
|---|---|---|
| Endpoints | y(A, B) on `original`, y(A', B') on the edit | Any arm exists |
| Route | y(A', B), y(A, B') | C1 to C4 pass. Secondary decomposition only, under the ruling on blocks |
| Token | y(S', rest), y(S, rest'), y(Q', rest), y(Q, rest') | C1 to C3 and the region test pass. Secondary decomposition only |

**When Stage 1 runs at all [audit 2026-11-08].** The draft made Stage 1 wait on "the study lead's scope decision" with no time for it. A2's analysis output reports the `mask_eyesmouth_neutral` contrast, so a decision taken after reading A2 could select on the very TE that A7 would trace. The rule is therefore: the decision to run or not run each admitted Stage 1 arm is recorded in `DECISIONS.md` either at the freeze or after Stage 0 and before any A2 analysis output is opened, and it may depend only on the Stage 0 table and compute. It may be conditional on the preconditions (RQ0, gates, Gate B, sham), which are then applied by code, not read by eye. If no entry exists by the time any A2 analysis output is opened, Stage 1 is not run and RQ3 rests on Stage 0 alone.

Q is a placebo: a random set of tokens outside S, the same size as S, fixed by a seed keyed on the clip, for criterion 6 of the mediation list in `causal_methods.md`.

**Sentinel.** On every `sentinel_every`th item, counting from the first, in A2's canonical item order (positions 0, 10, 20, and so on, 20 items): pixels from x', every block held at x's values. It must equal the same job's unrounded native `original` cell on that item within `completeness_tolerance_nats` on every one of the 106 word logprobs [audit 2026-11-08: the draft compared against A2's rounded stored cell with an unquantified rounding allowance]. For the token arm the sentinel holds S and the rest both at x's values. Any failure stops that arm for that model, and it is reported as not testable. This reads only an equality that must hold, and so no effect.

## Estimands

For item i, edit x to x', and a partition {A, B} of a complete cut, with y the CLR of the acted family, by A2's `clr_of`. A prime marks a block taken from x'.

    TE    = y(A', B') - y(A, B)          total effect, as in A2

    PE_A  = y(A', B)  - y(A, B)          A alone: sufficiency
    PE_B  = y(A, B')  - y(A, B)          B alone: sufficiency
    N_A   = y(A', B') - y(A, B')         lost if A is held: necessity
    N_B   = y(A', B') - y(A', B)         lost if B is held: necessity

    I     = TE - PE_A - PE_B             interaction
    phi_A = (PE_A + N_A) / 2             A's share, in nats
    phi_B = (PE_B + N_B) / 2             B's share, in nats

    phi_A + phi_B = TE                   exactly, for every item

The sum holds by cancellation, so nothing is fitted. Shares are in nats on the outcome scale. Since I is hidden inside the two phi terms, it has its own column: both blocks can look alike with phi near TE / 2 when each suffices alone (I opposite in sign to TE) or when neither does (I the same sign as TE). Vaidyanathan et al. (notes level) make the same point about interaction terms under patching. No shared item set is lost to the identity: an item needs all four cells, or it leaves all terms and is counted.

Actor effect, as A2: mean over the actor's items, then each actor counts once.

**Primary, per model and arm:** phi_A, phi_B and I. TE is estimated with them, and gates the reading. For the token arm, A is S, B is the rest, and one more column compares the cue tokens with the placebo: phi_S - phi_Q.

**Riders, reported with actor-bootstrap 95% CI.** PE_A, PE_B, N_A, N_B. Relative L2 change of each block under the edit, as a check that the edit reached the block. Relative L2 of each hybrid from its nearest native state, which gives the size of the departure from any image the encoder can produce.

**Named reading of the blocks.** On Qwen, A is the final vision output and B the three intermediate ViT levels. On Gemma, both streams are functions of one exit tensor, so the split is in how the language model takes it in, not in what vision computed. Neither split has a human meaning.

## Inference

- **Unit.** The actor, 17 actors.
- **Test.** One matrix, 17 rows (actors), shared by both models [audit 2026-11-08: the draft said "for each eligible model" and also "maximum 16 columns", which only adds up across models]. Per model: one column for TE, three for the route arm (phi_A, phi_B, I) and four for the token arm (phi_S, phi_rest, I, phi_S - phi_Q), so at most 8 per model and 16 in all. Columns exist only for arms that ran. All two-sided, because A7 states no direction. Signs flip by whole rows, all 2^17 flips, by A2's `signflip_maxt`, step-down maxT, familywise alpha .05. The two models share the 17 actors, so one row flip applies to both, and the correction is joint. Models are not pooled into any estimate, as in A5.
- **Mean.** "Mean TE", "mean phi" and so on are the mean over the 17 actor effects.
- **Direction.** A share counts as in the effect's direction if the sign of its mean matches the sign of mean TE.
- **Floors.** Two-sided about 1.5e-5, so the actor count does not limit these tests.
- **Interval.** Actor-bootstrap by A2's `actor_interval`, `bootstrap_draws` draws, seed 42. 95% for direction, 90% for equivalence, by A2's `equivalence_pass`.
- **TE margin.** A2's own margin for that model and route, 10% of `mismatch_margin_basis`, read from A2's analysis output and not recomputed. If A2's output has no such value for the model on `single_face`, the effect cannot be Absent, and an otherwise Absent effect is Unresolved [audit 2026-11-08].
- **Share margin [proposed].** `share_margin_fraction` of |mean TE| for that model. It is on the outcome's own scale, so A2's disclosed fault of a margin from another outcome does not arise. The margin is computed once from TE before any block equivalence is read.
- **Partial crossing.** Labels are pooled within actor, and no contrast crosses the two actor sets.
- **No human side.** There is no noise ceiling and no human comparison.

## Classes and verdict

Applied per model and arm.

**Effect class.**

| Class | Requires |
|---|---|
| Present | TE step-down p < .05, 95% CI excludes 0, and `mask_eyesmouth_first` and `mask_eyesmouth_blur` have the same sign as TE on A2's cells (A2's fill rule; the sign is that of the mean actor effect of each fill against `original` on A2's stored cells, computed by A2's `actor_effects`, with no test). If either fill is missing for the model, the effect cannot be Present |
| Absent | TE not significant, 90% CI inside the TE margin |
| Unresolved | Anything else |

**Carrying class,** only when the effect is Present. A block "passes" if its phi has step-down p < .05 and a 95% CI that excludes 0.

| Class | Requires |
|---|---|
| A only | phi_A passes in the effect's direction, phi_B's 90% CI inside the share margin |
| B only | Mirror |
| Both | phi_A and phi_B pass, both in the effect's direction |
| Opposed | One passes in the effect's direction, and the other passes against it. The total is a net of two opposing effects |
| Unresolved | Anything else |

**Interaction rider,** attached to every result with a Present effect, and it never changes the verdict.

| Rider | Requires |
|---|---|
| Additive | I's 90% CI inside the share margin |
| Redundant | I passes, opposite in sign to TE. Each block can do much of the job alone |
| Synergic | I passes, same sign as TE. The blocks need each other |
| Interaction unresolved | Anything else |

**Verdict.**

| Situation | Verdict | Reading |
|---|---|---|
| Stage 0 failed for this arm, Stage 0 has no result for the model, or a sentinel failed | **Not testable** | No complete cut on this input. Required wording under G7 |
| Stage 0 passed for this arm, but the arm was not run by the recorded decision under "When Stage 1 runs at all" | **Not run** | Reported with the decision's date and reason. Not a null [audit 2026-11-08] |
| A precondition failed | **Not tested** | See the next section |
| Effect Absent | **No effect to carry** | Null at the level of the edit. Nothing to trace |
| Effect Unresolved | **Inconclusive**, `te_unresolved` | |
| Effect Present, A only | **Carried by A** | The edit's effect on this readout passes through A. B carries under the margin here, on this edit. Null for B |
| Effect Present, B only | **Carried by B** | Mirror |
| Effect Present, Both | **Both blocks**, with its rider | The rider says whether redundant, synergic or additive |
| Effect Present, Opposed | **Opposed blocks** | Net effect hides two opposed routes |
| Effect Present, carrying Unresolved | **Inconclusive**, `carrying_unresolved` | |
| Any carrying verdict (A only, B only, Both, Opposed) in an arm where either block failed C5, or where a block's relative change under the edit is zero on any item | **Inconclusive**, `block_dead` | A null for a block that cannot move the output is not a null. Applied to every carrying verdict in that arm, not only to the block whose null is claimed [audit 2026-11-08, the more conservative of the two readings] |

**Order of precedence [audit 2026-11-08].** The draft did not say which row wins when several apply. Rows are applied in this order, and the first that applies is the verdict: (1) Not testable (Stage 0 failed for the arm, or a sentinel failed); (2) Not tested (a precondition failed); (3) Inconclusive `items_missing`; (4) the effect class: Absent gives No effect to carry, Unresolved gives `te_unresolved`; (5) with the effect Present, `block_dead`; (6) the carrying class. The interaction rider and the token arm's `not_local` rider are attached after this and never change it.

**Token arm.** "Carried by A" reads "carried by cue tokens". It stands as a cue-level claim only if phi_S - phi_Q also passes, with phi_S above phi_Q in the effect's direction. Otherwise it reads "no spatial specificity over a same-size random set", and the rider `not_local` is attached.

**Reachability, by logic only.** Recovery has not run, so this is an argument. Every row has an input that produces it, as the observers in Recovery show on paper. As one check of arithmetic on equivalence: with 17 actors a 90% interval has a half-width of about 1.645 / sqrt(17) = .40 actor SDs, so a block with true share 0 reaches "inside the margin" when its actor SD is under .20 / .40 = half of |TE|. Whether real data meet this is unknown, because TE and its spread are unknown. If recovery shows a row cannot be reached at realistic sizes, the rule changes before freezing, and that change is dated.

## Preconditions

A missing gate has not passed, as in A2.

| State | A7 result |
|---|---|
| RQ0 not passed on `single_face` through `rq0_status`, or G2 to G4 or delivery not passed | **Not tested**. Stage 0 still stands as mechanics, since it reads no analysis clip. No Stage 1 run for that model |
| Gate B fails on `single_face` | **Not tested**. A2 stops the blind readout for RQ1 and RQ2 at this point, and A7's outcome is the same quantity. A5 takes a failed Gate B further because it can be the RQ4 answer. For RQ3 it cannot, because there would be no valid effect to divide |
| Sham fails | **Not tested**. A7 relies on the mask pipeline |
| Fewer than `min_item_share` of the 200 items have all cells | **Inconclusive**, `items_missing`, with failures listed by label and reason |

**If A1 fails and A3 passes,** `single_face` stays admitted by the frozen table. A7 runs only on that route, so nothing changes.

**If every eligible model is Not tested,** RQ3 is reported as such with the Stage 0 table, and no Stage 1 hours are spent.

## Null and Inconclusive

There are three distinct nulls, and none is a default.

1. **Null at the edit: No effect to carry.** All of: preconditions met, at least one Stage 1 arm ran and its sentinels passed, TE not significant, and its 90% CI inside A2's margin. [audit 2026-11-08: the draft also required "both blocks live on Stage 0", which the verdict table does not; TE is computed from the two native endpoints and passes through no hold, so block liveness cannot bear on it. The verdict table governs.]
2. **Null at a block: Carried by A, for B, or the mirror.** All of: preconditions met, effect Present, the block's 90% CI inside the share margin, both blocks live on Stage 0 (C5), and each block's relative L2 change under the edit greater than zero on every item (the rider defined under Estimands; otherwise `block_dead`). This says only that the block carries under a fifth of this effect here. It does not say the block carries no cue information. Interaction is read beside it, because a block with phi near 0 can have large opposing PE and N.
3. **Null at the location: `not_local`.** Token arm only, as above.

**Inconclusive.** Exactly these reason codes: `te_unresolved`, `carrying_unresolved`, `block_dead`, `items_missing`. (`edit_too_weak` and `edit_failed` are Stage 0 reasons for the region test not passing, not Stage 1 verdicts.)

**Inconclusive is not the default here.** A failed gate gives Not tested. A missing site gives Not testable. An arm not run by recorded decision gives Not run. Every other verdict needs only named cells, and each is exercised by a named observer in Recovery. An output row with no verdict or reason code is a code fault and stops the analysis.

## What is reported in each outcome

| Outcome | What the paper says |
|---|---|
| Stage 0 fails at the exit on a model | On this input no complete vision cut exists for that model, so RQ3 is not testable there. This also bears on A5: the `image_free` row would not be the cut forward, and A5's W needs a new caveat |
| Exit passes, whatever the secondary arms below show | The only complete cut is the whole path, whose contrast is zero by construction. RQ3 is not testable beyond the interface. This is the RQ3 result under the ruling on blocks; the rows below are secondary |
| Route arm runs, No effect to carry | Joint eyes and mouth edits do not move the blind readout enough to trace. Pixels matter by Gate B if it passed, but this edit does not reach the acted family. It is a statement about the edit |
| Route arm runs, Carried by or Both blocks | That sentence, scoped: this edit, this readout, this model, as a decomposition of the complete exit cut, in the supplement. It is about which tensor, and has no human reading. It appears with Interaction and hybrid distance. RQ3 itself stays "not testable" beyond the interface |
| Region test fails | Tokens at the cue's location do not contain the cue's effect, on calibration clips, at this edit. Cue-level mediation in the sense of RQ3 is not testable by a token cut on this model. Global attention in the vision stack would account for this |
| Region test passes, Stage 1 token arm clean | A token set at the cue's location holds the edit's effect to within 5% on calibration clips, and the decomposition of the complete exit cut places the effect in it. Under the ruling on blocks this is not a cut and not an RQ3 cut result; it is reported as secondary, and DECISIONS 2026-11-07 says G7 would then be reopened by a dated entry. It would also need a second generator to count as mediation under the notes' seven criteria, and A7 does not provide one [audit 2026-11-08: the draft called this "a cue-level cut" and "the first non-vacuous answer to RQ3's literal question"] |

**Sentences A7 never licenses.** "The cue does not reach the answer." "The model does not use route R" without "on this edit". "The path is causally necessary", from the complete cut. "Like a human", in any form. Any depth statement about the language model. "Blocking route R" or "a cut of the cue", for any block or token set [audit 2026-11-08].

## Imported, unchanged

Frozen A2 and gate code is imported and never edited. Hashes are in the design block.

| From | Names | Use |
|---|---|---|
| `scripts/vision_gates.py` | `adapter_for`, `vision_intervention`, `LayerLab`, `swap_tokens`, `image_positions`, `strip_pixels`, `stable_int`, `_cut_fn` | Exit cut, holds, token blocks |
| `scripts/run_stimulus_interventions.py` | `load_handle`, `load_clip`, `build_media`, `encode_media`, `score_blind`, `score_sequences`, `family_scores`, `BLIND_PROMPT`, `BLIND_PREFILL`, `LEXICON`, `PIXEL_KEYS` | Same blind readout on every forward |
| `scripts/stimulus_interventions.py` | `apply_condition`, `region_mask`, `signflip_maxt`, `clr` | Edits, regions, statistics |
| `scripts/analyse_stimulus_interventions.py` | `Cells`, `clr_of`, `actor_interval`, `equivalence_pass`, `load_shards`, `gate_b`, `rq0_status`, `outside_gates`, `delivery_passed`, `sham_passed`, `below_token_floor`, `align_flagged`, `canonical_rows`, `SMOKE_ITEMS` | Loading, gates, CLR, intervals |
| `scripts/readout_ladder.py` | `face_views` and its record format | Calibration crops |

A7 rows use A2's row format, with conditions named `hold_*`, so `Cells` and `clr_of` read them unchanged.

## Sensitivities

All are reported beside the primary, none changes a verdict, and all but the last use no new forward.

- Excluding the 3 smoke items.
- Excluding cells with `align_flagged`.
- Items whose human mode is the target; and the outcome read against the human modal label.
- Canonical word only.
- Clips with at least 8 tokens across the face, by `below_token_floor`, with the share excluded. On Gemma this removes about 10% of clips (`input_delivery_gemma4.json`, share .8971).
- Whole-judgement change: the RMS over the 21 family CLRs of each term, in place of the acted family, so a route's effect on other families is visible.
- Optional, with its cost below: the 43 excluded items, as a repeat, if run.

## Recovery

No recovery has run. The parameters are in the design block and are proposed. Recovery is on two levels, because A7 has two ways to fail: a fault in the hooks, and a fault in the verdict logic.

**Level 1: a toy network, CPU only, for Stage 0.** A small torch module with two routes and one hidden third route of settable strength, through the real `vision_intervention` and `LayerLab` code paths.

| Toy | Expected |
|---|---|
| No third route, `toy_leak_nats` 0 | C1 to C4 pass |
| Third route at .0005 nats | Passes, showing what the tolerance allows |
| Third route at .002 and .05 nats | C2 to C4 fail. Not testable |
| Self-attention over tokens, leak set by `toy_region_leak` .0 and .03 | Region test passes |
| Same at .08 and .30 | Region test fails |
| A dead block | C5 fails, `block_dead` |
| Non-deterministic forward | C1 fails, and the run stops |

**Level 2: synthetic observers, for verdicts.** Real 200 items and 17 actors. An observer is a table of y(A, B) on the four cells, with actor and item noise from the design block, and TE sizes from `effect_grid`, since no real size is known. Each runs through `a7_analyse.py` unchanged, with the real family size of up to 16 columns.

**Units and margin in recovery [audit 2026-11-08].** The draft gave `effect_grid`, `actor_sd` and `item_sd` without units, and no TE margin for synthetic data, though "No effect to carry" needs one (in real data it comes from A2's output). Fixed: all three are in nats of the acted family's CLR (`effect_units`), and recovery uses a TE margin of `te_margin_nats` = 0.2 nats. The SDs apply to the item-level TE and to each item-level term, not to each cell [audit 2026-11-08: the draft did not say]. Arithmetic for reachability, not a result: with actor SD .3 and item SD .5 over 6 to 15 items, an actor effect has SD of about .33 to .36 nats, so a 90% interval over 17 actors has a half-width near 1.645 x .36 / sqrt(17), about .14, under the .2 margin. At `pass_effect` 2.0 the share margin is .4 nats, and the false-carry shares .25 and .30 put the true phi_B at .5 and .6, above it. Whether real data meet the TE margin is unknown until A2 runs.

| Observer | Construction, as shares of TE | Expected |
|---|---|---|
| A only | phi_A 1, phi_B 0, I 0 | Carried by A, Additive |
| B only | Mirror | Carried by B, Additive |
| Additive | .5, .5, I 0 | Both blocks, Additive |
| Redundant | PE_A = PE_B = TE | Both blocks, Redundant |
| Synergic | PE_A = PE_B = 0 | Both blocks, Synergic |
| Opposed | phi_A 1.5, phi_B -.5 | Opposed blocks |
| Prior-only | No image dependence on any cell | No effect to carry |
| Edit does nothing | TE 0, large identity variation | No effect to carry |
| Dead block | B has no effect, and C5 failed | Inconclusive, `block_dead` |
| Dead block, effect via both | phi_A .5, phi_B .5, and C5 failed for B | Inconclusive, `block_dead` (the conservative precedence rule) |
| Edit null, dead block | TE 0, C5 failed for B | No effect to carry (block liveness does not bear on TE) |
| False carry | phi_B at `false_carry_shares`, truly above the margin | Carried by A at most `false_positive_max` |
| Token local | phi_S 1, phi_Q 0 | Carried by cue tokens |
| Token not local | phi_S = phi_Q | Rider `not_local` |
| All-null family | TE 0 and all terms 0 in every column, `all_null_replicates` replicates | Any column significant at most `false_positive_max` |

Pass rule, as in A5 and A6, at `pass_effect`: expected verdict in at least `expected_verdict_min` of replicates, any other named verdict in at most `other_named_verdict_max`, Inconclusive in at most `inconclusive_max`, and every named verdict reached at least once. Smaller effects report power and are not judged. Noise has an actor part, because A2's recovery used independent noise and so showed mechanics, not power.

**End to end.** `tests/test_a7.py` must include: the identity phi_A + phi_B = TE to 1e-9; a verdict run from a fixture on the real pinned gate paths, returning True for a real pass, because A2's second audit found a path that made every verdict Inconclusive; a test that an observer can pass Gate B4 and have every block term zero, and the reverse, so no A7 quantity duplicates an A2 check; a test that a block never reaches the output as "complete"; and, added by the 2026-11-08 audit, tests that (a) the complete-cut criterion fails when a single word logprob on a single clip exceeds 1e-3 nats, (b) the verdict precedence gives the first applicable row, (c) a missing Stage 0 file, a missing A2 margin or a missing fill gives Not testable, Unresolved or not Present respectively, never a pass, (d) `a7_stage0.py` refuses any clip id outside `crema_calibration_clips.csv`, and (e) every RQ3 output row with a block result carries the label "decomposition", never "cut".

If an observer fails, the failure is reported. Any change comes before freezing, is dated, and is not made after any A2 cell is read.

## Order of work

**Dependencies.**

| Item | Needs | Blocked by the queue? |
|---|---|---|
| Audit of this text | Nothing | No |
| Code and tests | Audit | No |
| Recovery | Code | No, CPU only |
| Freeze, tag `a7-freeze` | Recovery; before any full A2 cell is read | No. If A2 cells are read first, A7 is exploratory and says so |
| Stage 0 (**the next evidence step; needs the cluster**) | Freeze; code synced to the cluster with hashes checked (ROADMAP 2.1). No dependence on A1 or A2 results | Short GPU jobs, one per model |
| Stage 1 (secondary decomposition only) | Stage 0 pass; G1 for RQ0; A2's run and Gate B (G4); a `DECISIONS.md` entry made before any A2 analysis output is opened (see "When Stage 1 runs at all") | Yes |
| Analysis | Stage 1 complete | No, CPU only |

**Rules.** A7 adds new files only. Nothing read by pending rotation jobs changes on the cluster. A2 cells are read for A7 only by `a7_analyse.py`, after A7 is frozen. A7 does not read A2's verdicts to pick an edit.

**Code to write.**

| File | Purpose |
|---|---|
| `scripts/a7_design.py` | Parses the design block from this file, refuses a hash mismatch, checks pinned files |
| `scripts/a7_cuts.py` | Hold operators per model, on top of `vision_gates.py`: route holds on Qwen, stream holds on Gemma with L0 hidden and all 42 per-layer tensors, token holds. Refuses any language model site after the entry |
| `scripts/a7_stage0.py` | C1 to C5, the token map, the region test. Refuses any clip id not in `crema_calibration_clips.csv` |
| `scripts/a7_run.py` | Stage 1 hybrids and sentinels, in A2's row format, resumable, sharded |
| `scripts/a7_analyse.py` | Estimands, family, classes, verdicts. Refuses until frozen |
| `scripts/a7_recovery.py` | Both levels of recovery |
| `slurm_jobs/study3_a7.sh` | Array script, with a 59 minute limit per task |
| `tests/test_a7.py` | As under Recovery |

**Guard.** `a7_cuts.py` must refuse Gemma image-position sites at L24 or later, which STATE already forbids, and must refuse any site in the language model past the entry on either model, so that an incomplete cut cannot be run by mistake.

## Cost

**Basis.** A2's measured seconds per `single_face` cell: Gemma 3.3, Qwen 5.4, and `video_face`: 14.6, 17.0 (A100, bfloat16, full forwards). These include the MCQ, which A7 does not run, so they are upper bounds. A blind-only rate has not been measured. A5 quotes .31 against .59 for one blind cell over 243 items on both models, which suggests about half. The table uses the upper bound, plus A2's 10% allowance. Since the audit, the held tensors are captured during the two same-job endpoint forwards, which are costed as cells, so no separate capture forward is added. Hook overhead is unmeasured and falls inside the allowance if it holds. Model load was 26.6 s and 30.8 s (gates). The token map adds 96 vision-only passes per model, also inside the allowance.

| Item | Cells per model | Gemma GPU-h | Qwen GPU-h | Total GPU-h | Status |
|---|---|---|---|---|---|
| Stage 0, exit and route (C1 to C5): 24 clips by 7 | 168 | .17 | .28 | **.45** | **Needed for the paper** |
| Stage 0, region test: 24 by 4 | 96 | .10 | .16 | .26 | Optional. Needed to say anything on cue-level mediation other than "not shown" |
| Stage 1, endpoints in the same job: 200 by 2 [audit 2026-11-08] | 400 | .40 | .66 | 1.06 | Needed for any Stage 1 arm |
| Stage 1, route arm: 200 by 2, plus 20 sentinels | 420 | .42 | .69 | 1.12 | Optional, secondary. Needs C4 |
| Stage 1, token arm: 200 by 4, plus 20 sentinels [audit 2026-11-08: sentinels added] | 820 | .83 | 1.35 | 2.18 | Conditional, secondary. Runs only if the region test passes |
| Exploratory holds on P1 to P4, route arm with endpoints: 61 items by 4 [audit 2026-11-08: the draft ran these "if A2 gives Matches", which selects on A2's verdicts; now exploratory only and labelled so] | 244 | .25 | .40 | .65 | Optional, exploratory |
| The 43 excluded items, route arm with endpoints: 43 by 4 | 172 | .17 | .28 | .46 | Optional |
| Exit completeness on `video_face`: 24 by 4 | 96 | .43 | .50 | .93 | Optional. No contrast follows from it |
| Analysis, recovery | | 0 | 0 | 0 | CPU only |

| Scope | Gemma | Qwen | Total GPU-h |
|---|---|---|---|
| Scoped limit: Stage 0 exit and route only | .17 | .28 | **.45** |
| Stage 0 complete (the recommended cluster step) | .27 | .44 | **.71** |
| Stage 0 complete, plus endpoints and route arm | 1.09 | 1.79 | 2.88 |
| Also the token arm, if admitted | 1.92 | 3.14 | 5.06 |
| Everything listed | 2.77 | 4.32 | 7.1 |

Rows recomputed by the audit on 2026-11-08 with the draft's own formula: seconds per cell x cells / 3600 x 1.1, from the rounded row values. The draft's totals were 1.83, 3.96 and 5.4 before the endpoint and sentinel cells were added.

Molmo2: 0. For scale, A2 is about 31 GPU-hours.

**Reading the labels at lowest cost.** The draft took the endpoints from A2's cells at 0 hours; the audit moves them into the same job (2 cells per item, see Stage 1), and A2's cells become a descriptive cross-check. Each hybrid carries all 21 family scores, so every label reading (acted, human modal, whole-judgement) comes from one cell, and the MCQ is dropped. Flagged and Neutral items are not run. If only the route arm runs, 2 hybrid cells per item, beside the 2 endpoints, is the minimum the estimand allows: fewer gives PE or N for one block, and loses I.

**Fair share.** `BARON-COHEN-SL3-GPU` has fair share 0 (`STATE.md`). Stage 0 needs about 16 minutes on Gemma and 27 on Qwen at the upper bound, region test included (.27 and .44 GPU-hours). The array script caps a task at 59 minutes, and Stage 1 is sharded by item so that each shard's upper-bound estimate is under that cap: if every arm runs, about 1.65 GPU-hours on Gemma and 2.70 on Qwen, so at least 2 and 3 shards [audit 2026-11-08: recomputed after the endpoint cells were added]. `STATE.md` records a 59 minute job starting while 12 hour tasks waited, which suggests backfill. That is an inference from one case, so wall time cannot be promised. Hours are within free allocation, so no spending arises.

## Recommendation

**G7 should become a scoped limit in the main paper, supported by the Stage 0 table, with the route arm as an optional supplement.** It should not be a confirmatory headline, and a second paper is not justified on what is known.

- **Compute is not the reason.** Everything listed costs about 7.1 GPU-hours (5.4 before the audit added same-job endpoints and token-arm sentinels). The reason is what the answer would mean.
- **The literal question is vacuous,** and the depth question is not testable.
- **The route question has no bearing on the overarching question.** It is about model engineering, and it has no human reading. It would be a sound result of little consequence to the paper's claim.
- **The cue question is the one worth answering, and it will probably not pass** because of global attention in both vision stacks. Stage 0 settles this for .26 GPU-hours, which is good value on its own.
- **The paper already allows this.** `paper/v2/outline.md` states that no scenario claims a route through the model unless the amendment finishes. A scoped limit with a table of sites, with exact figures on where every other cut fails, is itself useful: it shows that an image-position patch inside the language model can return a false null on these two models.

**Concretely.** Freeze A7 and run Stage 0 complete (.71 GPU-hours). If the region test passes on a model, run the token arm. The notes located no cue-specific mediation through image tokens on faces (searched 2026-11-04, not rechecked here), so that outcome would change this recommendation. If it fails, run the route arm only if queue time is free, as a secondary decomposition, by a decision recorded before any A2 analysis output is opened. The decision and reason go in `DECISIONS.md`, per `GOALS.md`, by the study lead. This draft does not make that entry.

## Limits

- Stage 0 runs on CREMA-D calibration clips and Stage 1 on EU-Emotion. The sentinels check C4 on analysis items at run time. The leak test has no such check before Stage 1, and the token arm's phi of the rest estimates it afterwards.
- The stage 0 edit is a blur fill and the Stage 1 edit a neutral fill. The first needs no donor. A blur leaves low spatial frequencies in place, so its token footprint may differ from a neutral fill.
- Hybrids are states the encoder cannot produce. The edit is local, so they should be close to native states, and the rider measures how close. A result near a large departure should be read with care.
- Blocks are defined by architecture. Information shared by both blocks is not separated, so "carried by A" does not mean "unique to A". On Qwen, main and deepstack come from one ViT.
- Phi depends on holding the other block at x or x', averaged over both. Other orders do not exist with two blocks.
- One edit, one route, one readout. The MCQ is not bridged, since a hold would double the cost, and its option sets identify the category (A1, defect 2).
- Prefill completion may not be the same task as recognition (A2, Limits).
- CLR of the acted family on all labels pools complex and basic. No human prediction is attached, so the scope tags of A2 do not apply, and no reading by label is confirmatory.
- 17 actors bound everything, and the result is about this stimulus set.
- The model code was read at `transformers` 5.9.0, while the gates ran at 5.10.1.

## Fixed before freezing

1. Audit of this text, including the ruling on block admissibility. **Done 2026-11-08** (local audit; stricter reading taken). A second, independent audit of the code against this text is still wanted before the freeze, as A2 had.
2. Confirm the **[proposed]** values (confirmed by the 2026-11-08 audit, subject to recovery), then code, tests and recovery, with outputs under `research/reproduction/a7_recovery.json`. Local, CPU only.
3. Check that `mask_eyesmouth_blur` applies to calibration clips with the landmarks the ladder's records hold, and that they contain the rings the mask needs. **Cluster:** the committed ladder summaries hold lexicon mass by clip, not landmarks, and the clips are not in the local repo, so this check needs the clip files on the cluster. It reads frames and landmarks only, no model output. If it cannot run before the freeze, a failure at Stage 0 means the region test has not passed, with reason `edit_failed`, and is not a reason to change the edit.
4. Check that the Gemma L0 hold with all per-layer tensors held works as the `LayerLab` code implies: a toy test first (local), then Stage 0's C4 (cluster).
5. Check the 3-item figure of 256 and 576 tokens holds on the calibration clips. **Cluster**, as part of Stage 0: a clip whose token count differs from `image_tokens` stops Stage 0 for that model as a fault, since pairing clips for holds needs one layout.
6. Record `design_sha256`, and tag `a7-freeze`, before any full A2 cell is read. If A2's cells have been read first, A7 becomes exploratory and says so. This is time-critical: A2's full run follows G1 (ROADMAP 2.3), so the local items 1, 2 and 4 must be finished before the Qwen rotation arrays are analysed and A2 starts, or A7 is exploratory.

## Faults from A2's audit, and how A7 avoids them

| A2 fault | Here |
|---|---|
| A check that duplicated another | A test shows no A7 quantity equals a Gate B check |
| Null rule uncoded or unexecutable | Three named nulls, each from named outputs |
| Verdicts ignored gates | Gates are inputs, and missing means not passed |
| Paths that made every verdict Inconclusive | End-to-end test on the real pinned paths |
| Dose passed on noise | C5 is measured against a native donor difference, and the signed TE gates the reading |
| Margin from another outcome | Share margin from TE, the outcome's own scale |
| Constants outside the hash | One hashed block, parsed by the code |
| Recovery showed mechanics only | Graded effects, actor noise, a full family, plus a toy network for the hooks |

Also from this project's own record: the Qwen L4 pilot patched a site that leaked and was at first read as a null. `a7_cuts.py` refuses such sites.

## Read status of sources

F full text read, P partial, A abstract only, N known to this draft only through `causal_methods.md`, which is itself mostly excerpts. **Nothing marked N was read for this draft, and each is used only for the idea stated.** Each must be read at the level claimed before any of it goes in the paper.

| Source | Status | Used for |
|---|---|---|
| Schyns, Snoek and Daube 2022 | F, per `research/paper/v2/references.md` | Degrees of similarity, in RQ text only |
| Geiger et al. 2021, NeurIPS | N | Interchange interventions |
| Goldowsky-Dill et al. 2023, arXiv 2304.05969 | N | Path patching |
| Vaidyanathan et al. 2026, arXiv 2606.27510 | N | Interaction terms under patching |
| Mueller 2024, arXiv 2407.04690 | N, and the notes state the full experiments were unread | Overdetermination |
| Rushing and Nanda, arXiv 2402.15390 | N, and the notes state the full experiments were unread | Self-repair, avoid zero ablation |
| Grant et al. 2025, arXiv 2511.04638 | N | A null patch can matter in other contexts |
| Neo et al., ICLR 2025, arXiv 2410.07149 | N | Information spreads beyond an object's tokens |
| "Where To Look?", arXiv 2608.10758 | N, preprint with undefined metrics by the notes' account | Causal tokens outside target regions |

## Audit 2026-11-08

Local audit for ROADMAP step 0.12. No cluster access, no model run, no A1, A2 or A7 output opened. Nothing in this section is a result.

**Checked, and found consistent.**

- Design hash: the block as it stood hashed to `9126e416...d799`, as stated. After the audit's additions it hashes to **`9e40cfcfbe8761703da22cf2400c3af6cc7f05019df8b186f91e421dce50bb09`** (sha256 of the bytes between the fences, with a final newline). It is restated again at the freeze.
- Pinned files: all four A2 code hashes, both other code hashes and all five evidence hashes match the files on disk. `a3_summary_sha256` equals A2's `A3_SUMMARY_SHA256` in `scripts/stimulus_design.py` and A5's pin.
- Imported names: every name in "Imported, unchanged" exists in the module named (`BLIND_PROMPT` and `BLIND_PREFILL` are bound by tuple assignment; `SMOKE_ITEMS` is defined in `stimulus_design.py` and imported into `analyse_stimulus_interventions.py`; `face_views` is in `readout_ladder.py`). `actor_effects`, used by the fill-rule clarification, exists in the analysis module.
- Counts: A2's lexicon has 21 families and 106 words; A2's B1 rule is 1e-3; A2 stores word logprobs at 4 decimals. From `data/eu_emotions_human_entropy.json` (file names only, through `actor_id_from_eu_filename`): 243 items; Kind 9, Unfriendly 10, Jealous 7 (26 flagged) and Neutral 17; 200 items remain, all 17 actors, 6 to 15 items each. `crema_calibration_clips.csv` has 24 clips with 24 different actor prefixes. No label, vote or response distribution was read.
- Arithmetic: the draft's cost rows follow its formula; its equivalence half-width (.40 actor SDs) and floor (2 / 2^17, about 1.5e-5) are right.
- Against DECISIONS 2026-11-07 ("G7 becomes a scoped limit, resting on Stage 0") and GOALS G7: consistent once block admissibility is ruled on; the ruling below matches the decision's fallback.

**Inconsistencies, gaps and fixes.** Each fix is in the text, marked "[audit 2026-11-08]".

| # | Found | Fix | Kind |
|---|---|---|---|
| 1 | Block admissibility under G7 left open | Stricter reading: blocks and token sets are not cuts; route and token arms kept only as a secondary decomposition, never an RQ3 result | Design choice, more conservative option |
| 2 | "Cue cut" allowed a 5% leak while the text says a cut with leakage is not a cut | Renamed "cue token set"; enters only as a block of the complete exit cut | Clear fix |
| 3 | No single numeric definition of a complete cut | Section "Complete cut: the numeric criterion": all 24 clips, all 106 word logprobs, worst absolute difference at most 1e-3 nats on C1 and C2, C3 (exit) or C4 (blocks) | Clear fix |
| 4 | "Exit cut", "hold" and `image_free` used without definition | Defined from the frozen code (`_cut_fn`, `strip_pixels`, `baseline_cells`) | Clear fix |
| 5 | Endpoints from A2's stored, rounded cells, hybrids from the A7 run: every estimand mixes two runs; the sentinel's "rounding allowance" was unquantified | Endpoints and hybrids in the same job, unrounded; sentinels compare against the same job's native cell; A2's cells a descriptive cross-check; cost rows updated | Design choice, more conservative option |
| 6 | Stage 1 waited on a "scope decision" with no time fixed, so it could follow a look at A2's output for the same contrast | Decision recorded in DECISIONS before any A2 analysis output is opened, depending only on Stage 0 and compute; otherwise Stage 1 does not run | Clear fix |
| 7 | P1 to P4 holds were run "if A2 gives Matches", which selects on A2's verdicts against the draft's own rule | Exploratory only, labelled so | Clear fix |
| 8 | "One matrix ... for each eligible model" against "maximum 16 columns" | One joint matrix over the shared 17 actors, at most 8 columns per model | Clear fix |
| 9 | Null 1 required both blocks live; the verdict table did not | Verdict table governs; TE passes through no hold | Clear fix |
| 10 | No precedence among verdict rows; `block_dead` scope unclear | Precedence fixed; `block_dead` applies to every carrying verdict in an arm with a dead block | Clear fix, conservative reading |
| 11 | No verdict for an arm that Stage 0 admits but that is not run | Verdict "Not run", not a null | Clear fix |
| 12 | Token-map positions, change measure and distance undefined; S_b share cap not said per clip | Four quadrant centres; summed L2; Chebyshev distance; cap on every clip | Clear fix |
| 13 | Region-test ratio had no floor on its denominator; bootstrap unit, draws, seed and method unstated | `region_edit_min_fraction` floor on the C5 scale; clip (equal to actor) percentile bootstrap, 10,000 draws, seed 42 | Clear fix, new constant [proposed] |
| 14 | C5 statistic ambiguous (ratio of means or mean of ratios, interval or not) | Ratio of means over 24 clips, point estimate | Clear fix |
| 15 | Fill-rule sign and missing-input cases undefined (fill, A2 margin) | Sign of mean actor effect; a missing fill means not Present, a missing margin means not Absent | Clear fix |
| 16 | Recovery effect units and TE margin undefined | Nats of acted-family CLR; recovery TE margin 0.2 nats [proposed], with reachability arithmetic | Clear fix, new constant |
| 17 | Token-arm sentinels absent | 20 per model added, costed | Clear fix |
| 18 | Opening paragraph read the stored fraction as a leak percentage, which the Evidence section warns against | Reworded to the stored fraction | Clear fix |
| 19 | The token-arm outcome row called a passing token set "a cue-level cut" and "the first non-vacuous answer to RQ3" | Reworded under the ruling; G7 reopening goes through DECISIONS | Clear fix |

**Scripts and tests.** No A7 script or test exists: no file matching `*a7*` in `scripts/`, `research/scripts/` or `tests/`. Nothing was run. A Stage 0 mechanics plan is in Stage 0, and the tests the code must carry are listed under Recovery, "End to end".

**Local work still to do before the freeze** (not open questions, just work): the code listed under "Code to write", Level 1 and Level 2 recovery on CPU, the toy test of the Gemma L0 and per-layer hold, an independent audit of code against text, then the freeze and tag `a7-freeze` before any full A2 cell is read.

**Open, cluster only.**

1. Stage 0 itself, per model: C1 to C5, the token map and the region test on the 24 calibration clips. This is the next evidence step, and it decides whether any complete cut exists on `single_face` and the blind readout. Until it runs, RQ3 is "not testable" on both models.
2. That `mask_eyesmouth_blur` applies to the calibration clips with the landmarks and rings it needs (Fixed before freezing, item 3).
3. That every calibration clip gives 256 (Gemma) and 576 (Qwen) image tokens on `single_face` (item 5).
4. Code sync and hash check on the cluster before any A7 job (ROADMAP 2.1).
5. A blind-only seconds-per-cell figure, if wanted, to replace the upper bounds in Cost.
