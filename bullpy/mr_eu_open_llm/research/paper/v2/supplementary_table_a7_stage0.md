# Supplementary Table: A7 Stage 0, completeness of vision-path cuts

<!-- Built by research/scripts/make_stage0_table.py from the locked Stage 0 files; do not edit by hand. -->

On `single_face` with the blind readout, frozen A7 code (tag `a7-freeze`), run on 2026-11-10. A site is a complete cut only if every clip and every lexicon word stays within the tolerance of the image-free reference. A block is not a cut (DECISIONS 2026-11-08). Blocks: Gemma 4: A = hidden_at_image_positions_entering_L0, B = per_layer_input_at_image_positions_all_layers; Qwen3-VL: A = main, B = deepstack_0, deepstack_1, deepstack_2.

| Quantity | Gemma 4 | Qwen3-VL |
|---|---|---|
| Clips (calibration set, outside the analysis set) | 24 | 24 |
| Completeness tolerance (nats, all 106 words) | .001 | .001 |
| C1, determinism, native against a repeat: worst change (nats); pass | .000; pass | .000; pass |
| C2, exit cut equals the image-free forward: worst change (nats); pass | .000; pass | .000; pass |
| C3, exit cut does not depend on the image: worst change (nats); pass | .000; pass | .000; pass |
| C4, holding both blocks is complete: worst change (nats); pass | .000; pass | .000; pass |
| C5, block A live: ratio of denominator (floor .10) | .230; live | .802; live |
| C5, block B live: ratio of denominator (floor .10) | .885; live | .273; live |
| Exit cut complete; blocks exhaust it | True; True | True; True |
| Complete cuts shown | exit | exit |
| Region test: pass; reason | False; leak_over_limit | False; token_map_failed |
| Token map: hit rate over trials | 1.000 of 96 | .760 of 96 |
| Leak at buffers 0 and 1: point (upper) | .139 (.181); .088 (.113) | not computed |
| Arms admitted: route; token | True; False | True; False |

Inputs:

- `research/reproduction/a7_stage0_gemma4.json` (sha256 `fff021f8...`)
- `research/reproduction/a7_stage0_qwen3vl.json` (sha256 `3762c7bd...`)
