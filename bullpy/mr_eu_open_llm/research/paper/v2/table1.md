# Table 1 | What each model has shown

<!-- Built by research/scripts/make_table1.py from locked outputs only; do not edit by hand. -->

Each cell copies a class or value from a locked output; no rule is added (Results, section 8). Models are never pooled. "Pending" means the analysis has not run. RQ1 on natural clips supports alignment with the human pattern, not use of the same cues. A4 headlines need WC and EA to agree; "Not aligned" is always on these renders. A2 counts are of the 16 confirmatory contrasts per route.

| Model | RQ0: A1 delta (Holm *P*) | RQ0: A3 delta (Holm *P*) | RQ0 reading | RQ1, natural clips: pattern score (*P*) | RQ1, randomised (A4): basic; conversational | A4 on a second sample (A9) | RQ2 (A2), after the fill rule: Gate B; Matches, Opposes, Null, Inconclusive |
|---|---|---|---|---|---|---|---|
| Qwen3-VL | .402 (.00009) | .244 (.00009) | Face use on both inputs | .549 (.0002), confirmatory | pending | pending | pending |
| Gemma 4 | .228 (.00009) | .216 (.00014) | Face use on both inputs | .418 (.0006), confirmatory | pending | pending | pending |
| Molmo2 | .071 (.00009) | .063 (.0001) | Face use on both inputs | .543 (.0002), confirmatory | not eligible (vision gates) | not eligible (vision gates) | not eligible (vision gates) |

Inputs:

- `research/reproduction/face_input_validity_summary.json` (sha256 `c1c64e69...`)
- `research/reproduction/rotation_summary.json` (sha256 `81500bdd...`)
- `research/reproduction/same_clip_family_model_gemma4.json` (sha256 `0f9c0a26...`)
- `research/reproduction/same_clip_family_model_molmo2.json` (sha256 `9ad3d59d...`)
- `research/reproduction/same_clip_family_model_qwen3vl.json` (sha256 `bc0d6795...`)
