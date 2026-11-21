# Table 1 | What each model has shown

<!-- Built by research/scripts/make_table1.py from locked outputs only; do not edit by hand. -->

Each cell copies a class or value from a locked output; no rule is added (Results). Models are never pooled. "Pending" means the analysis has not run. The natural-clip score supports alignment with the human pattern, not use of the same cues; whether it is confirmatory follows the face-dependence tests, not the vision gates (Methods). Randomised-cue headlines need the Western Caucasian (WC) and East Asian (EA) results to agree; "Not aligned" is always on these renders. Stimulus-edit counts are of the 16 confirmatory contrasts per model, 10 on the single face (`single_face`) and 6 on video (`video_face`). Methods labels the columns A1, A3, the RQ0 reading, natural-clip RQ1, A4, A9 and A2.

| Model | Face dependence, composite input: delta (Holm *P*) | Face dependence, single face: delta (Holm *P*) | Reading | Natural clips: pattern score (*P*) | Randomised cues: basic; conversational | Second human sample | Stimulus edits, after the fill rule: Gate B; Matches, Opposes, Null, Inconclusive |
|---|---|---|---|---|---|---|---|
| Qwen3-VL | .402 (.00009) | .244 (.00009) | Face use on both inputs | .549 (.0002), confirmatory | Inconclusive; Inconclusive | Not aligned on these renders; replication not testable | single_face: Gate B pass; 4, 0, 0, 6<br>video_face: Gate B pass; 0, 0, 0, 6 |
| Gemma 4 | .228 (.00009) | .216 (.00014) | Face use on both inputs | .418 (.0006), confirmatory | Aligned; Mixed | Aligned; replicated on a second human sample (same lab and generator) | single_face: Gate B pass; 5, 0, 0, 5<br>video_face: Gate B pass; 0, 1, 0, 5 |
| Molmo2 | .071 (.00009) | .063 (.0001) | Face use on both inputs | .543 (.0002), confirmatory | not eligible (vision gates) | not eligible (vision gates) | not eligible (vision gates) |

Inputs:

- `research/reproduction/a2_extras_gemma4.json` (sha256 `792ee9ad...`)
- `research/reproduction/a2_extras_qwen3vl.json` (sha256 `0d2e8d0d...`)
- `research/reproduction/face_input_validity_summary.json` (sha256 `c1c64e69...`)
- `research/reproduction/rotation_summary.json` (sha256 `81500bdd...`)
- `research/reproduction/same_clip_family_model_gemma4.json` (sha256 `0f9c0a26...`)
- `research/reproduction/same_clip_family_model_molmo2.json` (sha256 `9ad3d59d...`)
- `research/reproduction/same_clip_family_model_qwen3vl.json` (sha256 `bc0d6795...`)
- `research/reproduction/stimulus_interventions_gemma4.json` (sha256 `4ceb8840...`)
- `research/reproduction/stimulus_interventions_qwen3vl.json` (sha256 `ff6a7741...`)
- `results/a4/summary.json` (sha256 `08148303...`)
- `results/a9/summary.json` (sha256 `1d1a9995...`)
