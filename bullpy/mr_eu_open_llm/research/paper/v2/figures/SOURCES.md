# Figure sources

Written by `research/scripts/make_figures.py`. Each row is one read of a committed input file. Rebuild with `python research/scripts/make_figures.py --all`.

Figures 2, 4, 5 and 6 get a section only when built with `--real`, which refuses until every locked output in `LOCKED` exists. Synthetic test renders (`--synthetic`) are written to a temporary folder, never here.

## Figure 1

| Panel | Input file | sha256 | Script line |
|---|---|---|---|
| a, target position | `research/reproduction/prompt_confound_audit.json` | `17af332f89538948c6313166e89947939c5db8463dd841f99442d59e6ce3aa00` | `research/scripts/make_figures.py:172` |
| a, number of options | `data/eu_emotions_human_entropy.json` | `f9d91b4e47816fd01b773f0a63f8fff39c7e0cfa2c56a2136b1f52ba5cc33744` | `research/scripts/make_figures.py:178` |
| b, option-set counts | `research/reproduction/prompt_confound_audit.json` | `17af332f89538948c6313166e89947939c5db8463dd841f99442d59e6ce3aa00` | `research/scripts/make_figures.py:196` |
| b, option sets per label | `data/eu_emotions_human_entropy.json` | `f9d91b4e47816fd01b773f0a63f8fff39c7e0cfa2c56a2136b1f52ba5cc33744` | `research/scripts/make_figures.py:197` |
| c, delivery Gemma 4 | `research/reproduction/input_delivery_gemma4.json` | `94f55b9fd96767e0c19c0bbeb34eebd5a47dbbbf5b14058d53d6f15a8b40a8ca` | `research/scripts/make_figures.py:233` |
| c, delivery Qwen3-VL | `research/reproduction/input_delivery_qwen3vl.json` | `802f3231db29f2b27d67e8a6cfd70c78a9ea72c9e3fb4b1cadf7f8c966f1f801` | `research/scripts/make_figures.py:233` |
| c, delivery Molmo2 | `research/reproduction/input_delivery_molmo2.json` | `200ede896d13ccbbea15f7a5e3a561bac9dc7cc35c7ecc3286f4382dd963ec9b` | `research/scripts/make_figures.py:233` |
| d, human p_target per label | `data/eu_emotions_human_entropy.json` | `f9d91b4e47816fd01b773f0a63f8fff39c7e0cfa2c56a2136b1f52ba5cc33744` | `research/scripts/make_figures.py:270` |
| d, weak-label list | `scripts/run_stimulus_interventions.py` | `f7880c94e9671cc294cd4eaeece2108d7cd1d7a4c7e6f9e30a1c495964e6cd82` | `research/scripts/make_figures.py:271` |
| e, eight-paper table | `research/reproduction/literature_table.csv` | `48a7005d3a5ea695a4646f3e42e4117a91929c3ee7b786826e2a7cb2995820db` | `research/scripts/make_figures.py:314` |

## Figure 2

| Panel | Input file | sha256 | Script line |
|---|---|---|---|
| a to c, A1 rotation summary | `research/reproduction/rotation_summary.json` | `81500bdd229f88c5aa4038736d4ccc0eb2e04080613dc95380cc8c3dd067072a` | `research/scripts/make_figures.py:714` |
| a, b, d, A3 face input validity summary | `research/reproduction/face_input_validity_summary.json` | `c1c64e6975b32356a3314268c054f0dee483ddf74b11cd66d03b2faf972e64a4` | `research/scripts/make_figures.py:715` |

## Figure 3

| Panel | Input file | sha256 | Script line |
|---|---|---|---|
| a, schematic of proposition 2 | none: diagram of `research/theory/causal_feature_alignment.md`, no plotted numbers | - | `research/scripts/make_figures.py:390` |
| b, proposition 2 | `research/reproduction/cfa_identifiability.json` | `31bd4b4098026b3438c0293a13b7e379676ff8724cb67d6b88c8f602ead9ea45` | `research/scripts/make_figures.py:416` |
| c, theorem 1 | `research/reproduction/cfa_identifiability.json` | `31bd4b4098026b3438c0293a13b7e379676ff8724cb67d6b88c8f602ead9ea45` | `research/scripts/make_figures.py:440` |
| d, lambda_min per cue table | `research/reproduction/cfa_identifiability.json` | `31bd4b4098026b3438c0293a13b7e379676ff8724cb67d6b88c8f602ead9ea45` | `research/scripts/make_figures.py:492` |
| e, condition number per cue table | `research/reproduction/cfa_identifiability.json` | `31bd4b4098026b3438c0293a13b7e379676ff8724cb67d6b88c8f602ead9ea45` | `research/scripts/make_figures.py:492` |

## Figure 4

| Panel | Input file | sha256 | Script line |
|---|---|---|---|
| pattern score, Gemma 4 | `research/reproduction/same_clip_family_model_gemma4.json` | `0f9c0a2658cb15c4ac5e0ec03d87fa1f4a521a71ba5e758d54993e96ac37519d` | `research/scripts/make_figures.py:897` |
| pattern score, Qwen3-VL | `research/reproduction/same_clip_family_model_qwen3vl.json` | `bc0d67951cad79abb28cae1697f96946ec87e6f83eb6096d9cd772ad5cedb810` | `research/scripts/make_figures.py:897` |
| pattern score, Molmo2 | `research/reproduction/same_clip_family_model_molmo2.json` | `9ad3d59dd71b8c2a4f897ac8f05d1c9ed805f9e566c75ceb3651d2eafaf66e3f` | `research/scripts/make_figures.py:897` |
| reference band, route recovery 1x | `research/reproduction/same_clip_family_route_pattern_within_category.json` | `2bed8ba2dea187d48bd968c2048cb9ec49945ed0e22bb68e0a9946d1209cba8a` | `research/scripts/make_figures.py:889` |
| reference band, route recovery 0.75x | `research/reproduction/same_clip_family_route_pattern_075x_within_category.json` | `3cec1cf52258446df5ad7d917a048af0866fc61c43753add164362b168b56762` | `research/scripts/make_figures.py:889` |
