# Figure sources

Written by `research/scripts/make_figures.py`. Each row is one read of a committed input file. Rebuild with `python research/scripts/make_figures.py --all`.

## Figure 1

| Panel | Input file | sha256 | Script line |
|---|---|---|---|
| a, target position | `research/reproduction/prompt_confound_audit.json` | `17af332f89538948c6313166e89947939c5db8463dd841f99442d59e6ce3aa00` | `research/scripts/make_figures.py:125` |
| a, number of options | `data/eu_emotions_human_entropy.json` | `f9d91b4e47816fd01b773f0a63f8fff39c7e0cfa2c56a2136b1f52ba5cc33744` | `research/scripts/make_figures.py:131` |
| b, option-set counts | `research/reproduction/prompt_confound_audit.json` | `17af332f89538948c6313166e89947939c5db8463dd841f99442d59e6ce3aa00` | `research/scripts/make_figures.py:149` |
| b, option sets per label | `data/eu_emotions_human_entropy.json` | `f9d91b4e47816fd01b773f0a63f8fff39c7e0cfa2c56a2136b1f52ba5cc33744` | `research/scripts/make_figures.py:150` |
| c, delivery Gemma 4 | `research/reproduction/input_delivery_gemma4.json` | `94f55b9fd96767e0c19c0bbeb34eebd5a47dbbbf5b14058d53d6f15a8b40a8ca` | `research/scripts/make_figures.py:186` |
| c, delivery Qwen3-VL | `research/reproduction/input_delivery_qwen3vl.json` | `802f3231db29f2b27d67e8a6cfd70c78a9ea72c9e3fb4b1cadf7f8c966f1f801` | `research/scripts/make_figures.py:186` |
| c, delivery Molmo2 | `research/reproduction/input_delivery_molmo2.json` | `200ede896d13ccbbea15f7a5e3a561bac9dc7cc35c7ecc3286f4382dd963ec9b` | `research/scripts/make_figures.py:186` |
| d, human p_target per label | `data/eu_emotions_human_entropy.json` | `f9d91b4e47816fd01b773f0a63f8fff39c7e0cfa2c56a2136b1f52ba5cc33744` | `research/scripts/make_figures.py:223` |
| d, weak-label list | `scripts/run_stimulus_interventions.py` | `f7880c94e9671cc294cd4eaeece2108d7cd1d7a4c7e6f9e30a1c495964e6cd82` | `research/scripts/make_figures.py:224` |

## Figure 3

| Panel | Input file | sha256 | Script line |
|---|---|---|---|
| a, schematic of proposition 2 | none: diagram of `research/theory/causal_feature_alignment.md`, no plotted numbers | - | `research/scripts/make_figures.py:285` |
| b, proposition 2 | `research/reproduction/cfa_identifiability.json` | `31bd4b4098026b3438c0293a13b7e379676ff8724cb67d6b88c8f602ead9ea45` | `research/scripts/make_figures.py:310` |
| c, theorem 1 | `research/reproduction/cfa_identifiability.json` | `31bd4b4098026b3438c0293a13b7e379676ff8724cb67d6b88c8f602ead9ea45` | `research/scripts/make_figures.py:334` |
| d, lambda_min per cue table | `research/reproduction/cfa_identifiability.json` | `31bd4b4098026b3438c0293a13b7e379676ff8724cb67d6b88c8f602ead9ea45` | `research/scripts/make_figures.py:386` |
| e, condition number per cue table | `research/reproduction/cfa_identifiability.json` | `31bd4b4098026b3438c0293a13b7e379676ff8724cb67d6b88c8f602ead9ea45` | `research/scripts/make_figures.py:386` |
