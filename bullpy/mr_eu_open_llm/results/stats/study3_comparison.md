# Study 3 comparison (auto-generated)

Train: **MindReading** LoRA (`data/mindreading`). Eval transfer: **EU-Emotions** 118 trials, 4AFC stage-2.

| Model | Modality | B0 (base→FT) | Δ pp | Peak probe (base→FT) | RSA mean ρ | Patching fix |
|-------|----------|--------------|------|----------------------|------------|--------------|
| gemma4 | multimodal | 43.2%→2.5% | -40.7 | — | — | — |
| qwen2vl | video_only | 41.5%→44.9% | 3.4 | — | — | — |
| llavanext | video_only | 25.4%→25.4% | 0.0 | L3 33.1%→36.4% | 0.46 | 0.0% (empty outputs) |

## Missing local files

- **gemma4**: `/Users/eb2007/playground/bullpy/mr_eu_open_llm/results/rsa/baseline_vs_finetuned_gemma4_4afc.json, /Users/eb2007/playground/bullpy/mr_eu_open_llm/results/probes/baseline_gemma4_4afc/gemma4, /Users/eb2007/playground/bullpy/mr_eu_open_llm/results/probes/finetuned_gemma4_4afc/gemma4, /Users/eb2007/playground/bullpy/mr_eu_open_llm/results/patching/patching_results_gemma4_v2_4afc.json`
- **qwen2vl**: `/Users/eb2007/playground/bullpy/mr_eu_open_llm/results/rsa/baseline_vs_finetuned_qwen2vl_4afc.json, /Users/eb2007/playground/bullpy/mr_eu_open_llm/results/probes/baseline_qwen2vl_4afc/qwen2vl, /Users/eb2007/playground/bullpy/mr_eu_open_llm/results/probes/finetuned_qwen2vl_4afc/qwen2vl, /Users/eb2007/playground/bullpy/mr_eu_open_llm/results/patching/patching_results_qwen2vl_v2_4afc.json`

Regenerate: `python -m scripts.study3_summarize`
Fast pull: `./sync.sh pull-study3`