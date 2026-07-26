# Study 3 comparison (auto-generated)

Train: **MindReading** LoRA. Eval: **EU-Emotions** 118 trials, 4AFC stage-2.

| Model | Modality | B0 (base→FT) | Δ pp | Peak probe | Low/H high tertile (base) | RSA ρ | Patch fix |
|-------|----------|--------------|------|------------|---------------------------|-------|-----------|
| qwen2vl | video_only | 41.5%→44.9% | 3.4 | L20 18.6%→30.5% | —/— | 0.64 | 0.0% |
| llavanext | video_only | 25.4%→25.4% | 0.0 | L3 33.1%→36.4% | —/— | 0.46 | 0.0% (invalid outputs) |
| gemma4 | multimodal | 43.2%→2.5% | -40.7 | L12 18.6%→30.5% | —/— | 0.27 | 6.1% |

Regenerate: `python -m scripts.study3_summarize`
Fast pull: `./sync.sh pull-study3`
