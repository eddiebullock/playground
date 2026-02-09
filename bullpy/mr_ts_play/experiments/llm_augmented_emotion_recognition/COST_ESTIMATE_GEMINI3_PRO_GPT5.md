# Cost Estimate: Gemini 3 Pro & GPT-5 on EU Emotions and MindReading

Rough cost to run **Gemini 3 Pro** and **GPT-5** (full) on:
- **EU emotions:** 118 trials (video + audio where available)
- **MindReading:** 583 trials (decodable subset; video + audio where available)

---

## Token assumptions (per trial, multimodal)

Same as existing multimodal runs (video + audio, with reasoning):

| Per trial | Input (tokens) | Output (tokens) |
|-----------|----------------|-----------------|
| EU / MindReading | ~2,028 | ~155 |

- **Input:** 4 frames (~1,028) + audio (~750) + prompt (~250)
- **Output:** label + reasoning (~155)

**Totals:**
- **EU emotions (118 trials):** 118 × 2,028 = **239,304** input; 118 × 155 = **18,290** output
- **MindReading (583 trials):** 583 × 2,028 = **1,182,324** input; 583 × 155 = **90,365** output

---

## Pricing (per 1M tokens, standard API)

| Model | Input | Output |
|-------|--------|--------|
| **Gemini 3 Pro** | $2.00 | $12.00 |
| **GPT-5** | $1.25 | $10.00 |

*(Gemini 3 Pro: prompts ≤200K tokens; our batches are under that. Check [Google AI pricing](https://ai.google.dev/gemini-api/docs/pricing) and [OpenAI pricing](https://platform.openai.com/docs/pricing) for latest.)*

---

## Cost per run (single dataset, one model)

### EU emotions (118 trials)

| Model | Input cost | Output cost | **Total** |
|-------|------------|-------------|-----------|
| **Gemini 3 Pro** | (239K/1M)×$2 ≈ $0.48 | (18.3K/1M)×$12 ≈ $0.22 | **~$0.70** |
| **GPT-5** | (239K/1M)×$1.25 ≈ $0.30 | (18.3K/1M)×$10 ≈ $0.18 | **~$0.48** |

### MindReading (583 trials)

| Model | Input cost | Output cost | **Total** |
|-------|------------|-------------|-----------|
| **Gemini 3 Pro** | (1.18M/1M)×$2 ≈ $2.37 | (90.4K/1M)×$12 ≈ $1.08 | **~$3.45** |
| **GPT-5** | (1.18M/1M)×$1.25 ≈ $1.48 | (90.4K/1M)×$10 ≈ $0.90 | **~$2.38** |

---

## Total cost: all four runs

| Run | Est. cost |
|-----|-----------|
| Gemini 3 Pro on EU emotions | ~$0.70 |
| Gemini 3 Pro on MindReading | ~$3.45 |
| GPT-5 on EU emotions | ~$0.48 |
| GPT-5 on MindReading | ~$2.38 |
| **Total (all four)** | **~$7.01** |

---

## Summary table

| Dataset | Trials | Gemini 3 Pro | GPT-5 |
|---------|--------|---------------|-------|
| EU emotions | 118 | **~$0.70** | **~$0.48** |
| MindReading | 583 | **~$3.45** | **~$2.38** |
| **Both** | — | **~$4.15** | **~$2.86** |

**All four runs together: ~\$7.**

---

## Notes

- **Cache:** Gemini can cache inputs on repeat runs (cheaper input); first run is full price.
- **Token variance:** Real usage can be ±15–20%; add a small buffer if you need a hard budget.
- **MindReading:** Only 583 trials are processed (videos that decode); 680 are skipped.
- **Pricing checks:** Confirm [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing) and [OpenAI pricing](https://platform.openai.com/docs/pricing) before running; names (e.g. “Gemini 3 Pro”, “GPT-5”) and prices may change.
