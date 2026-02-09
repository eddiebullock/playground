# MindReading: Cost estimate — Opus 4.5, GPT-5, Gemini 3 Pro

Worst-case cost to run **Anthropic Opus 4.5**, **GPT-5**, and **Gemini 3 Pro** on the MindReading stimuli (same setup as EU: multimodal where supported, same prompt/reasoning).

---

## Setup

- **Trials:** `mindreading_emotions_test.json` (1,263 trials).
- **Processed:** ~583 trials typically succeed (rest fail video decode). Cost is based on **583 processed trials**.
- **Per trial (multimodal):** ~2,028 input tokens, ~155 output tokens (4 frames + audio + prompt; label + reasoning).

**Totals (583 trials):**
- Input: 583 × 2,028 = **1,182,324** tokens  
- Output: 583 × 155 = **90,365** tokens  

---

## Pricing (per 1M tokens, standard API)

| Model | Input | Output |
|-------|--------|--------|
| **Gemini 3 Pro** | $2.00 | $12.00 |
| **GPT-5** | $1.25 | $10.00 |
| **Anthropic Opus 4.5** | $5.00 | $25.00 |

*(Check [Google](https://ai.google.dev/gemini-api/docs/pricing), [OpenAI](https://platform.openai.com/docs/pricing), [Anthropic](https://www.anthropic.com/pricing) for current prices.)*

---

## Cost per model (583 trials)

| Model | Input cost | Output cost | **Total** |
|-------|------------|-------------|-----------|
| **Gemini 3 Pro** | (1.18M/1M)×$2 ≈ $2.37 | (90.4K/1M)×$12 ≈ $1.08 | **~$3.45** |
| **GPT-5** | (1.18M/1M)×$1.25 ≈ $1.48 | (90.4K/1M)×$10 ≈ $0.90 | **~$2.38** |
| **Anthropic Opus 4.5** | (1.18M/1M)×$5 ≈ $5.91 | (90.4K/1M)×$25 ≈ $2.26 | **~$8.17** |

---

## Worst-case total (all three)

| Run | Est. cost |
|-----|-----------|
| Gemini 3 Pro (video + audio) | ~$3.45 |
| GPT-5 (video-only) | ~$2.38 |
| Anthropic Opus 4.5 (video-only) | ~$8.17 |
| **Total (base)** | **~$14.00** |

**Worst-case buffer:** +20% for token variance and longer reasoning → **~$17.00** total.

So **worst case for all three on MindReading: ~\$17.**

---

## Notes

- **Gemini 3 Pro** gets video + audio; **GPT-5** and **Opus 4.5** are video-only in this setup (no audio sent to API).
- **Processed N:** Only trials that decode successfully (~583) are sent to the API; the rest are skipped. Actual cost scales with processed count.
- **Cache:** Gemini can cache inputs on repeat runs (lower input cost). First run is full price.
- Confirm model IDs and pricing before running: `gemini-3-pro-preview`, `gpt-5`, `claude-opus-4-5`.
