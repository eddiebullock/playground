# QC: gpt5

- Source: `/Users/eb2007/playground/bullpy/mr_eu_open_llm/study4_rmet/results/model/gpt5/rmet_eval_gpt5_full_seed42.json`
- Deterministic accuracy: **0.250** (9/36; chance=0.25)
- Binomial vs chance (greater): p=0.5637
- Binomial vs chance (two-sided): p=1.0000
- Mean sample entropy: 0.362 (26% of ln4 max); zero-entropy items: 15/36
- Mean modal agreement: 0.833
- Modal agreement when wrong: 0.8458333333333333
- Modal accuracy: 0.3333333333333333

## Verdict
- accuracy not above chance (binomial greater p>=.05)
- samples are low-entropy / high modal agreement → systematic (not near-random) responding
- wrong items still show high within-item agreement → consistently wrong, not guessing

## Comparison (acc / entropy / modal agree when wrong)

| model | acc_det | p>chance | mean H | H/Hmax | modal_agree | agree|wrong | modal_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| gpt5 | 0.250 | 0.564 | 0.362 | 0.26 | 0.833 | 0.846 | 0.333 |
| claude_opus | 0.444 | 0.009 | 0.028 | 0.02 | 0.983 | 0.970 | 0.444 |
| gemini_flash | 0.389 | 0.046 | 0.431 | 0.31 | 0.792 | 0.752 | 0.361 |
| qwen3vl | 0.472 | 0.003 | 0.059 | 0.04 | 0.972 | 0.953 | 0.472 |
| gemma4 | 0.389 | 0.046 | 0.290 | 0.21 | 0.858 | 0.848 | 0.361 |
| molmo2 | 0.278 | 0.412 | 0.754 | 0.54 | 0.664 | 0.674 | 0.250 |
