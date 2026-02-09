# EU Emotions: Gemini 3 Pro & GPT-5 — Score validation

Checks on the reported accuracies for **Gemini 3 Pro** (82.05%) and **GPT-5** (75.22%) on the EU emotion test set (118 trials).

---

## 1. Arithmetic

| Model         | Correct | Valid | Accuracy   | Check        |
|---------------|---------|--------|------------|-------------|
| **Gemini 3 Pro** | 96      | 117   | 82.05%     | 96/117 = 0.8205 ✓ |
| **GPT-5**        | 85      | 113   | 75.22%     | 85/113 = 0.7522 ✓ |

**Valid** = number of trials with `is_correct` not `null` (i.e. the model returned a label that could be matched to the candidate set and scored).  
**Accuracy** = correct / valid (trials with null predictions are excluded from both numerator and denominator).  
So the reported percentages are **arithmetically correct**.

---

## 2. Why valid &lt; 118?

Some trials have **null** predictions: the API returned text that did not parse to a label in the candidate list (e.g. no clear `EMOTION: <label>` or label not in the four options).

| Model         | Processed | Valid | Null (no usable label) |
|---------------|-----------|--------|--------------------------|
| **Gemini 3 Pro** | 118       | 117    | 1 (train_274, Disgusted)  |
| **GPT-5**        | 118       | 113    | 5 (see below)             |

**Gemini null:** `train_274` (correct_label: Disgusted) — `reasoning` is also null, so the response did not yield a parseable emotion.  
**GPT-5 nulls:** 5 trials where `predicted_label` and `reasoning` are null (e.g. train_353 Hurt, and four others with correct_labels Jealous, Happy Low Intensity, Happy, Disgusted Low Intensity). Different trials from Gemini’s single null.

So:
- **Robustness:** Nulls are due to parse/response format, not a bug in the accuracy formula. Excluding them from the denominator is correct.
- **Validity:** Accuracy is “among trials where the model gave a valid forced-choice answer.” That is a standard and valid way to report.

---

## 3. How accuracy is computed (code)

From `run_multimodal_experiment.py`:

- `total_valid = len([p for p in predictions if p.get('is_correct') is not None])`
- `correct_predictions = [p for p in predictions if p.get('is_correct') is True]`
- `accuracy = len(correct_predictions) / total_valid`

So:
- Only trials with a non-null `is_correct` count in the denominator.
- Correctness is `predicted_idx == correct_idx` (or equivalent) when a label was parsed.  
This matches the intended metric.

---

## 4. Fairness of head-to-head comparison

- **Gemini 3 Pro:** 117 valid trials (1 null).  
- **GPT-5:** 113 valid trials (5 null).  
So the two accuracies are **not** on exactly the same set of trials.

For a **strict head-to-head** you can:
- Restrict to trials that have a **valid prediction in both** runs (intersection of valid trials). That gives **112 trials** (118 − 1 − 5, with no overlap between the null sets).
- Recompute accuracy for each model on those 112 trials only.

Reported as-is:
- **82.05%** (Gemini 3 Pro) and **75.22%** (GPT-5) are **valid and robust**.
- For papers/tables you can either:
  - Report as above and add a short note: “Accuracy on trials with a valid forced-choice response (Gemini 117, GPT-5 113).”
  - Or report “on the 112 trials with valid response from both models” and give two accuracies on that common set.

---

## 5. Sanity checks (no red flags)

- **No path leakage:** Model input is frames (pixels), optional audio bytes, and prompt with candidate labels only (see `DATA_SENT_TO_API.md`). No file paths or filenames are sent.
- **Forced choice:** Each trial has four candidate labels; the model must pick one. Parsing only accepts a match from that set.
- **Correct vs valid:** `correct` counts only trials where the parsed label equals the ground-truth label among the four options. So the “correct” count is valid.

---

## 6. Summary

| Check                    | Status |
|--------------------------|--------|
| Arithmetic (96/117, 85/113) | ✓      |
| Denominator = valid only  | ✓      |
| Nulls excluded from metric | ✓    |
| Parsing / code logic     | ✓      |
| Same trial set (118)     | ✓ (different valid N per model) |
| Strict head-to-head      | Optional: restrict to 112 trials valid in both |

**Conclusion:** The scores **82.05%** (Gemini 3 Pro) and **75.22%** (GPT-5) are **valid and robust**. The only nuance is that valid N differs (117 vs 113); for a strict comparison, report on the 112 trials valid in both, or state the valid N per model when reporting these numbers.
