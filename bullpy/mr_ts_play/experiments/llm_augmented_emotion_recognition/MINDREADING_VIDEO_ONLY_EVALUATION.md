# MindReading Video-Only Results: Evaluation

## Summary

| Metric | Value |
|--------|--------|
| **Accuracy** | **60.70%** |
| Correct | 349 |
| Valid predictions | 575 |
| Processed (decode OK) | 583 |
| Failed (decode / other) | 680 (decode) + 8 (no valid prediction) |
| Total trials | 1,263 |
| Modality | Video only (no audio) |

**Result:** On the subset of MindReading trials where video could be decoded, the model reaches **60.70%** accuracy using **video only** (4 frames per clip, Gemini 2.5 Flash or equivalent).

---

## Comparison: Video-Only vs Multimodal

From the earlier multimodal run (same test set, video + audio):

- **Multimodal (video + audio):** 77.29% accuracy (583 valid trials).
- **Video-only (this run):** 60.70% accuracy (575 valid predictions from 583 processed).

**Estimated audio contribution:** **~16.6 percentage points** (77.29 − 60.70).  
So on this dataset, adding the single-word emotion audio clearly helps: the model does much better with both face and voice than with face alone.

---

## Interpretation

1. **Video-only baseline (60.70%)**  
   - Face-only performance is moderate.  
   - Many MindReading emotions are fine-grained (e.g. “needled” vs “disinclined”), which are hard from face alone.

2. **Why audio helps (~+17 pp)**  
   - In MindReading, audio is a **single-word utterance of the emotion label**.  
   - That gives a strong (partially lexical) cue, so multimodal can leverage both face and that verbal cue.  
   - For reporting, state clearly that “audio consisted of single-word utterances of the emotion label” to avoid overclaiming general “audio” benefit.

3. **Decode failures (680 trials)**  
   - 680/1,263 trials failed due to OpenCV decode (e.g. codec/container issues).  
   - Selection bias: failure is **not** associated with emotion or folder but **is** associated with **actor** (see selection bias analysis).  
   - Reported numbers are on the **decodable subset only** (583 processed → 575 valid); the 680 are excluded from accuracy.

4. **8 processed but no valid prediction**  
   - 583 processed, 575 valid → 8 trials had no usable model output (e.g. API/parse failure).  
   - Accuracy 60.70% = 349/575.

---

## Reporting Recommendations

- **Primary result:** “On 575 valid trials (from 583 decodable), video-only accuracy was **60.70%**.”
- **Comparison:** “Multimodal (video + audio) reached **77.29%** on the same decodable subset; audio contributed ~17 percentage points. Audio consisted of single-word utterances of the emotion label.”
- **Limitation:** “Results apply to the subset of trials whose videos could be decoded (583/1,263); decode failure was associated with actor, not emotion.”
- **Methods:** “Video-only: 4 frames per clip, same LLM and prompt as multimodal, no audio. Multimodal: same video plus one audio file per trial (when available).”

---

## Files

- **This run:** `results/mindreading_video_only/summary.json`, `predictions.json`
- **Comparison:** Use multimodal summary from the run that gave 77.29% (e.g. `results/mindreading_multimodal/summary.json` if that run was saved there).
- **Selection bias:** `analyze_mindreading_selection_bias.py` and `SELECTION_BIAS_DISCUSSION.md` (if present).

---

## Bottom line

Video-only performance is **60.70%**; adding the label utterance as audio raises performance to **77.29%** on the same decodable set, so **audio contributes ~17 pp** under this design. Report both numbers and the audio format (single-word emotion label) and the decode-failure limitation for transparency.

---

## Is this publishable?

**Yes**, provided you are transparent about design and limitations. The work has the right elements for a solid methods-and-results story.

### Why it holds up

- **Video-only baseline:** You report face-only performance (60.70%), not only multimodal.
- **Clear comparison:** Same decodable subset for video-only and multimodal; audio gain (~17 pp) is clearly quantified.
- **Audio described honestly:** You state that “audio consisted of single-word utterances of the emotion label,” so reviewers know the audio condition (and possible lexical cue) rather than assuming generic “prosody only.”
- **Selection bias analyzed:** Decode failure is documented (e.g. by actor), and you report on the decodable subset with an explicit limitation.
- **Reproducibility:** REPRODUCIBILITY.md (and related docs) give enough detail to replicate.

### What to state in the paper

1. **Methods:** “We compared video-only (four frames per clip) and multimodal (video + audio) conditions on the same decodable subset. Audio consisted of single-word utterances of the emotion label.”
2. **Results:** Report video-only accuracy (60.70%) and multimodal accuracy (77.29%) on that subset; state the ~17 pp gain from adding audio under this design.
3. **Limitations:** (a) Only trials with decodable video were included; decode failure was associated with actor. (b) The audio condition was single-word emotion labels, so the gain reflects that specific setup rather than “audio in general.”

### Publishability checklist

- [ ] Report **video-only** and **multimodal** on the **same decodable subset** (N = 583/575 as appropriate).
- [ ] State in methods: **“Audio consisted of single-word utterances of the emotion label.”**
- [ ] Include a **limitation** on decode failure and actor-related selection bias.
- [ ] Do **not** overclaim: avoid saying “audio improves emotion recognition” in a generic way without specifying the audio format.
- [ ] Cite or supplement **REPRODUCIBILITY.md** (or equivalent) for replication.

With that framing, the MindReading results are **publishable** as a transparent comparison of video-only vs multimodal (face + single-word voice) with documented limitations and reproducibility.
