# Figure captions

Computers in Human Behavior manuscript figures. Asterisk convention: * p < .05, ** p < .01, *** p < .001 (Bonferroni-corrected where stated in Results). Chance performance = 25% (4-AFC). Accuracies use Wilson 95% confidence intervals unless noted.

## Main text

**Figure 1.** Model accuracy on EU-Emotion compared to human benchmarks.

*Note.* Left: video-only accuracies for Gemini 3 Flash, GPT-5, GPT-5 Mini, and Claude Opus 4.5 (*N* = 118 trials per model) versus the O’Reilly et al. (2016) facial-expression validation benchmark (63.00%, *N* = 1,231). Right: Gemini 3 Flash audio-only accuracy (*N* = 118) versus the Lassalle et al. (2019) UK vocal-expression benchmark (45.19%, *N* = 427). Error bars are Wilson 95% CIs for model accuracies. * *p* < .05 versus the modality-matched human benchmark; † marginal (*p* = .051). Dashed line = chance (25%). Human bars are point estimates without model CIs.

**Figure 2.** Model accuracy by mental state on the EU-Emotion task (video-only).

*Note.* Cell colour encodes per-model accuracy (%) for each mental state (cividis scale; 0% = light, 100% = dark). Rows sorted by mean accuracy across the four models (descending). M = mean accuracy across models; SD = standard deviation across models. Neutral excluded (*n* = 26 states). Trial counts per state ranged from 2–8 per model. Most difficult states included Unfriendly, Jealous, Interested, and Angry; easiest included Sad, Sad Low Intensity, Disgusted, and Worried (all *M* = 100%).

**Figure 3.** Performance tiers across 359 mental states on the Mindreading task (video-only).

*Note.* Performance tiers classify 359 mental states (*N* = 581 trials per model; fair video-evaluated subset) by mean accuracy across Gemini 3 Flash, GPT-5, GPT-5 Mini, and Claude Opus 4.5. Tier counts: Perfect = 101; High = 84; Moderate = 72; Low = 61; Zero = 41. T-marker audio-only trials without video input are excluded.

## Supplementary

**Figure S1.** Full per-mental-state accuracy on EU-Emotion (video-only), all four models.

*Note.* Horizontal bars show accuracy (%) for each non-neutral mental state and intensity variant under video-only input (*N* = 118 trials per model). Models: Gemini 3 Flash, GPT-5, GPT-5 Mini, Claude Opus 4.5. Dashed line = chance (25%). Neutral excluded. Source: `results/full_run/*_eu_emotion_video_only_results.csv` (not the pooled `per_emotion_breakdown.csv`, which mixes conditions and lacks a dataset column).

**Figure S2.** Full per-mental-state accuracy on Mindreading (video-only fair subset), all four models.

*Note.* Heatmap of accuracy (%) for each of 359 mental states × four models on the fair video-evaluated subset (*N* = 581 trials per model). Rows sorted by mean accuracy ascending. Colour scale: cividis (0–100%). This is the full-resolution companion to main-text Figure 3.

**Figure S3.** Modality ablation comparison for Gemini 3 Flash across EU-Emotion and Mindreading.

*Note.* Accuracy (%) with Wilson 95% CIs under audio-only, video-only, and multimodal input. EU-Emotion: *N* = 118 per condition. Mindreading: audio-only *N* = 1,263; multimodal *N* = 1,240 valid; video-only fair subset *N* = 581. Pairwise modality contrasts and Bonferroni-corrected *p*-values are reported in Results / Table 3. Mindreading audio-inclusive conditions should be interpreted with the spoken-label confound caveat.

**Figure S4.** Cross-dataset generalisation (EU-Emotion vs Mindreading), video-only, all four models.

*Note.* Paired bars show video-only accuracy on EU-Emotion (*N* = 118) versus Mindreading fair subset (*N* = 581). Asterisks mark Bonferroni-corrected two-proportion *z*-tests (* *p* < .05). Only Claude Opus 4.5 showed a significant EU–Mindreading difference after correction (−14.55 pp, *p*_bonf = .013).

**Figure S5.** *Not generated.* No upright/inverted orientation condition is present in `results/full_run/*_results.csv`. Available conditions are video_only, audio_only, and multimodal only.
