# Methods Section: Detailed Technical Appendix

**Generated:** 2026-02-09  
**Codebase:** `/Users/eb2007/playground/bullpy/mr_ts_play`

---

## APPENDIX A: EXACT PROMPT TEMPLATES (VERBATIM)

### A.1 Google Gemini Prompts

#### A.1.1 Video-Only Prompt (With Reasoning)
**Location:** `experiments/llm_augmented_emotion_recognition/models/llm_wrapper.py` lines 281-294

```
Analyze these video frames and identify the emotion being expressed.

The emotion must be one of these options: {labels_str}

IMPORTANT: Provide your answer FIRST, then explain your reasoning.

Format your response EXACTLY as:
EMOTION: [single emotion label from the list above]
REASONING: [your detailed reasoning explaining what you observed and why this emotion matches]

What to observe:
- Facial expressions (eyes, mouth, eyebrows)
- Body language and posture
- Overall emotional tone
```

#### A.1.2 Multimodal Prompt (Video + Audio, With Reasoning)
**Location:** `experiments/llm_augmented_emotion_recognition/models/llm_wrapper.py` lines 259-279

```
Analyze these video frames AND the accompanying audio to identify the emotion being expressed.

The emotion must be one of these options: {labels_str}

IMPORTANT: Provide your answer FIRST, then explain your reasoning.

Format your response EXACTLY as:
EMOTION: [single emotion label from the list above]
REASONING: [your detailed reasoning explaining what you observed in both visual and audio cues and why this emotion matches]

What to observe:
VISUAL CUES:
- Facial expressions (eyes, mouth, eyebrows)
- Body language and posture
- Overall emotional tone

AUDIO CUES:
- Voice tone and prosody
- Volume and intensity
- Speech patterns and rhythm
- Emotional quality of the voice
```

#### A.1.3 Video-Only Prompt (Without Reasoning)
**Location:** `experiments/llm_augmented_emotion_recognition/models/llm_wrapper.py` lines 307-316

```
Analyze these video frames and identify the emotion being expressed.

The emotion must be one of these options: {labels_str}

Consider:
- Facial expressions (eyes, mouth, eyebrows)
- Body language and posture
- Overall emotional tone

Respond with ONLY the single emotion label from the list above.
```

#### A.1.4 Multimodal Prompt (Video + Audio, Without Reasoning)
**Location:** `experiments/llm_augmented_emotion_recognition/models/llm_wrapper.py` lines 296-305

```
Analyze these video frames AND the accompanying audio to identify the emotion being expressed.

The emotion must be one of these options: {labels_str}

Consider:
- Visual cues: Facial expressions, body language, posture
- Audio cues: Voice tone, prosody, volume, speech patterns

Respond with ONLY the single emotion label from the list above.
```

### A.2 OpenAI Prompts

#### A.2.1 Video-Only Prompt (With Reasoning)
**Location:** `experiments/llm_augmented_emotion_recognition/models/llm_wrapper.py` lines 485-498

```
Analyze these video frames and identify the emotion being expressed.

The emotion must be one of these options: {labels_str}

IMPORTANT: Provide your answer FIRST, then explain your reasoning.

Format your response EXACTLY as:
EMOTION: [single emotion label from the list above]
REASONING: [your detailed reasoning explaining what you observed and why this emotion matches]

What to observe:
- Facial expressions (eyes, mouth, eyebrows)
- Body language and posture
- Overall emotional tone
```

#### A.2.2 Multimodal Prompt (Video + Audio, With Reasoning)
**Location:** `experiments/llm_augmented_emotion_recognition/models/llm_wrapper.py` lines 471-483

```
Analyze these video frames AND the accompanying audio to identify the emotion being expressed.

The emotion must be one of these options: {labels_str}

IMPORTANT: Provide your answer FIRST, then explain your reasoning.

Format your response EXACTLY as:
EMOTION: [single emotion label from the list above]
REASONING: [your detailed reasoning explaining what you observed and why this emotion matches]

What to observe:
- Visual cues: Facial expressions (eyes, mouth, eyebrows), body language, posture
- Audio cues: Tone of voice, prosody, volume, speech patterns, emotional quality of the voice
```

#### A.2.3 Video-Only Prompt (Without Reasoning)
**Location:** `experiments/llm_augmented_emotion_recognition/models/llm_wrapper.py` lines 511-519

```
Analyze these video frames and identify the emotion being expressed.

The emotion must be one of these options: {labels_str}

Consider:
- Facial expressions (eyes, mouth, eyebrows)
- Body language and posture
- Overall emotional tone

Respond with ONLY the single emotion label from the list above.
```

#### A.2.4 Multimodal Prompt (Video + Audio, Without Reasoning)
**Location:** `experiments/llm_augmented_emotion_recognition/models/llm_wrapper.py` lines 500-509

```
Analyze these video frames AND the accompanying audio to identify the emotion being expressed.

The emotion must be one of these options: {labels_str}

Consider:
- Visual cues: Facial expressions, body language, posture
- Audio cues: Voice tone, prosody, volume, speech patterns

Respond with ONLY the single emotion label from the list above.
```

### A.3 Anthropic Claude Prompts

#### A.3.1 Video-Only Prompt (With Reasoning)
**Location:** `experiments/llm_augmented_emotion_recognition/models/llm_wrapper.py` lines 592-605

```
Analyze these video frames and identify the emotion being expressed.

The emotion must be one of these options: {labels_str}

IMPORTANT: Provide your answer FIRST, then explain your reasoning.

Format your response EXACTLY as:
EMOTION: [single emotion label from the list above]
REASONING: [your detailed reasoning explaining what you observed and why this emotion matches]

What to observe:
- Facial expressions (eyes, mouth, eyebrows)
- Body language and posture
- Overall emotional tone
```

**Note:** Claude API does not support audio input, so only video-only prompts are used.

#### A.3.2 Video-Only Prompt (Without Reasoning)
**Location:** `experiments/llm_augmented_emotion_recognition/models/llm_wrapper.py` lines 607-616

```
Analyze these video frames and identify the emotion being expressed.

The emotion must be one of these options: {labels_str}

Consider:
- Facial expressions (eyes, mouth, eyebrows)
- Body language and posture
- Overall emotional tone

Respond with ONLY the single emotion label from the list above.
```

### A.4 System Messages

**None used** - All prompts are user messages only. No system prompts are included in any API calls.

### A.5 Message Format Specifications

#### Google Gemini Format
```python
# From llm_wrapper.py lines 332-354
parts = [
    {"text": prompt},
    *image_parts,  # List of {"inline_data": {"mime_type": "image/jpeg", "data": base64_string}}
]
# If audio:
parts.append({
    "inline_data": {
        "mime_type": mime_type,  # "audio/mpeg", "audio/wav", "audio/mp4", "audio/ogg"
        "data": audio_base64
    }
})

payload = {
    "contents": [{"parts": parts}],
    "generationConfig": {...},
    "safetySettings": [...]
}
```

#### OpenAI Format
```python
# From llm_wrapper.py lines 533-544
content = [
    {"type": "text", "text": prompt},
    *[{"type": "image_url", "image_url": {"url": url, "detail": detail}} for url in image_urls]
]
# If audio:
content.append({
    "type": "input_audio",
    "input_audio": {
        "data": audio_base64,
        "format": audio_format  # "wav", "mp3", "m4a", "ogg"
    }
})

messages = [{"role": "user", "content": content}]
```

#### Anthropic Claude Format
```python
# From llm_wrapper.py lines 622-644
message = client.messages.create(
    model=self.vision_model,
    max_tokens=1000 if include_reasoning else 50,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            *image_media  # List of {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_string}}
        ]
    }]
)
```

---

## APPENDIX B: HYPERPARAMETERS TABLE

### B.1 Complete Hyperparameters by Model-Task Combination

| Model | Task | Temperature | Max Tokens (Reasoning) | Max Tokens (No Reasoning) | Vision Detail | Max Frames | Request Delay (s) | Max Retries | Base Delay (s) | Max Backoff (s) | Timeout (s) |
|-------|------|-------------|------------------------|---------------------------|---------------|------------|-------------------|-------------|----------------|-----------------|-------------|
| **Google Gemini 2.5 Flash** | All | 0.1 | 1000 | 200 | N/A | 4 | 2.5 | 10 | 3.0 | 90.0 | 60 |
| **Google Gemini 3 Flash** | All | 0.1 | 1000 | 200 | N/A | 4 | 2.5 | 10 | 3.0 | 90.0 | 60 |
| **Google Gemini 3 Pro** | All | 0.1 | 1000 | 200 | N/A | 4 | 2.5 | 10 | 3.0 | 90.0 | 60 |
| **OpenAI GPT-4o-mini** | All | 0.1 | 1000 | 50 | low/high | 4 | N/A | SDK | SDK | SDK | SDK |
| **OpenAI GPT-4o** | All | 0.1 | 1000 | 50 | low/high | 4 | N/A | SDK | SDK | SDK | SDK |
| **OpenAI GPT-5 Mini** | All | 0.1 | 1000 | 50 | low | 4 | N/A | SDK | SDK | SDK | SDK |
| **OpenAI GPT-5** | All | 1.0* | 1000 | 50 | low | 4 | N/A | SDK | SDK | SDK | SDK |
| **Anthropic Claude Sonnet** | All | Default | 1000 | 50 | N/A | 4 | N/A | SDK | SDK | SDK | SDK |
| **Anthropic Claude Opus 4.5** | All | Default | 1000 | 50 | N/A | 4 | N/A | SDK | SDK | SDK | SDK |

*Note: GPT-5 (full) uses default temperature (1.0) - API does not allow temperature parameter to be set.

**File Locations:**
- Google Gemini: `llm_wrapper.py` lines 367-368, 391-396, 398-400, 404
- OpenAI: `llm_wrapper.py` lines 558, 560-561
- Anthropic: `llm_wrapper.py` line 638

### B.2 Safety Settings (Google Gemini Only)

```python
# From llm_wrapper.py lines 370-387
safetySettings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    }
]
```

---

## APPENDIX C: EXTERNAL DEPENDENCIES

### C.1 Core Dependencies (requirements.txt)

**File Location:** `requirements.txt` (root directory)

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pyyaml>=6.0
opencv-python>=4.8.0
pillow>=10.0.0
transformers>=4.30.0
einops>=0.7.0
```

### C.2 Additional Dependencies (Not in requirements.txt, but used)

These are imported in the code but not explicitly listed in requirements.txt:

- **requests** - Used for Google Gemini REST API calls
  - **Location:** `llm_wrapper.py` line 19
  - **Usage:** HTTP POST requests to Gemini API

- **python-dotenv** - Used for loading environment variables
  - **Location:** `llm_wrapper.py` line 20
  - **Usage:** Loading API keys from `.env` files

- **openai** - OpenAI Python SDK
  - **Location:** `llm_wrapper.py` line 460
  - **Usage:** OpenAI API client

- **anthropic** - Anthropic Python SDK
  - **Location:** `llm_wrapper.py` line 583
  - **Usage:** Anthropic Claude API client

- **base64** - Standard library (no install needed)
  - **Usage:** Encoding images and audio for API

- **io.BytesIO** - Standard library (no install needed)
  - **Usage:** In-memory image/audio buffer handling

### C.3 Python Version

- **Python:** 3.8+ (as documented in `REPRODUCIBILITY.md`)

### C.4 Environment Variables

- **GOOGLE_API_KEY** - Google AI API key
- **OPENAI_API_KEY** - OpenAI API key
- **ANTHROPIC_API_KEY** - Anthropic API key
- **GOOGLE_GEMINI_REQUEST_DELAY** - Optional delay between Gemini requests (default: 2.5)

---

## APPENDIX D: HARD-CODED VALUES AND MAGIC NUMBERS

### D.1 Video Processing

| Value | Purpose | Location | Justification |
|-------|---------|----------|---------------|
| **4** | Number of frames per video | `run_mindreading_multimodal_experiment.py` line 146, `llm_wrapper.py` line 74, 320, 527, 623 | Uniform sampling across video length |
| **[:4]** | Frame limit in loops | `llm_wrapper.py` lines 320, 527, 623 | Ensures max 4 frames sent to API |
| **JPEG** | Image format | `llm_wrapper.py` lines 322, 529, 625 | Standard format for API encoding |
| **BGR to RGB** | Color space conversion | `run_mindreading_multimodal_experiment.py` line 86 | OpenCV uses BGR, PIL uses RGB |

### D.2 Random Seeds

| Value | Purpose | Location | Justification |
|-------|---------|----------|---------------|
| **42** | Global random seed | Multiple files | Standard ML seed for reproducibility |
| **hash(trial_id) % (2**31)** | Per-trial seed | `run_mindreading_multimodal_experiment.py` line 269 | Deterministic per-trial randomization |

### D.3 Candidate Label Generation

| Value | Purpose | Location | Justification |
|-------|---------|----------|---------------|
| **3** | Number of foil labels | `run_mindreading_multimodal_experiment.py` line 267 | 4-alternative forced-choice (1 correct + 3 foils) |
| **4** | Total candidate labels | `run_mindreading_multimodal_experiment.py` line 278 | Standard forced-choice format |

### D.4 API Rate Limiting and Retries

| Value | Purpose | Location | Justification |
|-------|---------|----------|---------------|
| **2.5** | Default request delay (seconds) | `llm_wrapper.py` line 392 | 25 RPM limit = 2.4s between requests, rounded to 2.5s |
| **10** | Max retries | `llm_wrapper.py` line 398 | Reasonable limit for transient failures |
| **3.0** | Base delay (seconds) | `llm_wrapper.py` line 399 | Initial backoff delay |
| **90.0** | Max backoff (seconds) | `llm_wrapper.py` line 400 | Maximum wait time between retries |
| **60** | Request timeout (seconds) | `llm_wrapper.py` line 404 | Reasonable timeout for API calls |

### D.5 Cache Versioning

| Value | Purpose | Location | Justification |
|-------|---------|----------|---------------|
| **"mindreading_multimodal_1.0"** | Cache version (multimodal) | `run_mindreading_multimodal_experiment.py` line 197 | Allows cache invalidation |
| **"mindreading_video_only_1.0"** | Cache version (video-only) | `run_mindreading_multimodal_experiment.py` line 197 | Separate cache for video-only experiments |

### D.6 Accuracy Rounding

| Value | Purpose | Location | Justification |
|-------|---------|----------|---------------|
| **4** | Decimal places for accuracy | `run_mindreading_multimodal_experiment.py` line 389 | Standard precision for reporting |

### D.7 Progress Logging

| Value | Purpose | Location | Justification |
|-------|---------|----------|---------------|
| **50** | Logging interval | `run_mindreading_multimodal_experiment.py` line 252 | Log every 50 skipped trials to reduce verbosity |

### D.8 Audio File Matching

| Value | Purpose | Location | Justification |
|-------|---------|----------|---------------|
| **"1"** | Default audio folder | `run_mindreading_multimodal_experiment.py` line 120 | MindReading dataset has 3 audio folders (1, 2, 3) |
| **"1w.aif"** | Audio file suffix pattern | `mindreading_audio_matcher.py` line 75 | MindReading audio naming convention |

### D.9 Frame Sampling

| Value | Purpose | Location | Justification |
|-------|---------|----------|---------------|
| **int(i * total_frames / num_frames)** | Frame index calculation | `run_mindreading_multimodal_experiment.py` line 79 | Uniform sampling across video |

---

## APPENDIX E: EVALUATION METRICS IMPLEMENTATIONS

### E.1 Overall Accuracy

**Location:** `experiments/llm_augmented_emotion_recognition/evaluation/metrics.py` lines 18-32

```python
def compute_accuracy(predictions: List[Dict]) -> float:
    """
    Compute overall accuracy.
    
    Args:
        predictions: List of prediction dictionaries with 'correct_label' and 'predicted_label'
    
    Returns:
        Overall accuracy (0.0 to 1.0)
    """
    if not predictions:
        return 0.0
    
    correct = sum(1 for p in predictions if p['correct_label'] == p['predicted_label'])
    return correct / len(predictions)
```

**Also implemented in:** `run_mindreading_multimodal_experiment.py` lines 368-370

```python
correct_predictions = [p for p in predictions if p.get('is_correct') is True]
total_valid = len([p for p in predictions if p.get('is_correct') is not None])
accuracy = len(correct_predictions) / total_valid if total_valid > 0 else 0.0
```

### E.2 Per-Emotion Accuracy

**Location:** `experiments/llm_augmented_emotion_recognition/evaluation/metrics.py` lines 35-56

```python
def compute_per_emotion_accuracy(predictions: List[Dict]) -> Dict[str, float]:
    """
    Compute accuracy for each emotion label.
    
    Args:
        predictions: List of prediction dictionaries
    
    Returns:
        Dictionary mapping emotion labels to accuracy scores
    """
    emotion_counts = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred in predictions:
        emotion = pred['correct_label']
        emotion_counts[emotion]['total'] += 1
        if pred['correct_label'] == pred['predicted_label']:
            emotion_counts[emotion]['correct'] += 1
    
    return {
        emotion: counts['correct'] / counts['total'] if counts['total'] > 0 else 0.0
        for emotion, counts in emotion_counts.items()
    }
```

**Also implemented in:** `run_mindreading_multimodal_experiment.py` lines 372-389

```python
from collections import defaultdict
by_emotion = defaultdict(lambda: {"count": 0, "correct": 0})
for p in predictions:
    correct_label = p.get("correct_label") or ""
    if not correct_label:
        continue
    is_correct = p.get("is_correct")
    if is_correct is None:
        continue
    by_emotion[correct_label]["count"] += 1
    if is_correct is True:
        by_emotion[correct_label]["correct"] += 1
per_emotion = {}
for emotion in sorted(by_emotion.keys()):
    d = by_emotion[emotion]
    n, c = d["count"], d["correct"]
    per_emotion[emotion] = {"count": n, "correct": c, "accuracy": round(c / n, 4) if n > 0 else 0.0}
```

### E.3 Confusion Matrix

**Location:** `experiments/llm_augmented_emotion_recognition/evaluation/metrics.py` lines 59-97

```python
def compute_confusion_matrix(predictions: List[Dict], normalize: bool = True) -> pd.DataFrame:
    """
    Compute confusion matrix for predictions.
    
    Args:
        predictions: List of prediction dictionaries
        normalize: If True, normalize rows to show proportions
    
    Returns:
        DataFrame with confusion matrix
    """
    all_labels = set()
    for pred in predictions:
        all_labels.add(pred['correct_label'])
        all_labels.add(pred['predicted_label'])
    
    all_labels = sorted(list(all_labels))
    label_to_idx = {label: i for i, label in enumerate(all_labels)}
    
    n = len(all_labels)
    cm = np.zeros((n, n), dtype=float)
    
    for pred in predictions:
        true_idx = label_to_idx[pred['correct_label']]
        pred_idx = label_to_idx[pred['predicted_label']]
        cm[true_idx, pred_idx] += 1
    
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums
    
    cm_df = pd.DataFrame(
        cm,
        index=all_labels,
        columns=all_labels,
    )
    
    return cm_df
```

**Note:** Normalization is row-normalized (each row sums to 1.0), representing the proportion of predictions for each true label.

### E.4 Correctness Determination

**Location:** `run_mindreading_multimodal_experiment.py` lines 307-335

```python
# Find predicted index in candidate_labels
predicted_idx = None
if predicted_label:
    try:
        predicted_idx = candidate_labels.index(predicted_label)
    except ValueError:
        # Try case-insensitive match
        for idx, label in enumerate(candidate_labels):
            if label.lower() == predicted_label.lower():
                predicted_idx = idx
                break

# Find correct index
correct_idx = trial.get('correct_idx')
if correct_idx is None:
    try:
        correct_idx = candidate_labels.index(correct_label)
    except ValueError:
        # Fallback: try case-insensitive
        for idx, label in enumerate(candidate_labels):
            if label.lower() == correct_label.lower():
                correct_idx = idx
                break
        if correct_idx is None:
            logger.warning(f"Trial {trial_id}: Correct label '{correct_label}' not in candidate_labels")
            correct_idx = None

is_correct = (predicted_idx == correct_idx) if predicted_idx is not None and correct_idx is not None else None
```

---

## APPENDIX F: EXPERIMENTAL CONDITIONS, ABLATIONS, AND PARAMETER SWEEPS

### F.1 Modality Ablation

#### F.1.1 Video-Only Condition
- **Purpose:** Baseline comparison to assess audio contribution
- **Implementation:** Same trials, same model, `--video-only` flag (disables audio)
- **Cache Version:** `mindreading_video_only_1.0` (separate from multimodal cache)
- **Results:** MindReading dataset - 60.70% accuracy (video-only) vs 77.29% (multimodal)
- **Audio Contribution:** ~16.6 percentage points
- **File Location:** `experiments/llm_augmented_emotion_recognition/MINDREADING_VIDEO_ONLY_EVALUATION.md`

#### F.1.2 Multimodal Condition
- **Purpose:** Full evaluation with video + audio
- **Implementation:** `--use-audio` flag, audio files matched to trials
- **Cache Version:** `mindreading_multimodal_1.0`
- **Results:** MindReading dataset - 77.29% accuracy
- **File Location:** `experiments/llm_augmented_emotion_recognition/REPRODUCIBILITY.md`

### F.2 Prompt Variations (Optimization Study)

**Note:** These were tested on validation set (54 trials) for optimization, not on final test set.

#### F.2.1 Baseline Prompt
- **Status:** Evaluated
- **Validation Accuracy:** 68.52% (37/54)
- **Description:** Default prompt with EMOTION/REASONING format
- **File Location:** `experiments/llm_augmented_emotion_recognition/OPTIMIZATION_STATUS.md`

#### F.2.2 Explicit Distinctions Prompt
- **Status:** Evaluated
- **Validation Accuracy:** 70.37% (38/54)
- **Improvement:** +1.85 percentage points
- **Modifications:** Adds explicit instructions for confusing emotion pairs:
  - Afraid vs. Surprised distinction
  - Worried vs. Afraid distinction
  - Worried vs. Afraid Low Intensity distinction
- **File Location:** `experiments/llm_augmented_emotion_recognition/scripts/create_enhanced_prompts.py` lines 43-91

#### F.2.3 Intensity-Aware Prompt
- **Status:** Planned (not evaluated on final test set)
- **Modifications:** Explicitly considers intensity as a dimension
- **File Location:** `experiments/llm_augmented_emotion_recognition/scripts/create_enhanced_prompts.py` lines 94-122

#### F.2.4 Temporal Analysis Prompt
- **Status:** Planned (not evaluated on final test set)
- **Modifications:** Requests frame-by-frame progression analysis
- **File Location:** `experiments/llm_augmented_emotion_recognition/scripts/create_enhanced_prompts.py` lines 123-152

#### F.2.5 Combined Prompt
- **Status:** Planned (not evaluated on final test set)
- **Modifications:** Combines all individual modifications
- **File Location:** `experiments/llm_augmented_emotion_recognition/scripts/create_enhanced_prompts.py` lines 254-313

**File Location:** `experiments/llm_augmented_emotion_recognition/PROMPT_OPTIMIZATION_PROTOCOL.md`

### F.3 Model Comparison (No Parameter Sweeps)

**Note:** No systematic parameter sweeps were performed. Models were evaluated with fixed hyperparameters.

#### F.3.1 Provider Comparison
- **Google:** Gemini 2.5 Flash, Gemini 3 Flash, Gemini 3 Pro
- **OpenAI:** GPT-4o-mini, GPT-4o, GPT-5 Mini, GPT-5
- **Anthropic:** Claude Sonnet, Claude Opus 4.5

#### F.3.2 Model Size Comparison
- **Mini vs Full:** GPT-5 Mini (72.03%) vs GPT-5 (75.22%) on EU-Emotion
- **Flash vs Pro:** Gemini 3 Flash (77.12%) vs Gemini 3 Pro (82.05%) on EU-Emotion

### F.4 Dataset Variations

#### F.4.1 EU-Emotion Dataset
- **Test Set:** 118 trials (`eu_emotion_test_final.json`)
- **Validation Set:** 54 trials (`eu_emotion_val_for_optimization.json`) - used for optimization only
- **Emotions:** 27 emotion classes
- **Modality:** Video + audio (where supported)

#### F.4.2 MindReading Dataset
- **Test Set:** 1,263 trials (`mindreading_emotions_test.json`)
- **Processed:** ~583 trials (rest fail video decode)
- **Valid Predictions:** ~572 trials (some have null predictions)
- **Emotions:** 425 unique emotion labels
- **Modality:** Video + audio (where supported)

### F.5 Audio Folder Variation

- **Options:** Audio folders 1, 2, 3 (MindReading dataset)
- **Default:** Folder 1
- **Implementation:** `--audio-folder` argument (choices: '1', '2', '3')
- **File Location:** `run_mindreading_multimodal_experiment.py` lines 118-122

### F.6 Frame Count Variation

- **Default:** 4 frames per video
- **Configurable:** `--num-frames` argument
- **Note:** All reported results use 4 frames
- **File Location:** `run_mindreading_multimodal_experiment.py` lines 144-147

### F.7 Vision Detail Variation (OpenAI Only)

- **Options:** `low` (default) or `high`
- **Impact:** Affects image processing cost and quality
- **Default:** `low` (used in all reported experiments)
- **File Location:** `llm_wrapper.py` line 51, `llm_config.yaml` line 21

### F.8 Reasoning Inclusion Variation

- **Options:** `include_reasoning=True` (default) or `False`
- **Impact:** Changes max_tokens (1000 vs 50/200) and prompt format
- **Default:** `True` (all reported experiments include reasoning)
- **File Location:** `llm_wrapper.py` lines 257, 295, 368, 558, 638

---

## APPENDIX G: RESPONSE PARSING IMPLEMENTATION

### G.1 Emotion and Reasoning Parser

**Location:** `experiments/llm_augmented_emotion_recognition/models/llm_wrapper.py` lines 655-711

```python
def _parse_emotion_and_reasoning(self, text: str, candidate_labels: List[str], include_reasoning: bool = True) -> Tuple[Optional[str], Optional[str]]:
    """Parse emotion label and reasoning from LLM response.
    
    Returns:
        Tuple of (predicted_label, reasoning)
    """
    if not text:
        return None, None
    
    reasoning = None
    emotion = None
    
    text_upper = text.upper()
    
    # Try to parse structured response - EMOTION should come FIRST now
    if include_reasoning and ("EMOTION:" in text_upper or "emotion:" in text):
        # Split on EMOTION: (case-insensitive) - emotion should be first
        parts = re.split(r'EMOTION:\s*', text, flags=re.IGNORECASE)
        if len(parts) >= 2:
            # First part might have REASONING: prefix, second part is emotion
            emotion_part = parts[1].strip()
            # Try to split emotion from reasoning if both are present
            if "REASONING:" in emotion_part.upper() or "reasoning:" in emotion_part:
                emotion_reasoning = re.split(r'REASONING:\s*', emotion_part, flags=re.IGNORECASE, maxsplit=1)
                if len(emotion_reasoning) >= 2:
                    emotion = self._parse_emotion_from_response(emotion_reasoning[0].strip(), candidate_labels)
                    reasoning = emotion_reasoning[1].strip() if emotion_reasoning[1].strip() else None
                else:
                    emotion = self._parse_emotion_from_response(emotion_part, candidate_labels)
                    reasoning = None
            else:
                # Just emotion, no reasoning marker
                emotion = self._parse_emotion_from_response(emotion_part, candidate_labels)
                # Check if there's reasoning after the emotion (without marker)
                if len(parts) > 2:
                    reasoning = parts[2].strip()
                elif len(emotion_part) > 100:  # If emotion_part is long, it might contain reasoning
                    # Try to extract just the emotion label
                    emotion = self._parse_emotion_from_response(emotion_part.split('\n')[0], candidate_labels)
                    reasoning = '\n'.join(emotion_part.split('\n')[1:]).strip() if len(emotion_part.split('\n')) > 1 else None
    elif include_reasoning and ("REASONING:" in text_upper or "reasoning:" in text):
        # Has REASONING: but no EMOTION: - response might be incomplete or in wrong order
        # Try to extract reasoning and search for emotion anywhere in the text
        reasoning_parts = re.split(r'REASONING:\s*', text, flags=re.IGNORECASE, maxsplit=1)
        if len(reasoning_parts) >= 2:
            reasoning = reasoning_parts[1].strip()
        else:
            reasoning = text
        # Try to find emotion anywhere in the text
        emotion = self._parse_emotion_from_response(text, candidate_labels)
    else:
        # Fallback: try to parse emotion from entire text
        emotion = self._parse_emotion_from_response(text, candidate_labels)
        if include_reasoning:
            reasoning = text  # Use full response as reasoning
    
    return emotion, reasoning
```

### G.2 Emotion Label Parser

**Location:** `experiments/llm_augmented_emotion_recognition/models/llm_wrapper.py` lines 713-736

```python
def _parse_emotion_from_response(self, text: str, candidate_labels: List[str]) -> Optional[str]:
    """Parse emotion label from LLM response."""
    if not text:
        return None
    
    text_lower = text.lower().strip()
    
    # Remove common punctuation
    text_clean = text_lower.rstrip('.,!?;:')
    
    # Try exact match first
    for label in candidate_labels:
        if label.lower().strip() == text_clean:
            return label
    
    # Try partial match (in case response is "The emotion is X")
    for label in candidate_labels:
        label_lower = label.lower().strip()
        if label_lower in text_lower or text_lower in label_lower:
            return label
    
    # If no match, return the cleaned text (might still work)
    logger.warning(f"Could not parse emotion from response: '{text}'")
    return text_clean
```

---

**END OF DETAILED TECHNICAL APPENDIX**
