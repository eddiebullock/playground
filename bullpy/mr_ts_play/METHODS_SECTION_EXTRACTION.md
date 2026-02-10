# Methods Section: LLM Evaluation Benchmark Study

**Generated:** 2026-02-09  
**Codebase:** `/Users/eb2007/playground/bullpy/mr_ts_play`

---

## 1. MODELS EVALUATED

### 1.1 LLM Models Evaluated

#### Google Models

**1. Google Gemini 2.5 Flash** (Baseline/Initial Model)
- **Model Identifier:** `gemini-2.5-flash`
- **Provider:** Google AI (Generative Language API)
- **API Endpoint:** `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`
- **Type:** API-based, cloud-hosted
- **Vision Capability:** Yes (multimodal vision-language model)
- **Audio Capability:** Yes (supports audio input via base64 encoding)
- **Datasets Evaluated:** MindReading (primary), EU-Emotion (baseline)
- **File Location:** `experiments/llm_augmented_emotion_recognition/models/llm_wrapper.py` (lines 235-451)

**2. Google Gemini 3 Flash**
- **Model Identifier:** `gemini-3-flash-preview`
- **Provider:** Google AI
- **Vision Capability:** Yes
- **Audio Capability:** Yes
- **Datasets Evaluated:** EU-Emotion
- **Results:** 77.12% accuracy (118 valid trials, video + audio)
- **File Location:** `results/eu_emotion_gemini3_flash/summary.json`, `RUN_EU_THREE_MODELS.md`

**3. Google Gemini 3 Pro**
- **Model Identifier:** `gemini-3-pro-preview`
- **Provider:** Google AI
- **Vision Capability:** Yes
- **Audio Capability:** Yes
- **Datasets Evaluated:** EU-Emotion, MindReading
- **Results (EU-Emotion):** 82.05% accuracy (117 valid trials, video + audio)
- **File Location:** `results/eu_emotion_gemini3_pro/summary.json`, `RUN_EU_GEMINI3_PRO_GPT5.md`, `RUN_MINDREADING_THREE_MODELS.md`

#### OpenAI Models

**4. OpenAI GPT-4o-mini** (Early Comparison Model)
- **Model Identifier:** `gpt-4o-mini`
- **Provider:** OpenAI
- **Vision Capability:** Yes
- **Audio Capability:** No (video-only in this implementation)
- **Vision Detail:** `low` (default) or `high`
- **Datasets Evaluated:** EU-Emotion (early experiments)
- **File Location:** `experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml`

**5. OpenAI GPT-4o** (Early Comparison Model)
- **Model Identifier:** `gpt-4o`
- **Provider:** OpenAI
- **Vision Capability:** Yes
- **Audio Capability:** No (video-only in this implementation)
- **Vision Detail:** `low` (default) or `high`
- **Datasets Evaluated:** EU-Emotion (early experiments)
- **File Location:** `experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml`

**6. OpenAI GPT-5 Mini**
- **Model Identifier:** `gpt-5-mini`
- **Provider:** OpenAI
- **Vision Capability:** Yes
- **Audio Capability:** No (video-only in this implementation)
- **Datasets Evaluated:** EU-Emotion
- **Results:** 72.03% accuracy (118 valid trials, video-only)
- **File Location:** `results/eu_emotion_gpt5_mini/summary.json`, `RUN_EU_THREE_MODELS.md`, `EU_EMOTIONS_RESULTS_SUMMARY.md`

**7. OpenAI GPT-5** (Full Model)
- **Model Identifier:** `gpt-5`
- **Provider:** OpenAI
- **Vision Capability:** Yes
- **Audio Capability:** No (video-only in this implementation; note: `gpt-audio` variant exists but not used)
- **Datasets Evaluated:** EU-Emotion, MindReading
- **Results (EU-Emotion):** 75.22% accuracy (113 valid trials, video-only)
- **Results (MindReading):** 65.73% accuracy (572 valid predictions out of 583 processed, video-only)
- **File Location:** `results/eu_emotion_gpt5/summary.json`, `results/mindreading_gpt5/summary.json`, `RUN_EU_GEMINI3_PRO_GPT5.md`, `RUN_MINDREADING_THREE_MODELS.md`

#### Anthropic Models

**8. Anthropic Claude Sonnet** (Early Comparison Model)
- **Model Identifier:** `claude-3-5-sonnet-20241022`
- **Provider:** Anthropic
- **Vision Capability:** Yes (images only)
- **Audio Capability:** No (Claude API does not support audio input)
- **Datasets Evaluated:** EU-Emotion (early experiments)
- **File Location:** `experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml`

**9. Anthropic Claude Opus 4.5**
- **Model Identifier:** `claude-opus-4-5`
- **Provider:** Anthropic
- **Vision Capability:** Yes (images only)
- **Audio Capability:** No (Claude API does not support audio input)
- **Datasets Evaluated:** EU-Emotion, MindReading
- **Results (EU-Emotion):** 72.97% accuracy (111 valid trials, video-only)
- **File Location:** `results/eu_emotion_opus_4_5/summary.json`, `RUN_EU_THREE_MODELS.md`, `RUN_MINDREADING_THREE_MODELS.md`, `EU_EMOTIONS_RESULTS_SUMMARY.md`

### 1.2 Model Evaluation Summary

#### Models Evaluated on EU-Emotion Dataset (118 trials)
| Provider | Model | Valid N | Accuracy | Modality | Results Location |
|----------|-------|---------|----------|----------|------------------|
| Google | Gemini 3 Flash | 118 | 77.12% | Video + Audio | `results/eu_emotion_gemini3_flash/` |
| Google | Gemini 3 Pro | 117 | 82.05% | Video + Audio | `results/eu_emotion_gemini3_pro/` |
| OpenAI | GPT-5 Mini | 118 | 72.03% | Video-only | `results/eu_emotion_gpt5_mini/` |
| OpenAI | GPT-5 | 113 | 75.22% | Video-only | `results/eu_emotion_gpt5/` |
| Anthropic | Opus 4.5 | 111 | 72.97% | Video-only | `results/eu_emotion_opus_4_5/` |

**File Location:** `experiments/llm_augmented_emotion_recognition/EU_EMOTIONS_RESULTS_SUMMARY.md`

#### Models Evaluated on MindReading Dataset (1,263 trials, ~583 processed)
| Provider | Model | Valid N | Accuracy | Modality | Results Location |
|----------|-------|---------|----------|----------|------------------|
| Google | Gemini 2.5 Flash | ~572 | ~65.7% | Video + Audio | Baseline model |
| Google | Gemini 3 Pro | ~583 | Reported | Video + Audio | `results/mindreading_gemini3_pro/` |
| OpenAI | GPT-5 | 572 | 65.73% | Video-only | `results/mindreading_gpt5/summary.json` |
| Anthropic | Opus 4.5 | ~583 | Reported | Video-only | `results/mindreading_opus_4_5/` |

**File Location:** `results/mindreading_gpt5/summary.json`, `RUN_MINDREADING_THREE_MODELS.md`

### 1.2 Model Configurations

#### Google Gemini Configuration
```python
# From llm_wrapper.py lines 362-388
generationConfig = {
    "temperature": 0.1,
    "maxOutputTokens": 1000 if include_reasoning else 200,
}
safetySettings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]
```

#### OpenAI Configuration
```python
# From llm_wrapper.py lines 555-562
openai_create_kwargs = {
    "model": self.vision_model,
    "messages": messages,
    "max_completion_tokens": 1000 if include_reasoning else 50,
}
# Note: GPT-5 models only allow default temperature (1.0)
# GPT-4o, GPT-4o-mini, GPT-5-mini use temperature=0.1
# GPT-5 (full) uses default temperature (1.0) - not configurable
if "gpt-5" not in (self.vision_model or "").lower():
    openai_create_kwargs["temperature"] = 0.1
```

**Note on GPT-5 Models:**
- **GPT-5 Mini:** Uses temperature=0.1 (same as GPT-4o models)
- **GPT-5 (full):** Uses default temperature (1.0) - API does not allow temperature parameter
- **Audio Support:** GPT-5 models do not support `input_audio` in this implementation (video-only)

#### Anthropic Configuration
```python
# From llm_wrapper.py lines 636-646
message = client.messages.create(
    model=self.vision_model,
    max_tokens=1000 if include_reasoning else 50,
    messages=[...]
)
```

**Note on Anthropic Models:**
- **Claude Sonnet:** `claude-3-5-sonnet-20241022` (early experiments)
- **Claude Opus 4.5:** `claude-opus-4-5` (main evaluation)
- **Audio Support:** Claude API does not support audio input (video-only for all Claude models)
- **Temperature:** Not explicitly set (uses model default)

**File Location:** `experiments/llm_augmented_emotion_recognition/models/llm_wrapper.py`

---

## 2. TASKS & DATASETS

### 2.1 MindReading Dataset

#### Dataset Description
- **Dataset Name:** MindReading Emotions Library
- **Source:** Internal dataset (path: `/Volumes/MindReading/Emotions`)
- **Task Type:** 4-alternative forced-choice emotion recognition
- **Modalities:** Video-only and multimodal (video + audio)

#### Dataset Statistics
- **Total trials:** 1,263 (test set)
- **Processed:** 583 trials
- **Failed:** 680 trials (video decoding issues)
- **Valid predictions:** 572 trials
- **Emotions:** 425 unique emotion labels

**File Location:** `results/mindreading_gpt5/summary.json`

#### Data Structure
Each trial contains:
```json
{
  "trial_id": "mindreading_trial_02030",
  "stimulus_path": "05/0507601/0507601M4Vcommitted.mov",
  "correct_label": "committed",
  "candidate_labels": ["committed", "vulnerable", "cheeky", "appreciated"],
  "correct_idx": 0,
  "emotion": "committed",
  "code": "0507601",
  "folder": "05"
}
```

**File Location:** `data/trial_definitions/mindreading_emotions_test.json`

#### Audio Data
- **Location:** `/Volumes/MindReading/Emotions/Audio`
- **Format:** Single-word utterances of emotion labels
- **Folders:** Three audio folders (1, 2, 3) - default uses folder 1
- **Note:** "Audio consisted of single-word utterances of the emotion label." (from REPRODUCIBILITY.md)

**File Location:** `experiments/llm_augmented_emotion_recognition/REPRODUCIBILITY.md` (line 33)

#### Train/Test Split
- **Split Method:** Generated with seed 42
- **Script:** `experiments/llm_augmented_emotion_recognition/scripts/create_mindreading_trials.py`
- **Output Files:**
  - `mindreading_emotions_train.json`
  - `mindreading_emotions_test.json`
  - `mindreading_emotions_all.json`

**File Location:** `experiments/llm_augmented_emotion_recognition/REPRODUCIBILITY.md` (lines 35-46)

### 2.2 EU-Emotion Dataset

#### Dataset Description
- **Dataset Name:** EU-Emotion Stimulus Set
- **Source:** `/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions`
- **Task Type:** 4-alternative forced-choice emotion recognition
- **Emotions:** 27 emotion classes

#### Dataset Statistics
- **Test Set (Final):** 118 trials
- **Validation Set (for optimization):** 54 trials
- **Test Set (for fusion):** Separate test set available

**File Location:** `data/trial_definitions/eu_emotion_test_final.json`

#### Data Structure
Each trial contains:
```json
{
  "stimulus_path": "emotions 36/HD Version - Face, Body, Social/Faces - HD Version/EDITED/Joking/EF12_cut.mp4",
  "correct_label": "Joking",
  "trial_id": "train_10"
}
```

**File Location:** `data/trial_definitions/eu_emotion_test_final.json` (lines 7-50)

#### Split Organization
- **Validation Set:** `eu_emotion_val_for_optimization.json` (54 trials) - Used for prompt optimization
- **Test Set:** `eu_emotion_test_final.json` (118 trials) - Used ONCE for final evaluation
- **Reorganization Date:** 2026-01-17
- **Purpose:** Ensures no test set optimization (proper ML evaluation practice)

**File Location:** `experiments/llm_augmented_emotion_recognition/scripts/reorganize_splits.py`

### 2.3 CAM Dataset (Referenced)

#### Dataset Description
- **Dataset Name:** CAM Face-Voice Battery
- **Source:** Golan et al. (2006) methodology
- **Task Type:** 4-alternative forced-choice emotion recognition
- **Concepts:** 20 emotion concepts, 5 trials per concept = 100 total trials

#### Dataset Statistics
- **Total trials:** 100
- **Face trials:** 51 (all valid)
- **Voice trials:** 49 (some corrupted)
- **Train/Test Split:** 80/20 (concept-balanced)

**File Location:** `experiments/cam_human_like/training/CAM_DATASET_BREAKDOWN.md`

---

## 3. EVALUATION PROTOCOL

### 3.1 Prompting Strategy

#### Primary Prompt (Default with Reasoning)
**Zero-shot evaluation** - No few-shot examples provided.

**Prompt Template (Video-only):**
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

**Prompt Template (Multimodal - Video + Audio):**
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

**File Location:** `experiments/llm_augmented_emotion_recognition/models/llm_wrapper.py` (lines 256-316)

#### Prompt Version
- **Version:** `default_with_reasoning` (prompt_version: default_with_reasoning)
- **Location:** `llm_wrapper._classify_google_gemini` method
- **Format:** EMOTION: label, REASONING: text

**File Location:** `experiments/llm_augmented_emotion_recognition/scripts/run_mindreading_multimodal_experiment.py` (lines 7-9)

### 3.2 Prompt Variations (Available but not used in primary experiments)

#### 1. Baseline Prompt
- Standard prompt with EMOTION/REASONING format
- **Status:** Already evaluated (68.52% on old test set)

#### 2. Explicit Distinctions Prompt
- Adds explicit instructions for confusing emotion pairs
- Distinctions for: Afraid vs. Surprised, Worried vs. Afraid, etc.

#### 3. Intensity-Aware Prompt
- Explicitly considers intensity as a dimension
- Requests intensity assessment (low vs. full)

#### 4. Temporal Analysis Prompt
- Requests frame-by-frame progression analysis
- Identifies peak expression and transitions

#### 5. Combined Prompt
- Combines all modifications above

**File Location:** `experiments/llm_augmented_emotion_recognition/scripts/create_enhanced_prompts.py`

### 3.3 System Prompts
- **None used** - Direct user prompts only, no system prompts

### 3.4 Input/Output Format

#### Input Format
- **Video:** 4 frames per video (uniform sampling)
- **Frame Format:** JPEG (converted from video frames)
- **Audio:** Base64-encoded audio files (when available)
- **Candidate Labels:** List of 4 emotion labels (1 correct + 3 foils)

#### Output Format
- **Structured Response:** `EMOTION: [label] REASONING: [text]`
- **Parsing:** Case-insensitive matching to candidate labels
- **Fallback:** If no exact match, assigns to first candidate

**File Location:** `experiments/llm_augmented_emotion_recognition/models/llm_wrapper.py` (lines 655-736)

### 3.5 Prompt Engineering
- **Optimization Protocol:** Systematic prompt optimization on validation set
- **Validation Set:** `eu_emotion_val_for_optimization.json` (54 trials)
- **Test Set:** `eu_emotion_test_final.json` (118 trials) - Used ONCE after optimization

**File Location:** `experiments/llm_augmented_emotion_recognition/PROMPT_OPTIMIZATION_PROTOCOL.md`

---

## 4. HYPERPARAMETERS & SETTINGS

### 4.1 Temperature Settings

#### Google Gemini
- **Temperature:** `0.1` (low temperature for deterministic outputs)
- **Location:** `llm_wrapper.py` line 367

#### OpenAI Models
- **Temperature:** `0.1` (for GPT-4o, GPT-4o-mini)
- **Exception:** GPT-5 models use default temperature (1.0) - not configurable
- **Location:** `llm_wrapper.py` lines 560-561

#### Anthropic Claude
- **Temperature:** Not explicitly set (uses default)
- **Location:** `llm_wrapper.py` line 636

### 4.2 Max Tokens / Sequence Length

#### Google Gemini
- **With reasoning:** `1000` tokens
- **Without reasoning:** `200` tokens
- **Location:** `llm_wrapper.py` line 368

#### OpenAI
- **With reasoning:** `1000` completion tokens
- **Without reasoning:** `50` completion tokens
- **Location:** `llm_wrapper.py` line 558

#### Anthropic Claude
- **With reasoning:** `1000` tokens
- **Without reasoning:** `50` tokens
- **Location:** `llm_wrapper.py` line 638

### 4.3 Sampling Parameters
- **Top-p:** Not explicitly set (uses model defaults)
- **Top-k:** Not explicitly set (uses model defaults)
- **Repetition penalty:** Not set (uses model defaults)

### 4.4 Stop Sequences
- **Not used** - Models generate until max_tokens or natural completion

### 4.5 Video Processing Parameters

#### Frame Extraction
- **Number of frames:** `4` frames per video (default)
- **Sampling method:** Uniform sampling
- **Frame indices:** `[int(i * total_frames / num_frames) for i in range(num_frames)]`
- **Location:** `run_mindreading_multimodal_experiment.py` lines 64-92

#### Frame Format
- **Format:** JPEG
- **Conversion:** BGR to RGB (OpenCV to PIL)
- **Location:** `run_mindreading_multimodal_experiment.py` lines 85-87

### 4.6 Audio Processing Parameters
- **Format:** Base64-encoded audio files
- **MIME types supported:** `.mp3`, `.wav`, `.m4a`, `.ogg`
- **Location:** `llm_wrapper.py` lines 341-347

### 4.7 Random Seeds
- **Seed:** `42` (used consistently across all experiments)
- **Usage:**
  - Trial generation
  - Candidate label generation
  - Train/test splits
- **Location:** Multiple files, documented in `REPRODUCIBILITY.md` line 15

**File Location:** `experiments/llm_augmented_emotion_recognition/REPRODUCIBILITY.md`

### 4.8 Batch Sizes
- **Not applicable** - Sequential API calls (not batched)
- **Rate limiting:** 2.5s delay between Gemini requests (configurable via `GOOGLE_GEMINI_REQUEST_DELAY` env var)
- **Location:** `llm_wrapper.py` lines 391-396

---

## 5. METRICS & SCORING

### 5.1 Primary Metrics

#### Overall Accuracy
- **Definition:** Proportion of correct predictions
- **Formula:** `correct_predictions / total_valid_predictions`
- **Implementation:**
```python
correct_predictions = [p for p in predictions if p.get('is_correct') is True]
total_valid = len([p for p in predictions if p.get('is_correct') is not None])
accuracy = len(correct_predictions) / total_valid if total_valid > 0 else 0.0
```
- **File Location:** `run_mindreading_multimodal_experiment.py` lines 368-370

#### Per-Emotion Accuracy
- **Definition:** Accuracy computed separately for each emotion label
- **Implementation:**
```python
by_emotion = defaultdict(lambda: {"count": 0, "correct": 0})
for p in predictions:
    correct_label = p.get("correct_label") or ""
    is_correct = p.get("is_correct")
    if is_correct is None:
        continue
    by_emotion[correct_label]["count"] += 1
    if is_correct is True:
        by_emotion[correct_label]["correct"] += 1
per_emotion = {
    emotion: {"count": n, "correct": c, "accuracy": round(c / n, 4) if n > 0 else 0.0}
    for emotion, (n, c) in by_emotion.items()
}
```
- **File Location:** `run_mindreading_multimodal_experiment.py` lines 372-389

### 5.2 Additional Metrics (Available)

#### Confusion Matrix
- **Normalization:** Row-normalized (proportions)
- **Implementation:** `experiments/llm_augmented_emotion_recognition/evaluation/metrics.py` lines 59-97

#### Top-k Accuracy
- **Not computed** - 4-alternative forced-choice task (top-1 only)

#### F1 Score
- **Available but not used** - See `src/evaluation/metrics.py` lines 88-91

### 5.3 Scoring Implementation

#### Correctness Determination
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

is_correct = (predicted_idx == correct_idx) if predicted_idx is not None and correct_idx is not None else None
```
- **File Location:** `run_mindreading_multimodal_experiment.py` lines 307-335

### 5.4 Statistical Significance Testing
- **Not implemented** - No statistical tests reported

### 5.5 Inter-Annotator Agreement
- **Not applicable** - No human evaluation involved

---

## 6. IMPLEMENTATION DETAILS

### 6.1 Programming Language and Version
- **Language:** Python 3.8+
- **File Location:** `experiments/llm_augmented_emotion_recognition/REPRODUCIBILITY.md` line 7

### 6.2 Key Libraries and Versions

#### Core Dependencies (from requirements.txt)
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

#### Additional Dependencies (for LLM experiments)
- `opencv-python` - Video frame extraction
- `PIL` (Pillow) - Image processing
- `requests` - HTTP API calls (for Google Gemini REST API)
- `python-dotenv` - Environment variable loading
- `openai` - OpenAI API client (for GPT models)
- `anthropic` - Anthropic API client (for Claude models)

**File Location:** `requirements.txt` and `experiments/llm_augmented_emotion_recognition/README.md`

### 6.3 Hardware Used
- **Not explicitly documented** - Experiments run on local machines and HPC clusters
- **GPU:** Not required (API-based models)
- **CPU:** Standard CPU sufficient for video frame extraction

### 6.4 API Rate Limiting and Batching

#### Google Gemini
- **Rate Limit:** 25 requests per minute (RPM)
- **Request Delay:** 2.5 seconds between requests (configurable via `GOOGLE_GEMINI_REQUEST_DELAY` env var)
- **Implementation:**
```python
request_delay = float(os.environ.get("GOOGLE_GEMINI_REQUEST_DELAY", "2.5"))
if request_delay > 0:
    time.sleep(request_delay)
```
- **File Location:** `llm_wrapper.py` lines 391-396

#### Retry Logic
- **Max Retries:** 10 attempts
- **Base Delay:** 3.0 seconds
- **Max Backoff:** 90.0 seconds
- **Exponential Backoff:** `delay = min(base_delay ** (attempt + 1), max_backoff)`
- **Implementation:**
```python
max_retries = 10
base_delay = 3.0
max_backoff = 90.0
for attempt in range(max_retries):
    try:
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 429:
            if attempt < max_retries - 1:
                delay = min(base_delay ** (attempt + 1), max_backoff)
                logger.warning(f"Rate limited. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
```
- **File Location:** `llm_wrapper.py` lines 398-426

#### OpenAI and Anthropic
- **Rate Limiting:** Handled by SDK (automatic retries)
- **No explicit delay** - SDK manages rate limits

### 6.5 Caching Mechanisms

#### Cache Strategy
- **Location:** `{cache_dir}/emotion_classification_{cache_key}_{cache_version}.json`
- **Cache Key Components:**
  - Provider name
  - Vision model name
  - Video path
  - Audio path (if applicable)
  - Candidate labels (sorted)
  - Cache version
  - Prompt hash (if custom prompt)
- **Cache Version:** `mindreading_multimodal_1.0` (multimodal) or `mindreading_video_only_1.0` (video-only)

#### Cache Implementation
```python
def _get_cache_key(self, video_path, candidate_labels, custom_prompt=None, audio_path=None):
    key_parts = [
        self.provider,
        self.vision_model,
        video_path or "no_path",
        audio_path or "no_audio",
        ",".join(sorted(candidate_labels)),
        self.cache_version
    ]
    if custom_prompt:
        prompt_hash = hashlib.md5(custom_prompt.encode()).hexdigest()[:8]
        key_parts.append(f"prompt_{prompt_hash}")
    key_str = "|".join(key_parts)
    return hashlib.md5(key_str.encode()).hexdigest()
```
- **File Location:** `llm_wrapper.py` lines 101-117

#### Cache Behavior
- **Failures not cached:** Failed API calls (predicted_label=None) are not cached to allow retries
- **Resume capability:** Scripts can resume from existing predictions
- **File Location:** `llm_wrapper.py` lines 228-231

### 6.6 Error Handling

#### API Errors
- **429 Rate Limit:** Exponential backoff retry (up to 10 attempts)
- **Network Errors:** Logged and retried
- **Timeout:** 60 seconds per request
- **Safety Filter Blocks:** Logged as warning, returns None

#### File Errors
- **Missing Videos:** Raises FileNotFoundError (or skipped with `--skip-failed`)
- **Corrupted Videos:** Logged as warning, skipped
- **Missing Audio:** Logged as warning, continues with video-only

#### Implementation
```python
try:
    # API call
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 429 and attempt < max_retries - 1:
        # Retry with backoff
    else:
        logger.error(f"Error calling API: {e}")
        return None, None
except Exception as e:
    logger.error(f"Error: {e}")
    return None, None
```
- **File Location:** `llm_wrapper.py` lines 402-426

### 6.7 Total Runtime/Computational Cost

#### Cost Estimates (from documentation)
- **Gemini 2.5 Flash:** ~$0.13 per test run (MindReading dataset)
- **GPT-4o-mini:** ~$0.02 per experiment run
- **GPT-4o:** ~$0.25 per experiment run
- **Gemini 3 Pro:** ~$0.50 per test run (estimated)
- **GPT-5.2 Pro:** ~$6.02 per test run (estimated)

**File Location:** `experiments/llm_augmented_emotion_recognition/PREMIUM_MODELS_COST.md`

#### Runtime
- **Not explicitly documented** - Depends on API response times and rate limits
- **Bottleneck:** API rate limits (2.5s delay between Gemini requests)

---

## 7. EXPERIMENTAL DESIGN

### 7.1 Number of Runs
- **Single run per experiment** - No multiple runs reported
- **Reproducibility:** Ensured via caching (same results on re-run)

### 7.2 Randomization Procedures

#### Trial Generation
- **Seed:** 42
- **Method:** Hash-based seeding for per-trial randomization
```python
trial_seed = hash(trial_id) % (2**31)
random.seed(trial_seed)
```
- **File Location:** `run_mindreading_multimodal_experiment.py` lines 268-279

#### Candidate Label Generation
- **Method:** Correct label + 3 random foils
- **Selection:** Random sampling from all emotions (excluding correct label)
- **Shuffling:** Random shuffle of candidate labels
- **File Location:** `run_mindreading_multimodal_experiment.py` lines 260-279

### 7.3 Cross-Validation
- **Not used** - Single train/test split

### 7.4 Handling Non-Deterministic Outputs
- **Temperature:** Low temperature (0.1) for deterministic outputs
- **Caching:** All API responses cached to ensure reproducibility
- **Note:** GPT-5 models use default temperature (1.0) - may have some variability

### 7.5 Ablation Studies

#### Modality Comparison
- **Video-only baseline:** Same trials, no audio
- **Multimodal:** Video + audio
- **Comparison script:** `compare_mindreading_modalities.py`
- **File Location:** `experiments/llm_augmented_emotion_recognition/REPRODUCIBILITY.md` lines 110-119

#### Prompt Variations
- **Multiple prompt variations tested** (see Section 3.2)
- **Optimization on validation set** before final test evaluation

---

## 8. DATA PIPELINE

### 8.1 Step-by-Step Flow

#### Step 1: Trial Definition Creation
```bash
python experiments/llm_augmented_emotion_recognition/scripts/create_mindreading_trials.py \
  --data-root "/Volumes/MindReading/Emotions" \
  --output-dir data/trial_definitions \
  --seed 42
```
- **Output:** JSON files with trial definitions
- **File Location:** `experiments/llm_augmented_emotion_recognition/REPRODUCIBILITY.md` lines 35-46

#### Step 2: (Optional) Video Re-encoding
- **Purpose:** Fix corrupted videos that fail to decode
- **Script:** `reencode_mindreading_videos.py`
- **Output:** Re-encoded videos and updated trial definitions
- **File Location:** `experiments/llm_augmented_emotion_recognition/REPRODUCIBILITY.md` lines 48-71

#### Step 3: Video Frame Extraction
- **Method:** Uniform sampling of 4 frames per video
- **Implementation:**
```python
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(Image.fromarray(frame_rgb))
```
- **File Location:** `run_mindreading_multimodal_experiment.py` lines 64-92

#### Step 4: Audio File Matching (Multimodal)
- **Method:** Find audio files matching trial IDs
- **Script:** `find_audio_files_for_trials()` from `mindreading_audio_matcher.py`
- **Matching Strategy:**
  1. Extract emotion code (7 digits) from video path (e.g., `0100104` from `01/0100104/0100104M1Vhumiliating.mov`)
  2. Extract emotion label from video filename (e.g., `humiliating` from `0100104M1Vhumiliating.mov`)
  3. Look for audio file: `{code}{emotion}_{folder}w.aif` (e.g., `0100104humiliating_1w.aif`)
  4. Audio location: `{audio_base_dir}/{audio_folder}/emotion/` (default folder: "1")
- **File Location:** `run_mindreading_multimodal_experiment.py` lines 179-191, `mindreading_audio_matcher.py` lines 44-96

#### Step 5: LLM Classification
- **Input:** Video frames (and optionally audio)
- **Process:** API call with prompt and media
- **Output:** Predicted label and reasoning
- **File Location:** `run_mindreading_multimodal_experiment.py` lines 295-301

#### Step 6: Result Collection
- **Format:** JSON predictions file
- **Fields:** trial_id, stimulus_path, candidate_labels, correct_label, predicted_label, is_correct, scores, reasoning
- **File Location:** `run_mindreading_multimodal_experiment.py` lines 337-350

#### Step 7: Metric Calculation
- **Metrics:** Overall accuracy, per-emotion accuracy
- **Output:** summary.json, per_emotion.json
- **File Location:** `run_mindreading_multimodal_experiment.py` lines 367-427

### 8.2 Intermediate Processing Steps
- **Frame conversion:** BGR (OpenCV) → RGB (PIL)
- **Image encoding:** PIL Image → JPEG → Base64 (for API)
- **Audio encoding:** Audio file → Base64 (for API)

### 8.3 Quality Control Measures
- **Video validation:** Check if video can be opened and has frames
- **File size check:** Minimum 50KB for valid files (mentioned in documentation)
- **Error handling:** Skip failed trials with `--skip-failed` flag
- **Resume capability:** Load existing predictions to resume interrupted runs

### 8.4 Output Storage
- **Predictions:** `predictions.json` (full prediction details)
- **Summary:** `summary.json` (aggregate metrics)
- **Per-emotion:** `per_emotion.json` (per-emotion accuracy)
- **Location:** `{output_dir}/` (default: `results/mindreading_multimodal/`)

---

## 9. REPRODUCIBILITY INFORMATION

### 9.1 Random Seeds

#### Seed Value
- **Seed:** `42` (used consistently)

#### Seed Usage Locations
1. **Trial generation:** `create_mindreading_trials.py` (seed 42)
2. **Candidate label generation:** `random.seed(42)` in main script
3. **Per-trial randomization:** `trial_seed = hash(trial_id) % (2**31)`
4. **Train/test splits:** `seed=42` in split creation functions

**File Locations:**
- `experiments/llm_augmented_emotion_recognition/REPRODUCIBILITY.md` line 15
- `run_mindreading_multimodal_experiment.py` line 218
- `src/utils/seed.py` (utility function)

### 9.2 Software Versions

#### Requirements File
- **Location:** `requirements.txt` (root directory)
- **Content:** See Section 6.2

#### Python Version
- **Version:** Python 3.8+
- **File Location:** `experiments/llm_augmented_emotion_recognition/REPRODUCIBILITY.md` line 7

### 9.3 Environment Variables

#### API Keys
- **Google:** `GOOGLE_API_KEY`
- **OpenAI:** `OPENAI_API_KEY`
- **Anthropic:** `ANTHROPIC_API_KEY`
- **Location:** `.env` file (multiple possible locations checked)

#### Configuration Variables
- **GOOGLE_GEMINI_REQUEST_DELAY:** Delay between Gemini requests (default: 2.5)
- **File Location:** `llm_wrapper.py` line 392

### 9.4 Configuration Files

#### LLM Config
- **File:** `experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml`
- **Contents:** Provider settings, model names, cache configuration

#### Experiment Configs
- **CAM:** `configs/cam_config.yaml`, `cam_config_emotion.yaml`, etc.
- **EU-Emotion:** `experiments/eu_emotion_model_comparison/configs/comparison_config.yaml`

### 9.5 Docker/Conda Environments
- **Not documented** - Standard Python virtual environment assumed

### 9.6 Cache Versioning
- **Cache Version:** `mindreading_multimodal_1.0` (multimodal) or `mindreading_video_only_1.0` (video-only)
- **Purpose:** Allows cache invalidation when methodology changes
- **File Location:** `run_mindreading_multimodal_experiment.py` line 197

---

## 10. EXCLUSIONS & FILTERING

### 10.1 Excluded Examples

#### Failed Video Decoding
- **Count:** 680 out of 1,263 trials (MindReading test set)
- **Reason:** Videos fail to decode with OpenCV
- **Handling:** Skipped with `--skip-failed` flag
- **File Location:** `results/mindreading_gpt5/summary.json` (failed: 680)

#### Missing Audio Files
- **Handling:** Falls back to video-only when audio not found
- **Logging:** "Found audio files for {audio_found}/{len(trials)} trials"
- **File Location:** `run_mindreading_multimodal_experiment.py` lines 190-191

#### Corrupted Files
- **Minimum Size:** 50KB (mentioned in documentation)
- **Handling:** Skipped during trial generation or processing

### 10.2 Failed API Calls

#### Rate Limiting (429)
- **Handling:** Exponential backoff retry (up to 10 attempts)
- **Caching:** Failures not cached (allows retry on next run)
- **File Location:** `llm_wrapper.py` lines 405-412

#### Safety Filter Blocks
- **Handling:** Logged as warning, returns None
- **Caching:** Not cached (allows retry)
- **File Location:** `llm_wrapper.py` lines 442-444

#### Network Errors
- **Handling:** Retried with exponential backoff
- **Timeout:** 60 seconds per request

### 10.3 Invalid/Malformed Model Outputs

#### Parsing Failures
- **Handling:** Case-insensitive matching, fallback to first candidate
- **Logging:** Warning if predicted label not in candidates
- **File Location:** `llm_wrapper.py` lines 713-736

#### Missing Predictions
- **Handling:** `is_correct = None` for invalid predictions
- **Exclusion:** Invalid predictions excluded from accuracy calculation
- **File Location:** `run_mindreading_multimodal_experiment.py` line 335

### 10.4 Quality Filtering Criteria

#### Valid Predictions
- **Criteria:** `is_correct is not None`
- **Exclusion:** Predictions with `is_correct = None` excluded from accuracy
- **File Location:** `run_mindreading_multimodal_experiment.py` line 369

#### Per-Emotion Filtering
- **Criteria:** Only emotions with `count > 0` included
- **Handling:** Zero-division protection in accuracy calculation
- **File Location:** `run_mindreading_multimodal_experiment.py` line 389

---

## APPENDIX: File Paths Reference

### Key Implementation Files
- **LLM Wrapper:** `experiments/llm_augmented_emotion_recognition/models/llm_wrapper.py`
- **Main Experiment Script:** `experiments/llm_augmented_emotion_recognition/scripts/run_mindreading_multimodal_experiment.py`
- **Metrics:** `experiments/llm_augmented_emotion_recognition/evaluation/metrics.py`
- **Reproducibility Guide:** `experiments/llm_augmented_emotion_recognition/REPRODUCIBILITY.md`

### Configuration Files
- **LLM Config:** `experiments/llm_augmented_emotion_recognition/configs/llm_config.yaml`
- **EU-Emotion Config:** `experiments/eu_emotion_model_comparison/configs/comparison_config.yaml`

### Data Files
- **MindReading Test:** `data/trial_definitions/mindreading_emotions_test.json`
- **EU-Emotion Test:** `data/trial_definitions/eu_emotion_test_final.json`
- **EU-Emotion Val:** `data/trial_definitions/eu_emotion_val_for_optimization.json`

### Results Files
- **Example Summary:** `results/mindreading_gpt5/summary.json`

---

## NOTES & INCONSISTENCIES

### Flagged Issues

1. **GPT-5 Temperature:** GPT-5 models cannot set temperature (must use default 1.0), while other models use 0.1
   - **Location:** `llm_wrapper.py` lines 560-561
   - **Impact:** Potential non-determinism for GPT-5 models

2. **Failed Trials:** High failure rate (680/1263 = 54%) in MindReading dataset
   - **Location:** `results/mindreading_gpt5/summary.json`
   - **Impact:** Results based on subset of data (583 processed, 572 valid)

3. **Multiple Config Files:** Different config files for different experiments
   - **Impact:** Need to verify which config is used for which experiment

4. **Cache Version Inconsistency:** Different cache versions for video-only vs multimodal
   - **Location:** `run_mindreading_multimodal_experiment.py` line 197
   - **Impact:** Separate caches maintained (intentional design)

5. **Audio Support Variability:** 
   - Google Gemini: Supports audio
   - OpenAI GPT-4o/GPT-4o-mini: No audio support (video-only)
   - OpenAI GPT-audio: Supports audio
   - Anthropic Claude: No audio support (video-only)
   - **Impact:** Multimodal experiments only valid for certain models

---

**END OF METHODS SECTION EXTRACTION**
