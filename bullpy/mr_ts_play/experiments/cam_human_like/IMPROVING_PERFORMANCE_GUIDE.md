# Guide: Improving CAM Performance Beyond Zero-Shot

This guide explains how to achieve 50-75% accuracy using emotion-specific models and fine-tuning.

## Current Performance

- **Zero-shot CLIP Base**: 37% face accuracy
- **Zero-shot CLIP Large**: 37% face accuracy (similar - model size doesn't help much for zero-shot)
- **Target**: 50-70% with emotion-specific training

## Approach 1: Emotion-Specific Models (50-70%)

### What Are Emotion-Specific Models?

Models that were **pre-trained or fine-tuned on emotion recognition datasets**. These understand emotions better than general vision-language models.

### Recommended Models

1. **FER (Facial Expression Recognition) Models**
   - Trained on FER2013, AffectNet, or similar datasets
   - Examples: ResNet-50/101 trained on emotions, EfficientNet-emotion
   - Expected: 50-60% on CAM (better on basic emotions, struggles with subtle ones)

2. **Emotion-CLIP Variants**
   - CLIP fine-tuned on emotion datasets
   - Examples: EmoCLIP, Emotion-CLIP
   - Expected: 55-65% (combines CLIP's vision-language with emotion knowledge)

3. **Video Emotion Models**
   - Models that understand temporal dynamics
   - Examples: I3D, SlowFast, Video-CLIP
   - Expected: 60-70% (videos show emotion over time)

### How to Use Emotion-Specific Models

#### Option A: Use Pre-trained Emotion Models

```python
# Example: Using a pre-trained FER model
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load emotion-specific model
processor = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")
model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression")

# Adapt to CAM: Map model's emotion classes to CAM concepts
# (May need to map basic emotions to CAM's complex emotions)
```

#### Option B: Create Emotion-Specific Wrapper

Create a new wrapper in `models/emotion_wrapper.py`:

```python
from .base import ModelWrapper, ModelOutput

class EmotionModelWrapper(ModelWrapper):
    """
    Wrapper for emotion-specific models.
    
    These models output emotion probabilities directly,
    which we can map to CAM concepts.
    """
    
    def __init__(self, model_name: str, emotion_to_cam_mapping: Dict, device: str = "cpu"):
        self.emotion_to_cam = emotion_to_cam_mapping
        super().__init__(model_name, device)
    
    def score_labels(self, stimulus_path, candidate_labels, modality="both"):
        # Get emotion predictions from model
        emotions = self.model.predict(stimulus_path)
        
        # Map to CAM concepts and score
        scores = {}
        for label in candidate_labels:
            # Map CAM concept to model's emotion classes
            score = self._map_to_emotion_score(label, emotions)
            scores[label] = score
        
        return ModelOutput(label_scores=scores)
```

### Where to Find Emotion Models

1. **HuggingFace Hub**: Search "emotion recognition", "facial expression"
   - `trpakov/vit-face-expression`
   - `j-hartmann/emotion-english-distilroberta-base`
   - `michellejieli/emotion_text_classifier`

2. **Papers with Code**: 
   - Search "facial expression recognition"
   - Look for models trained on FER2013, AffectNet, RAF-DB

3. **GitHub Repositories**:
   - Search "emotion recognition pytorch"
   - Many open-source implementations

### Implementation Steps

1. **Choose a model** (e.g., FER model or emotion-CLIP)
2. **Create wrapper** in `models/emotion_wrapper.py`
3. **Map emotions** to CAM concepts (may need manual mapping)
4. **Update config** to use emotion model
5. **Run experiment**

## Approach 2: Fine-Tuning on Emotion Data (60-75%)

### What is Fine-Tuning?

**Train CLIP (or another model) on emotion recognition datasets** to adapt it specifically for emotion recognition. This is more work but gives best results.

### Datasets for Fine-Tuning

1. **FER2013** (Facial Expression Recognition)
   - 7 basic emotions: happy, sad, angry, fear, surprise, disgust, neutral
   - 35,887 images
   - Download: https://www.kaggle.com/datasets/msambare/fer2013

2. **AffectNet**
   - 1M+ images, 7 basic + 7 compound emotions
   - More diverse than FER2013
   - Requires registration: http://mohammadmahoor.com/affectnet/

3. **RAF-DB** (Real-world Affective Faces)
   - 29,672 images, 7 basic emotions
   - More realistic than FER2013
   - Download: http://www.whdeng.cn/raf/model1.html

4. **Your Own Dataset**
   - Use the Mindreading/CAM dataset itself!
   - Train on train split, validate on val split
   - This would be most aligned with CAM

### Fine-Tuning Strategy

#### Option A: Fine-Tune CLIP on Emotion Data

```python
# Pseudo-code for fine-tuning CLIP
import torch
from transformers import CLIPModel, CLIPProcessor

# Load CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Prepare emotion dataset
# Images + emotion labels (e.g., "happy", "sad", etc.)

# Fine-tune vision encoder + text encoder
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(10):
    for images, emotion_labels in dataloader:
        # Encode images and emotion text
        image_features = model.get_image_features(**processor(images=images))
        text_features = model.get_text_features(**processor(text=emotion_labels))
        
        # Compute similarity and loss
        logits = image_features @ text_features.T
        loss = cross_entropy(logits, emotion_indices)
        
        loss.backward()
        optimizer.step()

# Save fine-tuned model
model.save_pretrained("clip-emotion-finetuned")
```

#### Option B: Fine-Tune on CAM Dataset Itself

**Best approach for CAM replication:**

1. **Use train split** of your dataset (with all 410 concepts or just 20 CAM)
2. **Fine-tune CLIP** to predict emotions from video frames
3. **Validate on val split**
4. **Test on test split** (or all 100 CAM trials)

```python
# Fine-tune on CAM dataset
from src.data.dataset import MindreadingDataset
from torch.utils.data import DataLoader

# Load CAM train split
train_dataset = MindreadingDataset(
    data_root=config['data']['root'],
    split_file='data/splits/train.csv',
    modality='V',  # Face trials
)

# Fine-tune CLIP
# (Implementation similar to above, but using CAM emotions)
```

### Implementation Steps

1. **Download emotion dataset** (FER2013, AffectNet, or use CAM train split)
2. **Create fine-tuning script** (`experiments/finetune_clip_emotions.py`)
3. **Fine-tune CLIP** on emotion data (10-20 epochs)
4. **Save fine-tuned model**
5. **Update config** to use fine-tuned model
6. **Run CAM experiment**

### Expected Results

| Approach | Expected Accuracy | Effort |
|----------|------------------|--------|
| Pre-trained FER model | 50-60% | Low (just use existing model) |
| Emotion-CLIP variant | 55-65% | Medium (find/download model) |
| Fine-tune CLIP on FER2013 | 60-70% | High (need to train) |
| Fine-tune CLIP on CAM train | 65-75% | High (best alignment) |

## Recommended Workflow

### Quick Win (50-60%): Use Pre-trained Model

1. Find emotion recognition model on HuggingFace
2. Create `EmotionModelWrapper`
3. Map model's emotions to CAM concepts
4. Run experiment

**Time**: 1-2 hours
**Expected**: 50-60% accuracy

### Best Results (65-75%): Fine-Tune on CAM

1. Use CAM train split (or FER2013 as proxy)
2. Fine-tune CLIP on emotion recognition
3. Save fine-tuned model
4. Run CAM experiment with fine-tuned model

**Time**: 4-8 hours (training time)
**Expected**: 65-75% accuracy

## Code Structure

```
experiments/cam_human_like/
├── models/
│   ├── emotion_wrapper.py      # NEW: Wrapper for emotion models
│   └── ...
├── training/                    # NEW: Fine-tuning scripts
│   ├── finetune_clip_emotions.py
│   └── finetune_on_cam.py
└── ...
```

## Next Steps

1. **Try pre-trained emotion model first** (quickest path to 50-60%)
2. **If you need 65-75%**: Fine-tune on emotion data
3. **For best results**: Fine-tune on CAM train split itself

The key insight: **Zero-shot models struggle with complex emotions, but models trained on emotion data can reach 60-75%**, which is much closer to human performance (88%).

