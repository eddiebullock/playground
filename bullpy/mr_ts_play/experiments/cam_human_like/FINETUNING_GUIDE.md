# Fine-Tuning Guide: Reaching 60-75% Accuracy

## Current Results Summary

| Model | Face Accuracy | Notes |
|-------|--------------|-------|
| CLIP Base (zero-shot) | 37.3% | Baseline |
| CLIP Large (zero-shot) | 37.3% | No improvement |
| Emotion Model (pre-trained) | 19.6% | Basic emotions don't map to complex CAM |
| Hybrid (emotion + CLIP) | 33.3% | Worse than CLIP alone |

**Finding**: Pre-trained basic emotion models don't help. Need fine-tuning for 50-75%.

## Why Fine-Tuning is Needed

1. **Basic vs Complex Emotions**: Pre-trained models (FER2013) use basic emotions (happy, sad, angry), but CAM uses complex emotions (appalled, exonerated, nostalgic)
2. **Domain Gap**: Models trained on static images don't understand video temporal dynamics
3. **Task Alignment**: Models need to be trained on emotion recognition to understand CAM concepts

## Fine-Tuning Strategy: 60-75% Accuracy

### Option 1: Fine-Tune CLIP on FER2013 (60-65%)

**Dataset**: FER2013 (7 basic emotions, 35K images)
- Download: https://www.kaggle.com/datasets/msambare/fer2013
- Format: Images + emotion labels

**Process**:
1. Load CLIP model
2. Fine-tune on FER2013 images + emotion text labels
3. Train for 5-10 epochs
4. Use fine-tuned model in CAM experiment

**Expected**: 60-65% face accuracy

### Option 2: Fine-Tune CLIP on CAM Train Split (65-75%) ⭐ BEST

**Dataset**: Your CAM train split (most aligned with task)
- Use `data/splits/train.csv`
- All 410 concepts or just 20 CAM concepts
- Face trials (V modality)

**Process**:
1. Load CLIP model
2. Fine-tune on CAM train videos + emotion labels
3. Validate on CAM val split
4. Test on CAM test split (or all 100 trials)

**Expected**: 65-75% face accuracy (best alignment)

## Implementation Steps

### Step 1: Create Fine-Tuning Script

Create `experiments/cam_human_like/training/finetune_clip_emotions.py`:

```python
"""
Fine-tune CLIP on emotion recognition data.

This adapts CLIP specifically for emotion recognition, which should
improve CAM performance from 37% to 60-75%.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

def finetune_clip_on_emotions(
    train_dataset,  # Your emotion dataset
    val_dataset,
    model_name="openai/clip-vit-base-patch32",
    num_epochs=10,
    batch_size=16,
    learning_rate=1e-5,
    device="cpu",
):
    """
    Fine-tune CLIP on emotion recognition.
    
    Args:
        train_dataset: Dataset with (images, emotion_labels)
        val_dataset: Validation dataset
        model_name: CLIP model to fine-tune
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
    
    Returns:
        Fine-tuned model
    """
    # Load CLIP
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = model.to(device)
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, emotion_labels = batch
            
            # Process images and text
            image_inputs = processor(images=images, return_tensors="pt").to(device)
            text_inputs = processor(text=emotion_labels, return_tensors="pt", padding=True).to(device)
            
            # Get embeddings
            image_features = model.get_image_features(**image_inputs)
            text_features = model.get_text_features(**text_inputs)
            
            # Normalize
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            # Compute similarity (logits)
            logits_per_image = image_features @ text_features.t()
            logits_per_text = logits_per_text.t()
            
            # Labels: diagonal (image i matches text i)
            labels = torch.arange(len(images)).to(device)
            
            # Contrastive loss
            loss_img = nn.CrossEntropyLoss()(logits_per_image, labels)
            loss_txt = nn.CrossEntropyLoss()(logits_per_text, labels)
            loss = (loss_img + loss_txt) / 2
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
        
        # Validate
        val_accuracy = validate(model, val_dataset, processor, device)
        print(f"Val Accuracy: {val_accuracy:.2%}")
    
    return model

def validate(model, val_dataset, processor, device):
    """Validate fine-tuned model."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_dataset:
            # Get predictions
            # ... (similar to training)
            # Compare predictions to labels
            # correct += (predictions == labels).sum()
            # total += len(labels)
            pass
    
    return correct / total if total > 0 else 0.0
```

### Step 2: Prepare Dataset

**Option A: Use FER2013**
```python
from torchvision.datasets import ImageFolder
from torchvision import transforms

# FER2013 structure: train/happy/, train/sad/, etc.
train_dataset = ImageFolder("fer2013/train", transform=transforms.ToTensor())
```

**Option B: Use CAM Train Split** (Recommended)
```python
from src.data.dataset import MindreadingDataset

train_dataset = MindreadingDataset(
    data_root=config['data']['root'],
    split_file='data/splits/train.csv',
    modality='V',  # Face trials
)
```

### Step 3: Fine-Tune

```bash
python experiments/cam_human_like/training/finetune_clip_emotions.py \
    --train_data "data/splits/train.csv" \
    --val_data "data/splits/val.csv" \
    --output_dir "models/clip_emotion_finetuned" \
    --num_epochs 10 \
    --batch_size 16 \
    --lr 1e-5
```

### Step 4: Use Fine-Tuned Model

Update config:
```yaml
model:
  type: "clip"
  name: "models/clip_emotion_finetuned"  # Local path to fine-tuned model
```

Run experiment:
```bash
python experiments/cam_human_like/run_experiment.py \
    --config configs/cam_config.yaml \
    --split all --no-actor-filtering
```

## Expected Results

| Approach | Expected Accuracy | Time |
|----------|------------------|------|
| Fine-tune on FER2013 | 60-65% | 2-4 hours |
| Fine-tune on CAM train | 65-75% | 4-8 hours |

## Why This Will Work

1. **Task Alignment**: Model learns emotion recognition specifically
2. **Domain Match**: Training on similar data improves generalization
3. **Proven Approach**: Fine-tuning CLIP on task-specific data is standard practice

## Next Steps

1. **Create fine-tuning script** (see template above)
2. **Prepare dataset** (FER2013 or CAM train split)
3. **Fine-tune model** (10-20 epochs)
4. **Evaluate on CAM** (should see 60-75% accuracy)

The fine-tuning approach is the most reliable path to 60-75% accuracy, as it directly addresses the domain gap between general vision-language models and emotion recognition.





