# Fine-Tuning Guide for Vision Models

This guide explains how to fine-tune vision models (I3D, TimeSformer, ResNet, ViT, etc.) on the EU-Emotion dataset.

## Current Situation

Most vision models in the comparison are **pretrained on ImageNet/Kinetics** (general vision tasks), not emotions. They currently return uniform scores as placeholders.

**To get meaningful results, you need to fine-tune them on emotion data first.**

## Option 1: Fine-Tune Using Existing CLIP Script (Easiest)

You already have a fine-tuning script! You can adapt it for other models.

### Current CLIP Fine-Tuning

Your existing script: `experiments/cam_human_like/training/finetune_clip_emotions.py`

**Current CLIP model**: `openai/clip-vit-base-patch32` (base/smaller version)

### Should You Use a Larger CLIP Model?

**Short answer: Probably not worth it for fine-tuning.**

From your previous experiments:
- CLIP Base (zero-shot): 37.3% accuracy
- CLIP Large (zero-shot): 37.3% accuracy (no improvement!)

**For fine-tuning, base model is usually sufficient:**
- Fine-tuning adapts the model to your task
- Larger models are more expensive and slower
- Base model + fine-tuning typically matches or exceeds large model zero-shot

**However, if you want to try larger CLIP models:**

Available CLIP models:
- `openai/clip-vit-base-patch32` (current) - 86M params
- `openai/clip-vit-base-patch16` - 86M params (higher resolution)
- `openai/clip-vit-large-patch14` - 307M params (larger)
- `openai/clip-vit-large-patch14-336` - 307M params (larger + higher res)

**Recommendation**: Stick with `clip-vit-base-patch32` unless you have specific reasons to use larger models.

## Option 2: Fine-Tune Other Vision Models

### For ResNet, ViT, EfficientNet (Image Models)

These models need to be fine-tuned for **multi-class emotion classification** (27 classes for EU-Emotion).

**Approach:**
1. Extract frames from videos
2. Train a classifier on top of pretrained features
3. Fine-tune the entire model or just the classifier head

**Example script structure:**

```python
# Fine-tune ResNet50 for emotion recognition
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader

# Load pretrained ResNet50
model = models.resnet50(weights="IMAGENET1K_V2")

# Replace classifier for 27 emotion classes
num_emotions = 27
model.fc = nn.Linear(model.fc.in_features, num_emotions)

# Fine-tune on EU-Emotion dataset
# (You'd need to create a dataset loader for EU-Emotion)
```

### For I3D, TimeSformer, X3D (Video Models)

These need **video-level fine-tuning** (process full video sequences, not just frames).

**Approach:**
1. Use pretrained video model
2. Replace final classification layer for 27 emotions
3. Fine-tune on video sequences

**Example for TimeSformer:**

```python
from transformers import TimesformerForVideoClassification

# Load pretrained TimeSformer
model = TimesformerForVideoClassification.from_pretrained(
    "facebook/timesformer-base-finetuned-k400"
)

# Modify for 27 emotion classes
model.classifier = nn.Linear(model.config.hidden_size, 27)

# Fine-tune on EU-Emotion videos
```

## Practical Steps

### Step 1: Prepare Training Data

You need to create train/val splits from EU-Emotion:

```python
# Create train/val splits (80/20)
# Use existing trial creation scripts as reference
# experiments/cam_human_like/training/create_eu_emotion_trials.py
```

### Step 2: Create Fine-Tuning Script

For each model type, create a fine-tuning script similar to your CLIP script:

**Template structure:**
1. Load pretrained model
2. Modify final layer for 27 emotions
3. Create data loader for EU-Emotion
4. Train with appropriate loss (CrossEntropyLoss)
5. Save best model

### Step 3: Run Fine-Tuning

```bash
# Example for ResNet50
python experiments/eu_emotion_model_comparison/training/finetune_resnet.py \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root /path/to/EU_emotions \
    --output_dir models/resnet50_emotion_finetuned \
    --num_epochs 20 \
    --batch_size 16 \
    --learning_rate 1e-4
```

### Step 4: Update Model Wrappers

After fine-tuning, update the model wrappers to load your fine-tuned models:

```python
# In cnn_vit_wrappers.py
class ResNetModel(BaseEmotionModel):
    def __init__(self, variant="resnet50", fine_tuned_path=None, ...):
        if fine_tuned_path:
            # Load your fine-tuned model
            self.model = torch.load(fine_tuned_path)
        else:
            # Use pretrained (returns uniform scores)
            self.model = models.resnet50(weights="IMAGENET1K_V2")
```

## Quick Start: Fine-Tune ResNet50 (Simplest)

Here's a minimal example to get you started:

### 1. Create Fine-Tuning Script

Create `experiments/eu_emotion_model_comparison/training/finetune_resnet.py`:

```python
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from PIL import Image
import cv2

# Dataset class for EU-Emotion
class EUEmotionDataset(Dataset):
    def __init__(self, trial_file, data_root, num_frames=8):
        with open(trial_file) as f:
            self.trials = json.load(f)['trials']
        self.data_root = Path(data_root)
        self.num_frames = num_frames
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Get all unique emotions
        self.emotions = sorted(set(t['correct_label'] for t in self.trials))
        self.emotion_to_idx = {e: i for i, e in enumerate(self.emotions)}
    
    def __len__(self):
        return len(self.trials)
    
    def __getitem__(self, idx):
        trial = self.trials[idx]
        video_path = self.data_root / trial['stimulus_path']
        
        # Extract frames
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        cap.release()
        
        # Use first frame (or average multiple frames)
        frame = frames[0] if frames else Image.new('RGB', (224, 224))
        frame_tensor = self.transform(frame)
        
        # Get label
        emotion = trial['correct_label']
        label = self.emotion_to_idx[emotion]
        
        return frame_tensor, label

# Fine-tuning function
def finetune_resnet(
    train_trials,
    val_trials,
    data_root,
    output_dir,
    num_epochs=20,
    batch_size=16,
    learning_rate=1e-4,
    device="cpu",
):
    # Load pretrained ResNet50
    model = models.resnet50(weights="IMAGENET1K_V2")
    
    # Get number of classes from training data
    with open(train_trials) as f:
        train_data = json.load(f)['trials']
    num_classes = len(set(t['correct_label'] for t in train_data))
    
    # Replace classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    # Create datasets
    train_dataset = EUEmotionDataset(train_trials, data_root)
    val_dataset = EUEmotionDataset(val_trials, data_root)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Path(output_dir) / "best_model.pth")
    
    print(f"Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_trials", required=True)
    parser.add_argument("--val_trials", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    finetune_resnet(
        args.train_trials,
        args.val_trials,
        args.data_root,
        args.output_dir,
        args.num_epochs,
        args.batch_size,
        args.learning_rate,
        device,
    )
```

### 2. Create Train/Val Splits

You'll need to split your EU-Emotion trials into train/val. You can adapt your existing trial creation scripts.

### 3. Run Fine-Tuning

```bash
python experiments/eu_emotion_model_comparison/training/finetune_resnet.py \
    --train_trials data/trial_definitions/eu_emotion_train.json \
    --val_trials data/trial_definitions/eu_emotion_val.json \
    --data_root /Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions \
    --output_dir models/resnet50_emotion_finetuned \
    --num_epochs 20 \
    --batch_size 16 \
    --device auto
```

### 4. Update Model Wrapper

After fine-tuning, modify `cnn_vit_wrappers.py` to load your fine-tuned model:

```python
class ResNetModel(BaseEmotionModel):
    def __init__(self, variant="resnet50", fine_tuned_path=None, ...):
        if fine_tuned_path and Path(fine_tuned_path).exists():
            # Load fine-tuned model
            model = models.resnet50(weights=None)  # Don't load pretrained
            model.fc = nn.Linear(model.fc.in_features, 27)  # 27 emotions
            model.load_state_dict(torch.load(fine_tuned_path))
            self.model = model.to(self.device)
        else:
            # Use pretrained (placeholder)
            self.model = models.resnet50(weights="IMAGENET1K_V2")
            self.model.fc = nn.Identity()
            self.model = self.model.to(self.device)
```

## Summary

1. **CLIP Model Size**: Your current `clip-vit-base-patch32` is fine. Larger models likely won't help much after fine-tuning.

2. **Fine-Tuning Vision Models**: 
   - Create fine-tuning scripts for each model type
   - Prepare train/val splits from EU-Emotion
   - Fine-tune for 27 emotion classes
   - Update model wrappers to load fine-tuned models

3. **Start Simple**: Fine-tune ResNet50 first (easiest), then expand to other models.

4. **Time Investment**: Each model will take 2-8 hours to fine-tune depending on dataset size and hardware.

## Alternative: Use Pre-trained Emotion Models

Instead of fine-tuning, you could use models already trained on emotions:
- FER2013 models (already integrated)
- AffectNet models
- EmoVLM-KD

These work out of the box but may need label mapping for EU-Emotion's 27 emotions.
