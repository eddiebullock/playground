# Prompt for Next Agent: Improving Baseline Model Performance

## Context: Project Status

You are working on a **PhD-level computational psychiatry project** for emotion recognition from video using the Cambridge Mindreading (CAM) / Mindreading dataset. The project is in Python using PyTorch.

### What Has Been Completed ✅

1. **Dataset Setup**: 
   - 4,944 video files (.mov format) with 405 emotion classes
   - Dataset path: `/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions`
   - Actor-independent train/val/test splits created and saved in `data/splits/`
   - Split creation logic fixed to ensure all splits have V (visual) videos

2. **Baseline Pipeline**:
   - Data loading: `src/data/dataset.py` (PyTorch Dataset class)
   - Model: `src/models/baseline.py` (SimpleFrameClassifier using ResNet18)
   - Training script: `experiments/baseline.py`
   - All components working end-to-end

3. **First Baseline Run**:
   - Successfully trained for 2 epochs
   - Results saved in `results/baseline/`
   - Current performance: ~0.1-0.4% accuracy (very low, as expected for initial baseline)

### Current State

**Dataset Details**:
- **Train**: 582 samples, 410 classes (V modality only)
- **Val**: 831 samples  
- **Test**: 1,059 samples
- **Modalities**: V (Visual) and T (Textual) - currently only using V
- **Videos**: 320x240 resolution, ~5 seconds, 25 FPS
- **Challenge**: 410 emotion classes with very few samples per class (1-3 in test set)

**Model Architecture**:
- `SimpleFrameClassifier` in `src/models/baseline.py`
- ResNet18 backbone (frozen, pretrained on ImageNet)
- Linear classifier on top
- Processes 8 frames per video, averages predictions

**Environment**:
- Python 3.13 in virtual environment (`venv/`)
- PyTorch with MPS (Metal Performance Shaders) support (Apple Silicon)
- All dependencies installed in `requirements.txt`

### Key Issues Fixed

1. **Split Creation**: Fixed `src/data/create_splits.py` to ensure modality balance across splits (V-only and T-only actors split separately)
2. **Data Loading**: Verified all splits load correctly with V modality filtering

### Important Files

- `experiments/baseline.py` - Main training script
- `src/models/baseline.py` - Model definitions (SimpleFrameClassifier, BaselineVideoClassifier)
- `src/data/dataset.py` - Dataset loader
- `src/data/create_splits.py` - Split creation (already fixed)
- `configs/baseline_config.yaml` - Configuration template
- `results/baseline/` - Previous experiment results

## Your Task: Improve Baseline Model Performance

The current baseline achieves only ~0.1-0.4% accuracy, which is expected given:
- Only 2 epochs of training
- Frozen backbone (only classifier trained)
- 410 classes with very few samples per class
- Simple frame averaging (no temporal modeling)

### Recommended Next Steps (in priority order):

1. **Increase Training Duration**:
   - Train for more epochs (10-20) with early stopping
   - Monitor validation loss to prevent overfitting
   - Update `experiments/baseline.py` to add early stopping logic

2. **Fine-tune Backbone**:
   - Unfreeze ResNet18 layers (or at least last few layers)
   - Use lower learning rate for backbone (e.g., 1e-5) vs classifier (1e-3)
   - Implement learning rate scheduling

3. **Improve Temporal Modeling**:
   - Replace simple frame averaging with better temporal pooling
   - Consider using `BaselineVideoClassifier` class (already defined but not used)
   - Or add LSTM/Transformer for temporal sequences

4. **Data Augmentation**:
   - Check `src/utils/transforms.py` and enhance augmentation
   - Add spatial augmentations (random crop, flip, color jitter)
   - Be careful not to change emotion semantics

5. **Class Imbalance Handling**:
   - Add class weights to CrossEntropyLoss
   - Consider focal loss for hard examples
   - Monitor per-class performance

6. **Hyperparameter Tuning**:
   - Learning rate: try 1e-4, 1e-3, 1e-2
   - Batch size: try 8, 16, 32 (currently 4)
   - Number of frames: try 16, 32 (currently 8)
   - Optimizer: try AdamW with weight decay

7. **Model Architecture Improvements**:
   - Try ResNet50 instead of ResNet18
   - Try different backbones (EfficientNet, Vision Transformer)
   - Add dropout for regularization

### Implementation Guidelines

1. **Start Simple**: Begin with increasing epochs and fine-tuning, as these are most likely to help
2. **Track Experiments**: 
   - Save results to `results/` with descriptive names
   - Log hyperparameters and results
   - Use the existing logging infrastructure in `src/utils/logging.py`
3. **Reproducibility**: 
   - Always set seeds (use `src/utils/seed.py`)
   - Save experiment configs
4. **Validation**: 
   - Use validation set for model selection
   - Don't touch test set until final evaluation
5. **Code Quality**:
   - Follow existing code style
   - Add docstrings for new functions
   - Keep modular structure

### Expected Outcomes

After improvements, you should see:
- Accuracy > 1% (still low due to 410 classes, but improvement)
- Better validation loss curves
- More stable training

### Testing Your Changes

Run experiments with:
```bash
# Activate venv first
source venv/bin/activate

# Run baseline with your improvements
python experiments/baseline.py \
  --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions" \
  --splits_dir data/splits \
  --batch_size 8 \
  --num_epochs 10 \
  --num_frames 8 \
  --seed 42
```

### Notes

- The dataset uses V (Visual) modality videos - these are standard video files showing actors performing emotions
- T (Textual) modality videos exist but are much smaller and use a codec OpenCV can't read easily - focus on V for now
- Actor independence is critical - splits ensure no actor appears in multiple sets
- The task is very challenging (410 classes) so even small improvements are meaningful

### Questions to Consider

- Should we fine-tune the entire backbone or just last layers?
- What's the best temporal pooling strategy for this task?
- How to handle the extreme class imbalance (1-3 samples per class in test)?
- Should we use both V and T modalities together?

Good luck! Focus on incremental improvements and track what works.









