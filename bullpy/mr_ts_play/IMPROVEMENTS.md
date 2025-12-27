# Baseline Model Improvements

This document describes the improvements made to the baseline model to enhance performance.

## Implemented Improvements

### 1. Early Stopping ✅
- **Implementation**: Added `--early_stop_patience` argument (default: 5 epochs)
- **Behavior**: Stops training if validation accuracy doesn't improve for N consecutive epochs
- **Benefit**: Prevents overfitting and saves training time

### 2. Fine-Tuning with Differential Learning Rates ✅
- **Implementation**: 
  - `--freeze_backbone` flag to control fine-tuning (default: False, so fine-tuning is enabled)
  - `--lr` for classifier learning rate (default: 1e-3)
  - `--backbone_lr` for backbone learning rate (default: 1e-5)
- **Behavior**: 
  - When fine-tuning, backbone uses 100x lower learning rate than classifier
  - Allows backbone to adapt slowly while classifier learns quickly
- **Benefit**: Better feature extraction for emotion recognition task

### 3. Learning Rate Scheduling ✅
- **Implementation**: `ReduceLROnPlateau` scheduler
- **Behavior**: Reduces learning rate by 50% when validation loss plateaus (patience: 3 epochs)
- **Benefit**: Better convergence and fine-grained optimization

### 4. Data Augmentation ✅
- **Implementation**: 
  - `--use_augmentation` flag
  - `VideoRandomCrop` for spatial augmentation
  - `VideoColorJitter` for color augmentation (brightness, contrast, saturation, hue)
- **Behavior**: 
  - Training uses augmentation, validation/test use center crop
  - Augmentations preserve emotion semantics
- **Benefit**: Better generalization and robustness

### 5. Class Weights for Imbalanced Classes ✅
- **Implementation**: `--use_class_weights` flag
- **Behavior**: 
  - Computes inverse frequency weights from training set
  - Applies weights to CrossEntropyLoss
- **Benefit**: Better handling of classes with few samples

### 6. Model Enhancements ✅
- **Dropout**: Added `--dropout` argument (default: 0.0) for regularization
- **Backbone Selection**: Added `--backbone` argument (resnet18 or resnet50)
- **More Epochs**: Default increased from 5 to 20 epochs

## Usage Examples

### Basic Fine-Tuning (Recommended)
```bash
python experiments/baseline.py \
  --data_root "/path/to/dataset" \
  --splits_dir data/splits \
  --batch_size 8 \
  --num_epochs 20 \
  --lr 1e-3 \
  --backbone_lr 1e-5 \
  --use_augmentation \
  --use_class_weights \
  --early_stop_patience 5
```

### Frozen Backbone (Faster, Lower Performance)
```bash
python experiments/baseline.py \
  --freeze_backbone \
  --num_epochs 20 \
  --lr 1e-3 \
  --use_augmentation
```

### ResNet50 with Dropout
```bash
python experiments/baseline.py \
  --backbone resnet50 \
  --dropout 0.3 \
  --num_epochs 20 \
  --use_augmentation \
  --use_class_weights
```

## Expected Improvements

With these changes, you should see:
- **Higher accuracy**: Fine-tuning and augmentation should improve from ~0.1-0.4% to >1%
- **Better convergence**: Learning rate scheduling and early stopping improve training stability
- **More robust predictions**: Data augmentation improves generalization
- **Better class balance**: Class weights help with rare emotions

## Next Steps

After running with these improvements:
1. **Analyze results**: Check if accuracy improved and training is stable
2. **Hyperparameter tuning**: Try different learning rates, batch sizes, number of frames
3. **Temporal modeling**: Consider using `BaselineVideoClassifier` instead of `SimpleFrameClassifier`
4. **Architecture experiments**: Try different backbones (EfficientNet, Vision Transformer)

## Configuration

All improvements are controlled via command-line arguments, so you can:
- Enable/disable features independently
- Experiment with different combinations
- Track which improvements help most

See `python experiments/baseline.py --help` for all available options.










