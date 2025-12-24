# Quick Start: Running Improved Baseline

## Recommended Command

Run the improved baseline with all enhancements enabled:

```bash
# Activate virtual environment (if using one)
source venv/bin/activate

# Run with recommended settings
python experiments/baseline.py \
  --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions" \
  --splits_dir data/splits \
  --batch_size 8 \
  --num_epochs 20 \
  --lr 1e-3 \
  --backbone_lr 1e-5 \
  --num_frames 8 \
  --use_augmentation \
  --use_class_weights \
  --early_stop_patience 5 \
  --seed 42
```

## What's Different from Before?

### Enabled by Default (No Flags Needed)
- ✅ **Fine-tuning**: Backbone now trains with lower learning rate (1e-5 vs 1e-3)
- ✅ **More epochs**: 20 instead of 5
- ✅ **Early stopping**: Stops if no improvement for 5 epochs
- ✅ **Learning rate scheduling**: Automatically reduces LR on plateau
- ✅ **MPS support**: Auto-detects Apple Silicon GPU

### Optional Flags (Recommended)
- `--use_augmentation`: Data augmentation for better generalization
- `--use_class_weights`: Handles class imbalance
- `--dropout 0.3`: Regularization (try if overfitting)

### Optional Flags (Advanced)
- `--freeze_backbone`: Disable fine-tuning (faster, lower performance)
- `--backbone resnet50`: Use ResNet50 instead of ResNet18
- `--num_frames 16`: More frames per video (slower, potentially better)

## Expected Runtime

- **With MPS (Apple Silicon)**: ~2-5 minutes per epoch
- **With CPU**: ~10-20 minutes per epoch
- **Total**: ~40-100 minutes for 20 epochs (or until early stopping)

## Monitoring Progress

Watch for:
- **Training accuracy**: Should increase over epochs
- **Validation accuracy**: Should track training (if not, may be overfitting)
- **Early stopping**: Will stop automatically if no improvement
- **LR reduction**: Scheduler will print when reducing learning rate

## Results Location

All results saved to `results/baseline/`:
- `train_history.csv`: Training metrics per epoch
- `best_model.pth`: Best model checkpoint
- `test_results.txt`: Final test set performance
- `confusion_matrix.png`: Confusion matrix (if ≤50 classes)

## Troubleshooting

### Out of Memory
- Reduce `--batch_size` (try 4 or 2)
- Reduce `--num_frames` (try 4)

### Slow Training
- Use `--freeze_backbone` (faster but lower performance)
- Reduce `--num_epochs`
- Use smaller backbone: `--backbone resnet18` (default)

### Low Accuracy
- Try `--use_augmentation` and `--use_class_weights`
- Increase `--num_frames` to 16
- Try `--backbone resnet50`
- Increase `--num_epochs` (with early stopping, it will stop when needed)

## Next Steps After Running

1. Check `results/baseline/train_history.csv` for training curves
2. Review `results/baseline/test_results.txt` for final metrics
3. If accuracy is still low (<1%), consider:
   - More frames per video
   - Different backbone (ResNet50, EfficientNet)
   - Temporal modeling (use `BaselineVideoClassifier`)
   - Multimodal approaches (combine V and T)









