# Prototypical Networks Baseline

## What is Prototypical Networks?

Prototypical Networks is a few-shot learning method that:
1. **Learns embeddings** where similar emotions cluster together
2. **Computes class prototypes** (mean embedding per class)
3. **Classifies by distance** to prototypes (closest prototype wins)

**Why it works for your problem:**
- Designed for few-shot learning (7 samples per class)
- Learns similarity structure, not just classification boundaries
- More robust with limited data than standard classification

## Expected Performance

With 7 samples per class:
- **Top-1 accuracy**: 5-15% (vs random: 0.24%)
- **Top-5 accuracy**: 20-40% (vs random: 1.2%)
- **Top-10 accuracy**: 35-55% (vs random: 2.4%)

This is **20-60x better than random** and shows the model is learning!

## Usage

### Basic Run
```bash
python experiments/prototypical_baseline.py \
  --data_root "/path/to/dataset" \
  --splits_dir data/splits \
  --batch_size 16 \
  --num_epochs 30 \
  --lr 1e-4 \
  --use_augmentation \
  --seed 42
```

### Recommended Settings
```bash
python experiments/prototypical_baseline.py \
  --data_root "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions" \
  --splits_dir data/splits \
  --batch_size 16 \
  --num_epochs 30 \
  --lr 1e-4 \
  --embedding_dim 512 \
  --backbone resnet18 \
  --dropout 0.3 \
  --num_frames 8 \
  --use_augmentation \
  --seed 42
```

### With ResNet50 (Better, Slower)
```bash
python experiments/prototypical_baseline.py \
  --backbone resnet50 \
  --embedding_dim 2048 \
  --batch_size 8 \
  --num_epochs 30
```

## Key Differences from Standard Baseline

1. **Embedding-based**: Learns continuous embeddings, not just classification
2. **Distance-based classification**: Uses Euclidean distance to prototypes
3. **Top-k evaluation**: Reports top-1, top-5, top-10 accuracy (more meaningful)
4. **Prototype computation**: Uses all training data to compute prototypes

## Output

Results saved to `results/prototypical/`:
- `best_model.pth`: Best model checkpoint
- `train_history.csv`: Training metrics per epoch
- `test_results.txt`: Final test set performance with all metrics

## Comparison to Standard Baseline

| Method | Top-1 Acc | Top-5 Acc | Top-10 Acc |
|--------|-----------|-----------|------------|
| Standard (frozen) | 0.1-0.4% | - | - |
| Standard (fine-tuned) | 1-3% | - | - |
| **Prototypical** | **5-15%** | **20-40%** | **35-55%** |

Prototypical Networks should significantly outperform standard classification!

## Next Steps

After running prototypical networks:
1. Compare to standard baseline
2. Try metric learning (contrastive/triplet loss)
3. Experiment with embedding dimensions
4. Add hierarchical evaluation
5. Compare to human performance

