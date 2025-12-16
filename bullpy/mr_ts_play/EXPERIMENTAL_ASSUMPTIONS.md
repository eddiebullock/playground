# Experimental Assumptions and Design Decisions

This document explicitly states the experimental assumptions, task formulation, and data leakage prevention strategies for Study 2.

## Task Formulation

### Unit of Prediction
- **Clip-level classification**: Each video clip is treated as a single sample
- **No temporal modeling across clips**: Clips are independent (no sequence modeling)
- **Single label per clip**: Each clip has exactly one emotion label (multi-class, not multi-label)

### Task Type
- **Multi-class classification**: 405 emotion classes
- **Not multi-label**: Each sample belongs to exactly one class
- **Not regression**: Discrete emotion categories, not continuous dimensions

### Input Modalities
- **Primary**: Video (visual frames)
- **Optional**: Audio (if available in .mov files)
- **Future**: Text descriptions or LLM-generated features

### Output Format
- **Single emotion label**: One of 405 emotion classes
- **Softmax probabilities**: Model outputs probability distribution over classes
- **Prediction**: Argmax of probability distribution

## Data Leakage Prevention

### 1. Actor-Independent Splits
**Critical**: No actor should appear in both training and test sets.

**Rationale**: 
- Different actors may have different acting styles, facial features, or expressions
- If an actor appears in both train and test, the model may learn actor-specific features rather than emotion-specific features
- This would inflate performance estimates

**Implementation**:
- Group samples by actor
- Split actors into train/val/test sets
- Ensure complete actor separation

**Actor distribution** (from inspection):
- 13 unique actors: A, B, C, M, P, R, S, U, V, W, X, Y, Z
- Uneven distribution (e.g., M: 819, P: 830, Y: 814 vs. B: 9, W: 8, X: 1)
- Need stratified split to maintain class balance while respecting actor independence

### 2. Held-Out Test Set
**Critical**: Test set is completely untouched until final evaluation.

**Rules**:
- No hyperparameter tuning on test set
- No model selection on test set
- No early stopping based on test performance
- Test set used only for final performance reporting

**Split strategy**:
- Train: 70% (for training and hyperparameter tuning)
- Val: 15% (for model selection and early stopping)
- Test: 15% (held out until final evaluation)

### 3. Scenario-Level Considerations
**Note**: Each scenario maps to exactly one emotion (1:1 mapping).

**Potential issue**: If we split by scenario, we might have perfect separation, but this could be too strict.

**Decision**: 
- Primary split: Actor-based (ensures actor independence)
- Secondary consideration: Monitor scenario distribution to ensure emotion balance
- If scenario-level split is needed, document it explicitly

### 4. Temporal Independence
**Status**: Already satisfied by dataset structure.

- Clips are independent scenarios
- No temporal sequences or dependencies
- No need for special handling

## Evaluation Metrics

### Primary Metrics
1. **Accuracy**: Overall classification accuracy
2. **Per-class F1**: F1 score for each emotion class
3. **Macro F1**: Average F1 across all classes
4. **Weighted F1**: F1 weighted by class frequency

### Secondary Metrics
1. **Confusion matrix**: Visualize class confusion patterns
2. **Top-k accuracy**: Accuracy when considering top-k predictions
3. **Per-actor accuracy**: Performance breakdown by actor (for analysis)
4. **Per-emotion accuracy**: Performance breakdown by emotion

### Human Benchmark
- **Target**: Compare model performance to human performance on CAM
- **Status**: To be implemented in future work
- **Method**: Use published human performance data or conduct new human evaluation

## Baseline Model Assumptions

### Architecture
- **Encoder**: Pretrained video encoder (e.g., I3D, SlowFast, CLIP-ViT)
- **Classifier**: Linear layer on top of encoder features
- **No fine-tuning**: Encoder frozen initially (can be fine-tuned later)

### Training
- **Optimizer**: Adam or AdamW
- **Learning rate**: 1e-4 to 1e-3 (tuned on validation set)
- **Batch size**: 16-32 (depending on GPU memory)
- **Epochs**: Early stopping based on validation loss
- **Loss**: Cross-entropy

### Data Augmentation
- **Spatial**: Random crop, horizontal flip (if appropriate)
- **Temporal**: Uniform sampling of frames
- **No augmentation that changes emotion**: Be careful with transformations

## Reproducibility

### Random Seeds
- **Python random**: `random.seed(42)`
- **NumPy**: `np.random.seed(42)`
- **PyTorch**: `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`
- **DataLoader**: `worker_init_fn` to set seeds per worker

### Logging
- **Experiment configs**: Save all hyperparameters to YAML/JSON
- **Results**: Log metrics, losses, confusion matrices
- **Checkpoints**: Save model checkpoints at best validation performance
- **Version control**: Git for code, separate versioning for data splits

### Environment
- **Python version**: 3.8+
- **PyTorch version**: Document exact version
- **CUDA version**: Document if using GPU
- **Dependencies**: `requirements.txt` with versions

## Future Extensions

### Multimodal Approaches
- **Video + Audio**: Combine visual and audio features
- **Video + Text**: Use LLM-generated descriptions or embeddings
- **Fusion strategies**: Early fusion, late fusion, attention-based fusion

### LLM-Augmented Models
- **Prompt engineering**: Generate emotion descriptions from video
- **Few-shot learning**: Use LLM for few-shot classification
- **Ensemble**: Combine vision model with LLM predictions

### Advanced Architectures
- **Transformer-based**: Video transformers (TimeSformer, VideoMAE)
- **Attention mechanisms**: Self-attention, cross-modal attention
- **Contrastive learning**: Self-supervised pretraining on video

## Open Questions

1. **Modality selection**: Should we use both V and T modalities, or focus on one?
2. **Frame sampling**: How many frames per clip? Uniform or keyframe extraction?
3. **Temporal modeling**: Should we model temporal dynamics within clips?
4. **Class imbalance**: Some emotions have 12 videos, others 24. Should we balance?
5. **Actor balance**: Some actors have very few samples. How to handle?

## Decisions Log

- **2024-XX-XX**: Initial assumptions documented
- **2024-XX-XX**: Decided on actor-independent splits as primary strategy
- **2024-XX-XX**: Baseline uses frozen pretrained encoder + linear classifier

