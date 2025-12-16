# Mindreading Mental State Recognition (Study 2)

PhD-level computational psychiatry project evaluating modern ML and LLM-augmented pipelines on complex mental-state recognition using the Cambridge Mindreading (CAM) / Mindreading stimuli.

## Project Overview

This project implements and evaluates machine learning models for recognizing mental states from video stimuli in the Mindreading/CAM dataset. The goal is to develop robust, multimodal approaches that can match or exceed human-level performance on this challenging task.

## Dataset

**Location**: `/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions`

**Summary** (from inspection):
- **4,944 video files** (.mov format)
- **412 unique scenarios**
- **405 unique emotion classes**
- **13 actors** (A, B, C, M, P, R, S, U, V, W, X, Y, Z)
- **2 modalities**: Visual (V) and Textual (T), balanced (2,442 each)
- **Class balance**: 12-24 videos per emotion (ratio: 0.500)
- **Label format**: Embedded in filenames (no separate label files)
- **Structure**: `{scenario_id}{actor}{instance}{V|T}{emotion}.mov`

See `dataset_inspection_report.txt` for detailed analysis.

## Project Structure

```
mr_ts_play/
├── data/
│   ├── raw/              # Symlinks or references to original dataset
│   ├── processed/        # Preprocessed features, extracted frames, etc.
│   └── splits/           # Train/val/test splits (CSV files)
├── src/
│   ├── data/             # Data loading and preprocessing modules
│   ├── models/           # Model architectures
│   ├── training/         # Training loops and utilities
│   ├── evaluation/       # Evaluation metrics and analysis
│   └── utils/            # General utilities
├── experiments/          # Experiment scripts and configurations
├── configs/              # YAML/JSON configuration files
├── notebooks/            # Jupyter notebooks for exploration
├── results/              # Experiment outputs, logs, checkpoints
└── README.md
```

## Experimental Assumptions

### Task Formulation
- **Unit of prediction**: Clip-level (each video clip is one sample)
- **Task type**: Multi-class classification (405 emotion classes)
- **Input**: Video clips (.mov files) with optional audio
- **Output**: Single emotion label per clip

### Data Leakage Prevention
1. **Actor-independent splits**: Ensure no actor appears in both train and test sets
2. **Held-out test set**: Test set untouched until final evaluation
3. **Scenario-level splits**: Consider scenario-level splits to avoid scenario leakage
4. **Temporal independence**: No temporal dependencies between clips (already satisfied by dataset structure)

### Evaluation Target
- **Primary metric**: Classification accuracy
- **Secondary metrics**: Per-class F1, confusion matrix, top-k accuracy
- **Human benchmark**: Compare against human performance on CAM (to be implemented)

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- FFmpeg (for video processing)
- See `requirements.txt` for full dependencies

### Installation
```bash
pip install -r requirements.txt
```

### Quick Start
1. **Inspect dataset**: `python inspect_dataset.py`
2. **Create data splits**: `python src/data/create_splits.py`
3. **Run baseline**: `python experiments/baseline.py`

## Baseline Model

The baseline uses:
- Pretrained video encoder (e.g., I3D, SlowFast, or CLIP)
- Linear classifier on top of extracted features
- Standard train/val/test split with actor independence

See `experiments/baseline.py` for implementation.

## Next Steps

1. ✅ Dataset inspection and validation
2. ✅ Project scaffolding
3. ⏳ Implement data loading pipeline
4. ⏳ Create actor-independent splits
5. ⏳ Implement baseline model
6. ⏳ Add multimodal extensions (audio, text)
7. ⏳ LLM-augmented approaches
8. ⏳ Human benchmark comparison

## Reproducibility

- Random seeds set in all experiments
- Configuration files for all hyperparameters
- Experiment logging to `results/`
- Version control for code and configs

## License

Research project - see project documentation for usage terms.

