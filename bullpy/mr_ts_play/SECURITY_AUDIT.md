# Git Security Audit - What's Being Tracked

## ✅ Safe to Share (Currently Tracked)

### Code Files
- All Python source code (`src/`, `experiments/`)
- Configuration files (`configs/baseline_config.yaml`)
- Documentation (`.md` files)
- Scripts and utilities

### Data Files (Safe)
- `data/basic_emotion_mapping.json` - Emotion mapping (no personal data)
- `data/splits/train.csv` - Training split (relative paths only)
- `data/splits/val.csv` - Validation split (relative paths only)
- `data/splits/test.csv` - Test split (relative paths only)
- `data/splits/actor_splits.txt` - Actor assignments (just actor codes)

**Note**: CSV files contain:
- Relative paths to videos (e.g., `01/0100104/0100104R6Thumiliating.mov`)
- Emotion labels (public research data)
- Actor codes (anonymous: A, B, C, etc.)
- **No personal information, no absolute paths to your computer**

### Config Files
- `configs/baseline_config.yaml` - Contains:
  - Absolute path to dataset (your local path - OK for research)
  - Hyperparameters (public)
  - **No API keys, passwords, or secrets**

## ❌ Properly Ignored (Not Tracked)

### Large Files
- ✅ Model checkpoints (`.pth`, `.pt`, `.ckpt`) - Ignored
- ✅ Video files (`.mov`, `.mp4`, `.avi`) - Ignored
- ✅ Processed data (`data/processed/`) - Ignored
- ✅ Results (`results/*`) - Ignored (except `.gitkeep`)

### Sensitive Files
- ✅ Environment files (`.env`) - Ignored
- ✅ Secret configs (`*secret*.yaml`, `*secret*.json`) - Ignored
- ✅ Credentials (`*credentials*`, `*password*`, `*token*`) - Ignored
- ✅ Keys (`*.key`, `*.pem`) - Ignored

### Generated Files
- ✅ Python cache (`__pycache__/`) - Ignored
- ✅ Logs (`*.log`) - Ignored
- ✅ Jupyter notebooks (`*.ipynb`) - Ignored

## ⚠️ Potential Concerns (Minor)

### 1. Absolute Paths in Config
**File**: `configs/baseline_config.yaml`
**Content**: Contains your local absolute path:
```yaml
root: "/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/..."
```

**Risk**: Low - This is your username and file structure, but not sensitive data
**Recommendation**: Consider using relative paths or environment variables if sharing publicly

### 2. Dataset Location Revealed
**File**: CSV files and configs
**Content**: Shows dataset is in OneDrive/University of Cambridge folder

**Risk**: Very Low - This is public research data location
**Recommendation**: Fine as-is for research repository

## ✅ Security Checklist

- [x] No API keys or secrets in tracked files
- [x] No passwords or credentials
- [x] No personal information (names, emails, etc.)
- [x] No large binary files (videos, models)
- [x] No sensitive data files
- [x] Model checkpoints properly ignored
- [x] Results properly ignored
- [x] Environment files ignored

## Summary

**Everything looks safe!** 

The repository contains:
- ✅ Public research code
- ✅ Public emotion labels and mappings
- ✅ Relative paths to videos (not the videos themselves)
- ✅ Hyperparameters and configs (no secrets)
- ✅ Documentation

**Nothing sensitive is being tracked.** The only "personal" information is:
- Your username in absolute paths (low risk)
- Dataset location (public research data)

Both are acceptable for a research repository.

