# Dataset Inspection Summary

## Dataset Location
**Path**: `/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/mindreading_transporter_files/Mindreading emotions library/Emotions`

**Note**: The path in the original request had a typo (`transporters` vs `transporter`). The correct path uses `transporter`.

## Dataset Statistics

### Overview
- **Total video files**: 4,944
- **Unique scenarios**: 412
- **Unique emotion classes**: 405
- **Unique actors**: 13 (A, B, C, M, P, R, S, U, V, W, X, Y, Z)
- **Modalities**: Visual (V) and Textual (T)
  - Visual: 2,442 videos
  - Textual: 2,442 videos
  - Perfectly balanced

### File Structure
```
Emotions/
├── 01/          # Emotion category 01
│   ├── 0100104/ # Scenario 0100104
│   │   ├── 0100104M1Vhumiliating.mov
│   │   ├── 0100104P5Thumiliating.mov
│   │   └── ...
│   └── ...
├── 02/
└── ...
```

### Filename Format
`{scenario_id}{actor}{instance}{V|T}{emotion}.mov`

**Example**: `0100104M1Vhumiliating.mov`
- `0100104`: Scenario ID
- `M`: Actor code
- `1`: Instance number
- `V`: Modality (V=Visual, T=Textual)
- `humiliating`: Emotion label

### Label Format
- **Embedded in filenames**: No separate label files
- **Single label per clip**: Each video has exactly one emotion
- **1:1 scenario-emotion mapping**: Each scenario maps to exactly one emotion

### Class Balance
- **Min videos per emotion**: 12
- **Max videos per emotion**: 24
- **Average videos per emotion**: 12.1
- **Balance ratio (min/max)**: 0.500

**Assessment**: Reasonably balanced, though some emotions have twice as many samples as others.

### Actor Distribution
| Actor | Videos |
|-------|--------|
| M     | 819    |
| P     | 830    |
| Y     | 814    |
| U     | 772    |
| S     | 575    |
| Z     | 345    |
| R     | 270    |
| C     | 235    |
| A     | 120    |
| V     | 86     |
| B     | 9      |
| W     | 8      |
| X     | 1      |

**Note**: Very uneven distribution. Some actors (B, W, X) have very few samples.

## Dataset Usability

### ✅ What Exists
- Video files in .mov format
- Clear emotion labels embedded in filenames
- Consistent file structure
- Balanced visual/textual modalities
- Reasonable class balance

### ⚠️ Potential Issues
1. **Actor imbalance**: Some actors have very few samples (B: 9, W: 8, X: 1)
2. **Video format**: .mov files may require QuickTime or FFmpeg
3. **No separate label files**: Labels must be extracted from filenames
4. **Hyphenated emotions**: Some emotions contain hyphens (e.g., "easy-going")

### ✅ Dataset is Usable
The dataset is **usable as-is** with the following considerations:

1. **Label extraction**: Implemented in `src/data/dataset.py` and `src/data/create_splits.py`
2. **Actor-independent splits**: Critical for avoiding data leakage (implemented)
3. **Video loading**: Uses OpenCV (cv2) which handles .mov files
4. **Class balance**: Acceptable for multi-class classification

## Preprocessing Requirements

### Minimal Preprocessing Needed
1. **Frame extraction**: Sample frames from videos (implemented)
2. **Resize**: Resize to standard size (e.g., 224x224) for pretrained models
3. **Normalization**: ImageNet normalization for pretrained models

### No Preprocessing Required For
- Label extraction (done programmatically)
- Format conversion (OpenCV handles .mov)
- Audio extraction (not needed for baseline)

## Recommendations

1. **Use actor-independent splits**: Essential to avoid learning actor-specific features
2. **Handle actor imbalance**: Consider stratified splitting or weighting
3. **Start with visual modality**: Focus on "V" modality first, then add "T" or combine
4. **Frame sampling**: Use 8-16 frames per video for baseline (balance between information and efficiency)
5. **Class weighting**: Consider class weights in loss function if needed

## Next Steps

1. ✅ Dataset inspection complete
2. ✅ Project structure created
3. ✅ Data loading pipeline implemented
4. ✅ Baseline model ready
5. ⏳ Run baseline experiment to verify end-to-end pipeline
6. ⏳ Analyze results and iterate

