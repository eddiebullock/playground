# MindReading/Emotions Dataset Analysis

## Executive Summary

**The dataset at `/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/MindReading/Emotions` is NOT appropriate as a direct replacement for the current EU-Emotion experiment**. 

**However, it IS related to your CAM dataset** - it appears to be a **more complete version** of the same MindReading dataset:
- **CAM dataset**: 4,968 files, 410 emotions
- **MindReading/Emotions**: 5,801 files, 425 emotions  
- **Overlap**: All 4,944 CAM files are in MindReading/Emotions, plus 523 additional files
- **Emotions**: All 410 CAM emotions are in MindReading/Emotions, plus 15 additional emotions

## Key Findings

### Dataset Identity
- **This is a MORE COMPLETE version of the MindReading/CAM dataset**:
  - Contains ALL files from CAM dataset (4,944 overlapping filenames)
  - Plus 523 additional files
  - All 410 CAM emotions plus 15 additional emotions
- **NOT the EU-Emotion Stimulus Set** - which is a separate, curated dataset with 27 emotions
- The current EU-Emotion experiment expects a different structure: `emotions*/HD Version - Face, Body, Social/...`

### Dataset Statistics
- **Total video files**: 5,801 files
- **Successfully parsed**: 5,480 files  
- **Unique emotions**: **425 emotions** (vs. 20 expected in EU-Emotion)
- **Unique scenarios**: 412 scenarios
- **Unique actors**: 13 actors
- **Modalities**: 
  - V (Visual/Face): 2,535 files
  - T (Textual/Voice): 2,945 files

### Structure
```
MindReading/Emotions/
├── 01-24/              # Numbered directories (scenario groups)
│   └── [scenario_id]/   # Subdirectories with scenario IDs
│       └── *.mov        # Video files
├── scenarios/          # 24 scenario videos
├── Audio/             # 2,394 audio files
├── Stories/           # 600 audio files
├── Rewards/           # Mixed media files
└── definitions/       # 100 audio files
```

### Filename Format
Files follow the CAM/MindReading format:
```
[scenario_id][actor][instance][modality][emotion].mov
Example: 0302501Y6Vgrateful.mov
```

## Comparison with Current EU-Emotion Experiment

| Aspect | Current EU-Emotion Experiment | MindReading/Emotions Dataset |
|--------|-------------------------------|------------------------------|
| **Expected Structure** | `emotions*/HD Version - Face, Body, Social/Faces - HD Version/EDITED/EmotionName/*.mp4` | Numbered directories (01-24) with scenario subdirectories |
| **Number of Emotions** | ~20-27 emotions (curated subset) | **425 emotions** (full MindReading set) |
| **Dataset Type** | EU-Emotion Stimulus Set (separate dataset) | MindReading/CAM dataset (source material) |
| **Filename Format** | Similar (emotion in filename) | Same format as CAM |
| **Modalities** | Face + Voice | Face (V) + Voice (T) |
| **Use Case** | Pre-training for CAM | Direct CAM experiments |

## Appropriateness Assessment

### ❌ NOT Appropriate as Direct Replacement Because:

1. **Different Structure**: The EU-Emotion experiment expects a specific directory structure (`emotions*/HD Version...`) that this dataset doesn't have
2. **Too Many Emotions**: 425 emotions vs. ~20-27 expected - would require significant filtering/curation
3. **Same Source Material**: This appears to be the same MindReading/CAM dataset you're already using, just in a different organization
4. **Different Purpose**: EU-Emotion is meant to be an external dataset for pre-training; this is the target dataset itself

### ✅ However, It Could Be Used If:

1. **You want to expand your emotion set**: Use all 425 emotions instead of the curated 27
2. **You adapt the dataset loader**: Modify `eu_emotion_dataset.py` to handle this structure
3. **You filter to specific emotions**: Select a subset of the 425 emotions that match your needs
4. **You use it as an enhanced CAM dataset**: This has all CAM files plus 523 additional files and 15 additional emotions

## Recommendations

### Option 1: Use as Enhanced CAM Dataset (Recommended)
This dataset contains all your CAM files plus additional content. You could:
- Use it as a more complete version of CAM (523 additional files, 15 additional emotions)
- The existing `MindreadingDataset` class should handle it (same filename format)
- No changes needed to current pipeline - it's a superset of CAM

### Option 2: Adapt for EU-Emotion Experiment
If you want to use this for the EU-Emotion experiment:
1. **Filter emotions**: Select ~20-27 emotions that match your experiment goals
2. **Modify dataset loader**: Update `eu_emotion_dataset.py` to handle the numbered directory structure
3. **Create emotion mapping**: Map the 425 emotions to your target emotion set
4. **Reorganize or adapt**: Either reorganize files or update the loader to work with current structure

### Option 3: Find the Actual EU-Emotion Dataset
The current experiment expects:
- Path: `/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions`
- Structure: `emotions*/HD Version - Face, Body, Social/Faces - HD Version/EDITED/EmotionName/`

Check if this path exists and has the expected structure.

## Next Steps

1. **Verify EU-Emotion dataset location**: Check if `/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions` exists with the expected structure
2. **Clarify experiment goals**: 
   - Do you want to use all 425 emotions?
   - Or filter to a specific subset?
   - Or use this as the CAM dataset instead?
3. **If using this dataset**: Adapt the dataset loader to handle the numbered directory structure

## Technical Details

The dataset uses the same filename parsing as CAM:
- Scenario ID: 7 digits (e.g., `0302501`)
- Actor: 1-2 letters (e.g., `Y`)
- Instance: 1-2 digits (e.g., `6`)
- Modality: `V` (visual/face) or `T` (textual/voice)
- Emotion: emotion name (e.g., `grateful`)

The existing `MindreadingDataset` class in `src/data/dataset.py` should be able to parse these files correctly.
