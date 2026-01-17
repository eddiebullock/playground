# EU-Emotion vs MindReading/Emotions Dataset Comparison

## EU-Emotion Dataset Structure

### Location
`/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/EU_emotions`

### Structure
```
EU_emotions/
├── emotions 1, emotions 2, ..., emotions 38, ... (multiple numbered directories)
│   └── HD Version - Face, Body, Social/
│       └── Faces - HD Version/
│           ├── EDITED/
│           │   ├── EmotionName1/  (e.g., "Happy", "Angry Low Intensity")
│           │   │   └── *.mp4
│           │   ├── EmotionName2/
│           │   └── ...
│           └── Original/
│               ├── EmotionName1/
│               │   └── *.mov
│               └── ...
├── EU Emotion - UK Voices/  (audio files)
└── Still Images - Faces/    (image files)
```

### Key Characteristics
- **Total video files**: ~959 files (mp4 + mov)
- **Emotions**: ~27 curated emotions (e.g., "Happy", "Happy Low Intensity", "Angry", "Angry Low Intensity", "Frustrated", "Joking", "Hurt", "Ashamed", "Interested", "Disgusted Low Intensity", etc.)
- **Structure**: Organized by emotion name in folders
- **Modalities**: Face videos (HD), Voice audio, Body gestures, Social scenes
- **Format**: 
  - EDITED folder: `.mp4` files
  - Original folder: `.mov` files
- **Purpose**: Curated stimulus set for emotion recognition research
- **Source**: Autism Research Centre (EU-Emotion Stimulus Set)

### Filename Format
- Files are organized by emotion name in folders
- No standardized filename format (various naming conventions)
- Example: `HF23v2_cut.mp4` (in "Frustrated" folder)

---

## MindReading/Emotions Dataset Structure

### Location
`/Users/eb2007/Library/CloudStorage/OneDrive-UniversityofCambridge/Documents/PhD/data/MindReading/Emotions`

### Structure
```
MindReading/Emotions/
├── 01, 02, 03, ..., 24/  (numbered directories)
│   └── [scenario_id]/     (7-digit scenario IDs)
│       └── *.mov          (video files)
├── scenarios/            (24 scenario videos)
├── Audio/                (2,394 audio files)
├── Stories/              (600 audio files)
├── Rewards/             (mixed media)
└── definitions/          (100 audio files)
```

### Key Characteristics
- **Total video files**: 5,801 files
- **Emotions**: 425 unique emotions (comprehensive set)
- **Structure**: Organized by scenario groups (numbered directories) and scenario IDs
- **Modalities**: 
  - V (Visual/Face): 2,535 files
  - T (Textual/Voice): 2,945 files
- **Format**: All `.mov` files
- **Purpose**: Complete MindReading/CAM dataset for emotion recognition
- **Source**: Autism Research Centre (MindReading/CAM dataset)

### Filename Format
Standardized format: `[scenario_id][actor][instance][modality][emotion].mov`
- Example: `0302501Y6Vgrateful.mov`
  - `0302501`: Scenario ID (7 digits)
  - `Y`: Actor code
  - `6`: Instance number
  - `V`: Modality (V=Visual/Face, T=Textual/Voice)
  - `grateful`: Emotion name

---

## Side-by-Side Comparison

| Aspect | EU-Emotion | MindReading/Emotions |
|--------|------------|---------------------|
| **Total Files** | ~959 videos | 5,801 videos |
| **Emotions** | ~27 curated | 425 comprehensive |
| **Structure** | Emotion-based folders | Scenario-based directories |
| **Organization** | `emotions*/HD Version.../Faces - HD Version/EDITED/EmotionName/` | `01-24/[scenario_id]/` |
| **Filename Format** | Variable (e.g., `HF23v2_cut.mp4`) | Standardized: `[scenario][actor][instance][modality][emotion].mov` |
| **File Formats** | `.mp4` (EDITED), `.mov` (Original) | `.mov` only |
| **Modalities** | Face (HD), Voice, Body, Social | Face (V), Voice (T) |
| **Curated vs Raw** | Curated subset | Complete dataset |
| **Purpose** | Pre-training stimulus set | Full research dataset |
| **Compatibility** | Works with `EUEmotionDataset` | Works with `MindreadingDataset` |

---

## Which Dataset is Better?

### For EU-Emotion Experiment: **EU-Emotion Dataset** ✅

**Reasons:**
1. **Designed for the experiment**: The EU-Emotion experiment was specifically designed for this dataset
2. **Curated emotions**: 27 carefully selected emotions match the experiment goals
3. **Ready to use**: The `EUEmotionDataset` loader already supports this structure
4. **Quality control**: Curated subset likely has better quality control
5. **Low intensity variants**: Includes important "Low Intensity" emotion variants
6. **Multiple modalities**: Has face, voice, body, and social scenes organized separately

**Current Status**: Your experiment already uses this dataset successfully (374 training trials with 27 emotions)

---

### For CAM/MindReading Experiments: **MindReading/Emotions Dataset** ✅

**Reasons:**
1. **Complete dataset**: All 425 emotions from the full MindReading dataset
2. **More data**: 5,801 files vs 4,968 in CAM (523 additional files)
3. **Standardized format**: Consistent filename format makes parsing easier
4. **Comprehensive**: Includes all scenarios, actors, and emotion variations
5. **Better for research**: More complete dataset for comprehensive emotion recognition

**Current Status**: This is a superset of your CAM dataset (contains all CAM files + 523 more)

---

## Recommendation Summary

### Use EU-Emotion Dataset When:
- ✅ Running the **EU-Emotion experiment** (current setup)
- ✅ Need **curated, high-quality** emotion stimuli
- ✅ Want **specific emotion set** (27 emotions with low-intensity variants)
- ✅ Need **multiple modalities** organized separately

### Use MindReading/Emotions Dataset When:
- ✅ Running **CAM/MindReading experiments**
- ✅ Need **comprehensive emotion coverage** (425 emotions)
- ✅ Want **maximum data** for training
- ✅ Need **standardized filename format** for easier processing

### For Your Current Experiment:
**Stick with EU-Emotion Dataset** - Your experiment is already set up for it and working well. The 27 curated emotions are specifically chosen for the experiment goals, and the dataset structure is already integrated into your pipeline.

**However**, if you want to expand your experiment:
- You could use MindReading/Emotions as an **additional training source**
- Filter MindReading/Emotions to match the 27 EU-Emotion emotions
- Use it for **data augmentation** or **larger-scale training**

---

## Technical Compatibility

### EU-Emotion Dataset
- ✅ **Compatible**: `EUEmotionDataset` class handles this structure
- ✅ **Working**: Current experiment uses this successfully
- ✅ **Trial definitions**: Already created for 27 emotions

### MindReading/Emotions Dataset  
- ✅ **Compatible**: `MindreadingDataset` class handles this structure
- ⚠️ **Requires adaptation**: Would need to filter to 27 emotions or adapt experiment
- ⚠️ **Different structure**: Would need to modify `EUEmotionDataset` or create adapter

---

## Conclusion

**For your current EU-Emotion experiment: EU-Emotion dataset is better** - it's designed for it, already integrated, and working well.

**For comprehensive emotion research: MindReading/Emotions dataset is better** - it's more complete, has more data, and covers all emotions.

**Best approach**: Use both! EU-Emotion for the specific experiment, MindReading/Emotions for comprehensive research and as a larger training source.
