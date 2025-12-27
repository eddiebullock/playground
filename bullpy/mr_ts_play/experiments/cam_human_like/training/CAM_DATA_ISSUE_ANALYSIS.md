# CAM Dataset Issue Analysis

## Problem Identified

**10 out of 20 CAM test trials are failing** due to corrupted video files.

### Root Cause

All 10 failed trials are **voice modality** trials using corrupted `.mov` files:
- Files ending in `T` (e.g., `P6T`, `U6T`) are **corrupted/incomplete** (4-16KB)
- These files exist locally but are too small to contain valid video/audio content
- OpenCV cannot open them (file size validation fails at 50KB threshold)

### Failed Trials

| Trial ID | Emotion | Corrupted File | Size |
|----------|---------|----------------|------|
| trial_015 | confronted | 1202701P6Tconfronted.mov | 10.5 KB |
| trial_034 | resentful | 0101702U6Tresentful.mov | 9.1 KB |
| trial_100 | exonerated | 0305802U6Texonerated.mov | 13.7 KB |
| trial_020 | nostalgic | 2300501P5Tnostalgic.mov | 16.2 KB |
| trial_069 | subservient | 1004701P4Tsubservient.mov | 8.2 KB |
| trial_080 | appalled | 1101401R6Tappalled.mov | 15.9 KB |
| trial_029 | uneasy | 0601901P3Tuneasy.mov | 11.6 KB |
| trial_065 | appealing | 1900201A6Tappealing.mov | 10.8 KB |
| trial_060 | guarded | 2100102P5Tguarded.mov | 11.2 KB |
| trial_055 | empathic | 0703001U5Tempathic.mov | 4.8 KB |

### Available Solutions

#### Option 1: Use Valid Video Files (V files)
- **All 10 trials have valid `.mov` files ending in `V`** (179-318KB)
- These are face videos but contain audio tracks
- Pattern: `{scenario_id}{actor}V{emotion}.mov` (e.g., `1202701M1Vconfronted.mov`)
- **Pros**: Files are valid, CLIP can process them
- **Cons**: These are face videos, not pure voice files

#### Option 2: Use Audio Files (.aif)
- **All 10 trials have corresponding `.aif` audio files** in `Audio/2/emotion/`
- Pattern: `{scenario_id}{emotion}_2w.aif` (e.g., `1202701confronted_2w.aif`)
- **Pros**: These are actual voice/audio files
- **Cons**: CLIP cannot process audio files directly (would need audio model)

#### Option 3: Fix Trial Definitions
- Update trial definitions to use valid `V` files instead of corrupted `T` files
- Keep same actors where possible, or use alternative actors
- **Pros**: Maintains dataset integrity
- **Cons**: Changes trial definitions (may affect reproducibility)

## Impact on Results

### Current Results (10 valid trials)
- Overall Accuracy: **50.00%**
- Face Accuracy: **33.3%** (below 37% baseline)
- Voice Accuracy: **75%** (well above baseline)
- **Sample size too small** (10 trials) for reliable metrics

### Expected Results (20 valid trials)
- With all 20 trials, metrics would be more reliable
- Face accuracy might improve with more face trials
- Voice accuracy might normalize with more voice trials

## Recommended Fix

**Update trial definitions to use valid `V` files** for voice trials:
1. For each failed trial, find a valid `V` file with the same emotion
2. Prefer files with the same actor if available
3. Update `cam_trial_definitions_20concepts.json`
4. Re-run evaluation on HPC

## Files to Update

- `data/cam_trial_definitions_20concepts.json` - Update stimulus_path for 10 failed trials
- Consider creating a script to automatically find and replace corrupted files



