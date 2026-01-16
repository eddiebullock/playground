# How to Recover Deleted Files

## Status: Files Were Deleted But Can Be Recovered!

**Good news:** The files are still in git history, so they can be recovered.

## What Was Deleted

**From `experiments/eu_emotion_model_comparison/`:**
- Many documentation files (README.md, FINETUNING_GUIDE.md, etc.)
- Some scripts (run_comparison.py, generate_report.py, etc.)
- Some model files (evaluator.py, metrics.py, etc.)
- Config files (comparison_config.yaml)

**From root:**
- `analyze_all_results.py`
- `GOOGLE_CLOUD_TROUBLESHOOTING.md`
- `GOOGLE_SAFETY_FIX.md`

**Critical files that still exist:**
- ✅ `finetune_video_models_task_specific.py` (exists)
- ⚠️ `hpc_finetune_video_models.sh` (exists but empty - needs restoration)
- ❌ `test_adaptive_fusion.py` (missing - needs recreation)

## How to Recover

### Option 1: Restore All from Git (Recommended)

**Run this in your terminal (outside Cursor):**

```bash
cd /Users/eb2007/playground/bullpy/mr_ts_play

# Restore all deleted files
git checkout HEAD -- experiments/eu_emotion_model_comparison/
git checkout HEAD -- analyze_all_results.py
git checkout HEAD -- GOOGLE_CLOUD_TROUBLESHOOTING.md
git checkout HEAD -- GOOGLE_SAFETY_FIX.md
```

### Option 2: Restore Specific Files

```bash
# Restore specific files
git checkout HEAD -- experiments/eu_emotion_model_comparison/training/hpc_finetune_video_models.sh
git checkout HEAD -- experiments/eu_emotion_model_comparison/README.md
# etc.
```

### Option 3: See What Was Deleted

```bash
# List all deleted files
git status | grep "deleted:"

# See what a deleted file looked like
git show HEAD:experiments/eu_emotion_model_comparison/README.md
```

## Critical Files to Restore

**Must restore:**
1. `experiments/eu_emotion_model_comparison/training/hpc_finetune_video_models.sh` (currently empty!)
2. `experiments/llm_augmented_emotion_recognition/scripts/test_adaptive_fusion.py` (we created this - not in git, need to recreate)

**Nice to have:**
- Documentation files (README.md, guides, etc.)
- Scripts (run_comparison.py, etc.)

## Recreate Missing Files

**The adaptive fusion script** (`test_adaptive_fusion.py`) was created today and isn't in git. I'll recreate it for you.

## Quick Recovery Command

**Run this in your terminal:**

```bash
cd /Users/eb2007/playground/bullpy/mr_ts_play

# Restore everything from git
git checkout HEAD -- .

# This will restore all deleted files that were tracked in git
```

**Note:** This will also restore any other files you may have intentionally deleted, so review `git status` first if needed.
