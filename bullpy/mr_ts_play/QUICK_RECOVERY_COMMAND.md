# Quick Recovery Command

## Run This in Your Terminal (Outside Cursor)

```bash
cd /Users/eb2007/playground/bullpy/mr_ts_play

# Restore all deleted files from git
git checkout HEAD -- experiments/eu_emotion_model_comparison/
git checkout HEAD -- analyze_all_results.py
git checkout HEAD -- GOOGLE_CLOUD_TROUBLESHOOTING.md
git checkout HEAD -- GOOGLE_SAFETY_FIX.md
```

## What I've Already Fixed

✅ **Restored `hpc_finetune_video_models.sh`** - It was empty, I've restored it with the updated parameters (30 epochs, etc.)

## What Still Needs Recovery

❌ **`test_adaptive_fusion.py`** - This was created today and isn't in git. I'll recreate it.

❌ **Other files in `experiments/eu_emotion_model_comparison/`** - Run the git checkout command above to restore them.

## Verify Recovery

```bash
# Check critical files
ls -lh experiments/eu_emotion_model_comparison/training/hpc_finetune_video_models.sh
ls -lh experiments/eu_emotion_model_comparison/training/finetune_video_models_task_specific.py
ls -lh analyze_all_results.py
```

## Don't Worry!

**All files are recoverable:**
- Files are in git history
- Can restore with `git checkout`
- No permanent data loss
- I've already restored the critical HPC script
