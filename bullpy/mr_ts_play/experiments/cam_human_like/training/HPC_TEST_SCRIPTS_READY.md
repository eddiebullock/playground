# HPC Test Scripts - Ready to Transfer

## ✅ What's Been Fixed

1. **Evaluation Script (`evaluate_on_cam.py`)**: 
   - ✅ Fixed import error - now uses `CLIPModel` and `CLIPProcessor` directly from `transformers`
   - ✅ No more dependency on `CLIPWrapper` or `run_experiment` modules
   - ✅ Works for both CAM and EU-Emotion datasets

2. **EU-Emotion Test Script (`hpc_eu_emotion_test.sh`)**:
   - ✅ Updated with additional RDS path checks
   - ✅ Ready to test the full pipeline (training + evaluation)

3. **CAM Test Script (`hpc_cam_test.sh`)**:
   - ✅ NEW: Created quick test script for CAM replication
   - ✅ 2 epochs, ~1-2 hours runtime
   - ✅ Tests full pipeline before committing to 6-hour job

## 📋 Files to Transfer

Transfer these updated/new files to HPC:

```bash
# From your local machine
rsync -avz --progress \
  experiments/cam_human_like/training/evaluate_on_cam.py \
  experiments/cam_human_like/training/hpc_eu_emotion_test.sh \
  experiments/cam_human_like/training/hpc_eu_emotion_test.slurm \
  experiments/cam_human_like/training/hpc_cam_test.sh \
  experiments/cam_human_like/training/hpc_cam_test.slurm \
  eb2007@login-cpu.hpc.cam.ac.uk:~/mr_ts_play/experiments/cam_human_like/training/
```

## 🧪 Running the Tests

### EU-Emotion Test (Quick Test - 2 epochs)

```bash
# On HPC
cd ~/mr_ts_play
sbatch experiments/cam_human_like/training/hpc_eu_emotion_test.slurm

# Monitor progress
squeue -u eb2007
tail -f eu_emotion_test_<JOBID>.out
```

**Expected runtime**: ~30-60 minutes  
**What it tests**: 
- Trial generation
- Training (2 epochs)
- Evaluation (the part that failed before)

### CAM Test (Quick Test - 2 epochs)

```bash
# On HPC
cd ~/mr_ts_play
sbatch experiments/cam_human_like/training/hpc_cam_test.slurm

# Monitor progress
squeue -u eb2007
tail -f cam_test_<JOBID>.out
```

**Expected runtime**: ~1-2 hours  
**What it tests**:
- CAM split creation
- Training (2 epochs)
- Evaluation

## ✅ Success Criteria

Both tests should:
1. ✅ Generate trial definitions/splits successfully
2. ✅ Complete training without errors
3. ✅ Complete evaluation without import errors
4. ✅ Save results to `results/eu_emotion_test/` or `results/cam_test/`

## 🚀 After Tests Pass

Once both tests pass, you can run the full replications:

```bash
# Full EU-Emotion replication (10 epochs, ~6-10 hours)
sbatch experiments/cam_human_like/training/hpc_eu_emotion_replication.slurm

# Full CAM replication (10 epochs, ~6-10 hours)
sbatch experiments/cam_human_like/training/hpc_cam_replication.slurm
```

## 📝 Notes

- Both test scripts use **2 epochs** for quick verification
- Both use **CPU nodes** (icelake partition)
- Evaluation script is now fixed and should work for both datasets
- If evaluation fails, check the error logs in `*_test_<JOBID>.err`


