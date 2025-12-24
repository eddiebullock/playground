# Verify Test Results and Run Full Replication

## Step 1: Check Test Job Results

Run these commands on HPC to verify the tests completed successfully:

```bash
cd ~/mr_ts_play

# Quick check - see final output
echo "=== CAM Test (Job 19803784) ==="
tail -30 cam_test_19803784.out
echo ""
echo "=== EU-Emotion Test (Job 19803785) ==="
tail -30 eu_emotion_test_19803785.out
echo ""

# Check for success indicators
echo "=== Success Check ==="
grep -i "complete\|success\|accuracy" cam_test_19803784.out | tail -5
grep -i "complete\|success\|accuracy" eu_emotion_test_19803785.out | tail -5
echo ""

# Check if models were saved
echo "=== Model Checkpoints ==="
ls -lh results/cam_test/model_checkpoints/best_model/ 2>/dev/null | head -3
ls -lh results/eu_emotion_test/model_checkpoints/best_model/ 2>/dev/null | head -3
echo ""

# Check evaluation files
echo "=== Evaluation Files ==="
ls -lh results/*/model_checkpoints/*evaluation*.json 2>/dev/null
echo ""

# Check for errors
echo "=== Error Check ==="
grep -i "error\|exception\|failed" cam_test_19803784.out cam_test_19803784.err | tail -5
grep -i "error\|exception\|failed" eu_emotion_test_19803785.out eu_emotion_test_19803785.err | tail -5
```

## Step 2: Check Accuracy Results

```bash
# View evaluation results
cat results/cam_test/model_checkpoints/cam_evaluation_test.json 2>/dev/null | python3 -m json.tool
cat results/eu_emotion_test/model_checkpoints/eu_emotion_evaluation_test.json 2>/dev/null | python3 -m json.tool

# Or just check accuracy
grep -i "accuracy" results/cam_test/model_checkpoints/*evaluation*.json 2>/dev/null
grep -i "accuracy" results/eu_emotion_test/model_checkpoints/*evaluation*.json 2>/dev/null
```

## Step 3: Verify Pipeline Structure Matches

The full replication scripts should match the test scripts. Verify:

```bash
# Check that both use the same structure
echo "=== CAM Scripts Comparison ==="
echo "Test script steps:"
grep "^# Step" experiments/cam_human_like/training/hpc_cam_test.sh
echo ""
echo "Replication script steps:"
grep "^# Step" experiments/cam_human_like/training/hpc_cam_replication.sh
echo ""

echo "=== EU-Emotion Scripts Comparison ==="
echo "Test script steps:"
grep "^# Step" experiments/cam_human_like/training/hpc_eu_emotion_test.sh
echo ""
echo "Replication script steps:"
grep "^# Step" experiments/cam_human_like/training/hpc_eu_emotion_replication.sh
```

## Step 4: Run Full Replication

Once tests are verified successful, run the full replications:

```bash
cd ~/mr_ts_play

# Submit full CAM replication (10 epochs, 10 trials per concept)
sbatch experiments/cam_human_like/training/hpc_cam_replication.slurm

# Submit full EU-Emotion replication (10 epochs, 10 trials per emotion)
sbatch experiments/cam_human_like/training/hpc_eu_emotion_replication.slurm

# Monitor jobs
squeue -u eb2007

# Check job status
sacct -u eb2007 --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

## Expected Test Results

### CAM Test
- **Trials**: 100 total (80 train, 20 test)
- **Epochs**: 2
- **Expected accuracy**: Should be above random (25%) and zero-shot baseline (37%)
- **Model saved**: `results/cam_test/model_checkpoints/best_model/`
- **Evaluation**: `results/cam_test/model_checkpoints/cam_evaluation_test.json`

### EU-Emotion Test
- **Trials**: ~135 total (108 train, 27 test) - 5 per emotion
- **Epochs**: 2
- **Expected accuracy**: Should be above random (25%)
- **Model saved**: `results/eu_emotion_test/model_checkpoints/best_model/`
- **Evaluation**: `results/eu_emotion_test/model_checkpoints/eu_emotion_evaluation_test.json`

## Full Replication Differences

| Aspect | Test | Full Replication |
|--------|------|------------------|
| **Epochs** | 2 | 10 |
| **CAM Trials/Concept** | 5 | 10 |
| **EU-Emotion Trials/Emotion** | 5 | 10 |
| **Runtime** | 1-4 hours | 6-12 hours |
| **Expected Accuracy** | Above baseline | 70-85% (CAM), 60-75% (EU-Emotion) |

## Quick Verification Script

Run this all-in-one check:

```bash
cd ~/mr_ts_play

echo "=========================================="
echo "TEST RESULTS VERIFICATION"
echo "=========================================="
echo ""

# Check completion
echo "1. Job Status:"
sacct -j 19803784,19803785 --format=JobID,JobName,State,ExitCode

echo ""
echo "2. Final Output:"
tail -10 cam_test_19803784.out
tail -10 eu_emotion_test_19803785.out

echo ""
echo "3. Model Checkpoints:"
[ -d "results/cam_test/model_checkpoints/best_model" ] && echo "✅ CAM model exists" || echo "❌ CAM model missing"
[ -d "results/eu_emotion_test/model_checkpoints/best_model" ] && echo "✅ EU-Emotion model exists" || echo "❌ EU-Emotion model missing"

echo ""
echo "4. Evaluation Files:"
[ -f "results/cam_test/model_checkpoints/cam_evaluation_test.json" ] && echo "✅ CAM evaluation exists" || echo "❌ CAM evaluation missing"
[ -f "results/eu_emotion_test/model_checkpoints/eu_emotion_evaluation_test.json" ] && echo "✅ EU-Emotion evaluation exists" || echo "❌ EU-Emotion evaluation missing"

echo ""
echo "5. Accuracy Results:"
if [ -f "results/cam_test/model_checkpoints/cam_evaluation_test.json" ]; then
    python3 -c "import json; d=json.load(open('results/cam_test/model_checkpoints/cam_evaluation_test.json')); print(f\"CAM Accuracy: {d.get('accuracy', 'N/A'):.2%}\")" 2>/dev/null
fi
if [ -f "results/eu_emotion_test/model_checkpoints/eu_emotion_evaluation_test.json" ]; then
    python3 -c "import json; d=json.load(open('results/eu_emotion_test/model_checkpoints/eu_emotion_evaluation_test.json')); print(f\"EU-Emotion Accuracy: {d.get('accuracy', 'N/A'):.2%}\")" 2>/dev/null
fi

echo ""
echo "=========================================="
echo "If all checks pass, you can run full replication!"
echo "=========================================="
```

