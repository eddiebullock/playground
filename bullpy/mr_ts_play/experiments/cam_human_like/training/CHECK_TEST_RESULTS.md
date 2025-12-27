# Commands to Check Test Job Results

## Jobs Completed

- **Job 19803784**: `cam_test` - COMPLETED ✅
- **Job 19803785**: `eu_emotion_test` - COMPLETED ✅

## Step 1: Check Job Output Files

```bash
cd ~/mr_ts_play

# Check CAM test results
cat cam_test_19803784.out | tail -50
cat cam_test_19803784.err | tail -20

# Check EU-Emotion test results
cat eu_emotion_test_19803785.out | tail -50
cat eu_emotion_test_19803785.err | tail -20
```

## Step 2: Check for Success Indicators

```bash
# Look for success messages
grep -i "complete\|success\|accuracy\|evaluation" cam_test_19803784.out
grep -i "complete\|success\|accuracy\|evaluation" eu_emotion_test_19803785.out

# Check if models were saved
ls -lh results/cam_test/model_checkpoints/best_model/
ls -lh results/eu_emotion_test/model_checkpoints/best_model/

# Check if evaluation files exist
ls -lh results/cam_test/model_checkpoints/*evaluation*.json
ls -lh results/eu_emotion_test/model_checkpoints/*evaluation*.json
```

## Step 3: Check Final Accuracy

```bash
# Check CAM test accuracy
grep -i "accuracy\|overall" results/cam_test/model_checkpoints/*evaluation*.json 2>/dev/null
cat results/cam_test/model_checkpoints/*evaluation*.json 2>/dev/null | python3 -m json.tool

# Check EU-Emotion test accuracy
grep -i "accuracy\|overall" results/eu_emotion_test/model_checkpoints/*evaluation*.json 2>/dev/null
cat results/eu_emotion_test/model_checkpoints/*evaluation*.json 2>/dev/null | python3 -m json.tool
```

## Step 4: Verify Pipeline Completed

```bash
# Check if all steps completed
grep -E "Step [123]|Complete|evaluation" cam_test_19803784.out
grep -E "Step [123]|Complete|evaluation" eu_emotion_test_19803785.out

# Check for errors
grep -i "error\|exception\|failed\|traceback" cam_test_19803784.out cam_test_19803784.err
grep -i "error\|exception\|failed\|traceback" eu_emotion_test_19803785.out eu_emotion_test_19803785.err
```

## Quick Success Check

```bash
cd ~/mr_ts_play

# All-in-one check
echo "=== CAM Test Results ==="
tail -20 cam_test_19803784.out
echo ""
echo "=== EU-Emotion Test Results ==="
tail -20 eu_emotion_test_19803785.out
echo ""
echo "=== Model Checkpoints ==="
ls -lh results/cam_test/model_checkpoints/best_model/ 2>/dev/null | head -5
ls -lh results/eu_emotion_test/model_checkpoints/best_model/ 2>/dev/null | head -5
echo ""
echo "=== Evaluation Files ==="
ls -lh results/*/model_checkpoints/*evaluation*.json 2>/dev/null
```

## Expected Success Indicators

✅ **Success if you see:**
- "CAM Test Complete!" or "EU-Emotion Quick Test Complete!"
- "Evaluation complete!"
- Model checkpoint directory exists: `results/*/model_checkpoints/best_model/`
- Evaluation JSON file exists: `results/*/model_checkpoints/*evaluation*.json`
- Accuracy reported in output or JSON file

❌ **Failure if you see:**
- "Error:" messages
- "Failed" or "Exception" in output
- No model checkpoint directory
- No evaluation file



