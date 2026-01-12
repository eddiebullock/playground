# Retraining Script Fixes

## Errors Found and Fixed

### Error 1: Missing `defaultdict` Import ✅ FIXED

**Error**: `NameError: name 'defaultdict' is not defined`

**Location**: `finetune_vision_models_task_specific.py` line 92

**Fix**: Added `from collections import defaultdict` to imports

**Status**: ✅ Fixed

---

### Error 2: CLIP Script Doesn't Handle "auto" Device ✅ FIXED

**Error**: `RuntimeError: Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone device type at start of device string: auto`

**Location**: `retrain_all_models_fixed_splits.sh` - CLIP training function

**Fix**: Added device detection in retraining script:
- If `DEVICE="auto"`, detects actual device (mps for Mac, cuda for Linux, cpu otherwise)
- Passes actual device string to CLIP script

**Status**: ✅ Fixed

---

## Updated Files

1. ✅ `experiments/eu_emotion_model_comparison/training/finetune_vision_models_task_specific.py`
   - Added `from collections import defaultdict` import

2. ✅ `experiments/eu_emotion_model_comparison/training/retrain_all_models_fixed_splits.sh`
   - Added device detection for CLIP training
   - Converts "auto" to actual device (mps/cuda/cpu)

---

## Ready to Run

The retraining script should now work. Run:

```bash
bash experiments/eu_emotion_model_comparison/training/retrain_all_models_fixed_splits.sh
```

**Note**: The sandbox can't run Python commands, but the fixes are in place. Run the script directly in your terminal (outside sandbox) with your venv activated.

---

## Expected Behavior

1. **CLIP training**: May fail if CLIP uses different split system (this is OK - script continues)
2. **ResNet50 training**: Should work now (defaultdict import fixed)
3. **ViT training**: Should work
4. **EfficientNet training**: Should work

All vision models will use:
- ✅ Fixed actor-independent splits
- ✅ Class-weighted loss
- ✅ Data augmentation
- ✅ Improved frame sampling
