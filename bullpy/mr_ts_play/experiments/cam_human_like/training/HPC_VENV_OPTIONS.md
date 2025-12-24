# Python Environment Options: /home vs RDS

## Why RDS for Python Environment?

### Current Situation
- **/home quota**: 50GB limit, currently at ~46GB (after removing EU_emotions)
- **Python venv size**: ~5-10GB (PyTorch + dependencies)
- **RDS available**: ~10TB available

### Option 1: Venv on RDS (Recommended)

**Advantages:**
- ✅ **No quota issues**: RDS has 10TB vs 50GB on /home
- ✅ **Future-proof**: Room for larger models, more packages
- ✅ **Better for large files**: RDS optimized for large I/O
- ✅ **Shared access**: Other project members can use same venv

**Disadvantages:**
- ⚠️  Slightly more complex setup
- ⚠️  Need to create symlink

### Option 2: Venv on /home (Simpler)

**Advantages:**
- ✅ **Simpler**: Just `python3 -m venv venv` in project directory
- ✅ **No symlinks needed**: Everything in one place
- ✅ **Faster access**: /home is typically faster for small files

**Disadvantages:**
- ⚠️  **Quota risk**: With venv (~5-10GB), you'd be at ~51-56GB (over 50GB quota)
- ⚠️  **Less room**: No space for model checkpoints, results
- ⚠️  **Future limitations**: Can't install more packages easily

## Recommendation

**Use RDS for venv** because:
1. You're already close to /home quota (46GB/50GB)
2. Venv + model checkpoints + results would exceed quota
3. RDS has plenty of space (10TB)
4. Data (EU_emotions, CAM) can also go on RDS

## But You Can Use /home If You Prefer

If you want simplicity and don't plan to store large files in /home:

```bash
# On HPC
cd ~/mr_ts_play
module load python/3.11.9/gcc/nptrdpll
python3 -m venv --without-pip venv
source venv/bin/activate
curl -sS https://bootstrap.pypa.io/get-pip.py | python3
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers pillow opencv-python tqdm numpy pandas
```

**Just make sure:**
- Store model checkpoints on RDS: `results/` → RDS
- Store large outputs on RDS
- Keep /home for code and small files only

## Hybrid Approach (Best of Both)

- **Code**: `/home/eb2007/mr_ts_play` (small, easy to edit)
- **Venv**: RDS (large, ~5-10GB)
- **Data**: RDS (EU_emotions, large datasets)
- **Results/Models**: RDS (checkpoints can be GB each)

This gives you:
- ✅ Code in /home (fast, easy access)
- ✅ Large files on RDS (plenty of space)
- ✅ Under /home quota


