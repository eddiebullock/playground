# HPC Partition Strategy for CSD3

## Current Partition Status (from sinfo output)

### GPU Partitions

| Partition | Idle Nodes | Allocated | Status | Queue Time |
|-----------|------------|-----------|--------|------------|
| **`ukaea-amp`** | **22** | 0 | ✅ **Best choice** | **Fastest** |
| `ampere` | 0 | 50 | ⚠️ All busy | Longer wait |
| `ampere-long` | 0 | 1 | ⚠️ Limited | Very long wait |
| `ukaea-mi300x` | 7 | 0 | ✅ Available | Fast (if accessible) |

### CPU Partitions (for reference)

| Partition | Idle Nodes | Use Case |
|-----------|------------|----------|
| `icelake` | 90 | CPU jobs |
| `cclake` | 0 | CPU jobs (all busy) |
| `sapphire` | 42 | CPU jobs |

## Recommended Strategy

### Primary Choice: `ukaea-amp`

**Why**: 22 idle nodes available = much faster queue time

```bash
#SBATCH --partition=ukaea-amp
```

**Advantages**:
- ✅ 22 idle nodes (vs 0 in ampere)
- ✅ Faster queue positioning
- ✅ Same GPU hardware (NVIDIA A100)
- ✅ Same resource requirements work

**Check if accessible**:
```bash
sinfo -p ukaea-amp
# If you see nodes, you can use it
```

### Fallback: `ampere`

**If `ukaea-amp` is not accessible to your account**:

```bash
#SBATCH --partition=ampere
```

**Note**: Currently 50 nodes allocated, 0 idle = longer queue time

### Alternative: `ukaea-mi300x`

**If available and accessible** (7 idle nodes):

```bash
#SBATCH --partition=ukaea-mi300x
```

**Note**: Different GPU architecture (AMD MI300X), may need different CUDA setup

## Updated SLURM Scripts

I've created two versions:

1. **`hpc_cam_replication.slurm`** (default): Uses `ukaea-amp` partition
2. **`hpc_cam_replication_ampere.slurm`**: Uses `ampere` partition (fallback)

## How to Choose

### Step 1: Check Partition Access

```bash
# On HPC, check which partitions you can use
sinfo -p ukaea-amp
sinfo -p ampere
sinfo -p ukaea-mi300x
```

### Step 2: Check Current Availability

```bash
# See current status
sinfo | grep -E "PARTITION|ukaea-amp|ampere"

# Check idle nodes specifically
sinfo -p ukaea-amp -o "%P %A"  # Shows available/idle
```

### Step 3: Submit to Best Available

```bash
# Try ukaea-amp first (fastest queue)
sbatch experiments/cam_human_like/training/hpc_cam_replication.slurm

# If that fails (permission denied), use ampere
sbatch experiments/cam_human_like/training/hpc_cam_replication_ampere.slurm
```

## Queue Time Estimates

Based on current status:

| Partition | Estimated Queue Time |
|-----------|---------------------|
| `ukaea-amp` | **Minutes to 1 hour** (22 idle nodes) |
| `ampere` | **Hours to days** (0 idle, 50 allocated) |
| `ampere-long` | **Very long** (1 node, infinite time limit) |

## Resource Optimization Still Applies

Even with better partition, keep resource requests minimal:
- 4 CPUs (not 8)
- 16GB RAM (not 32GB)
- 8 hours (not 24h)

This ensures fastest queue time within the chosen partition.

## Monitoring Queue

```bash
# Check your job position
squeue -u eb2007

# Check partition availability
sinfo -p ukaea-amp
sinfo -p ampere

# Check job details
scontrol show job <job_id>
```

## Troubleshooting

### "Invalid partition" error

If `ukaea-amp` is not accessible:
```bash
# Use ampere partition instead
sbatch experiments/cam_human_like/training/hpc_cam_replication_ampere.slurm
```

### Job stuck in queue

Check partition status:
```bash
sinfo -p ukaea-amp
# If all nodes allocated, queue time will be longer
```

Consider:
- Using `ampere` partition (if ukaea-amp unavailable)
- Reducing resource requests further
- Using interactive session for testing

