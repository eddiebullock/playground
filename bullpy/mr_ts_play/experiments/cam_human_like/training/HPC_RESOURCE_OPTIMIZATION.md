# HPC Resource Optimization Guide

## CSD3 System Overview

Based on [Cambridge HPC Documentation](https://docs.hpc.cam.ac.uk/hpc/):

- **GPU Nodes**: NVIDIA A100 GPUs (Ampere architecture)
- **Scheduler**: SLURM
- **Partitions**: `ampere` (GPU nodes), `cclake`, `icelake`, `sapphirerapids` (CPU nodes)
- **Key Principle**: Request only what you need to improve queue positioning

## Resource Optimization Strategy

### Why Optimize?

**Queue positioning** on HPC depends on:
1. **Resource requests**: Smaller requests = faster queue time
2. **Time limits**: Shorter realistic time = faster scheduling
3. **Partition selection**: Correct partition = better availability

### Optimized Resource Requests

For CLIP fine-tuning (CAM replication):

| Resource | Requested | Rationale |
|----------|-----------|-----------|
| **GPUs** | 1 | Single GPU sufficient for batch_size=16 |
| **CPUs** | 4 | Enough for data loading, not excessive |
| **Memory** | 16GB | Sufficient for batch_size=16, not wasteful |
| **Time** | 8 hours | Realistic for 10 epochs, shorter = faster queue |
| **Partition** | `ampere` | GPU nodes on CSD3 |

### Comparison: Before vs After

| Resource | Before | After | Impact |
|----------|--------|-------|--------|
| CPUs | 8 | 4 | ✅ Faster queue (less resource contention) |
| Memory | 32GB | 16GB | ✅ Faster queue (less memory pressure) |
| Time | 24h | 8h | ✅ Faster queue (shorter jobs prioritized) |
| Partition | `gpu` | `ampere` | ✅ More specific = better targeting |

## CSD3-Specific Considerations

### Partition Selection

According to CSD3 documentation:
- **`ampere`**: NVIDIA A100 GPU nodes (use for GPU jobs)
- **`cclake`**: Cascade Lake CPU nodes
- **`icelake`**: Ice Lake CPU nodes
- **`sapphirerapids`**: Sapphire Rapids CPU nodes

**For GPU jobs**: Use `ampere` partition explicitly.

### Module Loading

CSD3 uses environment modules. Check available modules:

```bash
module avail python
module avail cuda
module avail cudnn
```

Common modules for PyTorch:
```bash
module load python/3.9  # or latest available
module load cuda/11.8   # or latest available
```

### PyTorch on CSD3

CSD3 has specific PyTorch documentation. Check:
- [PyTorch on CSD3](https://docs.hpc.cam.ac.uk/hpc/software-apps/pytorch.html)

Install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Batch Size Optimization

### Memory Considerations

With **16GB RAM**:
- `batch_size=16` works well for CLIP fine-tuning
- Each sample: ~8 frames × 224×224 images = ~1-2MB per sample
- Batch of 16: ~16-32MB per batch (well within 16GB)

### If OOM Errors Occur

Reduce batch size in `hpc_cam_replication.sh`:
```bash
BATCH_SIZE=8  # Instead of 16
```

## Time Limit Strategy

### Realistic Time Estimates

For CAM replication (10 epochs, batch_size=16):
- **Training**: ~2-4 hours
- **Evaluation**: ~10-20 minutes
- **Total**: ~3-5 hours

**Request 8 hours** to allow for:
- Queue wait time
- Data loading overhead
- Checkpoint saving
- Safety margin

### If Job Times Out

Increase time limit:
```bash
#SBATCH --time=12:00:00  # Instead of 8:00:00
```

## Monitoring and Optimization

### Check Resource Usage

After job runs, check actual usage:
```bash
# Check job efficiency
seff <job_id>

# Check GPU utilization
nvidia-smi  # During job execution
```

### Adjust Based on Results

If resources are underutilized:
- Reduce CPU count (if <50% usage)
- Reduce memory (if <50% usage)
- This improves queue positioning for future jobs

## Alternative: Interactive Testing

For initial testing, use interactive session:

```bash
# Request interactive GPU session (shorter queue)
srun --gres=gpu:1 --time=2:00:00 --cpus-per-task=4 --mem=16G --partition=ampere --pty bash

# Then run script interactively
cd ~/mr_ts_play
bash experiments/cam_human_like/training/hpc_cam_replication.sh
```

**Advantages**:
- Faster queue (interactive jobs often prioritized)
- Can test before submitting long job
- Immediate feedback

## Queue Monitoring

```bash
# Check queue status
squeue -u eb2007

# Check partition availability
sinfo -p ampere

# Check your job details
scontrol show job <job_id>
```

## Best Practices Summary

1. ✅ **Request minimum necessary resources**
2. ✅ **Use realistic time limits** (not maximum)
3. ✅ **Select correct partition** (`ampere` for GPU)
4. ✅ **Start with interactive session** for testing
5. ✅ **Monitor resource usage** and adjust
6. ✅ **Use appropriate batch size** for available memory

## Troubleshooting

### Job Stuck in Queue

- Check if partition is available: `sinfo -p ampere`
- Reduce resource requests if possible
- Check if other users have priority

### Out of Memory

- Reduce `BATCH_SIZE` in script
- Request more memory: `#SBATCH --mem=32G` (but slower queue)

### GPU Not Available

- Check: `nvidia-smi` during job
- Verify CUDA: `python3 -c "import torch; print(torch.cuda.is_available())"`
- Check module loading


