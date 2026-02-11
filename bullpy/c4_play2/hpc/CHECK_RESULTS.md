# How to Check Job Results

## Step 1: Check Job Status (Finished or Failed?)

```bash
# Navigate to c4 directory
cd /home/eb2007/c4

# Check if job finished successfully
sacct -j 20710759 --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS

# Or check all your recent jobs
sacct -u eb2007 -S $(date -d '1 day ago' +%Y-%m-%d) --format=JobID,JobName,State,ExitCode,Elapsed
```

## Step 2: Check Final Output

```bash
cd /home/eb2007/c4

# View full output to see what happened
cat logs/comprehensive_optimization_20710759.out

# Or view last 100 lines
tail -100 logs/comprehensive_optimization_20710759.out

# Check for errors
cat logs/comprehensive_optimization_20710759.err
```

## Step 3: Check if Results Were Created

```bash
cd /home/eb2007/c4

# Check results directory
ls -lh results/

# Check models directory
ls -lh models/

# Look for timestamped files
ls -lht results/ | head -10
ls -lht models/ | head -10
```

## Step 4: View Results (if they exist)

```bash
# View JSON results (if created)
cat results/comprehensive_results_*.json | head -50

# View CSV summary (if created)
cat results/comprehensive_results_*.csv

# Or use less for scrolling
less results/comprehensive_results_*.csv
```

## Step 5: Check What Happened

```bash
# One-liner to check everything
cd /home/eb2007/c4 && \
echo "=== Job Status ===" && \
sacct -j 20710759 --format=JobID,JobName,State,ExitCode,Elapsed && \
echo -e "\n=== Output File ===" && \
ls -lh logs/comprehensive_optimization_20710759.out && \
echo -e "\n=== Last 20 lines of output ===" && \
tail -20 logs/comprehensive_optimization_20710759.out && \
echo -e "\n=== Results Files ===" && \
ls -lh results/ 2>/dev/null || echo "No results directory" && \
echo -e "\n=== Model Files ===" && \
ls -lh models/ 2>/dev/null | head -10 || echo "No models directory"
```

## What to Look For

### Success Indicators:
- ExitCode: 0:0 (in sacct)
- Output ends with "Optimization completed!"
- Results files exist in `results/`
- Model files exist in `models/`

### Failure Indicators:
- ExitCode: non-zero (e.g., 1:0, 137:0)
- Output ends with error messages
- No results files created
- Error log has content

## If Job Timed Out

If the job hit the 6-hour limit:
- Check how far it got (which models completed)
- Results may be partial but still useful
- You can resume from where it left off
