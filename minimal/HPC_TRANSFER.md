# What to transfer to HPC (CSD3)

The laptop runs the Driver, combiner, and replay harness.
The GPU session runs **only** the Thinker HTTP server.

## Copy this and nothing else

```
thinker/
  server.py
  model.py
  schemas.py
  requirements.txt
  run_hpc.sh
  README.md
```

From your laptop:

```bash
ssh eb2007@login.hpc.cam.ac.uk "mkdir -p ~/rds/hpc-work/driver-thinker-practice"
scp -r /Users/eb2007/playground/minimal/thinker \
  eb2007@login.hpc.cam.ac.uk:~/rds/hpc-work/driver-thinker-practice/
```

Put it on **rds/hpc-work**, not `$HOME` on the login node (quota + no GPU).

Do **not** copy: `driver/`, `integration/`, `replay/`, `data/`, garage checkpoints, footage. The laptop sends JPEGs over the tunnel; the GPU does not need the clip.

## On CSD3: GPU session, then install, then serve

1. Leave the login node. Request an interactive GPU job (Wilkes3 / `sintr` — use whatever your local docs say). Note the GPU hostname.
2. On that GPU node:

```bash
cd ~/rds/hpc-work/driver-thinker-practice/thinker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# If Qwen2-VL import fails: pip install -U "transformers>=4.45"
export HF_HOME=$HOME/rds/hpc-work/hf-cache
export TOKENIZERS_PARALLELISM=false
bash run_hpc.sh
```

First start downloads `Qwen/Qwen2-VL-2B-Instruct` into `HF_HOME` (~4–5 GB). Keep the session alive.

3. Smoke test **on the GPU node**:

```bash
curl -s http://127.0.0.1:8000/health
# optional: scp one jpg and
curl -s -F "image=@frame.jpg" -F "frame_index=0" http://127.0.0.1:8000/think
```

You want `HAZARD=...; ACTION=...; SPEED_CAP_MPS=...` in the JSON `raw_text` field.

## Laptop tunnel

The Thinker binds `127.0.0.1:8000` on the GPU node. From the laptop:

```bash
# replace gpu-q-XX with the hostname sintr printed
bash scripts/tunnel.sh gpu-q-XX
```

Leave that terminal open. Then:

```bash
curl -s http://127.0.0.1:8000/health
python -m replay.harness --config configs/public.yaml --thinker-url http://127.0.0.1:8000
```

If health fails, the harness still runs: last-known advisory, or safe default `SLOW` @ 3 m/s. It must not hang when the HPC job dies.

## What stays mock until the GPU is up

`bash scripts/run_mock_thinker.sh` on the laptop is the same HTTP contract with a hash-based fake model. Use it to debug the loop without burning GPU time. `--mock` is **not** Qwen-VL.

## Later: team Thinker

Same `server.py`. Change `--model-id` to the team's checkpoint path (or edit `QwenInferencer`). Keep the `HAZARD=...; ACTION=...` line format so `integration/combiner.py` does not change.
