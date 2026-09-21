# Driver + Thinker practice stack (public models only)

Location: `/Users/eb2007/playground/minimal`

Personal practice repo for an async **Driver** (trajectory) + **Thinker** (Qwen-VL advisory) loop.
Nothing here is the team's private checkpoints, code, or footage.

| Machine | What runs |
|---|---|
| Laptop (CPU) | Driver, combiner, replay harness, optional mock Thinker server |
| CSD3 GPU session | Real Thinker only (`thinker/` folder) |

## What you copy to HPC

**Only the `thinker/` directory.** That is the entire GPU-side program.

Details: [HPC_TRANSFER.md](HPC_TRANSFER.md)

## Layout

```
thinker/          COPY TO HPC. FastAPI server. Loads Qwen once.
driver/           Laptop. DummyDriver today; GarageDriver when weights exist.
integration/      Async client + combiner (advisory waypoints + hard speed cap).
replay/           Footage fetch + frame loop + JSONL logs.
configs/          public.yaml now; team.yaml.example for later.
```

## Laptop setup

```bash
cd /Users/eb2007/playground/minimal
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-laptop.txt

python -m replay.fetch_footage          # public clip -> data/frames/
bash scripts/run_mock_thinker.sh        # terminal A: HTTP mock on :8000
curl -s http://127.0.0.1:8000/health
curl -s -F "image=@data/frames/000001.jpg" -F "frame_index=0" \
  http://127.0.0.1:8000/think

python -m replay.harness --config configs/public.yaml   # terminal B
```

In-process mock (no HTTP), to prove the backend swap is one flag:

```bash
python -m replay.harness --config configs/public.yaml --thinker local-mock
```

## Models (public)

- **Thinker:** `Qwen/Qwen2-VL-2B-Instruct` (Hugging Face). No `Qwen2.5-VL-2B` exists.
- **Driver today:** `DummyDriver` (kinematic waypoints). Interface matches the real one.
- **Driver optional:** official TransFuser++ from [carla_garage](https://github.com/autonomousvision/carla_garage) `leaderboard_2`, one seed `model_0030_0.pth`. No CARLA, **no lidar** (zeros). `scripts/fetch_driver_checkpoint.sh` — large, skip until you want it.

## Footage

`replay/fetch_footage.py` grabs whatever is reachable without an account:

1. comma2k19 example dashcam (best if GitHub serves the HEVC)
2. Intel `car-detection.mp4` fallback (CCTV, not ego-cam)
3. comma2k19 `preview.png` duplicated as a placeholder sequence

Drop your own clip in `data/raw/` and re-run the fetch script, or copy nuScenes `CAM_FRONT` stills into `data/frames/` (nuScenes is CC BY-NC-SA — personal practice only).

## Contract

Thinker `POST /think` (multipart image) returns:

```
HAZARD=...; ACTION=...; SPEED_CAP_MPS=...
```

plus timestamps. Combiner:

- does **not** replace the Driver trajectory
- may **nudge** waypoint `y` (NUDGE_LEFT / NUDGE_RIGHT)
- may **hard-cap** speed (SLOW / STOP / explicit SPEED_CAP_MPS)
- if the server is slow, down, or the HPC session dies: keep last-known advisory, else `SLOW` at 3 m/s

## Logs

Each replay line in `data/logs/*.jsonl` has `driver`, `thinker` (fresh/stale), `combined`, and `timings_ms` (driver, thinker inference, network RTT, loop).

## Later, with team access

See `configs/team.yaml.example`. You should only swap Driver weights/class, Thinker `model_id` / prompt, and `frames_dir`. Do not rewrite the async loop.
