#!/usr/bin/env bash
# Optional. Downloads official TransFuser++ weights (CC BY 4.0) from
# autonomousvision/carla_garage. Several GB. Not required for today's dummy path.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/checkpoints"
mkdir -p "$DEST"
ZIP="$DEST/pretrained_models.zip"
URL="https://s3.eu-central-1.amazonaws.com/avg-projects-2/garage_2/models/pretrained_models.zip"
echo "Downloading $URL"
curl -L --fail --retry 3 -o "$ZIP" "$URL"
unzip -o "$ZIP" -d "$DEST"
echo "Pick one seed folder (config.json + model_0030_0.pth) and set"
echo "  driver.checkpoint_dir  in configs/public.yaml"
echo "Also clone garage code:"
echo "  git clone --branch leaderboard_2 --depth 1 https://github.com/autonomousvision/carla_garage.git $ROOT/vendor/carla_garage"
echo "Then set driver.backend: garage"
