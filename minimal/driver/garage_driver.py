"""Optional TransFuser++ (carla_garage) loader.

PUBLIC PRACTICE LIMITS
- No CARLA. No lidar. A zero lidar-BEV is passed into the network.
- Trajectory quality on real dashcam frames will be poor. The point is the
  I/O contract: RGB(+speed+command) -> waypoints / BEV / boxes.
- Official eval still needs CARLA 0.9.15 + leaderboard. Do not do that today.

Expected layout after you fetch weights (see scripts/fetch_driver_checkpoint.sh):

    data/checkpoints/tfpp_seed0/config.json
    data/checkpoints/tfpp_seed0/model_0030_0.pth
    vendor/carla_garage/team_code/   (cloned repo, leaderboard_2 branch)

If those paths are missing, load_driver(backend=garage) raises a clear error
and the replay harness should stay on backend=dummy.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Optional

import cv2
import numpy as np

from .interface import COMMANDS, DriverOutput


class GarageDriver:
    backend_name = "garage"

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.checkpoint_dir = cfg.get("checkpoint_dir", "data/checkpoints/tfpp_seed0")
        self.team_code = cfg.get("garage_team_code", "vendor/carla_garage/team_code")
        self.image_w = int(cfg.get("image_width", 1024))
        self.image_h = int(cfg.get("image_height", 256))
        self.device_name = cfg.get("device", "cpu")
        self.model = self._load()

    def _load(self) -> Any:
        ckpt_json = os.path.join(self.checkpoint_dir, "config.json")
        weights = self._find_weights()
        if not os.path.isdir(self.team_code):
            raise FileNotFoundError(
                f"garage team_code not found at {self.team_code}. "
                "Clone autonomousvision/carla_garage (branch leaderboard_2) into vendor/."
            )
        if not os.path.isfile(ckpt_json) or weights is None:
            raise FileNotFoundError(
                f"TransFuser++ checkpoint not found in {self.checkpoint_dir}. "
                "Run scripts/fetch_driver_checkpoint.sh (large download)."
            )
        team_code_abs = os.path.abspath(self.team_code)
        parent = os.path.dirname(team_code_abs)
        for path in (team_code_abs, parent):
            if path not in sys.path:
                sys.path.insert(0, path)
        self._install_carla_stub()

        import torch

        # Imported from garage after sys.path patch.
        from model import LidarCenterNet  # type: ignore
        from config import GlobalConfig  # type: ignore

        with open(ckpt_json, "r", encoding="utf-8") as f:
            saved = json.load(f)
        config = GlobalConfig()
        if isinstance(saved, dict):
            for key, value in saved.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        net = LidarCenterNet(config)
        if getattr(config, "sync_batch_norm", False):
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        state = torch.load(weights, map_location=self.device_name)
        net.load_state_dict(state, strict=True)
        net.to(self.device_name)
        net.eval()
        self.torch = torch
        self.config = config
        print(f"[driver] loaded garage weights {weights} on {self.device_name}")
        return net

    def _find_weights(self) -> Optional[str]:
        if not os.path.isdir(self.checkpoint_dir):
            return None
        for name in sorted(os.listdir(self.checkpoint_dir)):
            if name.startswith("model") and name.endswith(".pth"):
                return os.path.join(self.checkpoint_dir, name)
        return None

    @staticmethod
    def _install_carla_stub() -> None:
        """sensor_agent / some model files import carla. We are not running CARLA."""
        if "carla" in sys.modules:
            return
        import types

        stub = types.ModuleType("carla")
        stub.Client = object  # type: ignore[attr-defined]
        sys.modules["carla"] = stub

    def predict(
        self,
        frame_bgr: np.ndarray,
        speed_mps: float,
        command: str = "LANEFOLLOW",
    ) -> DriverOutput:
        torch = self.torch
        t0 = time.time()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.image_w, self.image_h))
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        rgb_t = rgb_t.to(self.device_name)

        # PUBLIC PRACTICE: zeros instead of a real lidar histogram.
        lidar_channels = int(getattr(self.config, "lidar_seq_len", 1)) * 2
        lidar_res = int(getattr(self.config, "lidar_resolution_width", 256))
        lidar = torch.zeros(
            (1, lidar_channels, lidar_res, lidar_res),
            dtype=torch.float32,
            device=self.device_name,
        )

        cmd_idx = COMMANDS.get(command, COMMANDS["LANEFOLLOW"])
        command_t = torch.tensor([cmd_idx], device=self.device_name)
        velocity = torch.tensor([[speed_mps]], dtype=torch.float32, device=self.device_name)
        # Dummy GPS target 20 m ahead — garage uses this as a route hint.
        target_point = torch.tensor([[20.0, 0.0]], dtype=torch.float32, device=self.device_name)

        with torch.inference_mode():
            kwargs = dict(
                rgb=rgb_t,
                lidar_bev=lidar,
                target_point=target_point,
                ego_vel=velocity,
                command=command_t,
            )
            if getattr(self.config, "two_tp_input", False):
                kwargs["target_point_next"] = target_point
            out = self.model.forward(**kwargs)

        pred_wp = out[0]
        pred_target_speed = out[1] if len(out) > 1 else None
        pred_bev = out[4] if len(out) > 4 else None
        wps = pred_wp[0].detach().cpu().numpy()
        if wps.ndim == 2 and wps.shape[1] == 2:
            yaw = np.zeros((wps.shape[0], 1), dtype=np.float32)
            wps = np.concatenate([wps.astype(np.float32), yaw], axis=1)
        speed = float(speed_mps)
        if pred_target_speed is not None:
            ts = pred_target_speed.detach().cpu().numpy().reshape(-1)
            speed = float(ts[0]) if ts.size else speed
        bev = None
        if pred_bev is not None:
            bev_np = pred_bev.detach().cpu().numpy()
            bev = bev_np[0].argmax(axis=0).astype(np.uint8) if bev_np.ndim == 4 else bev_np[0]
        return DriverOutput(
            waypoints=wps.astype(np.float32),
            target_speed_mps=speed,
            bev_semantic=bev,
            boxes=[],
            inference_ms=(time.time() - t0) * 1000.0,
            backend=self.backend_name,
            extras={"lidar": "zeros", "command": command},
        )
