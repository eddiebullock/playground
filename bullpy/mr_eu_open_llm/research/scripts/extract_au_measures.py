#!/usr/bin/env python3
"""Measure facial action units on the 243 EU-Emotion clips with two detectors (RQ1).

Detectors:
  libreface   MediaPipe FaceMesh alignment, then ResNet-18 intensity (12 AUs, trained
              on DISFA) and RepVGG occurrence (12 AUs, trained on BP4D). USC research
              licence. Also head pose, from the landmarks as upstream computes it.
  openface3   RetinaFace crop, then the EfficientNet multitask model (8 AUs, gaze).
              CMU academic noncommercial licence. Landmarks are not needed for AUs.

The clips are measured and nothing else. Frames stay in memory. No frame, crop or
aligned face is written anywhere, except in --verify-upstream, which writes three
frames to node-local $TMPDIR for the upstream path-based code and deletes them.

Steps:
  extract    one shard of clips; per-frame numbers go to --frames-dir (cluster only)
  aggregate  per-clip, per-cue aggregates, for all frames and for the frames the
             models saw (eval JSON frame_indices), to --out-csv
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import itertools
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

LF_INTENSITY_AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]  # upstream order
LF_OCCURRENCE_AUS = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]  # upstream order
OF3_AUS = [1, 2, 4, 6, 9, 12, 25, 26]  # order of au_labels in upstream demo2.py
ACTOR_RE = re.compile(r"([A-Za-z])F\s*\d+", re.IGNORECASE)
MIN_FACE_RATE = 0.5

# MediaPipe FaceMesh edges as in libreface.detect_mediapipe_image.get_aligned_image.
LIPS = [(61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405),
        (405, 321), (321, 375), (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
        (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291), (78, 95),
        (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402), (402, 318),
        (318, 324), (324, 308), (78, 191), (191, 80), (80, 81), (81, 82), (82, 13),
        (13, 312), (312, 311), (311, 310), (310, 415), (415, 308)]
LEFT_EYE = [(263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
            (381, 382), (382, 362), (263, 466), (466, 388), (388, 387), (387, 386),
            (386, 385), (385, 384), (384, 398), (398, 362)]
RIGHT_EYE = [(33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
             (154, 155), (155, 133), (33, 246), (246, 161), (161, 160), (160, 159),
             (159, 158), (158, 157), (157, 173), (173, 133)]
POSE_POINTS = {33, 263, 1, 61, 291, 199}


def actor_of(trial_id: str) -> str:
    match = ACTOR_RE.search(trial_id)
    return match.group(1).upper() if match else ""


def unique_points(edges):
    return list(dict.fromkeys(itertools.chain.from_iterable(edges)))


def au_name(number: int) -> str:
    return f"AU{number:02d}"


# --------------------------------------------------------------------------
# Detectors. Heavy imports stay inside so `aggregate` needs only numpy and pandas.
# --------------------------------------------------------------------------


@contextlib.contextmanager
def retinaface_paths(weights: Path):
    """RetinaFace reads ./weights/mobilenetV1X0.25_pretrain.tar relative to the cwd."""
    link = weights / "weights"
    if not link.exists():
        link.symlink_to(".")
    previous = os.getcwd()
    os.chdir(weights)
    try:
        yield
    finally:
        os.chdir(previous)


class LibreFace:
    """Upstream LibreFace models with alignment done in memory."""

    INTENSITY_OPTS = {
        "seed": 0, "data_root": "", "data": "DISFA", "fold": "all", "image_size": 256,
        "crop_size": 224, "num_labels": 12, "sigma": 10.0, "model_name": "resnet",
        "dropout": 0.1, "hidden_dim": 128, "half_precision": False, "num_epochs": 30,
        "interval": 500, "threshold": 0, "batch_size": 64, "learning_rate": "3e-5",
        "weight_decay": "1e-4", "loss": "unweighted", "clip": 1.0, "when": 10,
        "patience": 5, "fm_distillation": False, "num_workers": 0,
        "weights_download_id": "14qEnWRew2snhdMdOVyqKFJ5rq5VZrfAX",
    }
    OCCURRENCE_OPTS = {
        "seed": 0, "data_root": "", "data": "BP4D", "fold": "all", "image_size": 256,
        "crop_size": 224, "num_labels": 12, "sigma": 10.0, "model_name": "repvgg",
        "dropout": 0.1, "hidden_dim": 128, "half_precision": False, "num_epochs": 30,
        "interval": 500, "threshold": 0, "batch_size": 64, "learning_rate": "3e-5",
        "weight_decay": "1e-4", "loss": "unweighted", "clip": 1.0, "when": 10,
        "patience": 5, "fm_distillation": False, "num_workers": 0,
        "weights_download_id": "1vInW2UbjrD13uO3KiIoqBOVZ6gM_aWKC",
    }

    def __init__(self, weights: Path, device: str):
        import mediapipe as mp
        import torch
        from PIL import Image
        from torchvision import transforms

        # Upstream calls Image.ANTIALIAS, removed in Pillow 10. It was LANCZOS.
        if not hasattr(Image, "ANTIALIAS"):
            Image.ANTIALIAS = Image.Resampling.LANCZOS
        from libreface.AU_Detection.solver_inference_image import (
            solver_in_domain_image as OccurrenceSolver,
        )
        from libreface.AU_Recognition.solver_inference_image import (
            solver_inference_image as IntensitySolver,
        )
        from libreface.detect_mediapipe_image import image_align

        self.torch, self.Image, self.image_align = torch, Image, image_align
        self.device = device

        class Opts:
            def __init__(self, values):
                self.__dict__.update(values)

        self.intensity = IntensitySolver(Opts({
            **self.INTENSITY_OPTS, "device": device,
            "ckpt_path": str(weights / "AU_Recognition/weights/resnet.pt"),
        })).to(device).eval()
        self.occurrence = OccurrenceSolver(Opts({
            **self.OCCURRENCE_OPTS, "device": device,
            "ckpt_path": str(weights / "AU_Detection/weights/repvgg.pt"),
        })).to(device).eval()
        # Upstream loads checkpoints in run(), not in the constructor.
        self.intensity.load_best_ckpt()
        self.occurrence.load_best_ckpt()
        self.intensity.eval()
        self.occurrence.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, refine_landmarks=True, max_num_faces=1,
            min_detection_confidence=0.5,
        )
        self.indices = unique_points(LEFT_EYE) + unique_points(RIGHT_EYE) + unique_points(LIPS)

    def align(self, rgb: np.ndarray):
        """Return (aligned PIL image, pitch, yaw, roll), or None without a face."""
        import cv2

        result = self.mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        points = result.multi_face_landmarks[0].landmark
        landmark = np.array([[points[i].x, points[i].y] for i in self.indices])
        h, w = rgb.shape[:2]
        face_2d, face_3d = [], []
        for idx, lm in enumerate(points):
            if idx in POSE_POINTS:
                x, y = int(lm.x * w), int(lm.y * h)
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])
        camera = np.array([[w, 0, h / 2], [0, w, w / 2], [0, 0, 1]], dtype=float)
        _, rotation, _ = cv2.solvePnP(
            np.array(face_3d, dtype=np.float64), np.array(face_2d, dtype=np.float64),
            camera, np.zeros((4, 1), dtype=np.float64),
        )
        angles = cv2.RQDecomp3x3(cv2.Rodrigues(rotation)[0])[0]
        aligned = self.image_align(self.Image.fromarray(rgb), landmark)
        return aligned, angles[0] * 360, angles[1] * 360, angles[2] * 360

    def predict(self, images):
        batch = self.torch.stack([self.transform(im) for im in images]).to(self.device)
        with self.torch.no_grad():
            intensity = self.intensity.model(batch)
            intensity = self.torch.clamp(intensity * 5.0, 0.0, 5.0).cpu().numpy()
            occurrence = self.occurrence.model(batch).cpu().numpy()
        return intensity, occurrence


class OpenFace3:
    """Upstream RetinaFace and multitask model, on in-memory frames."""

    def __init__(self, weights: Path, device: str):
        import torch
        from torchvision import transforms
        from openface.face_detection import FaceDetector
        from openface.model.MTL import MTL
        from openface.Pytorch_Retinaface.layers.functions.prior_box import PriorBox
        from openface.Pytorch_Retinaface.utils.box_utils import decode
        from openface.Pytorch_Retinaface.utils.nms.py_cpu_nms import py_cpu_nms

        self.torch, self.decode, self.nms, self.PriorBox = torch, decode, py_cpu_nms, PriorBox
        self.device = device
        with retinaface_paths(weights):
            self.detector = FaceDetector(str(weights / "Alignment_RetinaFace.pth"), device=device)
        self.model = MTL().to(device)
        self.model.load_state_dict(
            torch.load(weights / "MTL_backbone.pth", map_location=device, weights_only=True)
        )
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.priors = {}

    def _priors(self, shape):
        if shape not in self.priors:
            box = self.PriorBox(self.detector.cfg, image_size=shape)
            self.priors[shape] = box.forward().to(self.device)
        return self.priors[shape]

    def crop(self, bgr: np.ndarray):
        """Return (BGR crop, confidence) as FaceDetector.get_face does, or None."""
        torch = self.torch
        image = np.float32(bgr) - np.float32((104, 117, 123))
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            loc, conf, _ = self.detector.model(tensor)
        h, w = bgr.shape[:2]
        cfg = self.detector.cfg
        boxes = self.decode(loc.data.squeeze(0), self._priors((h, w)).data, cfg["variance"])
        boxes = (boxes * torch.Tensor([w, h, w, h]).to(self.device)).cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        keep = np.where(scores > self.detector.confidence_threshold)[0]
        if len(keep) == 0:
            return None
        dets = np.hstack((boxes[keep], scores[keep, None])).astype(np.float32)
        det = dets[self.nms(dets, self.detector.nms_threshold)][0]
        if det[4] < self.detector.vis_threshold:
            return None
        x0, y0, x1, y1 = det[:4].astype(int)
        crop = bgr[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)]
        return (crop, float(det[4])) if crop.size else None

    def predict(self, crops):
        import cv2

        tensors = [self.transform(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
        with self.torch.no_grad():
            _, gaze, au = self.model(self.torch.stack(tensors).to(self.device))
        return au.cpu().numpy(), gaze.cpu().numpy()


def frame_columns():
    cols = ["lf_face"]
    cols += [f"lf_{au_name(a)}" for a in LF_INTENSITY_AUS]
    cols += [f"lfocc_{au_name(a)}" for a in LF_OCCURRENCE_AUS]
    cols += ["lf_head_pitch", "lf_head_yaw", "lf_head_roll"]
    cols += ["of_face", "of_conf"]
    cols += [f"of_{au_name(a)}" for a in OF3_AUS]
    cols += ["of_gaze_yaw", "of_gaze_pitch"]
    return cols


def measure_chunk(libreface, openface, frames, start_index):
    """Per-frame numbers for a list of BGR frames. Nothing is retained."""
    import cv2

    table = pd.DataFrame(np.nan, index=range(len(frames)), columns=frame_columns())
    table.insert(0, "frame_index", np.arange(start_index, start_index + len(frames)))
    table["lf_face"], table["of_face"] = 0.0, 0.0

    aligned, positions = [], []
    for row, frame in enumerate(frames):
        result = libreface.align(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if result is not None:
            aligned.append(result[0])
            positions.append(row)
            table.loc[row, ["lf_head_pitch", "lf_head_yaw", "lf_head_roll"]] = result[1:]
    if aligned:
        intensity, occurrence = libreface.predict(aligned)
        table.loc[positions, "lf_face"] = 1.0
        table.loc[positions, [f"lf_{au_name(a)}" for a in LF_INTENSITY_AUS]] = intensity
        table.loc[positions, [f"lfocc_{au_name(a)}" for a in LF_OCCURRENCE_AUS]] = occurrence

    crops, positions = [], []
    for row, frame in enumerate(frames):
        result = openface.crop(frame)
        if result is not None:
            crops.append(result[0])
            positions.append(row)
            table.loc[row, "of_conf"] = result[1]
    if crops:
        au, gaze = openface.predict(crops)
        table.loc[positions, "of_face"] = 1.0
        table.loc[positions, [f"of_{au_name(a)}" for a in OF3_AUS]] = au
        table.loc[positions, ["of_gaze_yaw", "of_gaze_pitch"]] = gaze
    return table


def load_trials(manifest: Path):
    trials = json.loads(manifest.read_text(encoding="utf-8"))["trials"]
    return sorted(trials, key=lambda t: t["trial_id"])


def done_clips(path: Path):
    """Clips whose rows are complete. A clip is appended only after it finishes."""
    if not path.exists():
        return set()
    return set(pd.read_csv(path, usecols=["trial_id"])["trial_id"].unique())


def extract(args):
    import cv2
    import torch

    torch.set_num_threads(max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))))
    trials = load_trials(args.manifest)[args.shard :: args.n_shards]
    if args.max_clips:
        trials = trials[: args.max_clips]
    args.frames_dir.mkdir(parents=True, exist_ok=True)
    out = args.frames_dir / f"frames_shard{args.shard:02d}of{args.n_shards:02d}.csv"
    finished = done_clips(out)
    libreface = LibreFace(args.libreface_weights, args.device)
    openface = OpenFace3(args.openface_weights, args.device)
    print(f"device={args.device} clips={len(trials)} already_done={len(finished)}", flush=True)
    for position, trial in enumerate(trials):
        trial_id = trial["trial_id"]
        if trial_id in finished:
            continue
        started = time.time()
        capture = cv2.VideoCapture(str(args.data_root / trial["stimulus_path"]))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {trial_id}")
        declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        parts, chunk, first, seen = [], [], 0, 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            chunk.append(frame)
            seen += 1
            if len(chunk) == args.batch_size:
                parts.append(measure_chunk(libreface, openface, chunk, first))
                first, chunk = seen, []
        capture.release()
        if chunk:
            parts.append(measure_chunk(libreface, openface, chunk, first))
        if not parts:
            raise RuntimeError(f"No frames read from {trial_id}")
        table = pd.concat(parts, ignore_index=True)
        table.insert(0, "trial_id", trial_id)
        table.insert(2, "declared_frames", declared)
        table.round(5).to_csv(out, mode="a", index=False, header=not out.exists())
        elapsed = time.time() - started
        print(
            f"[{position + 1}/{len(trials)}] {seen} frames, {elapsed:.1f}s "
            f"({1000 * elapsed / seen:.0f} ms/frame) "
            f"lf_face={table.lf_face.mean():.3f} of_face={table.of_face.mean():.3f}",
            flush=True,
        )
    print("Shard done.", flush=True)


def verify_upstream(args):
    """Compare the in-memory paths with upstream's path-based code on three frames.

    Also confirms that sequential decoding gives the frame that evaluate.py gets by
    seeking. Temporary images live in node-local $TMPDIR and are removed.
    """
    import cv2
    import torch
    from libreface import get_au_intensities
    from libreface.detect_mediapipe_image import get_aligned_image
    from openface.face_detection import FaceDetector
    from openface.multitask_model import MultitaskPredictor

    libreface = LibreFace(args.libreface_weights, args.device)
    openface = OpenFace3(args.openface_weights, args.device)
    with retinaface_paths(args.openface_weights):
        upstream_faces = FaceDetector(
            str(args.openface_weights / "Alignment_RetinaFace.pth"), device=args.device
        )
    upstream_mtl = MultitaskPredictor(
        str(args.openface_weights / "MTL_backbone.pth"), device=args.device
    )
    trials = load_trials(args.manifest)[:: max(1, 243 // 3)][:3]
    root = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    report = []
    try:
        for trial in trials:
            path = str(args.data_root / trial["stimulus_path"])
            capture = cv2.VideoCapture(path)
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            target = count // 2
            sequential = None
            for index in range(target + 1):
                ok, sequential = capture.read()
                if not ok:
                    raise RuntimeError(f"Decode stopped at {index} in {trial['trial_id']}")
            capture.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, sought = capture.read()
            capture.release()
            same = bool(ok and np.array_equal(sequential, sought))

            image = os.path.join(root, "probe.png")
            cv2.imwrite(image, sequential)
            aligned_path, pose, _ = get_aligned_image(image, temp_dir=root)
            reference = get_au_intensities(
                aligned_path, device=args.device,
                weights_download_dir=str(args.libreface_weights),
            )
            reference = np.array([reference[f"au_{a}_intensity"] for a in LF_INTENSITY_AUS])
            rgb = cv2.cvtColor(sequential, cv2.COLOR_BGR2RGB)
            aligned, pitch, yaw, roll = libreface.align(rgb)
            intensity, _ = libreface.predict([aligned])

            crop_ref, _ = upstream_faces.get_face(image)
            _, gaze_ref, au_ref = upstream_mtl.predict(crop_ref)
            crop, _ = openface.crop(sequential)
            au, gaze = openface.predict([crop])
            report.append({
                "sequential_equals_seek": same,
                "lf_max_abs_diff": float(np.abs(intensity[0] - reference).max()),
                "lf_pose_max_abs_diff": float(
                    np.abs(np.array([pitch, yaw, roll])
                           - np.array([pose["pitch"], pose["yaw"], pose["roll"]])).max()
                ),
                "lf_intensity_range": [float(reference.min()), float(reference.max())],
                "of_max_abs_diff": float(np.abs(au[0] - au_ref.cpu().numpy()[0]).max()),
                "of_gaze_max_abs_diff": float(np.abs(gaze[0] - gaze_ref.cpu().numpy()[0]).max()),
                "of_au_range": [float(au.min()), float(au.max())],
            })
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(json.dumps(report, indent=1), flush=True)
    del torch


# --------------------------------------------------------------------------
# Aggregates
# --------------------------------------------------------------------------


def model_frame_indices(pattern: str):
    """frame_indices per trial from eval JSONs. Responses in them are never read.

    Every eval must agree, so one 'model_1fps' frame set describes all of them.
    """
    # A mismatch run stores the indices of the swapped-in video, so it cannot be used.
    paths = sorted(p for p in glob.glob(pattern) if not re.search(r"mismatch|smoke", p))
    if not paths:
        raise FileNotFoundError(f"No eval JSON matches {pattern}")
    merged = {}
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for trial in payload.get("trials") or payload["results"]:
            if trial.get("error"):
                continue
            indices = list(trial["frame_indices"])
            if merged.setdefault(trial["trial_id"], indices) != indices:
                raise ValueError(f"frame_indices disagree for {trial['trial_id']} in {path}")
    return merged, paths


def long_measures(frames: pd.DataFrame, selected: dict):
    specs = []
    for column in frames.columns:
        for prefix, detector, kind in (
            ("lfocc_", "libreface", "occurrence"),
            ("lf_", "libreface", "intensity"),
            ("of_", "openface3", "intensity"),
        ):
            if column.startswith(prefix) and "_AU" in f"_{column[len(prefix):]}":
                specs.append((column, column[len(prefix):], detector, kind, prefix[:2] + "_face"))
                break
    specs += [
        ("lf_head_pitch", "head_pitch", "libreface", "pose", "lf_face"),
        ("lf_head_yaw", "head_yaw", "libreface", "pose", "lf_face"),
        ("lf_head_roll", "head_roll", "libreface", "pose", "lf_face"),
        ("of_gaze_yaw", "gaze_yaw", "openface3", "gaze", "of_face"),
        ("of_gaze_pitch", "gaze_pitch", "openface3", "gaze", "of_face"),
    ]
    rows = []
    for trial_id, clip in frames.groupby("trial_id", sort=True):
        chosen = selected[trial_id]
        for frame_set, part in (("all", clip), ("model_1fps", clip[clip.frame_index.isin(chosen)])):
            if frame_set == "model_1fps" and len(part) != len(set(chosen)):
                raise ValueError(f"{trial_id}: {len(part)} of {len(set(chosen))} model frames present")
            for column, cue, detector, kind, face in specs:
                values = part[column].dropna().to_numpy()
                n_face = int(part[face].sum())
                rows.append({
                    "trial_id": trial_id, "actor": actor_of(trial_id), "detector": detector,
                    "kind": kind, "cue": cue, "frame_set": frame_set, "n_frames": len(part),
                    "n_face": n_face, "face_rate": round(n_face / len(part), 4),
                    "mean": round(float(values.mean()), 5) if len(values) else np.nan,
                    "sd": round(float(values.std()), 5) if len(values) else np.nan,
                    "p90": round(float(np.quantile(values, 0.9)), 5) if len(values) else np.nan,
                    "max": round(float(values.max()), 5) if len(values) else np.nan,
                })
    return pd.DataFrame(rows)


def frame_agreement(frames: pd.DataFrame):
    """Between-detector correlation per AU across frames, raw and within clip."""
    out = {}
    both = frames[(frames.lf_face == 1) & (frames.of_face == 1)]
    for au in OF3_AUS:
        a, b = f"lf_{au_name(au)}", f"of_{au_name(au)}"
        within_a = both[a] - both.groupby("trial_id")[a].transform("mean")
        within_b = both[b] - both.groupby("trial_id")[b].transform("mean")
        out[au_name(au)] = {
            "frames": int(len(both)),
            "pearson_all_frames": round(float(both[a].corr(both[b])), 4),
            "pearson_within_clip": round(float(within_a.corr(within_b)), 4),
        }
    return out


def aggregate(args):
    paths = sorted(args.frames_dir.glob("frames_shard*.csv"))
    if not paths:
        raise FileNotFoundError(f"No shard files in {args.frames_dir}")
    frames = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    frames = frames.drop_duplicates(["trial_id", "frame_index"])
    wanted = [t["trial_id"] for t in load_trials(args.manifest)]
    present = set(frames.trial_id)
    missing = [t for t in wanted if t not in present]
    if missing and not args.allow_partial:
        raise ValueError(f"{len(missing)} clips missing, for example {missing[:3]}")
    short = frames.groupby("trial_id").agg(n=("frame_index", "size"), d=("declared_frames", "first"))
    short = short[short.n != short.d]
    selected, sources = model_frame_indices(args.evals)
    frames = frames[frames.trial_id.isin(selected)]
    table = long_measures(frames, selected)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False)
    face_rates = table[table.frame_set == "all"].drop_duplicates(["trial_id", "detector"])
    summary = {
        "n_clips": int(table.trial_id.nunique()),
        "n_frames_all": int(len(frames)),
        "n_frames_model": int(sum(len(set(selected[t])) for t in frames.trial_id.unique())),
        "frame_index_sources": sources,
        "clips_with_decoded_count_differing_from_declared": int(len(short)),
        "face_rate_by_detector": {
            k: {"mean": round(float(v.mean()), 4), "min": round(float(v.min()), 4),
                "clips_below_min_rate": int((v < MIN_FACE_RATE).sum())}
            for k, v in face_rates.groupby("detector")["face_rate"]
        },
        "frame_level_agreement": frame_agreement(frames),
    }
    print(json.dumps(summary, indent=1))
    if args.out_summary:
        args.out_summary.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=["extract", "aggregate", "verify"])
    parser.add_argument("--manifest", type=Path, default=Path("data/eu_emotions_full_manifest.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data/eu_emotions"))
    parser.add_argument("--frames-dir", type=Path, default=Path("results/au/frames"))
    parser.add_argument("--libreface-weights", type=Path)
    parser.add_argument("--openface-weights", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument(
        "--evals",
        default="results/baseline/eu_emotions/*/eval_v2_eu_emotions_*_video_only_seed42*.json",
        help="eval JSONs supplying frame_indices; responses are not read",
    )
    parser.add_argument("--out-csv", type=Path, default=Path("research/reproduction/eu_au_measures.csv"))
    parser.add_argument("--out-summary", type=Path)
    parser.add_argument("--allow-partial", action="store_true", help="for smoke runs only")
    args = parser.parse_args(argv)
    {"extract": extract, "aggregate": aggregate, "verify": verify_upstream}[args.step](args)


if __name__ == "__main__":
    sys.exit(main())
