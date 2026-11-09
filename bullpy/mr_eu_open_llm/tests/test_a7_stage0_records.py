"""The unpinned adapter from ladder geometry to A7 Stage 0 records, on synthetic geometry."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "scripts"))

import a7_stage0_records as adapter  # noqa: E402
from scripts import a7_stage0 as s0  # noqa: E402
from scripts import readout_ladder as rl  # noqa: E402
from scripts import run_stimulus_interventions as run  # noqa: E402
from scripts import stimulus_interventions as si  # noqa: E402

CSV = ROOT / "research" / "reproduction" / "crema_calibration_clips.csv"
HEIGHT, WIDTH = 360, 480
BASE = [0, 10, 20, 30]


def calibration_ids():
    with CSV.open(encoding="utf-8") as handle:
        return [row["clip_id"] for row in csv.DictReader(handle)]


def frame_for(trial_id, index):
    seed = sum(map(ord, trial_id)) * 100 + index
    return np.random.default_rng(seed).integers(0, 256, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)


def landmarks_for(index):
    rng = np.random.default_rng(index)
    points = np.column_stack([rng.uniform(150, 330, 478), rng.uniform(80, 280, 478)])
    return np.round(points, 1).tolist()


def geometry_record(trial_id, apex=20):
    return {
        "trial_id": trial_id, "actor": trial_id.split("_")[0], "label": "",
        "n_frames": 40, "fps": 30.0, "width": WIDTH, "height": HEIGHT, "base_indices": BASE,
        "frame_sha256": {str(i): si.array_sha256(frame_for(trial_id, i)) for i in BASE},
        "landmarks": {str(i): landmarks_for(i) for i in BASE},
        "failed_frames": [], "apex": apex,
        "face_box": [140, 60, 340, 260], "eye_box": [150, 100, 330, 160], "face_width_px": 180.0,
        "aperture_check": {"wide": {"nominal": 1.5, "measured": {"left": 1.4, "right": 1.5}}},
        "path": f"data/crema_d_calibration/{trial_id}.flv",
    }


def geometry(ids=None):
    return {"items": {t: geometry_record(t) for t in (ids or calibration_ids())}}


def test_converts_and_passes_clip_id_check():
    records = adapter.convert(geometry())
    ids = [r["clip_id"] for r in records]
    assert len(records) == 24
    assert s0.check_clip_ids(ids) == calibration_ids()


def test_copies_record_unchanged_and_adds_apex_landmarks():
    source = geometry()
    records = adapter.convert(json.loads(json.dumps(source)))
    for out in records:
        original = source["items"][out["clip_id"]]
        assert {k: v for k, v in out.items() if k not in ("clip_id", "apex_landmarks")} == original
        assert out["clip_id"] == original["trial_id"]
        # Same mapping as A2: _landmark_table keyed by int frame, then .get(apex) in build_media.
        expected = run._landmark_table(original).get(original["apex"])
        assert np.array_equal(np.asarray(out["apex_landmarks"], dtype=np.float64), expected)


def test_face_views_and_region_mask_accept_converted_record(monkeypatch):
    record = adapter.convert(geometry())[0]

    def fake_read_frames(path, indices):
        return {i: frame_for(Path(path).stem, i) for i in indices}, {"height": HEIGHT, "width": WIDTH}

    monkeypatch.setattr(si, "read_frames", fake_read_frames)
    view, shifted = rl.face_views(record)
    assert view.shape[1] == si.FACE_SIZE and view.shape == shifted.shape
    # The region test's landmark use, as a7_stage0.main does it.
    landmarks = np.asarray(record["apex_landmarks"], dtype=np.float64)
    alpha = si.region_mask((HEIGHT, WIDTH), landmarks, "eyesmouth")
    assert alpha.shape == (HEIGHT, WIDTH) and alpha.max() > 0


@pytest.mark.parametrize("change, reason", [
    (lambda r: r.update(clip_failure="video_unreadable"), "failed"),
    (lambda r: r.update(box_failure="fewer_than_half_frames_have_a_face"), "failed"),
    (lambda r: r.update(apex=None), "no apex"),
    (lambda r: r["landmarks"].update({"20": None}), "no usable landmarks"),
    (lambda r: r["landmarks"].pop("20"), "no landmarks entry"),
    (lambda r: r.pop("face_box"), "missing keys"),
    (lambda r: r.pop("frame_sha256"), "missing keys"),
    (lambda r: r.pop("path"), "missing keys"),
    (lambda r: r.update(trial_id="other"), "record trial_id"),
])
def test_refuses_bad_items(change, reason):
    source = geometry()
    change(source["items"][calibration_ids()[3]])
    with pytest.raises(adapter.Refused, match=reason):
        adapter.convert(source)


def test_refuses_empty_or_wrong_shape():
    for bad in ({}, {"items": {}}, [], {"items": []}):
        with pytest.raises(adapter.Refused):
            adapter.convert(bad)


def test_main_writes_list_and_will_not_overwrite(tmp_path):
    src, out = tmp_path / "geometry.json", tmp_path / "out" / "records.json"
    src.write_text(json.dumps(geometry()), encoding="utf-8")
    assert adapter.main(["--geometry", str(src), "--output", str(out)]) == 0
    records = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(records, list) and len(records) == 24
    before = out.read_bytes()
    assert adapter.main(["--geometry", str(src), "--output", str(out)]) == 1
    assert out.read_bytes() == before


def test_main_refuses_without_writing(tmp_path):
    source = geometry()
    source["items"][calibration_ids()[0]]["clip_failure"] = "video_unreadable"
    src, out = tmp_path / "geometry.json", tmp_path / "records.json"
    src.write_text(json.dumps(source), encoding="utf-8")
    assert adapter.main(["--geometry", str(src), "--output", str(out)]) == 1
    assert not out.exists()
