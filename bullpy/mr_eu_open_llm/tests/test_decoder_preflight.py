"""The decoder preflight must reject junk frames."""

import importlib.util
from pathlib import Path

import numpy as np

path = Path(__file__).resolve().parents[1] / "research/scripts/decoder_preflight.py"
spec = importlib.util.spec_from_file_location("decoder_preflight", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def frames(*levels):
    return [np.full((4, 4, 3), level, dtype=np.uint8) for level in levels]


def test_good_reads_pass():
    assert mod.check_reads([frames(90, 110), frames(90, 110)]) == []


def test_reads_that_differ_fail():
    assert "frames differ between reads" in mod.check_reads([frames(90, 110), frames(90, 111)])


def test_blank_frames_fail():
    assert "blank frame" in mod.check_reads([frames(0, 0), frames(0, 0)])


def test_all_frames_identical_fail():
    assert "all sampled frames identical" in mod.check_reads([frames(117, 117), frames(117, 117)])
