"""Tests for the headless AU renderer. Tests that need assets skip when they are absent."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "scripts"))
import human_au_effects as hae  # noqa: E402
import render_au_faces as raf  # noqa: E402


def have_assets() -> bool:
    try:
        raf.assets_root()
    except FileNotFoundError:
        return False
    return True


needs_assets = pytest.mark.skipif(not have_assets(), reason="set AU_RENDER_ASSETS to run")


# Pure logic: no assets needed.


def test_columns_match_the_effects_script():
    assert raf.AU_COLS == hae.AU_COLS
    assert len(raf.AU_COLS) == 33


def test_every_column_has_a_coverage_row_and_none_is_dropped():
    assert set(raf.COVERAGE) == set(raf.AU_COLS)
    assert {status for status, _ in raf.COVERAGE.values()} <= {"direct", "approximated", "missing"}
    assert raf.COVERAGE["AU22"][0] == "approximated"
    assert raf.COVERAGE["AU27"][0] == "approximated"


def test_every_family_has_an_expected_region():
    assert {c[:4] for c in raf.AU_COLS} == set(raf.EXPECTED_REGION)


def test_config_check_rejects_bad_input():
    with pytest.raises(ValueError, match="unknown AU"):
        raf.check_config({"AU99": 0.5})
    with pytest.raises(ValueError, match="outside 0 to 1"):
        raf.check_config({"AU12L": 1.5})
    with pytest.raises(ValueError, match="outside 0 to 1"):
        raf.check_config({"AU12L": float("nan")})


def test_parse_au():
    assert raf.parse_au("AU12L=1,AU12R=0.5") == {"AU12L": 1.0, "AU12R": 0.5}
    assert raf.parse_au("") == {}


def test_read_target_skips_comments(tmp_path):
    path = tmp_path / "x.target"
    path.write_text("# comment\n3 0.1 -.2 0\n\n7 1 2 3\n")
    ids, offsets = raf.read_target(str(path))
    assert ids.tolist() == [3, 7]
    assert offsets[0].tolist() == pytest.approx([0.1, -0.2, 0.0])


def test_read_mhclo_handles_single_and_triple_rows(tmp_path):
    path = tmp_path / "x.mhclo"
    path.write_text(
        "name x\nx_scale 1 2 0.5\ny_scale 3 4 0.5\nz_scale 5 6 0.5\nmaterial m.mhmat\nverts 0\n"
        "1 2 3 0.5 0.25 0.25 0.1 0.2 0.3\n9\n"
    )
    clothes = raf.read_mhclo(path)
    assert clothes.refs.tolist() == [[1, 2, 3], [9, 0, 1]]
    assert clothes.weights[1].tolist() == [1.0, 0.0, 0.0]
    assert clothes.material == "m.mhmat"
    assert clothes.scale_data[1] == (3, 4, 0.5)


def test_fit_uses_rest_spans_so_opening_the_jaw_does_not_stretch_offsets():
    clothes = raf.Clothes(
        refs=np.array([[0, 1, 2]]),
        weights=np.array([[1.0, 0, 0]], np.float32),
        offsets=np.array([[0, 1.0, 0]], np.float32),
        scale_data=((0, 1, 1.0), (0, 1, 1.0), (0, 1, 1.0)),
        material="",
    )
    rest = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], np.float32)
    deformed = rest.copy()
    deformed[1, 1] = 5
    fitted = raf.fit_clothes(clothes, deformed, rest)
    assert fitted[0].tolist() == pytest.approx([0, 1, 0])


def test_macro_weights_for_a_pure_adult_sum_to_one_per_family():
    identity = raf.Identity("t", 0.0, (("caucasian", 1.0),), ("s",))
    weights = raf.macro_target_weights(identity)
    assert weights["macrodetails/caucasian-female-young"] == 1.0
    assert weights["macrodetails/universal-female-young-averagemuscle-averageweight"] == 1.0
    assert not any("male-" in k and "female" not in k for k in weights)


def test_macro_weights_split_gender():
    identity = raf.Identity("t", 0.25, (("asian", 1.0),), ("s",), muscle=0.75)
    weights = raf.macro_target_weights(identity)
    assert weights["macrodetails/asian-male-young"] == pytest.approx(0.25)
    assert weights["macrodetails/universal-female-young-maxmuscle-averageweight"] == pytest.approx(0.75 * 0.5)


def test_catmull_clark_on_a_cube():
    cube = np.array([[0, 1, 2, 3], [7, 6, 5, 4], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]])
    S, quads = raf.catmull_clark(cube, 8)
    assert S.shape == (8 + 12 + 6, 8)
    assert quads.shape == (24, 4)
    assert np.asarray(S.sum(1)).ravel() == pytest.approx(np.ones(26))


def test_catmull_clark_keeps_a_flat_grid_flat_and_rows_affine():
    quads = np.array([[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]])
    grid = np.array([[x, y, 0] for y in range(3) for x in range(3)], float)
    S, _ = raf.catmull_clark(quads, 9)
    moved = S @ grid
    assert np.abs(moved[:, 2]).max() == 0
    assert np.asarray(S.sum(1)).ravel() == pytest.approx(np.ones(S.shape[0]))
    assert moved[4].tolist() == pytest.approx([1, 1, 0])  # centre is an interior point


def test_subdivide_uvs_order():
    uv = np.array([[[0, 0], [1, 0], [1, 1], [0, 1]]], np.float32)
    out = raf.subdivide_uvs(uv)
    assert out.shape == (4, 4, 2)
    assert out[0].tolist() == [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]


def test_fragments_cover_a_triangle_area():
    points = np.array([[10, 10, 1], [110, 10, 1], [10, 110, 1]], np.float32)
    tri, px, py, weights, area = raf.fragments(points, np.array([[0, 1, 2]]), 200)
    assert len(px) == pytest.approx(5000, abs=120)
    assert weights.sum(1) == pytest.approx(np.ones(len(px)))
    assert area[0] == pytest.approx(10000)


def test_nearer_part_wins_and_alpha_cuts_through():
    tris = np.array([[0, 2, 1], [0, 3, 2]])  # negative image area, so front-facing
    square = np.array([[10, 10, 0], [50, 10, 0], [50, 50, 0], [10, 50, 0]], np.float32)
    uv = np.zeros((2, 3, 2), np.float32)
    normals = np.tile(np.array([0, 0, 1], np.float32), (4, 1))
    frame, z = np.zeros((64, 64, 3), np.float32), np.full((64, 64), np.inf, np.float32)
    white = np.full((2, 2, 4), 255, np.uint8)
    black = np.zeros((2, 2, 4), np.uint8)
    black[..., 3] = 255
    clear = np.zeros((2, 2, 4), np.uint8)

    near = square.copy()
    near[:, 2] = 1
    far = square.copy()
    far[:, 2] = 2
    raf.draw_part(raf.Part("n", tris, uv, white), near, normals, frame, z, 1.0)
    raf.draw_part(raf.Part("f", tris, uv, black), far, normals, frame, z, 1.0)
    assert frame[30, 30].tolist() == pytest.approx([1, 1, 1])
    nearest = square.copy()
    nearest[:, 2] = 0.5
    raf.draw_part(raf.Part("a", tris, uv, clear, alpha_test=True), nearest, normals, frame, z, 1.0)
    assert frame[30, 30].tolist() == pytest.approx([1, 1, 1])


def test_texture_rows_flip_against_obj_v():
    texture = np.zeros((2, 2, 4), np.uint8)
    texture[0] = 255  # top image row
    assert raf.sample_texture(texture, np.array([[0.2, 0.999]], np.float32))[0, 0] > 0.9
    assert raf.sample_texture(texture, np.array([[0.2, 0.001]], np.float32))[0, 0] < 0.1


def test_libreface_hook_is_a_marked_stub(tmp_path):
    with pytest.raises(NotImplementedError, match="LibreFace"):
        raf.libreface_hook(tmp_path, tmp_path / "x.csv")


# With assets.


@needs_assets
def test_output_format_and_black_background(tmp_path):
    out = tmp_path / "face.png"
    image = raf.render_face({"AU12L": 1, "AU12R": 1}, "female_a", 0, out)
    assert image.shape == (768, 768, 3)
    assert image.dtype == np.uint8
    saved = Image.open(out)
    assert saved.size == (768, 768) and saved.mode == "RGB"
    assert np.array_equal(np.asarray(saved), image)
    for corner in (image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]):
        assert corner.tolist() == [0, 0, 0]
    assert 0.15 < (image.sum(2) > 0).mean() < 0.45


@needs_assets
def test_same_inputs_give_identical_files_and_seed_changes_them(tmp_path):
    digests = []
    for name, seed in (("a.png", 3), ("b.png", 3), ("c.png", 4)):
        path = tmp_path / name
        raf.render_face({"AU04": 0.5}, "male_a", seed, path)
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
    assert digests[0] == digests[1]
    assert digests[0] != digests[2]


@needs_assets
def test_identities_cover_sex_and_differ():
    assert {raf.IDENTITIES[n].gender for n in ("female_a", "female_b", "male_a", "male_b")} == {0.0, 1.0}
    assert "neutral" in raf.IDENTITIES
    shapes = {n: raf.build_avatar(n, 0).rest for n in ("neutral", "female_a", "female_b", "male_a", "male_b")}
    names = list(shapes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert np.abs(shapes[a] - shapes[b]).max() > 0.05


@needs_assets
def test_neutral_lips_stay_closed_on_every_identity():
    for name in raf.IDENTITIES:
        assert raf.lip_aperture(raf.build_avatar(name, 0).rest) <= 1e-3


@needs_assets
def test_stray_pelvis_vertices_are_gone():
    y = raf.base_mesh().vertices[:, 1]
    for column in raf.AU_COLS:
        moved = np.linalg.norm(raf.au_offsets({column: 1.0}), axis=1)
        assert moved[y < raf.HEAD_MIN_Y].max(initial=0) == 0, column
        assert moved.max() > 0, column


@needs_assets
@pytest.mark.parametrize("family", sorted(raf.LATERAL_TARGETS))
def test_laterality_is_the_avatars_side(family):
    x = raf.base_mesh().vertices[:, 0]
    left = np.linalg.norm(raf.au_offsets({family + "L": 1.0}), axis=1)
    right = np.linalg.norm(raf.au_offsets({family + "R": 1.0}), axis=1)
    assert (left * x).sum() > 0 > (right * x).sum()


@needs_assets
def test_bilateral_split_is_continuous():
    a = raf.au_offsets({"AU12L": 0.5, "AU12R": 0.5})
    b = raf.au_offsets({"AU12L": 0.5, "AU12R": 0.499})
    assert np.abs(a - b).max() < 0.01


@needs_assets
def test_au26_and_au27_share_the_jaw():
    jaw_27 = raf.au_offsets({"AU27": 1.0})[raf.CHIN_VERTEX, 1]
    both = raf.au_offsets({"AU26": 1.0, "AU27": 1.0})[raf.CHIN_VERTEX, 1]
    assert both == pytest.approx(jaw_27)
    jaw_26 = raf.au_offsets({"AU26": 1.0})[raf.CHIN_VERTEX, 1]
    assert jaw_27 < jaw_26 < 0


@needs_assets
def test_au27_differs_from_au26_in_shape_not_only_size():
    avatar = raf.build_avatar("neutral", 0)
    m26 = raf.deformed_mesh({"AU26": 1.0}, avatar)
    m27 = raf.deformed_mesh({"AU27": 1.0}, avatar)
    assert raf.lip_aperture(m27) > 1.3 * raf.lip_aperture(m26)
    assert raf.mouth_width(m27) < 0.9 * raf.mouth_width(m26)


@needs_assets
def test_au22_narrows_and_pushes_the_lips_forward():
    avatar = raf.build_avatar("neutral", 0)
    moved = raf.deformed_mesh({"AU22": 1.0}, avatar)
    assert raf.mouth_width(moved) < 0.85 * raf.mouth_width(avatar.rest)
    assert moved[raf.LOWER_LIP_INNER, 2] > avatar.rest[raf.LOWER_LIP_INNER, 2]


@needs_assets
@pytest.mark.parametrize("column", ["AU12L", "AU22", "AU26", "AU27", "AU01", "AU43"])
def test_single_au_validity_on_one_identity(column):
    neutral = raf.render_face({}, "neutral", 0)
    masks = raf.region_masks(raf.build_avatar("neutral", 0))
    row, _ = raf.validate_au(column, "neutral", neutral, masks)
    assert row["passed"], row


@needs_assets
def test_coverage_targets_exist_on_disk():
    for column in raf.AU_COLS:
        assert raf.au_offsets({column: 1.0}).any()
