import json

import cv2
import numpy as np
import pytest

from solarian.core.optical_remap import shipped_checkerboard_path

from solarian_umi.slam.camera_calibration import (
    BoardView,
    StaticMask,
    board_object_points,
    calibrate_views,
    detect_board,
    find_calibration,
    is_near_duplicate,
    stand_in_from_remap,
    uncovered_cells,
)
from solarian_umi.slam.fisheye_model import Fisheye, build_remap

LUMI = Fisheye(
    fx=349.1, fy=349.3, cx=772.6, cy=595.7,
    k=(-0.0088, -0.0016, -0.00084, -0.000049), width=1440, height=1080,
)


def _synthetic_views(model: Fisheye, n: int, seed: int = 0) -> list[BoardView]:
    rng = np.random.default_rng(seed)
    corners = board_object_points()
    center = corners.mean(axis=0)
    views = []
    while len(views) < n:
        rvec = rng.normal(0, 0.5, 3)
        aim = rng.uniform(-1.4, 1.4, 2)
        distance = rng.uniform(0.15, 0.4)
        R, _ = cv2.Rodrigues(rvec)
        tvec = np.array([np.sin(aim[0]), np.sin(aim[1]), 1.0])
        tvec = tvec / np.linalg.norm(tvec) * distance - R @ center
        cam = corners @ R.T + tvec
        if (cam[:, 2] <= 0.02).any():
            continue
        uv, valid = model.project(cam / np.linalg.norm(cam, axis=1, keepdims=True))
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < model.width) & (uv[:, 1] >= 0) & (uv[:, 1] < model.height)
        if not (valid & inside).all():
            continue
        noise = rng.normal(0, 0.1, uv.shape)
        views.append(BoardView(image_points=uv + noise, object_points=corners))
    return views


def test_calibrate_views_recovers_a_wide_fisheye_from_whole_board_views():
    views = _synthetic_views(LUMI, 60)
    model, rms, used = calibrate_views(views, (LUMI.width, LUMI.height))

    assert rms < 0.2
    assert used >= 40
    assert model.fx == pytest.approx(LUMI.fx, rel=0.01)
    assert model.cx == pytest.approx(LUMI.cx, abs=2.0)
    assert model.cy == pytest.approx(LUMI.cy, abs=2.0)
    probe = np.array([[0.6, 0.0, 0.8], [0.0, -0.9, 0.44], [0.95, 0.3, 0.1]])
    uv_true, _ = LUMI.project(probe)
    uv_fit, _ = model.project(probe)
    assert np.abs(uv_fit - uv_true).max() < 1.5


def test_uncovered_cells_skips_the_black_corners_of_the_image_circle():
    lit = build_remap(LUMI, LUMI).valid
    centered = BoardView(
        image_points=np.array([[700.0, 500.0], [740.0, 560.0]]),
        object_points=np.zeros((2, 3)),
    )
    none = np.zeros_like(lit)
    assert len(uncovered_cells([centered], np.ones_like(lit), none)) == 12
    assert 0 < len(uncovered_cells([centered], lit, none)) < 12
    full = _synthetic_views(LUMI, 200, seed=1)
    assert uncovered_cells(full, lit, none) == []


def test_a_static_gripper_at_the_rim_excuses_the_cells_it_covers():
    lit = build_remap(LUMI, LUMI).valid
    rng = np.random.default_rng(0)
    motion = StaticMask()
    for _ in range(20):
        frame = rng.integers(0, 255, (1080, 1440), dtype=np.uint8)
        frame = cv2.GaussianBlur(frame, (0, 0), 3)
        frame[810:, 300:1140] = np.tile(np.arange(840) % 64 * 4, (270, 1)).astype(np.uint8)
        frame[~lit] = 0
        motion.add(frame)
    static = motion.mask(lit)

    assert static[950, 720] and not static[540, 720]
    centered = BoardView(image_points=np.array([[700.0, 500.0]]), object_points=np.zeros((1, 3)))
    missing = uncovered_cells([centered], lit, static)
    assert (1, 3) not in missing and (2, 3) not in missing
    assert (1, 0) in missing


def test_detected_corners_follow_the_object_point_order():
    board = cv2.imread(str(shipped_checkerboard_path()), cv2.IMREAD_GRAYSCALE)
    h, w = board.shape
    target = np.float32([[200, 150], [1150, 220], [1100, 900], [260, 820]])
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), target)
    view = detect_board(cv2.warpPerspective(board, H, (1440, 1080), borderValue=255))

    assert view is not None and len(view.image_points) == 54
    # Object points to image by a homography means the order is a planar grid,
    # up to the 180 degree flip findChessboardCorners may pick.
    G, _ = cv2.findHomography(view.object_points[:, :2], view.image_points)
    projected = cv2.perspectiveTransform(view.object_points[None, :, :2], G)[0]
    assert np.abs(projected - view.image_points).max() < 0.5


def test_near_duplicate_views_are_recognised():
    view = _synthetic_views(LUMI, 1)[0]
    moved = BoardView(image_points=view.image_points + [200.0, 0.0], object_points=view.object_points)
    assert is_near_duplicate(view, [view], 1440)
    assert not is_near_duplicate(moved, [view], 1440)


def test_stand_in_scales_checkerboard_intrinsics_to_the_recorded_mode(tmp_path):
    remap = tmp_path / "remap.json"
    remap.write_text(json.dumps({"camera_intrinsics": LUMI.resized(2028, 1520).to_umi_json(rms=0.3, n_views=20, fps=60)}))
    doc = stand_in_from_remap(remap, serial="LUMI1")

    model = Fisheye.from_umi_json(doc["intrinsics"])
    assert (model.width, model.height) == (1440, 1080)
    assert model.fx == pytest.approx(LUMI.fx)
    assert model.cy == pytest.approx(LUMI.cy)
    assert doc["camera_serial"] == "LUMI1"


def test_find_calibration_prefers_the_session_then_conf_and_checks_serial(tmp_path):
    session, conf = tmp_path / "session", tmp_path / "conf"
    (conf / "calibrations" / "cameras").mkdir(parents=True)
    session.mkdir()
    (conf / "calibrations" / "cameras" / "LUMI1.json").write_text(json.dumps({"camera_serial": "LUMI1"}))
    assert find_calibration(session, "LUMI1", [conf]) == conf / "calibrations" / "cameras" / "LUMI1.json"

    (session / "camera_calibration.json").write_text(json.dumps({"camera_serial": "LUMI2"}))
    with pytest.raises(SystemExit, match="LUMI2"):
        find_calibration(session, "LUMI1", [conf])
    with pytest.raises(SystemExit, match="No calibration"):
        find_calibration(tmp_path / "other", "LUMI9", [conf])


def test_remap_to_the_same_camera_is_identity_inside_the_lens():
    remap = build_remap(LUMI, LUMI)
    ys, xs = np.nonzero(remap.valid)
    assert remap.coverage > 0.7
    assert np.abs(remap.map_x[ys, xs] - xs).max() < 1e-2
    assert np.abs(remap.map_y[ys, xs] - ys).max() < 1e-2
