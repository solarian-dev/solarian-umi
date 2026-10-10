"""Contact sheet for Solarian vs UMI gold policy images."""

from pathlib import Path

import numpy as np
import pytest

from solarian_umi_skill.skill import UMISkill
from solarian_umi_skill.umi.common.cv2_util import letterbox_to_square
from solarian_umi_skill.umi.compare_policy_image import (
    describe_pair,
    format_summary,
    write_contact_sheet,
)


def _rgb(h, w, color):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def test_describe_pair_reports_size_match():
    gold = _rgb(224, 224, (10, 10, 10))
    gold[180:220, 20:60] = (255, 80, 0)
    gold[180:220, 164:204] = (255, 80, 0)
    sol = gold.copy()
    lines = describe_pair(gold, sol)
    assert "224x224" in lines[0]
    assert "match" in lines[0]


def test_describe_pair_reports_size_mismatch():
    lines = describe_pair(_rgb(224, 224, 0), _rgb(270, 480, 0))
    assert "MISMATCH" in lines[0]
    assert "480x270" in lines[0]


def test_write_contact_sheet(tmp_path):
    gold = _rgb(224, 224, (8, 8, 8))
    gold[40:180, 40:180] = (40, 80, 120)
    gold[190:220, 30:70] = (240, 90, 20)
    sol = gold.copy()
    sol[190:220, 30:70] = (240, 90, 20)
    sol[10:30, 10:30] = (255, 255, 255)

    gold_path = tmp_path / "gold.png"
    sol_path = tmp_path / "solarian_camera0_rgb_t0.png"
    import cv2

    cv2.imwrite(str(gold_path), cv2.cvtColor(gold, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(sol_path), cv2.cvtColor(sol, cv2.COLOR_RGB2BGR))

    dump = tmp_path / "dump"
    dump.mkdir()
    stats = write_contact_sheet(dump, gold_path=gold_path, solarian_path=sol_path)
    assert Path(stats["compare_path"]).is_file()
    assert (dump / "umi_gold_camera0_rgb_t0.png").is_file()
    assert stats["gold_shape"] == (224, 224, 3)
    assert stats["solarian_shape"] == (224, 224, 3)
    assert any("Size:" in line for line in stats["lines"])


def test_format_summary_includes_compare_lines():
    text = format_summary(
        policy_shape=(1, 3, 224, 224),
        policy_dtype=np.float32,
        policy_min=0.0,
        policy_max=1.0,
        policy_mean=0.25,
        masked_shape=(1, 270, 480, 3),
        raw_shape=(1, 270, 480, 3),
        compare_lines=["Size: gold 224x224 vs solarian 224x224 (match)"],
    )
    assert "shape=(1, 3, 224, 224)" in text
    assert "masked HWC" in text
    assert "224x224" in text


def test_dump_policy_writes_directory(tmp_path):
    skill = UMISkill.__new__(UMISkill)
    skill.debug_image_path = str(tmp_path)
    skill.debug_image_max_dumps = 5
    skill.umi_gold_image = None
    skill._debug_dump_count = 0
    skill._cam_module = "cam"
    skill.observations = {}

    rgb = np.zeros((1, 3, 224, 224), dtype=np.float32)
    rgb[0, 0, 80:120, 80:120] = 1.0
    masked = np.zeros((1, 270, 480, 3), dtype=np.uint8)
    skill._dump_policy_input_image(
        {"camera0_rgb": rgb},
        {"camera0_rgb": masked},
    )
    assert (tmp_path / "solarian_camera0_rgb_t0.png").is_file()
    assert (tmp_path / "solarian_masked_hwc.png").is_file()
    assert (tmp_path / "summary.txt").is_file()
    assert "shape=(1, 3, 224, 224)" in (tmp_path / "summary.txt").read_text()


def test_dump_policy_draws_the_box_the_policy_was_sent(tmp_path):
    import cv2

    skill = UMISkill.__new__(UMISkill)
    skill.debug_image_path = str(tmp_path)
    skill.debug_image_max_dumps = 5
    skill.umi_gold_image = None
    skill._debug_dump_count = 0
    skill.observations = {}
    rgb = np.zeros((1, 3, 224, 224), dtype=np.float32)

    skill._dump_policy_input_image(
        {"camera0_rgb": rgb, "bbox": np.array([[72, 99, 85, 116]], np.float32)},
        {"camera0_rgb": np.zeros((1, 224, 224, 3), np.uint8)},
    )

    img = cv2.imread(str(tmp_path / "solarian_camera0_rgb_t0.png"))
    red = img[..., 2] > 127
    ys, xs = np.nonzero(red)
    assert [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1] == [72, 99, 85, 116]


def test_dump_policy_respects_max_dumps(tmp_path):
    skill = UMISkill.__new__(UMISkill)
    skill.debug_image_path = str(tmp_path)
    skill.debug_image_max_dumps = 1
    skill.umi_gold_image = None
    skill._debug_dump_count = 0
    skill._cam_module = "cam"
    skill.observations = {}
    rgb = np.zeros((1, 3, 224, 224), dtype=np.float32)
    obs = {"camera0_rgb": rgb}
    skill._dump_policy_input_image(obs, obs)
    skill._dump_policy_input_image(obs, obs)
    assert (tmp_path / "solarian_camera0_rgb_t0.png").is_file()
    assert not (tmp_path / "solarian_camera0_rgb_t1.png").exists()


def test_dump_policy_obs_writes_one_npz_per_inference(tmp_path):
    skill = UMISkill.__new__(UMISkill)
    skill.dump_policy_obs = str(tmp_path / "obs")
    skill._policy_obs_dump_count = 0
    obs_dict = {
        "camera0_rgb": np.full((1, 3, 224, 224), 0.5, np.float32),
        "bbox": np.array([[72, 99, 85, 116]], np.float32),
        "robot0_eef_pos": np.zeros((2, 3), np.float32),
    }
    env_obs = {
        "robot0_eef_pos": np.array([[0.0, 0.0, 0.0], [0.30, -0.05, 0.12]]),
        "robot0_eef_rot_axis_angle": np.array([[0.0, 0.0, 0.0], [3.1, 0.0, 0.1]]),
        "timestamp": np.array([1.0, 1.1]),
    }
    raw = np.ones((16, 10), np.float32)

    skill._dump_policy_obs(obs_dict, raw, env_obs=env_obs)
    skill._dump_policy_obs(obs_dict, raw, env_obs=env_obs)

    files = sorted((tmp_path / "obs").glob("*.npz"))
    assert [f.name for f in files] == ["00000.npz", "00001.npz"]
    with np.load(files[0]) as z:
        np.testing.assert_array_equal(z["obs.bbox"], obs_dict["bbox"])
        np.testing.assert_array_equal(z["obs.camera0_rgb"], obs_dict["camera0_rgb"])
        np.testing.assert_array_equal(z["raw_action"], raw)
        np.testing.assert_allclose(z["eef_pose"], [0.30, -0.05, 0.12, 3.1, 0.0, 0.1])
        assert float(z["timestamp"]) == 1.1


def _pipeline_skill(*, letterbox: bool = True):
    skill = UMISkill.__new__(UMISkill)
    skill.obs_image_resolution = (224, 224)
    skill._policy_image_tf = None
    skill._policy_image_tf_key = None
    skill.no_mirror = True
    skill.mirror_swap = False
    skill.letterbox_to_square = letterbox
    return skill


def test_resize_to_policy_is_center_square_224():
    skill = _pipeline_skill(letterbox=False)
    # 480×270: left strip red, center green, right blue.
    frame = np.zeros((1, 270, 480, 3), dtype=np.uint8)
    frame[0, :, :105] = (255, 0, 0)
    frame[0, :, 105:375] = (0, 255, 0)
    frame[0, :, 375:] = (0, 0, 255)
    out = skill._resize_to_policy(frame)
    assert out.shape == (1, 224, 224, 3)
    # Center of the square crop is the green band, not the side strips.
    center = out[0, 112, 112]
    np.testing.assert_array_equal(center, [0, 255, 0])


def test_letterbox_keeps_side_vignette():
    """16:9 black sides (fisheye tunnel) must survive into the 224 square."""
    skill = _pipeline_skill(letterbox=True)
    frame = np.zeros((1, 270, 480, 3), dtype=np.uint8)
    frame[0, :, 80:400] = 255
    out = skill._resize_to_policy(frame)
    assert out.shape == (1, 224, 224, 3)
    assert out[0, 112, 8].max() < 40
    assert out[0, 112, 112].min() > 200


def test_center_crop_drops_side_vignette():
    skill = _pipeline_skill(letterbox=False)
    frame = np.zeros((1, 270, 480, 3), dtype=np.uint8)
    frame[0, :, 80:400] = 255
    out = skill._resize_to_policy(frame)
    assert out[0, 112, 8].min() > 200


def test_letterbox_to_square_pads_16_9():
    img = np.full((270, 480, 3), 255, dtype=np.uint8)
    out = letterbox_to_square(img)
    assert out.shape == (480, 480, 3)
    assert out[0, 240].max() == 0
    assert out[240, 240].min() == 255


def test_four_three_center_crop_keeps_inscribed_circle():
    """4:3 capture + UMI crop: fisheye filling the height becomes a porthole."""
    skill = _pipeline_skill(letterbox=False)
    h, w = 360, 480
    yy, xx = np.ogrid[:h, :w]
    circle = (xx - w / 2.0) ** 2 + (yy - h / 2.0) ** 2 <= (h / 2.0) ** 2
    frame = np.zeros((1, h, w, 3), dtype=np.uint8)
    frame[0][circle] = 255
    out = skill._resize_to_policy(frame)
    assert out.shape == (1, 224, 224, 3)
    assert out[0, 2, 2].max() < 40
    assert out[0, 2, 221].max() < 40
    assert out[0, 112, 112].min() > 200


def test_masks_run_on_square_policy_frame():
    skill = _pipeline_skill()
    # White 16:9 → crop/resize → mask. Mirror/gripper polygons black the
    # sides and bottom of the 224 square (not the 16:9 letterbox).
    white = np.full((1, 270, 480, 3), 255, dtype=np.uint8)
    square = skill._resize_to_policy(white)
    masked = skill._apply_image_masks(square.copy())
    assert masked.shape == (1, 224, 224, 3)
    # Interior should stay white; left edge / bottom-center should be blacked.
    assert masked[0, 80, 112].min() > 200
    assert masked[0, 160, 5].max() < 16
    assert masked[0, 220, 112].max() < 16


def test_mask_before_crop_differs_from_umi_order():
    """Regression: masking 16:9 then cropping ≠ crop then mask on 224."""
    skill = _pipeline_skill()
    white = np.full((1, 270, 480, 3), 255, dtype=np.uint8)
    umi = skill._apply_image_masks(skill._resize_to_policy(white))
    wrong = skill._resize_to_policy(skill._apply_image_masks(white))
    assert not np.array_equal(umi, wrong)


def _synthetic_gripper(*, dx: int = 0, dy: int = 0) -> np.ndarray:
    """224×224 with orange pads, white shells, and 3 black screws per side."""
    img = np.full((224, 224, 3), 20, dtype=np.uint8)
    yy, xx = np.ogrid[:224, :224]

    def paint(base_x: int) -> None:
        x = base_x + dx
        y = 160 + dy
        img[y : y + 50, x : x + 42] = (230, 230, 230)
        img[y + 8 : y + 48, x + 28 : x + 42] = (255, 90, 20)
        for cx, cy in ((x + 12, y + 12), (x + 12, y + 32), (x + 24, y + 22)):
            img[(xx - cx) ** 2 + (yy - cy) ** 2 <= 9] = (8, 8, 8)

    paint(28)
    paint(154)
    return img


def _synthetic_gripper_polygons(*, dx: int = 0, dy: int = 0):
    from solarian_umi_skill.umi.gripper_align import RegionPolygons

    polys = RegionPolygons()
    for base_x in (28, 154):
        x = base_x + dx
        y = 160 + dy
        polys.add(
            "white",
            [(x, y), (x + 42, y), (x + 42, y + 50), (x, y + 50)],
        )
        polys.add(
            "orange",
            [(x + 28, y + 8), (x + 42, y + 8), (x + 42, y + 48), (x + 28, y + 48)],
        )
    return polys


def test_align_metrics_identical_is_full_overlap():
    from solarian_umi_skill.umi.gripper_align import compute_align_metrics

    gold = _synthetic_gripper()
    polys = _synthetic_gripper_polygons()
    m = compute_align_metrics(gold, gold.copy(), gold_polys=polys, live_polys=polys)
    assert m.orange_iou == pytest.approx(1.0, abs=0.02)
    assert m.orange_coverage == pytest.approx(1.0, abs=0.02)
    assert m.white_iou == pytest.approx(1.0, abs=0.05)
    assert m.white_coverage == pytest.approx(1.0, abs=0.05)
    assert m.n_screws_gold == 6
    assert m.n_screws_live == 6
    assert m.screw_mean_px == pytest.approx(0.0, abs=0.6)
    assert m.screw_overlap == pytest.approx(1.0, abs=0.05)


def test_policy_pipeline_process_frame_is_224():
    from solarian_umi_skill.umi.policy_image import PolicyImagePipeline

    pipe = PolicyImagePipeline(letterbox_to_square=False)
    frame = np.zeros((360, 480, 3), dtype=np.uint8)
    frame[:, 80:400] = 255
    out = pipe.process_frame(frame)
    assert out.shape == (224, 224, 3)


def test_remapped_gopro_frame_is_224_policy_input():
    """Optical remap output (1920x1080) then UMI crop/mask is 224x224."""
    from solarian_umi_skill.umi.policy_image import PolicyImagePipeline
    from solarian.core.optical_remap import (
        apply_optical_remap,
        identity_aligner,
        load_coverage_mask,
        load_shipped_gopro_intrinsics,
    )

    gopro = load_shipped_gopro_intrinsics()
    aligner = identity_aligner(gopro, gopro)
    frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
    frame[400:700, 700:1200] = (200, 180, 160)
    warped = apply_optical_remap(frame, aligner, load_coverage_mask())
    rgb = warped[..., ::-1]
    out = PolicyImagePipeline(letterbox_to_square=False).process_frame(rgb)
    assert out.shape == (224, 224, 3)


def test_render_contact_sheet_includes_align_lines():
    from solarian_umi_skill.umi.compare_policy_image import render_contact_sheet

    gold = _rgb(224, 224, (10, 10, 10))
    sheet = render_contact_sheet(gold, gold, extra_lines=["Orange fingers: IoU 100.0%"])
    assert sheet.shape[1] == 224 * 3
    assert sheet.shape[0] > 224


def test_align_metrics_shift_lowers_overlap():
    from solarian_umi_skill.umi.gripper_align import compute_align_metrics

    gold = _synthetic_gripper()
    live = _synthetic_gripper(dx=8, dy=4)
    gold_p = _synthetic_gripper_polygons()
    live_p = _synthetic_gripper_polygons(dx=8, dy=4)
    m = compute_align_metrics(gold, live, gold_polys=gold_p, live_polys=live_p)
    assert m.orange_iou < 0.85
    assert m.white_iou < 0.85
    assert m.screw_mean_px > 4.0
    assert m.screw_overlap < 0.9


def test_polygon_mask_fills_quad():
    from solarian_umi_skill.umi.gripper_align import RegionPolygons, polygons_to_mask

    polys = RegionPolygons()
    polys.add("orange", [(10, 10), (40, 10), (40, 40), (10, 40)])
    mask = polys.mask((50, 50), "orange")
    assert bool(mask[20, 20])
    assert not bool(mask[5, 5])
    filled = polygons_to_mask([[(0, 0), (9, 0), (9, 9), (0, 9)]], (10, 10))
    assert int(filled.sum()) == 100


def test_render_contact_sheet_mask_views():
    from solarian_umi_skill.umi.compare_policy_image import render_contact_sheet

    gold = _synthetic_gripper()
    extra = ["Orange fingers: IoU 100.0%"]
    polys = _synthetic_gripper_polygons()
    absdiff = render_contact_sheet(
        gold, gold, extra_lines=extra, view="absdiff", gold_polys=polys, live_polys=polys
    )
    mask = render_contact_sheet(
        gold, gold, extra_lines=extra, view="mask_iou", gold_polys=polys, live_polys=polys
    )
    both = render_contact_sheet(
        gold, gold, extra_lines=extra, view="both", gold_polys=polys, live_polys=polys
    )
    assert absdiff.shape[1] == 224 * 3
    assert mask.shape[1] == 224 * 3
    assert both.shape[1] == 224 * 3
    assert mask.shape[0] == absdiff.shape[0]
    assert both.shape[0] > absdiff.shape[0]


def test_align_polygons_roundtrip(tmp_path):
    from solarian_umi_skill.umi.gripper_align import (
        RegionPolygons,
        load_align_polygons,
        save_align_polygons,
    )

    gold = RegionPolygons(orange=[[(10, 20), (30, 20), (30, 40)]], white=[[(1, 2), (5, 2), (5, 8)]])
    live = RegionPolygons(orange=[[(11, 21), (31, 21), (31, 41)]])
    path = tmp_path / "align_polygons.json"
    save_align_polygons(path, gold, live)
    g2, l2 = load_align_polygons(path)
    assert g2.orange == [[(10, 20), (30, 20), (30, 40)]]
    assert g2.white == [[(1, 2), (5, 2), (5, 8)]]
    assert l2.orange == [[(11, 21), (31, 21), (31, 41)]]
