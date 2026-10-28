"""UMISkill with targets: placement, checks against the checkpoint, and each
target as the policy sees it."""

import numpy as np
import pytest

from solarian.core.data import BoundingBoxData, MaskData, PointData
from solarian.core.execution import ExecutionPlace

from solarian_umi_skill import UMISkill
from solarian_umi_skill.umi.policy_image import PolicyImagePipeline


def _conf(tmp_path, monkeypatch, text):
    conf = tmp_path / "conf" / "skills"
    conf.mkdir(parents=True, exist_ok=True)
    (conf / "target.yaml").write_text(
        "arm: right_arm\ngripper: right_gripper\ncamera: right_wrist_camera\n" + text,
        encoding="utf-8",
    )
    monkeypatch.setenv("SOLARIAN_CONF_PATH", str(conf.parent))
    monkeypatch.delenv("SOLARIAN_DEPLOYMENT", raising=False)


def test_targets_need_an_explicit_placement(tmp_path, monkeypatch):
    _conf(tmp_path, monkeypatch, "targets: {bbox: bbox}\n")
    with pytest.raises(ValueError, match="execution_place"):
        UMISkill(params={"config": "target.yaml"})


def test_a_skill_without_targets_runs_on_the_robot_by_default(tmp_path, monkeypatch):
    _conf(tmp_path, monkeypatch, "")
    assert UMISkill(params={"config": "target.yaml"}).execution_place is ExecutionPlace.ROBOT


def test_placement_from_a_missing_deployment_key_fails(tmp_path, monkeypatch):
    _conf(tmp_path, monkeypatch, "execution_place: ${deployment.umi_target_execution_place}\n")
    with pytest.raises(Exception, match="umi_target_execution_place"):
        UMISkill(params={"config": "target.yaml"})


def test_execution_place_comes_from_the_config(tmp_path, monkeypatch):
    _conf(tmp_path, monkeypatch, "execution_place: server\ntargets: {bbox: bbox}\n")
    skill = UMISkill(params={"config": "target.yaml"})
    assert skill.execution_place is ExecutionPlace.SERVER
    assert skill.registry_name() == "umi"


def test_the_host_running_it_wins_over_the_config(tmp_path, monkeypatch):
    _conf(tmp_path, monkeypatch, "execution_place: robot\ntargets: {bbox: bbox}\n")
    skill = UMISkill(params={"config": "target.yaml"}, execution_place="server")
    assert skill.execution_place is ExecutionPlace.SERVER


BBOX_META = {"bbox": {"shape": [4], "type": "low_dim", "bbox_mode": "mask_channel", "horizon": 1}}
MASK_META = {"mask": {"shape": [1, 224, 224], "type": "mask", "horizon": 1}}
POINT_META = {"point": {"shape": [2], "type": "low_dim", "point_mode": "mask_channel", "horizon": 1}}


class _Field:
    def __init__(self, value, timestamp=1.0):
        self.value = value
        self.timestamp = timestamp


class _Spec:
    def __init__(self, data_class, place=ExecutionPlace.SERVER):
        self.data_class = data_class
        self.execution_place = place


def _checked(tmp_path, monkeypatch, targets, obs_meta, fields, place="server"):
    _conf(tmp_path, monkeypatch, f"execution_place: {place}\ntargets: {targets}\n")
    skill = UMISkill(params={"config": "target.yaml"})
    skill.camera_horizon = 1
    observation = type("Obs", (), {"get_calculated_fields_spec": staticmethod(lambda: fields)})
    monkeypatch.setattr("solarian_umi_skill.skill.Observation", {"right_wrist_camera": observation()})
    skill._check_targets({"camera0_rgb": {"horizon": 1, "type": "rgb"}, **obs_meta})
    return skill


def test_the_targets_must_be_exactly_the_checkpoints(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="maps"):
        _checked(tmp_path, monkeypatch, "{bbox: bbox}", {**BBOX_META, **POINT_META},
                 {"bbox": _Spec(BoundingBoxData)})
    with pytest.raises(ValueError, match="observes targets none"):
        _checked(tmp_path, monkeypatch, "{bbox: bbox}", {}, {"bbox": _Spec(BoundingBoxData)})


def test_a_point_target_needs_a_point_field(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="needs a PointData field"):
        _checked(tmp_path, monkeypatch, "{point: bbox}", POINT_META, {"bbox": _Spec(BoundingBoxData)})


def test_a_robot_skill_cannot_read_a_relay_field(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="computed on the server"):
        _checked(tmp_path, monkeypatch, "{point: target_point}", POINT_META,
                 {"target_point": _Spec(PointData)}, place="robot")


def test_a_box_and_a_point_on_one_image_pass(tmp_path, monkeypatch):
    _checked(tmp_path, monkeypatch, "{bbox: bbox, point: target_point}", {**BBOX_META, **POINT_META},
             {"bbox": _Spec(BoundingBoxData), "target_point": _Spec(PointData), "m": _Spec(MaskData)})


def _skill(tmp_path, monkeypatch, targets, obs_meta, pipe):
    _conf(tmp_path, monkeypatch, f"execution_place: server\ntargets: {targets}\n")
    skill = UMISkill(params={"config": "target.yaml"})
    # What setup has fetched from the policy server when it builds the first observation.
    skill.shape_meta = {"obs": obs_meta}
    skill._ensure_policy_pipeline = lambda: pipe
    return skill


def test_a_box_target_is_read_before_setup_finishes(tmp_path, monkeypatch):
    pipe = type("Pipe", (), {"bbox": staticmethod(lambda raw, hw, warp: np.asarray(raw, np.float32))})()
    skill = _skill(tmp_path, monkeypatch, "{bbox: bbox}", BBOX_META, pipe)

    extras = skill._camera_obs_extras([{"bbox": _Field(np.array([1, 2, 3, 4], np.int32))}], (10, 10), None)

    assert extras["bbox"].tolist() == [[1, 2, 3, 4]]


def test_a_mask_target_is_the_field_as_a_mask(tmp_path, monkeypatch):
    skill = _skill(tmp_path, monkeypatch, "{mask: target_mask}", MASK_META,
                   PolicyImagePipeline(output_res=(224, 224)))
    mask = np.zeros((480, 640), np.uint8)
    mask[200:280, 280:360] = 1
    frames = [{"target_mask": _Field(None, 1.0)}, {"target_mask": _Field(mask, 2.0)}]

    extras = skill._camera_obs_extras(frames, (480, 640), None)

    assert set(extras) == {"mask"}
    out = extras["mask"]
    assert out.shape == (2, 1, 224, 224) and out.dtype == np.float32
    # No target yet: the dataset's empty mask, not the whole frame.
    assert out[0].sum() == 0
    ys, xs = np.nonzero(out[1, 0])
    # 640x480 -> 299x224, center crop x offset 37.
    assert xs.min() == pytest.approx(280 * 224 / 480 - 37, abs=1.5)
    assert ys.min() == pytest.approx(200 * 224 / 480, abs=1.5)


def test_a_box_and_a_point_are_both_sent(tmp_path, monkeypatch):
    skill = _skill(tmp_path, monkeypatch, "{bbox: bbox, point: target_point}", {**BBOX_META, **POINT_META},
                   PolicyImagePipeline(output_res=(224, 224)))
    frame = {"bbox": _Field(np.array([320, 240, 400, 320], np.int32)),
             "target_point": _Field(np.array([480, 240, 90], np.float32))}

    extras = skill._camera_obs_extras([frame], (480, 640), None)

    assert extras["bbox"].shape == (1, 4) and extras["point"].shape == (1, 2)
    assert extras["point"][0].tolist() == pytest.approx([480 * 224 / 480 - 37, 112], abs=1.0)


def test_an_unchanged_target_reuses_the_mapped_one(tmp_path, monkeypatch):
    calls = []

    class Pipe:
        def target_mask(self, mask, hw, warp):
            calls.append("mask")
            return np.zeros((224, 224), np.float32)

        def point(self, point, hw, warp):
            calls.append("point")
            return np.zeros(2, np.float32)

    skill = _skill(tmp_path, monkeypatch, "{mask: target_mask, point: target_point}",
                   {**MASK_META, **POINT_META}, Pipe())
    frame = {"target_mask": _Field(np.ones((8, 8), np.uint8), 5.0),
             "target_point": _Field(np.array([2, 3, 100], np.float32))}
    skill._camera_obs_extras([frame, frame], (8, 8), None)
    skill._camera_obs_extras([frame], (8, 8), None)
    assert calls == ["mask", "point"]


def test_target_mask_follows_the_resize_crop_and_warp():
    pipe = PolicyImagePipeline(output_res=(224, 224))
    mask = np.zeros((480, 480), np.uint8)
    mask[:, :240] = 1
    flip = lambda img: img[:, ::-1].copy()
    out = pipe.target_mask(mask, (480, 480), warp=flip)
    assert out.dtype == np.float32 and set(np.unique(out).tolist()) <= {0.0, 1.0}
    # A flipped left half is the right half (less what the gripper paint hides).
    assert out[:, :110].sum() == 0
    assert out[:100, 114:].mean() > 0.95


def test_target_mask_is_cleared_under_the_gripper_paint():
    pipe = PolicyImagePipeline(output_res=(224, 224))
    out = pipe.target_mask(np.ones((224, 224), np.uint8), (224, 224))
    painted = pipe.mask(np.full((1, 224, 224, 3), 255, np.uint8))[0, ..., 0] <= 127
    assert painted.any()
    assert out[painted].sum() == 0
    assert out[~painted].all()


def test_target_mask_from_another_resolution_is_refused():
    pipe = PolicyImagePipeline(output_res=(224, 224))
    with pytest.raises(ValueError, match="target mask"):
        pipe.target_mask(np.ones((10, 10), np.uint8), (480, 640))


def test_empty_box_is_the_whole_policy_frame():
    pipe = PolicyImagePipeline(output_res=(224, 224))
    out = pipe.bbox(np.zeros(4, dtype=np.int32), (480, 640))
    assert out.tolist() == [0, 0, 224, 224]
    assert out.dtype == np.float32


def test_box_follows_the_resize_and_center_crop():
    # 640x480 -> resize to 299x224 -> center crop to 224x224 (x offset 37).
    pipe = PolicyImagePipeline(output_res=(224, 224))
    out = pipe.bbox(np.array([320, 240, 400, 320]), (480, 640))
    x0, y0, x1, y1 = out.tolist()
    assert x0 == pytest.approx(320 * 224 / 480 - 37, abs=1.5)
    assert y0 == pytest.approx(112, abs=1.5)
    assert x1 == pytest.approx(400 * 224 / 480 - 37, abs=1.5)
    assert y1 == pytest.approx(320 * 224 / 480, abs=1.5)


def test_box_cropped_away_falls_back_to_whole_frame():
    pipe = PolicyImagePipeline(output_res=(224, 224))
    # Left 37 policy px (~80 capture px) are cropped off.
    out = pipe.bbox(np.array([0, 100, 40, 200]), (480, 640))
    assert out.tolist() == [0, 0, 224, 224]


def test_box_goes_through_the_warp():
    pipe = PolicyImagePipeline(output_res=(224, 224))
    flip = lambda img: img[:, ::-1].copy()
    out = pipe.bbox(np.array([0, 0, 240, 240]), (480, 480), warp=flip)
    # A flipped left half is the right half.
    assert out.tolist() == pytest.approx([112, 0, 224, 112], abs=1)


def test_a_point_follows_the_resize_and_center_crop():
    pipe = PolicyImagePipeline(output_res=(224, 224))
    out = pipe.point(np.array([320, 240, 90]), (480, 640))
    assert out.dtype == np.float32
    assert out.tolist() == pytest.approx([320 * 224 / 480 - 37, 112], abs=1.0)


def test_no_point_and_a_cropped_point_are_the_datasets_no_point():
    pipe = PolicyImagePipeline(output_res=(224, 224))
    assert pipe.point(np.zeros(3), (480, 640)).tolist() == [-1, -1]
    assert pipe.point(np.array([20, 240, 100]), (480, 640)).tolist() == [-1, -1]


def test_a_point_goes_through_the_warp():
    pipe = PolicyImagePipeline(output_res=(224, 224))
    flip = lambda img: img[:, ::-1].copy()
    out = pipe.point(np.array([100, 240, 100]), (480, 480), warp=flip)
    assert out.tolist() == pytest.approx([(480 - 100) * 224 / 480, 112], abs=1.0)
