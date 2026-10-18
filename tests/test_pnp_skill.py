"""Point and pick: placement from config, and the box as the policy sees it."""

import numpy as np
import pytest

from solarian.core.execution import ExecutionPlace

from solarian_umi_skill import UMIPnPSkill, UMISkill
from solarian_umi_skill.umi.policy_image import PolicyImagePipeline


def _conf(tmp_path, monkeypatch, text):
    conf = tmp_path / "conf" / "skills"
    conf.mkdir(parents=True)
    (conf / "pnp.yaml").write_text(
        "arm: right_arm\ngripper: right_gripper\ncamera: right_wrist_camera\n" + text,
        encoding="utf-8",
    )
    monkeypatch.setenv("SOLARIAN_CONF_PATH", str(conf.parent))
    monkeypatch.delenv("SOLARIAN_DEPLOYMENT", raising=False)


def test_pnp_placement_must_be_explicit(tmp_path, monkeypatch):
    _conf(tmp_path, monkeypatch, "")
    with pytest.raises(ValueError, match="execution_place"):
        UMIPnPSkill(params={"config": "pnp.yaml"})


def test_pnp_placement_from_a_missing_deployment_key_fails(tmp_path, monkeypatch):
    _conf(tmp_path, monkeypatch, "execution_place: ${deployment.pnp_execution_place}\n")
    with pytest.raises(Exception, match="pnp_execution_place"):
        UMIPnPSkill(params={"config": "pnp.yaml"})


def test_execution_place_comes_from_the_config(tmp_path, monkeypatch):
    _conf(tmp_path, monkeypatch, "execution_place: robot\n")
    skill = UMIPnPSkill(params={"config": "pnp.yaml"})
    assert skill.execution_place is ExecutionPlace.ROBOT
    assert skill.registry_name() == "umi_pnp"
    assert UMISkill(params={"config": "pnp.yaml"}).execution_place is ExecutionPlace.ROBOT


def test_the_host_running_it_wins_over_the_config(tmp_path, monkeypatch):
    _conf(tmp_path, monkeypatch, "execution_place: robot\n")
    skill = UMIPnPSkill(params={"config": "pnp.yaml"}, execution_place="server")
    assert skill.execution_place is ExecutionPlace.SERVER


BBOX_META = {"bbox": {"shape": [4], "type": "low_dim", "bbox_mode": "mask_channel"}}
MASK_META = {"mask": {"shape": [1, 224, 224], "type": "mask"}}


def test_the_box_is_readable_before_setup_finishes(tmp_path, monkeypatch):
    """The base setup builds the first observation, box included."""
    _conf(tmp_path, monkeypatch, "execution_place: server\n")
    skill = UMIPnPSkill(params={"config": "pnp.yaml"})
    # What the base setup has fetched from the policy server by then.
    skill.shape_meta = {"obs": BBOX_META}
    skill._ensure_policy_pipeline = lambda: type(
        "Pipe", (), {"bbox": staticmethod(lambda raw, hw, warp: np.asarray(raw, np.float32))}
    )()
    box = type("Box", (), {"value": np.array([1, 2, 3, 4], np.int32)})()

    extras = skill._camera_obs_extras([{"bbox": box}], (10, 10), None)

    assert extras["bbox"].tolist() == [[1, 2, 3, 4]]


class _Field:
    def __init__(self, value, timestamp=1.0):
        self.value = value
        self.timestamp = timestamp


def test_a_mask_checkpoint_gets_the_target_field_as_a_mask(tmp_path, monkeypatch):
    _conf(tmp_path, monkeypatch, "execution_place: server\ntarget_field: target_mask\n")
    skill = UMIPnPSkill(params={"config": "pnp.yaml"})
    skill.shape_meta = {"obs": MASK_META}
    skill._ensure_policy_pipeline = lambda: PolicyImagePipeline(output_res=(224, 224))
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


def test_an_unchanged_mask_stamp_reuses_the_mapped_mask(tmp_path, monkeypatch):
    _conf(tmp_path, monkeypatch, "execution_place: server\ntarget_field: target_mask\n")
    skill = UMIPnPSkill(params={"config": "pnp.yaml"})
    skill.shape_meta = {"obs": MASK_META}
    calls = []

    class Pipe:
        def target_mask(self, mask, hw, warp):
            calls.append(hw)
            return np.zeros((224, 224), np.float32)

    skill._ensure_policy_pipeline = Pipe
    frame = {"target_mask": _Field(np.ones((8, 8), np.uint8), 5.0)}
    skill._camera_obs_extras([frame, frame], (8, 8), None)
    skill._camera_obs_extras([frame], (8, 8), None)
    assert len(calls) == 1


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
