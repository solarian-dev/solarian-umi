"""The mask task: the object's segmentation instead of a box as the 4th ViT channel."""

from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from solarian_umi_skill.diffusion_policy.common.replay_buffer import ReplayBuffer
from solarian_umi_skill.diffusion_policy.dataset.umi_dataset import UmiDataset
from solarian_umi_skill.diffusion_policy.model.vision.timm_obs_encoder import (
    TimmObsEncoder,
    rasterize_bbox_to_mask,
)
from solarian.core.ai.system1.trainer_registry import load_train_config
from solarian_umi_skill.trainer import UmiTrainer, has_mask, uses_mask

TRAINING = Path(__file__).resolve().parents[1] / "solarian_umi_skill" / "templates" / "training"


def _encoder_meta(target: str) -> dict:
    obs = {
        "camera0_rgb": {"shape": [3, 224, 224], "horizon": 1, "type": "rgb"},
        "robot0_gripper_width": {"shape": [1], "horizon": 2, "type": "low_dim"},
    }
    if target == "bbox":
        obs["bbox"] = {"shape": [4], "horizon": 1, "type": "low_dim",
                       "bbox_mode": "mask_channel", "target_rgb_key": "camera0_rgb"}
    else:
        obs["mask"] = {"shape": [1, 224, 224], "horizon": 1, "type": "mask",
                       "target_rgb_key": "camera0_rgb"}
    return {"obs": obs, "action": {"shape": [10], "horizon": 16}}


def _encoder(target: str, feature_aggregation=None, meta=None) -> TimmObsEncoder:
    return TimmObsEncoder(
        shape_meta=meta or _encoder_meta(target),
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        frozen=False,
        global_pool="",
        transforms=None,
        feature_aggregation=feature_aggregation,
    ).eval()


@pytest.mark.parametrize("feature_aggregation", [None, "box_pool"])
def test_a_mask_of_the_box_conditions_exactly_like_the_box(feature_aggregation):
    """Same weights, same channel: a bbox checkpoint loads strict and agrees."""
    torch.manual_seed(0)
    box_enc = _encoder("bbox", feature_aggregation)
    mask_enc = _encoder("mask", feature_aggregation)
    mask_enc.load_state_dict(box_enc.state_dict(), strict=True)
    # The widened patch embed starts at zero for the 4th channel; give it
    # weight so the comparison actually exercises the target.
    with torch.no_grad():
        proj = box_enc.key_model_map["camera0_rgb"].patch_embed.proj
        proj.weight[:, 3].normal_()
    mask_enc.load_state_dict(box_enc.state_dict(), strict=True)

    boxes = torch.tensor([[16, 16, 64, 64], [100, 40, 208, 120]], dtype=torch.float32).view(2, 1, 4)
    rgb = torch.rand(2, 1, 3, 224, 224)
    width = torch.rand(2, 2, 1)
    with torch.no_grad():
        from_box = box_enc({"camera0_rgb": rgb, "robot0_gripper_width": width, "bbox": boxes})
        from_mask = mask_enc({"camera0_rgb": rgb, "robot0_gripper_width": width,
                              "mask": rasterize_bbox_to_mask(boxes, H=224, W=224)})

    assert box_enc.output_shape() == mask_enc.output_shape()
    torch.testing.assert_close(from_mask, from_box)


def test_a_mask_follows_its_shape_not_its_box():
    torch.manual_seed(0)
    enc = _encoder("mask", "box_pool")
    with torch.no_grad():
        enc.key_model_map["camera0_rgb"].patch_embed.proj.weight[:, 3].normal_()
    obs = {"camera0_rgb": torch.rand(1, 1, 3, 224, 224), "robot0_gripper_width": torch.rand(1, 2, 1)}
    diagonal = torch.zeros(1, 1, 1, 224, 224)
    for i in range(224):
        diagonal[..., i, max(0, i - 8):i + 8] = 1.0
    full = torch.ones_like(diagonal)
    with torch.no_grad():
        a = enc({**obs, "mask": diagonal})
        b = enc({**obs, "mask": full})
    # Same tight box (the whole frame), different pixels: different features.
    assert not torch.allclose(a, b)


def test_mask_shape_must_match_the_target_image():
    meta = _encoder_meta("mask")
    meta["obs"]["mask"]["shape"] = [1, 112, 112]
    with pytest.raises(RuntimeError, match=r"must have shape \[1, 224, 224\]"):
        _encoder("mask", meta=meta)


def test_two_targets_for_one_image_are_refused():
    meta = _encoder_meta("mask")
    meta["obs"]["bbox"] = _encoder_meta("bbox")["obs"]["bbox"]
    with pytest.raises(RuntimeError, match="camera0_rgb"):
        _encoder("mask", meta=meta)


# ---- dataset ---------------------------------------------------------------

FRAMES = 20
ACTION_HORIZON = 4


def _write_zarr(path, n_episodes: int = 2, with_mask: bool = True) -> str:
    buffer = ReplayBuffer.create_empty_numpy()
    for ep in range(n_episodes):
        pose = np.zeros((FRAMES, 6), dtype=np.float32)
        episode = {
            "camera0_rgb": np.zeros((FRAMES, 8, 8, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros((FRAMES, 3), dtype=np.float32),
            "robot0_eef_rot_axis_angle": np.zeros((FRAMES, 3), dtype=np.float32),
            "robot0_gripper_width": np.full((FRAMES, 1), 0.05, dtype=np.float32),
            "robot0_demo_start_pose": pose,
            "robot0_demo_end_pose": pose,
            "bbox": np.tile(np.array([1, 1, 4, 4], dtype=np.float32), (FRAMES, 1)),
        }
        if with_mask:
            mask = np.zeros((FRAMES, 8, 8), dtype=np.uint8)
            mask[:, 1:4, 1 + ep:4 + ep] = 1
            episode["mask"] = mask
        buffer.add_episode(episode)
    with zarr.ZipStore(str(path), mode="w") as store:
        buffer.save_to_store(store)
    return str(path)


def _dataset_meta() -> dict:
    def low_dim(shape, horizon=2, **extra):
        return {"shape": shape, "horizon": horizon, "latency_steps": 0,
                "down_sample_steps": 1, "type": "low_dim", **extra}

    return {
        "obs": {
            "camera0_rgb": {"shape": [3, 8, 8], "horizon": 1, "latency_steps": 0,
                            "down_sample_steps": 1, "type": "rgb"},
            "robot0_eef_pos": low_dim([3]),
            "robot0_eef_rot_axis_angle": low_dim([6], raw_shape=[3], rotation_rep="rotation_6d"),
            "robot0_gripper_width": low_dim([1]),
            "mask": {"shape": [1, 8, 8], "horizon": 1, "latency_steps": 0,
                     "down_sample_steps": 1, "type": "mask", "target_rgb_key": "camera0_rgb"},
        },
        "action": {"shape": [10], "horizon": ACTION_HORIZON, "latency_steps": 0,
                   "down_sample_steps": 1, "rotation_rep": "rotation_6d"},
    }


def test_dataset_samples_the_mask_per_frame_as_a_channel(tmp_path):
    dataset = UmiDataset(
        shape_meta=_dataset_meta(), dataset_path=_write_zarr(tmp_path / "m.zarr.zip"),
        pose_repr={"obs_pose_repr": "relative", "action_pose_repr": "relative"}, val_ratio=0.0,
    )
    first = dataset[0]["obs"]["mask"]
    last = dataset[len(dataset) - 1]["obs"]["mask"]

    assert tuple(first.shape) == (1, 1, 8, 8)
    assert first.dtype == torch.float32
    assert first.sum().item() == 9 and first[0, 0, 1:4, 1:4].all()
    # The second episode's mask is shifted one pixel right.
    assert last[0, 0, 1:4, 2:5].all()
    assert "mask" in dataset.mask_keys and "mask" not in dataset.sampler_lowdim_keys

    normalizer = dataset.get_normalizer()
    x = torch.rand(2, 1, 1, 8, 8)
    torch.testing.assert_close(normalizer["mask"].normalize(x), x)


# ---- trainer ---------------------------------------------------------------

def _cfg(name: str = "umi_pick_mask", *overrides: str):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(TRAINING)):
        cfg = compose(config_name=name, overrides=list(overrides))
    return load_train_config(cfg)[2]


def test_the_mask_template_observes_a_mask_and_no_box():
    cfg = _cfg()
    umi = cfg.umi_config()

    obs = umi["task"]["shape_meta"]["obs"]
    assert "bbox" not in obs
    assert obs["mask"]["type"] == "mask" and obs["mask"]["shape"] == [1, 224, 224]
    assert obs["mask"]["target_rgb_key"] == "camera0_rgb"
    assert "mask" in umi["task"]["dataset"]["shape_meta"]["obs"]
    assert umi["policy"]["obs_encoder"]["feature_aggregation"] == "box_pool"
    assert umi["bbox_mode"] == "mask"
    assert uses_mask(cfg.task)


def test_a_box_template_does_not_use_a_mask():
    assert not uses_mask(_cfg("umi_pick").task)


def test_has_mask_reads_the_zip_index(tmp_path):
    assert has_mask(_write_zarr(tmp_path / "with.zarr.zip"))
    assert not has_mask(_write_zarr(tmp_path / "without.zarr.zip", with_mask=False))


def test_mask_training_refuses_a_dataset_without_masks(tmp_path, monkeypatch):
    def fail(*_a, **_k):
        raise AssertionError("training should not start")

    monkeypatch.setattr("solarian_umi_skill.training.run_training", fail)
    with_mask = _write_zarr(tmp_path / "with.zarr.zip")
    without = _write_zarr(tmp_path / "without.zarr.zip", with_mask=False)
    cfg = _cfg("umi_pick_mask", f"task.dataset_mix=[{{path:'{with_mask}',weight:1.0}}]",
               f"task.dataset.val_dataset_path='{without}'")
    with pytest.raises(ValueError, match="zarr_add_masks"):
        UmiTrainer(cfg, output_dir=str(tmp_path)).train()
