"""Weighted sampling across several zarrs and a normalizer fit on their union."""

from collections import Counter

import numpy as np
import pytest
import torch
import zarr

from solarian_umi_skill.diffusion_policy.common.replay_buffer import ReplayBuffer
from solarian_umi_skill.diffusion_policy.dataset.umi_dataset import UmiDataset, mix_sampler

FRAMES = 20
ACTION_HORIZON = 4


def _write_zarr(path, n_episodes: int, gripper_width: float) -> str:
    buffer = ReplayBuffer.create_empty_numpy()
    for _ in range(n_episodes):
        pose = np.zeros((FRAMES, 6), dtype=np.float32)
        buffer.add_episode({
            "camera0_rgb": np.zeros((FRAMES, 8, 8, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros((FRAMES, 3), dtype=np.float32),
            "robot0_eef_rot_axis_angle": np.zeros((FRAMES, 3), dtype=np.float32),
            "robot0_gripper_width": np.full((FRAMES, 1), gripper_width, dtype=np.float32),
            "robot0_demo_start_pose": pose,
            "robot0_demo_end_pose": pose,
        })
    with zarr.ZipStore(str(path), mode="w") as store:
        buffer.save_to_store(store)
    return str(path)


def _dataset(path) -> UmiDataset:
    def low_dim(shape, **extra):
        return {"shape": shape, "horizon": 2, "latency_steps": 0,
                "down_sample_steps": 1, "type": "low_dim", **extra}

    shape_meta = {
        "obs": {
            "camera0_rgb": {"shape": [3, 8, 8], "horizon": 1, "latency_steps": 0,
                            "down_sample_steps": 1, "type": "rgb"},
            "robot0_eef_pos": low_dim([3]),
            "robot0_eef_rot_axis_angle": low_dim([6], raw_shape=[3], rotation_rep="rotation_6d"),
            "robot0_gripper_width": low_dim([1]),
        },
        "action": {"shape": [10], "horizon": ACTION_HORIZON, "latency_steps": 0,
                   "down_sample_steps": 1, "rotation_rep": "rotation_6d"},
    }
    return UmiDataset(
        shape_meta=shape_meta, dataset_path=path,
        pose_repr={"obs_pose_repr": "relative", "action_pose_repr": "relative"},
    )


def test_parts_are_drawn_by_weight_not_by_size():
    torch.manual_seed(0)
    sampler = mix_sampler([1000, 10], [0.3, 0.7], num_samples=20000)

    indices = list(sampler)

    assert len(indices) == 20000
    assert np.mean([i >= 1000 for i in indices]) == pytest.approx(0.7, abs=0.02)


def test_a_single_part_is_drawn_without_replacement_each_pass():
    counts = Counter(mix_sampler([50], [1.0], num_samples=150))

    assert set(counts) == set(range(50))
    assert set(counts.values()) == {3}


def test_one_epoch_is_the_total_length_without_a_sample_budget():
    assert len(mix_sampler([30, 20], [1.0, 1.0], num_samples=None)) == 50


def test_normalizer_is_fit_on_the_union(tmp_path):
    narrow = _dataset(_write_zarr(tmp_path / "a.zarr.zip", n_episodes=2, gripper_width=0.05))
    wide = _dataset(_write_zarr(tmp_path / "b.zarr.zip", n_episodes=1, gripper_width=0.09))
    data = [narrow.normalizer_data(), wide.normalizer_data()]

    union = narrow.fit_normalizer({k: np.concatenate([d[k] for d in data]) for k in data[0]})
    alone = narrow.get_normalizer()

    stats = union["robot0_gripper_width"].params_dict["input_stats"]
    assert stats["min"].item() == pytest.approx(0.05)
    assert stats["max"].item() == pytest.approx(0.09)
    assert alone["robot0_gripper_width"].params_dict["input_stats"]["max"].item() == \
        pytest.approx(0.05)
