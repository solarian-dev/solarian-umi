"""UmiDataset val_dataset_path: validate on a separate zarr, train on all of dataset_path."""

import numpy as np
import zarr

from solarian_umi_skill.diffusion_policy.common.replay_buffer import ReplayBuffer
from solarian_umi_skill.diffusion_policy.dataset.umi_dataset import UmiDataset

FRAMES = 40
ACTION_HORIZON = 4


def _write_zarr(path, n_episodes: int, bbox_x0: float) -> str:
    buffer = ReplayBuffer.create_empty_numpy()
    for _ in range(n_episodes):
        pose = np.zeros((FRAMES, 6), dtype=np.float32)
        buffer.add_episode({
            "camera0_rgb": np.zeros((FRAMES, 8, 8, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros((FRAMES, 3), dtype=np.float32),
            "robot0_eef_rot_axis_angle": np.zeros((FRAMES, 3), dtype=np.float32),
            "robot0_gripper_width": np.full((FRAMES, 1), 0.05, dtype=np.float32),
            "robot0_demo_start_pose": pose,
            "robot0_demo_end_pose": pose,
            "bbox": np.tile(np.array([bbox_x0, 0, 4, 4], dtype=np.float32), (FRAMES, 1)),
        })
    with zarr.ZipStore(str(path), mode="w") as store:
        buffer.save_to_store(store)
    return str(path)


def _shape_meta() -> dict:
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
            "bbox": low_dim([4], horizon=1, bbox_mode="mask_channel",
                            target_rgb_key="camera0_rgb"),
        },
        "action": {"shape": [10], "horizon": ACTION_HORIZON, "latency_steps": 0,
                   "down_sample_steps": 1, "rotation_rep": "rotation_6d"},
    }


def _dataset(train_path, **kwargs) -> UmiDataset:
    return UmiDataset(
        shape_meta=_shape_meta(), dataset_path=train_path,
        pose_repr={"obs_pose_repr": "relative", "action_pose_repr": "relative"},
        **kwargs,
    )


SAMPLES_PER_EPISODE = FRAMES - (ACTION_HORIZON - 1)


def test_separate_val_zarr_is_the_whole_validation_set(tmp_path):
    train = _write_zarr(tmp_path / "train.zarr.zip", n_episodes=5, bbox_x0=1)
    val = _write_zarr(tmp_path / "val.zarr.zip", n_episodes=2, bbox_x0=2)

    dataset = _dataset(train, val_ratio=0.4, val_dataset_path=val)
    val_set = dataset.get_validation_dataset()

    assert len(dataset) == 5 * SAMPLES_PER_EPISODE
    assert len(val_set) == 2 * SAMPLES_PER_EPISODE
    assert val_set[0]["obs"]["bbox"][0, 0].item() == 2
    assert dataset[0]["obs"]["bbox"][0, 0].item() == 1


def test_without_val_zarr_val_ratio_splits_dataset_path(tmp_path):
    train = _write_zarr(tmp_path / "train.zarr.zip", n_episodes=5, bbox_x0=1)

    dataset = _dataset(train, val_ratio=0.4)

    assert len(dataset) == 3 * SAMPLES_PER_EPISODE
    assert len(dataset.get_validation_dataset()) == 2 * SAMPLES_PER_EPISODE


def test_training_frame_cache_is_not_applied_to_a_separate_val_zarr(tmp_path):
    train = _write_zarr(tmp_path / "train.zarr.zip", n_episodes=2, bbox_x0=1)
    val = _write_zarr(tmp_path / "val.zarr.zip", n_episodes=1, bbox_x0=2)
    dataset = _dataset(train, val_dataset_path=val)
    val_set = dataset.get_validation_dataset()
    cache = np.zeros((2 * FRAMES, ACTION_HORIZON, 10), dtype=np.float32)

    dataset.set_action_optimized(cache)
    val_set.set_action_optimized(cache)

    assert dataset.has_action_optimized_cache
    assert not val_set.has_action_optimized_cache
