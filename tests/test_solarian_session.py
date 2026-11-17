"""umi-slam's solarian session: what it writes, and training reading it back."""

import json
import pickle
import subprocess
import sys
import zipfile
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import pytest
import zarr
from hydra import compose, initialize_config_dir

from solarian_umi.diffusion_policy.common.replay_buffer import ReplayBuffer
from solarian_umi.diffusion_policy.dataset.solarian_session import (
    SessionArchive,
    episodes_missing_target,
    has_target,
    is_session,
    load_session_replay_buffer,
    write_targets,
)
from solarian_umi.diffusion_policy.dataset import umi_dataset
from solarian_umi.diffusion_policy.dataset.umi_dataset import UmiDataset
from solarian.core.ai.system1.trainer_registry import load_train_config
from solarian_umi.slam.support.policy_frame import PolicyFrame
from solarian_umi.trainer import UmiTrainer, _missing_target_hint, has_target_array

SCRIPTS = Path(__file__).resolve().parents[1] / "solarian_umi" / "slam" / "scripts"
VIDEO_WH = (64, 48)
OUT_WH = (32, 24)
FPS = 30
N_FRAMES = 40
#: Plan episodes as (frame_start, frame_end) in the one demo video.
SEGMENTS = [(3, 17), (20, 38)]
TAG = np.array([[40, 10], [50, 10], [50, 20], [40, 20]], dtype=np.float32)


def _frame(i: int) -> np.ndarray:
    """Smooth gradients, so a re-encode stays close to its source."""
    w, h = VIDEO_WH
    x = np.linspace(0, 255, w)[None, :, None]
    y = np.linspace(0, 255, h)[:, None, None]
    img = np.broadcast_to((x * 0.5 + y * 0.3 + i * 3) % 256, (h, w, 3)).copy()
    img[..., 1] = 128
    return img.astype(np.uint8)


@pytest.fixture(scope="module")
def slam_session(tmp_path_factory):
    """The files steps 00-06 leave behind, for one demo of one Lumi."""
    root = tmp_path_factory.mktemp("lumi_session")
    demo = root / "demos" / "demo_A"
    demo.mkdir(parents=True)
    with av.open(str(demo / "raw_video.mp4"), mode="w") as container:
        stream = container.add_stream("libx264", rate=Fraction(FPS))
        stream.width, stream.height = VIDEO_WH
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "0"}
        for i in range(N_FRAMES):
            for packet in stream.encode(av.VideoFrame.from_ndarray(_frame(i), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    tags = [{"frame_idx": i, "time": i / FPS,
             "tag_dict": {7: {"corners": TAG + i % 3}} if i % 2 else {}}
            for i in range(N_FRAMES)]
    (demo / "tag_detection.pkl").write_bytes(pickle.dumps(tags))

    rng = np.random.default_rng(0)
    plan = []
    for start, end in SEGMENTS:
        n = end - start
        plan.append({
            "episode_timestamps": 1.7e9 + np.arange(start, end) / FPS,
            "grippers": [{
                "tcp_pose": rng.normal(size=(n, 6)),
                "gripper_width": rng.uniform(0.0, 0.09, size=n),
                "demo_start_pose": rng.normal(size=6),
                "demo_end_pose": rng.normal(size=6),
            }],
            "cameras": [{"video_path": "demo_A/raw_video.mp4", "video_start_end": (start, end)}],
        })
    (root / "dataset_plan.pkl").write_bytes(pickle.dumps(plan))
    return root


def _run(script: str, *args) -> None:
    subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)], check=True)


@pytest.fixture(scope="module")
def session(slam_session):
    out = slam_session / "dataset.solarian.zip"
    _run("07_generate_session.py", "-o", out, slam_session)
    return out


@pytest.fixture(scope="module")
def zarr_buffer(slam_session):
    out = slam_session / "dataset.zarr.zip"
    _run("07_generate_replay_buffer.py", "-o", out, "-or", f"{OUT_WH[0]},{OUT_WH[1]}",
         slam_session)
    with zarr.ZipStore(str(out), mode="r") as store:
        return ReplayBuffer.copy_from_store(src_store=store, store=zarr.MemoryStore())


def test_the_output_is_one_zip_with_the_session_at_its_root(session):
    assert session.is_file() and is_session(session)
    assert not any(p.name.startswith(".solarian-") for p in session.parent.iterdir())
    with zipfile.ZipFile(session) as zf:
        names = zf.namelist()
        assert "session.json" in names and "manifest.jsonl" in names
        assert "episodes/000000/episode.parquet" in names
        assert zf.getinfo("episodes/000000/video/camera0.rgb.mp4").compress_type == \
            zipfile.ZIP_STORED


def test_the_session_has_the_solarian_layout(session):
    with SessionArchive(session) as archive:
        meta = json.loads(archive.read_text("session.json"))
        assert meta["cameras"]["camera0.rgb"]["width"] == VIDEO_WH[0]
        assert meta["cameras"]["camera0.rgb"]["height"] == VIDEO_WH[1]
        assert meta["obs_fields"]["robot0"] == ["x", "y", "z", "rx", "ry", "rz", "gripper_width"]
        manifest = archive.read_text("manifest.jsonl").splitlines()
        assert [json.loads(line)["episode"] for line in manifest] == [0, 1]

        episodes = archive.episodes()
        assert episodes == ["episodes/000000", "episodes/000001"]
        for episode, (start, end) in zip(episodes, SEGMENTS):
            with archive.open(f"{episode}/episode.parquet") as fh:
                table = pq.read_table(fh)
            assert table.num_rows == end - start
            assert table.column("cam.camera0.rgb.frame_index").to_pylist() == \
                list(range(end - start))
            assert set(meta["schema"]) == set(table.schema.names)
            with archive.open(f"{episode}/video/camera0.rgb.mp4") as fh, av.open(fh) as container:
                frames = list(container.decode(video=0))
            assert len(frames) == end - start
            assert (frames[0].width, frames[0].height) == VIDEO_WH
            tags = json.loads(archive.read_text(f"{episode}/tags/camera0.rgb.json"))
            assert len(tags) == end - start
            assert ("7" in tags[0]) == bool(start % 2)
            episode_meta = json.loads(archive.read_text(f"{episode}/episode.json"))
            assert len(episode_meta["umi"]["robot0"]["demo_start_pose"]) == 6


def test_an_unzipped_session_loads_the_same(session, tmp_path):
    with zipfile.ZipFile(session) as zf:
        zf.extractall(tmp_path / "dataset")
    assert is_session(tmp_path / "dataset")

    zipped = load_session_replay_buffer(session, OUT_WH)
    unzipped = load_session_replay_buffer(tmp_path / "dataset", OUT_WH)

    for key in zipped.keys():
        np.testing.assert_array_equal(zipped[key][:], unzipped[key][:], err_msg=key)


def test_the_loaded_session_matches_the_zarr(session, zarr_buffer):
    loaded = load_session_replay_buffer(session, OUT_WH)

    assert list(loaded.episode_ends[:]) == list(zarr_buffer.episode_ends[:])
    assert set(loaded.keys()) == set(zarr_buffer.keys())
    for key in zarr_buffer.keys():
        if key == "camera0_rgb":
            continue
        np.testing.assert_array_equal(loaded[key][:], zarr_buffer[key][:], err_msg=key)
    images, expected = loaded["camera0_rgb"][:], zarr_buffer["camera0_rgb"][:]
    assert images.shape == expected.shape == (sum(e - s for s, e in SEGMENTS),
                                              OUT_WH[1], OUT_WH[0], 3)
    # The session re-encodes its range, so pixels agree up to codec noise.
    assert np.abs(images.astype(int) - expected.astype(int)).mean() < 3


def test_the_episodes_missing_a_target_are_named(slam_session, tmp_path):
    session = tmp_path / "dataset.solarian.zip"
    _run("07_generate_session.py", "-o", session, slam_session)
    start, end = SEGMENTS[1]
    write_targets(session, 1, point=np.zeros((end - start, 2), dtype=np.float32))

    assert episodes_missing_target(session, "point") == ["episodes/000000"]
    assert not has_target(session, "point")
    assert "episodes 000000 have none" in _missing_target_hint("point", "point", str(session))


def test_targets_are_cropped_to_the_policy_resolution(slam_session, tmp_path):
    session = tmp_path / "dataset.solarian.zip"
    _run("07_generate_session.py", "-o", session, slam_session)
    w, h = VIDEO_WH
    for index, (start, end) in enumerate(SEGMENTS):
        n = end - start
        mask = np.zeros((n, h, w), dtype=np.uint8)
        mask[:, 20:40, 10:30] = 1
        bbox = np.tile(np.array([10, 20, 30, 40, 0.5], dtype=np.float32), (n, 1))
        bbox[0] = -1
        point = np.tile(np.array([20, 30], dtype=np.float32), (n, 1))
        point[0] = -1
        write_targets(session, index, bbox=bbox, mask=mask, point=point)
    assert has_target(session, "point") and has_target_array(str(session), "mask")
    with pytest.raises(FileExistsError):
        write_targets(session, 0, bbox=bbox)

    loaded = load_session_replay_buffer(session, OUT_WH)

    # 64x48 to 32x24 keeps the whole frame, so pixels halve.
    np.testing.assert_allclose(loaded["bbox"][1], [5, 10, 15, 20, 0.5])
    np.testing.assert_array_equal(loaded["bbox"][0], [-1] * 5)
    np.testing.assert_allclose(loaded["point"][1], [10, 15])
    np.testing.assert_array_equal(loaded["point"][0], [-1, -1])
    masks = loaded["mask"][:]
    assert masks.shape == (loaded.n_steps, OUT_WH[1], OUT_WH[0])
    assert masks[:, 10:20, 5:15].all() and masks.sum() == loaded.n_steps * 100


def test_a_point_the_crop_cuts_off_is_no_point():
    # 64x48 to 24x24 keeps the centre 48x48, from x=8.
    frame = PolicyFrame(in_res=(64, 48), out_res=(24, 24))
    np.testing.assert_allclose(frame.point([[32, 24], [4, 24], [60, 10], [-1, -1]]),
                               [[12, 12], [-1, -1], [-1, -1], [-1, -1]])


def _shape_meta() -> dict:
    def low_dim(shape, **extra):
        return {"shape": shape, "horizon": 2, "latency_steps": 0,
                "down_sample_steps": 1, "type": "low_dim", **extra}

    return {
        "obs": {
            "camera0_rgb": {"shape": [3, OUT_WH[1], OUT_WH[0]], "horizon": 1,
                            "latency_steps": 0, "down_sample_steps": 1, "type": "rgb"},
            "robot0_eef_pos": low_dim([3]),
            "robot0_eef_rot_axis_angle": low_dim([6], raw_shape=[3], rotation_rep="rotation_6d"),
            "robot0_gripper_width": low_dim([1]),
        },
        "action": {"shape": [10], "horizon": 4, "latency_steps": 0,
                   "down_sample_steps": 1, "rotation_rep": "rotation_6d"},
    }


POSE_REPR = {"obs_pose_repr": "relative", "action_pose_repr": "relative"}


def test_umi_dataset_and_trainer_take_a_solarian_zip(session, monkeypatch):
    dataset = UmiDataset(
        shape_meta=_shape_meta(), dataset_path=str(session), val_dataset_path=str(session),
        pose_repr=POSE_REPR,
    )
    sample = dataset[0]
    assert sample["obs"]["camera0_rgb"].shape == (1, 3, OUT_WH[1], OUT_WH[0])
    assert len(dataset.get_validation_dataset()) > 0
    assert not has_target_array(str(session), "mask")

    started = []
    monkeypatch.setattr("solarian_umi.training.run_training",
                        lambda cfg, out: started.append(cfg))
    training = Path(__file__).resolve().parents[1] / "solarian_umi" / "templates" / "training"
    with initialize_config_dir(version_base=None, config_dir=str(training)):
        composed = compose(config_name="umi", overrides=[f"task.dataset_mix=[{{path:'{session}',weight:1.0}}]"])
    cfg = load_train_config(composed)[2]
    UmiTrainer(cfg, output_dir=str(session.parent / "out")).train()
    assert started


def test_a_cached_session_is_decoded_once(session, tmp_path, monkeypatch):
    first = UmiDataset(shape_meta=_shape_meta(), dataset_path=str(session),
                       cache_dir=str(tmp_path), pose_repr=POSE_REPR)

    def fail(*_a, **_k):
        raise AssertionError("the cache should be reused")

    monkeypatch.setattr(
        "solarian_umi.diffusion_policy.dataset.umi_dataset.load_session_replay_buffer",
        fail)
    second = UmiDataset(shape_meta=_shape_meta(), dataset_path=str(session),
                        cache_dir=str(tmp_path), pose_repr=POSE_REPR)

    np.testing.assert_array_equal(first.replay_buffer["camera0_rgb"][:],
                                  second.replay_buffer["camera0_rgb"][:])


def test_an_interrupted_cache_write_is_redone(session, tmp_path, monkeypatch):
    real = umi_dataset.open_replay_buffer

    def crash(dataset_path, store, out_res):
        real(dataset_path, store, out_res)
        raise RuntimeError("killed before the cache was renamed into place")

    monkeypatch.setattr(umi_dataset, "open_replay_buffer", crash)
    with pytest.raises(RuntimeError):
        UmiDataset(shape_meta=_shape_meta(), dataset_path=str(session),
                   cache_dir=str(tmp_path), pose_repr=POSE_REPR)
    assert not list(tmp_path.glob("*.zarr"))

    monkeypatch.setattr(umi_dataset, "open_replay_buffer", real)
    dataset = UmiDataset(shape_meta=_shape_meta(), dataset_path=str(session),
                         cache_dir=str(tmp_path), pose_repr=POSE_REPR)
    assert len(list(tmp_path.glob("*.zarr"))) == 1
    assert dataset.replay_buffer.n_episodes > 0
