"""umi-slam wiring and the Docker image prompt. No SLAM and no Docker."""

import subprocess
from pathlib import Path

import pytest

from solarian_umi_skill.slam.docker import (
    LUMI_ORB_SLAM3_IMAGE,
    OPENICC_IMAGE,
    ORB_SLAM3_IMAGE,
    ensure_image,
)
from solarian_umi_skill.slam.run import pipeline_argv, run_session

HEADER = "Adapted from Universal Manipulation Interface."
PACKAGE = Path(__file__).resolve().parents[1] / "solarian_umi_skill"


def test_copied_files_carry_the_umi_attribution():
    slam_license = (PACKAGE / "slam" / "LICENSE.umi").read_text(encoding="utf-8")
    train_license = (PACKAGE / "diffusion_policy" / "LICENSE.umi").read_text(encoding="utf-8")
    assert slam_license == train_license
    assert slam_license.startswith("MIT License")
    assert "Columbia Artificial Intelligence and Robotics Lab" in slam_license
    for rel in (
        "slam/run_pipeline.py",
        "slam/scripts/07_generate_replay_buffer.py",
        "diffusion_policy/workspace/train_diffusion_unet_image_workspace.py",
        "diffusion_policy/config/train_diffusion_unet_timm_umi_cup_wild.yaml",
    ):
        text = (PACKAGE / rel).read_text(encoding="utf-8")
        assert text.startswith(f"# {HEADER}")
        assert "https://github.com/real-stanford/universal_manipulation_interface" in text


def test_the_default_dataset_is_a_solarian_session(tmp_path):
    session = tmp_path / "cup_session"
    calibration = tmp_path / "calibration"
    slam, dataset = pipeline_argv(session, session / "dataset.solarian.zip", calibration)

    assert slam[1].endswith("run_pipeline.py")
    assert slam[2:] == [str(session), "-c", str(calibration)]
    assert dataset[1].endswith("07_generate_session.py")
    assert dataset[2:] == ["-o", str(session / "dataset.solarian.zip"), str(session)]


def test_format_zarr_writes_the_replay_buffer(tmp_path):
    session = tmp_path / "cup_session"
    _slam, zarr = pipeline_argv(
        session, session / "dataset.zarr.zip", tmp_path, dataset_format="zarr")

    assert zarr[1].endswith("07_generate_replay_buffer.py")
    assert zarr[2:] == ["-o", str(session / "dataset.zarr.zip"), str(session)]


def test_lumi_camera_and_slam_settings_are_forwarded(tmp_path):
    session = tmp_path / "lumi_session"
    calibration = tmp_path / "calibration"
    settings = tmp_path / "lumi_slam.yaml"
    slam, _zarr = pipeline_argv(
        session,
        session / "dataset.zarr.zip",
        calibration,
        camera="lumi",
        slam_settings=settings,
    )
    assert slam[2:] == [
        str(session),
        "-c",
        str(calibration),
        "--camera",
        "lumi",
        "--slam-settings",
        str(settings),
    ]


def test_the_gripper_tag_threshold_is_forwarded(tmp_path):
    slam, _dataset = pipeline_argv(tmp_path, tmp_path / "out.zip", tmp_path, gripper_tag_threshold=0.7)
    assert slam[-2:] == ["--gripper-tag-threshold", "0.7"]


def test_a_failed_slam_stage_does_not_write_the_zarr(tmp_path, monkeypatch):
    session = tmp_path / "session"
    session.mkdir()
    calls = []

    def fake_run(cmd, check=False):
        calls.append((cmd, check))
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr("solarian_umi_skill.slam.run.ensure_image", lambda *_a, **_k: None)
    monkeypatch.setattr("solarian_umi_skill.slam.run.subprocess.run", fake_run)

    with pytest.raises(SystemExit) as raised:
        run_session(session, session / "dataset.zarr.zip", tmp_path)
    assert raised.value.code == 1
    assert len(calls) == 1
    assert calls[0][0][1].endswith("run_pipeline.py")


@pytest.mark.parametrize(
    ("camera", "images"),
    [
        ("gopro", [OPENICC_IMAGE, ORB_SLAM3_IMAGE]),
        ("lumi", [LUMI_ORB_SLAM3_IMAGE]),
    ],
)
def test_each_camera_asks_only_for_its_images(tmp_path, monkeypatch, camera, images):
    session = tmp_path / "session"
    session.mkdir()
    asked = []
    monkeypatch.setattr("solarian_umi_skill.slam.run.ensure_image", lambda name, **_k: asked.append(name))
    monkeypatch.setattr(
        "solarian_umi_skill.slam.run.subprocess.run",
        lambda cmd, check=False: subprocess.CompletedProcess(cmd, 1),
    )

    with pytest.raises(SystemExit):
        run_session(session, session / "dataset.zarr.zip", tmp_path, camera=camera)
    assert asked == images


def test_missing_image_is_pulled_only_after_yes(monkeypatch):
    pulls = []
    monkeypatch.setattr("solarian_umi_skill.slam.docker.shutil.which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr("solarian_umi_skill.slam.docker.image_present", lambda _name: False)
    monkeypatch.setattr(
        "solarian_umi_skill.slam.docker.subprocess.run",
        lambda cmd, **_k: pulls.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )

    ensure_image(ORB_SLAM3_IMAGE, input_fn=lambda _prompt: "y")
    ensure_image(OPENICC_IMAGE, input_fn=lambda _prompt: "yes")

    assert pulls == [
        ["docker", "pull", ORB_SLAM3_IMAGE],
        ["docker", "pull", OPENICC_IMAGE],
    ]


def test_missing_image_is_not_pulled_when_declined(monkeypatch):
    monkeypatch.setattr("solarian_umi_skill.slam.docker.shutil.which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr("solarian_umi_skill.slam.docker.image_present", lambda _name: False)

    def fail(*_a, **_k):
        raise AssertionError("docker pull should not run")

    monkeypatch.setattr("solarian_umi_skill.slam.docker.subprocess.run", fail)
    with pytest.raises(SystemExit, match="Not downloading"):
        ensure_image(ORB_SLAM3_IMAGE, input_fn=lambda _prompt: "n")


def test_a_present_image_is_not_pulled(monkeypatch):
    monkeypatch.setattr("solarian_umi_skill.slam.docker.shutil.which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr("solarian_umi_skill.slam.docker.image_present", lambda _name: True)

    def fail(*_a, **_k):
        raise AssertionError("docker pull should not run")

    monkeypatch.setattr("solarian_umi_skill.slam.docker.subprocess.run", fail)
    ensure_image(ORB_SLAM3_IMAGE, input_fn=fail)
