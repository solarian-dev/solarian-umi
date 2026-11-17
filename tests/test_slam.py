"""umi-slam wiring and the Docker image prompt. No SLAM and no Docker."""

import subprocess
import zipfile
from pathlib import Path

import pytest

from solarian_umi.commands.slam import UmiSlamCommand, extract_lumi
from solarian_umi.slam.docker import (
    LUMI_ORB_SLAM3_IMAGE,
    OPENICC_IMAGE,
    ORB_SLAM3_IMAGE,
    ensure_image,
)
from solarian_umi.slam.run import pipeline_argv, run_session

HEADER = "Adapted from Universal Manipulation Interface."
PACKAGE = Path(__file__).resolve().parents[1] / "solarian_umi"


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
    ):
        text = (PACKAGE / rel).read_text(encoding="utf-8")
        assert text.startswith(f"# {HEADER}")
        assert "https://github.com/real-stanford/universal_manipulation_interface" in text
    # A training file opens with its run; the attribution heads the UMI part.
    for path in sorted((PACKAGE / "templates" / "training").glob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        assert f"# {HEADER}" in text
        assert text.index(f"# {HEADER}") < text.index("\nname: train_diffusion_unet_timm")


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


def test_the_rolling_shutter_is_undone_only_when_asked(tmp_path):
    plain, _dataset = pipeline_argv(tmp_path, tmp_path / "out.zip", tmp_path, camera="lumi")
    undone, _dataset = pipeline_argv(tmp_path, tmp_path / "out.zip", tmp_path, camera="lumi", rolling_shutter=True)
    assert "--rolling-shutter" not in plain
    assert undone[-1] == "--rolling-shutter"


def test_the_map_library_is_forwarded(tmp_path):
    slam, _dataset = pipeline_argv(tmp_path, tmp_path / "out.zip", tmp_path, map_library=tmp_path / "maps")
    assert slam[-2:] == ["--map-library", str(tmp_path / "maps")]


def test_a_failed_slam_stage_does_not_write_the_zarr(tmp_path, monkeypatch):
    session = tmp_path / "session"
    session.mkdir()
    calls = []

    def fake_run(cmd, check=False):
        calls.append((cmd, check))
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr("solarian_umi.slam.run.ensure_image", lambda *_a, **_k: None)
    monkeypatch.setattr("solarian_umi.slam.run.subprocess.run", fake_run)

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
    monkeypatch.setattr("solarian_umi.slam.run.ensure_image", lambda name, **_k: asked.append(name))
    monkeypatch.setattr(
        "solarian_umi.slam.run.subprocess.run",
        lambda cmd, check=False: subprocess.CompletedProcess(cmd, 1),
    )

    with pytest.raises(SystemExit):
        run_session(session, session / "dataset.zarr.zip", tmp_path, camera=camera)
    assert asked == images


def _lumi_zip(path: Path, wrap: str = "") -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{wrap}catalog.json", "{}")
        zf.writestr(f"{wrap}raw_videos/GX010001.MP4", b"mp4")
        zf.writestr(f"{wrap}raw_videos/GX010001.imu.json", "{}")
    return path


def _run_command(monkeypatch, argv):
    calls = []

    def fake_run_session(session, output, calibration, **kwargs):
        calls.append({"session": session, "output": output, **kwargs,
                      "files": sorted(p.relative_to(session).as_posix()
                                      for p in session.rglob("*") if p.is_file())})

    monkeypatch.setattr("solarian_umi.commands.slam.run_session", fake_run_session)
    cmd = UmiSlamCommand()
    cmd.parse_args(argv)
    cmd.handle(None)
    return calls[0]


@pytest.mark.parametrize("wrap", ["", "cups/"])
def test_a_lumi_zip_is_extracted_and_defaults_to_lumi(tmp_path, monkeypatch, wrap):
    archive = _lumi_zip(tmp_path / "cups.lumi.zip", wrap)
    work = tmp_path / "work"

    call = _run_command(monkeypatch, [str(archive), "--work-dir", str(work), "--yes"])

    assert call["camera"] == "lumi"
    assert call["assume_yes"] is True
    assert call["output"] == tmp_path / "cups.solarian.zip"
    assert call["session"] == work / wrap.rstrip("/")
    assert call["files"] == ["catalog.json", "raw_videos/GX010001.MP4", "raw_videos/GX010001.imu.json"]
    assert work.is_dir()


def test_a_temporary_work_dir_is_removed_after_success(tmp_path, monkeypatch):
    archive = _lumi_zip(tmp_path / "cups.lumi.zip")

    call = _run_command(monkeypatch, [str(archive), "--format", "zarr"])

    assert call["output"] == tmp_path / "cups.zarr.zip"
    assert not call["session"].exists()


def test_a_directory_keeps_gopro_and_its_own_output(tmp_path, monkeypatch):
    call = _run_command(monkeypatch, [str(tmp_path)])

    assert call["session"] == tmp_path
    assert call["camera"] == "gopro"
    assert call["output"] == tmp_path / "dataset.solarian.zip"


def test_a_zip_without_catalog_is_refused(tmp_path):
    archive = tmp_path / "x.lumi.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("session.json", "{}")
    with pytest.raises(SystemExit, match="not a lumi dataset"):
        extract_lumi(archive, tmp_path / "work")


def test_missing_image_is_pulled_only_after_yes(monkeypatch):
    pulls = []
    monkeypatch.setattr("solarian_umi.slam.docker.shutil.which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr("solarian_umi.slam.docker.image_present", lambda _name: False)
    monkeypatch.setattr(
        "solarian_umi.slam.docker.subprocess.run",
        lambda cmd, **_k: pulls.append(cmd) or subprocess.CompletedProcess(cmd, 0),
    )

    ensure_image(ORB_SLAM3_IMAGE, input_fn=lambda _prompt: "y")
    ensure_image(OPENICC_IMAGE, input_fn=lambda _prompt: "yes")

    assert pulls == [
        ["docker", "pull", ORB_SLAM3_IMAGE],
        ["docker", "pull", OPENICC_IMAGE],
    ]


def test_missing_image_is_not_pulled_when_declined(monkeypatch):
    monkeypatch.setattr("solarian_umi.slam.docker.shutil.which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr("solarian_umi.slam.docker.image_present", lambda _name: False)

    def fail(*_a, **_k):
        raise AssertionError("docker pull should not run")

    monkeypatch.setattr("solarian_umi.slam.docker.subprocess.run", fail)
    with pytest.raises(SystemExit, match="Not downloading"):
        ensure_image(ORB_SLAM3_IMAGE, input_fn=lambda _prompt: "n")


def test_a_present_image_is_not_pulled(monkeypatch):
    monkeypatch.setattr("solarian_umi.slam.docker.shutil.which", lambda _name: "/usr/bin/docker")
    monkeypatch.setattr("solarian_umi.slam.docker.image_present", lambda _name: True)

    def fail(*_a, **_k):
        raise AssertionError("docker pull should not run")

    monkeypatch.setattr("solarian_umi.slam.docker.subprocess.run", fail)
    ensure_image(ORB_SLAM3_IMAGE, input_fn=fail)
