"""Run the copied SLAM pipeline, then write the training zarr."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from solarian_umi_skill.slam.docker import (
    OPENICC_IMAGE,
    ORB_SLAM3_IMAGE,
    ensure_image,
    ensure_lumi_slam_image,
)

PACKAGE = Path(__file__).resolve().parent


def calibration_dir() -> Path:
    return PACKAGE / "calibration"


def pipeline_argv(
    session: Path,
    output: Path,
    calibration: Path,
    camera: str = "gopro",
    slam_settings: Path | None = None,
) -> tuple[list[str], list[str]]:
    """The two commands: steps 00–06, then the replay-buffer zarr."""
    slam = [
        sys.executable,
        str(PACKAGE / "run_pipeline.py"),
        str(session),
        "-c",
        str(calibration),
    ]
    if camera != "gopro":
        slam.extend(["--camera", camera])
    if slam_settings is not None:
        slam.extend(["--slam-settings", str(slam_settings)])
    return (
        slam,
        [
            sys.executable,
            str(PACKAGE / "scripts" / "07_generate_replay_buffer.py"),
            "-o",
            str(output),
            str(session),
        ],
    )


def run_session(
    session: Path,
    output: Path,
    calibration: Path | None = None,
    camera: str = "gopro",
    slam_settings: Path | None = None,
) -> None:
    if not session.is_dir():
        raise FileNotFoundError(
            f"Session directory not found: {session}. "
            "Pass a directory of GoPro or Lumi collector MP4s."
        )
    calibration = calibration or calibration_dir()
    if not calibration.is_dir():
        raise FileNotFoundError(f"Calibration directory not found: {calibration}.")
    # IMU extraction runs before mapping, so ask for that image first.
    # Lumi IMU data comes from the collector's sidecars instead.
    if camera == "gopro":
        ensure_image(OPENICC_IMAGE)
    ensure_image(ORB_SLAM3_IMAGE)
    if camera == "lumi":
        ensure_lumi_slam_image()
    slam_cmd, zarr_cmd = pipeline_argv(
        session, output, calibration, camera=camera, slam_settings=slam_settings
    )
    slam = subprocess.run(slam_cmd)
    if slam.returncode != 0:
        raise SystemExit(slam.returncode)
    subprocess.run(zarr_cmd, check=True)
