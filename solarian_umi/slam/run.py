"""Run the copied SLAM pipeline, then write the training dataset."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from solarian_umi.slam.docker import (
    LUMI_ORB_SLAM3_IMAGE,
    OPENICC_IMAGE,
    ORB_SLAM3_IMAGE,
    ensure_image,
)

PACKAGE = Path(__file__).resolve().parent


def calibration_dir() -> Path:
    return PACKAGE / "calibration"


#: The last step's script per output format.
DATASET_SCRIPTS = {
    "solarian": "07_generate_session.py",
    "zarr": "07_generate_replay_buffer.py",
}


def pipeline_argv(
    session: Path,
    output: Path,
    calibration: Path,
    camera: str = "gopro",
    slam_settings: Path | None = None,
    dataset_format: str = "solarian",
    gripper_tag_threshold: float | None = None,
    rolling_shutter: bool = False,
    map_library: Path | None = None,
) -> tuple[list[str], list[str]]:
    """The two commands: steps 00–06, then the dataset in ``dataset_format``."""
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
    if gripper_tag_threshold is not None:
        slam.extend(["--gripper-tag-threshold", str(gripper_tag_threshold)])
    if rolling_shutter:
        slam.append("--rolling-shutter")
    if map_library is not None:
        slam.extend(["--map-library", str(map_library)])
    return (
        slam,
        [
            sys.executable,
            str(PACKAGE / "scripts" / DATASET_SCRIPTS[dataset_format]),
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
    dataset_format: str = "solarian",
    gripper_tag_threshold: float | None = None,
    rolling_shutter: bool = False,
    map_library: Path | None = None,
    assume_yes: bool = False,
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
    images = [OPENICC_IMAGE, ORB_SLAM3_IMAGE] if camera == "gopro" else [LUMI_ORB_SLAM3_IMAGE]
    for image in images:
        if assume_yes:
            ensure_image(image, input_fn=lambda _prompt: "y")
        else:
            ensure_image(image)
    slam_cmd, dataset_cmd = pipeline_argv(
        session, output, calibration, camera=camera, slam_settings=slam_settings,
        dataset_format=dataset_format, gripper_tag_threshold=gripper_tag_threshold,
        rolling_shutter=rolling_shutter, map_library=map_library,
    )
    slam = subprocess.run(slam_cmd)
    if slam.returncode != 0:
        raise SystemExit(slam.returncode)
    subprocess.run(dataset_cmd, check=True)
