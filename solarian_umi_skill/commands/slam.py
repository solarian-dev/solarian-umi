"""``solarian umi-slam`` — GoPro or Lumi MP4s to a UMI training dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import DictConfig

from solarian.cli import BaseCommand, CommandRegistry

from solarian_umi_skill.slam.run import DATASET_SCRIPTS, calibration_dir, run_session

DEFAULT_OUTPUT = {"solarian": "dataset.solarian.zip", "zarr": "dataset.zarr.zip"}


@CommandRegistry.register("umi-slam")
class UmiSlamCommand(BaseCommand):
    help = "Turn a directory of GoPro or Lumi MP4s into a UMI training dataset."
    category = "Training & Data"
    uses_config = False
    description = """
    Runs the UMI SLAM pipeline shipped in this plugin, then writes the
    dataset. The input is a directory of GoPro or Lumi collector MP4s, not a
    solarian record session.

    The default output is a solarian session packed into dataset.solarian.zip:
    per episode, the trajectory as parquet and the remapped video at full
    resolution, which training decodes to the policy resolution when it
    loads. --format zarr writes the 224x224 dataset.zarr.zip instead.

    ORB-SLAM3 runs in the Docker image chicheng/orb_slam3:latest and GoPro
    IMU extraction in chicheng/openicc:latest. With --camera lumi, ORB-SLAM3
    runs in solariandev/orb_slam3:lumi-inliers15, the same image with a lower
    inlier floor after relocalization. An image that is already on
    this machine is used as it is. A missing image is downloaded only after
    you agree.
    """
    usage = ("<session_dir> [--output PATH] [--format solarian|zarr] [--calibration-dir DIR] "
             "[--camera gopro|lumi] [--slam-settings PATH] [--gripper-tag-threshold F]")
    options = {
        "session_dir": "Directory of GoPro or Lumi collector MP4s.",
        "--output PATH": "Dataset path. Defaults to <session_dir>/dataset.solarian.zip, or "
                         "<session_dir>/dataset.zarr.zip for zarr.",
        "--format solarian|zarr": "solarian (default): a .solarian.zip session with "
                                  "full-resolution video. zarr: the 224x224 UMI replay buffer.",
        "--calibration-dir DIR": "ArUco config, and GoPro intrinsics when --camera gopro. "
                                 "Defaults to the calibration files shipped in this plugin.",
        "--camera gopro|lumi": "GoPro uses the Max Lens yaml in the ORB-SLAM3 image. "
                               "Lumi remaps the videos into GoPro geometry with the camera's "
                               "calibration (<session>/camera_calibration.json or "
                               "conf/cameras/<serial>.json, see umi-calibrate-lumi), times the "
                               "IMU from the .imu.json sidecars and writes its own yaml.",
        "--slam-settings PATH": "ORB-SLAM3 yaml to mount into the container instead. "
                                "Defaults to the GoPro Max Lens yaml, or the Lumi yaml.",
        "--gripper-tag-threshold F": "Fraction of gripper calibration frames in which each finger "
                                     "tag must be detected. Defaults to 0.8.",
    }
    examples = (
        "solarian umi-slam datasets/cup_session",
        "solarian umi-slam datasets/cup_session --camera lumi",
        "solarian umi-slam datasets/cup_session --format zarr --output datasets/cup.zarr.zip",
    )

    def parse_args(self, argv: list[str]) -> list[str]:
        parser = argparse.ArgumentParser(prog="solarian umi-slam")
        parser.add_argument("session_dir")
        parser.add_argument("--output", "-o", default=None)
        parser.add_argument("--format", choices=tuple(DATASET_SCRIPTS), default="solarian")
        parser.add_argument("--calibration-dir", default=None)
        parser.add_argument("--camera", choices=("gopro", "lumi"), default="gopro")
        parser.add_argument("--slam-settings", default=None)
        parser.add_argument("--gripper-tag-threshold", type=float, default=None)
        ns = parser.parse_args(argv)
        self.session_dir = ns.session_dir
        self.output = ns.output
        self.format = ns.format
        self.calibration_dir = ns.calibration_dir
        self.camera = ns.camera
        self.slam_settings = ns.slam_settings
        self.gripper_tag_threshold = ns.gripper_tag_threshold
        return []

    def handle(self, cfg: DictConfig) -> None:
        session = Path(self.session_dir).expanduser()
        output = (
            Path(self.output).expanduser()
            if self.output
            else session / DEFAULT_OUTPUT[self.format]
        )
        calibration = (
            Path(self.calibration_dir).expanduser()
            if self.calibration_dir
            else calibration_dir()
        )
        slam_settings = (
            Path(self.slam_settings).expanduser() if self.slam_settings else None
        )
        run_session(
            session,
            output,
            calibration,
            camera=self.camera,
            slam_settings=slam_settings,
            dataset_format=self.format,
            gripper_tag_threshold=self.gripper_tag_threshold,
        )
