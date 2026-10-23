"""``solarian umi-slam`` — GoPro or Lumi MP4s to a UMI training dataset."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

from omegaconf import DictConfig

from solarian.cli import BaseCommand, CommandRegistry

from solarian_umi_skill.slam.run import DATASET_SCRIPTS, calibration_dir, run_session

OUTPUT_SUFFIX = {"solarian": ".solarian.zip", "zarr": ".zarr.zip"}
LUMI_SUFFIX = ".lumi.zip"


def extract_lumi(archive: Path, dest: Path) -> Path:
    """Unpack a ``.lumi.zip`` into ``dest``; the directory holding its ``catalog.json``."""
    if not zipfile.is_zipfile(archive):
        raise SystemExit(f"{archive} is not a zip archive.")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    # Zipping the folder itself, rather than its contents, wraps everything one level down.
    roots = [dest, *(p for p in sorted(dest.iterdir()) if p.is_dir())]
    for root in roots:
        if (root / "catalog.json").is_file() and (root / "raw_videos").is_dir():
            return root
    raise SystemExit(f"{archive} is not a lumi dataset: no catalog.json and raw_videos/.")


@CommandRegistry.register("umi-slam")
class UmiSlamCommand(BaseCommand):
    help = "Turn GoPro or Lumi MP4s (a directory or a .lumi.zip) into a UMI training dataset."
    category = "Training & Data"
    uses_config = False
    description = """
    Runs the UMI SLAM pipeline shipped in this plugin, then writes the
    dataset. The input is a directory of GoPro or Lumi collector MP4s, or a
    Lumi collector .lumi.zip, not a solarian record session.

    A .lumi.zip is extracted into a work directory first, because the
    pipeline writes its intermediate files into the session. The work
    directory is a temporary one unless --work-dir is given; it is removed
    after a successful run and kept after a failed one. The camera defaults
    to lumi for a .lumi.zip.

    The default output is a solarian session packed into dataset.solarian.zip:
    per episode, the trajectory as parquet and the remapped video at full
    resolution, which training decodes to the policy resolution when it
    loads. --format zarr writes the 224x224 dataset.zarr.zip instead.

    ORB-SLAM3 runs in the Docker image chicheng/orb_slam3:latest and GoPro
    IMU extraction in chicheng/openicc:latest. With --camera lumi, ORB-SLAM3
    runs in solariandev/orb_slam3:lumi-inliers15, the same image with a lower
    inlier floor after relocalization. An image that is already on
    this machine is used as it is. A missing image is downloaded only after
    you agree, or without asking with --yes.
    """
    usage = ("<session> [--output PATH] [--format solarian|zarr] [--calibration-dir DIR] "
             "[--camera gopro|lumi] [--slam-settings PATH] [--gripper-tag-threshold F] "
             "[--rolling-shutter] [--map-library DIR] [--work-dir DIR] [--yes]")
    options = {
        "session": "Directory of GoPro or Lumi collector MP4s, or a Lumi <name>.lumi.zip.",
        "--output PATH": "Dataset path. Defaults to <session>/dataset.solarian.zip (dataset.zarr.zip "
                         "for zarr), or <name>.solarian.zip beside a .lumi.zip.",
        "--format solarian|zarr": "solarian (default): a .solarian.zip session with "
                                  "full-resolution video. zarr: the 224x224 UMI replay buffer.",
        "--calibration-dir DIR": "ArUco config, and GoPro intrinsics when --camera gopro. "
                                 "Defaults to the calibration files shipped in this plugin.",
        "--camera gopro|lumi": "Defaults to lumi for a .lumi.zip, gopro otherwise. "
                               "GoPro uses the Max Lens yaml in the ORB-SLAM3 image. "
                               "Lumi remaps the videos into GoPro geometry with the camera's "
                               "calibration (<session>/camera_calibration.json or "
                               "conf/calibrations/cameras/<serial>.json, see umi-calibrate-lumi), times the "
                               "IMU from the .imu.json sidecars and writes its own yaml.",
        "--slam-settings PATH": "ORB-SLAM3 yaml to mount into the container instead. "
                                "Defaults to the GoPro Max Lens yaml, or the Lumi yaml.",
        "--gripper-tag-threshold F": "Fraction of gripper calibration frames in which each finger "
                                     "tag must be detected. Defaults to 0.8.",
        "--rolling-shutter": "Lumi: undo the sensor's rolling shutter in the remap, with the gyro's "
                             "rotation while the rows were read. The readout is measured either way.",
        "--map-library DIR": "Keeps the best map of each mapping take here, with the settings it was "
                             "built with. A later run on the same take, such as a merged dataset that "
                             "shares it, scores that map against its new ones and keeps the better.",
        "--work-dir DIR": "Where a .lumi.zip is extracted and processed. Kept afterwards. "
                          "Defaults to a temporary directory.",
        "--yes": "Download missing Docker images without asking, for unattended runs.",
    }
    examples = (
        "solarian umi-slam datasets/cup_session",
        "solarian umi-slam datasets/cup_session --camera lumi",
        "solarian umi-slam datasets/cups.lumi.zip",
        "solarian umi-slam datasets/cup_session --format zarr --output datasets/cup.zarr.zip",
    )

    def parse_args(self, argv: list[str]) -> list[str]:
        parser = argparse.ArgumentParser(prog="solarian umi-slam")
        parser.add_argument("session")
        parser.add_argument("--output", "-o", default=None)
        parser.add_argument("--format", choices=tuple(DATASET_SCRIPTS), default="solarian")
        parser.add_argument("--calibration-dir", default=None)
        parser.add_argument("--camera", choices=("gopro", "lumi"), default=None)
        parser.add_argument("--slam-settings", default=None)
        parser.add_argument("--gripper-tag-threshold", type=float, default=None)
        parser.add_argument("--rolling-shutter", action="store_true")
        parser.add_argument("--map-library", default=None)
        parser.add_argument("--work-dir", default=None)
        parser.add_argument("--yes", "-y", action="store_true")
        ns = parser.parse_args(argv)
        self.session = ns.session
        self.output = ns.output
        self.format = ns.format
        self.calibration_dir = ns.calibration_dir
        self.camera = ns.camera
        self.slam_settings = ns.slam_settings
        self.gripper_tag_threshold = ns.gripper_tag_threshold
        self.rolling_shutter = ns.rolling_shutter
        self.map_library = ns.map_library
        self.work_dir = ns.work_dir
        self.yes = ns.yes
        return []

    def handle(self, cfg: DictConfig) -> None:
        source = Path(self.session).expanduser()
        lumi_zip = source.name.endswith(LUMI_SUFFIX)
        if lumi_zip:
            stem = source.name[: -len(LUMI_SUFFIX)]
            default_output = source.parent / f"{stem}{OUTPUT_SUFFIX[self.format]}"
            work = (
                Path(self.work_dir).expanduser()
                if self.work_dir
                else Path(tempfile.mkdtemp(prefix=f"{stem}-", suffix=".lumi"))
            )
            print(f"Extracting {source} into {work}")
            session = extract_lumi(source, work)
        else:
            default_output = source / f"dataset{OUTPUT_SUFFIX[self.format]}"
            session = source
        output = Path(self.output).expanduser() if self.output else default_output
        calibration = (
            Path(self.calibration_dir).expanduser()
            if self.calibration_dir
            else calibration_dir()
        )
        slam_settings = (
            Path(self.slam_settings).expanduser() if self.slam_settings else None
        )
        # A failure raises out of here, so a temporary work dir stays for inspection.
        run_session(
            session,
            output,
            calibration,
            camera=self.camera or ("lumi" if lumi_zip else "gopro"),
            slam_settings=slam_settings,
            dataset_format=self.format,
            gripper_tag_threshold=self.gripper_tag_threshold,
            rolling_shutter=self.rolling_shutter,
            map_library=Path(self.map_library).expanduser() if self.map_library else None,
            assume_yes=self.yes,
        )
        if lumi_zip and not self.work_dir:
            shutil.rmtree(work, ignore_errors=True)
