"""Pull the Docker images named in the original UMI SLAM scripts.

Map and tracking use ``chicheng/orb_slam3:latest``. GoPro IMU extraction
uses ``chicheng/openicc:latest``. Each video is ``docker run --rm``; nothing
is left running. An image that is already local is used as it is. A missing
image is downloaded only after the operator agrees.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ORB_SLAM3_IMAGE = "chicheng/orb_slam3:latest"
# Stock ORB-SLAM3 rejects the first second of a loaded atlas below 30 inliers.
# Lumi locks on around 26, so that second ends the clip. This image uses the
# same floor of 15 as the rest of an IMU-initialized track.
LUMI_ORB_SLAM3_IMAGE = "lumi-orb-slam3:inliers15"
OPENICC_IMAGE = "chicheng/openicc:latest"
GOPRO_MAXLENS_720 = (
    "/ORB_SLAM3/Examples/Monocular-Inertial/"
    "gopro10_maxlens_fisheye_setting_v1_720.yaml"
)
CONTAINER_SETTINGS = "/slam/settings.yaml"


def image_present(name: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def ensure_image(name: str, *, allow_pull: bool = True, input_fn=input) -> None:
    """Run *name* only if it is local, or the operator agrees to download it."""
    if shutil.which("docker") is None:
        raise FileNotFoundError(
            "Docker is not installed. ORB-SLAM3 and GoPro IMU extraction "
            f"run in {ORB_SLAM3_IMAGE} and {OPENICC_IMAGE}."
        )
    if image_present(name):
        return
    if not allow_pull:
        raise SystemExit(
            f"Docker image {name} is not on this machine, and downloading was declined."
        )
    answer = input_fn(
        f"Docker image {name} is not on this machine. "
        "Download it and run it here? [y/N] "
    )
    if str(answer).strip().lower() not in {"y", "yes"}:
        raise SystemExit(f"Not downloading {name}.")
    subprocess.run(["docker", "pull", name], check=True)


def ensure_lumi_slam_image() -> None:
    """Build the Lumi ORB-SLAM3 image locally when it is not already present."""
    if shutil.which("docker") is None:
        raise FileNotFoundError(
            "Docker is not installed. Lumi SLAM runs in "
            f"{LUMI_ORB_SLAM3_IMAGE}, built from {ORB_SLAM3_IMAGE}."
        )
    if image_present(LUMI_ORB_SLAM3_IMAGE):
        return
    if not image_present(ORB_SLAM3_IMAGE):
        raise SystemExit(
            f"Docker image {ORB_SLAM3_IMAGE} is not on this machine. "
            f"{LUMI_ORB_SLAM3_IMAGE} is built from it."
        )
    dockerfile = Path(__file__).with_name("Dockerfile.lumi")
    subprocess.run(
        [
            "docker", "build",
            "-t", LUMI_ORB_SLAM3_IMAGE,
            "-f", str(dockerfile),
            str(dockerfile.parent),
        ],
        check=True,
    )


def slam_setting_mount(settings_path: str | Path | None) -> tuple[list[str], str]:
    """Docker volume arguments, and the ``--setting`` path inside the container.

    ``None`` is the GoPro Max Lens yaml already in the image. A host file is
    bind-mounted because the container cannot see this machine's paths.
    """
    if settings_path is None:
        return [], GOPRO_MAXLENS_720
    host = Path(settings_path).expanduser().resolve()
    if not host.is_file():
        raise FileNotFoundError(f"SLAM settings yaml not found: {host}")
    return ["--volume", f"{host}:{CONTAINER_SETTINGS}:ro"], CONTAINER_SETTINGS
