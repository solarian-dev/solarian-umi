"""ORB-SLAM3 settings and mask for remapped Lumi videos. No SLAM."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from solarian_umi_skill.slam.docker import CONTAINER_SETTINGS, GOPRO_MAXLENS_720, slam_setting_mount
from solarian_umi_skill.slam.fisheye_model import Fisheye
from solarian_umi_skill.slam.lumi_camera import (
    MOUNT_PRIOR_R_B_C,
    SLAM_SIZE,
    camera_origin_in_imu_frame,
    pad_imu_json,
    slam_mask,
    slam_yaml_text,
)
from solarian_umi_skill.slam.run import calibration_dir


def test_camera_origin_is_the_measured_mount():
    origin = camera_origin_in_imu_frame(MOUNT_PRIOR_R_B_C)
    assert origin == pytest.approx(np.array([-0.002, 0.0163, 0.0]))


def test_mask_is_the_gopro_mask_plus_the_uncovered_border():
    coverage = np.zeros((1080, 1440), dtype=bool)
    cv2.circle(coverage.view(np.uint8), (720, 540), 500, 1, thickness=-1)
    mask = slam_mask(coverage)
    assert mask.shape == (1080, 1440)
    assert mask[540, 720] == 0
    assert mask[540, 720 + 495] == 255  # inside coverage, within the edge margin
    assert mask[540, 720 + 450] == 0
    assert mask[0, 0] == 255
    assert mask[1060, 720] == 255  # GoPro finger mask


def test_yaml_is_the_gopro_camera_with_the_lumi_imu(tmp_path):
    gopro = json.loads((calibration_dir() / "gopro_intrinsics_2_7k.json").read_text(encoding="utf-8"))
    camera = Fisheye.from_umi_json(gopro).resized(*SLAM_SIZE)
    tilt = cv2.Rodrigues(np.array([0.0, 0.4, 0.0]))[0]
    r_b_c = MOUNT_PRIOR_R_B_C @ tilt
    text = slam_yaml_text(camera, r_b_c, fps=30.0, imu_hz=200)

    path = tmp_path / "lumi_slam.yaml"
    path.write_text(text, encoding="utf-8")
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    assert storage.getNode("Camera1.fx").real() == pytest.approx(282.9, abs=0.1)
    assert storage.getNode("Camera.width").real() == 960
    assert storage.getNode("Camera.fps").isInt()
    assert storage.getNode("Camera.fps").real() == 30
    assert storage.getNode("IMU.Frequency").isReal()
    assert storage.getNode("IMU.NoiseGyro").real() == pytest.approx(0.0003)
    t_b_c = storage.getNode("IMU.T_b_c1").mat()
    storage.release()
    assert t_b_c[:3, :3] == pytest.approx(r_b_c, abs=1e-6)
    assert t_b_c[:3, 3] == pytest.approx(r_b_c @ -np.array([0.0, 0.0163, 0.002]), abs=1e-6)


def test_pad_imu_json_covers_frames_past_the_last_sample():
    def samples(cts):
        return [{"value": [1.0, 2.0, 3.0], "cts": t} for t in cts]

    # 30 fps, 10 frames: last frame is at 9/30 s. Samples stop at 100 ms.
    imu = {
        "1": {
            "streams": {
                "ACCL": {"samples": samples([0.0, 5.0, 100.0])},
                "GYRO": {"samples": samples([0.0, 5.0, 100.0])},
            }
        }
    }
    added = pad_imu_json(imu, n_frames=10, fps=30.0)
    gyro = imu["1"]["streams"]["GYRO"]["samples"]
    assert added > 0
    assert gyro[-1]["cts"] / 1000.0 > 9 / 30
    assert gyro[-1]["value"] == [1.0, 2.0, 3.0]
    assert len(imu["1"]["streams"]["ACCL"]["samples"]) == len(gyro)
    assert pad_imu_json(imu, n_frames=10, fps=30.0) == 0


def test_slam_setting_mount(tmp_path: Path):
    volume, setting = slam_setting_mount(None)
    assert volume == []
    assert setting == GOPRO_MAXLENS_720

    host = tmp_path / "lumi_slam.yaml"
    host.write_text("%YAML:1.0\n", encoding="utf-8")
    volume, setting = slam_setting_mount(host)
    assert volume == ["--volume", f"{host.resolve()}:{CONTAINER_SETTINGS}:ro"]
    assert setting == CONTAINER_SETTINGS

    with pytest.raises(FileNotFoundError):
        slam_setting_mount(tmp_path / "missing.yaml")
