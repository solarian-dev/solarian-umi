"""ORB-SLAM3 settings and mask for Lumi videos remapped into GoPro geometry.

The camera section is the GoPro Max Lens model the videos were remapped to,
at the 960x720 gopro_slam tracks on. The IMU section is the Lumi's: its
rate and the camera-to-IMU rotation estimated from the demos, with the
measured chip position, and the MPU-6050's noise.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from solarian_umi.slam.fisheye_model import Fisheye
from solarian_umi.slam.support.cv_util import draw_predefined_mask

SLAM_SIZE = (960, 720)

# The GoPro file's figures, far above the MPU-6050's noise at rest (9e-5 rad/s/√Hz,
# 2e-3 m/s²/√Hz). In motion its accelerometer is worse than that: it reads about
# 3.5% over g when still. Trusting it more (6e-3 m/s²/√Hz) made demos diverge
# when the camera turned to look straight down: 17 against 8 of 148 on one session.
# The gyro figure barely matters.
NOISE_GYRO = 1.5e-3  # rad/s/√Hz
NOISE_ACC = 1.7e-2  # m/s²/√Hz
GYRO_WALK = 5.0e-5  # rad/s²/√Hz
ACC_WALK = 5.5e-3  # m/s³/√Hz

# IMU chip in the camera frame (x right, y down, z toward the scene), meters:
# 16.3 mm image-down, 2 mm toward the scene. The remap keeps the optical
# center and axes, so this holds for the GoPro-geometry video too.
IMU_IN_CAMERA_M = np.array([0.0, 0.0163, 0.002])
# Camera vector into the stored (GPMF ZXY) IMU frame as designed: stored x is
# camera z, y is camera -y, z is camera x. The estimate from the gyro is
# compared against it.
MOUNT_PRIOR_R_B_C = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
)

SLAM_YAML_NAME = "lumi_slam.yaml"
# Finger tags' depth in the camera frame, meters. The lens sits where the
# GoPro's does, so this is the GoPro gripper's value; sessions recorded with
# the lens 9 mm further forward need 0.063. Gripper width readings only keep
# tags within 8 mm of it.
FINGER_TAG_Z_M = 0.072
# Pixels this far inside the coverage edge are dropped too: features on the
# black border are fixed in the image and would vote for zero motion.
COVERAGE_MARGIN_PX = 12

# 3000 features split the 30 fps Lumi map into several; 5000 segfaults this
# build while bundle-adjusting.
_ORB_AND_VIEWER = """
ORBextractor.nFeatures: 1250
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

System.thFarPoints: 20.0

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -3.5
Viewer.ViewpointF: 500.0
Viewer.imageViewScale: 1.0
"""


def camera_origin_in_imu_frame(r_b_c: np.ndarray) -> np.ndarray:
    """Camera origin in the stored IMU frame, meters."""
    return r_b_c @ (-IMU_IN_CAMERA_M)


def slam_mask(coverage: np.ndarray) -> np.ndarray:
    """255 where ORB-SLAM3 must ignore the frame, at the video's size.

    The GoPro finger mask (drawn at 2.7k) plus everything the Lumi lens did
    not cover. The Lumi gripper has no side mirrors; masking where a GoPro's
    would be drops 9% of the image.
    """
    height, width = coverage.shape
    gopro = draw_predefined_mask(
        np.zeros((2028, 2704), dtype=np.uint8), color=255, mirror=False, gripper=False, finger=True
    )
    mask = cv2.resize(gopro, (width, height), interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((2 * COVERAGE_MARGIN_PX + 1,) * 2, np.uint8)
    uncovered = cv2.dilate((~coverage).astype(np.uint8), kernel) > 0
    mask[uncovered] = 255
    return mask


def yaml_real(value: float) -> str:
    # OpenCV YAML stores a bare integer as an int. ORB-SLAM3 float
    # settings, including IMU.Frequency, require a real.
    text = format(float(value), ".10g")
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return text


def slam_yaml_text(camera: Fisheye, r_b_c: np.ndarray, *, fps: float, imu_hz: float) -> str:
    t = camera_origin_in_imu_frame(r_b_c)
    matrix = np.eye(4)
    matrix[:3, :3] = r_b_c
    matrix[:3, 3] = t
    data = ", ".join(format(value, ".10g") for value in matrix.ravel())
    k1, k2, k3, k4 = camera.k
    return f"""%YAML:1.0

# Camera: GoPro Max Lens (gopro_intrinsics_2_7k.json) at {camera.width}x{camera.height};
# the Lumi videos are remapped into it.
# T_b_c1: camera to the stored IMU frame, rotation estimated from the gyro,
# chip at (0, 0.0163, 0.002) m in the camera frame.
File.version: "1.0"
Camera.type: "KannalaBrandt8"

Camera1.fx: {yaml_real(camera.fx)}
Camera1.fy: {yaml_real(camera.fy)}
Camera1.cx: {yaml_real(camera.cx)}
Camera1.cy: {yaml_real(camera.cy)}

Camera1.k1: {yaml_real(k1)}
Camera1.k2: {yaml_real(k2)}
Camera1.k3: {yaml_real(k3)}
Camera1.k4: {yaml_real(k4)}

Camera.width: {camera.width}
Camera.height: {camera.height}
Camera.fps: {int(round(fps))}
Camera.RGB: 1

IMU.T_b_c1: !!opencv-matrix
   rows: 4
   cols: 4
   dt: f
   data: [ {data} ]

IMU.NoiseGyro: {yaml_real(NOISE_GYRO)}
IMU.NoiseAcc: {yaml_real(NOISE_ACC)}
IMU.GyroWalk: {yaml_real(GYRO_WALK)}
IMU.AccWalk: {yaml_real(ACC_WALK)}
IMU.Frequency: {yaml_real(imu_hz)}
{_ORB_AND_VIEWER}"""


def imu_streams(imu_json: dict) -> dict:
    for value in imu_json.values():
        if isinstance(value, dict) and "streams" in value:
            return value["streams"]
    raise ValueError("imu_data.json has no IMU streams")


def pad_imu_json(imu_json: dict, n_frames: int, fps: float) -> int:
    """Repeat the last sample until one timestamp is strictly after the last frame.

    gopro_slam indexes the IMU array with no end check. A frame with no
    samples leaves the preintegration null, and PredictStateIMU segfaults.
    cts is milliseconds.
    """
    streams = imu_streams(imu_json)
    gyro = streams["GYRO"]["samples"]
    if n_frames < 2 or fps <= 0 or len(gyro) < 2:
        return 0
    cts = np.array([sample["cts"] for sample in gyro], dtype=np.float64)
    steps = np.diff(cts)
    steps = steps[steps > 0]
    if len(steps) == 0:
        return 0
    step = float(np.median(steps))
    last_frame_ms = (n_frames - 1) / fps * 1000.0
    if float(cts[-1]) > last_frame_ms:
        return 0
    series = [
        stream["samples"]
        for stream in streams.values()
        if isinstance(stream, dict) and isinstance(stream.get("samples"), list) and stream["samples"]
    ]
    added = 0
    cursor = float(cts[-1])
    while cursor <= last_frame_ms:
        cursor += step
        added += 1
        for samples in series:
            sample = dict(samples[-1])
            sample["cts"] = cursor
            samples.append(sample)
    return added


def extend_imu_through_video(video_dir: Path) -> int:
    """Keep imu_data.json covering every frame of raw_video.mp4. See pad_imu_json."""
    capture = cv2.VideoCapture(str(video_dir / "raw_video.mp4"))
    n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    imu_path = video_dir / "imu_data.json"
    imu_json = json.loads(imu_path.read_text(encoding="utf-8"))
    added = pad_imu_json(imu_json, n_frames, fps)
    if added:
        imu_path.write_text(json.dumps(imu_json), encoding="utf-8")
        print(f"Extended IMU by {added} samples through the last frame of {video_dir.name}")
    return added
