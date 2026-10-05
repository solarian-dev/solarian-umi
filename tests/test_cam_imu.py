import json

import cv2
import numpy as np
import pytest

from solarian_umi_skill.slam.fisheye_model import Fisheye, build_remap
from solarian_umi_skill.slam.lumi_imu import ImuTrack, frame_slots, imu_document, read_sidecar, to_video_time
from solarian_umi_skill.slam.support.cam_imu import (
    CameraRotation,
    _pair_omega,
    estimate_offset,
    estimate_rotation,
    paired_rates,
    rotation_angle_deg,
)

SMALL = Fisheye(fx=174.6, fy=174.6, cx=360.0, cy=270.0, k=(-0.0088, -0.0016, -0.00084, -0.000049), width=720, height=540)
R_B_C = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])


def _motion(t: np.ndarray) -> np.ndarray:
    """Handheld-like camera angular velocity, rad/s."""
    return np.stack(
        [
            1.2 * np.sin(2 * np.pi * 0.7 * t) + 0.4 * np.sin(2 * np.pi * 2.3 * t),
            0.9 * np.sin(2 * np.pi * 1.1 * t + 1.0),
            0.6 * np.sin(2 * np.pi * 0.4 * t + 2.0) + 0.3 * np.sin(2 * np.pi * 3.1 * t),
        ],
        axis=1,
    )


def _camera(fps: float, seconds: float) -> CameraRotation:
    dt = 1.0 / fps
    t = (np.arange(int(seconds * fps)) + 0.5) * dt
    fine = np.linspace(-dt / 2, dt / 2, 21)
    omega = np.mean([_motion(t + f) for f in fine], axis=0)
    return CameraRotation(t_s=t, omega=omega, dt_s=np.full(len(t), dt))


def test_offset_and_rotation_recovered_from_a_shifted_rotated_gyro():
    rng = np.random.default_rng(3)
    cam = _camera(30.0, 8.0)
    offset = -0.237
    # Irregular polling times, like the collector's sidecar.
    t_imu = np.cumsum(rng.uniform(0.001, 0.009, 2000)) - 0.5
    gyr = _motion(t_imu + offset) @ R_B_C.T + np.array([0.01, -0.02, 0.005])
    gyr += rng.normal(0, 0.02, gyr.shape)

    estimate = estimate_offset(cam, t_imu, gyr)
    assert estimate.reliable
    assert estimate.offset_s == pytest.approx(offset, abs=0.003)

    R, residual = estimate_rotation([paired_rates(cam, t_imu, gyr, estimate.offset_s)])
    assert rotation_angle_deg(R, R_B_C) < 1.0
    assert residual < 0.1


def test_pair_omega_measures_a_known_rotation_between_two_fisheye_frames():
    rng = np.random.default_rng(0)
    texture = cv2.GaussianBlur(rng.uniform(0, 255, (540, 720)).astype(np.uint8), (0, 0), 2.0)
    texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX)
    rvec = np.array([0.02, -0.035, 0.015])
    R, _ = cv2.Rodrigues(rvec)
    frame_b = build_remap(SMALL, SMALL, rotation_src_from_dst=R).apply(texture)
    frame_a = build_remap(SMALL, SMALL).apply(texture)

    assert np.allclose(_pair_omega(frame_a, frame_b, SMALL), rvec, atol=2e-3)


def test_frame_timestamps_put_imu_samples_on_the_video_frame_grid():
    frames = 10.0 + 0.034 * np.arange(5)  # frames 34 ms apart, nominal 30 fps
    t = np.array([9.99, 10.0, 10.017, 10.136, 10.2])
    video = to_video_time(t, frames, 30.0)
    assert video == pytest.approx([-0.01, 0.0, 0.5 / 30, 4 / 30, 4 / 30 + 0.064])
    assert to_video_time(t, None, 30.0) == pytest.approx(t - 9.99)


def test_frame_slots_skip_a_dropped_frame_without_drifting():
    frames = np.array([0.0, 0.0334, 0.0667, 0.1667, 0.2001])  # two frames missing after the third
    assert frame_slots(frames, 30.0).tolist() == [0, 1, 2, 5, 6]
    slow_clock = np.arange(300) * 0.0337  # 1% off 30 fps adds up, but never one whole step
    assert frame_slots(slow_clock, 30.0).tolist() == list(range(300))


def test_imu_after_a_dropped_frame_lands_on_the_repeated_slot():
    frames = np.array([0.0, 1 / 30, 4 / 30, 5 / 30])
    video = to_video_time(np.array([1 / 30, 2.5 / 30, 4 / 30]), frames, 30.0)
    assert video == pytest.approx([1 / 30, 2.5 / 30, 4 / 30])


def test_imu_document_starts_at_zero_and_covers_the_last_frame():
    t = np.array([-0.02, 0.012, 0.0125, 0.017, 0.022, 0.5])
    track = ImuTrack(t, np.arange(18.0).reshape(6, 3), np.arange(18.0).reshape(6, 3) / 10, np.full(6, 36.0))
    doc = imu_document(track, serial="LUMI1", n_frames=31, fps=30.0)
    gyro = doc["1"]["streams"]["GYRO"]["samples"]
    cts = np.array([s["cts"] for s in gyro])

    assert cts[0] == 0.0
    assert cts[-1] > 1000.0
    assert np.all(np.diff(cts) > 0)
    assert 12.5 not in cts  # within a millisecond of the 12 ms read
    assert gyro[0]["value"] == pytest.approx([0.3, 0.4, 0.5])
    assert gyro[-1]["value"] == pytest.approx([1.5, 1.6, 1.7])
    assert doc["1"]["CASN"] == "LUMI1"


def _write_sidecar(path, acc, gyr):
    def samples(values):
        return [{"value": list(v), "cts": 5.0 * k, "temperature [°C]": 36.0} for k, v in enumerate(values)]

    path.write_text(json.dumps({"1": {"streams": {"ACCL": {"samples": samples(acc)}, "GYRO": {"samples": samples(gyr)}}}}))


def test_sidecar_accel_is_scaled_to_read_gravity(tmp_path):
    rng = np.random.default_rng(0)
    acc = np.tile([-7.2, 7.2, -0.4], (200, 1)) + rng.normal(0, 0.05, (200, 3))
    _write_sidecar(tmp_path / "a.imu.json", acc, np.zeros((200, 3)))

    track = read_sidecar(tmp_path / "a.imu.json", 30.0)

    assert np.median(np.linalg.norm(track.acc, axis=1)) == pytest.approx(9.80665)
    assert track.acc_scale == pytest.approx(9.80665 / np.median(np.linalg.norm(acc, axis=1)))
    assert not track.acc_misread


def test_a_take_with_accel_and_gyro_swapped_is_misread(tmp_path):
    _write_sidecar(tmp_path / "a.imu.json", np.full((200, 3), 0.03), np.tile([-2.9, 3.4, -0.4], (200, 1)))

    assert read_sidecar(tmp_path / "a.imu.json", 30.0).acc_misread
