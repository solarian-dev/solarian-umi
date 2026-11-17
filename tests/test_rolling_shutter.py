import numpy as np
import pytest

from solarian_umi.slam.fisheye_model import Fisheye, build_remap
from solarian_umi.slam.rolling_shutter import (
    RollingShutter,
    ShutterRemap,
    _rotate,
    estimate_rolling_shutter,
    frame_camera_rates,
)
from solarian_umi.slam.support.cam_imu import ROW_BANDS, CameraRotation

SMALL = Fisheye(fx=174.6, fy=174.6, cx=360.0, cy=270.0, k=(-0.0088, -0.0016, -0.00084, -0.000049), width=720, height=540)
R_B_C = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])


def _motion(t):
    return np.stack(
        [
            1.2 * np.sin(2 * np.pi * 0.7 * t) + 0.4 * np.sin(2 * np.pi * 2.3 * t),
            0.9 * np.sin(2 * np.pi * 1.1 * t + 1.0),
            0.6 * np.sin(2 * np.pi * 0.4 * t + 2.0) + 0.3 * np.sin(2 * np.pi * 3.1 * t),
        ],
        axis=1,
    )


def test_readout_and_reference_row_come_from_when_each_band_of_rows_turned():
    readout, ref_row, offset = 0.022, 0.25, -0.12
    dt = 1 / 30
    t = (np.arange(240) + 0.5) * dt
    rows = (np.arange(ROW_BANDS) + 0.5) / ROW_BANDS
    rng = np.random.default_rng(1)
    timed = []
    for _ in range(6):
        t_imu = np.cumsum(rng.uniform(0.003, 0.007, 1700)) - 0.5
        gyr = _motion(t_imu + offset) @ R_B_C.T + rng.normal(0, 0.01, (len(t_imu), 3))
        # A band read later shows the turn the gyro measured later.
        bands = np.stack([_motion(t + readout * (r - ref_row)) for r in rows], axis=1)
        cam = CameraRotation(t_s=t, omega=_motion(t), dt_s=np.full(len(t), dt), band_omega=bands)
        timed.append((cam, t_imu, gyr, offset))

    shutter = estimate_rolling_shutter(timed)

    assert shutter.readout_s == pytest.approx(readout, abs=0.002)
    assert shutter.ref_row == pytest.approx(ref_row, abs=0.05)


def test_too_few_timed_demos_leave_the_shutter_in():
    assert estimate_rolling_shutter([]) is None


def test_camera_rate_is_the_gyro_in_camera_axes_over_the_readout():
    shutter = RollingShutter(readout_s=0.02, ref_row=0.3)
    t_imu = np.linspace(-1.0, 2.0, 600)
    gyr = np.tile(R_B_C @ np.array([0.5, -1.0, 2.0]), (600, 1))

    rates = frame_camera_rates(shutter, np.arange(10) / 30, t_imu, gyr, R_B_C)

    assert rates == pytest.approx(np.tile([0.5, -1.0, 2.0], (10, 1)))


def test_each_pixel_samples_where_its_ray_was_when_that_row_was_read():
    shutter = RollingShutter(readout_s=0.022, ref_row=0.25)
    base = build_remap(SMALL, SMALL)
    corrected = ShutterRemap(base, SMALL, shutter)
    omega = np.array([0.8, -2.0, 0.5])

    still = corrected.frame(np.zeros(3))
    turning = corrected.frame(omega)

    assert np.array_equal(still.map_x, base.map_x) and np.array_equal(still.map_y, base.map_y)
    for x, y in [(360, 270), (200, 60), (520, 480), (100, 300)]:
        uv = np.array([[turning.map_x[y, x], turning.map_y[y, x]]], dtype=np.float64)
        delay = shutter.row_delay_s(uv[0, 1] / SMALL.height)
        ray, _ = SMALL.unproject(np.array([[base.map_x[y, x], base.map_y[y, x]]], dtype=np.float64))
        expected, _ = SMALL.project(_rotate(ray, -delay * omega[None, :]))
        assert uv == pytest.approx(expected, abs=0.05)
    # 2 rad/s about the vertical axis over most of the readout moves the bottom rows by pixels.
    assert abs(turning.map_x[500, 360] - base.map_x[500, 360]) > 3
