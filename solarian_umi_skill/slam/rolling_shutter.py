"""The Lumi sensor's rolling shutter, measured from video and gyro and undone in the remap.

The IMX477 reads its rows top to bottom over about 22 ms, so in a fast turn
the bottom of a frame sees the scene a few degrees later than the top.
ORB-SLAM3 takes every pixel of a frame as seen at once. The remap renders
each frame as one instant instead: every pixel samples the native pixel that
saw its ray when the row it fell on was read, given the gyro's rotation over
the readout. Camera translation during the readout is left in; it needs depth.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from solarian_umi_skill.slam.fisheye_model import Fisheye, Remap
from solarian_umi_skill.slam.support.cam_imu import ROW_BANDS, CameraRotation, estimate_offset, gyro_over_intervals

# Band offsets are searched this close to the demo's whole-frame offset.
BAND_SEARCH_S = 0.03
MIN_BAND_CORRELATION = 0.8
MIN_TIMED_DEMOS = 5
# Output pixels between the points the correction is computed at; it varies
# smoothly over the frame and is interpolated between them.
GRID_STEP_PX = 8


@dataclass(frozen=True)
class RollingShutter:
    readout_s: float  # first row to last
    # Fraction of the frame height, from the top, whose read time the demo
    # offsets refer to. The whole-frame fit leans to the rows with most features.
    ref_row: float

    def row_delay_s(self, row: np.ndarray | float) -> np.ndarray | float:
        """Seconds after the reference row that rows (fractions of the height) were read."""
        return self.readout_s * (row - self.ref_row)


def estimate_rolling_shutter(timed: list[tuple[CameraRotation, np.ndarray, np.ndarray, float]]) -> RollingShutter | None:
    """From (camera rotation, IMU times, gyro, whole-frame offset) per demo.

    Each band of rows lines up with the gyro at its own offset, lower bands
    later. The bottom band is left out: on a Lumi the gripper fills it, and the
    nearby table moves there with translation as much as with rotation.
    """
    top = 0.5 / ROW_BANDS
    per_height, top_offsets, offsets = [], [], []
    for cam, t_imu, gyr, offset in timed:
        bands = [
            estimate_offset(
                replace(cam, omega=cam.band_omega[:, k]), t_imu, gyr,
                search=(offset - BAND_SEARCH_S, offset + BAND_SEARCH_S), step_s=0.001,
            )
            for k in (0, 1)
        ]
        if min(b.correlation for b in bands) < MIN_BAND_CORRELATION:
            continue
        # video_time = imu_time + offset, so a row read later has a smaller offset.
        per_height.append((bands[0].offset_s - bands[1].offset_s) * ROW_BANDS)
        top_offsets.append(bands[0].offset_s)
        offsets.append(offset)
    if len(per_height) < MIN_TIMED_DEMOS:
        return None
    readout = float(np.median(per_height))
    if readout <= 0:
        return None
    ref_row = float(np.median([top + (a - o) / readout for a, o in zip(top_offsets, offsets)]))
    return RollingShutter(readout_s=readout, ref_row=ref_row)


def frame_camera_rates(
    shutter: RollingShutter,
    frame_video_s: np.ndarray,
    t_imu: np.ndarray,
    gyr: np.ndarray,
    r_b_c: np.ndarray,
) -> np.ndarray:
    """(K, 3) camera-frame angular velocity over each frame's readout, rad/s.

    ``t_imu`` is on the video clock (offset applied).
    """
    centers = frame_video_s + shutter.row_delay_s(0.5)
    mean_gyro = gyro_over_intervals(t_imu, gyr, centers, np.full(len(centers), shutter.readout_s))
    # gyro = R_b_c @ camera
    return mean_gyro @ r_b_c


class ShutterRemap:
    """Per-frame remaps: ``base`` corrected for the camera turning while the rows were read."""

    def __init__(self, base: Remap, src: Fisheye, shutter: RollingShutter):
        self.base = base
        self.src = src
        self.shutter = shutter
        height, width = base.map_x.shape
        self.size = (width, height)
        grid_w, grid_h = -(-width // GRID_STEP_PX), -(-height // GRID_STEP_PX)
        # Where cv2.resize puts the samples of a grid_w x grid_h image scaled to width x height.
        gx = (np.arange(grid_w) + 0.5) * width / grid_w - 0.5
        gy = (np.arange(grid_h) + 0.5) * height / grid_h - 0.5
        xs, ys = np.meshgrid(gx.astype(np.float32), gy.astype(np.float32))
        sample = [
            cv2.remap(m, xs, ys, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            for m in (base.map_x, base.map_y)
        ]
        self.grid_shape = (grid_h, grid_w)
        self.rays, self.modelled = src.unproject(np.stack(sample, axis=-1).reshape(-1, 2).astype(np.float64))
        # Shifts are taken from where the model puts each ray, so a still camera shifts nothing.
        self.grid_uv, _ = src.project(self.rays)

    def frame(self, omega_c: np.ndarray) -> Remap:
        """The remap for a frame the camera turned through at ``omega_c`` rad/s while it was read."""
        uv = self.grid_uv
        # Two passes: the row a ray lands on sets when it was read, which moves the ray.
        for _ in range(2):
            delay = self.shutter.row_delay_s(uv[:, 1] / self.src.height)
            # A direction fixed in the world turns by -omega t in the camera frame.
            seen = _rotate(self.rays, -delay[:, None] * omega_c[None, :])
            uv, _ = self.src.project(seen)
        shift = np.where(self.modelled[:, None], uv - self.grid_uv, 0.0)
        shift = shift.reshape(*self.grid_shape, 2).astype(np.float32)
        shift = cv2.resize(shift, self.size, interpolation=cv2.INTER_LINEAR)
        return Remap(
            map_x=self.base.map_x + shift[..., 0],
            map_y=self.base.map_y + shift[..., 1],
            valid=self.base.valid,
        )


def _rotate(rays: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues rotation of each ray by its own rotation vector."""
    angle = np.linalg.norm(rotvec, axis=1, keepdims=True)
    axis = np.divide(rotvec, angle, out=np.zeros_like(rotvec), where=angle > 1e-12)
    cos, sin = np.cos(angle), np.sin(angle)
    return (
        rays * cos
        + np.cross(axis, rays) * sin
        + axis * np.sum(axis * rays, axis=1, keepdims=True) * (1 - cos)
    )
