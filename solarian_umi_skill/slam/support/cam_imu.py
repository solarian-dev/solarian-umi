"""Camera-to-IMU time offset and rotation from the video and the gyro.

The camera's angular velocity comes from frame-to-frame rotation of tracked
features. The gyro measures the same rotation in its own axes, so:

- the lag that best lines up the two angular speeds is the time offset,
  video_time = imu_time + offset;
- with the offset applied, the rotation taking camera vectors onto gyro
  vectors is the camera-to-IMU rotation (Kabsch, gyro bias removed by
  centering each demo).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np

from solarian_umi_skill.slam.fisheye_model import Fisheye

# Recordings start the encoder before the IMU thread, but the first frame
# can trail by most of a second; the IMU never starts much after the video.
OFFSET_SEARCH_S = (-1.2, 0.3)
MIN_PEAK_CORRELATION = 0.6
MIN_INLIERS = 25
# Horizontal bands of rows, top first, each measured on its own features: a
# rolling shutter captures the lower bands later.
ROW_BANDS = 3
MIN_BAND_FEATURES = 25


@dataclass(frozen=True)
class CameraRotation:
    t_s: np.ndarray  # (M,) video seconds, midway between the frame pair
    omega: np.ndarray  # (M, 3) rad/s in the camera frame; NaN where tracking failed
    dt_s: np.ndarray  # (M,) seconds between the frame pair
    # (M, ROW_BANDS, 3) the same from the features in each band of rows; None when not measured.
    band_omega: np.ndarray | None = None


def camera_angular_velocity(
    video: Path, model: Fisheye, frame_video_s: np.ndarray | None = None, *, width: int = 720
) -> CameraRotation:
    """Per frame pair: x2 = R x1 + t from the essential matrix, omega = -log(R) / dt.

    ``frame_video_s`` is each decoded frame's video time; None spaces them 1 / fps.
    """
    height = int(round(width * model.height / model.width))
    small = model.resized(width, height)
    rotations: list[tuple[np.ndarray, np.ndarray]] = []
    prev = None
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        fps = float(stream.base_rate)
        for frame in container.decode(stream):
            gray = cv2.resize(frame.to_ndarray(format="gray"), (width, height), interpolation=cv2.INTER_AREA)
            if prev is not None:
                rotations.append(_pair_omega(prev, gray, small))
            prev = gray
    n_frames = len(rotations) + 1
    frame_t = np.arange(n_frames) / fps if frame_video_s is None else frame_video_s[:n_frames]
    dt = np.diff(frame_t)
    omega = np.array([r[0] for r in rotations]).reshape(-1, 3) / dt[:, None]
    bands = np.array([r[1] for r in rotations]).reshape(-1, ROW_BANDS, 3) / dt[:, None, None]
    return CameraRotation(t_s=frame_t[:-1] + dt / 2, omega=omega, dt_s=dt, band_omega=bands)


def _pair_omega(a: np.ndarray, b: np.ndarray, model: Fisheye) -> tuple[np.ndarray, np.ndarray]:
    """Rotation vector of the camera from frame a to frame b, radians (NaN if unknown),
    and the (ROW_BANDS, 3) rotation-only fit to the features in each band of rows."""
    nan = np.full(3, np.nan)
    no_bands = np.full((ROW_BANDS, 3), np.nan)
    pa = cv2.goodFeaturesToTrack(a, maxCorners=400, qualityLevel=0.01, minDistance=12, blockSize=7)
    if pa is None or len(pa) < MIN_INLIERS:
        return nan, no_bands
    pb, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, pa, None, winSize=(21, 21), maxLevel=3)
    back, ok_back, _ = cv2.calcOpticalFlowPyrLK(b, a, pb, None, winSize=(21, 21), maxLevel=3)
    good = (ok.ravel() == 1) & (ok_back.ravel() == 1) & (np.linalg.norm(back - pa, axis=2).ravel() < 0.5)
    rows = pa.reshape(-1, 2)[good, 1]
    ra, va = model.unproject(pa.reshape(-1, 2)[good])
    rb, vb = model.unproject(pb.reshape(-1, 2)[good])
    bands = no_bands.copy()
    band = np.minimum((rows / model.height * ROW_BANDS).astype(int), ROW_BANDS - 1)
    for k in range(ROW_BANDS):
        sel = va & vb & (band == k)
        if sel.sum() >= MIN_BAND_FEATURES:
            bands[k] = _trimmed_rotation(ra[sel], rb[sel])
    # Normalized pinhole points need rays in front of the camera.
    use = va & vb & (ra[:, 2] > 0.15) & (rb[:, 2] > 0.15)
    if use.sum() < MIN_INLIERS:
        return nan, bands
    na = ra[use, :2] / ra[use, 2:]
    nb = rb[use, :2] / rb[use, 2:]
    E, inliers = cv2.findEssentialMat(na, nb, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=1.0 / model.fx)
    if E is None or E.shape != (3, 3) or inliers is None or inliers.sum() < MIN_INLIERS:
        return _rotation_only(ra[use], rb[use]), bands
    _, R, _, pose_inliers = cv2.recoverPose(E, na, nb, np.eye(3), mask=inliers.copy())
    if int((pose_inliers > 0).sum()) < MIN_INLIERS:
        return _rotation_only(ra[use], rb[use]), bands
    rvec, _ = cv2.Rodrigues(R)
    return -rvec.ravel(), bands


def _rotation_only(ra: np.ndarray, rb: np.ndarray) -> np.ndarray:
    """Kabsch rb = R ra, for pairs too still for an essential matrix."""
    R = kabsch(ra, rb)
    rvec, _ = cv2.Rodrigues(R)
    return -rvec.ravel()


def _trimmed_rotation(ra: np.ndarray, rb: np.ndarray) -> np.ndarray:
    """Rotation-only fit refitted without the features it explains worst.

    A band has too few features for RANSAC, and corners that stay put (the
    image circle's edge, the gripper) pull a plain fit toward no rotation.
    """
    R = kabsch(ra, rb)
    err = np.linalg.norm(rb - ra @ R.T, axis=1)
    keep = err <= 2.0 * np.median(err)
    if keep.sum() >= MIN_BAND_FEATURES:
        R = kabsch(ra[keep], rb[keep])
    rvec, _ = cv2.Rodrigues(R)
    return -rvec.ravel()


def kabsch(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rotation R minimizing |dst - R src|, for (N, 3) arrays."""
    H = src.T @ dst
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def gyro_over_intervals(t_imu: np.ndarray, gyr: np.ndarray, centers: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Mean gyro vector over [c - dt/2, c + dt/2], comparable to one frame pair."""
    cum = np.concatenate(
        [np.zeros((1, 3)), np.cumsum(0.5 * (gyr[1:] + gyr[:-1]) * np.diff(t_imu)[:, None], axis=0)]
    )
    lo = np.stack([np.interp(centers - dt / 2, t_imu, cum[:, i]) for i in range(3)], axis=1)
    hi = np.stack([np.interp(centers + dt / 2, t_imu, cum[:, i]) for i in range(3)], axis=1)
    return (hi - lo) / dt[:, None]


@dataclass(frozen=True)
class OffsetEstimate:
    offset_s: float
    correlation: float

    @property
    def reliable(self) -> bool:
        return self.correlation >= MIN_PEAK_CORRELATION


def estimate_offset(
    cam: CameraRotation,
    t_imu: np.ndarray,
    gyr: np.ndarray,
    search: tuple[float, float] = OFFSET_SEARCH_S,
    step_s: float = 0.002,
) -> OffsetEstimate:
    """Offset maximizing the correlation of camera and gyro angular speed."""
    ok = np.isfinite(cam.omega).all(axis=1)
    if ok.sum() < 20:
        return OffsetEstimate(0.0, 0.0)
    centers = cam.t_s[ok]
    dt = cam.dt_s[ok]
    speed_cam = np.linalg.norm(cam.omega[ok], axis=1)
    lags = np.arange(search[0], search[1] + step_s / 2, step_s)
    scores = np.full(len(lags), -1.0)
    for i, lag in enumerate(lags):
        # A frame at video time c saw the gyro at imu time c - lag.
        imu_times = centers - lag
        inside = (imu_times - dt / 2 >= t_imu[0]) & (imu_times + dt / 2 <= t_imu[-1])
        if inside.sum() < 20:
            continue
        speed_imu = np.linalg.norm(gyro_over_intervals(t_imu, gyr, imu_times[inside], dt[inside]), axis=1)
        a, b = speed_cam[inside], speed_imu
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        scores[i] = float(np.corrcoef(a, b)[0, 1])
    best = int(np.argmax(scores))
    offset = float(lags[best])
    if 0 < best < len(lags) - 1:
        y0, y1, y2 = scores[best - 1 : best + 2]
        denom = y0 - 2 * y1 + y2
        if denom < 0:
            offset += 0.5 * (y0 - y2) / denom * step_s
    return OffsetEstimate(offset, float(scores[best]))


def paired_rates(cam: CameraRotation, t_imu: np.ndarray, gyr: np.ndarray, offset_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Camera and gyro angular velocity for the same frame pairs, both centered."""
    ok = np.isfinite(cam.omega).all(axis=1)
    imu_times = cam.t_s - offset_s
    ok &= (imu_times - cam.dt_s / 2 >= t_imu[0]) & (imu_times + cam.dt_s / 2 <= t_imu[-1])
    w_cam = cam.omega[ok]
    w_imu = gyro_over_intervals(t_imu, gyr, imu_times[ok], cam.dt_s[ok])
    return w_cam - w_cam.mean(axis=0), w_imu - w_imu.mean(axis=0)


def estimate_rotation(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, float]:
    """R_b_c with gyro = R_b_c @ camera, and the RMS residual in rad/s."""
    w_cam = np.concatenate([p[0] for p in pairs])
    w_imu = np.concatenate([p[1] for p in pairs])
    R = kabsch(w_cam, w_imu)
    residual = float(np.sqrt(np.mean(np.sum((w_imu - w_cam @ R.T) ** 2, axis=1))))
    return R, residual


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
