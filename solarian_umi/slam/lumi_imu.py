"""imu_data.json for Lumi demos from the collector's .imu.json sidecars.

The MP4's gpmd track packs a second of samples per payload, and OpenICC
spreads them evenly over that second. The sidecar keeps each sample's read
time on the collector's monotonic clock, which is what ORB-SLAM3 needs.

cts in the output is milliseconds of video time. The remapped video has a
frame in every slot of the fps grid, repeating the previous frame where the
collector dropped one, so frame k of the recording is at slot_k / fps.
Without frame timestamps in the sidecar, video time zero is the first IMU
sample plus an offset estimated from the gyro (see cam_imu).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np

SIDECAR_SUFFIX = ".imu.json"
# Polling reads that land within a millisecond of the previous one repeat it.
MIN_SAMPLE_GAP_S = 1e-3
GRAVITY_M_S2 = 9.80665
# Past this the take did not record acceleration at all (a misaligned FIFO
# swaps accel and gyro), rather than a sensor that reads a little off.
MAX_ACC_SCALE_ERROR = 0.2


@dataclass(frozen=True)
class ImuTrack:
    t_s: np.ndarray  # (N,) video seconds
    acc: np.ndarray  # (N, 3) m/s², stored (ZXY) axes, scaled by acc_scale
    gyr: np.ndarray  # (N, 3) rad/s, stored axes
    temp_c: np.ndarray  # (N,)
    acc_scale: float = 1.0

    def shifted(self, offset_s: float) -> "ImuTrack":
        return ImuTrack(self.t_s + offset_s, self.acc, self.gyr, self.temp_c, self.acc_scale)

    @property
    def acc_misread(self) -> bool:
        return abs(1.0 / self.acc_scale - 1.0) > MAX_ACC_SCALE_ERROR


def video_frame_count(video: Path) -> tuple[int, float]:
    """Frames and the nominal rate; a recording that dropped frames averages below it."""
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        fps = float(stream.base_rate)
        n = stream.frames or sum(1 for packet in container.demux(stream) if packet.size)
    return int(n), fps


def frame_slots(frame_t_s: np.ndarray, fps: float) -> np.ndarray:
    """Each recorded frame's place on the fps grid, as the collector counts it.

    Rounded step by step, so the sensor's clock running a little off fps
    never adds up to a phantom drop over a long take.
    """
    steps = np.maximum(1, np.rint(np.diff(frame_t_s) * fps)).astype(np.int64)
    return np.concatenate([[0], np.cumsum(steps)])


def sidecar_frame_times(path: Path) -> np.ndarray | None:
    """Sensor time of each recorded frame on the IMU clock; None for older sidecars."""
    frames = json.loads(path.read_text(encoding="utf-8"))["1"].get("frames")
    return np.array([f["t_s"] for f in frames], dtype=np.float64) if frames else None


def read_sidecar(path: Path, fps: float) -> ImuTrack:
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data["1"].get("frames")
    streams = data["1"]["streams"]
    gyro = streams["GYRO"]["samples"]
    accel = streams["ACCL"]["samples"]
    t = np.array([s["cts"] for s in gyro], dtype=np.float64) / 1000.0
    acc = np.array([s["value"] for s in accel], dtype=np.float64)
    # The MPU-6050 reads gravity ~3% high. Demos hold the gripper at much the
    # same tilt, so the error sits along gravity, and ORB-SLAM3 integrating it
    # in a 6 s demo loaded onto a map runs the pose off between keyframes.
    acc_scale = GRAVITY_M_S2 / float(np.median(np.linalg.norm(acc, axis=1)))
    return ImuTrack(
        t_s=to_video_time(t, np.array([f["t_s"] for f in frames]) if frames else None, fps),
        acc=acc * acc_scale,
        gyr=np.array([s["value"] for s in gyro], dtype=np.float64),
        temp_c=np.array([s["temperature [°C]"] for s in gyro], dtype=np.float64),
        acc_scale=acc_scale,
    )


def to_video_time(t_imu: np.ndarray, frame_t: np.ndarray | None, fps: float) -> np.ndarray:
    """IMU clock seconds to video seconds.

    With per-frame sensor times on the IMU clock (collectors that log them),
    frame k maps to its slot / fps and samples between frames interpolate, so
    late frames do not shear the IMU against the video and a dropped one
    spans the repeated slot.
    """
    if frame_t is None:
        return t_imu - t_imu[0]
    grid = frame_slots(frame_t, fps) / fps
    out = np.interp(t_imu, frame_t, grid)
    before = t_imu < frame_t[0]
    after = t_imu > frame_t[-1]
    out[before] = t_imu[before] - frame_t[0]
    out[after] = grid[-1] + (t_imu[after] - frame_t[-1])
    return out


def imu_document(track: ImuTrack, *, serial: str, n_frames: int, fps: float) -> dict:
    """OpenICC-shaped JSON covering frame 0 through the last frame.

    gopro_slam reads samples between consecutive frames with no bounds check,
    so the series starts at 0 and runs past the last frame; the ends repeat
    the nearest sample.
    """
    keep = track.t_s >= 0.0
    t, acc, gyr, temp = track.t_s[keep], track.acc[keep], track.gyr[keep], track.temp_c[keep]
    if len(t) < 2:
        raise ValueError("fewer than two IMU samples inside the video")
    gap_ok = np.concatenate([[True], np.diff(t) >= MIN_SAMPLE_GAP_S])
    t, acc, gyr, temp = t[gap_ok], acc[gap_ok], gyr[gap_ok], temp[gap_ok]
    step = float(np.median(np.diff(t)))
    head = np.arange(0.0, t[0] - 0.5 * step, step)
    end = (n_frames - 1) / fps + step
    tail = np.arange(t[-1] + step, end + step, step) if t[-1] <= end else np.empty(0)
    times = np.concatenate([head, t, tail])
    index = np.concatenate([np.zeros(len(head), int), np.arange(len(t)), np.full(len(tail), len(t) - 1)])

    def samples(values: np.ndarray) -> list[dict]:
        return [
            {"value": values[i].tolist(), "cts": float(ms), "temperature [°C]": float(temp[i])}
            for ms, i in zip(times * 1000.0, index)
        ]

    return {
        "1": {
            "streams": {
                "ACCL": {"samples": samples(acc), "name": "Accelerometer", "units": "m/s2"},
                "GYRO": {"samples": samples(gyr), "name": "Gyroscope", "units": "rad/s"},
            },
            "device name": "LUMI Collector",
            "CASN": serial,
        }
    }


def session_sidecars(session: Path) -> list[Path]:
    found = list(session.glob(f"*{SIDECAR_SUFFIX}")) + list((session / "raw_videos").rglob(f"*{SIDECAR_SUFFIX}"))
    return sorted({p.resolve(): p for p in found}.values())


def demo_sidecars(session: Path) -> dict[Path, Path]:
    """Demo directory to its sidecar, through the raw_videos symlinks step 00 left.

    The mapping and gripper calibration clips may be renamed (mapping.mp4,
    gripper_calibration.mp4) away from their sidecar's name, so each gets the
    remaining sidecar whose span best matches its video.
    """
    by_stem = {p.name[: -len(SIDECAR_SUFFIX)]: p for p in session_sidecars(session)}
    pairs: dict[Path, Path] = {}
    # A demo moved to excluded_demos leaves a dangling link. It is not paired,
    # but its sidecar stays claimed so a renamed clip cannot pick it.
    claimed: set[Path] = set()
    for link in (session / "raw_videos").rglob("*"):
        if not link.is_symlink() or link.suffix.lower() != ".mp4":
            continue
        sidecar = by_stem.get(link.stem)
        if sidecar is not None:
            claimed.add(sidecar)
            if link.exists():
                pairs[link.resolve().parent] = sidecar
    demos = session / "demos"
    renamed = [d.resolve() for d in [demos / "mapping", *sorted(demos.glob("gripper_calibration*"))]]
    spare = [p for p in by_stem.values() if p not in claimed]
    for demo in renamed:
        if not demo.is_dir() or demo in pairs or not spare:
            continue
        n, fps = video_frame_count(demo / "raw_video.mp4")
        duration = n / fps

        def mismatch(path: Path) -> float:
            gyro = json.loads(path.read_text(encoding="utf-8"))["1"]["streams"]["GYRO"]["samples"]
            return abs((gyro[-1]["cts"] - gyro[0]["cts"]) / 1000.0 - duration)

        pairs[demo] = min(spare, key=mismatch)
        spare.remove(pairs[demo])
    return pairs


def write_imu_json(demo_dir: Path, track: ImuTrack, serial: str) -> Path:
    n, fps = video_frame_count(demo_dir / "raw_video.mp4")
    out = demo_dir / "imu_data.json"
    out.write_text(json.dumps(imu_document(track, serial=serial, n_frames=n, fps=fps)), encoding="utf-8")
    return out
