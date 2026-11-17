"""Lumi session preparation between step 00 and the GoPro SLAM steps.

Measure on the recordings each demo's camera-to-IMU time offset, one
camera-to-IMU rotation and the sensor's rolling-shutter readout; remap the
videos into GoPro geometry, optionally with the readout undone; build
imu_data.json from the sidecars, and write the ORB-SLAM3 yaml. After this, steps 02 to 07 treat the session
as GoPro footage with a 30 fps camera and a 200 Hz IMU.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from solarian_umi.slam.camera_calibration import find_calibration, load_native_model
from solarian_umi.slam.fisheye_model import Fisheye
from solarian_umi.slam.lumi_camera import MOUNT_PRIOR_R_B_C, SLAM_SIZE, SLAM_YAML_NAME, slam_yaml_text
from solarian_umi.slam.lumi_imu import (
    ImuTrack,
    demo_sidecars,
    frame_slots,
    read_sidecar,
    sidecar_frame_times,
    video_frame_count,
    write_imu_json,
)
from solarian_umi.slam.remap_video import gopro_target, native_path, remap_session
from solarian_umi.slam.rolling_shutter import estimate_rolling_shutter, frame_camera_rates
from solarian_umi.slam.support.cam_imu import (
    CameraRotation,
    OffsetEstimate,
    camera_angular_velocity,
    estimate_offset,
    estimate_rotation,
    paired_rates,
    rotation_angle_deg,
)
from solarian_umi.slam.support.mp4_meta import camera_serial

CAMERA_IMU_NAME = "camera_imu.json"
EXCLUDED_DEMOS_DIR = "excluded_demos"
# A correlation peak this far from the session's prior is a false match.
PRIOR_WINDOW_S = 0.15


@dataclass
class DemoTiming:
    demo: Path
    track: ImuTrack
    cam: CameraRotation
    # IMU span past the last frame: without sidecar frame times only the video
    # end lines up with the IMU, so a demo's offset is about (end_lag - span_excess).
    span_excess_s: float
    found: OffsetEstimate
    # Sidecar frame times put frames on the IMU clock, so every take shares one
    # offset however long the IMU runs on after the last frame.
    framed: bool
    frame_video_s: np.ndarray  # (K,) video time of each recorded frame
    offset_s: float = 0.0
    source: str = ""


def _measure(demo: Path, sidecar: Path, model: Fisheye) -> DemoTiming:
    native = native_path(demo)
    n, fps = video_frame_count(native)
    track = read_sidecar(sidecar, fps)
    frame_t = sidecar_frame_times(sidecar)
    if frame_t is None:
        frame_video_s = np.arange(n) / fps
    else:
        slots = frame_slots(frame_t, fps)
        frame_video_s = slots / fps
        # The remap fills every slot the collector dropped.
        n = int(slots[-1]) + 1
    cam = camera_angular_velocity(native, model, None if frame_t is None else frame_video_s)
    return DemoTiming(
        demo=demo,
        track=track,
        cam=cam,
        span_excess_s=float(track.t_s[-1] - (n - 1) / fps),
        found=estimate_offset(cam, track.t_s, track.gyr),
        framed=frame_t is not None,
        frame_video_s=frame_video_s,
    )


def _settle_offsets(timings: list[DemoTiming]) -> float:
    """Pick each demo's offset; returns the session's end lag in seconds."""
    reliable = [t for t in timings if t.found.reliable]
    if not reliable:
        raise SystemExit("No demo has enough camera rotation to line the video up with the gyro.")
    end_lag = float(np.median([t.found.offset_s + t.span_excess_s for t in reliable]))
    shared = float(np.median([t.found.offset_s for t in reliable]))
    for t in timings:
        prior = shared if t.framed else end_lag - t.span_excess_s
        if t.found.reliable and abs(t.found.offset_s - prior) <= PRIOR_WINDOW_S:
            t.offset_s, t.source = t.found.offset_s, "gyro"
            continue
        near = estimate_offset(t.cam, t.track.t_s, t.track.gyr, search=(prior - 0.1, prior + 0.1))
        if near.reliable:
            t.offset_s, t.source = near.offset_s, "gyro near prior"
        else:
            t.offset_s, t.source = prior, "end of video"
    return end_lag


def prepare_lumi_session(
    session: Path, gopro_intrinsics: Path, conf_dirs: list[Path], *, undo_rolling_shutter: bool = False
) -> Path:
    """Returns the ORB-SLAM3 yaml. The rolling shutter is measured either way."""
    demos = session / "demos"
    serial = camera_serial(native_path(demos / "mapping"))
    calibration = find_calibration(session, serial, conf_dirs)
    print(f"Lumi camera {serial}: {calibration}")
    pairs = demo_sidecars(session)
    missing = sorted(p.name for p in demos.iterdir() if p.is_dir() and p.resolve() not in pairs)
    if missing:
        raise SystemExit(f"No .imu.json sidecar for: {', '.join(missing)}")

    native = load_native_model(calibration)
    with ThreadPoolExecutor(max_workers=6) as pool:
        timings = list(pool.map(lambda item: _measure(item[0], item[1], native), sorted(pairs.items())))
    misread = [t for t in timings if t.track.acc_misread]
    for t in misread:
        if t.demo.name == "mapping":
            raise SystemExit(f"The mapping video's IMU did not record acceleration ({pairs[t.demo].name}).")
    if misread:
        excluded = session / EXCLUDED_DEMOS_DIR
        excluded.mkdir(exist_ok=True)
        for t in misread:
            t.demo.rename(excluded / t.demo.name)
        print(
            f"Left out {len(misread)} demo(s) whose IMU did not record acceleration "
            f"(moved to {excluded}): {', '.join(pairs[t.demo].name for t in misread)}"
        )
        timings = [t for t in timings if not t.track.acc_misread]
    end_lag = _settle_offsets(timings)

    used = [t for t in timings if t.source.startswith("gyro")]
    r_b_c, residual = estimate_rotation(
        [paired_rates(t.cam, t.track.t_s, t.track.gyr, t.offset_s) for t in used]
    )
    tilt = rotation_angle_deg(r_b_c, MOUNT_PRIOR_R_B_C)
    shutter = estimate_rolling_shutter([(t.cam, t.track.t_s, t.track.gyr, t.offset_s) for t in used])
    counts = {s: sum(t.source == s for t in timings) for s in ("gyro", "gyro near prior", "end of video")}
    print(
        f"IMU timing: video ends {end_lag * 1000:.0f} ms before the IMU stops "
        f"(offsets from gyro {counts['gyro']}, gyro near that {counts['gyro near prior']}, "
        f"from the video end {counts['end of video']}). "
        f"Camera-to-IMU rotation is {tilt:.1f} deg from the designed mount, "
        f"residual {residual:.3f} rad/s over {len(used)} demos. "
        + (
            "Too few demos time the rows apart to measure the rolling shutter."
            if shutter is None
            else f"Rows read over {shutter.readout_s * 1000:.1f} ms; offsets refer to the row "
            f"{shutter.ref_row:.0%} down the frame."
        )
    )

    undo = shutter if undo_rolling_shutter else None
    frame_rates = {}
    if undo is not None:
        frame_rates = {
            t.demo: frame_camera_rates(undo, t.frame_video_s, t.track.t_s + t.offset_s, t.track.gyr, r_b_c)
            for t in timings
        }
    remap_session(session, calibration, gopro_intrinsics, pairs, undo, frame_rates)

    for t in timings:
        write_imu_json(t.demo, t.track.shifted(t.offset_s), serial)

    # Only the noise scaling reads the rate; polling jitter moves the median a few Hz.
    imu_hz = round(float(1.0 / np.median(np.concatenate([np.diff(t.track.t_s) for t in timings]))), -1)
    _, fps = video_frame_count(demos / "mapping" / "raw_video.mp4")
    slam_camera = gopro_target(gopro_intrinsics).resized(*SLAM_SIZE)
    yaml_path = demos / "mapping" / SLAM_YAML_NAME
    yaml_path.write_text(slam_yaml_text(slam_camera, r_b_c, fps=fps, imu_hz=imu_hz), encoding="utf-8")

    (session / CAMERA_IMU_NAME).write_text(
        json.dumps(
            {
                "camera_serial": serial,
                "calibration": str(calibration),
                "r_b_c": r_b_c.tolist(),
                "degrees_from_designed_mount": tilt,
                "rotation_residual_rad_s": residual,
                "end_lag_s": end_lag,
                "rolling_shutter": None if shutter is None else {
                    "readout_s": shutter.readout_s,
                    "ref_row": shutter.ref_row,
                    "undone": undo is not None,
                },
                "demos": {
                    t.demo.name: {
                        "offset_s": t.offset_s,
                        "source": t.source,
                        "correlation": t.found.correlation,
                        "span_excess_s": t.span_excess_s,
                    }
                    for t in timings
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return yaml_path
