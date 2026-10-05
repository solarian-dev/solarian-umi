"""Lumi videos warped into GoPro Max Lens geometry before SLAM.

Every Lumi demo then looks like a GoPro 2.7k demo at 1440x1080: the GoPro
intrinsics, masks and SLAM settings apply unchanged, and policies trained on
it run on any camera that can be remapped the same way. Pixels the Lumi lens
does not see are black.

Each demo keeps its recording as raw_video_native.mp4; raw_video.mp4 is
replaced by the remap, with the native gpmd track (camera serial) and
timecode copied so step 06 reads the same metadata.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import cv2
import numpy as np

from solarian_umi_skill.slam.camera_calibration import load_native_model
from solarian_umi_skill.slam.fisheye_model import Fisheye, Remap, build_remap, lit_mask
from solarian_umi_skill.slam.lumi_imu import frame_slots, sidecar_frame_times

GOPRO_SIZE = (1440, 1080)
NATIVE_NAME = "raw_video_native.mp4"
COVERAGE_NAME = "gopro_coverage.png"


def gopro_target(gopro_intrinsics: Path) -> Fisheye:
    return Fisheye.from_umi_json(json.loads(gopro_intrinsics.read_text(encoding="utf-8"))).resized(*GOPRO_SIZE)


def sample_frames(video: Path, *, n: int = 60) -> list[np.ndarray]:
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        step = max(1, (stream.frames or n * 30) // n)
        return [
            frame.to_ndarray(format="rgb24")
            for index, frame in enumerate(container.decode(stream))
            if index % step == 0
        ]


def native_path(demo_dir: Path) -> Path:
    native = demo_dir / NATIVE_NAME
    return native if native.is_file() else demo_dir / "raw_video.mp4"


def session_remap(session: Path, calibration: Path, gopro_intrinsics: Path) -> Remap:
    demos = session / "demos"
    lit = lit_mask(sample_frames(native_path(demos / "mapping")))
    remap = build_remap(load_native_model(calibration), gopro_target(gopro_intrinsics), src_lit=lit)
    cv2.imwrite(str(session / COVERAGE_NAME), remap.valid.astype(np.uint8) * 255)
    return remap


def write_remapped(native: Path, out: Path, remap: Remap, frame_t_s: np.ndarray | None, *, crf: int = 12) -> None:
    """Constant-rate remap of ``native``; a frame the collector dropped repeats the one before it.

    ``frame_t_s`` is the sidecar's sensor time of each recorded frame (None
    writes every frame once).
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is required to write the remapped videos (apt install ffmpeg).")
    with av.open(str(native)) as container:
        stream = container.streams.video[0]
        # The collector's nominal rate; average_rate falls below it when frames were dropped.
        fps = stream.base_rate
        repeats = None if frame_t_s is None else np.append(np.diff(frame_slots(frame_t_s, float(fps))), 1)
        timecode = stream.metadata.get("timecode")
        width, height = remap.valid.shape[1], remap.valid.shape[0]
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
            "-framerate", str(fps), "-i", "-",
            "-i", str(native),
            "-map", "0:v:0", "-map", "1:d?",
            "-c:v", "libx264", "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-c:d", "copy", "-map_metadata", "1",
        ]
        if timecode:
            cmd += ["-timecode", timecode]
        # QuickTime, like the collector output: the MP4 muxer refuses gpmd.
        cmd += ["-f", "mov", str(out)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        decoded = 0
        try:
            for frame in container.decode(stream):
                data = remap.apply(frame.to_ndarray(format="rgb24")).tobytes()
                copies = 1 if repeats is None or decoded >= len(repeats) else int(repeats[decoded])
                for _ in range(copies):
                    proc.stdin.write(data)
                decoded += 1
        finally:
            proc.stdin.close()
            code = proc.wait()
    if code != 0:
        out.unlink(missing_ok=True)
        raise SystemExit(f"ffmpeg failed ({code}) remapping {native}")
    if repeats is not None and decoded != len(repeats):
        out.unlink(missing_ok=True)
        raise SystemExit(f"{native} has {decoded} frames but its sidecar times {len(repeats)}")


def remap_demo(demo_dir: Path, remap: Remap, frame_t_s: np.ndarray | None) -> None:
    video = demo_dir / "raw_video.mp4"
    native = demo_dir / NATIVE_NAME
    if not native.is_file():
        video.rename(native)
    elif video.is_file():
        return
    tmp = demo_dir / "raw_video.remap.tmp.mp4"
    write_remapped(native, tmp, remap, frame_t_s)
    # Step 00 and 06 date a timecode without creation_time from the mtime.
    stat = native.stat()
    os.utime(tmp, (stat.st_atime, stat.st_mtime))
    tmp.rename(video)


def remap_session(session: Path, calibration: Path, gopro_intrinsics: Path, sidecars: dict[Path, Path]) -> Remap:
    """Remap every demo in place; ``sidecars`` maps each resolved demo directory to its sidecar."""
    remap = session_remap(session, calibration, gopro_intrinsics)
    target = gopro_target(gopro_intrinsics)
    circle = build_remap(target, target).valid
    demo_dirs = sorted({p.parent for p in session.glob("demos/*/raw_video*.mp4")})
    print(
        f"Remapping {len(demo_dirs)} videos into GoPro geometry; the Lumi lens sees "
        f"{remap.valid.sum() / circle.sum():.1%} of the GoPro image circle."
    )
    # Each worker is a decode, cv2.remap and x264 pipeline of its own.
    with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 4) // 4)) as pool:
        list(pool.map(lambda d: remap_demo(d, remap, sidecar_frame_times(sidecars[d.resolve()])), demo_dirs))
    return remap
