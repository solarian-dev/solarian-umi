"""Build several ORB-SLAM3 maps from the mapping video and keep the one demos track on best.

Map building is not repeatable: on one Lumi session, five atlases from the
same mapping video let 2, 70, 94, 95 and 103 of 148 demos pass the trajectory
check, and running a demo again on the same atlas recovers few. A map whose
visual-inertial initialization went wrong fails nearly every demo. Each
candidate is scored by tracking an evenly spaced sample of demos on it, which
ranked those five atlases in the same order as tracking every demo.

Good maps are rare enough that a map library keeps the best map of each
mapping take, so a later run on the same take (a merged dataset that shares
it, a rerun) starts from that map instead of hoping to draw as good a one.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from solarian_umi_skill.slam.lumi_prepare import CAMERA_IMU_NAME
from solarian_umi_skill.slam.remap_video import native_path
from solarian_umi_skill.slam.trajectory_check import demo_trajectories

CANDIDATES_DIR = "map_candidates"
MAP_NAME = "map_atlas.osa"
# The ORB-SLAM3 yaml a map was built with. Demos must track with the same one: on
# a map built with another camera-to-IMU rotation, 0.85 deg away cost about a
# fifth of the demos that passed, 2.5 deg nearly all of them.
MAP_SETTINGS = "map_slam_settings.yaml"
# Step 02 writes these next to the atlas; the kept candidate's replace the mapping dir's.
MAP_OUTPUTS = (MAP_NAME, "mapping_camera_trajectory.csv", "slam_stdout.txt", "slam_stderr.txt", MAP_SETTINGS)
LIBRARY_CANDIDATE = "library"
LIBRARY_INFO = "map.json"
SELECTION_NAME = "map_selection.json"
DEMO_INPUTS = ("raw_video.mp4", "imu_data.json")


@dataclass
class Candidate:
    folder: Path
    score: int = 0

    @property
    def settings(self) -> Path | None:
        """None: the yaml in the Docker image."""
        path = self.folder / MAP_SETTINGS
        return path if path.is_file() else None


def _link_inputs(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in DEMO_INPUTS:
        if not (dst / name).exists():
            os.link(src / name, dst / name)


def _settings_option(settings: Path | None) -> list[str]:
    return [] if settings is None else ["--slam_settings", str(settings)]


def _copy_map(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in MAP_OUTPUTS:
        (dst / name).unlink(missing_ok=True)
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)


def mapping_key(session: Path) -> str:
    """The mapping take and the remap it went through; maps built from one key are interchangeable."""
    digest = hashlib.sha256()
    with native_path(session / "demos" / "mapping").open("rb") as video:
        for chunk in iter(lambda: video.read(1 << 20), b""):
            digest.update(chunk)
    camera_imu = session / CAMERA_IMU_NAME
    if camera_imu.is_file():
        fit = json.loads(camera_imu.read_text(encoding="utf-8"))
        digest.update(Path(fit["calibration"]).read_bytes())
        shutter = fit["rolling_shutter"]
        if shutter is not None and shutter["undone"]:
            digest.update(b"rolling shutter undone")
    return digest.hexdigest()[:16]


def probe_demos(demo_dir: Path, count: int) -> list[Path]:
    """``count`` demos spread evenly over the session, so each part of the recording counts."""
    demos = sorted(d for d in demo_dir.glob("demo_*") if (d / "raw_video.mp4").is_file())
    if len(demos) <= count:
        return demos
    return [demos[i * len(demos) // count] for i in range(count)]


def _save_to_library(candidate: Candidate, entry: Path, info: dict) -> None:
    """Replace ``entry`` whole, so a run reading it never sees half of two maps."""
    staged = entry.with_name(entry.name + ".new")
    shutil.rmtree(staged, ignore_errors=True)
    _copy_map(candidate.folder, staged)
    (staged / LIBRARY_INFO).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    old = entry.with_name(entry.name + ".old")
    if entry.exists():
        shutil.rmtree(old, ignore_errors=True)
        entry.rename(old)
    staged.rename(entry)
    shutil.rmtree(old, ignore_errors=True)


def select_map(
    mapping_dir: Path,
    demo_dir: Path,
    scripts_dir: Path,
    slam_options: list[str],
    settings: Path | None,
    *,
    candidates: int,
    probes: int,
    library_entry: Path | None = None,
) -> dict:
    """Build ``candidates`` atlases side by side with ``settings``, score each and the map
    library's for this mapping take, and copy the best into ``mapping_dir`` with the yaml
    it was built with. A new map that wins replaces the library's.

    Returns what the selection saw, for the session's ``map_selection.json``.
    """
    root = mapping_dir / CANDIDATES_DIR
    builds = []
    for k in range(candidates):
        folder = root / f"map{k}"
        _link_inputs(mapping_dir, folder)
        if settings is not None:
            shutil.copy2(settings, folder / MAP_SETTINGS)
        builds.append(subprocess.Popen([
            sys.executable, str(scripts_dir / "02_create_map.py"),
            "--input_dir", str(folder), "--map_path", str(folder / MAP_NAME),
            *_settings_option(settings), *slam_options,
        ]))
    built = [Candidate(root / f"map{k}") for k, build in enumerate(builds) if build.wait() == 0]
    built = [c for c in built if (c.folder / MAP_NAME).is_file()]
    # First, so it keeps its place on a tie.
    if library_entry is not None and (library_entry / MAP_NAME).is_file():
        stored = Candidate(root / LIBRARY_CANDIDATE)
        _copy_map(library_entry, stored.folder)
        built.insert(0, stored)
    if not built:
        raise SystemExit("No candidate map was built.")

    sample = probe_demos(demo_dir, probes)
    for candidate in built:
        probe_dir = candidate.folder / "probe"
        for demo in sample:
            _link_inputs(demo, probe_dir / demo.name)
        result = subprocess.run([
            sys.executable, str(scripts_dir / "03_batch_slam.py"),
            "--input_dir", str(probe_dir), "--map_path", str(candidate.folder / MAP_NAME),
            *_settings_option(candidate.settings), *slam_options,
        ])
        if result.returncode != 0:
            raise SystemExit(result.returncode)
        candidate.score = sum(t.ok for t in demo_trajectories(probe_dir))

    best = max(built, key=lambda c: c.score)
    _copy_map(best.folder, mapping_dir)
    saved = library_entry is not None and best.folder.name != LIBRARY_CANDIDATE
    if saved:
        _save_to_library(best, library_entry, {
            "built_for": str(demo_dir.parent),
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "probe_demos_passing": best.score,
            "probe_demos": len(sample),
        })
    print(
        "Map candidates, probe demos passing out of "
        f"{len(sample)}: {', '.join(f'{c.folder.name} {c.score}' for c in built)}. Keeping {best.folder.name}."
        + (f" Saved it to the map library ({library_entry})." if saved else "")
    )
    return {
        "probe_demos": [d.name for d in sample],
        "candidates": {c.folder.name: c.score for c in built},
        "kept": best.folder.name,
        "map_library_entry": None if library_entry is None else str(library_entry),
        "saved_to_library": saved,
    }
