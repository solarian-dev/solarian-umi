"""Step 00: which clip becomes the map and which the gripper calibration. No video decoding."""

import datetime
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "solarian_umi_skill" / "slam" / "scripts" / "00_process_videos.py"
START = datetime.datetime(2026, 10, 14, 18, 0, 0)


@pytest.fixture
def process_videos(monkeypatch):
    spec = importlib.util.spec_from_file_location("process_videos", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # File name GX01.MP4 starts 1 s after START, GX02.MP4 2 s, and so on.
    monkeypatch.setattr(
        module,
        "mp4_get_start_datetime",
        lambda path: START + datetime.timedelta(seconds=int(Path(path).read_bytes()[:2])),
    )
    monkeypatch.setattr(module, "camera_serial", lambda _path: "LUMI0001")
    return lambda session: module.main.main([str(session)], standalone_mode=False)


def write_clip(path: Path, start_s: int, size: int) -> None:
    path.write_bytes(f"{start_s:02d}".encode() + b"x" * size)


def demo_dirs(session: Path) -> dict[str, str]:
    """Demo directory name prefix to the raw_videos link that points into it."""
    return {
        link.resolve().parent.name.split("_LUMI")[0]: link.name
        for link in (session / "raw_videos").rglob("*")
        if link.is_symlink()
    }


def test_without_named_clips_the_largest_maps_and_the_earliest_calibrates(tmp_path, process_videos):
    write_clip(tmp_path / "GX01.MP4", 1, 10)
    write_clip(tmp_path / "GX02.MP4", 2, 100)
    write_clip(tmp_path / "GX03.MP4", 3, 10)
    process_videos(tmp_path)

    links = demo_dirs(tmp_path)
    assert links["mapping"] == "mapping.mp4"
    assert links["gripper_calibration"] == "GX01.MP4"
    assert links["demo"] == "GX03.MP4"


def test_named_clips_win_over_size_and_start_time(tmp_path, process_videos):
    write_clip(tmp_path / "GX01.MP4", 1, 100)
    write_clip(tmp_path / "MAPPING.MP4", 2, 10)
    write_clip(tmp_path / "gripper_calibration.MP4", 3, 10)
    process_videos(tmp_path)

    assert (tmp_path / "raw_videos" / "mapping.mp4").read_bytes()[:2] == b"02"
    assert (tmp_path / "raw_videos" / "gripper_calibration" / "gripper_calibration.mp4").read_bytes()[:2] == b"03"
    assert sorted(p.name for p in (tmp_path / "demos").iterdir()) == [
        "demo_LUMI0001_2026.10.14_18.00.01.000000",
        "gripper_calibration_LUMI0001_2026.10.14_18.00.03.000000",
        "mapping",
    ]
