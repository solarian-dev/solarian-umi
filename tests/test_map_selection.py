import json
import subprocess
from pathlib import Path

import pytest

from solarian_umi_skill.slam import map_selection
from solarian_umi_skill.slam.map_selection import MAP_SETTINGS, mapping_key, probe_demos, select_map

HEADER = "frame_idx,timestamp,state,is_lost,is_keyframe,x,y,z,q_x,q_y,q_z,q_w\n"


def _demo(demos: Path, name: str) -> Path:
    demo = demos / name
    demo.mkdir(parents=True)
    (demo / "raw_video.mp4").write_bytes(b"video")
    (demo / "imu_data.json").write_text("{}")
    return demo


def _arg(cmd, flag):
    return Path(cmd[cmd.index(flag) + 1])


def test_probes_spread_over_the_whole_session(tmp_path):
    demos = [_demo(tmp_path, f"demo_{i:03d}") for i in range(10)]
    _demo(tmp_path, "mapping")

    assert [d.name for d in probe_demos(tmp_path, 4)] == ["demo_000", "demo_002", "demo_005", "demo_007"]
    assert probe_demos(tmp_path, 20) == demos


@pytest.fixture
def fake_slam(monkeypatch):
    """Maps hold their candidate's name and ``map2`` fails to build. Demos track on the
    maps in ``passing``; ``tracked_with`` notes each map's settings when tracked."""

    class Build:
        def __init__(self, cmd):
            folder = _arg(cmd, "--input_dir")
            self.code = 1 if folder.name == "map2" else 0
            if not self.code:
                _arg(cmd, "--map_path").write_text(folder.name)
                (folder / "mapping_camera_trajectory.csv").write_text(folder.name)

        def wait(self):
            return self.code

    state = {"passing": {"map1"}, "tracked_with": {}}

    def track(cmd):
        name = _arg(cmd, "--map_path").read_text()
        state["tracked_with"][name] = (
            _arg(cmd, "--slam_settings").read_text() if "--slam_settings" in cmd else None)
        step = 0.001 if name in state["passing"] else 1.0
        for demo in _arg(cmd, "--input_dir").glob("demo_*"):
            rows = [f"{i},{i / 30:.4f},2,false,false,{step * i},0,0.3,0,0,0,1" for i in range(10)]
            (demo / "camera_trajectory.csv").write_text(HEADER + "\n".join(rows) + "\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(map_selection.subprocess, "Popen", Build)
    monkeypatch.setattr(map_selection.subprocess, "run", track)
    return state


def _session(tmp_path):
    demos = tmp_path / "demos"
    mapping = _demo(demos, "mapping")
    for i in range(6):
        _demo(demos, f"demo_{i}")
    settings = tmp_path / "lumi_slam.yaml"
    settings.write_text("session yaml")
    return mapping, demos, settings


def _library_entry(tmp_path, map_text="stored", settings_text="stored yaml"):
    entry = tmp_path / "library" / "key"
    entry.mkdir(parents=True)
    (entry / "map_atlas.osa").write_text(map_text)
    (entry / MAP_SETTINGS).write_text(settings_text)
    return entry


def test_the_map_most_probe_demos_track_on_is_kept(tmp_path, fake_slam):
    mapping, demos, settings = _session(tmp_path)

    selection = select_map(mapping, demos, tmp_path, [], settings, candidates=3, probes=4)

    assert selection["candidates"] == {"map0": 0, "map1": 4}
    assert selection["kept"] == "map1"
    assert (mapping / "map_atlas.osa").read_text() == "map1"
    assert (mapping / "mapping_camera_trajectory.csv").read_text() == "map1"
    assert (mapping / MAP_SETTINGS).read_text() == "session yaml"
    assert not list(demos.glob("demo_*/camera_trajectory.csv"))


def test_the_library_map_is_tracked_with_the_settings_it_was_built_with(tmp_path, fake_slam):
    mapping, demos, settings = _session(tmp_path)
    entry = _library_entry(tmp_path)
    fake_slam["passing"] = {"map1", "stored"}

    selection = select_map(mapping, demos, tmp_path, [], settings, candidates=3, probes=4, library_entry=entry)

    assert fake_slam["tracked_with"] == {"stored": "stored yaml", "map0": "session yaml", "map1": "session yaml"}
    # A tie keeps the map that already did well before.
    assert selection["kept"] == "library"
    assert not selection["saved_to_library"]
    assert (mapping / "map_atlas.osa").read_text() == "stored"
    assert (mapping / MAP_SETTINGS).read_text() == "stored yaml"


def test_a_new_map_that_beats_the_library_replaces_it(tmp_path, fake_slam):
    mapping, demos, settings = _session(tmp_path)
    entry = _library_entry(tmp_path)

    selection = select_map(mapping, demos, tmp_path, [], settings, candidates=3, probes=4, library_entry=entry)

    assert selection["candidates"] == {"library": 0, "map0": 0, "map1": 4}
    assert selection["saved_to_library"]
    assert (entry / "map_atlas.osa").read_text() == "map1"
    assert (entry / MAP_SETTINGS).read_text() == "session yaml"
    info = json.loads((entry / "map.json").read_text())
    assert (info["probe_demos_passing"], info["probe_demos"]) == (4, 4)
    assert sorted(p.name for p in entry.parent.iterdir()) == ["key"]


def test_the_first_map_of_a_take_starts_its_library_entry(tmp_path, fake_slam):
    mapping, demos, settings = _session(tmp_path)
    entry = tmp_path / "library" / "key"
    entry.parent.mkdir()

    select_map(mapping, demos, tmp_path, [], settings, candidates=2, probes=4, library_entry=entry)

    assert (entry / "map_atlas.osa").read_text() == "map1"


def test_no_built_map_stops_the_pipeline(tmp_path, monkeypatch):
    class Failed:
        def __init__(self, cmd):
            pass

        def wait(self):
            return 1

    monkeypatch.setattr(map_selection.subprocess, "Popen", Failed)
    demos = tmp_path / "demos"
    with pytest.raises(SystemExit, match="No candidate map"):
        select_map(_demo(demos, "mapping"), demos, tmp_path, [], None, candidates=2, probes=4)


def _keyed_session(root: Path, video: bytes, calibration: str, undone: bool) -> Path:
    mapping = root / "demos" / "mapping"
    mapping.mkdir(parents=True)
    (mapping / "raw_video_native.mp4").write_bytes(video)
    (mapping / "raw_video.mp4").write_bytes(b"remapped " + root.name.encode())
    (root / "calibration.json").write_text(calibration)
    (root / "camera_imu.json").write_text(json.dumps({
        "calibration": str(root / "calibration.json"),
        "rolling_shutter": {"readout_s": 0.02, "ref_row": 0.5, "undone": undone},
    }))
    return root


def test_a_mapping_take_has_one_key_whatever_session_it_is_in(tmp_path):
    alone = _keyed_session(tmp_path / "alone", b"take", "cal", False)
    merged = _keyed_session(tmp_path / "merged", b"take", "cal", False)
    other_take = _keyed_session(tmp_path / "other_take", b"another take", "cal", False)
    recalibrated = _keyed_session(tmp_path / "recalibrated", b"take", "new cal", False)
    undone = _keyed_session(tmp_path / "undone", b"take", "cal", True)

    assert mapping_key(alone) == mapping_key(merged)
    assert len({mapping_key(s) for s in (alone, other_take, recalibrated, undone)}) == 4
