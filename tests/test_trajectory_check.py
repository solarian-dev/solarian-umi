import pytest

from solarian_umi.slam.trajectory_check import demo_trajectories, exclude_failed, format_report, gate

HEADER = "frame_idx,timestamp,state,is_lost,is_keyframe,x,y,z,q_x,q_y,q_z,q_w\n"


def _write(demos, name, rows):
    path = demos / name / "camera_trajectory.csv"
    path.parent.mkdir(parents=True)
    lines = [
        f"{i},{i / 30:.4f},2,{'true' if lost else 'false'},false,{x},{y},{z},0,0,0,1"
        for i, (lost, x, y, z) in enumerate(rows)
    ]
    path.write_text(HEADER + "\n".join(lines) + "\n", encoding="utf-8")


def test_a_smooth_tabletop_demo_passes_and_a_diverged_one_fails(tmp_path):
    demos = tmp_path / "demos"
    _write(demos, "demo_a", [(False, 0.001 * i, 0.0, 0.3) for i in range(100)])
    # Lost frames are ignored for extent; one 2 m jump between tracked frames fails.
    _write(demos, "demo_b", [(True, 50.0, 0, 0)] + [(False, 0.0, 0.0, 0.0), (False, 2.0, 0.0, 0.0)])

    a, b = demo_trajectories(demos)
    assert a.ok and a.extent_m == pytest.approx(0.099) and a.max_step_m == pytest.approx(0.001)
    assert not b.ok and b.max_step_m == pytest.approx(2.0) and b.lost_fraction == pytest.approx(1 / 3)
    assert "1 fail" in format_report([a, b])


def test_gate_stops_when_most_demos_fail(tmp_path):
    demos = tmp_path / "demos"
    _write(demos, "demo_a", [(False, 0.0, 0.0, 0.0)] * 10)
    _write(demos, "demo_b", [(False, 0.0, 0.0, 0.0), (False, 3.0, 0.0, 0.0)])
    _write(demos, "demo_c", [(True, 0.0, 0.0, 0.0)] * 10)
    trajectories = demo_trajectories(demos)
    with pytest.raises(SystemExit, match="2 of 3"):
        gate(trajectories)
    gate(trajectories[:1])
    with pytest.raises(SystemExit, match="SLAM produced nothing"):
        gate([])


def test_failed_demos_move_out_of_the_dataset(tmp_path):
    demos, excluded = tmp_path / "demos", tmp_path / "excluded_demos"
    _write(demos, "demo_a", [(False, 0.0, 0.0, 0.0)] * 10)
    _write(demos, "demo_b", [(False, 0.0, 0.0, 0.0), (False, 3.0, 0.0, 0.0)])

    assert exclude_failed(demos, demo_trajectories(demos), excluded) == ["demo_b"]
    assert [p.name for p in demos.iterdir()] == ["demo_a"]
    assert (excluded / "demo_b" / "camera_trajectory.csv").is_file()
    assert exclude_failed(demos, demo_trajectories(demos), excluded) == []
