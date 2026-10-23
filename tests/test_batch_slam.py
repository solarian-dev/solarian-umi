"""Step 03: why a demo got no trajectory, and that a timed-out container does not outlive it. No Docker."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "solarian_umi_skill" / "slam" / "scripts" / "03_batch_slam.py"


@pytest.fixture
def batch_slam():
    spec = importlib.util.spec_from_file_location("batch_slam", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_reason_is_gopro_slams_not_the_shutdown_noise_after_it(batch_slam, tmp_path):
    stdout = tmp_path / "slam_stdout.txt"
    stdout.write_text(
        "n_lost_frames=60\n"
        "Lost tracking on 60 >= 60 frames. Terminating!\n"
        "mpLoopCloser is not finished\n\n"
    )
    assert batch_slam.failure_reason(stdout) == "Lost tracking on 60 >= 60 frames. Terminating!"


def test_without_a_reason_the_last_line_is_reported(batch_slam, tmp_path):
    stdout = tmp_path / "slam_stdout.txt"
    stdout.write_text("Atlas loaded!\npKFcur->mnId247\n")
    assert batch_slam.failure_reason(stdout) == "pKFcur->mnId247"
    assert batch_slam.failure_reason(tmp_path / "missing.txt") == ""


def test_a_timeout_kills_the_container(batch_slam, tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "timeout" in kwargs:
            raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(batch_slam.subprocess, "run", fake_run)
    result = batch_slam.runner(
        ["docker", "run", "--name", "umi-slam-x"], tmp_path,
        tmp_path / "out.txt", tmp_path / "err.txt", 1.0, "umi-slam-x")

    assert isinstance(result, subprocess.TimeoutExpired)
    assert calls[-1] == ["docker", "kill", "umi-slam-x"]
