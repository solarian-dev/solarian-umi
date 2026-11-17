"""Cup-arrange re-infers on UMI's steps_per_inference grid, not remaining horizon."""

import time
from types import SimpleNamespace

import numpy as np
import pytest

from solarian_umi.skill import UMISkill


def _skill(*, steps_per_inference: int = 16, policy_dt: float = 0.1):
    skill = UMISkill.__new__(UMISkill)
    skill.steps_per_inference = steps_per_inference
    skill.policy_dt = policy_dt
    skill._next_inference_mono = time.monotonic()
    skill._startup = None
    return skill


def test_the_first_chunk_logs_the_start_up_phases_once(caplog):
    skill = _skill()
    skill._startup = [("constructed", 0.0), ("first observation", 0.0)]
    skill._run_inference = lambda: ["wp"]

    with caplog.at_level("INFO", logger="solarian_umi.skill"):
        skill.control_step(0.1)
        skill._next_inference_mono = 0.0
        skill.control_step(0.1)

    (line,) = [r.getMessage() for r in caplog.records if "start-up" in r.getMessage()]
    assert "first observation" in line and "first step" in line and "first chunk" in line
    assert skill._startup is None


def test_first_control_step_infers_immediately():
    skill = _skill()
    calls = []
    skill._run_inference = lambda: calls.append("infer") or ["wp"]

    assert skill.control_step(0.1) == ["wp"]
    assert calls == ["infer"]


def test_control_step_skips_until_steps_per_inference_elapse():
    skill = _skill(steps_per_inference=16, policy_dt=0.1)
    calls = []
    skill._run_inference = lambda: calls.append("infer") or ["wp"]

    assert skill.control_step(0.1) == ["wp"]
    assert skill.control_step(0.1) is None
    assert calls == ["infer"]


def test_next_inference_stays_on_the_cycle_grid():
    """Due time advances from the cycle start, not from inference completion."""
    skill = _skill(steps_per_inference=16, policy_dt=0.1)
    t0 = time.monotonic()
    skill._next_inference_mono = t0

    def slow_infer():
        time.sleep(0.05)
        return ["wp"]

    skill._run_inference = slow_infer
    skill.control_step(0.1)

    assert skill._next_inference_mono == pytest.approx(t0 + 1.6, abs=1e-4)
    assert time.monotonic() < skill._next_inference_mono


def test_over_budget_cycle_keeps_the_original_grid():
    """Two back-to-back overruns still land on t0 + 2N, not now + N."""
    skill = _skill(steps_per_inference=16, policy_dt=0.1)
    t0 = time.monotonic() - 1.61
    skill._next_inference_mono = t0
    skill._run_inference = lambda: ["wp"]

    skill.control_step(0.1)
    skill.control_step(0.1)

    assert skill._next_inference_mono == pytest.approx(t0 + 3.2, abs=1e-4)


def test_failed_inference_retries_on_the_next_tick():
    skill = _skill()
    skill._run_inference = lambda: (_ for _ in ()).throw(RuntimeError("boom"))

    assert skill.control_step(0.1) is None
    skill._run_inference = lambda: ["wp"]
    assert skill.control_step(0.1) == ["wp"]


class _FakeBuffer:
    """A full ring of evenly spaced samples, oldest first."""

    get_max_k = 32

    def __init__(self, period, newest):
        n = self.get_max_k
        self._stamps = [newest - (n - 1 - i) * period for i in range(n)]

    def last_k(self, k):
        return [SimpleNamespace(t=s) for s in self._stamps[-k:]]


def _window_skill(policy_dt: float = 0.1, execution_speed_scale: float = 1.0):
    skill = UMISkill.__new__(UMISkill)
    skill.policy_dt = policy_dt
    skill.history_dt = policy_dt * execution_speed_scale
    return skill


def _oldest_read(buffer, k):
    return buffer.last_k(k)[0].t


# Buffer stamps here are what the skill sees, which is what each module
# publishes: already backdated by its own ``receive_latency``. So the camera's
# newest stamp *is* the anchor, and the arm's stamps run ahead of it by
# however stale the newest frame is.
def test_a_one_frame_policy_reads_one_frame():
    """The other 31 in the ring are 2.3 MB each, copied for nothing."""
    skill = _window_skill()
    cam = _FakeBuffer(period=1 / 30, newest=1000.0)

    k = skill._history_window(cam, lambda o: o.t, 1, 1, 1000.0)

    assert k == 1


def test_arm_window_reaches_back_past_a_stale_camera_anchor():
    """Horizons count back from the camera's stamp, not the arm's.

    The arm publishes 100x faster than the camera is late, so sizing its
    window against its own newest sample leaves the interpolator
    extrapolating the oldest pose the policy asked for.
    """
    skill = _window_skill()
    arm = _FakeBuffer(period=0.01, newest=1000.0)
    anchor = 1000.0 - 0.149

    k = skill._history_window(arm, lambda o: o.t, 2, 1, anchor)

    assert _oldest_read(arm, k) <= anchor - 0.1


def test_a_multi_frame_camera_horizon_still_spans_its_history():
    skill = _window_skill()
    cam = _FakeBuffer(period=1 / 30, newest=1000.0)

    k = skill._history_window(cam, lambda o: o.t, 2, 1, 1000.0)

    assert _oldest_read(cam, k) <= 1000.0 - 0.1


def test_the_window_never_asks_for_more_than_the_ring_holds():
    """last_k clamps, but the caller should not be relying on that."""
    skill = _window_skill()
    arm = _FakeBuffer(period=0.01, newest=1000.0)

    k = skill._history_window(arm, lambda o: o.t, 16, 4, 1000.0 - 5.0)

    assert k == arm.get_max_k


def test_a_slowed_replay_reads_a_proportionally_longer_history():
    skill = _window_skill(policy_dt=0.1, execution_speed_scale=4.0)
    arm = _FakeBuffer(period=0.02, newest=1000.0)

    k = skill._history_window(arm, lambda o: o.t, 2, 1, 1000.0)

    assert _oldest_read(arm, k) <= 1000.0 - 0.4


def test_a_ring_too_short_for_the_history_says_how_big_it_must_be(caplog):
    """32 samples of a 90 Hz arm is 0.35 s; a 6x replay asks for 0.6 s."""
    skill = _window_skill(policy_dt=0.1, execution_speed_scale=6.0)
    arm = _FakeBuffer(period=1 / 90, newest=1000.0)

    with caplog.at_level("WARNING", logger="solarian_umi.skill"):
        skill._history_window(arm, lambda o: o.t, 2, 1, 1000.0)

    assert "Raise get_max_k" in caplog.text


def _stamped(value, t):
    return SimpleNamespace(value=value, timestamp=t)


class _Ring:
    def __init__(self, samples):
        self._samples = samples

    def last_k(self, k):
        return self._samples[-k:]


def _obs_skill(*, policy_dt, execution_speed_scale, arm_speed):
    """Camera at 30 Hz, arm at 100 Hz moving along x at ``arm_speed`` m/s."""
    skill = _window_skill(policy_dt, execution_speed_scale)
    skill.params = {"camera": "cam", "arm": "arm", "gripper": "grip"}
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    cam = [SimpleNamespace(image=_stamped(frame, 10.0 - i / 30)) for i in range(32)][::-1]
    arm_t = [10.0 - i * 0.01 for i in range(128)][::-1]
    arm = [
        SimpleNamespace(
            x=_stamped(arm_speed * t, t), y=_stamped(0.0, t), z=_stamped(0.0, t),
            rx=_stamped(0.0, t), ry=_stamped(0.0, t), rz=_stamped(0.0, t),
        )
        for t in arm_t
    ]
    grip = [
        SimpleNamespace(gripper=_stamped(0.0, t), aperture_mm=_stamped(50.0, t))
        for t in arm_t
    ]
    skill.observations = {"cam": _Ring(cam), "arm": _Ring(arm), "grip": _Ring(grip)}
    skill._camera_window = skill._arm_window = skill._gripper_window = 128
    skill.camera_horizon, skill.arm_horizon, skill.gripper_horizon = 1, 2, 2
    skill.camera_down_sample_steps = 1
    skill.arm_down_sample_steps = skill.gripper_down_sample_steps = 1
    skill._ensure_camera_remap = lambda size: None
    skill._camera_obs_extras = lambda hist, frame_hw, remap: {}
    skill._resize_to_policy = lambda images: images
    skill._apply_image_masks = lambda images: images
    skill._obs_width_m = lambda aperture_mm: aperture_mm / 1000.0
    return skill


def test_a_slowed_replay_shows_the_policy_its_training_speed():
    """Training history is one action step apart. Replayed 4x slower, the arm
    covers a step's distance in 4 policy_dt, so sampling 1 policy_dt apart
    would show a quarter of the motion: a hand about to stop and release."""
    skill = _obs_skill(policy_dt=0.1, execution_speed_scale=4.0, arm_speed=0.05)

    obs = skill.get_umi_obs()

    step = np.diff(obs["robot0_eef_pos"][:, 0])[0]
    assert step == pytest.approx(0.05 * 0.4, abs=1e-6)


def _chunk_skill(*, policy_dt: float = 0.1, execution_speed_scale: float = 1.5):
    skill = UMISkill.__new__(UMISkill)
    skill.policy_dt = policy_dt
    skill.execution_speed_scale = execution_speed_scale
    skill._eval_t_start = 100.0
    skill._chunk_sent = True
    skill.arm_action_latency = 0.09
    skill.gripper_action_latency = 0.05
    return skill


def test_the_first_chunk_starts_after_now_instead_of_dropping_its_head():
    """The arm stood still through the first inference, so the steps between
    the stale observation and now were never run; dropping them makes the arm
    dash to the first kept step at its speed cap."""
    skill = _chunk_skill(policy_dt=0.1, execution_speed_scale=1.5)
    skill._chunk_sent = False
    poses = np.arange(16, dtype=np.float64).reshape(16, 1)

    kept, timestamps = skill._prepare_chunk(
        poses, obs_last_timestamp=10.0, curr_time=10.5
    )

    np.testing.assert_array_equal(kept.reshape(-1), np.arange(16))
    # now + arm latency (the larger) + one policy step
    assert timestamps[0] == pytest.approx(10.5 + 0.09 + 0.1)
    assert np.diff(timestamps) == pytest.approx(np.full(15, 0.15))


def test_a_fresh_first_observation_keeps_the_umi_anchor():
    skill = _chunk_skill(policy_dt=0.1, execution_speed_scale=1.5)
    skill._chunk_sent = False
    poses = np.arange(4, dtype=np.float64).reshape(4, 1)

    _, timestamps = skill._prepare_chunk(poses, obs_last_timestamp=11.0, curr_time=10.5)

    assert timestamps[0] == pytest.approx(11.0)


def test_first_action_timestamp_is_the_last_obs_time():
    """UMI: arange(len) so action 0 lands on obs_timestamps[-1], not +dt."""
    skill = _chunk_skill(policy_dt=0.1, execution_speed_scale=1.5)
    poses = np.arange(16, dtype=np.float64).reshape(16, 1)
    _, timestamps = skill._prepare_chunk(poses, obs_last_timestamp=10.0, curr_time=9.0)

    assert timestamps[0] == pytest.approx(10.0)
    assert timestamps[1] == pytest.approx(10.15)
    assert len(timestamps) == 16


def test_prepare_chunk_drops_past_waypoints():
    skill = _chunk_skill(policy_dt=0.1, execution_speed_scale=1.5)
    poses = np.arange(8, dtype=np.float64).reshape(8, 1)
    # obs at 10.0; steps at 10.00, 10.15, ... 11.05. now+10ms = 10.41 → keep 10.45+
    kept, timestamps = skill._prepare_chunk(
        poses, obs_last_timestamp=10.0, curr_time=10.4
    )

    np.testing.assert_array_equal(kept.reshape(-1), np.arange(3, 8))
    assert timestamps[0] == pytest.approx(10.45)


def _send_skill(*, arm_latency: float, gripper_latency: float):
    skill = UMISkill.__new__(UMISkill)
    skill.arm_action_latency = arm_latency
    skill.gripper_action_latency = gripper_latency
    skill.params = {"arm": "right_arm", "gripper": "right_gripper"}
    skill._arm_action_fields = ["x", "y", "z", "rx", "ry", "rz"]
    skill._gripper_action_field = "gripper"
    skill._cmd_width_mm = lambda width_m: width_m * 1000.0
    return skill


# now=10.0, arm needs 90 ms of notice and the gripper 50 ms, so of these four
# the first is reachable by neither, the second by the gripper alone.
_SEND_STAMPS = np.array([10.04, 10.07, 10.12, 10.20])


def test_a_waypoint_the_arm_cannot_reach_still_goes_to_the_gripper():
    """The two latencies differ, so one mask for both would discard usable
    gripper waypoints along with the arm's unreachable ones."""
    skill = _send_skill(arm_latency=0.09, gripper_latency=0.05)
    actions = np.zeros((4, 7), dtype=np.float64)

    waypoints = skill.exec_actions(actions, _SEND_STAMPS, now=10.0)

    assert [sorted(w.data) for w in waypoints] == [
        ["right_gripper"],
        ["right_arm", "right_gripper"],
        ["right_arm", "right_gripper"],
    ]


def test_a_waypoint_no_module_can_reach_is_not_sent_at_all():
    skill = _send_skill(arm_latency=0.09, gripper_latency=0.05)
    actions = np.zeros((1, 7), dtype=np.float64)

    assert skill.exec_actions(actions, _SEND_STAMPS[:1], now=10.0) == []


def test_the_reported_chunk_counts_what_the_module_will_execute():
    """The count used to include waypoints the module then threw away."""
    skill = _send_skill(arm_latency=0.09, gripper_latency=0.05)

    arm_n, arm_lead_ms = skill._module_chunk(_SEND_STAMPS, 0.09, 10.0)
    grip_n, grip_lead_ms = skill._module_chunk(_SEND_STAMPS, 0.05, 10.0)

    assert (arm_n, grip_n) == (2, 3)
    assert arm_lead_ms == pytest.approx(120.0)
    assert grip_lead_ms == pytest.approx(70.0)


def test_a_module_that_gets_nothing_reports_no_lead():
    skill = _send_skill(arm_latency=0.09, gripper_latency=0.05)

    n, lead_ms = skill._module_chunk(_SEND_STAMPS[:1], 0.09, 10.0)

    assert n == 0
    assert np.isnan(lead_ms)


def test_latency_compensation_moves_the_target_earlier_by_the_measured_delay():
    skill = _send_skill(arm_latency=0.09, gripper_latency=0.05)
    actions = np.zeros((1, 7), dtype=np.float64)

    waypoint = skill.exec_actions(actions, np.array([10.20]), now=10.0)[0]

    assert waypoint.data["right_arm"]["data"]["x"]["timestamp"] == pytest.approx(10.11)
    grip = waypoint.data["right_gripper"]["data"]["gripper"]
    assert grip["timestamp"] == pytest.approx(10.15)


def test_prepare_chunk_over_budget_schedules_last_on_policy_grid():
    skill = _chunk_skill(policy_dt=0.1, execution_speed_scale=1.5)
    skill._eval_t_start = 10.0
    poses = np.arange(4, dtype=np.float64).reshape(4, 1)
    # Last stamp is 10.0 + 3*0.15 = 10.45; curr 11.35 is past the whole chunk.
    kept, timestamps = skill._prepare_chunk(
        poses, obs_last_timestamp=10.0, curr_time=11.35
    )

    np.testing.assert_array_equal(kept.reshape(-1), [3])
    # ceil((11.35 - 10.0) / 0.1) = 14 → 10.0 + 1.4
    assert timestamps[0] == pytest.approx(11.4)
