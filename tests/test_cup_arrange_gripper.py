"""Cup-arrange gripper width is UMI metres over a mm action space."""

import numpy as np

from solarian_umi_skill.skill import UMISkill


def _skill():
    skill = UMISkill.__new__(UMISkill)
    skill._gripper_calibration = {
        "min_gripper_width": 0.0,
        "max_gripper_width": 85.0,
        "policy_min_width_m": 0.0,
        "policy_max_width_m": 0.09,
    }
    return skill


def test_obs_fully_open_is_physical_metres():
    skill = _skill()
    # Fully open is 85 mm = 0.085 m, inside the policy's 0.09 m cap.
    np.testing.assert_allclose(skill._obs_width_m(85.0), 0.085, atol=1e-6)
    np.testing.assert_allclose(skill._obs_width_m(0.0), 0.0, atol=1e-6)


def test_obs_identity_inside_umi_range():
    skill = _skill()
    # An 80 mm opening (typical UMI "open") is in-range: report 0.08 m.
    np.testing.assert_allclose(skill._obs_width_m(80.0), 0.08, atol=1e-6)


def test_cmd_open_clips_to_physical_max():
    skill = _skill()
    # Policy "fully open" (0.09 m) is past the 85 mm hard stop.
    assert skill._cmd_width_mm(0.09) == 85.0
    assert skill._cmd_width_mm(0.15) == 85.0


def test_cmd_close_keeps_physical_metres():
    skill = _skill()
    # A 40 mm grasp stays 40 mm. Fully closed (0 m) stays 0 mm.
    assert skill._cmd_width_mm(0.04) == 40.0
    assert skill._cmd_width_mm(0.0) == 0.0


def test_over_grip_closes_more_than_policy_cmd():
    skill = _skill()
    skill._gripper_calibration["over_grip"] = -4.0
    np.testing.assert_allclose(skill._cmd_width_mm(0.04), 36.0, atol=1e-6)
    np.testing.assert_allclose(skill._cmd_width_mm(0.0), -4.0, atol=1e-6)


def test_over_grip_obs_matches_policy_command():
    """A physical 36 mm (after extra close) reports 40 mm to the policy."""
    skill = _skill()
    skill._gripper_calibration["over_grip"] = -4.0
    np.testing.assert_allclose(skill._obs_width_m(36.0), 0.04, atol=1e-6)


def test_larger_is_more_open():
    skill = _skill()
    assert skill._cmd_width_mm(0.08) > skill._cmd_width_mm(0.04)
    assert skill._obs_width_m(50.0) > skill._obs_width_m(20.0)


def test_publish_gripper_status_uses_mm(monkeypatch):
    skill = _skill()
    posted = {}

    def _set(key, value):
        posted[key] = value

    monkeypatch.setattr("solarian_umi_skill.skill.StatusBoard.set", _set)
    skill._publish_gripper_status(obs_width_m=0.08, cmd_width_m=0.04)
    assert posted["UMI gripper obs (mm)"] == 80.0
    assert posted["UMI gripper cmd (mm)"] == 40.0


