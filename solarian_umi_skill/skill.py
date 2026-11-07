"""
UMISkill — runs a UMI diffusion policy described by a project yaml.

The diffusion model is hosted by a ``DiffusionPolicyServer`` (from the
``solarian-policy-diffusion`` plugin) and reached over HTTP through a
:class:`~solarian.services.policy.PolicyClient`.

Arm and gripper motion match UMI's RTDE path: the skill schedules sparse
timestamped waypoints onto each module (``SCHEDULE_L`` / ``SCHEDULE_J``). The arm
module's ``PoseTrajectoryInterpolator`` and the gripper module's
``ScalarTrajectoryInterpolator`` upsample at their control rates. On every
``control_step``:

1. Every ``steps_per_inference * policy_dt`` seconds (UMI overlapping
   replan), run a policy inference and return one waypoint per *future*
   predicted (pose, gripper, target_time). Past steps are dropped so a
   new chunk replaces the unused tail instead of yanking the arm back.
2. Otherwise return nothing — modules own the dense streaming.

Which policy and which robot wiring run is the yaml passed as
``params["config"]`` (a filename in the project ``conf/``, or a path).
Copy ``conf/skills/umi.yaml`` to define another policy.

Targets
-------
A checkpoint trained with targets (see :mod:`solarian_umi_skill.targets`:
a box, a point or a mask marking something on ``camera0_rgb``) reads each
from a calculated field of the camera, named by the yaml's ``targets``
mapping ``{obs key: field}``, e.g. ``{bbox: bbox}`` to pick what a tracker
follows or ``{point: target_point}`` to place where a point tracker says. The
field's data class must match: ``BoundingBoxData``, ``PointData`` or
``MaskData``. Each is mapped into the policy image the same way the frame is.

Before the operator seeds a tracker a box is all zeros and the policy gets
the whole frame as its box; a point is all zeros and the policy gets
``[-1, -1]``; a mask is None and the policy gets an empty mask. Each is the
dataset's "no target". Starting the skill does not wait for one.

A skill with targets must set ``execution_place`` in its yaml: it has to run
where its fields are computed, so both confs read one deployment key.
"""

import logging
import os
import pathlib
import subprocess
import time

import numpy as np
import yaml

from solarian.core.action import Action, ActionType
from solarian.core.data import BoundingBoxData, MaskData, PointData
from solarian.core.execution import ExecutionPlace
from solarian.core.skill import Skill
from solarian.core.clock_sync import LatencyEma, synced_time
from solarian.core.conf_paths import conf_dir, module_config_path, skill_config_path
from solarian.core.module_yaml import read_module_field
from solarian.core.observation import Observation
from solarian.services import PolicyClient
from solarian.services.launch import server_spec
from solarian.utils.status_board import StatusBoard

from solarian_umi_skill.targets import target_keys

from solarian_umi_skill.umi.common.interpolation_util import (
    PoseInterpolator,
    get_interp1d,
)
from solarian_umi_skill.umi.policy_image import PolicyImagePipeline
from solarian_umi_skill.umi.real_world.real_inference_util import (
    get_real_obs_resolution,
    get_real_umi_action,
    get_real_umi_obs_dict,
)
from solarian_umi_skill.umi.utils import (
    solve_sphere_collision,
    solve_table_collision,
)


logger = logging.getLogger(__name__)

TARGET_DATA = {"box": BoundingBoxData, "point": PointData, "mask": MaskData}


def reachable(timestamps: np.ndarray, latency: float, now: float) -> np.ndarray:
    """Which of *timestamps* a module with *latency* can still hit.

    Its target is moved ``latency`` earlier before being sent, so a waypoint
    due sooner than that is already past on arrival and the module drops it.
    Deciding here instead costs the same waypoints but stops the chunk count
    claiming ones that never execute, and the two modules differ enough that
    one mask for both would throw away gripper waypoints the arm alone
    cannot reach.
    """
    return timestamps - latency > now


class UMISkill(Skill):
    """A UMI diffusion policy, configured by the yaml in ``params["config"]``."""

    policy_service = "diffusion-policy"
    config_name = "umi"
    execution_place = ExecutionPlace.ROBOT
    skill_id = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Occupancy is read from params before setup, so module names have to
        # be there as soon as the controller builds this instance.
        cfg = self._load_config()
        self.config = cfg
        for key in ("arm", "gripper", "camera"):
            if key not in self.params and key in cfg:
                self.params[key] = cfg[key]
        # Where to run is deployment config, so the same policy can be timed
        # on the robot and on the relay without a code change. A host that
        # constructs the skill to run it passes its own place, which wins.
        if kwargs.get("execution_place") is None and cfg.get("execution_place"):
            self.execution_place = ExecutionPlace(str(cfg["execution_place"]).lower())
        # Set here, not in setup: setup builds the first observation, and
        # with it the targets, before returning.
        self._targets: dict[str, str] = {str(k): str(v) for k, v in (cfg.get("targets") or {}).items()}
        if self._targets and kwargs.get("execution_place") is None and not cfg.get("execution_place"):
            raise ValueError(
                f"{self.params['config']} reads targets from camera fields, so it must set "
                "execution_place (server or robot), the same place those fields are computed"
            )
        #: Per obs key, ``(key, policy-frame target)``: a target changes far
        #: less often than inference runs, and mapping it costs a full-frame warp.
        self._target_cache: dict[str, tuple[tuple, np.ndarray]] = {}
        #: ``(phase, synced time it ended)`` from construction to the first
        #: chunk; synced, since construction and setup can be in different
        #: processes. ``None`` once logged.
        self._startup: list[tuple[str, float]] | None = [("constructed", synced_time())]

    def _startup_mark(self, phase: str) -> None:
        if self._startup is None:
            return
        self._startup.append((phase, synced_time()))
        logger.debug("UMISkill start-up: %s at +%.0f ms", phase,
                     (self._startup[-1][1] - self._startup[0][1]) * 1e3)

    @classmethod
    def tool_spec(cls, params) -> dict:
        """The contract of the policy ``params["config"]`` selects, from its yaml's ``agent:`` block.

        A policy runs until stopped, so the block's ``monitor`` questions or its
        ``max_duration_s`` are what end it. A checkpoint with targets takes a ``point`` in its
        camera's image, which :meth:`services_for` seeds each target's tracker at.
        """
        cfg = cls.load_config_file(skill_config_path(str(params["config"])))
        agent = cfg.get("agent") or {}
        spec = super().tool_spec(params)
        spec.update(
            description=agent.get("description"),
            examples=list(agent.get("examples") or []),
            ends_by_itself=bool(agent.get("ends_by_itself", False)),
            max_duration_s=agent.get("max_duration_s"),
            max_target_area=agent.get("max_target_area"),
            monitor=list(agent.get("monitor") or []),
            camera=params.get("camera") or cfg.get("camera"),
        )
        if cfg.get("targets"):
            spec["image_points"] = ["point"]
            spec["parameters"] = {
                "type": "object",
                "properties": {
                    "point": {
                        "type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
                        "description": agent.get("point") or "[x, y] pixels of the target in the camera image.",
                    },
                },
                "required": ["point"],
            }
        return spec

    @classmethod
    def services_for(cls, params) -> list[tuple[str, str, dict]]:
        if params.get("point") is None:
            return []
        cfg = cls.load_config_file(skill_config_path(str(params["config"])))
        camera = str(params.get("camera") or cfg["camera"])
        return [
            (camera, str(field_name), {"point": list(params["point"])})
            for field_name in (cfg.get("targets") or {}).values()
        ]

    @classmethod
    def load_config_file(cls, path: pathlib.Path) -> dict:
        """The yaml is the whole config; ``${deployment.*}`` is resolved against this host."""
        from solarian.utils.configs import resolve_with_deployment

        if not path.is_file():
            raise FileNotFoundError(
                f"UMISkill config not found: {path}. "
                "params['config'] is a filename in conf/skills/ or a path to a yaml."
            )
        with open(path, "r") as f:
            loaded = yaml.safe_load(f) or {}
        return resolve_with_deployment(loaded)

    def _load_config(self) -> dict:
        """Load ``params['config']``, then overlay other launch params."""
        raw = self.params.get("config")
        if not raw:
            raise ValueError(
                "UMISkill requires params['config']: a yaml filename in conf/skills/ "
                "(for example 'cup_arrange.yaml') or a path to one."
            )
        config = type(self).load_config_file(skill_config_path(str(raw)))
        for key, value in self.params.items():
            if key == "config":
                continue
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key] = type(self)._deep_merge(config[key], value)
            else:
                config[key] = value
        return config

    def setup(self):
        self._startup_mark("setup start")
        cfg = self._load_config()
        self.config = cfg

        for key, default in (
            ("arm", "right_arm"),
            ("gripper", "right_gripper"),
            ("camera", "right_wrist_camera"),
        ):
            if key not in self.params:
                self.params[key] = cfg.get(key, default)

        self._arm_action_fields = list(
            cfg.get("arm_action_fields", ["x", "y", "z", "rx", "ry", "rz"])
        )
        self._gripper_action_field = cfg.get("gripper_action_field", "gripper")

        self._gripper_calibration = dict(cfg.get("gripper_calibration", {}))
        self._logged_gripper = False

        self._managed_server_proc = None
        policy_server_url = self._maybe_start_managed_server(cfg)

        self.policy_client = PolicyClient(
            server_url=policy_server_url,
            token=cfg.get("policy_server_token"),
            connect_retries=int(cfg.get("connect_retries", 0)),
            retry_interval=float(cfg.get("retry_interval", 1.0)),
        )

        self.debug_image_path = cfg.get("debug_image_path")
        self.debug_image_max_dumps = int(cfg.get("debug_image_max_dumps", 5))
        self.umi_gold_image = cfg.get("umi_gold_image")
        self._debug_dump_count = 0
        self.dump_policy_obs = cfg.get("dump_policy_obs")
        self._policy_obs_dump_dir: pathlib.Path | None = None
        self._policy_obs_dump_count = 0

        try:
            model_info = self.policy_client.get_model_info()
            self.shape_meta = self.policy_client.get_shape_meta()
        except Exception:
            logger.exception(
                "Failed to retrieve model info from policy server at %s",
                policy_server_url,
            )
            raise
        self._startup_mark("policy server")

        self.obs_pose_repr = model_info.get("obs_pose_repr")
        self.action_pose_repr = model_info.get("action_pose_repr")
        logger.info(
            "Connected to policy server at %s (device=%s, model=%s, dataset=%s)",
            policy_server_url,
            model_info.get("device", "unknown"),
            model_info.get("model_name", "unknown"),
            model_info.get("dataset_path", "unknown"),
        )
        logger.info(
            "obs_pose_repr=%s action_pose_repr=%s",
            self.obs_pose_repr,
            self.action_pose_repr,
        )

        self.arms_config = cfg["arms"]
        self.frequency = float(cfg.get("frequency", self.frequency))
        self.policy_dt = 1.0 / self.frequency
        self.execution_speed_scale = float(cfg.get("execution_speed_scale", 1.0))
        # Training samples history one action step apart, so a replay stretched
        # by execution_speed_scale must stretch the history too, or the policy
        # reads its own slowed motion as a hand coming to a stop.
        self.history_dt = self.policy_dt * self.execution_speed_scale
        self.steps_per_inference = int(cfg.get("steps_per_inference", 6))
        self.discard_actions = bool(cfg.get("discard_actions", False))

        # Observation latency is gone from here on purpose: each module
        # subtracts its own ``receive_latency`` when it stamps, so the
        # timestamps reaching this skill already name the instant the reading
        # was true. Subtracting again would double-count it.
        self.arm_action_latency = read_module_field(
            self.params["arm"], "action_latency", 0.0
        )
        self.gripper_action_latency = read_module_field(
            self.params["gripper"], "action_latency", 0.0
        )

        self.camera_down_sample_steps = int(cfg.get("camera_down_sample_steps", 1))
        self.arm_down_sample_steps = int(cfg.get("arm_down_sample_steps", 1))
        self.gripper_down_sample_steps = int(cfg.get("gripper_down_sample_steps", 1))

        if "tx_left_right" in cfg:
            self.tx_robot1_robot0 = np.array(cfg["tx_left_right"])
        else:
            self.tx_robot1_robot0 = None

        self.no_mirror = bool(cfg.get("no_mirror", True))
        self.mirror_swap = bool(cfg.get("mirror_swap", False))
        # Pad 16:9 capture to a square before 224 resize. The Pi's 1080p mode
        # fills the short side with the fisheye; UMI's center-crop then discards
        # the black sides (the tunnel). Leave false for 4:3 capture (the UMI
        # crop then inscribes the circle, matching GoPro eval).
        self.letterbox_to_square = bool(cfg.get("letterbox_to_square", False))

        obs_meta = self.shape_meta["obs"]
        self.camera_horizon = int(obs_meta["camera0_rgb"]["horizon"])
        self.arm_horizon = int(obs_meta["robot0_eef_pos"]["horizon"])
        self.gripper_horizon = int(obs_meta["robot0_gripper_width"]["horizon"])
        self._check_targets(obs_meta)
        # Policy RGB size (wo, ho). UMI eval: resize-then-center-crop to this,
        # then paint gripper/mirror masks on the square frame.
        self.obs_image_resolution = get_real_obs_resolution(self.shape_meta)
        self._policy_pipeline = None
        self._camera_remap = None
        self._camera_remap_wh = None

        # Every millisecond between the capture and the schedule comes off
        # the front of the chunk in ``_prepare_chunk``, so these two are what
        # "the arm skipped the approach" looks like before it reaches the arm.
        self._obs_age = LatencyEma()
        self._pipeline = LatencyEma()
        self._t_buffers_ms = 0.0
        #: Planned arm waypoints already past when the chunk was scheduled,
        #: summed over the run. The cost of where this skill runs, in steps.
        self._dropped_waypoints = 0

        self.policy_client.reset()
        self._startup_mark("policy reset")
        self._wait_for_obs()
        self._startup_mark("observation history")

        # The instant every horizon counts back from, exactly as
        # ``get_umi_obs`` computes it.
        camera_buffer = self.observations[self.params["camera"]]
        anchor = camera_buffer.last_k(1)[-1].image.timestamp
        self._camera_window = self._history_window(
            camera_buffer,
            lambda o: o.image.timestamp,
            self.camera_horizon,
            self.camera_down_sample_steps,
            anchor,
        )
        self._arm_window = self._history_window(
            self.observations[self.params["arm"]],
            lambda o: o.x.timestamp,
            self.arm_horizon,
            self.arm_down_sample_steps,
            anchor,
        )
        self._gripper_window = self._history_window(
            self.observations[self.params["gripper"]],
            lambda o: o.gripper.timestamp,
            self.gripper_horizon,
            self.gripper_down_sample_steps,
            anchor,
        )
        logger.info(
            "Observation windows: camera %d, arm %d, gripper %d samples "
            "(ring holds %d)",
            self._camera_window,
            self._arm_window,
            self._gripper_window,
            self.observations[self.params["camera"]].get_max_k,
        )

        env_obs = self.get_umi_obs()
        self.episode_start_pose: list[np.ndarray] = []
        for robot_id in range(len(self.arms_config)):
            pose = np.concatenate(
                [
                    env_obs[f"robot{robot_id}_eef_pos"],
                    env_obs[f"robot{robot_id}_eef_rot_axis_angle"],
                ],
                axis=-1,
            )[-1]
            self.episode_start_pose.append(pose)
        self._startup_mark("first observation")

        now = synced_time()
        self._eval_t_start = now
        # UMI ``t_start``: inferences land on this monotonic grid every
        # ``steps_per_inference * policy_dt``.
        self._next_inference_mono = time.monotonic()
        #: Whether the arm has been handed a chunk yet; until then it is
        #: standing at the observed pose (see ``_prepare_chunk``).
        self._chunk_sent = False

        # UMI eval opens to max_gripper_width (0.09 m) before the first
        # inference. umi-slim SET_HOME is fully open (85 mm), inside that
        # range; still open to the policy max so the first obs matches training.
        # A skill that starts holding an object (place) must not, or it drops it.
        if not bool(cfg.get("open_gripper_on_start", True)):
            return
        open_m = self._policy_max_width_m()
        open_time = now + 0.5
        self.dispatch(
            Action(
                action_type=ActionType.MODULE_ACTION,
                data={
                    self.params["gripper"]: {
                        "action_type": ActionType.SCHEDULE_J,
                        "data": {
                            self._gripper_action_field: {
                                "value": float(self._cmd_width_mm(open_m)),
                                "timestamp": float(open_time),
                            }
                        },
                    }
                },
            )
        )
        self._publish_gripper_status(cmd_width_m=open_m)
        logger.info(
            "Opening gripper to policy max %.1f mm (umi-slim physical max is %.1f mm)",
            open_m * 1000.0,
            float(self._gripper_calibration["max_gripper_width"]) * 10.0,
        )

    # ------------------------------------------------------------------
    # Gripper: policy metres <-> module aperture mm
    #
    # Normalized servo <-> aperture is the gripper module's job, through its
    # measured `aperture_table`. This skill only converts units and applies the
    # UMI `over_grip` bias.
    # ------------------------------------------------------------------

    def _policy_min_width_m(self) -> float:
        return float(self._gripper_calibration.get("policy_min_width_m", 0.0))

    def _policy_max_width_m(self) -> float:
        # UMI eval_real.py uses 0.09 m as fully open.
        return float(self._gripper_calibration.get("policy_max_width_m", 0.09))

    def _over_grip_mm(self) -> float:
        """Extra close/open in mm (UMI ``over_grip``). Negative closes more."""
        return float(self._gripper_calibration.get("over_grip", 0.0))

    def _obs_width_m(self, aperture_mm: float) -> float:
        """Measured aperture in metres, clipped to the policy's trained range.

        Larger is more open (UMI convention). umi-slim can open past 90 mm;
        reporting that is out-of-distribution for cup-arrange. ``over_grip`` is
        subtracted (UMI driver) so the policy sees the width it commanded, not
        the extra squeeze.
        """
        width_m = (float(aperture_mm) - self._over_grip_mm()) / 1000.0
        return float(
            np.clip(width_m, self._policy_min_width_m(), self._policy_max_width_m())
        )

    def _cmd_width_mm(self, width_m: float) -> float:
        """Policy metres → umi-slim aperture mm, plus UMI ``over_grip``.

        Negative over_grip may go below ``min_gripper_width`` (extra close).
        The module still clamps the servo to [-100, 100].
        """
        width_m = float(
            np.clip(width_m, self._policy_min_width_m(), self._policy_max_width_m())
        )
        over = self._over_grip_mm()
        mm = width_m * 1000.0 + over
        lo = float(self._gripper_calibration["min_gripper_width"]) + min(over, 0.0)
        hi = float(self._gripper_calibration["max_gripper_width"])
        return float(np.clip(mm, lo, hi))

    def _publish_gripper_status(
        self, obs_width_m: float | None = None, cmd_width_m: float | None = None
    ) -> None:
        """Show UMI gripper obs/cmd (mm) on the robot StatusBoard."""
        if obs_width_m is not None:
            StatusBoard.set(
                "UMI gripper obs (mm)", round(float(obs_width_m) * 1000.0, 2)
            )
        if cmd_width_m is not None and np.isfinite(cmd_width_m):
            StatusBoard.set(
                "UMI gripper cmd (mm)", round(self._cmd_width_mm(cmd_width_m), 2)
            )

    # ------------------------------------------------------------------
    # Managed / co-located policy server
    # ------------------------------------------------------------------

    def _maybe_start_managed_server(self, cfg: dict) -> str:
        """Launch the conf's ``server:`` as a child process when it is managed.

        Returns the URL to reach the policy at: the managed server's, or the
        conf's ``policy_server_url`` for one running elsewhere.
        """
        block = cfg.get("server") or {}
        if not block.get("managed", False):
            return cfg["policy_server_url"]
        spec = server_spec(block, default_service=type(self).policy_service, source=str(self.params["config"]))
        cmd = spec.command()
        logger.info("Launching managed policy server: %s", " ".join(cmd))
        self._managed_server_proc = subprocess.Popen(cmd)
        return f"http://{spec.host}:{spec.port}"

    # ------------------------------------------------------------------
    # Observation handling
    # ------------------------------------------------------------------

    @property
    def _camera_min_samples(self) -> int:
        return self.camera_horizon + 2

    @property
    def _robot_min_samples(self) -> int:
        return self.arm_horizon + 2

    @property
    def _gripper_min_samples(self) -> int:
        return self.gripper_horizon + 2

    def _wait_for_obs(self, timeout: float = 10.0) -> None:
        """Block until each observation buffer has enough samples."""
        deadline = time.time() + timeout
        observations = self.observations
        while time.time() < deadline:
            arm_buf = observations.get(self.params["arm"])
            cam_buf = observations.get(self.params["camera"])
            grip_buf = observations.get(self.params["gripper"])
            if (
                arm_buf is not None
                and cam_buf is not None
                and grip_buf is not None
                and arm_buf.count >= self._robot_min_samples
                and cam_buf.count >= self._camera_min_samples
                and grip_buf.count >= self._gripper_min_samples
            ):
                return
            time.sleep(0.05)
        raise RuntimeError(
            "Timed out waiting for observation buffers to fill "
            f"(need camera>={self._camera_min_samples}, "
            f"arm>={self._robot_min_samples}, "
            f"gripper>={self._gripper_min_samples})"
        )

    def _history_window(
        self,
        buffer,
        stamp,
        horizon: int,
        down_sample: int,
        anchor: float,
    ) -> int:
        """Samples to read per inference to span the history the policy wants.

        ``last_k`` copies whole observations out of shared memory, so reading
        the whole ring to use one frame is the most expensive thing this
        skill does: the camera ring is 32 x 1014 x 760 x 3, or 74 MB, per
        inference. That cost is memory bandwidth, which is exactly what the
        archive encoder competes for, so an oversized window is worst under
        recording.

        The reach is measured back from this buffer's newest sample to the
        oldest instant ``get_umi_obs`` will resample at. Those are not the
        same clock: every horizon is anchored on the camera's stamp, which the
        camera module has already backdated by its ``receive_latency``, so a
        fast-publishing arm must still hold history reaching back past a camera
        frame that is 150 ms old, or the interpolator is asked to extrapolate.
        Doubled, so jitter and a late camera frame cannot leave the oldest
        sample short; that doubling is the whole margin, and no constant is
        added on top. A single-frame policy then reads one frame, which is
        right: the camera is what the anchor is measured from, so its newest
        sample is the anchor.
        """
        stamps = np.array(
            [stamp(o) for o in buffer.last_k(buffer.get_max_k)], dtype=np.float64
        )
        intervals = np.diff(stamps)
        period = float(np.median(intervals)) if intervals.size else 0.0
        if period <= 0.0:
            return buffer.get_max_k
        oldest_needed = anchor - (horizon - 1) * down_sample * self.history_dt
        reach_s = max(0.0, float(stamps[-1]) - oldest_needed)
        if buffer.get_max_k * period < reach_s:
            logger.warning(
                "UMISkill history needs %.2f s but the ring holds %d samples "
                "(%.2f s); older poses are clamped to the oldest sample. "
                "Raise get_max_k in that module's yaml to at least %d.",
                reach_s, buffer.get_max_k, buffer.get_max_k * period,
                int(np.ceil(2.0 * reach_s / period)) + horizon,
            )
        return int(min(buffer.get_max_k, np.ceil(2.0 * reach_s / period) + horizon))

    @staticmethod
    def _atomic_imwrite(path: pathlib.Path, bgr: np.ndarray) -> None:
        """Write ``bgr`` to ``path`` atomically (write to .tmp.<ext>, rename)."""
        import cv2

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        tmp_path = path.with_name(path.stem + ".tmp" + path.suffix)
        try:
            ok = cv2.imwrite(str(tmp_path), bgr)
            if not ok:
                logger.error(
                    "cv2.imwrite returned False for %s "
                    "(extension %s — unsupported codec? array dtype %s, shape %s)",
                    tmp_path, path.suffix, bgr.dtype, bgr.shape,
                )
                return
            os.replace(tmp_path, path)
        except Exception:
            logger.exception("Failed to write debug image to %s", path)

    _IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}

    def _resolve_debug_path(self) -> pathlib.Path | None:
        raw = self.debug_image_path
        if not raw:
            return None
        path = pathlib.Path(os.path.expanduser(str(raw)))
        if not path.is_absolute():
            root = os.environ.get("SOLARIAN_PROJECT_PATH") or os.getcwd()
            path = pathlib.Path(root) / path
        return path

    def _debug_path_is_dir(self, path: pathlib.Path) -> bool:
        if path.exists():
            return path.is_dir()
        return path.suffix.lower() not in self._IMAGE_SUFFIXES

    def _frames_to_strip(self, frames_hwc_rgb: np.ndarray) -> np.ndarray:
        """Concatenate ``(T, H, W, C)`` uint8 RGB frames into a strip."""
        if frames_hwc_rgb.shape[0] == 1:
            return frames_hwc_rgb[0]
        sep = np.zeros(
            (frames_hwc_rgb.shape[1], 4, frames_hwc_rgb.shape[3]),
            dtype=np.uint8,
        )
        tiles: list[np.ndarray] = []
        for i, f in enumerate(frames_hwc_rgb):
            if i > 0:
                tiles.append(sep)
            tiles.append(f)
        return np.concatenate(tiles, axis=1)

    def _write_rgb_strip(self, out_path: pathlib.Path, frames: np.ndarray) -> None:
        """Write a uint8 (T, H, W, C) RGB stack as a side-by-side strip."""
        try:
            import cv2
        except ImportError:
            return
        if frames.ndim != 4 or frames.shape[-1] != 3:
            return
        if frames.dtype != np.uint8:
            frames = np.clip(frames, 0, 255).astype(np.uint8)
        strip = self._frames_to_strip(frames)
        self._atomic_imwrite(out_path, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))

    def _dump_native_image(self, suffix: str, frames: np.ndarray) -> None:
        """Write a uint8 (T, H, W, C) RGB stack next to ``debug_image_path``."""
        path = self._resolve_debug_path()
        if path is None:
            return
        if self._debug_path_is_dir(path):
            name = {
                ".masked": "solarian_masked_hwc.png",
                ".raw": "solarian_raw_hwc.png",
            }.get(suffix, f"solarian{suffix}.png")
            out_path = path / name
        else:
            out_path = path.with_name(path.stem + suffix + path.suffix)
        self._write_rgb_strip(out_path, frames)

    def _dump_policy_obs(self, obs_dict_np: dict, raw_action, env_obs: dict) -> None:
        """One ``.npz`` per inference for solarian_base's ``scripts/diagnose``.

        ``obs.<key>`` is exactly what the policy server received, ``raw_action``
        the chunk it returned, ``eef_pose`` the current TCP pose (position m +
        rotation vector) in the robot base frame. Each run writes into a
        subfolder named by the time of its first inference.
        """
        if self._policy_obs_dump_dir is None:
            self._policy_obs_dump_dir = (
                pathlib.Path(self.dump_policy_obs).expanduser() / time.strftime("%Y%m%d-%H%M%S"))
            self._policy_obs_dump_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self._policy_obs_dump_dir / f"{self._policy_obs_dump_count:05d}.npz",
            **{f"obs.{key}": np.asarray(value) for key, value in obs_dict_np.items()},
            raw_action=np.asarray(raw_action),
            eef_pose=np.concatenate(
                [env_obs["robot0_eef_pos"][-1], env_obs["robot0_eef_rot_axis_angle"][-1]]
            ),
            timestamp=float(env_obs["timestamp"][-1]),
        )
        self._policy_obs_dump_count += 1

    def _dump_policy_input_image(self, obs_dict_np: dict, env_obs: dict) -> None:
        """Dump the policy input image plus raw / masked siblings for debugging."""
        path = self._resolve_debug_path()
        if path is None:
            return
        if self._debug_dump_count >= self.debug_image_max_dumps:
            return

        try:
            import cv2
        except ImportError:
            logger.warning(
                "debug_image_path set but cv2 is unavailable; skipping dump"
            )
            self.debug_image_path = None
            return

        dump_idx = self._debug_dump_count
        self._debug_dump_count += 1
        is_dir = self._debug_path_is_dir(path)
        if is_dir:
            path.mkdir(parents=True, exist_ok=True)

        policy_shape = policy_dtype = None
        policy_min = policy_max = policy_mean = 0.0
        rgb = obs_dict_np.get("camera0_rgb")
        policy_hwc = None
        if rgb is not None:
            arr = np.asarray(rgb)
            if arr.ndim == 4:
                policy_shape = tuple(arr.shape)
                policy_dtype = arr.dtype
                policy_min = float(arr.min())
                policy_max = float(arr.max())
                policy_mean = float(arr.mean())
                logger.info(
                    "policy camera tensor: shape=%s dtype=%s "
                    "min=%.3f max=%.3f mean=%.3f",
                    arr.shape, arr.dtype,
                    policy_min, policy_max, policy_mean,
                )
                frames = np.clip(arr, 0.0, 1.0)
                frames = (frames * 255.0).astype(np.uint8)
                frames = np.ascontiguousarray(np.moveaxis(frames, 1, -1))
                boxes = obs_dict_np.get("bbox")
                if boxes is not None:
                    # The 4th channel the policy rasterizes: exclusive x1/y1.
                    for frame, box in zip(frames, np.asarray(boxes).reshape(-1, 4)):
                        x0, y0, x1, y1 = (int(round(float(v))) for v in box)
                        cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 0), 1)
                    logger.info("policy bbox: %s", np.asarray(boxes).reshape(-1, 4).tolist())
                policy_hwc = frames
                if is_dir:
                    out = path / f"solarian_camera0_rgb_t{dump_idx}.png"
                else:
                    out = path
                self._write_rgb_strip(out, frames)

        masked = env_obs.get("camera0_rgb")
        masked_arr = None
        if masked is not None:
            masked_arr = np.asarray(masked)
            if dump_idx == 0:
                self._dump_native_image(".masked", masked_arr)

        raw_arr = None
        try:
            cam_hist = self.observations[self.params["camera"]].last_k(2)
            raw_arr = np.stack(
                [np.asarray(c.image.value) for c in cam_hist], axis=0
            )
            logger.info(
                "raw camera buffer frame: shape=%s dtype=%s "
                "min=%d max=%d mean=%.1f",
                raw_arr.shape, raw_arr.dtype,
                int(raw_arr.min()), int(raw_arr.max()), float(raw_arr.mean()),
            )
            if dump_idx == 0:
                self._dump_native_image(".raw", raw_arr)
        except Exception:
            logger.exception("Failed to dump raw camera buffer frame")

        if not is_dir or dump_idx != 0 or policy_hwc is None:
            return

        from solarian_umi_skill.umi.compare_policy_image import (
            format_summary,
            write_contact_sheet,
        )

        gold = self.umi_gold_image
        compare_lines = None
        try:
            gold_path = pathlib.Path(gold) if gold else None
            stats = write_contact_sheet(path, gold_path=gold_path)
            compare_lines = stats["lines"]
            logger.info("Wrote UMI image compare sheet to %s", stats["compare_path"])
        except FileNotFoundError as exc:
            logger.warning("Skipping compare sheet (%s)", exc)
        except Exception:
            logger.exception("Failed to write UMI image compare sheet")

        summary = format_summary(
            policy_shape=policy_shape,
            policy_dtype=policy_dtype,
            policy_min=policy_min,
            policy_max=policy_max,
            policy_mean=policy_mean,
            masked_shape=None if masked_arr is None else tuple(masked_arr.shape),
            raw_shape=None if raw_arr is None else tuple(raw_arr.shape),
            compare_lines=compare_lines,
        )
        try:
            (path / "summary.txt").write_text(summary)
        except Exception:
            logger.exception("Failed to write image-compare summary.txt")
        logger.info("Wrote policy image dump to %s", path)

    def _ensure_policy_pipeline(self) -> PolicyImagePipeline:
        key = (
            tuple(self.obs_image_resolution),
            bool(self.letterbox_to_square),
            bool(self.no_mirror),
            bool(self.mirror_swap),
        )
        pipe = getattr(self, "_policy_pipeline", None)
        if pipe is None or getattr(pipe, "_skill_key", None) != key:
            pipe = PolicyImagePipeline(
                output_res=self.obs_image_resolution,
                letterbox_to_square=self.letterbox_to_square,
                no_mirror=self.no_mirror,
                mirror_swap=self.mirror_swap,
            )
            pipe._skill_key = key
            self._policy_pipeline = pipe
        return pipe

    def _camera_sensor_conf(self) -> dict:
        """The wrist camera's ``sensors:`` block, read straight from YAML.

        Only the optical-remap keys are wanted and none of them interpolate,
        so this deliberately skips the deployment resolution a full config
        load would need.
        """
        import yaml

        path = module_config_path(self.params['camera'])
        with open(path, "r") as fp:
            doc = yaml.safe_load(fp) or {}
        sensors = doc.get("sensors") or {}
        return next(iter(sensors.values()), {})

    def _ensure_camera_remap(self, frame_wh: tuple[int, int]):
        """The warp into the plane the checkpoint was trained on.

        The capture path does not warp — see ``load_policy_remap``. Built on
        first use because the maps sample the source, and the source size is
        whatever the observation happens to carry.
        """
        if self._camera_remap_wh != frame_wh:
            from solarian.core.optical_remap import load_policy_remap

            sensor = self._camera_sensor_conf()
            out = (sensor.get("remap_width"), sensor.get("remap_height"))
            self._camera_remap = load_policy_remap(
                sensor.get("optical_remap"),
                frame_wh,
                out_wh=out if all(out) else None,
                conf_dir=conf_dir(),
            )
            self._camera_remap_wh = frame_wh
        return self._camera_remap

    def _resize_to_policy(self, images: np.ndarray) -> np.ndarray:
        """Map capture frames to policy RGB (224×224 for cup-wild)."""
        return self._ensure_policy_pipeline().resize(images)

    def _apply_image_masks(self, images: np.ndarray) -> np.ndarray:
        """Black out GoPro gripper body + side mirrors (UMI ``draw_predefined_mask``)."""
        return self._ensure_policy_pipeline().mask(images)

    def get_umi_obs(self) -> dict[str, np.ndarray]:
        """Build the env_obs dict expected by the UMI obs/action utilities.

        Not this skill's observations — those are the buffers on
        ``self.observations``. This resamples their history onto the
        timestamps the policy expects.
        """
        cam_buf = self.observations[self.params["camera"]]
        arm_buf = self.observations[self.params["arm"]]
        grip_buf = self.observations[self.params["gripper"]]

        t0 = time.perf_counter()
        cam_hist = cam_buf.last_k(self._camera_window)
        arm_hist = arm_buf.last_k(self._arm_window)
        grip_hist = grip_buf.last_k(self._gripper_window)
        self._t_buffers_ms = (time.perf_counter() - t0) * 1000.0
        if not cam_hist or not arm_hist or not grip_hist:
            raise RuntimeError(
                "Observation buffers became empty between waits and reads"
            )

        # Already backdated by each module's ``receive_latency`` at the point
        # of stamping, so these name the instant the reading was true.
        cam_ts = np.array([o.image.timestamp for o in cam_hist], dtype=np.float64)
        arm_ts = np.array([o.x.timestamp for o in arm_hist], dtype=np.float64)
        grip_ts = np.array([g.gripper.timestamp for g in grip_hist], dtype=np.float64)

        last_timestamp = float(cam_ts[-1])

        camera_obs_timestamps = last_timestamp - (
            np.arange(self.camera_horizon)[::-1]
            * self.camera_down_sample_steps
            * self.history_dt
        )
        arm_obs_timestamps = last_timestamp - (
            np.arange(self.arm_horizon)[::-1]
            * self.arm_down_sample_steps
            * self.history_dt
        )
        gripper_obs_timestamps = last_timestamp - (
            np.arange(self.gripper_horizon)[::-1]
            * self.gripper_down_sample_steps
            * self.history_dt
        )

        cam_idxs = [int(np.argmin(np.abs(cam_ts - t))) for t in camera_obs_timestamps]
        images = np.stack(
            [np.asarray(cam_hist[i].image.value) for i in cam_idxs], axis=0
        )
        if images.dtype != np.uint8:
            images = images.astype(np.uint8)

        # Into GoPro optics first: the masks and the 224 crop below are both
        # defined in that plane, and the capture path leaves frames in the
        # camera's own.
        frame_hw = (images.shape[1], images.shape[2])
        remap = self._ensure_camera_remap((frame_hw[1], frame_hw[0]))
        if remap is not None:
            images = np.stack([remap.apply(frame) for frame in images], axis=0)
        extras = self._camera_obs_extras(
            [cam_hist[i] for i in cam_idxs], frame_hw, remap
        )

        # Match UMI eval: square 224×224, then gripper/mirror polygons.
        images = self._resize_to_policy(images)
        images = self._apply_image_masks(images)

        arm_poses = np.array(
            [
                [
                    float(o.x.value), float(o.y.value), float(o.z.value),
                    float(o.rx.value), float(o.ry.value), float(o.rz.value),
                ]
                for o in arm_hist
            ],
            dtype=np.float64,
        )
        arm_ts_unique, unique_idx = np.unique(arm_ts, return_index=True)
        if arm_ts_unique.size < 2:
            raise RuntimeError(
                f"Need at least 2 arm samples with distinct timestamps to "
                f"interpolate, got {arm_ts_unique.size}"
            )
        pose_interp = PoseInterpolator(arm_ts_unique, arm_poses[unique_idx])
        arm_aligned = pose_interp(arm_obs_timestamps).astype(np.float32)
        eef_pos = arm_aligned[:, :3]
        eef_rot = arm_aligned[:, 3:]

        # `aperture_mm` is the module's calibrated opening. Re-deriving it from
        # the raw servo value would reintroduce the straight-line map that the
        # measured aperture table exists to replace.
        gripper_widths_m = np.array(
            [self._obs_width_m(float(g.aperture_mm.value)) for g in grip_hist],
            dtype=np.float32,
        ).reshape(-1, 1)
        grip_ts_unique, grip_unique_idx = np.unique(grip_ts, return_index=True)
        if grip_ts_unique.size < 2:
            raise RuntimeError(
                f"Need at least 2 gripper samples with distinct timestamps "
                f"to interpolate, got {grip_ts_unique.size}"
            )
        gripper_interp = get_interp1d(
            grip_ts_unique, gripper_widths_m[grip_unique_idx]
        )
        gripper_aligned = gripper_interp(gripper_obs_timestamps).astype(
            np.float32
        )

        return {
            "camera0_rgb": images,
            "robot0_eef_pos": eef_pos,
            "robot0_eef_rot_axis_angle": eef_rot,
            "robot0_gripper_width": gripper_aligned,
            "timestamp": camera_obs_timestamps,
            **extras,
        }

    def _check_targets(self, obs_meta: dict) -> None:
        """The yaml's ``targets`` read every target the checkpoint observes, from
        a camera field of the right data class that reaches where this runs."""
        kinds = target_keys(obs_meta)
        source = self.params["config"]
        if set(kinds) != set(self._targets):
            raise ValueError(
                f"The checkpoint observes targets {kinds or 'none'}, but {source} maps "
                f"{sorted(self._targets) or 'none'}; set targets: {{obs key: camera field}} "
                "for exactly the checkpoint's targets"
            )
        camera = self.params["camera"]
        fields = type(Observation[camera]).get_calculated_fields_spec()
        for key, field_name in self._targets.items():
            meta = obs_meta[key]
            if meta.get("target_rgb_key", "camera0_rgb") != "camera0_rgb":
                raise ValueError(
                    f"obs.{key} targets {meta['target_rgb_key']!r}; this skill drives one camera, camera0_rgb")
            if int(meta.get("horizon", self.camera_horizon)) != self.camera_horizon:
                raise ValueError(
                    f"obs.{key} horizon {meta['horizon']} differs from camera0_rgb's "
                    f"{self.camera_horizon}; one target is sent per frame")
            if field_name not in fields:
                raise ValueError(f"Camera '{camera}' has no calculated field '{field_name}' for obs.{key}")
            field = fields[field_name]
            data_class = TARGET_DATA[kinds[key]]
            if field.data_class is not data_class:
                raise ValueError(
                    f"obs.{key} is a {kinds[key]} target, which needs a {data_class.__name__} "
                    f"field, but {camera}.{field_name} is a {field.data_class.__name__}")
            # A field computed on the relay never reaches the robot's buffers;
            # one computed on the robot reaches both.
            if self.execution_place is ExecutionPlace.ROBOT and field.execution_place is not ExecutionPlace.ROBOT:
                raise ValueError(
                    f"{source} runs on the robot but {camera}.{field_name} is computed on the "
                    f"{field.execution_place.value}; set the field's execution_place to robot "
                    "or run the skill on the server")
            logger.info("Target obs.%s (%s) from %s.%s (computed on the %s; skill on the %s)",
                        key, kinds[key], camera, field_name, field.execution_place.value,
                        self.execution_place.value)

    def _camera_obs_extras(self, frames: list, frame_hw: tuple[int, int], remap) -> dict:
        """The targets, read off the camera observations the frames came from.

        ``frames`` are the camera ``ModuleObs`` picked for the image horizon,
        in that order; ``frame_hw`` is their capture size and ``remap`` the
        optical warp applied to them, or None.
        """
        if not self._targets:
            return {}
        pipe = self._ensure_policy_pipeline()
        warp = remap.apply if remap is not None else None
        kinds = target_keys(self.shape_meta["obs"])
        extras = {}
        for key, field_name in self._targets.items():
            kind = kinds[key]
            mapped = []
            for obs in frames:
                data = obs[field_name]
                if kind == "mask":
                    # A tracker result keeps the stamp of the frame it came
                    # from, so an unchanged stamp is an unchanged mask.
                    cache_key = (float(data.timestamp), frame_hw)
                else:
                    cache_key = (tuple(np.asarray(data.value).reshape(-1).tolist()), frame_hw)
                cached = self._target_cache.get(key)
                if cached is None or cached[0] != cache_key:
                    if kind == "mask":
                        target = pipe.target_mask(data.value, frame_hw, warp)
                    elif kind == "box":
                        target = pipe.bbox(np.asarray(data.value).reshape(4), frame_hw, warp)
                    else:
                        target = pipe.point(data.value, frame_hw, warp)
                    cached = self._target_cache[key] = (cache_key, target)
                mapped.append(cached[1])
            extras[key] = np.stack(mapped).astype(np.float32)
            if kind == "mask":
                extras[key] = extras[key][:, None]
                StatusBoard.set(f"UMI target {key} px", int(mapped[-1].sum()))
            else:
                StatusBoard.set(f"UMI target {key}", np.round(mapped[-1], 1).tolist())
        return extras

    # ------------------------------------------------------------------
    # Action handling
    # ------------------------------------------------------------------

    def exec_actions(
        self,
        actions: np.ndarray,
        timestamps: np.ndarray,
        now: float,
        compensate_latency: bool = True,
    ) -> list:
        """Turn a planned chunk into one waypoint action per predicted step.

        A chunk is a burst of setpoints stretching seconds into the future,
        not a single command for this instant, so the step returns a list and
        the modules' interpolators do the rest.
        """
        if len(actions) == 0:
            return []
        if len(actions) != len(timestamps):
            raise ValueError(
                f"actions/timestamps length mismatch: "
                f"{len(actions)} vs {len(timestamps)}"
            )

        arm_lat = self.arm_action_latency if compensate_latency else 0.0
        grip_lat = self.gripper_action_latency if compensate_latency else 0.0
        arm_ok = reachable(timestamps, arm_lat, now)
        grip_ok = reachable(timestamps, grip_lat, now)

        waypoints = []
        for action_vec, ts, send_arm, send_grip in zip(
            actions, timestamps, arm_ok, grip_ok
        ):
            ts = float(ts)
            data = {}

            if send_arm:
                pose = np.asarray(action_vec[:6], dtype=np.float64)
                data[self.params["arm"]] = {
                    "action_type": ActionType.SCHEDULE_L,
                    "data": {
                        field: {
                            "value": float(val),
                            "timestamp": float(ts - arm_lat),
                        }
                        for field, val in zip(self._arm_action_fields, pose)
                    },
                }
            if send_grip:
                data[self.params["gripper"]] = {
                    "action_type": ActionType.SCHEDULE_J,
                    "data": {
                        self._gripper_action_field: {
                            "value": float(self._cmd_width_mm(float(action_vec[6]))),
                            "timestamp": float(ts - grip_lat),
                        }
                    },
                }
            if data:
                waypoints.append(
                    Action(action_type=ActionType.MODULE_ACTION, data=data)
                )
        return waypoints

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(self) -> list:
        """Run a single policy inference and turn its chunk into waypoints."""
        t_start = time.perf_counter()
        obs = self.get_umi_obs()
        obs_timestamps = obs["timestamp"]

        obs_dict_np = get_real_umi_obs_dict(
            env_obs=obs,
            shape_meta=self.shape_meta,
            obs_pose_repr=self.obs_pose_repr,
            tx_robot1_robot0=self.tx_robot1_robot0,
            episode_start_pose=self.episode_start_pose,
        )

        self._dump_policy_input_image(obs_dict_np, env_obs=obs)
        prep_ms = (time.perf_counter() - t_start) * 1000.0 - self._t_buffers_ms

        t_predict = time.perf_counter()
        _, raw_action = self.policy_client.predict_with_raw(obs_dict_np)
        predict_ms = (time.perf_counter() - t_predict) * 1000.0
        if self.dump_policy_obs:
            self._dump_policy_obs(obs_dict_np, raw_action, env_obs=obs)
        action = get_real_umi_action(raw_action, obs, self.action_pose_repr)
        cmd_m = float(action[0, 6]) if action.size else float("nan")
        obs_m = float(obs["robot0_gripper_width"][-1])
        self._publish_gripper_status(obs_width_m=obs_m, cmd_width_m=cmd_m)
        if not self._logged_gripper:
            self._logged_gripper = True
            logger.info(
                "Gripper: obs=%.1f mm  policy_cmd=%.1f mm  module_cmd=%.1f mm "
                "(larger = more open; UMI max=%.1f mm)",
                obs_m * 1000.0,
                cmd_m * 1000.0,
                self._cmd_width_mm(cmd_m),
                self._policy_max_width_m() * 1000.0,
            )

        this_target_poses = action
        assert this_target_poses.shape[1] == len(self.arms_config) * 7

        if (
            self.discard_actions
            and len(this_target_poses) > self.steps_per_inference
        ):
            this_target_poses = this_target_poses[: self.steps_per_inference]
            action = action[: self.steps_per_inference]

        # solve_table_collision mutates ee_pose in place and returns nothing, so
        # the clamp is invisible unless z is read either side of it. Robot 0
        # only, to match the measured z the log line pairs these with.
        asked_z = float("inf")
        asked_z_width = float("nan")
        clamp_dz = 0.0
        for target_pose in this_target_poses:
            for arm_idx in range(len(self.arms_config)):
                ee_pose = target_pose[arm_idx * 7 : arm_idx * 7 + 6]
                width = float(target_pose[arm_idx * 7 + 6])
                z_asked = float(ee_pose[2])
                solve_table_collision(
                    ee_pose=ee_pose,
                    gripper_width=width,
                    height_threshold=self.arms_config[arm_idx][
                        "height_threshold"
                    ],
                )
                if arm_idx == 0:
                    if z_asked < asked_z:
                        asked_z, asked_z_width = z_asked, width
                    clamp_dz = max(clamp_dz, float(ee_pose[2]) - z_asked)
            solve_sphere_collision(
                ee_poses=target_pose.reshape([len(self.arms_config), -1]),
                arms_config=self.arms_config,
            )

        n_planned = len(this_target_poses)
        now = synced_time()
        this_target_poses, action_timestamps = self._prepare_chunk(
            this_target_poses,
            float(obs_timestamps[-1]),
            now,
        )

        waypoints = self.exec_actions(
            actions=this_target_poses,
            timestamps=action_timestamps,
            now=now,
            compensate_latency=True,
        )

        # Reads as: obs age is the anchor's distance into the past, and the
        # chunk is spaced policy_dt * execution_speed_scale, so every that
        # much of age deletes one more waypoint off the front. A first
        # waypoint far in the future means the arm is being handed the
        # middle of the predicted motion and never the approach to it.
        obs_age_ms = (now - float(obs_timestamps[-1])) * 1000.0
        # Per module, because their action_latency differs: the same chunk
        # reaches the gripper one or two waypoints deeper than the arm.
        arm_n, arm_lead_ms = self._module_chunk(
            action_timestamps, self.arm_action_latency, now
        )
        grip_n, grip_lead_ms = self._module_chunk(
            action_timestamps, self.gripper_action_latency, now
        )
        if arm_n:
            self._chunk_sent = True
        # predict is split because its three parts have nothing to do with
        # each other: encode is this board's CPU, transport is the link and
        # the payload size, model is the far end's GPU.
        timing = self.policy_client.last_timing
        logger.info(
            "Budget: buffers %.0f + prep %.0f + predict %.0f ms "
            "(encode %.0f + net %.0f + model %.0f, %.2f MB) | obs age "
            "%.0f ms | arm %d/%d first %+.0f ms, gripper %d/%d first %+.0f ms",
            self._t_buffers_ms,
            prep_ms,
            predict_ms,
            timing["encode_ms"],
            timing["transport_ms"],
            timing["server_ms"],
            timing["payload_mb"],
            obs_age_ms,
            arm_n,
            n_planned,
            arm_lead_ms,
            grip_n,
            n_planned,
            grip_lead_ms,
        )
        StatusBoard.set(
            "UMI obs age (ms)", self._obs_age.update(obs_age_ms).status_value()
        )
        StatusBoard.set(
            "UMI pipeline (ms)",
            self._pipeline.update(
                self._t_buffers_ms + prep_ms + predict_ms
            ).status_value(),
        )
        StatusBoard.set(
            "UMI chunk",
            f"arm {arm_n}/{n_planned} first {arm_lead_ms:+.0f} ms, "
            f"gripper {grip_n}/{n_planned} first {grip_lead_ms:+.0f} ms",
        )
        self._dropped_waypoints += n_planned - arm_n
        StatusBoard.set(
            f"UMI dropped waypoints ({self.execution_place.value})",
            self._dropped_waypoints,
        )
        # Reads as: the policy never asks low enough -> the camera-to-fingertip
        # lever arm is wrong (`solarian calibrate-hand-eye`); it asks low but
        # measured z never follows -> arm tracking or a truncated chunk; clamp
        # non-zero -> height_threshold is above the real table.
        logger.info(
            "Height: measured z %+.4f | policy asks %+.4f (gripper %.1f mm "
            "there) | table clamp +%.4f (threshold %+.4f) [m]",
            float(obs["robot0_eef_pos"][-1][2]),
            asked_z,
            asked_z_width * 1000.0,
            clamp_dz,
            float(self.arms_config[0]["height_threshold"]),
        )
        return waypoints

    def _module_chunk(
        self, timestamps: np.ndarray, latency: float, now: float
    ) -> tuple[int, float]:
        """How much of the chunk one module gets, and when its first is due.

        The lead is measured against the uncompensated time, so it reads as
        the distance into the future the module is being aimed at rather than
        the moment the message has to leave.
        """
        landing = timestamps[reachable(timestamps, latency, now)]
        if not len(landing):
            return 0, float("nan")
        return len(landing), (float(landing[0]) - now) * 1000.0

    def _prepare_chunk(
        self,
        poses: np.ndarray,
        obs_last_timestamp: float,
        curr_time: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Timestamp like UMI eval, then drop steps that are already past.

        First action is at ``obs_last_timestamp`` (``arange(len)``), except
        for the first chunk of a run, which starts after ``curr_time``. Over
        budget → last pose on the next ``1/frequency`` tick from
        ``_eval_t_start``.
        """
        if len(poses) == 0:
            return poses, np.array([], dtype=np.float64)

        anchor = obs_last_timestamp
        if not self._chunk_sent:
            # Valid only because the arm stood still through this inference,
            # so it is still at the pose the chunk is relative to. Anchored at
            # the observation, the steps it never ran would be dropped and
            # the arm would cover them in one dash at its speed cap.
            lead = max(self.arm_action_latency, self.gripper_action_latency)
            anchor = max(anchor, curr_time + lead + self.policy_dt)
        dt_execution = self.policy_dt * self.execution_speed_scale
        timestamps = (
            np.arange(len(poses), dtype=np.float64) * dt_execution + anchor
        )
        # 10 ms buffer matches eval_real ``action_exec_latency``, which is
        # honest for a UR5 on RTDE and a floor here: each module's own
        # action_latency is applied in ``exec_actions``. Kept because the
        # over-budget fallback below has to know the chunk is wholly past,
        # which is a property of the chunk and not of any one module.
        is_new = timestamps > (curr_time + 0.01)
        if not np.any(is_new):
            next_step_idx = int(
                np.ceil((curr_time - self._eval_t_start) / self.policy_dt)
            )
            action_timestamp = self._eval_t_start + next_step_idx * self.policy_dt
            logger.warning(
                "UMISkill over budget by %.3f s — last action in %.3f s",
                curr_time - timestamps[-1],
                action_timestamp - curr_time,
            )
            return poses[[-1]], np.array([action_timestamp], dtype=np.float64)
        return poses[is_new], timestamps[is_new]

    # ------------------------------------------------------------------
    # Skill loop
    # ------------------------------------------------------------------

    def control_step(self, dt):
        """Re-infer on UMI's overlapping cadence.

        UMI waits ``steps_per_inference * dt`` from each cycle start
        (``t_cycle_end``) and infers again while the previous chunk is still
        executing. Advance the due-time only after a successful inference so
        a failed call retries on the next 10 Hz tick instead of going silent
        for a full period.
        """
        if self._startup is not None and self._startup[-1][0] == "first observation":
            self._startup_mark("first step")
        if time.monotonic() < self._next_inference_mono:
            return None
        try:
            waypoints = self._run_inference()
        except Exception:
            logger.exception("UMISkill inference failed")
            return None
        if self._startup is not None:
            self._startup_mark("first chunk")
            logger.info("UMISkill start-up: %s", ", ".join(
                f"{phase} {(t - before) * 1e3:.0f} ms"
                for (_, before), (phase, t) in zip(self._startup, self._startup[1:])))
            self._startup = None

        period = self.steps_per_inference * self.policy_dt
        self._next_inference_mono += period
        # Staying on the grid means an overrun leaves the due-time already
        # past, and the next tick fires immediately — back-to-back inference
        # exactly when the board is short of CPU. That is UMI's cadence and
        # is left alone; it is reported because it is the shape a starved
        # board makes, and the chunk arriving late is what the arm feels.
        slip = time.monotonic() - self._next_inference_mono
        if slip > 0.0:
            logger.warning(
                "UMISkill is %.2f s behind its %.2f s inference cycle; the "
                "next one starts with no wait",
                slip,
                period,
            )
        return waypoints

    def teardown(self):
        proc = getattr(self, "_managed_server_proc", None)
        if proc is not None and proc.poll() is None:
            logger.info("Stopping managed policy server (pid=%s)", proc.pid)
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except Exception:
                proc.kill()
