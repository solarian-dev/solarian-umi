"""UMIPnPSkill — point and pick: a UMI policy conditioned on a target box.

The checkpoint was trained with ``bbox_mode: mask_channel`` (UMI a24410f):
the ViT patch embedding takes a fourth input channel, a binary mask of the
target box drawn over ``camera0_rgb``. At run time the box is the camera's
``bbox`` calculated field — typically a tracker the operator seeded by
clicking the object in the controller (``start_service``) — mapped into the
policy image the same way the frame is.

Before the first click the field is all zeros and the policy gets the whole
frame as its box, the placeholder it was trained on for "no target". Starting
the skill does not wait for a target.

Where it runs is the yaml's required ``execution_place``: on the relay
server, beside the tracker, the robot only captures and executes. It must
agree with where the box is computed, so both confs read one deployment key.
"""

import logging

import numpy as np

from solarian.core.execution import ExecutionPlace
from solarian.core.observation import Observation
from solarian.utils.status_board import StatusBoard

from solarian_umi_skill.skill import UMISkill

logger = logging.getLogger(__name__)


class UMIPnPSkill(UMISkill):
    """A UMI policy that picks the object inside the camera's tracked box."""

    config_name = "umi_pnp"
    execution_place = ExecutionPlace.SERVER
    skill_id = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if kwargs.get("execution_place") is None and not self.config.get("execution_place"):
            raise ValueError(
                f"{self.params.get('config') or 'umi_pnp.yaml'} must set "
                "execution_place (server or robot), the same place its bbox "
                "field is computed"
            )
        # Set here, not in setup: the base setup builds the first
        # observation, and with it the box, before returning.
        self._bbox_field = str(self.config.get("bbox_field", "bbox"))
        self._bbox_cache: tuple[tuple, np.ndarray] | None = None

    def setup(self):
        super().setup()
        meta = self.shape_meta["obs"].get("bbox")
        if meta is None or meta.get("bbox_mode") != "mask_channel":
            raise ValueError(
                "UMIPnPSkill needs a checkpoint trained with obs.bbox "
                "bbox_mode=mask_channel; this one's shape_meta has "
                f"{meta!r}. Use UMISkill for a policy without a target box."
            )
        if meta.get("target_rgb_key", "camera0_rgb") != "camera0_rgb":
            raise ValueError(
                f"obs.bbox targets {meta['target_rgb_key']!r}; this skill "
                "drives one camera, camera0_rgb"
            )
        if int(meta.get("horizon", self.camera_horizon)) != self.camera_horizon:
            raise ValueError(
                f"obs.bbox horizon {meta['horizon']} differs from camera0_rgb's "
                f"{self.camera_horizon}; one box is sent per frame"
            )
        camera = self.params["camera"]
        fields = type(Observation[camera]).get_calculated_fields_spec()
        if self._bbox_field not in fields:
            raise ValueError(
                f"Camera '{camera}' has no calculated field "
                f"'{self._bbox_field}' to read the target box from"
            )
        # A field computed on the relay never reaches the robot's buffers;
        # one computed on the robot reaches both.
        field_place = fields[self._bbox_field].execution_place
        if self.execution_place is ExecutionPlace.ROBOT and field_place is not ExecutionPlace.ROBOT:
            raise ValueError(
                f"UMIPnPSkill runs on the robot but {camera}.{self._bbox_field} "
                f"is computed on the {field_place.value}; set the field's "
                "execution_place to robot or run the skill on the server"
            )
        logger.info(
            "Target box from %s.%s (computed on the %s; skill on the %s)",
            camera,
            self._bbox_field,
            field_place.value,
            self.execution_place.value,
        )

    def _camera_obs_extras(self, frames: list, frame_hw: tuple[int, int], remap) -> dict:
        pipe = self._ensure_policy_pipeline()
        warp = remap.apply if remap is not None else None
        boxes = []
        for obs in frames:
            raw = np.asarray(obs[self._bbox_field].value).reshape(4)
            key = (tuple(int(v) for v in raw), frame_hw)
            # The box changes far less often than inference runs, and mapping
            # it costs a full-frame warp.
            if self._bbox_cache is None or self._bbox_cache[0] != key:
                self._bbox_cache = (key, pipe.bbox(raw, frame_hw, warp))
            boxes.append(self._bbox_cache[1])
        StatusBoard.set("UMI PnP target", np.asarray(frames[-1][self._bbox_field].value).tolist())
        return {"bbox": np.stack(boxes).astype(np.float32)}
