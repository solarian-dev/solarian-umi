"""UMIPnPSkill — point and pick: a UMI policy conditioned on a target.

The checkpoint's ViT patch embedding takes a fourth input channel marking the
target over ``camera0_rgb``: a box drawn as a filled rectangle (``obs.bbox``,
``bbox_mode: mask_channel``, UMI a24410f), or the object's segmentation
(``obs.mask``, ``type: mask``; trained with ``target: mask``). At run time
the target is the camera's calculated field named by ``target_field`` — a
``BoundingBoxData`` or a ``MaskData`` from a tracker the operator seeded by
clicking the object in the controller (``start_service``) — mapped into the
policy image the same way the frame is.

Before the first click a box is all zeros and the policy gets the whole frame
as its box; a mask is None and the policy gets an empty mask, the dataset's
"no target". Starting the skill does not wait for a target.

Where it runs is the yaml's required ``execution_place``: on the relay
server, beside the tracker, the robot only captures and executes. It must
agree with where the target is computed, so both confs read one deployment key.
"""

import logging

import numpy as np

from solarian.core.data import BoundingBoxData, MaskData
from solarian.core.execution import ExecutionPlace
from solarian.core.observation import Observation
from solarian.utils.status_board import StatusBoard

from solarian_umi_skill.skill import UMISkill

logger = logging.getLogger(__name__)


class UMIPnPSkill(UMISkill):
    """A UMI policy that picks the object the camera's tracker follows."""

    config_name = "umi_pnp"
    execution_place = ExecutionPlace.SERVER
    skill_id = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if kwargs.get("execution_place") is None and not self.config.get("execution_place"):
            raise ValueError(
                f"{self.params.get('config') or 'umi_pnp.yaml'} must set "
                "execution_place (server or robot), the same place its target "
                "field is computed"
            )
        # Set here, not in setup: the base setup builds the first
        # observation, and with it the target, before returning.
        self._target_field = str(self.config.get("target_field", "bbox"))
        #: ``(key, policy-frame target)``; the target changes far less often
        #: than inference runs, and mapping it costs a full-frame warp.
        self._target_cache: tuple[tuple, np.ndarray] | None = None

    def _uses_mask(self) -> bool:
        """The checkpoint conditions on ``obs.mask`` rather than ``obs.bbox``."""
        return self.shape_meta["obs"].get("mask", {}).get("type") == "mask"

    def setup(self):
        super().setup()
        obs_meta = self.shape_meta["obs"]
        if self._uses_mask():
            key, data_class = "mask", MaskData
        elif obs_meta.get("bbox", {}).get("bbox_mode") == "mask_channel":
            key, data_class = "bbox", BoundingBoxData
        else:
            raise ValueError(
                "UMIPnPSkill needs a checkpoint trained with a target: obs.bbox "
                "bbox_mode=mask_channel or obs.mask type=mask; this one's "
                f"shape_meta has bbox={obs_meta.get('bbox')!r}, "
                f"mask={obs_meta.get('mask')!r}. Use UMISkill for a policy "
                "without a target."
            )
        meta = obs_meta[key]
        if meta.get("target_rgb_key", "camera0_rgb") != "camera0_rgb":
            raise ValueError(
                f"obs.{key} targets {meta['target_rgb_key']!r}; this skill "
                "drives one camera, camera0_rgb"
            )
        if int(meta.get("horizon", self.camera_horizon)) != self.camera_horizon:
            raise ValueError(
                f"obs.{key} horizon {meta['horizon']} differs from camera0_rgb's "
                f"{self.camera_horizon}; one target is sent per frame"
            )
        camera = self.params["camera"]
        fields = type(Observation[camera]).get_calculated_fields_spec()
        if self._target_field not in fields:
            raise ValueError(
                f"Camera '{camera}' has no calculated field "
                f"'{self._target_field}' to read the target from"
            )
        field = fields[self._target_field]
        if field.data_class is not data_class:
            raise ValueError(
                f"The checkpoint conditions on obs.{key}, which needs a "
                f"{data_class.__name__} field, but {camera}.{self._target_field} "
                f"is a {field.data_class.__name__}; set target_field in "
                f"{self.params.get('config') or 'umi_pnp.yaml'}"
            )
        # A field computed on the relay never reaches the robot's buffers;
        # one computed on the robot reaches both.
        if self.execution_place is ExecutionPlace.ROBOT and field.execution_place is not ExecutionPlace.ROBOT:
            raise ValueError(
                f"UMIPnPSkill runs on the robot but {camera}.{self._target_field} "
                f"is computed on the {field.execution_place.value}; set the field's "
                "execution_place to robot or run the skill on the server"
            )
        logger.info(
            "Target %s from %s.%s (computed on the %s; skill on the %s)",
            key,
            camera,
            self._target_field,
            field.execution_place.value,
            self.execution_place.value,
        )

    def _camera_obs_extras(self, frames: list, frame_hw: tuple[int, int], remap) -> dict:
        pipe = self._ensure_policy_pipeline()
        warp = remap.apply if remap is not None else None
        if self._uses_mask():
            masks = []
            for obs in frames:
                data = obs[self._target_field]
                # A tracker result keeps the stamp of the frame it came from,
                # so an unchanged stamp is an unchanged mask.
                key = ("mask", float(data.timestamp), frame_hw)
                if self._target_cache is None or self._target_cache[0] != key:
                    self._target_cache = (key, pipe.target_mask(data.value, frame_hw, warp))
                masks.append(self._target_cache[1])
            StatusBoard.set("UMI PnP target px", int(masks[-1].sum()))
            return {"mask": np.stack(masks)[:, None].astype(np.float32)}
        boxes = []
        for obs in frames:
            raw = np.asarray(obs[self._target_field].value).reshape(4)
            key = ("bbox", tuple(int(v) for v in raw), frame_hw)
            if self._target_cache is None or self._target_cache[0] != key:
                self._target_cache = (key, pipe.bbox(raw, frame_hw, warp))
            boxes.append(self._target_cache[1])
        StatusBoard.set("UMI PnP target", np.asarray(frames[-1][self._target_field].value).tolist())
        return {"bbox": np.stack(boxes).astype(np.float32)}
