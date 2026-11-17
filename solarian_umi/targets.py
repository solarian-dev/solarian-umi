"""Which ``shape_meta`` obs entries are targets, and of what kind.

A target marks something on an rgb image (the object to pick, the place to
put it) and reaches the ViT as one extra input channel of that image
(``target_rgb_key``), in shape_meta order:

* ``box``: ``low_dim`` shape ``[4]`` with ``bbox_mode: mask_channel``,
  ``x0, y0, x1, y1`` policy pixels drawn as a filled rectangle;
* ``point``: ``low_dim`` shape ``[2]`` with ``point_mode: mask_channel``,
  ``x, y`` policy pixels drawn as a ``point_box_px`` square centred on it;
* ``mask``: ``type: mask`` shape ``[1, H, W]``, a 0/1 segmentation used as is.

An entry trains on the dataset array of its own name; a solarian session
holds ``bbox``, ``point`` and ``mask`` (``track`` and ``track_point``). Kept
free of torch so a skill can read a checkpoint's targets on a robot.
"""

from __future__ import annotations

from typing import Literal, Mapping

TargetKind = Literal["box", "point", "mask"]

#: Side of the square a point is drawn as, in policy pixels, when its entry does not say.
DEFAULT_POINT_BOX_PX = 16


def target_kind(attr: Mapping) -> TargetKind | None:
    """The kind of target an obs entry is, None for an ordinary one."""
    kind = attr.get("type", "low_dim")
    if kind == "mask":
        return "mask"
    if kind == "low_dim" and attr.get("bbox_mode", "low_dim") == "mask_channel":
        return "box"
    if kind == "low_dim" and attr.get("point_mode") == "mask_channel":
        return "point"
    return None


def target_keys(obs_meta: Mapping[str, Mapping]) -> dict[str, TargetKind]:
    """``{key: kind}`` of the target entries of ``shape_meta['obs']``, in its order."""
    return {key: kind for key, attr in obs_meta.items() if (kind := target_kind(attr)) is not None}
