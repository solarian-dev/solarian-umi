"""Subset of ``umi.common.cv_util`` containing only the image-mask helpers
needed at inference time.

These polygon helpers describe regions of the original UMI GoPro
training image (a 2028x2704 fish-eye crop) in *canonical* coordinates
— i.e. coordinates that have been normalized by the image height so
they survive arbitrary resize operations.  Three regions are defined:

* ``mirror``  — the two side mirrors that the GoPro UMI gripper has
  attached to it.  These mirrors show the operator's body during data
  collection; they are absent on a non-GoPro wrist camera.
* ``gripper`` — the GoPro-mounted gripper jaws that occupy the
  foreground of every training frame.  Always painted black at
  inference time so the policy doesn't try to attend to whatever
  gripper is in *its* foreground.
* ``finger``  — a narrower gripper-tip mask used by some policies.

At inference time the canonical coordinates are scaled to whatever
image resolution we happen to be working with via
``canonical_to_pixel_coords`` and filled with ``cv2.fillPoly``.
"""

from __future__ import annotations

import cv2
import numpy as np


def canonical_to_pixel_coords(
    coords: np.ndarray, img_shape: tuple[int, int] = (2028, 2704)
) -> np.ndarray:
    pts = np.asarray(coords) * img_shape[0] + np.array(img_shape[::-1]) * 0.5
    return pts


def pixel_coords_to_canonical(
    pts: np.ndarray, img_shape: tuple[int, int] = (2028, 2704)
) -> np.ndarray:
    coords = (np.asarray(pts) - np.array(img_shape[::-1]) * 0.5) / img_shape[0]
    return coords


def get_mirror_canonical_polygon() -> np.ndarray:
    left_pts = [
        [540, 1700],
        [680, 1450],
        [590, 1070],
        [290, 1130],
        [290, 1770],
        [550, 1770],
    ]
    resolution = [2028, 2704]
    left_coords = pixel_coords_to_canonical(left_pts, resolution)
    right_coords = left_coords.copy()
    right_coords[:, 0] *= -1
    coords = np.stack([left_coords, right_coords])
    return coords


def get_gripper_canonical_polygon() -> np.ndarray:
    left_pts = [
        [1352, 1730],
        [1100, 1700],
        [650, 1500],
        [0, 1350],
        [0, 2028],
        [1352, 2704],
    ]
    resolution = [2028, 2704]
    left_coords = pixel_coords_to_canonical(left_pts, resolution)
    right_coords = left_coords.copy()
    right_coords[:, 0] *= -1
    coords = np.stack([left_coords, right_coords])
    return coords


def get_finger_canonical_polygon(
    height: float = 0.37, top_width: float = 0.25, bottom_width: float = 1.4
) -> np.ndarray:
    resolution = [2028, 2704]
    img_h, img_w = resolution

    top_y = 1.0 - height
    bottom_y = 1.0
    width = img_w / img_h
    middle_x = width / 2.0
    top_left_x = middle_x - top_width / 2.0
    top_right_x = middle_x + top_width / 2.0
    bottom_left_x = middle_x - bottom_width / 2.0
    bottom_right_x = middle_x + bottom_width / 2.0

    top_y *= img_h
    bottom_y *= img_h
    top_left_x *= img_h
    top_right_x *= img_h
    bottom_left_x *= img_h
    bottom_right_x *= img_h

    points = [[
        [bottom_left_x, bottom_y],
        [top_left_x, top_y],
        [top_right_x, top_y],
        [bottom_right_x, bottom_y],
    ]]
    coords = pixel_coords_to_canonical(points, img_shape=resolution)
    return coords


def draw_predefined_mask(
    img: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 0),
    mirror: bool = True,
    gripper: bool = True,
    finger: bool = True,
    use_aa: bool = False,
) -> np.ndarray:
    """Paint over the requested predefined regions of ``img`` in-place.

    ``mirror`` paints the GoPro UMI side-mirror polygons.
    ``gripper`` paints the GoPro UMI gripper-jaw foreground polygons.
    ``finger`` paints the narrower finger-tip polygon.

    Polygons are scaled to ``img.shape[:2]`` from canonical coordinates,
    so this works at any image resolution.
    """
    all_coords = list()
    if mirror:
        all_coords.extend(get_mirror_canonical_polygon())
    if gripper:
        all_coords.extend(get_gripper_canonical_polygon())
    if finger:
        all_coords.extend(get_finger_canonical_polygon())

    for coords in all_coords:
        pts = canonical_to_pixel_coords(coords, img.shape[:2])
        pts = np.round(pts).astype(np.int32)
        flag = cv2.LINE_AA if use_aa else cv2.LINE_8
        cv2.fillPoly(img, [pts], color=color, lineType=flag)
    return img
