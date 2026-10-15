"""One GoPro-geometry video frame to the image the UMI policy trains on.

Shared by the zarr writer (``07_generate_replay_buffer.py``) and the solarian
session loader, so a session decoded at train time yields the same pixels the
zarr would have stored.
"""

from __future__ import annotations

from typing import Callable, Iterable

import cv2
import numpy as np

from solarian_umi_skill.slam.support.cv_util import (
    draw_predefined_mask,
    get_image_transform,
    inpaint_tag,
)


def crop_window(in_res: tuple[int, int], out_res: tuple[int, int]) -> tuple[int, int, int, int]:
    """``(x0, y0, w, h)`` of the source region :func:`get_image_transform` keeps."""
    iw, ih = in_res
    ow, oh = out_res
    ch = round(ih)
    cw = round(ih / oh * ow)
    return (iw - cw) // 2, (ih - ch) // 2, cw, ch


class PolicyFrame:
    """Inpaint ArUco tags, black out the gripper, crop and resize.

    ``in_res`` and ``out_res`` are ``(width, height)``. ``resize`` replaces the
    centre crop and resize, e.g. with a fisheye rectifier.
    """

    def __init__(
        self,
        in_res: tuple[int, int],
        out_res: tuple[int, int],
        no_mirror: bool = False,
        mirror_swap: bool = False,
        resize: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.in_res = tuple(in_res)
        self.out_res = tuple(out_res)
        self.no_mirror = no_mirror
        self.resize = resize or get_image_transform(in_res=in_res, out_res=out_res)
        self.is_mirror = None
        if mirror_swap:
            ow, oh = out_res
            mirror_mask = np.ones((oh, ow, 3), dtype=np.uint8)
            mirror_mask = draw_predefined_mask(
                mirror_mask, color=(0, 0, 0), mirror=True, gripper=False, finger=False)
            self.is_mirror = mirror_mask[..., 0] == 0

    def __call__(self, img: np.ndarray, tag_corners: Iterable[np.ndarray]) -> np.ndarray:
        for corners in tag_corners:
            img = inpaint_tag(img, np.asarray(corners))
        img = draw_predefined_mask(
            img, color=(0, 0, 0), mirror=self.no_mirror, gripper=True, finger=False)
        img = self.resize(img)
        if self.is_mirror is not None:
            img[self.is_mirror] = img[:, ::-1, :][self.is_mirror]
        return img

    def box(self, bbox: np.ndarray) -> np.ndarray:
        """Source-pixel boxes ``(..., 4+)`` in output pixels.

        Columns past ``(x0, y0, x1, y1)`` pass through. A row with a negative
        coordinate is primarykey's "no box" marker and is kept as it is.
        """
        x0, y0, cw, ch = crop_window(self.in_res, self.out_res)
        ow, oh = self.out_res
        bbox = np.asarray(bbox, dtype=np.float32)
        out = bbox.copy()
        scale = np.array([ow / cw, oh / ch, ow / cw, oh / ch], dtype=np.float32)
        xy = (bbox[..., :4] - np.array([x0, y0, x0, y0], dtype=np.float32)) * scale
        out[..., :4] = np.clip(xy, 0, np.array([ow, oh, ow, oh], dtype=np.float32))
        invalid = (bbox[..., :4] < 0).any(axis=-1)
        out[invalid] = bbox[invalid]
        return out

    def mask(self, mask: np.ndarray) -> np.ndarray:
        """A source-resolution ``(H, W)`` 0/1 mask, cropped and resized like the image."""
        x0, y0, cw, ch = crop_window(self.in_res, self.out_res)
        crop = np.ascontiguousarray(mask[y0:y0 + ch, x0:x0 + cw], dtype=np.uint8)
        return cv2.resize(crop, self.out_res, interpolation=cv2.INTER_NEAREST)
