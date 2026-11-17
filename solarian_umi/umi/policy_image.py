"""Capture RGB → policy 224×224 (UMI resize-then-center-crop + gripper mask).

Shared by ``UMISkill`` and ``solarian calibrate-umi-camera`` so the
live overlay matches ``compare.png`` without constructing the skill
(which waits on the policy server).
"""

from __future__ import annotations

import numpy as np

from solarian_umi.umi.common.cv_util import draw_predefined_mask
from solarian_umi.umi.common.cv2_util import get_image_transform, letterbox_to_square

DEFAULT_OUTPUT_RES = (224, 224)
#: Capture pixels. Large enough to survive the downscale to the policy frame,
#: small enough that the crop edge rarely cuts it and shifts the centroid.
POINT_DISK_RADIUS_PX = 6


class PolicyImagePipeline:
    """Stateful crop/mask helper (caches ``get_image_transform``)."""

    def __init__(
        self,
        output_res: tuple[int, int] = DEFAULT_OUTPUT_RES,
        *,
        letterbox_to_square: bool = False,
        no_mirror: bool = True,
        mirror_swap: bool = False,
    ):
        self.output_res = (int(output_res[0]), int(output_res[1]))
        self.letterbox_to_square = bool(letterbox_to_square)
        self.no_mirror = bool(no_mirror)
        self.mirror_swap = bool(mirror_swap)
        self._tf = None
        self._tf_key = None
        self._mirror_swap_is_mirror = None
        self._mirror_swap_mask_cache_key = None

    def resize(self, images: np.ndarray) -> np.ndarray:
        """Map capture frames to policy RGB. ``images`` is (T, H, W, C) uint8."""
        import cv2

        ow, oh = self.output_res
        out = []
        for frame in images:
            if self.letterbox_to_square:
                frame = letterbox_to_square(frame)
            elif (frame.shape[1], frame.shape[0]) != (ow, oh):
                key = (frame.shape[1], frame.shape[0], ow, oh)
                if self._tf_key != key:
                    self._tf = get_image_transform(
                        input_res=(frame.shape[1], frame.shape[0]),
                        output_res=(ow, oh),
                        bgr_to_rgb=False,
                    )
                    self._tf_key = key
                frame = self._tf(frame)
            if (frame.shape[1], frame.shape[0]) != (ow, oh):
                frame = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
            out.append(frame)
        return np.stack(out)

    def mask(self, images: np.ndarray) -> np.ndarray:
        """Black out GoPro gripper body + side mirrors on square frames."""
        out = np.ascontiguousarray(images.copy())
        h, w = out.shape[1:3]

        if self.mirror_swap:
            cache_key = (h, w)
            if self._mirror_swap_mask_cache_key != cache_key:
                tmp = np.ones((h, w, 3), dtype=np.uint8)
                draw_predefined_mask(
                    tmp, color=(0, 0, 0),
                    mirror=True, gripper=False, finger=False,
                )
                self._mirror_swap_is_mirror = tmp[..., 0] == 0
                self._mirror_swap_mask_cache_key = cache_key
            is_mirror = self._mirror_swap_is_mirror
            for i in range(out.shape[0]):
                frame = out[i]
                frame[is_mirror] = frame[:, ::-1, :][is_mirror]

        for i in range(out.shape[0]):
            draw_predefined_mask(
                out[i],
                color=(0, 0, 0),
                mirror=self.no_mirror,
                gripper=True,
                finger=False,
                use_aa=True,
            )
        return out

    def bbox(self, bbox, frame_hw: tuple[int, int], warp=None) -> np.ndarray:
        """A capture-pixel box as the policy sees it: ``[x0, y0, x1, y1]`` float32.

        The box is drawn as a mask and sent through the same ``warp`` (the
        optical remap, which is not linear) and :meth:`resize` as the frame,
        so it covers what it covered before. An empty box, or one the crop
        removes, becomes the whole policy frame: the placeholder the
        mask-channel policy was trained on for "no target".
        """
        import cv2

        ow, oh = self.output_res
        full = np.array([0, 0, ow, oh], dtype=np.float32)
        h, w = frame_hw
        x0, y0, x1, y1 = (int(v) for v in np.asarray(bbox).reshape(4))
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return full
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255
        if warp is not None:
            mask = warp(mask)
        mask = self.resize(mask[None])[0][..., 0]
        x, y, bw, bh = cv2.boundingRect((mask > 127).astype(np.uint8))
        if bw <= 0 or bh <= 0:
            return full
        return np.array([x, y, x + bw, y + bh], dtype=np.float32)

    def point(self, point, frame_hw: tuple[int, int], warp=None) -> np.ndarray:
        """A capture-pixel point as the policy sees it: ``[x, y]`` float32.

        ``point`` is a ``PointData`` value ``(x, y, visibility)``; only ``x, y``
        are read. A small disk around it goes through the same ``warp`` and
        :meth:`resize` as the frame and the point is the centroid of what
        arrives. All zeros (no point yet), or a point the crop removes, is
        ``[-1, -1]``: the dataset's "no point".
        """
        import cv2

        none = np.array([-1.0, -1.0], dtype=np.float32)
        raw = np.asarray(point, dtype=np.float64).reshape(-1)
        h, w = frame_hw
        x, y = raw[0], raw[1]
        if not raw.any() or not (0 <= x < w and 0 <= y < h):
            return none
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(img, (int(round(x)), int(round(y))), POINT_DISK_RADIUS_PX, (255, 255, 255), -1)
        if warp is not None:
            img = warp(img)
        weight = self.resize(img[None])[0][..., 0].astype(np.float64)
        total = weight.sum()
        if total <= 0:
            return none
        ys, xs = np.indices(weight.shape)
        return np.array([(xs * weight).sum() / total, (ys * weight).sum() / total], dtype=np.float32)

    def target_mask(self, mask, frame_hw: tuple[int, int], warp=None) -> np.ndarray:
        """A capture-pixel target mask as the policy sees it: ``(oh, ow)`` float32 0/1.

        Sent through the same ``warp`` and :meth:`resize` as the frame, then
        cleared where :meth:`mask` paints the gripper and mirrors black: the
        training masks were segmented on frames already painted. None or an
        empty mask stays all zeros, the "no target" the dataset stores for a
        frame without a box.
        """
        ow, oh = self.output_res
        empty = np.zeros((oh, ow), dtype=np.float32)
        if mask is None:
            return empty
        mask = np.asarray(mask)
        if mask.shape != tuple(frame_hw):
            raise ValueError(
                f"target mask is {mask.shape} but the frame is {tuple(frame_hw)}; "
                "the mask must come from the camera field the policy sees"
            )
        if not mask.any():
            return empty
        img = np.repeat((mask > 0).astype(np.uint8)[..., None] * 255, 3, axis=-1)
        if warp is not None:
            img = warp(img)
        img = self.resize(img[None])
        visible = self.mask(np.full_like(img, 255))
        return ((img[0, ..., 0] > 127) & (visible[0, ..., 0] > 127)).astype(np.float32)

    def __call__(self, images: np.ndarray) -> np.ndarray:
        return self.mask(self.resize(images))

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Single HWC uint8 capture → policy HWC uint8."""
        if frame.ndim == 3:
            images = frame[None]
        else:
            images = frame
        return self(images)[0]
