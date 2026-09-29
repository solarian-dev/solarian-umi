"""Capture RGB → policy 224×224 (UMI resize-then-center-crop + gripper mask).

Shared by ``UMISkill`` and ``solarian calibrate-umi-camera`` so the
live overlay matches ``compare.png`` without constructing the skill
(which waits on the policy server).
"""

from __future__ import annotations

import numpy as np

from solarian_umi_skill.umi.common.cv_util import draw_predefined_mask
from solarian_umi_skill.umi.common.cv2_util import get_image_transform, letterbox_to_square

DEFAULT_OUTPUT_RES = (224, 224)


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

    def __call__(self, images: np.ndarray) -> np.ndarray:
        return self.mask(self.resize(images))

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Single HWC uint8 capture → policy HWC uint8."""
        if frame.ndim == 3:
            images = frame[None]
        else:
            images = frame
        return self(images)[0]
