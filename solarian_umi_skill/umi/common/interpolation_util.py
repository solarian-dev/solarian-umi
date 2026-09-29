"""Vendored from the original UMI ``umi.common.interpolation_util`` module.

These helpers are used by :class:`solarian_umi_skill.skill.UMISkill` to resample
asynchronous proprioceptive streams onto the camera-anchored
observation timestamps that the diffusion policy was trained against.
"""

import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st


def get_interp1d(t, x):
    """1D linear interpolator with constant extrapolation past the ends.

    ``bounds_error=False`` plus ``fill_value=(x[0], x[-1])`` mirrors how the
    UMI codebase resamples gripper widths: out-of-range queries are clamped
    to the first / last sample rather than raising or producing NaN.
    """
    return si.interp1d(
        t, x,
        axis=0, bounds_error=False,
        fill_value=(x[0], x[-1]),
    )


class PoseInterpolator:
    """Linear-on-position, Slerp-on-rotation interpolator for 6-DoF poses.

    Inputs:
        t : (N,) array of timestamps
        x : (N, 6) array of poses laid out as ``[x, y, z, rx, ry, rz]``
            with the rotation expressed as a rotation vector.

    Calling the instance at one or more query timestamps returns a
    ``(..., 6)`` array in the same layout.  Query timestamps are clipped
    to the input range so out-of-range queries return the boundary pose
    instead of raising — same convention as :func:`get_interp1d`.
    """

    def __init__(self, t, x):
        pos = x[:, :3]
        rot = st.Rotation.from_rotvec(x[:, 3:])
        self.pos_interp = get_interp1d(t, pos)
        self.rot_interp = st.Slerp(t, rot)

    @property
    def x(self):
        return self.pos_interp.x

    def __call__(self, t):
        min_t = self.pos_interp.x[0]
        max_t = self.pos_interp.x[-1]
        t = np.clip(t, min_t, max_t)

        pos = self.pos_interp(t)
        rot = self.rot_interp(t)
        rvec = rot.as_rotvec()
        return np.concatenate([pos, rvec], axis=-1)
