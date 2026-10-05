"""Kannala-Brandt fisheye camera as unit rays, and remaps between two cameras.

Rays rather than pinhole coordinates: the GoPro Max Lens frame reaches past
90 degrees off axis, where x/z is undefined. Distortion is OpenCV's fisheye
model, theta_d = theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Fisheye:
    fx: float
    fy: float
    cx: float
    cy: float
    k: tuple[float, float, float, float]
    width: int
    height: int

    @classmethod
    def from_umi_json(cls, data: dict) -> "Fisheye":
        intr = data["intrinsics"]
        f = float(intr["focal_length"])
        return cls(
            fx=f,
            fy=f * float(intr["aspect_ratio"]),
            cx=float(intr["principal_pt_x"]),
            cy=float(intr["principal_pt_y"]),
            k=(
                float(intr["radial_distortion_1"]),
                float(intr["radial_distortion_2"]),
                float(intr["radial_distortion_3"]),
                float(intr["radial_distortion_4"]),
            ),
            width=int(data["image_width"]),
            height=int(data["image_height"]),
        )

    @classmethod
    def from_opencv(cls, K: np.ndarray, D: np.ndarray, width: int, height: int) -> "Fisheye":
        d = np.asarray(D, dtype=np.float64).reshape(-1)
        return cls(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            k=(float(d[0]), float(d[1]), float(d[2]), float(d[3])),
            width=int(width),
            height=int(height),
        )

    def to_umi_json(self, *, rms: float, n_views: int, fps: float) -> dict:
        return {
            "final_reproj_error": float(rms),
            "fps": float(fps),
            "image_height": self.height,
            "image_width": self.width,
            "intrinsic_type": "FISHEYE",
            "intrinsics": {
                "aspect_ratio": self.fy / self.fx,
                "focal_length": self.fx,
                "principal_pt_x": self.cx,
                "principal_pt_y": self.cy,
                "radial_distortion_1": self.k[0],
                "radial_distortion_2": self.k[1],
                "radial_distortion_3": self.k[2],
                "radial_distortion_4": self.k[3],
                "skew": 0.0,
            },
            "nr_calib_images": int(n_views),
            "stabelized": False,
        }

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def D(self) -> np.ndarray:
        return np.array(self.k, dtype=np.float64).reshape(4, 1)

    def resized(self, width: int, height: int) -> "Fisheye":
        """The same lens on a frame scaled to *width* x *height*, no crop.

        Per-axis, because the ISP scales 2028x1520 to 1440x1080, which is not
        exactly one factor. Pixel centers stay pixel centers.
        """
        sx = width / self.width
        sy = height / self.height
        return Fisheye(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=(self.cx + 0.5) * sx - 0.5,
            cy=(self.cy + 0.5) * sy - 0.5,
            k=self.k,
            width=int(width),
            height=int(height),
        )

    def distort_theta(self, theta: np.ndarray) -> np.ndarray:
        t2 = theta * theta
        k1, k2, k3, k4 = self.k
        return theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))

    def theta_limit(self) -> float:
        """Largest off-axis angle the model maps one-to-one into this frame.

        The polynomial is fitted inside the image. Past the farthest corner,
        or where it stops increasing, it can fold distant rays back in.
        """
        corners = np.array(
            [[0, 0], [self.width - 1, 0], [0, self.height - 1], [self.width - 1, self.height - 1]],
            dtype=np.float64,
        )
        mx = (corners[:, 0] - self.cx) / self.fx
        my = (corners[:, 1] - self.cy) / self.fy
        r_corner = float(np.max(np.hypot(mx, my)))
        thetas = np.linspace(0.0, np.pi, 4001)
        rd = self.distort_theta(thetas)
        rising = np.diff(rd) > 0
        stop = int(np.argmin(rising)) if not rising.all() else len(rising)
        thetas = thetas[: stop + 1]
        rd = rd[: stop + 1]
        if rd[-1] <= r_corner:
            return float(thetas[-1])
        return float(np.interp(r_corner, rd, thetas))

    def unproject(self, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Unit rays (N, 3) for pixels (N, 2), and which ones the model covers."""
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        mx = (pixels[:, 0] - self.cx) / self.fx
        my = (pixels[:, 1] - self.cy) / self.fy
        rd = np.hypot(mx, my)
        limit = self.theta_limit()
        table_t = np.linspace(0.0, limit, 2048)
        table_rd = self.distort_theta(table_t)
        valid = rd <= table_rd[-1]
        theta = np.interp(rd, table_rd, table_t)
        # Two Newton steps take the table lookup to float precision.
        k1, k2, k3, k4 = self.k
        for _ in range(2):
            t2 = theta * theta
            f = self.distort_theta(theta) - rd
            df = 1.0 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
            theta = theta - f / np.where(np.abs(df) < 1e-9, 1e-9, df)
        scale = np.where(rd > 1e-12, np.sin(theta) / np.maximum(rd, 1e-12), 1.0)
        rays = np.stack([mx * scale, my * scale, np.cos(theta)], axis=1)
        return rays, valid

    def project(self, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixels (N, 2) for rays (N, 3), and which ones land in the modelled frame."""
        rays = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
        r = np.hypot(rays[:, 0], rays[:, 1])
        theta = np.arctan2(r, rays[:, 2])
        rd = self.distort_theta(theta)
        scale = np.where(r > 1e-12, rd / np.maximum(r, 1e-12), 0.0)
        u = self.fx * rays[:, 0] * scale + self.cx
        v = self.fy * rays[:, 1] * scale + self.cy
        valid = (
            (theta <= self.theta_limit())
            & (u >= 0)
            & (v >= 0)
            & (u <= self.width - 1)
            & (v <= self.height - 1)
        )
        return np.stack([u, v], axis=1), valid


@dataclass(frozen=True)
class Remap:
    """cv2.remap tables that render *dst* frames from *src* frames."""

    map_x: np.ndarray
    map_y: np.ndarray
    valid: np.ndarray

    @property
    def coverage(self) -> float:
        return float(self.valid.mean())

    def apply(self, frame: np.ndarray) -> np.ndarray:
        out = cv2.remap(
            frame,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        out[~self.valid] = 0
        return out


def build_remap(
    src: Fisheye,
    dst: Fisheye,
    rotation_src_from_dst: np.ndarray | None = None,
    src_lit: np.ndarray | None = None,
) -> Remap:
    """Each *dst* pixel samples the *src* pixel that sees the same ray.

    *src_lit* is the part of the source frame the lens actually lights; the
    corners of a fisheye frame outside the image circle are black.
    """
    xs, ys = np.meshgrid(np.arange(dst.width), np.arange(dst.height))
    pixels = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    rays, valid = dst.unproject(pixels)
    if rotation_src_from_dst is not None:
        rays = rays @ np.asarray(rotation_src_from_dst, dtype=np.float64).T
    uv, in_src = src.project(rays)
    valid &= in_src
    if src_lit is not None:
        ui = np.clip(np.round(uv[:, 0]).astype(np.int64), 0, src.width - 1)
        vi = np.clip(np.round(uv[:, 1]).astype(np.int64), 0, src.height - 1)
        valid &= src_lit[vi, ui]
    shape = (dst.height, dst.width)
    return Remap(
        map_x=uv[:, 0].reshape(shape).astype(np.float32),
        map_y=uv[:, 1].reshape(shape).astype(np.float32),
        valid=valid.reshape(shape),
    )


def lit_mask(frames: list[np.ndarray], threshold: int = 12) -> np.ndarray:
    """Pixels some frame shows brighter than black: the lens image circle."""
    peak = None
    for frame in frames:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        peak = gray if peak is None else np.maximum(peak, gray)
    if peak is None:
        raise ValueError("no frames for the lit mask")
    mask = (peak > threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        raise ValueError("frames are black; no lit image circle")
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    # The image circle is convex; the hull fills scene regions that stayed dark.
    points = cv2.findNonZero((labels == biggest).astype(np.uint8))
    hull = cv2.convexHull(points)
    filled = np.zeros_like(mask)
    cv2.fillConvexPoly(filled, hull, 1)
    return filled.astype(bool)
