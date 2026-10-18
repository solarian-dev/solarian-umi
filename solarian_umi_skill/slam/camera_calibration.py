"""Per camera-and-lens calibration: Lumi intrinsics from solarian's 9x6 checkerboard on a screen.

One JSON per physical camera and lens, keyed by the serial the collector
stamps into the MP4. It carries the native intrinsics (UMI schema), which
the remap to GoPro geometry needs, and later the camera-to-IMU rotation
estimated from the demos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from solarian.core.fisheye import find_chessboard

from solarian_umi_skill.slam.fisheye_model import Fisheye, lit_mask

CALIBRATION_FILENAME = "camera_calibration.json"
# Inner corners of solarian's shipped checkerboard_9x6.png.
BOARD = (9, 6)
# Physical size cancels out of intrinsics.
SQUARE_M = 0.025
MAX_RMS_PX = 0.5
MIN_VIEWS = 15
# Outer cells of a 4x4 grid over the frame; the fisheye edge is where
# distortion terms are decided, so every one must see corners.
COVERAGE_GRID = 4
LUMI_SIZE = (1440, 1080)
LUMI_FPS = 30.0


@dataclass(frozen=True)
class BoardView:
    image_points: np.ndarray  # (N, 2) pixels
    object_points: np.ndarray  # (N, 3) board meters, z = 0


def board_object_points() -> np.ndarray:
    """Board corners in findChessboardCorners order: row by row, x fastest."""
    cols, rows = BOARD
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float64) * SQUARE_M
    return np.hstack([grid, np.zeros((len(grid), 1))])


def detect_board(gray: np.ndarray) -> BoardView | None:
    corners = find_chessboard(gray, BOARD)
    if corners is None:
        return None
    return BoardView(image_points=corners.reshape(-1, 2).astype(np.float64), object_points=board_object_points())


def is_near_duplicate(view: BoardView, views: list[BoardView], width: int) -> bool:
    """Same place and size as an earlier view: weight without information."""
    center = view.image_points.mean(axis=0)
    span = np.ptp(view.image_points, axis=0).max()
    for other in views:
        other_span = np.ptp(other.image_points, axis=0).max()
        if (
            np.linalg.norm(center - other.image_points.mean(axis=0)) < 0.02 * width
            and abs(span - other_span) < 0.1 * other_span
        ):
            return True
    return False


def _gray_frames(source: Path, every: int):
    if source.is_dir():
        for path in sorted(source.glob("*.png")):
            yield cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return
    import av

    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index % every == 0:
                yield frame.to_ndarray(format="gray")


class StaticMask:
    """What stays the same while the camera moves: the gripper in view.

    Compares local contrast, not brightness, because auto exposure changes the
    gripper's brightness from frame to frame. Only regions joined to the rim of
    the image circle count; the gripper enters from there, and specks of
    chance agreement in the scene do not. Until the camera has moved enough,
    nothing counts, so holding still does not excuse every cell.
    """

    WIDTH = 192
    MIN_SAMPLES = 12
    MIN_CONSISTENCY = 0.6

    def __init__(self) -> None:
        self._sum: np.ndarray | None = None
        self._abs: np.ndarray | None = None
        self._n = 0

    def add(self, gray: np.ndarray) -> None:
        small = cv2.resize(
            gray, (self.WIDTH, round(gray.shape[0] * self.WIDTH / gray.shape[1])), interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        mean = cv2.GaussianBlur(small, (0, 0), 4)
        spread = np.sqrt(cv2.GaussianBlur((small - mean) ** 2, (0, 0), 4)) + 4
        contrast = (small - mean) / spread
        if self._sum is None:
            self._sum, self._abs = np.zeros_like(contrast), np.zeros_like(contrast)
        self._sum += contrast
        self._abs += np.abs(contrast)
        self._n += 1

    def mask(self, lit: np.ndarray) -> np.ndarray:
        """Static pixels inside *lit*, at its size."""
        none = np.zeros_like(lit, dtype=bool)
        if self._n < self.MIN_SAMPLES:
            return none
        h, w = self._sum.shape
        consistent = (np.abs(self._sum) / (self._abs + 1e-3) > self.MIN_CONSISTENCY).astype(np.uint8)
        consistent = cv2.morphologyEx(consistent, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        consistent = cv2.morphologyEx(consistent, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        small_lit = cv2.resize(lit.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        rim = cv2.dilate(1 - small_lit, np.ones((5, 5), np.uint8)) & small_lit
        static = consistent & small_lit
        n, labels = cv2.connectedComponents(static)
        touching = np.unique(labels[(rim > 0) & (static > 0)])
        static = np.isin(labels, touching[touching > 0]).astype(np.uint8)
        # The lens vignettes a thin static ring at the rim; the gripper is bulky.
        blob = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        static = cv2.morphologyEx(static, cv2.MORPH_OPEN, blob)
        static = cv2.morphologyEx(static, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        static = cv2.resize(static, (lit.shape[1], lit.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool) & lit
        if static.sum() > 0.5 * lit.sum():
            return none
        return static


def detect_views(
    source: Path, *, every: int = 5, max_views: int = 120
) -> tuple[list[BoardView], np.ndarray, np.ndarray]:
    """Board views from a clip (every *every*-th frame) or a folder of PNGs, the lit mask and the static mask."""
    views: list[BoardView] = []
    sampled: list[np.ndarray] = []
    static = StaticMask()
    # Saved frames are each a deliberate view; clip frames are sampled densely.
    sample_every = 1 if source.is_dir() else 6
    for index, gray in enumerate(_gray_frames(source, every)):
        if index % sample_every == 0:
            sampled.append(gray)
            static.add(gray)
        view = detect_board(gray)
        if view is not None and not is_near_duplicate(view, views, gray.shape[1]):
            views.append(view)
    if not sampled:
        raise ValueError(f"{source} has no frames")
    if len(views) > max_views:
        keep = np.linspace(0, len(views) - 1, max_views).round().astype(int)
        views = [views[i] for i in keep]
    lit = lit_mask(sampled)
    return views, lit, static.mask(lit)


def cell_lit_fraction(lit: np.ndarray) -> np.ndarray:
    height, width = lit.shape
    n = COVERAGE_GRID
    return lit[: height // n * n, : width // n * n].reshape(n, height // n, n, width // n).mean(axis=(1, 3))


def cells_hit(views: list[BoardView], size: tuple[int, int]) -> np.ndarray:
    width, height = size
    n = COVERAGE_GRID
    hit = np.zeros((n, n), dtype=bool)
    for view in views:
        col = np.clip((view.image_points[:, 0] / width * n).astype(int), 0, n - 1)
        row = np.clip((view.image_points[:, 1] / height * n).astype(int), 0, n - 1)
        hit[row, col] = True
    return hit


def required_cells(lit: np.ndarray, static: np.ndarray) -> np.ndarray:
    """Outer grid cells the board can reach.

    Most of the cell is inside the image circle (the black frame corners of a
    circular fisheye never show the board), and the gripper covers less than
    half of that part.
    """
    n = COVERAGE_GRID
    cell_lit = cell_lit_fraction(lit)
    cell_static = cell_lit_fraction(static)
    outer = np.zeros((n, n), dtype=bool)
    outer[[0, -1], :] = True
    outer[:, [0, -1]] = True
    return outer & (cell_lit > 0.5) & (cell_static < 0.5 * cell_lit)


def uncovered_cells(views: list[BoardView], lit: np.ndarray, static: np.ndarray) -> list[tuple[int, int]]:
    """Required cells (col, row) no corner landed in."""
    missing = required_cells(lit, static) & ~cells_hit(views, (lit.shape[1], lit.shape[0]))
    return [(int(c), int(r)) for r, c in zip(*np.nonzero(missing))]


def _kb_project(intrinsics: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, points: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    fx, fy, cx, cy, k1, k2, k3, k4 = intrinsics
    cam = Rotation.from_rotvec(rvec).apply(points) + tvec
    r = np.hypot(cam[:, 0], cam[:, 1])
    theta = np.arctan2(r, cam[:, 2])
    t2 = theta * theta
    d = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    scale = d / np.maximum(r, 1e-12)
    return np.column_stack([fx * cam[:, 0] * scale + cx, fy * cam[:, 1] * scale + cy])


def _initial_pose(view: BoardView, model: Fisheye) -> np.ndarray | None:
    """Board pose from a pinhole PnP in a virtual camera turned to face the board.

    A pinhole only holds well inside 90 degrees, and this lens sees past it;
    facing the board puts every corner near the virtual axis.
    """
    from scipy.spatial.transform import Rotation

    rays, valid = model.unproject(view.image_points)
    if not valid.all():
        return None
    mean = rays.mean(axis=0)
    face, _ = Rotation.align_vectors([[0.0, 0.0, 1.0]], [mean / np.linalg.norm(mean)])
    turned = face.apply(rays)
    ok, rvec, tvec = cv2.solvePnP(
        view.object_points, turned[:, :2] / turned[:, 2:], np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return None
    back = face.inv()
    rotation = back * Rotation.from_rotvec(rvec.ravel())
    return np.concatenate([rotation.as_rotvec(), back.apply(tvec.ravel())])


def _fit(views: list[BoardView], size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Joint least squares of Kannala-Brandt intrinsics and every view's pose.

    Returns the 8 intrinsics and per-view RMS reprojection error in pixels.
    OpenCV's fisheye.calibrate initialises poses from all corners through a
    pinhole and fails on boards near this lens's 90 degree edge.
    """
    from scipy.optimize import least_squares

    width, height = size
    # Wide fisheye: focal length about a third of the frame height; equidistant start.
    f0 = 0.33 * height
    start = Fisheye(fx=f0, fy=f0, cx=width / 2, cy=height / 2, k=(0.0, 0.0, 0.0, 0.0), width=width, height=height)
    poses = [_initial_pose(v, start) for v in views]
    if any(p is None for p in poses):
        raise ValueError("a board view could not be placed; undo views at the very edge of the lens")
    x0 = np.concatenate([[f0, f0, width / 2, height / 2, 0, 0, 0, 0], *poses])

    def residuals(x: np.ndarray) -> np.ndarray:
        intrinsics = x[:8]
        out = []
        for i, view in enumerate(views):
            pose = x[8 + 6 * i : 14 + 6 * i]
            out.append((_kb_project(intrinsics, pose[:3], pose[3:], view.object_points) - view.image_points).ravel())
        return np.concatenate(out)

    n = len(views)
    sparsity = np.zeros((sum(2 * len(v.image_points) for v in views), 8 + 6 * n), dtype=bool)
    row = 0
    for i, view in enumerate(views):
        rows = 2 * len(view.image_points)
        sparsity[row : row + rows, :8] = True
        sparsity[row : row + rows, 8 + 6 * i : 14 + 6 * i] = True
        row += rows
    result = least_squares(residuals, x0, jac_sparsity=sparsity, x_scale="jac", method="trf", max_nfev=200)
    per_point = np.linalg.norm(result.fun.reshape(-1, 2), axis=1)
    bounds = np.cumsum([0] + [len(v.image_points) for v in views])
    per_view = np.array([np.sqrt(np.mean(per_point[a:b] ** 2)) for a, b in zip(bounds[:-1], bounds[1:])])
    return result.x[:8], per_view


def calibrate_views(
    views: list[BoardView], size: tuple[int, int], *, max_drop_fraction: float = 0.3
) -> tuple[Fisheye, float, int]:
    """Fisheye intrinsics with every term free; drops the worst views first.

    Returns the model, its RMS reprojection error in pixels and views used.
    """
    width, height = size
    current = list(views)
    budget = int(len(views) * max_drop_fraction)
    while True:
        if len(current) < MIN_VIEWS:
            raise ValueError(f"only {len(current)} usable board views; need {MIN_VIEWS}")
        intrinsics, per_view = _fit(current, size)
        rms = float(np.sqrt(np.mean(per_view**2)))
        if rms <= MAX_RMS_PX or budget == 0:
            fx, fy, cx, cy, *k = (float(v) for v in intrinsics)
            return Fisheye(fx=fx, fy=fy, cx=cx, cy=cy, k=tuple(k), width=width, height=height), rms, len(current)
        current.pop(int(np.argmax(per_view)))
        budget -= 1


def calibration_document(
    model: Fisheye,
    *,
    serial: str,
    source: str,
    rms: float,
    n_views: int,
    fps: float,
) -> dict:
    return {
        "camera_serial": serial,
        "source": source,
        "intrinsics": model.to_umi_json(rms=rms, n_views=n_views, fps=fps),
    }


def stand_in_from_remap(remap_json: Path, *, serial: str, size: tuple[int, int] = LUMI_SIZE) -> dict:
    """A calibration document from a solarian GoPro-remap JSON's camera intrinsics.

    Those come from a checkerboard on the raw 2028x1520 sensor mode; the
    collector records that mode scaled to *size*.
    """
    data = json.loads(remap_json.read_text(encoding="utf-8"))
    native = data["camera_intrinsics"]
    model = Fisheye.from_umi_json(native).resized(*size)
    return calibration_document(
        model,
        serial=serial,
        source=f"stand-in: {remap_json.name} scaled from {native['image_width']}x{native['image_height']}",
        rms=float(native["final_reproj_error"]),
        n_views=int(native["nr_calib_images"]),
        fps=LUMI_FPS,
    )


def find_calibration(session: Path, serial: str, conf_dirs: list[Path]) -> Path:
    """The session's own calibration, else the camera's file in a conf dir."""
    candidates = [session / CALIBRATION_FILENAME]
    candidates += [conf / "calibrations" / "cameras" / f"{serial}.json" for conf in conf_dirs]
    for path in candidates:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["camera_serial"] != serial:
                raise SystemExit(f"{path} is for camera {data['camera_serial']}, videos are from {serial}")
            return path
    tried = "\n  ".join(str(p) for p in candidates)
    raise SystemExit(
        f"No calibration for camera {serial}. Tried:\n  {tried}\n"
        f"Put the collector on Stream video and run solarian umi-calibrate-lumi --serial {serial}."
    )


def load_native_model(calibration: Path) -> Fisheye:
    return Fisheye.from_umi_json(json.loads(calibration.read_text(encoding="utf-8"))["intrinsics"])
