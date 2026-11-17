"""Live checkerboard capture from the Lumi collector's stream.

One fullscreen window: solarian's 9x6 checkerboard on the left for the Lumi
to look at, and on the right what the Lumi sees with the detected corners and
the edge cells still missing.

The lens sees the whole screen, so a picture of the board in the preview could
be detected as a second, curved board; blurring or edge-filtering it does not
stop the detector. Once the board is found it is painted over in the preview.
Before that the preview only holds a smaller copy of a board the detector
just failed to find, which it then fails to find too.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np
from solarian.core.optical_remap import shipped_checkerboard_path

from solarian_umi.slam.camera_calibration import (
    BOARD,
    COVERAGE_GRID,
    LUMI_FPS,
    MIN_VIEWS,
    BoardView,
    StaticMask,
    cells_hit,
    detect_board,
    is_near_duplicate,
    required_cells,
    uncovered_cells,
)
from solarian_umi.slam.fisheye_model import lit_mask

WINDOW = "umi-calibrate-lumi  [SPACE save  u undo  c calibrate  q quit]"
BOARD_SHARE = 0.62
DETECT_WIDTH = 720
GREEN = (80, 220, 80)
RED = (60, 60, 230)
YELLOW = (0, 210, 255)
WHITE = (255, 255, 255)


class LumiStream:
    """Newest decoded frame of the collector's H.264 stream (BGR, native size)."""

    def __init__(self, host: str, port: int):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise SystemExit("ffmpeg not found; install ffmpeg")
        self._proc = subprocess.Popen(
            [
                ffmpeg, "-hide_banner", "-loglevel", "info",
                # -fflags nobuffer decodes nothing from a raw H.264 socket.
                "-flags", "low_delay",
                # Raw H.264 has no timestamps; the demuxer would assume 25 fps.
                "-f", "h264", "-framerate", str(int(LUMI_FPS)), "-i", f"tcp://{host}:{port}",
                "-vf", "scale=in_range=tv:out_range=pc",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.size = self._read_size(host, port)
        self._frame: np.ndarray | None = None
        self._count = 0
        self._lock = threading.Lock()
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        threading.Thread(target=self._read_frames, daemon=True).start()

    def _read_size(self, host: str, port: int) -> tuple[int, int]:
        log = []
        for raw in self._proc.stderr:
            line = raw.decode(errors="replace")
            log.append(line)
            match = re.search(r"Video: .*?(\d{3,5})x(\d{3,5})", line)
            if match:
                return int(match.group(1)), int(match.group(2))
        self.close()
        raise SystemExit(
            f"No video from tcp://{host}:{port}. Put the collector on its Stream video screen.\n"
            + "".join(log[-5:])
        )

    def _drain_stderr(self) -> None:
        for _ in self._proc.stderr:
            pass

    def _read_frames(self) -> None:
        width, height = self.size
        n = width * height * 3
        while True:
            data = self._proc.stdout.read(n)
            if len(data) < n:
                return
            frame = np.frombuffer(data, np.uint8).reshape(height, width, 3)
            with self._lock:
                self._frame = frame
                self._count += 1

    def latest(self) -> tuple[int, np.ndarray | None]:
        with self._lock:
            return self._count, self._frame

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()


def _fit(image: np.ndarray, box_w: int, box_h: int) -> np.ndarray:
    scale = min(box_w / image.shape[1], box_h / image.shape[0])
    size = (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)


def _paste(canvas: np.ndarray, image: np.ndarray, x: int, y: int) -> None:
    canvas[y : y + image.shape[0], x : x + image.shape[1]] = image


def camera_preview(frame: np.ndarray, width: int, live: np.ndarray | None) -> tuple[np.ndarray, float]:
    """The frame at *width*, with the board at *live* (frame pixels) painted over."""
    scale = width / frame.shape[1]
    preview = cv2.resize(frame, (width, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    if live is None:
        return preview, scale
    points = live * scale
    grid = points.reshape(BOARD[1], BOARD[0], 2)
    # Corners are one square in from the board's edge; cover that border too.
    square = float(np.median(np.linalg.norm(np.diff(grid, axis=1), axis=2)))
    hull = cv2.convexHull(points.astype(np.int32))
    cv2.fillConvexPoly(preview, hull, (40, 90, 40))
    cv2.polylines(preview, [hull], True, (40, 90, 40), max(2, int(2.6 * square)))
    return preview, scale


def _draw_guidance(
    preview: np.ndarray,
    scale: float,
    size: tuple[int, int],
    views: list[BoardView],
    lit: np.ndarray | None,
    static: np.ndarray | None,
    live: np.ndarray | None,
) -> None:
    n = COVERAGE_GRID
    h, w = preview.shape[:2]
    if lit is not None:
        small_static = cv2.resize(static.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        preview[small_static] = (preview[small_static] * 0.4 + np.array([90, 40, 0])).astype(np.uint8)
        small = cv2.resize(lit.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, (120, 120, 120), 1)
        required = required_cells(lit, static)
        hit = cells_hit(views, size)
        for r, c in zip(*np.nonzero(required)):
            p0 = (int(c * w / n) + 2, int(r * h / n) + 2)
            p1 = (int((c + 1) * w / n) - 2, int((r + 1) * h / n) - 2)
            cv2.rectangle(preview, p0, p1, GREEN if hit[r, c] else RED, 2)
    for view in views:
        for x, y in view.image_points * scale:
            cv2.circle(preview, (int(x), int(y)), 1, YELLOW, -1)
    if live is not None:
        cv2.drawChessboardCorners(preview, BOARD, (live * scale).astype(np.float32).reshape(-1, 1, 2), True)


def _text(canvas: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]], x: int, y: int, size: float) -> None:
    step = int(34 * size)
    for i, (line, color) in enumerate(lines):
        cv2.putText(canvas, line, (x, y + i * step), cv2.FONT_HERSHEY_SIMPLEX, size, color, max(1, int(2 * size)), cv2.LINE_AA)


def render(
    window: tuple[int, int],
    board: np.ndarray,
    frame: np.ndarray,
    live: np.ndarray | None,
    views: list[BoardView],
    lit: np.ndarray | None,
    static: np.ndarray | None,
    status: tuple[str, tuple[int, int, int]],
) -> np.ndarray:
    win_w, win_h = window
    canvas = np.full((win_h, win_w, 3), 255, np.uint8)
    board_w = int(win_w * BOARD_SHARE)
    board_img = _fit(board, board_w - 20, win_h - 20)
    _paste(canvas, board_img, (board_w - board_img.shape[1]) // 2, (win_h - board_img.shape[0]) // 2)
    panel_x, panel_w = board_w, win_w - board_w
    canvas[:, panel_x:] = 0
    preview, scale = camera_preview(frame, panel_w - 20, live)
    _draw_guidance(preview, scale, (frame.shape[1], frame.shape[0]), views, lit, static, live)
    _paste(canvas, preview, panel_x + 10, 10)

    missing = None if lit is None else len(uncovered_cells(views, lit, static))
    _text(
        canvas,
        [
            ("BOARD - SPACE to save" if live is not None else "no board detected", GREEN if live is not None else YELLOW),
            (f"views {len(views)}/{MIN_VIEWS}", GREEN if len(views) >= MIN_VIEWS else WHITE),
            (f"red edge cells left: {'?' if missing is None else missing}", GREEN if missing == 0 else RED),
            ("Put corners in every red cell, tilted", WHITE),
            ("30-45 deg and straight, near and far.", WHITE),
            ("Blue: gripper, not required.", WHITE),
            ("c calibrates  u undo  q quit", WHITE),
            status,
        ],
        panel_x + 16,
        preview.shape[0] + 50,
        max(0.45, panel_w / 900),
    )
    return canvas


def capture(stream: LumiStream, frames_dir: Path, calibrate) -> object | None:
    """Run the window until *calibrate(views, lit, static)* returns a result, or the user quits.

    *calibrate* raises ValueError with a message for the status line when the
    views are not good enough yet.
    """
    board = cv2.imread(str(shipped_checkerboard_path()))
    frames_dir.mkdir(parents=True, exist_ok=True)
    width, height = stream.size
    views: list[BoardView] = []
    peak = np.zeros((height, width), np.uint8)
    motion = StaticMask()
    lit: np.ndarray | None = None
    static: np.ndarray | None = None
    status: tuple[str, tuple[int, int, int]] = ("Point the Lumi at the board.", WHITE)
    last_count = -1
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        while True:
            if not stream.alive:
                raise SystemExit("The Lumi stream ended. Is the collector still on Stream video?")
            _, _, win_w, win_h = cv2.getWindowImageRect(WINDOW)
            if win_w < 200 or win_h < 200:
                win_w, win_h = 1920, 1080
            count, frame = stream.latest()
            if frame is None:
                if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
                    return None
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if count != last_count and count % 15 == 0:
                np.maximum(peak, gray, out=peak)
                motion.add(gray)
                try:
                    lit = lit_mask([peak])
                    static = motion.mask(lit)
                except ValueError:
                    lit = static = None
            last_count = count
            detect_scale = DETECT_WIDTH / width
            small = cv2.resize(gray, (DETECT_WIDTH, int(round(height * detect_scale))), interpolation=cv2.INTER_AREA)
            found = detect_board(small)
            live = None if found is None else found.image_points / detect_scale

            cv2.imshow(WINDOW, render((win_w, win_h), board, frame, live, views, lit, static, status))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return None
            if key == ord(" "):
                view = detect_board(gray)
                if view is None:
                    status = ("Board not found at full resolution; hold still.", YELLOW)
                elif is_near_duplicate(view, views, width):
                    status = ("Same as an earlier view; move or tilt.", YELLOW)
                else:
                    cv2.imwrite(str(frames_dir / f"{len(views):04d}.png"), frame)
                    views.append(view)
                    status = (f"Saved view {len(views)}.", GREEN)
            elif key == ord("u") and views:
                views.pop()
                (frames_dir / f"{len(views):04d}.png").unlink(missing_ok=True)
                status = (f"Removed view {len(views) + 1}.", WHITE)
            elif key == ord("c"):
                if lit is None:
                    status = ("No lens image yet.", YELLOW)
                    continue
                busy = ("Calibrating, about 10 s...", WHITE)
                cv2.imshow(WINDOW, render((win_w, win_h), board, frame, live, views, lit, static, busy))
                cv2.waitKey(1)
                try:
                    return calibrate(views, lit, static)
                except ValueError as exc:
                    status = (str(exc), RED)
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
