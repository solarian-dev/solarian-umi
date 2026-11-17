"""Calibrate a wrist camera onto GoPro Hero 10 + Max Lens Mod optics.

Interactive flow:
  1. Pick a connected camera (same enumeration as ``list-cameras``)
  2. Optionally pick a GoPro for paired checker capture / homography
  3. Capture a 9x6 checkerboard (or reuse ``images=`` / ``gopro_images=``)
  4. Fisheye-calibrate the user camera, remap onto shipped GoPro HDMI 1080p
  5. Save remap JSON and point the camera YAML at it
  6. Live gold overlay (fingers / holders / screws) after the camera is mounted

Does not need ``run-server`` / ``run-robot`` (those lock the CSI camera).
"""

from __future__ import annotations

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.core.conf_paths import (
    calibrations_dir,
    conf_dir,
    module_config_path,
    skill_config_path,
)
from solarian.utils.configs import load_yaml

_WINDOW = "calibrate-umi-camera"
_DUMP_DEFAULT = ".cache/umi_image_compare"
_CAPTURE_ROOT = ".cache/umi_camera_calib"
_DEFAULT_CAMERA_YAML = "right_wrist_camera.yaml"
_REMAP_RESIZE = (640, 360)
_BOARD = (9, 6)
_MIN_SAMPLES = 8
_GOPRO_WH = (1920, 1080)
_OPENCV_PATTERN_URL = "https://github.com/opencv/opencv/blob/master/doc/pattern.png"


def _project_dir() -> Path:
    return Path(os.environ.get("SOLARIAN_PROJECT_PATH", os.getcwd()))


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    val = OmegaConf.select(cfg, key)
    return default if val is None else val


def _as_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _skill_config_path(cfg) -> Path:
    """``config=`` names the UMI yaml; otherwise the installed ``conf/skills/umi.yaml``."""
    raw = _cfg_get(cfg, "config")
    name = str(raw).strip() if raw else "umi.yaml"
    if name.lower() in {"", "null", "none"}:
        name = "umi.yaml"
    if not name.endswith((".yaml", ".yml")):
        name = f"{name}.yaml"
    return skill_config_path(name)


def _load_skill_cfg(cfg=None):
    path = _skill_config_path(cfg)
    if not path.is_file():
        return OmegaConf.create({})
    return load_yaml(path)


def _resolve_gold(cfg, skill_cfg) -> Path:
    from solarian_umi.umi.compare_policy_image import DEFAULT_GOLD

    raw = None
    if cfg is not None and OmegaConf.select(cfg, "gold") is not None:
        raw = str(cfg.gold).strip()
        if raw.lower() in {"", "null", "none"}:
            raw = None
    if raw:
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"Gold image not found: {path}")
        return path
    return DEFAULT_GOLD


def _camera_yaml_path(cfg, skill_cfg) -> Path:
    """``camera_yaml=`` (a module name or filename in conf/modules/, or a
    path), otherwise the skill's camera module."""
    raw = _cfg_get(cfg, "camera_yaml")
    if raw:
        p = Path(str(raw))
        if p.is_absolute() or len(p.parts) > 1:
            return p
        return module_config_path(p.stem)
    cam_module = skill_cfg.get("camera")
    if not cam_module:
        modules = skill_cfg.get("modules") or {}
        cam_module = modules.get("camera", "right_wrist_camera")
    return module_config_path(str(cam_module))


def _load_camera_capture_size(yaml_path: Path) -> tuple[int | None, int | None]:
    if not yaml_path.is_file():
        return None, None
    ycfg = load_yaml(yaml_path)
    cam = (ycfg.get("sensors") or {}).get("camera") or {}
    w = cam.get("width")
    h = cam.get("height")
    if w is None or h is None:
        return None, None
    return int(w), int(h)


def _prompt(msg: str) -> str:
    try:
        return input(msg).strip()
    except EOFError:
        print()
        return ""


def _wait_enter(msg: str) -> None:
    _prompt(msg)


def _upsert_yaml_key(text: str, key: str, value: str, indent: str = "    ") -> str:
    pattern = rf"(?m)^({re.escape(indent)}{re.escape(key)}:\s*).*$"
    if re.search(pattern, text):
        return re.sub(pattern, rf"\g<1>{value}", text, count=1)
    rh = rf"(?m)^({re.escape(indent)}resize_height:\s*.*)$"
    if re.search(rh, text):
        return re.sub(rh, rf"\1\n{indent}{key}: {value}", text, count=1)
    return text.rstrip() + f"\n{indent}{key}: {value}\n"


def _patch_camera_yaml(yaml_path: Path, remap_rel: str, resize_wh: tuple[int, int]) -> None:
    text = yaml_path.read_text(encoding="utf-8")
    text = _upsert_yaml_key(text, "optical_remap", remap_rel)
    text = _upsert_yaml_key(text, "resize_width", str(resize_wh[0]))
    text = _upsert_yaml_key(text, "resize_height", str(resize_wh[1]))
    yaml_path.write_text(text, encoding="utf-8")


def _print_checkerboard_links(session_dir: Path) -> None:
    from solarian.core.optical_remap import (
        CHECKERBOARD_URL,
        shipped_checkerboard_path,
    )

    src = shipped_checkerboard_path()
    dest = session_dir / src.name
    session_dir.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dest)
        file_url = dest.resolve().as_uri()
        pkg_url = src.resolve().as_uri()
    else:
        file_url = pkg_url = "(checkerboard PNG missing from package)"
    print(
        "Print this 9x6 (inner corners) checkerboard and glue it to a flat rigid backing:\n"
        f"  {file_url}\n"
        f"  {pkg_url}\n"
        f"  {CHECKERBOARD_URL}\n"
        "Do not scale non-uniformly when printing. Same 9x6 geometry the detector expects."
    )
    _wait_enter("Press Enter when the printed board is ready. ")


def _draw_label(img, text, color=(0, 255, 0)):
    import cv2

    out = img.copy()
    h, w = out.shape[:2]
    bar = max(22, h // 22)
    cv2.rectangle(out, (0, 0), (w, bar), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (8, bar - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.5, bar / 38.0),
        color,
        max(1, bar // 18),
        cv2.LINE_AA,
    )
    return out


def _board_overlay(preview_bgr, board):
    import cv2
    from solarian.core.fisheye import find_chessboard

    gray = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2GRAY)
    corners = find_chessboard(gray, board)
    detected = corners is not None
    if detected:
        cv2.drawChessboardCorners(preview_bgr, board, corners, True)
    return preview_bgr, detected


def _next_index(folder: Path) -> int:
    mx = -1
    if folder.is_dir():
        for n in os.listdir(folder):
            stem = os.path.splitext(n)[0]
            if stem.isdigit():
                mx = max(mx, int(stem))
    return mx + 1


def _to_preview(img, pw: int = 640):
    import cv2

    h, w = img.shape[:2]
    return cv2.resize(img, (pw, int(round(h * pw / w))))


def _iter_sized(cam: dict, width: int | None, height: int | None):
    from solarian.cli.commands.list_cameras import iter_camera_frames

    return iter_camera_frames(cam, width=width, height=height)


def _capture_solo(cam: dict, out_dir: Path, width: int | None, height: int | None) -> Path:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    window = "calibrate-umi-camera capture  [SPACE=save  u=undo  q/ESC=done]"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    frames = _iter_sized(cam, width, height)
    idx = _next_index(out_dir)
    saved: list[str] = []
    print("Capture. SPACE=save, u=undo, q/ESC=done. Need >= 8 varied views.")
    try:
        for frame in frames:
            if frame is None:
                continue
            prev, det = _board_overlay(_to_preview(frame.copy()), _BOARD)
            prev = _draw_label(
                prev,
                ("BOARD" if det else "move board into view") + f"  saved={len(saved)}",
                (0, 255, 0) if det else (0, 255, 255),
            )
            cv2.imshow(window, prev)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            if k == ord(" "):
                name = f"{idx:04d}.png"
                cv2.imwrite(str(out_dir / name), frame)
                saved.append(name)
                idx += 1
                print(f"  saved {name}  (total {len(saved)})")
            elif k == ord("u") and saved:
                name = saved.pop()
                p = out_dir / name
                if p.exists():
                    p.unlink()
                idx -= 1
                print(f"  undid {name}  (total {len(saved)})")
    finally:
        frames.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    print(f"Saved {len(saved)} frames -> {out_dir}")
    if len(saved) < _MIN_SAMPLES:
        raise RuntimeError(
            f"Need at least {_MIN_SAMPLES} checkerboard shots, got {len(saved)}."
        )
    return out_dir


def _capture_paired(
    user_cam: dict,
    gopro_cam: dict,
    user_dir: Path,
    gopro_dir: Path,
    user_wh: tuple[int | None, int | None],
) -> tuple[Path, Path]:
    import cv2
    import numpy as np

    user_dir.mkdir(parents=True, exist_ok=True)
    gopro_dir.mkdir(parents=True, exist_ok=True)
    window = "calibrate-umi-camera capture  [SPACE=save pair  u=undo  q/ESC=done]"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    u_iter = _iter_sized(user_cam, user_wh[0], user_wh[1])
    g_iter = _iter_sized(gopro_cam, _GOPRO_WH[0], _GOPRO_WH[1])
    idx = max(_next_index(user_dir), _next_index(gopro_dir))
    saved: list[str] = []
    print(
        "Paired capture. SPACE=save when BOTH see the board, u=undo, q/ESC=done. "
        "Need >= 8 varied views."
    )
    try:
        while True:
            try:
                uframe = next(u_iter)
                gframe = next(g_iter)
            except StopIteration:
                break
            if uframe is None or gframe is None:
                continue
            uprev, udet = _board_overlay(_to_preview(uframe.copy()), _BOARD)
            gprev, gdet = _board_overlay(_to_preview(gframe.copy()), _BOARD)
            H = max(uprev.shape[0], gprev.shape[0])
            uprev = cv2.copyMakeBorder(uprev, 0, H - uprev.shape[0], 0, 0, cv2.BORDER_CONSTANT)
            gprev = cv2.copyMakeBorder(gprev, 0, H - gprev.shape[0], 0, 0, cv2.BORDER_CONSTANT)
            uprev = _draw_label(
                uprev, "user" + ("  BOARD" if udet else ""),
                (0, 255, 0) if udet else (0, 255, 255),
            )
            gprev = _draw_label(
                gprev, "GoPro" + ("  BOARD" if gdet else ""),
                (0, 255, 0) if gdet else (0, 255, 255),
            )
            show = np.concatenate([gprev, uprev], axis=1)
            status = f"saved={len(saved)}"
            status += (
                "  BOTH DETECTED - SPACE to save"
                if (udet and gdet)
                else "  (move board so BOTH see it)"
            )
            cv2.rectangle(show, (0, show.shape[0] - 30), (show.shape[1], show.shape[0]), (0, 0, 0), -1)
            cv2.putText(
                show, status, (8, show.shape[0] - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.imshow(window, show)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break
            if k == ord(" "):
                name = f"{idx:04d}.png"
                cv2.imwrite(str(user_dir / name), uframe)
                cv2.imwrite(str(gopro_dir / name), gframe)
                saved.append(name)
                idx += 1
                print(f"  saved pair {name}  (total {len(saved)})")
            elif k == ord("u") and saved:
                name = saved.pop()
                for d in (user_dir, gopro_dir):
                    p = d / name
                    if p.exists():
                        p.unlink()
                idx -= 1
                print(f"  undid {name}  (total {len(saved)})")
    finally:
        u_iter.close()
        g_iter.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    print(f"Saved {len(saved)} pairs -> {user_dir} and {gopro_dir}")
    if len(saved) < _MIN_SAMPLES:
        raise RuntimeError(
            f"Need at least {_MIN_SAMPLES} paired shots, got {len(saved)}."
        )
    return user_dir, gopro_dir


def _screen_geom() -> tuple[int, int, int, int]:
    """Largest monitor as (width, height, x, y). Full HD fallback if screeninfo fails."""
    try:
        from screeninfo import get_monitors

        monitors = get_monitors()
        if monitors:
            m = max(monitors, key=lambda mon: int(mon.width) * int(mon.height))
            return int(m.width), int(m.height), int(m.x), int(m.y)
    except Exception:
        pass
    return 1920, 1080, 0, 0


def _fit_image_to_screen(img, margin: float = 0.88, max_scale: float = 4.0):
    """Resize ``img`` so it fits on the display (leaves room for chrome / dock)."""
    import cv2

    sw, sh, _, _ = _screen_geom()
    max_w = max(1, int(sw * margin))
    max_h = max(1, int(sh * margin))
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, max_scale)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    if (nw, nh) == (w, h):
        return img, nw, nh
    interp = cv2.INTER_NEAREST if scale >= 1.0 else cv2.INTER_AREA
    return cv2.resize(img, (nw, nh), interpolation=interp), nw, nh


def _map_overlay_click(x: int, y: int, layout: dict) -> tuple[str, int, int] | None:
    """Map a window click onto the gold or live 224×224 panel."""
    disp_h, disp_w = layout.get("display_hw") or (0, 0)
    sheet_h, sheet_w = layout.get("sheet_hw") or (0, 0)
    panel_h, panel_w = layout.get("panel_hw") or (0, 0)
    n_rows = int(layout.get("n_rows") or 1)
    if min(disp_h, disp_w, sheet_h, sheet_w, panel_h, panel_w) <= 0:
        return None
    sx = int(x * sheet_w / disp_w)
    sy = int(y * sheet_h / disp_h)
    bar_h = sheet_h - panel_h * n_rows
    if sy < bar_h or sx < 0 or sx >= sheet_w:
        return None
    py = sy - bar_h
    col = sx // panel_w
    if col >= 2:
        return None
    local_x = sx % panel_w
    local_y = py % panel_h if n_rows > 1 else py
    if not (0 <= local_x < panel_w and 0 <= local_y < panel_h):
        return None
    which = "gold" if col == 0 else "live"
    return which, local_x, local_y


def _solve_view_rotation(gold_polys, live_polys, aligner, panel_wh):
    """``(R, rms_before_px, rms_after_px, move)`` or None without two outline pairs.

    R turns GoPro-plane rays so the live outlines land on gold's: a camera
    tilted on its mount puts every outline off by the same angle, which no
    intrinsics can absorb but the remap's rotation can. Each gold outline is
    paired with the nearest live one of its class, and scored by the distance
    from every vertex to the nearest vertex of its partner, so the two traces
    need not share a vertex count or order. *move* carries live points through
    R, for outlines traced before it was applied.
    """
    import cv2
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    width, height = aligner.out_size
    # UMI's crop: fit the height to the panel, keep the centred square.
    scale = height / panel_wh[1]
    x0 = (width - height) / 2.0
    K, D = aligner.Kg, aligner.Dg

    def rays(pts):
        px = np.array([[x0 + x * scale, y * scale] for x, y in pts], dtype=np.float64)
        norm = cv2.fisheye.undistortPoints(px.reshape(-1, 1, 2), K, D).reshape(-1, 2)
        v = np.c_[norm, np.ones(len(norm))]
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def pixels(v):
        norm = v[:, :2] / v[:, 2:3]
        px = cv2.fisheye.distortPoints(norm.reshape(-1, 1, 2), K, D).reshape(-1, 2)
        return np.c_[(px[:, 0] - x0) / scale, px[:, 1] / scale]

    pairs = []
    for kind in ("orange", "white"):
        live = getattr(live_polys, kind)
        for gold in getattr(gold_polys, kind):
            if live:
                centre = np.mean(gold, axis=0)
                partner = min(live, key=lambda p: np.linalg.norm(np.mean(p, axis=0) - centre))
                pairs.append((np.asarray(gold, dtype=np.float64), rays(partner)))
    if len(pairs) < 2:
        return None

    def residuals(rotvec):
        rot = Rotation.from_rotvec(rotvec).as_matrix()
        out = []
        for gold, live_rays in pairs:
            gap = np.linalg.norm(pixels(live_rays @ rot.T)[:, None] - gold[None], axis=2)
            out.extend(gap.min(axis=1))
            out.extend(gap.min(axis=0))
        return np.asarray(out)

    before = residuals(np.zeros(3))
    fit = least_squares(residuals, np.zeros(3))
    rot = Rotation.from_rotvec(fit.x).as_matrix()

    def move(pts):
        return [tuple(int(round(c)) for c in p) for p in pixels(rays(pts) @ rot.T)]

    return (
        rot,
        float(np.sqrt(np.mean(before**2))),
        float(np.sqrt(np.mean(fit.fun**2))),
        move,
    )


def _run_overlay(
    *,
    gold,
    gold_path: Path,
    dump_dir: Path,
    skill_cfg,
    cam: dict | None,
    remap_path: Path | None,
    capture_wh: tuple[int | None, int | None],
) -> None:
    import json

    import cv2
    import numpy as np
    from scipy.spatial.transform import Rotation

    from solarian_umi.umi.compare_policy_image import (
        CONTACT_SHEET_VIEWS,
        _imwrite_rgb,
        render_contact_sheet,
    )
    from solarian_umi.umi.gripper_align import (
        ALIGN_POLYGONS_NAME,
        compute_align_metrics,
        extract_align_masks,
        load_align_polygons,
        save_align_polygons,
    )
    from solarian_umi.umi.policy_image import PolicyImagePipeline
    from solarian.core.optical_remap import OpticalRemap

    pipeline = PolicyImagePipeline(
        output_res=(gold.shape[1], gold.shape[0]) if gold.shape[0] == gold.shape[1] else (224, 224),
        letterbox_to_square=bool(skill_cfg.get("letterbox_to_square", False)),
        no_mirror=bool(skill_cfg.get("no_mirror", True)),
        mirror_swap=bool(skill_cfg.get("mirror_swap", False)),
    )
    remap = None
    if remap_path is not None and remap_path.is_file():
        remap = OpticalRemap(str(remap_path), conf_dir())
        print(
            f"Live overlay with remap {remap_path} "
            f"(model={remap.aligner.model})"
        )
    else:
        print("Live overlay WITHOUT optical remap (raw camera -> policy crop/mask)")

    if cam is None:
        raise RuntimeError("No camera selected for overlay.")

    polys_path = dump_dir / ALIGN_POLYGONS_NAME
    gold_polys, live_polys = load_align_polygons(polys_path)
    view_i = 0
    draw_kind = "orange"
    ghost_gold = True
    freeze_live = False
    frozen_live = None
    draft: dict = {"panel": None, "pts": []}
    state: dict = {"layout": {}, "live": None}

    def _persist() -> None:
        save_align_polygons(polys_path, gold_polys, live_polys)

    def _shown_live():
        live = state["live"]
        drawing_live = bool(draft["pts"]) and draft["panel"] == "live"
        if (freeze_live or drawing_live) and frozen_live is not None:
            return frozen_live
        return live

    def _close_polygon() -> None:
        pts = draft["pts"]
        panel = draft["panel"]
        if panel is None or len(pts) < 3:
            return
        target = gold_polys if panel == "gold" else live_polys
        target.add(draw_kind, pts)
        draft["pts"] = []
        draft["panel"] = None
        _persist()
        print(f"  closed {panel} {draw_kind} ({len(pts)} pts)  gold {gold_polys.counts()}  live {live_polys.counts()}")

    def _on_mouse(event, x, y, _flags, _userdata) -> None:
        nonlocal frozen_live
        mapped = _map_overlay_click(x, y, state["layout"])
        if mapped is None:
            return
        which, px, py = mapped
        if event == cv2.EVENT_RBUTTONDOWN:
            _close_polygon()
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if draft["pts"] and draft["panel"] != which:
            print(f"  finish the {draft['panel']} polygon first (right-click or c)")
            return
        if not draft["pts"]:
            draft["panel"] = which
            if which == "live" and state["live"] is not None:
                frozen_live = state["live"].copy()
        draft["pts"].append((int(px), int(py)))

    print(
        f"Gold: {gold_path}\n"
        f"Trace orange pads and white holders: click vertices, right-click or c to close.\n"
        f"Draw on gold (left) and live (middle). Gold outlines are ghosted on live.\n"
        f"o/w class   c close   z undo vertex   x del last poly   r reset\n"
        f"f freeze live   g ghost gold   m view   s save {dump_dir / 'compare.png'}   q quit\n"
        f"a turn the remap so the live outlines land on gold's (saves the remap)\n"
        f"Polygons: {polys_path}"
    )
    _wait_enter("Press Enter to open the live overlay. ")

    frames = _iter_sized(cam, capture_wh[0], capture_wh[1])
    live_policy = None
    last_lines: list[str] = []
    last_win: tuple[int, int] | None = None
    try:
        cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(_WINDOW, _on_mouse)
        for bgr in frames:
            if bgr is None:
                continue
            frame = bgr
            if remap is not None:
                frame = remap.apply(frame)
            rgb = np.ascontiguousarray(frame[..., ::-1])
            live_policy = pipeline.process_frame(rgb)
            state["live"] = live_policy
            if frozen_live is None:
                frozen_live = live_policy.copy()
            elif not freeze_live and not (draft["pts"] and draft["panel"] == "live"):
                frozen_live = live_policy.copy()
            shown = _shown_live()
            if shown is None:
                continue
            view = CONTACT_SHEET_VIEWS[view_i]
            masks = extract_align_masks(
                gold, shown, gold_polys=gold_polys, live_polys=live_polys
            )
            metrics = compute_align_metrics(gold, shown, masks=masks)
            draft_n = len(draft["pts"])
            status = (
                f"draw={draw_kind}  gold {gold_polys.counts()}  live {live_polys.counts()}  "
                f"draft={draft_n}pt  freeze={'on' if freeze_live or (draft_n and draft['panel']=='live') else 'off'}  "
                f"[o/w] [c]close [z]undo [x]del [r]reset [f]freeze [g]ghost [m]view"
            )
            last_lines = [*metrics.lines(), status]
            sheet = render_contact_sheet(
                gold,
                shown,
                extra_lines=last_lines,
                scale=1,
                view=view,
                masks=masks,
                gold_polys=gold_polys,
                live_polys=live_polys,
                draft_panel=draft["panel"],
                draft_pts=draft["pts"] or None,
                ghost_gold_on_live=ghost_gold,
            )
            display, win_w, win_h = _fit_image_to_screen(sheet)
            n_rows = 2 if view == "both" else 1
            state["layout"] = {
                "display_hw": display.shape[:2],
                "sheet_hw": sheet.shape[:2],
                "panel_hw": gold.shape[:2],
                "n_rows": n_rows,
            }
            if last_win != (win_w, win_h):
                cv2.resizeWindow(_WINDOW, win_w, win_h)
                if last_win is None:
                    _, _, sx, sy = _screen_geom()
                    cv2.moveWindow(_WINDOW, sx + 16, sy + 40)
                last_win = (win_w, win_h)
            cv2.imshow(_WINDOW, np.ascontiguousarray(display[:, :, ::-1]))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                _persist()
                if last_lines:
                    print("\n".join(last_lines))
                return
            if key == ord("m"):
                view_i = (view_i + 1) % len(CONTACT_SHEET_VIEWS)
            elif key == ord("o"):
                draw_kind = "orange"
            elif key == ord("w"):
                draw_kind = "white"
            elif key in (ord("c"), 13):
                _close_polygon()
            elif key in (ord("z"), 8):
                if draft["pts"]:
                    draft["pts"].pop()
                    if not draft["pts"]:
                        draft["panel"] = None
            elif key == ord("x"):
                panel = draft["panel"] or "gold"
                target = gold_polys if panel == "gold" else live_polys
                target.pop(draw_kind)
                _persist()
                print(f"  deleted last {panel} {draw_kind}  gold {gold_polys.counts()}  live {live_polys.counts()}")
            elif key == ord("r"):
                gold_polys.clear()
                live_polys.clear()
                draft["pts"] = []
                draft["panel"] = None
                _persist()
                print("  polygons reset")
            elif key == ord("f"):
                freeze_live = not freeze_live
                if freeze_live and live_policy is not None:
                    frozen_live = live_policy.copy()
                print(f"  freeze live {'on' if freeze_live else 'off'}")
            elif key == ord("g"):
                ghost_gold = not ghost_gold
            elif key == ord("a"):
                if remap is None or pipeline.letterbox_to_square:
                    print("  a needs an optical remap and UMI's centre crop (letterbox_to_square: false)")
                    continue
                solved = _solve_view_rotation(
                    gold_polys, live_polys, remap.aligner, (gold.shape[1], gold.shape[0])
                )
                if solved is None:
                    print("  a needs at least two outlines traced on both gold and live")
                    continue
                rot, before_px, after_px, move = solved
                pitch, yaw, roll = Rotation.from_matrix(rot).as_rotvec(degrees=True)
                with open(remap_path, "r") as fp:
                    data = json.load(fp)
                data["transform"] = (rot @ np.asarray(data["transform"], dtype=np.float64)).tolist()
                with open(remap_path, "w") as fp:
                    json.dump(data, fp, indent=2)
                remap = OpticalRemap(str(remap_path), conf_dir())
                for kind in ("orange", "white"):
                    polys = getattr(live_polys, kind)
                    polys[:] = [move(p) for p in polys]
                _persist()
                frozen_live = None
                print(
                    f"  view turned {pitch:+.2f} deg pitch, {yaw:+.2f} yaw, {roll:+.2f} roll: "
                    f"outlines {before_px:.1f} -> {after_px:.1f} px rms. Saved to {remap_path}.\n"
                    "  Live outlines were moved with it; redraw them to check. The hand-eye\n"
                    "  axes turn with the remap, so re-run solarian calibrate-hand-eye after."
                )
            elif key == ord("s") and shown is not None:
                dump_dir.mkdir(parents=True, exist_ok=True)
                out = dump_dir / "compare.png"
                _imwrite_rgb(
                    out,
                    render_contact_sheet(
                        gold,
                        shown,
                        extra_lines=last_lines,
                        view=view,
                        masks=masks,
                        gold_polys=gold_polys,
                        live_polys=live_polys,
                        ghost_gold_on_live=ghost_gold,
                    ),
                )
                _persist()
                print(f"Wrote {out}")
                print("\n".join(last_lines))
    finally:
        frames.close()
        try:
            cv2.destroyWindow(_WINDOW)
            cv2.waitKey(1)
        except Exception:
            pass


@CommandRegistry.register("calibrate-umi-camera", "controller")
class CalibrateUmiCameraCommand(BaseCommand):
    category = "Controller"
    setup_modules = ()
    help = (
        "Calibrate a connected camera onto GoPro Hero 10 + Max Lens Mod optics "
        "and overlay vs UMI gold."
    )
    description = """
    Walks the whole alignment in one sitting:

      1. Pick the camera to calibrate (same enumeration as list-cameras)
      2. Optionally pick a GoPro on the SAME mount for a paired homography
      3. Capture a 9x6 checkerboard from at least 8 varied views
      4. Fisheye-calibrate it and remap it onto the shipped GoPro HDMI 1080p model
      5. Write the remap JSON and point the camera yaml at it, at 640x360
      6. Open the live gold overlay to check fingers, holders and screws

    Run it with the robot stopped: run-robot and run-server hold the camera. Once a
    remap exists, overlay_only=true skips straight to step 6, which is how the
    camera gets aimed after it is mounted.

    In the overlay, click to trace polygons on the gold (left) and live (middle)
    panels: o/w pick the class, right-click or c closes one, z undoes a vertex,
    x deletes the last, r resets, f freezes the live frame, g ghosts the gold
    outlines, m cycles views, s saves a contact sheet, q quits. With the same
    outlines traced on both panels, a solves the rotation that carries the live
    ones onto gold's -- a camera tilted on its mount -- and writes it into the
    remap's transform.
    """
    params = {
        "config": "UMI skill yaml to read camera and image flags from; a filename "
                  "in conf/ or a path (default: umi.yaml). arm1's cup arrange "
                  "file is config=cup_arrange.yaml.",
        "gold": "Path to a custom gold policy image; defaults to the UMI image "
                "shipped with solarian-umi.",
        "camera_yaml": "Camera config to read the capture size from and write the "
                       "remap into; defaults to the skill's camera module.",
        "images": "Directory of checkerboard PNGs to reuse instead of capturing.",
        "gopro_images": "Matching GoPro captures for the paired homography, named "
                        "to pair with images.",
        "overlay_only": "true to skip capture and calibration and open the live "
                        "overlay against the remap the camera yaml already names "
                        "(default: false).",
        "skip_overlay": "true to stop after writing the remap (default: false).",
    }
    examples = (
        "solarian calibrate-umi-camera",
        "solarian calibrate-umi-camera overlay_only=true",
        "solarian calibrate-umi-camera images=.cache/umi_camera_calib/20260919_101530/user",
        "solarian calibrate-umi-camera camera_yaml=right_wrist_camera_arducam_lens.yaml",
    )

    def handle(self, cfg: DictConfig) -> None:
        from solarian.cli.commands.list_cameras import (
            _print_cameras,
            _prompt_camera_choice,
            find_cameras,
        )
        from solarian.core.optical_remap import build_remap_from_images

        skill_cfg = _load_skill_cfg(cfg)
        camera_yaml = _camera_yaml_path(cfg, skill_cfg)
        capture_wh = _load_camera_capture_size(camera_yaml)
        dump_dir = Path(str(skill_cfg.get("debug_image_path", _DUMP_DEFAULT)))
        if not dump_dir.is_absolute():
            dump_dir = _project_dir() / dump_dir

        overlay_only = _as_bool(_cfg_get(cfg, "overlay_only"))
        skip_overlay = _as_bool(_cfg_get(cfg, "skip_overlay"))
        images = _cfg_get(cfg, "images")
        gopro_images = _cfg_get(cfg, "gopro_images")

        gold = None
        gold_path = None
        try:
            gold_path = _resolve_gold(cfg, skill_cfg)
            from solarian_umi.umi.compare_policy_image import _imread_rgb

            gold = _imread_rgb(gold_path)
        except FileNotFoundError as exc:
            if not skip_overlay:
                print(f"Warning: {exc}")
            gold = None

        selected = None
        gopro_cam = None

        if overlay_only:
            if gold is None:
                raise FileNotFoundError("overlay_only needs a gold image")
            print("Scanning for cameras...\n")
            cameras = find_cameras()
            if not cameras:
                raise RuntimeError("No cameras detected.")
            _print_cameras(cameras)
            choice = _prompt_camera_choice(len(cameras))
            if choice is None:
                return
            selected = cameras[choice - 1]
            remap_raw = None
            if camera_yaml.is_file():
                ycfg = load_yaml(camera_yaml)
                remap_raw = ((ycfg.get("sensors") or {}).get("camera") or {}).get(
                    "optical_remap"
                )
            remap_path = None
            if remap_raw:
                from solarian.core.optical_remap import resolve_remap_path

                remap_path = resolve_remap_path(str(remap_raw), conf_dir())
            _run_overlay(
                gold=gold,
                gold_path=gold_path,
                dump_dir=dump_dir,
                skill_cfg=skill_cfg,
                cam=selected,
                remap_path=remap_path,
                capture_wh=capture_wh,
            )
            return

        session = _project_dir() / _CAPTURE_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
        user_dir: Path | None = Path(str(images)) if images else None
        gopro_dir: Path | None = Path(str(gopro_images)) if gopro_images else None

        if user_dir is None:
            print("Scanning for cameras...\n")
            cameras = find_cameras()
            if not cameras:
                raise RuntimeError("No cameras detected.")
            _print_cameras(cameras)
            print("Select the camera to calibrate (the one that will mount on the gripper).")
            choice = _prompt_camera_choice(len(cameras))
            if choice is None:
                return
            selected = cameras[choice - 1]
            print(
                f"Calibrating: [{choice}] {selected['model']} "
                f"({selected['backend']} {selected['port']})"
            )

            if gopro_dir is None:
                rest = [c for i, c in enumerate(cameras, start=1) if i != choice]
                if rest:
                    print(
                        "\nOptionally select a GoPro only if it is on the SAME mount "
                        "(same viewpoint) as the camera you just picked. A homography "
                        "from two cameras in different places will yaw the gripper "
                        "view off-center. Press Enter to skip (recommended)."
                    )
                    _print_cameras(rest)
                    raw = _prompt(
                        f"GoPro camera? (1-{len(rest)}, or Enter to skip): "
                    ).lower()
                    if raw.isdigit() and 1 <= int(raw) <= len(rest):
                        gopro_cam = rest[int(raw) - 1]
                        print(
                            f"GoPro: {gopro_cam['model']} "
                            f"({gopro_cam['backend']} {gopro_cam['port']})"
                        )
                    else:
                        print("Skipping GoPro; optics-only (identity) remap onto shipped K/D.")

            _print_checkerboard_links(session)
            if gopro_cam is not None:
                user_dir, gopro_dir = _capture_paired(
                    selected, gopro_cam,
                    session / "user", session / "gopro",
                    capture_wh,
                )
            else:
                user_dir = _capture_solo(
                    selected, session / "user", capture_wh[0], capture_wh[1]
                )
        else:
            user_dir = Path(str(images))
            if not user_dir.is_dir():
                raise NotADirectoryError(user_dir)
            print(f"Skipping capture; using images={user_dir}")
            if gopro_dir is not None:
                gopro_dir = Path(str(gopro_images))
                print(f"Paired GoPro images={gopro_dir}")

        cam_module = camera_yaml.stem if camera_yaml.suffix == ".yaml" else "camera"
        calib_dir = calibrations_dir()
        calib_dir.mkdir(parents=True, exist_ok=True)
        remap_path = calib_dir / f"{cam_module}_gopro_remap.json"
        geometry_dir = session / "geometry" if session.exists() or user_dir else None
        if geometry_dir is None:
            geometry_dir = _project_dir() / _CAPTURE_ROOT / "geometry"

        _aligner, data = build_remap_from_images(
            str(user_dir),
            gopro_dir=str(gopro_dir) if gopro_dir else None,
            board=_BOARD,
            out_path=str(remap_path),
            geometry_dir=str(geometry_dir) if gopro_dir else None,
        )
        metrics = data.get("metrics") or {}
        print("\n--- remap metrics ---")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        remap_rel = os.path.relpath(remap_path, _project_dir())
        if camera_yaml.is_file():
            _patch_camera_yaml(camera_yaml, remap_rel, _REMAP_RESIZE)
            print(
                f"\nUpdated {camera_yaml}:\n"
                f"    optical_remap: {remap_rel}\n"
                f"    resize_width: {_REMAP_RESIZE[0]}\n"
                f"    resize_height: {_REMAP_RESIZE[1]}"
            )
        else:
            print(
                f"\nAdd to the camera YAML:\n"
                f"    optical_remap: {remap_rel}\n"
                f"    resize_width: {_REMAP_RESIZE[0]}\n"
                f"    resize_height: {_REMAP_RESIZE[1]}"
            )

        if skip_overlay:
            return
        if gold is None:
            print("Skipping overlay (no gold image).")
            return
        if selected is None:
            print("Scanning for cameras for the gold overlay...\n")
            cameras = find_cameras()
            if not cameras:
                print("No cameras detected; skipping overlay.")
                return
            _print_cameras(cameras)
            choice = _prompt_camera_choice(len(cameras))
            if choice is None:
                return
            selected = cameras[choice - 1]

        _run_overlay(
            gold=gold,
            gold_path=gold_path,
            dump_dir=dump_dir,
            skill_cfg=skill_cfg,
            cam=selected,
            remap_path=remap_path,
            capture_wh=capture_wh,
        )
