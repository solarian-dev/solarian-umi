"""Side-by-side gold vs Solarian policy-input images.

The gold UMI eval dump is post-pipeline 224×224 RGB (what the policy saw).
Solarian dumps the same layout as ``solarian_camera0_rgb_t0.png``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

GOLD_NAME = "umi_gold_camera0_rgb_t0.png"
SOLARIAN_NAME = "solarian_camera0_rgb_t0.png"
COMPARE_NAME = "compare.png"
DEFAULT_GOLD = Path(__file__).resolve().parent / GOLD_NAME


def _imread_rgb(path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _imwrite_rgb(path: Path, rgb: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {path}")


def _black_frac(rgb: np.ndarray, thresh: int = 16) -> float:
    black = np.all(rgb < thresh, axis=2)
    return float(black.mean())


def _finger_notes(rgb: np.ndarray, thresh: int = 16) -> str:
    """Crude finger-base location: lowest non-black pixel in left/right thirds."""
    h, w = rgb.shape[:2]
    vis = ~np.all(rgb < thresh, axis=2)
    notes = []
    for label, x0, x1 in (
        ("left", 0, w // 3),
        ("right", 2 * w // 3, w),
    ):
        col = vis[:, x0:x1]
        if not col.any():
            notes.append(f"{label} finger: none")
            continue
        rows = np.where(col.any(axis=1))[0]
        notes.append(
            f"{label} finger y={int(rows[0])}..{int(rows[-1])} "
            f"(base {h - 1 - int(rows[-1])}px from bottom)"
        )
    return "; ".join(notes)


def describe_pair(gold: np.ndarray, solarian: np.ndarray) -> list[str]:
    gh, gw = gold.shape[:2]
    sh, sw = solarian.shape[:2]
    size_ok = (gh, gw) == (sh, sw)
    return [
        f"Size: gold {gw}x{gh} vs solarian {sw}x{sh}"
        + (" (match)" if size_ok else " (MISMATCH)"),
        f"Tunnel: gold black={_black_frac(gold):.1%}  "
        f"solarian black={_black_frac(solarian):.1%}  "
        "(fisheye circle vs black corners / crop tightness)",
        f"Mask: compare black polygons on gripper body and L/R edges "
        f"(not orange fingers)",
        f"Fingers gold: {_finger_notes(gold)}",
        f"Fingers solarian: {_finger_notes(solarian)}",
    ]


def _pad_to(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[: min(h, height), : min(w, width)] = rgb[: min(h, height), : min(w, width)]
    return canvas


def _caption_bar(width: int, lines: list[str], line_h: int = 18) -> np.ndarray:
    import cv2

    bar_h = 12 + line_h * len(lines)
    bar = np.zeros((bar_h, width, 3), dtype=np.uint8)
    y = 16
    for line in lines:
        cv2.putText(
            bar,
            line[:140],
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += line_h
    return bar


CONTACT_SHEET_VIEWS = ("absdiff", "mask_iou", "both")
_ORANGE_TINT = np.array([255, 140, 0], dtype=np.float32)
_WHITE_TINT = np.array([180, 210, 255], dtype=np.float32)
_ORANGE_LINE = (255, 120, 0)
_WHITE_LINE = (220, 240, 255)
_DRAFT_LINE = (255, 255, 0)
_GHOST_ORANGE = (180, 80, 0)
_GHOST_WHITE = (160, 180, 200)


def caption_bar_height(n_lines: int, line_h: int = 18) -> int:
    return 12 + line_h * n_lines


def _tint_mask(rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = rgb.astype(np.float32)
    if mask.any():
        out[mask] = out[mask] * (1.0 - alpha) + color * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay_gripper_masks(rgb: np.ndarray, orange: np.ndarray, white: np.ndarray) -> np.ndarray:
    out = _tint_mask(rgb, white, _WHITE_TINT)
    return _tint_mask(out, orange, _ORANGE_TINT)


def _iou_color_map(gold_mask: np.ndarray, live_mask: np.ndarray) -> np.ndarray:
    panel = np.zeros((*gold_mask.shape, 3), dtype=np.uint8)
    panel[gold_mask & ~live_mask] = (220, 40, 40)
    panel[live_mask & ~gold_mask] = (40, 90, 255)
    panel[gold_mask & live_mask] = (40, 220, 40)
    return panel


def gripper_iou_panel(masks) -> np.ndarray:
    """Stacked orange (top) / white (bottom) IoU maps: green overlap, red gold-only, blue live-only."""
    import cv2

    orange = _iou_color_map(masks.gold_orange, masks.live_orange)
    white = _iou_color_map(masks.gold_white, masks.live_white)
    h, w = orange.shape[:2]
    top = cv2.resize(orange, (w, h // 2), interpolation=cv2.INTER_NEAREST)
    bot = cv2.resize(white, (w, h - h // 2), interpolation=cv2.INTER_NEAREST)
    return np.concatenate([top, bot], axis=0)


def _draw_poly_list(rgb: np.ndarray, polygons, color, *, closed: bool = True) -> None:
    import cv2

    for poly in polygons or []:
        if len(poly) < 2:
            continue
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(rgb, [pts], closed and len(poly) >= 3, color, 1, cv2.LINE_AA)
        for x, y in poly:
            cv2.circle(rgb, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)


def draw_region_polygons(
    rgb: np.ndarray,
    polys,
    *,
    draft_pts=None,
    ghost=None,
) -> np.ndarray:
    """Outline closed polygons, optional in-progress draft, optional gold ghost."""
    out = rgb.copy()
    if ghost is not None:
        _draw_poly_list(out, ghost.orange, _GHOST_ORANGE)
        _draw_poly_list(out, ghost.white, _GHOST_WHITE)
    if polys is not None:
        _draw_poly_list(out, polys.orange, _ORANGE_LINE)
        _draw_poly_list(out, polys.white, _WHITE_LINE)
    if draft_pts:
        _draw_poly_list(out, [draft_pts], _DRAFT_LINE, closed=False)
    return out


def render_contact_sheet(
    gold: np.ndarray,
    solarian: np.ndarray,
    extra_lines: list[str] | None = None,
    *,
    scale: int = 1,
    view: str = "absdiff",
    masks=None,
    gold_polys=None,
    live_polys=None,
    draft_panel: str | None = None,
    draft_pts=None,
    ghost_gold_on_live: bool = True,
) -> np.ndarray:
    """RGB contact sheet: caption + gold | solarian | absdiff or mask IoU."""
    import cv2

    from solarian_umi.umi.gripper_align import extract_align_masks

    if view not in CONTACT_SHEET_VIEWS:
        view = "absdiff"
    lines = describe_pair(gold, solarian)
    if extra_lines:
        lines = [*lines, *extra_lines]
    h = max(gold.shape[0], solarian.shape[0])
    w = max(gold.shape[1], solarian.shape[1])
    g = _pad_to(gold, h, w)
    s = _pad_to(solarian, h, w)
    if masks is None:
        masks = extract_align_masks(g, s, gold_polys=gold_polys, live_polys=live_polys)
    g_show = overlay_gripper_masks(g, masks.gold_orange, masks.gold_white)
    s_show = overlay_gripper_masks(s, masks.live_orange, masks.live_white)
    g_show = draw_region_polygons(
        g_show,
        gold_polys,
        draft_pts=draft_pts if draft_panel == "gold" else None,
    )
    s_show = draw_region_polygons(
        s_show,
        live_polys,
        draft_pts=draft_pts if draft_panel == "live" else None,
        ghost=gold_polys if ghost_gold_on_live else None,
    )
    diff = np.abs(g.astype(np.int16) - s.astype(np.int16)).astype(np.uint8)
    iou = gripper_iou_panel(masks)
    absdiff_row = np.concatenate([g_show, s_show, diff], axis=1)
    iou_row = np.concatenate([g_show, s_show, iou], axis=1)
    if view == "mask_iou":
        panel = iou_row
        third = "orange/white IoU (G/R/B)"
    elif view == "both":
        panel = np.concatenate([absdiff_row, iou_row], axis=0)
        third = "|diff| + IoU"
    else:
        panel = absdiff_row
        third = "|gold-solarian|"
    headers = [
        f"gold (UMI eval)          solarian policy input          {third}",
        *lines,
    ]
    bar = _caption_bar(panel.shape[1], headers)
    sheet = np.concatenate([bar, panel], axis=0)
    if scale > 1:
        sheet = cv2.resize(
            sheet,
            (sheet.shape[1] * scale, sheet.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
    return sheet


def write_contact_sheet(
    dump_dir: Path,
    gold_path: Path | None = None,
    solarian_path: Path | None = None,
    extra_lines: list[str] | None = None,
) -> dict:
    """Write ``compare.png`` and return stats for ``summary.txt``."""
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)

    gold_src = Path(gold_path) if gold_path is not None else dump_dir / GOLD_NAME
    if not gold_src.is_file():
        gold_src = DEFAULT_GOLD
    if gold_src.is_file() and gold_src.resolve() != (dump_dir / GOLD_NAME).resolve():
        shutil.copy2(gold_src, dump_dir / GOLD_NAME)
        gold_src = dump_dir / GOLD_NAME

    solarian_src = (
        Path(solarian_path) if solarian_path is not None else dump_dir / SOLARIAN_NAME
    )
    gold = _imread_rgb(gold_src)
    solarian = _imread_rgb(solarian_src)
    lines = describe_pair(gold, solarian)
    if extra_lines:
        lines = [*lines, *extra_lines]
    sheet = render_contact_sheet(gold, solarian, extra_lines=extra_lines)
    out = dump_dir / COMPARE_NAME
    _imwrite_rgb(out, sheet)
    return {
        "gold_shape": tuple(gold.shape),
        "solarian_shape": tuple(solarian.shape),
        "lines": lines,
        "compare_path": str(out),
    }


def format_summary(
    *,
    policy_shape,
    policy_dtype,
    policy_min: float,
    policy_max: float,
    policy_mean: float,
    masked_shape=None,
    raw_shape=None,
    compare_lines: list[str] | None = None,
) -> str:
    parts = [
        "--- solarian policy camera0_rgb ---",
        f"shape={policy_shape} dtype={policy_dtype} "
        f"min={policy_min:.4f} max={policy_max:.4f} mean={policy_mean:.4f}",
    ]
    if masked_shape is not None:
        parts.append(f"masked HWC (pre-crop): shape={masked_shape}")
    if raw_shape is not None:
        parts.append(f"raw camera buffer: shape={raw_shape}")
    if compare_lines:
        parts.append("")
        parts.append("--- vs UMI gold ---")
        parts.extend(compare_lines)
    parts.append("")
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build gold vs Solarian policy-image contact sheet."
    )
    parser.add_argument(
        "dump_dir",
        nargs="?",
        default=".cache/umi_image_compare",
        help="Directory with solarian_camera0_rgb_t0.png (and optional gold copy).",
    )
    parser.add_argument(
        "--gold",
        default=None,
        help="Path to UMI gold camera0_rgb_t0.png (copied into dump_dir).",
    )
    args = parser.parse_args(argv)
    dump_dir = Path(args.dump_dir)
    gold = Path(args.gold) if args.gold else None
    stats = write_contact_sheet(dump_dir, gold_path=gold)
    print("\n".join(stats["lines"]))
    print(f"wrote {stats['compare_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
