"""Gripper-hardware superposition vs a gold UMI policy image.

Orange pads and white shells are user-drawn polygons on the 224×224
policy crop (gold and live independently). Screws are still found as
dark round blobs inside the white-holder polygons.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

_SCREW_V_MAX = 55
_SCREW_AREA_MIN = 3
_SCREW_AREA_MAX = 90
_SCREW_RADIUS = 4
_SCREWS_PER_SIDE = 3
ALIGN_POLYGONS_NAME = "align_polygons.json"


@dataclass
class RegionPolygons:
    """Closed polygons for orange pads and white holders on one image."""

    orange: list[list[tuple[int, int]]] = field(default_factory=list)
    white: list[list[tuple[int, int]]] = field(default_factory=list)

    def add(self, kind: str, pts: list[tuple[int, int]]) -> None:
        if len(pts) < 3:
            return
        getattr(self, kind).append([(int(x), int(y)) for x, y in pts])

    def pop(self, kind: str) -> None:
        polys = getattr(self, kind)
        if polys:
            polys.pop()

    def clear(self) -> None:
        self.orange.clear()
        self.white.clear()

    def counts(self) -> str:
        return f"o={len(self.orange)}/w={len(self.white)}"

    def mask(self, shape: tuple[int, int], kind: str) -> np.ndarray:
        return polygons_to_mask(getattr(self, kind), shape)

    def to_dict(self) -> dict:
        return {
            "orange": [[list(p) for p in poly] for poly in self.orange],
            "white": [[list(p) for p in poly] for poly in self.white],
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> RegionPolygons:
        if not data:
            return cls()

        def _polys(key: str) -> list[list[tuple[int, int]]]:
            out: list[list[tuple[int, int]]] = []
            for poly in data.get(key, []) or []:
                pts: list[tuple[int, int]] = []
                for p in poly:
                    if len(p) >= 2:
                        pts.append((int(p[0]), int(p[1])))
                if len(pts) >= 3:
                    out.append(pts)
            return out

        return cls(orange=_polys("orange"), white=_polys("white"))


@dataclass
class AlignMasks:
    gold_orange: np.ndarray
    live_orange: np.ndarray
    gold_white: np.ndarray
    live_white: np.ndarray


def polygons_to_mask(
    polygons: list[list[tuple[int, int]]],
    shape: tuple[int, int],
) -> np.ndarray:
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    for poly in polygons:
        if len(poly) < 3:
            continue
        pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    union = np.logical_or(a, b).sum(dtype=np.int64)
    if union == 0:
        return float("nan")
    return float(inter) / float(union)


def _coverage(gold: np.ndarray, live: np.ndarray) -> float:
    g = int(gold.sum())
    if g == 0:
        return float("nan")
    return float(np.logical_and(gold, live).sum(dtype=np.int64)) / float(g)


def _side_slice(w: int, side: str) -> slice:
    if side == "left":
        return slice(0, w // 2)
    return slice(w // 2, w)


def screw_centroids(
    rgb: np.ndarray,
    white: np.ndarray | None = None,
    *,
    side: str | None = None,
) -> np.ndarray:
    """(N, 2) float centroids ``(x, y)`` of dark blobs inside the white holders."""
    import cv2

    h, w = rgb.shape[:2]
    if white is None:
        white = np.zeros((h, w), dtype=bool)
    region = np.zeros_like(white)
    if side is None:
        region[:] = white
    else:
        sl = _side_slice(w, side)
        region[:, sl] = white[:, sl]
    if not region.any():
        return np.zeros((0, 2), dtype=np.float64)

    dil = cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    dark = np.all(rgb < _SCREW_V_MAX, axis=2) & dil
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dark.astype(np.uint8), connectivity=8
    )
    cands: list[tuple[float, np.ndarray]] = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < _SCREW_AREA_MIN or area > _SCREW_AREA_MAX:
            continue
        ys, xs = np.where(labels == i)
        if ys.size < 3:
            continue
        blob = (labels == i).astype(np.uint8)
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        peri = cv2.arcLength(cnts[0], True)
        circ = (4.0 * np.pi * area / (peri * peri)) if peri > 1e-6 else 0.0
        if circ < 0.35:
            continue
        cands.append((area, centroids[i].astype(np.float64)))
    cands.sort(key=lambda t: t[0], reverse=True)
    if not cands:
        return np.zeros((0, 2), dtype=np.float64)
    return np.stack([c for _, c in cands[:_SCREWS_PER_SIDE]], axis=0)


def _disk_mask(pts: np.ndarray, shape: tuple[int, int], radius: int) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    if pts.size == 0:
        return m
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    r2 = float(radius * radius)
    for x, y in pts:
        m |= (xx - x) ** 2 + (yy - y) ** 2 <= r2
    return m


def match_screws(
    gold_pts: np.ndarray,
    live_pts: np.ndarray,
    *,
    shape: tuple[int, int],
    radius: int = _SCREW_RADIUS,
) -> tuple[float, float]:
    """Return (mean pixel error, mean per-pair disk IoU). NaN if nothing to match."""
    if gold_pts.size == 0:
        return float("nan"), 1.0 if live_pts.size == 0 else 0.0
    if live_pts.size == 0:
        return float("nan"), 0.0
    cost = cdist(gold_pts, live_pts)
    ri, ci = linear_sum_assignment(cost)
    dists = cost[ri, ci]
    ious = []
    for g, l in zip(gold_pts[ri], live_pts[ci]):
        gm = _disk_mask(g.reshape(1, 2), shape, radius)
        lm = _disk_mask(l.reshape(1, 2), shape, radius)
        ious.append(_iou(gm, lm))
    n_unmatched = abs(len(gold_pts) - len(live_pts))
    ious.extend([0.0] * n_unmatched)
    return float(np.mean(dists)), float(np.mean(ious)) if ious else float("nan")


@dataclass
class AlignMetrics:
    orange_iou: float
    orange_coverage: float
    white_iou: float
    white_coverage: float
    screw_mean_px: float
    screw_overlap: float
    n_screws_gold: int
    n_screws_live: int

    def lines(self) -> list[str]:
        def pct(v: float) -> str:
            if v != v:  # NaN
                return "n/a"
            return f"{100.0 * v:.1f}%"

        px = "n/a" if self.screw_mean_px != self.screw_mean_px else f"{self.screw_mean_px:.1f}px"
        return [
            f"Orange fingers: IoU {pct(self.orange_iou)}  gold-coverage {pct(self.orange_coverage)}",
            f"White holders:  IoU {pct(self.white_iou)}  gold-coverage {pct(self.white_coverage)}",
            f"Screws ({self.n_screws_live}/{self.n_screws_gold}): "
            f"mean {px}  disk-overlap {pct(self.screw_overlap)}",
        ]


def extract_align_masks(
    gold_rgb: np.ndarray,
    live_rgb: np.ndarray,
    *,
    gold_polys: RegionPolygons | None = None,
    live_polys: RegionPolygons | None = None,
) -> AlignMasks:
    gold_polys = gold_polys or RegionPolygons()
    live_polys = live_polys or RegionPolygons()
    shape = gold_rgb.shape[:2]
    return AlignMasks(
        gold_orange=gold_polys.mask(shape, "orange"),
        live_orange=live_polys.mask(shape, "orange"),
        gold_white=gold_polys.mask(shape, "white"),
        live_white=live_polys.mask(shape, "white"),
    )


def compute_align_metrics(
    gold_rgb: np.ndarray,
    live_rgb: np.ndarray,
    *,
    gold_polys: RegionPolygons | None = None,
    live_polys: RegionPolygons | None = None,
    masks: AlignMasks | None = None,
) -> AlignMetrics:
    if gold_rgb.shape[:2] != live_rgb.shape[:2]:
        raise ValueError(
            f"gold {gold_rgb.shape[:2]} vs live {live_rgb.shape[:2]} (need same size)"
        )
    if masks is None:
        masks = extract_align_masks(
            gold_rgb, live_rgb, gold_polys=gold_polys, live_polys=live_polys
        )
    shape = gold_rgb.shape[:2]
    g_screws = []
    l_screws = []
    px_errs = []
    overlaps = []
    for side in ("left", "right"):
        gp = screw_centroids(gold_rgb, masks.gold_white, side=side)
        lp = screw_centroids(live_rgb, masks.live_white, side=side)
        g_screws.append(gp)
        l_screws.append(lp)
        mean_px, ov = match_screws(gp, lp, shape=shape)
        if mean_px == mean_px:
            px_errs.append(mean_px)
        if ov == ov:
            overlaps.append(ov)

    n_gold = int(sum(len(p) for p in g_screws))
    n_live = int(sum(len(p) for p in l_screws))
    return AlignMetrics(
        orange_iou=_iou(masks.gold_orange, masks.live_orange),
        orange_coverage=_coverage(masks.gold_orange, masks.live_orange),
        white_iou=_iou(masks.gold_white, masks.live_white),
        white_coverage=_coverage(masks.gold_white, masks.live_white),
        screw_mean_px=float(np.mean(px_errs)) if px_errs else float("nan"),
        screw_overlap=float(np.mean(overlaps)) if overlaps else float("nan"),
        n_screws_gold=n_gold,
        n_screws_live=n_live,
    )


def load_align_polygons(path: Path) -> tuple[RegionPolygons, RegionPolygons]:
    path = Path(path)
    if not path.is_file():
        return RegionPolygons(), RegionPolygons()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return RegionPolygons(), RegionPolygons()
    gold = RegionPolygons.from_dict(data.get("gold") if isinstance(data, dict) else None)
    live = RegionPolygons.from_dict(data.get("live") if isinstance(data, dict) else None)
    return gold, live


def save_align_polygons(path: Path, gold: RegionPolygons, live: RegionPolygons) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"gold": gold.to_dict(), "live": live.to_dict()}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
