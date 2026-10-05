"""Per-demo sanity numbers for ORB-SLAM3 camera trajectories.

A handheld tabletop demo moves the camera well under a meter, and at 30 to
60 fps a tracked frame moves a few millimeters. Meters of extent or
centimeter jumps mean the visual-inertial solution diverged, which the
dataset plan would otherwise turn into actions.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_EXTENT_M = 1.0
MAX_STEP_M = 0.05
MAX_LOST_FRACTION = 0.3
# The gate stops the pipeline when more than this share of demos fail.
MAX_BAD_DEMO_FRACTION = 0.5


@dataclass(frozen=True)
class DemoTrajectory:
    name: str
    n_frames: int
    lost_fraction: float
    extent_m: float
    max_step_m: float

    @property
    def ok(self) -> bool:
        return (
            self.lost_fraction <= MAX_LOST_FRACTION
            and self.extent_m <= MAX_EXTENT_M
            and self.max_step_m <= MAX_STEP_M
        )


def _is_lost(value: str) -> bool:
    return value.strip().lower() in ("true", "1")


def read_trajectory(csv_path: Path) -> DemoTrajectory:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    lost = np.array([_is_lost(row["is_lost"]) for row in rows], dtype=bool)
    xyz = np.array(
        [[float(row["x"]), float(row["y"]), float(row["z"])] for row in rows],
        dtype=np.float64,
    ).reshape(-1, 3)
    tracked = xyz[~lost]
    if len(tracked) == 0:
        extent = 0.0
        max_step = 0.0
    else:
        # Largest axis span, so a demo that only drifts along one axis still counts.
        extent = float(np.max(tracked.max(axis=0) - tracked.min(axis=0)))
        steps = np.linalg.norm(np.diff(tracked, axis=0), axis=1)
        max_step = float(steps.max()) if len(steps) else 0.0
    n = len(rows)
    return DemoTrajectory(
        name=csv_path.parent.name,
        n_frames=n,
        lost_fraction=float(lost.mean()) if n else 1.0,
        extent_m=extent,
        max_step_m=max_step,
    )


def demo_trajectories(demos_dir: Path) -> list[DemoTrajectory]:
    return [read_trajectory(path) for path in sorted(demos_dir.glob("demo_*/camera_trajectory.csv"))]


def format_report(trajectories: list[DemoTrajectory]) -> str:
    lines = [f"{'demo':<48} {'frames':>6} {'lost':>5} {'extent m':>9} {'max step m':>10}"]
    for t in trajectories:
        flag = "" if t.ok else "  FAIL"
        lines.append(
            f"{t.name:<48} {t.n_frames:>6} {t.lost_fraction:>5.0%} "
            f"{t.extent_m:>9.3f} {t.max_step_m:>10.3f}{flag}"
        )
    if trajectories:
        bad = sum(not t.ok for t in trajectories)
        lost = np.median([t.lost_fraction for t in trajectories])
        extent = np.median([t.extent_m for t in trajectories])
        step = np.median([t.max_step_m for t in trajectories])
        lines.append(
            f"{len(trajectories)} demos, {bad} fail "
            f"(limits: lost {MAX_LOST_FRACTION:.0%}, extent {MAX_EXTENT_M} m, "
            f"step {MAX_STEP_M} m). Median lost {lost:.0%}, extent {extent:.3f} m, "
            f"max step {step:.3f} m."
        )
    else:
        lines.append("No demo trajectories found.")
    return "\n".join(lines)


def gate(trajectories: list[DemoTrajectory]) -> None:
    """Stop before the dataset plan when most demos diverged."""
    if not trajectories:
        raise SystemExit("No demo has a camera_trajectory.csv; SLAM produced nothing.")
    bad = sum(not t.ok for t in trajectories)
    if bad / len(trajectories) > MAX_BAD_DEMO_FRACTION:
        raise SystemExit(
            f"{bad} of {len(trajectories)} demo trajectories fail the sanity limits. "
            "Not writing a dataset from them."
        )


def exclude_failed(demos_dir: Path, trajectories: list[DemoTrajectory], excluded_dir: Path) -> list[str]:
    """Move demos that fail the limits out of ``demos_dir``, so no later step turns them into data."""
    failed = [t.name for t in trajectories if not t.ok]
    if failed:
        excluded_dir.mkdir(exist_ok=True)
        for name in failed:
            (demos_dir / name).rename(excluded_dir / name)
    return failed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--gate", action="store_true", help="Exit non-zero when most demos fail.")
    ns = parser.parse_args(argv)
    trajectories = demo_trajectories(ns.session_dir.expanduser() / "demos")
    print(format_report(trajectories))
    if ns.gate:
        gate(trajectories)


if __name__ == "__main__":
    main()
