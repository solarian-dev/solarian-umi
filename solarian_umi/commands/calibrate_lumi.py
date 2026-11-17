"""``solarian umi-calibrate-lumi`` — Lumi camera intrinsics from solarian's checkerboard on a screen."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig

from solarian.cli import BaseCommand, CommandRegistry


@CommandRegistry.register("umi-calibrate-lumi")
class UmiCalibrateLumiCommand(BaseCommand):
    help = "Calibrate a Lumi camera and lens live against the 9x6 checkerboard on this screen."
    category = "Training & Data"
    uses_config = False
    description = """
    Writes one calibration file per camera and lens, which umi-slam --camera
    lumi uses to remap the videos into GoPro Max Lens geometry.

    1. Put the collector on its Stream video screen.
    2. Run this command. A fullscreen window shows solarian's 9x6 checkerboard
       on the left and, on the right, what the Lumi sees with the detected
       corners and a 4x4 grid over the frame. Red edge cells have no corners
       yet; green ones do. A detected board is painted over in the preview,
       since the lens sees the preview too. What stays the same while the
       camera moves (the gripper) is tinted blue, and edge cells it mostly
       covers are not required, so the camera can stay mounted.
    3. Hold the Lumi in front of the screen and press SPACE when the corners
       are drawn. Move the board image through every red cell, tilted 30 to
       45 degrees and straight, near and far. u undoes the last view.
    4. Press c once all edge cells are green and there are 15 views or more.
       The file goes to conf/calibrations/cameras/<serial>.json unless --output says
       otherwise; umi-slam also reads <session>/camera_calibration.json first.

    Saved frames stay in .cache/lumi_calibration/<serial>_<time>/; pass that
    folder (or a recorded clip) instead of streaming to calibrate again.
    --from-remap writes the file from an existing solarian GoPro remap JSON
    instead (its checkerboard intrinsics, scaled to the recorded 1440x1080).
    """
    usage = "--serial S [--host HOST] [--output PATH] | <frames-dir|clip.mp4> | --from-remap PATH --serial S --output PATH"
    options = {
        "source": "Folder of saved frames or a Lumi MP4 of the board, instead of the live stream.",
        "--serial S": "Camera serial. Required live; defaults to the one stamped into a clip.",
        "--host HOST": "Collector hostname for the live stream (default: solarian-lumi.local).",
        "--port PORT": "Collector stream port (default: 8888).",
        "--output PATH": "Calibration file. Defaults to conf/calibrations/cameras/<serial>.json.",
        "--from-remap PATH": "Stand-in from a GoPro remap JSON's camera_intrinsics.",
    }
    examples = (
        "solarian umi-calibrate-lumi --serial LUMI07FC9785",
        "solarian umi-calibrate-lumi .cache/lumi_calibration/LUMI07FC9785_20261002_113000 --serial LUMI07FC9785",
        "solarian umi-calibrate-lumi --from-remap conf/calibrations/right_wrist_camera_gopro_remap_arducam.json "
        "--serial LUMI07FC9785 --output datasets/session/camera_calibration.json",
    )

    def parse_args(self, argv: list[str]) -> list[str]:
        parser = argparse.ArgumentParser(prog="solarian umi-calibrate-lumi")
        parser.add_argument("source", nargs="?")
        parser.add_argument("--output", "-o", default=None)
        parser.add_argument("--serial", default=None)
        parser.add_argument("--host", default="solarian-lumi.local")
        parser.add_argument("--port", type=int, default=8888)
        parser.add_argument("--from-remap", default=None)
        self.ns = parser.parse_args(argv)
        if not (self.ns.source or self.ns.from_remap or self.ns.serial):
            parser.error("live calibration needs --serial (the LUMI... id in the recorded MP4s)")
        return []

    def handle(self, cfg: DictConfig) -> None:
        from solarian_umi.slam.camera_calibration import (
            LUMI_FPS,
            MAX_RMS_PX,
            calibrate_views,
            calibration_document,
            detect_views,
            stand_in_from_remap,
            uncovered_cells,
        )

        ns = self.ns

        def calibrate(views, lit, static):
            missing = uncovered_cells(views, lit, static)
            if missing:
                raise ValueError(
                    f"The board never reached {len(missing)} edge cells (column, row on a 4x4 grid): {missing}."
                )
            model, rms, used = calibrate_views(views, (lit.shape[1], lit.shape[0]))
            if rms > MAX_RMS_PX:
                raise ValueError(f"RMS {rms:.3f} px is above {MAX_RMS_PX}; undo blurry views or add more.")
            return model, rms, used

        if ns.from_remap:
            if not ns.serial or not ns.output:
                raise SystemExit("--from-remap needs --serial and --output")
            document = stand_in_from_remap(Path(ns.from_remap).expanduser(), serial=ns.serial)
        elif ns.source:
            from solarian_umi.slam.support.mp4_meta import camera_serial

            source = Path(ns.source).expanduser()
            if source.is_dir() and not ns.serial:
                raise SystemExit("a folder of frames needs --serial")
            serial = ns.serial or camera_serial(source)
            views, lit, static = detect_views(source)
            print(f"{len(views)} board views in {source.name}")
            try:
                model, rms, used = calibrate(views, lit, static)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            document = calibration_document(
                model, serial=serial, source=f"checkerboard: {source.name}", rms=rms, n_views=used, fps=LUMI_FPS
            )
        else:
            from solarian_umi.slam.live_calibration import LumiStream, capture

            frames_dir = Path(".cache/lumi_calibration") / f"{ns.serial}_{datetime.now():%Y%m%d_%H%M%S}"
            stream = LumiStream(ns.host, ns.port)
            print(f"Streaming {stream.size[0]}x{stream.size[1]} from {ns.host}; frames go to {frames_dir}")
            try:
                result = capture(stream, frames_dir, calibrate)
            finally:
                stream.close()
            if result is None:
                print(f"Stopped without calibrating. Saved frames: {frames_dir}")
                return
            model, rms, used = result
            document = calibration_document(
                model, serial=ns.serial, source=f"checkerboard live: {frames_dir}", rms=rms, n_views=used, fps=LUMI_FPS
            )

        intr = document["intrinsics"]["intrinsics"]
        print(
            f"RMS {document['intrinsics']['final_reproj_error']:.3f} px on {document['intrinsics']['nr_calib_images']} "
            f"views; f {intr['focal_length']:.1f}, center ({intr['principal_pt_x']:.1f}, {intr['principal_pt_y']:.1f})"
        )
        output = Path(ns.output or f"conf/calibrations/cameras/{document['camera_serial']}.json").expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {output}")
