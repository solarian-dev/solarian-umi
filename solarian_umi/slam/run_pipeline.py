# Adapted from Universal Manipulation Interface.
# Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab.
# https://github.com/real-stanford/universal_manipulation_interface
# Licensed under the MIT License. The full text is LICENSE.umi
# beside this package.

"""
Main script for UMI SLAM pipeline.
python run_slam_pipeline.py <session_dir>
"""

import sys
import os


# %%
import json
import pathlib
import click
import subprocess

from solarian_umi.slam.docker import LUMI_ORB_SLAM3_IMAGE
from solarian_umi.slam.lumi_camera import FINGER_TAG_Z_M
from solarian_umi.slam.lumi_prepare import EXCLUDED_DEMOS_DIR, prepare_lumi_session
from solarian_umi.slam.map_selection import MAP_SETTINGS, SELECTION_NAME, mapping_key, select_map
from solarian_umi.slam.remap_video import COVERAGE_NAME
from solarian_umi.slam.trajectory_check import demo_trajectories, exclude_failed, format_report, gate

# %%
@click.command()
@click.argument('session_dir', nargs=-1)
@click.option('-c', '--calibration_dir', type=str, default=None)
@click.option('--camera', type=click.Choice(['gopro', 'lumi']), default='gopro')
@click.option('--slam-settings', 'slam_settings', type=str, default=None)
@click.option('--gripper-tag-threshold', 'gripper_tag_threshold', type=float, default=None)
@click.option('--map-candidates', 'map_candidates', type=int, default=6,
              help='Maps built from the mapping video; the one most probe demos track on is kept.')
@click.option('--map-library', 'map_library', type=str, default=None,
              help='Directory keeping the best map of each mapping take, with the settings it was built '
                   'with; a later run on the same take scores that map against its new ones.')
@click.option('--probe-demos', 'probe_demos', type=int, default=24,
              help='Demos, spread over the session, each map candidate is scored on.')
@click.option('--retries', type=int, default=2,
              help='Passes that track again, on the same map, the demos still without a passing trajectory.')
@click.option('--rolling-shutter', 'rolling_shutter', is_flag=True,
              help='Lumi: undo the measured rolling shutter in the remap.')
def main(session_dir, calibration_dir, camera, slam_settings, gripper_tag_threshold, map_candidates, map_library,
         probe_demos, retries, rolling_shutter):
    script_dir = pathlib.Path(__file__).parent.joinpath('scripts')
    if calibration_dir is None:
        calibration_dir = pathlib.Path(__file__).parent.joinpath('calibration')
    else:
        calibration_dir = pathlib.Path(calibration_dir)
    assert calibration_dir.is_dir()

    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()

        print("############## 00_process_videos #############")
        script_path = script_dir.joinpath("00_process_videos.py")
        assert script_path.is_file()
        cmd = [
            sys.executable, str(script_path),
            str(session)
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise SystemExit(result.returncode)

        demo_dir = session.joinpath('demos')
        mapping_dir = demo_dir.joinpath('mapping')
        assert mapping_dir.is_dir()
        aruco_config = calibration_dir.joinpath('aruco_config.yaml')
        assert aruco_config.is_file()
        camera_intrinsics = calibration_dir.joinpath('gopro_intrinsics_2_7k.json')
        slam_settings_path = None
        if slam_settings is not None:
            slam_settings_path = pathlib.Path(os.path.expanduser(slam_settings)).absolute()
        coverage_mask = None
        finger_options = ['--nominal_z', str(FINGER_TAG_Z_M)] if camera == 'lumi' else []

        if camera == 'lumi':
            print("############# 01_prepare_lumi ###########")
            # Camera files live in the project the command runs from.
            lumi_settings = prepare_lumi_session(
                session, camera_intrinsics, [pathlib.Path.cwd().joinpath('conf')],
                undo_rolling_shutter=rolling_shutter)
            slam_settings_path = slam_settings_path or lumi_settings
            coverage_mask = session.joinpath(COVERAGE_NAME)
        else:
            print("############# 01_extract_gopro_imu ###########")
            script_path = script_dir.joinpath("01_extract_gopro_imu.py")
            assert script_path.is_file()
            cmd = [
                sys.executable, str(script_path),
                str(session)
            ]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                raise SystemExit(result.returncode)

        slam_options = []
        if coverage_mask is not None:
            slam_options.extend(['--coverage_mask', str(coverage_mask)])
        if camera == 'lumi':
            slam_options.extend(['--docker_image', LUMI_ORB_SLAM3_IMAGE])

        print("############# 02_create_map ###########")
        map_path = mapping_dir.joinpath('map_atlas.osa')
        if not map_path.is_file():
            library_entry = None
            if map_library is not None:
                library_entry = pathlib.Path(os.path.expanduser(map_library)).absolute().joinpath(
                    mapping_key(session))
            selection = select_map(mapping_dir, demo_dir, script_dir, slam_options, slam_settings_path,
                                   candidates=map_candidates, probes=probe_demos, library_entry=library_entry)
            session.joinpath(SELECTION_NAME).write_text(json.dumps(selection, indent=2) + '\n', encoding='utf-8')
            assert map_path.is_file()
        map_settings = mapping_dir.joinpath(MAP_SETTINGS)
        if map_settings.is_file():
            slam_settings_path = map_settings

        print("############# 03_batch_slam ###########")
        script_path = script_dir.joinpath("03_batch_slam.py")
        assert script_path.is_file()
        cmd = [
            sys.executable, str(script_path),
            '--input_dir', str(demo_dir),
            '--map_path', str(map_path),
            *slam_options,
        ]
        if slam_settings_path is not None:
            cmd.extend(['--slam_settings', str(slam_settings_path)])
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise SystemExit(result.returncode)

        # Tracking on a map is not repeatable: a demo that diverges on one run often passes on the next.
        for attempt in range(1, retries + 1):
            tracked = {t.name for t in demo_trajectories(demo_dir) if t.ok}
            retry = [d for d in demo_dir.glob('demo_*') if d.name not in tracked]
            if not retry:
                break
            print(f"Retry {attempt} of {retries}: tracking {len(retry)} demo(s) again that have no trajectory "
                  f"or fail the limits.")
            for demo in retry:
                demo.joinpath('camera_trajectory.csv').unlink(missing_ok=True)
            result = subprocess.run(cmd)
            if result.returncode != 0:
                raise SystemExit(result.returncode)

        print("############# 03b_check_trajectories ###########")
        trajectories = demo_trajectories(demo_dir)
        report = format_report(trajectories)
        print(report)
        session.joinpath('trajectory_report.txt').write_text(report + '\n', encoding='utf-8')
        gate(trajectories)
        excluded = session.joinpath(EXCLUDED_DEMOS_DIR)
        failed = exclude_failed(demo_dir, trajectories, excluded)
        if failed:
            print(f"Left out {len(failed)} demo(s) whose trajectories fail the limits (moved to {excluded}).")

        print("############# 04_detect_aruco ###########")
        script_path = script_dir.joinpath("04_detect_aruco.py")
        assert script_path.is_file()
        assert camera_intrinsics.is_file()

        cmd = [
            sys.executable, str(script_path),
            '--input_dir', str(demo_dir),
            '--camera_intrinsics', str(camera_intrinsics),
            '--aruco_yaml', str(aruco_config)
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise SystemExit(result.returncode)

        print("############# 05_run_calibrations ###########")
        script_path = script_dir.joinpath("05_run_calibrations.py")
        assert script_path.is_file()
        cmd = [
            sys.executable, str(script_path),
            str(session),
            *finger_options,
        ]
        if gripper_tag_threshold is not None:
            cmd.extend(['--tag_det_threshold', str(gripper_tag_threshold)])
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise SystemExit(result.returncode)

        print("############# 06_generate_dataset_plan ###########")
        script_path = script_dir.joinpath("06_generate_dataset_plan.py")
        assert script_path.is_file()
        cmd = [
            sys.executable, str(script_path),
            '--input', str(session),
            *finger_options,
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise SystemExit(result.returncode)

## %%
if __name__ == "__main__":
    main()
