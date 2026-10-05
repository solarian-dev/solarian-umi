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
import pathlib
import click
import subprocess

from solarian_umi_skill.slam.docker import LUMI_ORB_SLAM3_IMAGE
from solarian_umi_skill.slam.lumi_camera import FINGER_TAG_Z_M
from solarian_umi_skill.slam.lumi_prepare import EXCLUDED_DEMOS_DIR, prepare_lumi_session
from solarian_umi_skill.slam.remap_video import COVERAGE_NAME
from solarian_umi_skill.slam.trajectory_check import demo_trajectories, exclude_failed, format_report, gate

# %%
@click.command()
@click.argument('session_dir', nargs=-1)
@click.option('-c', '--calibration_dir', type=str, default=None)
@click.option('--camera', type=click.Choice(['gopro', 'lumi']), default='gopro')
@click.option('--slam-settings', 'slam_settings', type=str, default=None)
def main(session_dir, calibration_dir, camera, slam_settings):
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
                session, camera_intrinsics, [pathlib.Path.cwd().joinpath('conf')])
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
        if slam_settings_path is not None:
            slam_options.extend(['--slam_settings', str(slam_settings_path)])
        if coverage_mask is not None:
            slam_options.extend(['--coverage_mask', str(coverage_mask)])
        if camera == 'lumi':
            slam_options.extend(['--docker_image', LUMI_ORB_SLAM3_IMAGE])

        print("############# 02_create_map ###########")
        script_path = script_dir.joinpath("02_create_map.py")
        assert script_path.is_file()
        map_path = mapping_dir.joinpath('map_atlas.osa')
        if not map_path.is_file():
            cmd = [
                sys.executable, str(script_path),
                '--input_dir', str(mapping_dir),
                '--map_path', str(map_path),
                *slam_options,
            ]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                raise SystemExit(result.returncode)
            assert map_path.is_file()

        print("############# 03_batch_slam ###########")
        script_path = script_dir.joinpath("03_batch_slam.py")
        assert script_path.is_file()
        cmd = [
            sys.executable, str(script_path),
            '--input_dir', str(demo_dir),
            '--map_path', str(map_path),
            *slam_options,
        ]
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
