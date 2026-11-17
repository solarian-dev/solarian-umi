# Adapted from Universal Manipulation Interface.
# Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab.
# https://github.com/real-stanford/universal_manipulation_interface
# Licensed under the MIT License. The full text is LICENSE.umi
# beside this package.

"""
python scripts_slam_pipeline/00_process_videos.py -i data_workspace/toss_objects/20231113/mapping
"""

# %%
import sys
import os


# %%
import pathlib
import click
import subprocess
import multiprocessing
import concurrent.futures
from tqdm import tqdm
import numpy as np
import cv2
from solarian_umi.slam.docker import slam_setting_mount
from solarian_umi.slam.lumi_camera import extend_imu_through_video, slam_mask as remapped_slam_mask
from solarian_umi.slam.support.cv_util import draw_predefined_mask

# %%
@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for mapping video')
@click.option('-m', '--map_path', default=None, help='ORB_SLAM3 *.osa map atlas file')
@click.option('-d', '--docker_image', default="chicheng/orb_slam3:latest")
@click.option('-np', '--no_docker_pull', is_flag=True, default=False, help="pull docker image from docker hub")
@click.option('-nm', '--no_mask', is_flag=True, default=False, help="Whether to mask out gripper and mirrors. Set if map is created with bare GoPro no on gripper.")
@click.option('--slam_settings', default=None, help='Host ORB-SLAM3 yaml. Default is the GoPro Max Lens file in the image.')
@click.option('--coverage_mask', default=None, help='PNG, white where a remapped video has image. Its border joins the SLAM mask.')
def main(input_dir, map_path, docker_image, no_docker_pull, no_mask, slam_settings, coverage_mask):
    video_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    for fn in ['raw_video.mp4', 'imu_data.json']:
        assert video_dir.joinpath(fn).is_file()
    extend_imu_through_video(video_dir)

    if map_path is None:
        map_path = video_dir.joinpath('map_atlas.osa')
    else:
        map_path = pathlib.Path(os.path.expanduser(map_path)).absolute()
    map_path.parent.mkdir(parents=True, exist_ok=True)

    from solarian_umi.slam.docker import ensure_image
    ensure_image(docker_image, allow_pull=not no_docker_pull)

    mount_target = pathlib.Path('/data')
    csv_path = mount_target.joinpath('mapping_camera_trajectory.csv')
    video_path = mount_target.joinpath('raw_video.mp4')
    json_path = mount_target.joinpath('imu_data.json')
    mask_path = mount_target.joinpath('slam_mask.png')
    if not no_mask:
        mask_write_path = video_dir.joinpath('slam_mask.png')
        if coverage_mask is not None:
            slam_mask = remapped_slam_mask(cv2.imread(coverage_mask, cv2.IMREAD_GRAYSCALE) > 0)
        else:
            slam_mask = np.zeros((2028, 2704), dtype=np.uint8)
            slam_mask = draw_predefined_mask(
                slam_mask, color=255, mirror=True, gripper=False, finger=True)
        cv2.imwrite(str(mask_write_path.absolute()), slam_mask)

    map_mount_source = pathlib.Path(map_path)
    map_mount_target = pathlib.Path('/map').joinpath(map_mount_source.name)
    setting_volume, setting_path = slam_setting_mount(slam_settings)

    # run SLAM
    cmd = [
        'docker',
        'run',
        '--rm', # delete after finish
        '--volume', str(video_dir) + ':' + '/data',
        '--volume', str(map_mount_source.parent) + ':' + str(map_mount_target.parent),
        *setting_volume,
        docker_image,
        '/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam',
        '--vocabulary', '/ORB_SLAM3/Vocabulary/ORBvoc.txt',
        '--setting', setting_path,
        '--input_video', str(video_path),
        '--input_imu_json', str(json_path),
        '--output_trajectory_csv', str(csv_path),
        '--save_map', str(map_mount_target)
    ]
    if not no_mask:
        cmd.extend([
            '--mask_img', str(mask_path)
        ])

    stdout_path = video_dir.joinpath('slam_stdout.txt')
    stderr_path = video_dir.joinpath('slam_stderr.txt')

    result = subprocess.run(
        cmd,
        cwd=str(video_dir),
        stdout=stdout_path.open('w'),
        stderr=stderr_path.open('w')
    )
    print(result)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


# %%
if __name__ == "__main__":
    main()
