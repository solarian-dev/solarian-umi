# Adapted from Universal Manipulation Interface.
# Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab.
# https://github.com/real-stanford/universal_manipulation_interface
# Licensed under the MIT License. The full text is LICENSE.umi
# beside this package.

"""
python scripts_slam_pipeline/03_batch_slam.py -i data_workspace/fold_cloth_20231214/demos
"""
# %%
import sys
import os


# %%
import pathlib
import click
import subprocess
import multiprocessing
import uuid
import concurrent.futures
from tqdm import tqdm
import cv2
import av
import numpy as np
from solarian_umi.slam.docker import slam_setting_mount
from solarian_umi.slam.lumi_camera import extend_imu_through_video, slam_mask as remapped_slam_mask
from solarian_umi.slam.support.cv_util import draw_predefined_mask


# %%
def runner(cmd, cwd, stdout_path, stderr_path, timeout, container, **kwargs):
    try:
        return subprocess.run(cmd,                       
            cwd=str(cwd),
            stdout=stdout_path.open('w'),
            stderr=stderr_path.open('w'),
            timeout=timeout,
            **kwargs)
    except subprocess.TimeoutExpired as e:
        # The timeout only kills the docker client. The container would go on
        # tracking, or hang in Shutdown, and slow every demo after it.
        subprocess.run(['docker', 'kill', container], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return e


def failure_reason(stdout_path):
    """gopro_slam's own reason to stop, else its last line; shutdown messages follow the reason."""
    lines = [line.strip() for line in stdout_path.read_text(errors='replace').splitlines() if line.strip()] \
        if stdout_path.is_file() else []
    terminating = [line for line in lines if 'Terminating' in line]
    return (terminating or lines or [''])[-1]


# %%
@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for demos folder')
@click.option('-m', '--map_path', default=None, help='ORB_SLAM3 *.osa map atlas file')
@click.option('-d', '--docker_image', default="chicheng/orb_slam3:latest")
@click.option('-n', '--num_workers', type=int, default=None)
@click.option('-ml', '--max_lost_frames', type=int, default=60)
@click.option('-tm', '--timeout_multiple', type=float, default=16, help='timeout_multiple * duration = timeout')
@click.option('-np', '--no_docker_pull', is_flag=True, default=False, help="pull docker image from docker hub")
@click.option('--slam_settings', default=None, help='Host ORB-SLAM3 yaml. Default is the GoPro Max Lens file in the image.')
@click.option('--coverage_mask', default=None, help='PNG, white where a remapped video has image. Its border joins the SLAM mask.')
def main(input_dir, map_path, docker_image, num_workers, max_lost_frames, timeout_multiple, no_docker_pull, slam_settings, coverage_mask):
    input_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    input_video_dirs = [x.parent for x in input_dir.glob('demo*/raw_video.mp4')]
    input_video_dirs += [x.parent for x in input_dir.glob('map*/raw_video.mp4')]
    print(f'Found {len(input_video_dirs)} video dirs')
    
    if map_path is None:
        map_path = input_dir.joinpath('mapping', 'map_atlas.osa')
    else:
        map_path = pathlib.Path(os.path.expanduser(map_path)).absolute()
    assert map_path.is_file()

    if num_workers is None:
        num_workers = multiprocessing.cpu_count() // 2

    from solarian_umi.slam.docker import ensure_image
    ensure_image(docker_image, allow_pull=not no_docker_pull)
    setting_volume, setting_path = slam_setting_mount(slam_settings)

    submitted = dict()
    with tqdm(total=len(input_video_dirs)) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for video_dir in tqdm(input_video_dirs):
                video_dir = video_dir.absolute()
                extend_imu_through_video(video_dir)
                if video_dir.joinpath('camera_trajectory.csv').is_file():
                    print(f"camera_trajectory.csv already exists, skipping {video_dir.name}")
                    continue
                
                # softlink won't work in bind volume
                mount_target = pathlib.Path('/data')
                csv_path = mount_target.joinpath('camera_trajectory.csv')
                video_path = mount_target.joinpath('raw_video.mp4')
                json_path = mount_target.joinpath('imu_data.json')
                mask_path = mount_target.joinpath('slam_mask.png')
                mask_write_path = video_dir.joinpath('slam_mask.png')
                
                # find video duration
                with av.open(str(video_dir.joinpath('raw_video.mp4').absolute())) as container:
                    video = container.streams.video[0]
                    duration_sec = float(video.duration * video.time_base)
                timeout = duration_sec * timeout_multiple

                if coverage_mask is not None:
                    slam_mask = remapped_slam_mask(cv2.imread(coverage_mask, cv2.IMREAD_GRAYSCALE) > 0)
                else:
                    slam_mask = np.zeros((2028, 2704), dtype=np.uint8)
                    slam_mask = draw_predefined_mask(
                        slam_mask, color=255, mirror=True, gripper=False, finger=True)
                cv2.imwrite(str(mask_write_path.absolute()), slam_mask)

                map_mount_source = map_path
                map_mount_target = pathlib.Path('/map').joinpath(map_mount_source.name)

                # run SLAM
                container = f'umi-slam-{uuid.uuid4().hex[:12]}'
                cmd = [
                    'docker',
                    'run',
                    '--rm', # delete after finish
                    '--name', container,
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
                    '--load_map', str(map_mount_target),
                    '--mask_img', str(mask_path),
                    '--max_lost_frames', str(max_lost_frames)
                ]

                stdout_path = video_dir.joinpath('slam_stdout.txt')
                stderr_path = video_dir.joinpath('slam_stderr.txt')

                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))

                future = executor.submit(runner,
                    cmd, str(video_dir), stdout_path, stderr_path, timeout, container)
                futures.add(future)
                submitted[future] = video_dir

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    results = {video_dir: future.result() for future, video_dir in submitted.items()}
    failed = {video_dir: result for video_dir, result in results.items()
              if isinstance(result, subprocess.TimeoutExpired)
              or result.returncode != 0
              or not video_dir.joinpath('camera_trajectory.csv').is_file()}
    print(f"Done! {len(results) - len(failed)} of {len(results)} videos tracked.")
    for video_dir, result in sorted(failed.items(), key=lambda item: item[0].name):
        if isinstance(result, subprocess.TimeoutExpired):
            status = f"timeout after {result.timeout:.0f} s"
        else:
            status = f"exit {result.returncode}"
        print(f"  {video_dir.name}: {status}, no trajectory. {failure_reason(video_dir.joinpath('slam_stdout.txt'))}")

# %%
if __name__ == "__main__":
    main()
