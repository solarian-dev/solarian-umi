# Adapted from Universal Manipulation Interface.
# Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab.
# https://github.com/real-stanford/universal_manipulation_interface
# Licensed under the MIT License. The full text is LICENSE.umi
# beside this package.

# %%
import sys
import os


# %%
import pathlib
import click
import cv2
import numpy as np
import pickle
import json
import pandas as pd
from scipy.spatial.transform import Rotation
from solarian_umi.umi.common.pose_util import pose_to_mat


def geometric_median(points: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """Spatial median. One far pose does not pull it the way a mean does."""
    points = np.asarray(points, dtype=np.float64)
    median = np.median(points, axis=0)
    for _ in range(64):
        delta = points - median
        dist = np.linalg.norm(delta, axis=1)
        near = dist < tol
        if np.any(near):
            return points[near][0]
        weights = 1.0 / dist
        updated = np.sum(points * weights[:, None], axis=0) / np.sum(weights)
        if float(np.linalg.norm(updated - median)) < tol:
            return updated
        median = updated
    return median


def image_center(tag_detection: str) -> np.ndarray:
    video = pathlib.Path(tag_detection).with_name("raw_video.mp4")
    if video.is_file():
        capture = cv2.VideoCapture(str(video))
        width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        if width > 0 and height > 0:
            return np.array([width, height], dtype=np.float64) / 2
    # GoPro 2.7k, the resolution this filter was written for.
    return np.array([2704.0, 2028.0]) / 2

# %%
@click.command()
@click.option('-d', '--tag_detection', required=True, help='Tag detection pkl path')
@click.option('-c', '--csv_trajectory', default=None, help='CSV trajectory from SLAM (not mapping)')
@click.option('-o', '--output', required=True, help='output json')
@click.option('-tid', '--tag_id', type=int, default=13)
@click.option('-k', '--keyframe_only', is_flag=True, default=False)
def main(tag_detection, csv_trajectory, output, tag_id, keyframe_only):
    """
    Please use camera_trajectory.csv produced by re-localizing (initializing)
    the mapping video with the map_atlas.osa produced by mapping run.
    This is much more accurate than the mapping_camera_trajectory.csv produced by
    mapping run itself.
    """

    img_center = image_center(tag_detection)

    # load
    df = pd.read_csv(csv_trajectory)
    tag_detection_results = pickle.load(open(tag_detection, 'rb'))

    # filter pose
    is_valid = ~df['is_lost']
    if keyframe_only:
        is_valid &= df['is_keyframe']

    # convert to mat
    cam_pose_timestamps = df['timestamp'].loc[is_valid].to_numpy()
    cam_pos = df[['x','y','z']].loc[is_valid].to_numpy()
    cam_rot_quat_xyzw = df[['q_x', 'q_y', 'q_z', 'q_w']].loc[is_valid].to_numpy()
    cam_rot = Rotation.from_quat(cam_rot_quat_xyzw)
    cam_pose = np.zeros((cam_pos.shape[0], 4, 4), dtype=np.float32)
    cam_pose[:,3,3] = 1
    cam_pose[:,:3,3] = cam_pos
    cam_pose[:,:3,:3] = cam_rot.as_matrix()

    # match tum data to video idx
    video_timestamps = np.array([x['time'] for x in tag_detection_results])
    tum_video_idxs = list()
    for t in cam_pose_timestamps:
        tum_video_idxs.append(np.argmin(np.abs(video_timestamps - t)))

    # find corresponding tag detection
    all_tx_slam_tag = list()
    all_idxs = list()
    for tum_idx, video_idx in enumerate(tum_video_idxs):
        td = tag_detection_results[video_idx]
        tag_dict = td['tag_dict']
        if tag_id not in tag_dict:
            continue
        
        tag = tag_dict[tag_id]
        pose = np.concatenate([tag['tvec'], tag['rvec']])
        tx_cam_tag = pose_to_mat(pose)
        tx_slam_cam = cam_pose[tum_idx]

        # filter cam pose
        dist_to_cam = np.linalg.norm(tx_cam_tag[:3,3])
        if (dist_to_cam < 0.3) or  (dist_to_cam > 2):
            continue
        
        # filter tag location in image
        corners = tag['corners']
        tag_center_pix = corners.mean(axis=0)
        dist_to_center = np.linalg.norm(tag_center_pix - img_center) / img_center[1]
        if dist_to_center > 0.6:
            continue

        tx_slam_tag = tx_slam_cam @ tx_cam_tag
        all_tx_slam_tag.append(tx_slam_tag)
        all_idxs.append(tum_idx)
    if not all_tx_slam_tag:
        raise SystemExit(
            f"No tag {tag_id} poses left after the distance and image-center filters."
        )
    all_tx_slam_tag = np.array(all_tx_slam_tag)

    # find transform closest to the mean
    all_slam_tag_pos = all_tx_slam_tag[:,:3,3]
    median = geometric_median(all_slam_tag_pos)
    dists = np.linalg.norm((all_tx_slam_tag[:,:3,3] - median), axis=-1)
    threshold = np.quantile(dists, 0.9)
    is_valid = dists < threshold
    std = all_slam_tag_pos[is_valid].std(axis=0)
    mean = all_slam_tag_pos[is_valid].mean(axis=0)
    dists = np.linalg.norm((all_tx_slam_tag[is_valid][:,:3,3] - mean), axis=-1)
    nn_idx = np.argmin(dists)
    tx_slam_tag = all_tx_slam_tag[is_valid][nn_idx]
    print("Tag detection standard deviation (cm) < 0.9 quantile")
    print(std * 100)

    # save
    result = {
        'tx_slam_tag': tx_slam_tag.tolist()
    }
    json.dump(result, open(output, 'w'), indent=2)
    print(f"Saved result to {output}")


# %%
if __name__ == "__main__":
    main()
