import numpy as np
import scipy.spatial.transform as st
from solarian_umi_skill.umi.common.pose_util import pose_to_mat, mat_to_pose


def solve_table_collision(ee_pose, gripper_width, height_threshold):
    """
    Adjusts the end-effector's z-coordinate so the gripper fingers don't go below 
    the table surface, accounting for gripper orientation and width.
    Args:
        ee_pose: end-effector pose, shape (6,)
        gripper_width: gripper width, float
        height_threshold: height threshold, float
    Returns:
        ee_pose: end-effector pose, shape (6,)
    """
    # gripper finger thickness in meters
    finger_thickness = 25.5 / 1000
    keypoints = list()
    # 4 corners of the gripper fingers
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * gripper_width / 2, dy * finger_thickness / 2, 0))
    keypoints = np.asarray(keypoints)
    # convert rotation to matrix
    rot_mat = st.Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
    # transform keypoints to world coordinates
    transformed_keypoints = np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    # calculate the delta to avoid collision
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)
    # adjust the z-coordinate of the end-effector to avoid collision
    ee_pose[2] += delta

def solve_sphere_collision(ee_poses, arms_config):
    num_arms = len(arms_config)
    this_that_mat = np.identity(4)
    this_that_mat[:3, 3] = np.array([0, 0.89, 0]) # TODO: very hacky now!!!!

    for this_arm_idx in range(num_arms):
        for that_arm_idx in range(this_arm_idx + 1, num_arms):
            this_ee_mat = pose_to_mat(ee_poses[this_arm_idx][:6])
            this_sphere_mat_local = np.identity(4)
            this_sphere_mat_local[:3, 3] = np.asarray(arms_config[this_arm_idx]['sphere_center'])
            this_sphere_mat_global = this_ee_mat @ this_sphere_mat_local
            this_sphere_center = this_sphere_mat_global[:3, 3]

            that_ee_mat = pose_to_mat(ee_poses[that_arm_idx][:6])
            that_sphere_mat_local = np.identity(4)
            that_sphere_mat_local[:3, 3] = np.asarray(arms_config[that_arm_idx]['sphere_center'])
            that_sphere_mat_global = this_that_mat @ that_ee_mat @ that_sphere_mat_local
            that_sphere_center = that_sphere_mat_global[:3, 3]

            distance = np.linalg.norm(that_sphere_center - this_sphere_center)
            threshold = arms_config[this_arm_idx]['sphere_radius'] + arms_config[that_arm_idx]['sphere_radius']
            # print(that_sphere_center, this_sphere_center)
            if distance < threshold:
                print('avoid collision between two arms')
                half_delta = (threshold - distance) / 2
                normal = (that_sphere_center - this_sphere_center) / distance
                this_sphere_mat_global[:3, 3] -= half_delta * normal
                that_sphere_mat_global[:3, 3] += half_delta * normal
                
                ee_poses[this_arm_idx][:6] = mat_to_pose(this_sphere_mat_global @ np.linalg.inv(this_sphere_mat_local))
                ee_poses[that_arm_idx][:6] = mat_to_pose(np.linalg.inv(this_that_mat) @ that_sphere_mat_global @ np.linalg.inv(that_sphere_mat_local))
