import json
import numpy as np

from cosmos_framework.data.generator.action.pose_utils import (
    build_abs_pose_from_components,
    pose_abs_to_rel,
)


def nav_traj_to_cosmos_json(
    trajectory,
    output_json,
    pose_convention="backward_anchored",
):
    """
    Parameters
    ----------
    trajectory : list[(x_forward, y_right, yaw)]
        平面导航轨迹（绝对坐标）

    output_json : str

    pose_convention :
        backward_anchored
        backward_framewise
    """

    trajectory = np.asarray(trajectory, dtype=np.float32)

    assert trajectory.shape[1] == 3

    N = len(trajectory)

    #########################################
    # Cosmos xyz
    #########################################

    xyz = np.zeros((N, 3), dtype=np.float32)

    # nav forward -> cosmos z
    xyz[:, 2] = trajectory[:, 0]

    # nav right -> cosmos x
    xyz[:, 0] = trajectory[:, 1]

    # flat ground
    xyz[:, 1] = 0.0

    #########################################
    # Euler
    #########################################

    euler = np.zeros((N, 3), dtype=np.float32)

    # roll
    euler[:, 0] = 0

    # pitch
    euler[:, 1] = 0

    # yaw
    euler[:, 2] = trajectory[:, 2]

    #########################################
    # Build pose
    #########################################

    poses_abs = build_abs_pose_from_components(
        xyz,
        euler,
        "euler_xyz",
    )

    #########################################
    # Pose -> Cosmos action
    #########################################

    actions = pose_abs_to_rel(
        poses_abs,
        rotation_format="rot6d",
        pose_convention=pose_convention,
    )

    with open(output_json, "w") as f:
        json.dump(actions.tolist(), f, indent=2)

    return actions


if __name__ == "__main__":

    traj = []

    # 前进1.2m
    for i in range(61):
        traj.append([
            i * 0.02,   # x forward
            0.0,        # y right
            0.0         # yaw
        ])

    actions = nav_traj_to_cosmos_json(
        traj,
        "straight.json",
        pose_convention="backward_anchored",
    )

    print(actions.shape)
    print(actions[:5])