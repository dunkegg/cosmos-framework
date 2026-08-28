import json
import numpy as np

from cosmos_framework.data.generator.action.pose_utils import (
    build_abs_pose_from_components,
    pose_abs_to_rel,
)


def nav_xy_to_cosmos_action(
    trajectory_xy,
    pose_convention="backward_anchored",
    add_heading=True,
):
    """
    输入已经转换成：
        trajectory_xy[:, 0] = forward
        trajectory_xy[:, 1] = right

    Cosmos:
        x = right
        y = vertical
        z = forward
    """

    traj = np.asarray(trajectory_xy, dtype=np.float32)

    assert traj.ndim == 2
    assert traj.shape[1] == 2
    assert len(traj) >= 2

    forward = traj[:, 0]
    right = traj[:, 1]

    xyz = np.zeros(
        (len(traj), 3),
        dtype=np.float32,
    )

    xyz[:, 0] = right
    xyz[:, 1] = 0.0
    xyz[:, 2] = forward

    euler_xyz = np.zeros(
        (len(traj), 3),
        dtype=np.float32,
    )

    if add_heading:
        d_forward = np.diff(forward)
        d_right = np.diff(right)

        heading = np.arctan2(
            d_right,
            d_forward,
        )

        # T0 identity
        euler_xyz[0, 1] = 0.0
        euler_xyz[1:, 1] = heading

    poses_abs = build_abs_pose_from_components(
        xyz,
        euler_xyz,
        "euler_xyz",
    )

    actions = pose_abs_to_rel(
        poses_abs,
        rotation_format="rot6d",
        pose_convention=pose_convention,
    )

    return np.asarray(
        actions,
        dtype=np.float32,
    )


def nav_xy_to_cosmos_json(
    trajectory_xy,
    output_json,
    pose_convention="backward_anchored",
    add_heading=True,
):
    actions = nav_xy_to_cosmos_action(
        trajectory_xy,
        pose_convention=pose_convention,
        add_heading=add_heading,
    )

    with open(output_json, "w") as f:
        json.dump(
            actions.tolist(),
            f,
            indent=2,
        )

    print("actions:", actions.shape)
    print("first 3 actions:")
    print(actions[:3])

    return actions