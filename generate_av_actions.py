# import json
# import numpy as np

# from cosmos_framework.data.generator.action.pose_utils import (
#     build_abs_pose_from_components,
#     pose_abs_to_rel,
# )

# NUM_ACTIONS = 60
# NUM_POSES = NUM_ACTIONS + 1

# FPS = 10
# SPEED = 0.2  # m/s
# dt = 1.0 / FPS

# OUTPUT_PATH = "datasets/action/real_case/nav_straight_backward_anchored_60.json"

# # absolute poses
# xyz = np.zeros((NUM_POSES, 3), dtype=np.float32)

# # AV / OpenCV camera convention:
# # x = right
# # y = down
# # z = forward
# xyz[:, 2] = np.arange(NUM_POSES, dtype=np.float32) * SPEED * dt

# # no roll / pitch / yaw
# euler_xyz = np.zeros((NUM_POSES, 3), dtype=np.float32)

# poses_abs = build_abs_pose_from_components(
#     xyz,
#     euler_xyz,
#     "euler_xyz",
# )

# actions = pose_abs_to_rel(
#     poses_abs,
#     rotation_format="rot6d",
#     pose_convention="backward_anchored",
# )

# assert actions.shape == (60, 9)

# with open(OUTPUT_PATH, "w") as f:
#     json.dump(actions.tolist(), f, indent=2)

# print(actions.shape)
# print(actions[:10])
# print(actions[-1])

import numpy as np
from wzj_tools.convert_xy_to_av_action import nav_xy_to_cosmos_json
# ==========================================
# 参数
# ==========================================

NUM_ACTIONS = 60
NUM_POSES = NUM_ACTIONS + 1

STEP = 0.2      # 每帧前进2cm
SIDE = 0.00      # 每帧横向1cm


FORWARD_STEP = 0.02

# 最终只横向偏 0.20 m
MAX_SIDE = 0.20

right_front = []
left_front = []

for i in range(NUM_POSES):
    t = i / NUM_ACTIONS

    forward = i * FORWARD_STEP

    # 0 -> 0.20m，且前期变化很缓
    side = MAX_SIDE * (t ** 2)

    right_front.append([forward, +side])
    left_front.append([forward, -side])

nav_xy_to_cosmos_json(
    right_front,
    "datasets/action/real_case/right_front.json",
)
nav_xy_to_cosmos_json(
    left_front,
    "datasets/action/real_case/left_front.json",
)


SWITCH_T = 0.15

traj = []

for i in range(NUM_POSES):
    t = i / NUM_ACTIONS
    forward = i * FORWARD_STEP

    if t < SWITCH_T:
        # 0 -> +MAX_SIDE
        u = t / SWITCH_T
        side = MAX_SIDE * np.sin(0.5 * np.pi * u)

    else:
        # +MAX_SIDE -> -MAX_SIDE
        u = (t - SWITCH_T) / (1.0 - SWITCH_T)

        side = MAX_SIDE * np.cos(np.pi * u)

    traj.append([forward, side])

nav_xy_to_cosmos_json(
    traj,
    "datasets/action/real_case/s_curve.json",
)





print("Done!")