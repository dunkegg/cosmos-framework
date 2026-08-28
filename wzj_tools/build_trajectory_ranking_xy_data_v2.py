#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from cosmos_framework.data.generator.action.pose_utils import (
    build_abs_pose_from_components,
    pose_abs_to_rel,
)


def load_annotations(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_no, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"Invalid JSONL line {line_no}: {e}") from e
        return rows
    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise TypeError(f"Unsupported annotation root type: {type(obj)}")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def lerobot_paths(dataset_root: Path, episode_index: int, video_key: str, chunk_size: int = 1000) -> dict[str, Path]:
    chunk_idx = episode_index // chunk_size
    return {
        "parquet": dataset_root / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_index:06d}.parquet",
        "video": dataset_root / "videos" / f"chunk-{chunk_idx:03d}" / video_key / f"episode_{episode_index:06d}.mp4",
    }

def parse_local_traj(
    value,
) -> np.ndarray:
    """
    local_traj cell:
        np.ndarray(shape=(N,), dtype=object)

    each element:
        [x_forward, y_left, yaw]

    return:
        [N, 3] float32
    """

    rows = []

    for item in value:
        arr = np.asarray(
            item,
            dtype=np.float32,
        ).reshape(-1)

        if len(arr) < 3:
            raise ValueError(
                f"Invalid local_traj point: {arr}"
            )

        rows.append(
            arr[:3]
        )

    traj = np.stack(
        rows,
        axis=0,
    ).astype(np.float32)

    return traj
def load_episode_nav(
    parquet_path: Path,
    nav_key: str,
):
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    df = pd.read_parquet(parquet_path)

    if nav_key not in df.columns:
        raise KeyError(
            f"nav_key={nav_key!r} not found in {parquet_path}. "
            f"columns={list(df.columns)}"
        )

    return df[nav_key].tolist()


def raw_xy_to_local_forward_right(raw_xy: np.ndarray) -> np.ndarray:
    """
    Raw LeRobot:
      x+ = forward
      y+ = left

    Output:
      [:,0] = forward
      [:,1] = right

    Current frame becomes [0,0].
    """
    raw_xy = np.asarray(raw_xy, dtype=np.float32)
    assert raw_xy.ndim == 2 and raw_xy.shape[1] == 2 and len(raw_xy) >= 2
    local = raw_xy - raw_xy[0:1]
    forward = local[:, 0]
    right = -local[:, 1]
    return np.stack([forward, right], axis=-1).astype(np.float32)

def interpolate_xy_trajectory(
    trajectory_xy: np.ndarray,
    num_poses: int,
) -> np.ndarray:
    """
    Interpolate N sparse XY waypoints to num_poses poses.
    """

    traj = np.asarray(
        trajectory_xy,
        dtype=np.float32,
    )

    assert traj.ndim == 2
    assert traj.shape[1] == 2
    assert len(traj) >= 2

    src_t = np.linspace(
        0.0,
        1.0,
        len(traj),
    )

    dst_t = np.linspace(
        0.0,
        1.0,
        num_poses,
    )

    forward = np.interp(
        dst_t,
        src_t,
        traj[:, 0],
    )

    right = np.interp(
        dst_t,
        src_t,
        traj[:, 1],
    )

    return np.stack(
        [forward, right],
        axis=-1,
    ).astype(np.float32)
def nav_xy_to_cosmos_action(
    trajectory_xy: np.ndarray,
    pose_convention: str = "backward_anchored",
    add_heading: bool = True,
) -> np.ndarray:
    """
    trajectory_xy:
      [:,0] = forward
      [:,1] = right

    Cosmos:
      x = right
      y = vertical
      z = forward
    """
    traj = np.asarray(trajectory_xy, dtype=np.float32)
    assert traj.ndim == 2 and traj.shape[1] == 2 and len(traj) >= 2

    forward = traj[:, 0]
    right = traj[:, 1]

    xyz = np.zeros((len(traj), 3), dtype=np.float32)
    xyz[:, 0] = right
    xyz[:, 1] = 0.0
    xyz[:, 2] = forward

    euler_xyz = np.zeros((len(traj), 3), dtype=np.float32)

    if add_heading:
        d_forward = np.diff(forward)
        d_right = np.diff(right)
        heading = np.arctan2(d_right, d_forward)
        euler_xyz[0, 1] = 0.0
        euler_xyz[1:, 1] = heading

    poses_abs = build_abs_pose_from_components(xyz, euler_xyz, "euler_xyz")

    actions = pose_abs_to_rel(
        poses_abs,
        rotation_format="rot6d",
        pose_convention=pose_convention,
    )

    return np.asarray(actions, dtype=np.float32)


def nav_xy_to_cosmos_json(
    trajectory_xy: np.ndarray,
    output_json: Path,
    pose_convention: str = "backward_anchored",
    add_heading: bool = True,
) -> np.ndarray:
    actions = nav_xy_to_cosmos_action(
        trajectory_xy,
        pose_convention=pose_convention,
        add_heading=add_heading,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(actions.tolist(), f, indent=2)
    return actions


def fixed_pose_horizon(trajectory_xy: np.ndarray, num_actions: int) -> np.ndarray:
    """
    T XY poses -> T-1 Cosmos actions.
    For 60 actions, keep/pad to 61 poses.
    """
    required_poses = num_actions + 1
    traj = np.asarray(trajectory_xy, dtype=np.float32)
    if len(traj) == 0:
        raise ValueError("Empty trajectory")
    if len(traj) >= required_poses:
        return traj[:required_poses].copy()
    pad = np.repeat(traj[-1:], required_poses - len(traj), axis=0)
    return np.concatenate([traj, pad], axis=0).astype(np.float32)


def smooth_ramp(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones((length,), dtype=np.float32)
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return t * t * (3.0 - 2.0 * t)


def make_gt_xy(gt_xy: np.ndarray) -> np.ndarray:
    return gt_xy.copy()


def make_mild_xy(gt_xy: np.ndarray) -> np.ndarray:
    out = gt_xy.copy()
    ramp = smooth_ramp(len(out))
    out[:, 1] *= 0.85
    final_right = float(gt_xy[-1, 1])
    if abs(final_right) > 1e-4:
        opposite_sign = -1.0 if final_right > 0 else 1.0
        bias_amp = max(0.08 * abs(final_right), 0.01)
        out[:, 1] += opposite_sign * bias_amp * ramp
    else:
        out[:, 1] += 0.02 * ramp
    return out.astype(np.float32)


def make_medium_xy(gt_xy: np.ndarray) -> np.ndarray:
    out = gt_xy.copy()
    out[:, 0] *= 0.70
    out[:, 1] *= 0.55
    stop_idx = max(2, int(round(len(out) * 0.70)))
    if stop_idx < len(out):
        out[stop_idx:] = out[stop_idx - 1]
    return out.astype(np.float32)


def make_bad_xy(gt_xy: np.ndarray) -> np.ndarray:
    out = gt_xy.copy()
    final_right = float(gt_xy[-1, 1])
    out[:, 0] *= 0.80
    if abs(final_right) > 0.02:
        out[:, 1] *= -1.0
    else:
        ramp = smooth_ramp(len(out))
        forward_span = max(abs(float(gt_xy[-1, 0])), 0.5)
        out[:, 1] = 0.20 * forward_span * ramp
    return out.astype(np.float32)


CANDIDATE_BUILDERS = {
    "gt": make_gt_xy,
    "mild": make_mild_xy,
    "medium": make_medium_xy,
    "bad": make_bad_xy,
}

NOMINAL_RANK = {"gt": 0, "mild": 1, "medium": 2, "bad": 3}


def linspace_int(start: int, end: int, num: int) -> list[int]:
    if num <= 0:
        return []
    if end <= start:
        return [start]
    if num == 1:
        return [start]
    vals = [int(round(start + i / (num - 1) * (end - start))) for i in range(num)]
    return list(dict.fromkeys(vals))


def sample_current_frames(
    start_frame: int,
    end_frame: int,
    starts_per_subtask: int,
    min_future_frames: int,
) -> list[int]:
    """
    Sample current states only where enough real GT future remains.

    Example:
        subtask = [53, 106]
        min_future_frames = 20

    Valid current frames:
        53 ... 86

    Frames 87...105 are skipped because they would contain too little
    real future and too much repeated-pose padding.
    """
    if end_frame <= start_frame:
        return []

    latest = end_frame - min_future_frames

    if latest < start_frame:
        # The whole subtask is shorter than min_future_frames.
        # Keep only the subtask start as the best available sample.
        return [start_frame]

    return linspace_int(
        start_frame,
        latest,
        starts_per_subtask,
    )


def build_ranking_manifest(
    annotations: list[dict[str, Any]],
    dataset_root: Path,
    output_root: Path,
    video_key: str,
    nav_key: str,
    chunk_size: int,
    action_chunk_size: int,
    starts_per_subtask: int,
    min_future_frames: int,
    only_usable: bool,
) -> list[dict[str, Any]]:
    rows = []
    cache: dict[int, np.ndarray] = {}

    for ann in annotations:
        quality = ann.get("quality") or {}
        if only_usable and quality.get("usable_for_training", True) is False:
            continue

        episode_index = int(ann["episode_index"])
        episode_id = str(ann.get("episode_id", f"episode_{episode_index:06d}"))
        paths = lerobot_paths(dataset_root, episode_index, video_key, chunk_size)

        if episode_index not in cache:
            cache[episode_index] = load_episode_nav(paths["parquet"], nav_key)

        episode_nav = cache[episode_index]

        for subtask in ann.get("subtasks", []):
            if not subtask.get("resolved", True):
                continue

            subtask_id = str(subtask["subtask_id"])
            subtask_text = str(subtask["text"])
            s = int(subtask["start_frame"])
            e = int(subtask["end_frame"])

            if e <= s:
                continue

            for current_frame in sample_current_frames(
                s,
                e,
                starts_per_subtask,
                min_future_frames,
            ):
                raw_local_traj = parse_local_traj(episode_nav[current_frame])

                if len(raw_local_traj) < 2:
                    continue

                # local_traj:
                # dim0 = x forward
                # dim1 = y left
                # dim2 = yaw
                #
                # yaw 暂时不用，heading 根据 xy 重新计算
                raw_xy = raw_local_traj[:, :2]

                gt_xy = raw_xy_to_local_forward_right(raw_xy)

                gt_xy = interpolate_xy_trajectory(gt_xy,num_poses=action_chunk_size + 1)

                group_id = f"ep{episode_index:06d}_{subtask_id}_f{current_frame:06d}"
                group_dir = output_root / "ranking_actions" / group_id
                candidates = []

                for candidate_id, builder in CANDIDATE_BUILDERS.items():
                    candidate_xy = builder(gt_xy)
                    action_path = group_dir / f"{candidate_id}.json"

                    cosmos_action = nav_xy_to_cosmos_json(
                        candidate_xy,
                        action_path,
                        pose_convention="backward_anchored",
                        add_heading=True,
                    )

                    candidates.append(
                        {
                            "candidate_id": candidate_id,
                            "nominal_rank": NOMINAL_RANK[candidate_id],
                            "nominal_label": "strong_positive" if candidate_id == "gt" else "synthetic_candidate",
                            "action_path": str(action_path),
                            "cosmos_latent_path": None,
                            "teacher_score": None,
                            "teacher_reason": None,
                            "xy_final": candidate_xy[-1].astype(float).tolist(),
                            "cosmos_final_translation": cosmos_action[-1, :3].astype(float).tolist(),
                            "perturbation": candidate_id,
                        }
                    )

                rows.append(
                    {
                        "schema_version": "trajectory_ranking_xy_v1",
                        "group_id": group_id,
                        "episode_id": episode_id,
                        "episode_index": episode_index,
                        "instruction": ann.get("instruction", ""),
                        "subtask_id": subtask_id,
                        "subtask_text": subtask_text,
                        "current_frame": current_frame,
                        "subtask_range": {
                            "start_frame": s,
                            "end_frame": e,
                        },
                        "sampling": {
                            "starts_per_subtask": starts_per_subtask,
                            "min_future_frames": min_future_frames,
                            "remaining_gt_frames": int(e - current_frame),
                        },
                        "source": {
                            "parquet_path": str(
                                paths["parquet"]
                            ),

                            "video_path": str(
                                paths["video"]
                            ),

                            "video_key": video_key,

                            "nav_key": nav_key,

                            "raw_nav_convention": {
                                "dim0": "forward_x_positive",
                                "dim1": "left_y_positive",
                                "dim2": "yaw_ignored",
                            },

                            # 当前 frame 对应的 local_traj
                            "local_traj_frame": current_frame,

                            # 例如你现在就是 5
                            "local_traj_num_points": int(
                                len(raw_local_traj)
                            ),
                        },
                        "cosmos": {
                            "action_chunk_size": action_chunk_size,
                            "action_dim": 9,
                            "pose_convention": "backward_anchored",
                            "rotation_format": "rot6d",
                            "translation_convention": {
                                "x": "right",
                                "y": "vertical",
                                "z": "forward",
                            },
                        },
                        "nominal_ranking": ["gt", "mild", "medium", "bad"],
                        "candidates": candidates,
                        "quality": {
                            "annotation_confidence": quality.get("confidence"),
                            "annotation_usable_for_training": quality.get("usable_for_training", True),
                            "needs_review": quality.get("needs_review", False),
                            "ranking_supervision": "weak_synthetic_except_gt",
                        },
                    }
                )

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=Path("lerobot_data_r2r_50"))
    p.add_argument("--output-root", type=Path, default=Path("outputs/trajectory_ranking_xy"))
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--video-key", type=str, default="observation.images.chest_rgb")
    p.add_argument("--nav-key", type=str, default="action")
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--action-chunk-size", type=int, default=60)
    p.add_argument("--starts-per-subtask", type=int, default=3)
    p.add_argument(
        "--min-future-frames",
        type=int,
        default=20,
        help=(
            "Only sample current frames with at least this many real GT future "
            "frames remaining inside the subtask. Prevents late samples from "
            "being dominated by repeated-pose padding."
        ),
    )
    p.add_argument("--only-usable", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    annotations = load_annotations(args.annotations)
    print("loaded annotations:", len(annotations))

    rows = build_ranking_manifest(
        annotations=annotations,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        video_key=args.video_key,
        nav_key=args.nav_key,
        chunk_size=args.chunk_size,
        action_chunk_size=args.action_chunk_size,
        starts_per_subtask=args.starts_per_subtask,
        min_future_frames=args.min_future_frames,
        only_usable=args.only_usable,
    )

    manifest = args.manifest if args.manifest is not None else args.output_root / "trajectory_ranking_manifest.jsonl"
    count = write_jsonl(manifest, rows)

    print("saved ranking groups:", count)
    print("manifest:", manifest)

    if rows:
        print("example group:")
        print(json.dumps(rows[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
