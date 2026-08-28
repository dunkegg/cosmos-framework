#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cosmos_framework.data.generator.action.pose_utils import build_abs_pose_from_components, pose_abs_to_rel


CANDIDATE_BUILDERS = {}
NOMINAL_RANK = {"gt": 0, "mild": 1, "medium": 2, "bad": 3}


def load_annotations(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        rows = []
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"Invalid JSONL line {i}: {e}") from e
        return rows
    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise TypeError(f"Unsupported annotation root type: {type(obj)}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def lerobot_paths(dataset_root: Path, episode_index: int, video_key: str, chunk_size: int) -> dict[str, Path]:
    chunk_idx = episode_index // chunk_size
    return {
        "parquet": dataset_root / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_index:06d}.parquet",
        "video": dataset_root / "videos" / f"chunk-{chunk_idx:03d}" / video_key / f"episode_{episode_index:06d}.mp4",
    }


def load_episode_nav(parquet_path: Path, nav_key: str) -> list[Any]:
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    df = pd.read_parquet(parquet_path)
    if nav_key not in df.columns:
        raise KeyError(f"nav_key={nav_key!r} not found in {parquet_path}; columns={list(df.columns)}")
    return df[nav_key].tolist()


def parse_local_traj(value: Any) -> np.ndarray:
    rows = [np.asarray(item, dtype=np.float32).reshape(-1)[:3] for item in value]
    if not rows or any(len(x) < 3 for x in rows):
        raise ValueError(f"Invalid local_traj: {value}")
    return np.stack(rows, axis=0).astype(np.float32)


def raw_xy_to_local_forward_right(raw_xy: np.ndarray) -> np.ndarray:
    raw_xy = np.asarray(raw_xy, dtype=np.float32)
    assert raw_xy.ndim == 2 and raw_xy.shape[1] == 2 and len(raw_xy) >= 2
    local = raw_xy - raw_xy[0:1]
    return np.stack([local[:, 0], -local[:, 1]], axis=-1).astype(np.float32)


def interpolate_xy_trajectory(trajectory_xy: np.ndarray, num_poses: int) -> np.ndarray:
    traj = np.asarray(trajectory_xy, dtype=np.float32)
    assert traj.ndim == 2 and traj.shape[1] == 2 and len(traj) >= 2
    src_t = np.linspace(0.0, 1.0, len(traj))
    dst_t = np.linspace(0.0, 1.0, num_poses)
    forward = np.interp(dst_t, src_t, traj[:, 0])
    right = np.interp(dst_t, src_t, traj[:, 1])
    return np.stack([forward, right], axis=-1).astype(np.float32)


def nav_xy_to_cosmos_action(trajectory_xy: np.ndarray, pose_convention: str = "backward_anchored", add_heading: bool = True) -> np.ndarray:
    traj = np.asarray(trajectory_xy, dtype=np.float32)
    assert traj.ndim == 2 and traj.shape[1] == 2 and len(traj) >= 2

    forward, right = traj[:, 0], traj[:, 1]
    xyz = np.zeros((len(traj), 3), dtype=np.float32)
    xyz[:, 0], xyz[:, 2] = right, forward

    euler_xyz = np.zeros((len(traj), 3), dtype=np.float32)
    if add_heading:
        euler_xyz[1:, 1] = np.arctan2(np.diff(right), np.diff(forward))

    poses_abs = build_abs_pose_from_components(xyz, euler_xyz, "euler_xyz")
    actions = pose_abs_to_rel(poses_abs, rotation_format="rot6d", pose_convention=pose_convention)
    return np.asarray(actions, dtype=np.float32)


def save_cosmos_action(trajectory_xy: np.ndarray, output_json: Path) -> np.ndarray:
    actions = nav_xy_to_cosmos_action(trajectory_xy)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(actions.tolist(), indent=2), encoding="utf-8")
    return actions


def smooth_ramp(length: int) -> np.ndarray:
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
        sign = -1.0 if final_right > 0 else 1.0
        out[:, 1] += sign * max(0.08 * abs(final_right), 0.01) * ramp
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
        out[:, 1] = 0.20 * max(abs(float(gt_xy[-1, 0])), 0.5) * smooth_ramp(len(out))
    return out.astype(np.float32)


CANDIDATE_BUILDERS.update({"gt": make_gt_xy, "mild": make_mild_xy, "medium": make_medium_xy, "bad": make_bad_xy})


def linspace_int(start: int, end: int, num: int) -> list[int]:
    if num <= 0:
        return []
    if end <= start or num == 1:
        return [start]
    vals = [int(round(start + i / (num - 1) * (end - start))) for i in range(num)]
    return list(dict.fromkeys(vals))


def sample_current_frames(start_frame: int, end_frame: int, starts_per_subtask: int, min_future_frames: int) -> list[int]:
    if end_frame <= start_frame:
        return []
    latest = end_frame - min_future_frames
    if latest < start_frame:
        return [start_frame]
    return linspace_int(start_frame, latest, starts_per_subtask)


def build_ranking_manifest(
    annotations: list[dict[str, Any]], dataset_root: Path, output_root: Path, video_key: str, nav_key: str,
    chunk_size: int, action_chunk_size: int, starts_per_subtask: int, min_future_frames: int, only_usable: bool,
) -> list[dict[str, Any]]:
    rows, cache = [], {}

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

            subtask_id, subtask_text = str(subtask["subtask_id"]), str(subtask["text"])
            s, e = int(subtask["start_frame"]), int(subtask["end_frame"])
            if e <= s:
                continue

            for current_frame in sample_current_frames(s, e, starts_per_subtask, min_future_frames):
                raw_local_traj = parse_local_traj(episode_nav[current_frame])
                if len(raw_local_traj) < 2:
                    continue

                gt_xy = raw_xy_to_local_forward_right(raw_local_traj[:, :2])
                gt_xy = interpolate_xy_trajectory(gt_xy, num_poses=action_chunk_size + 1)

                group_id = f"ep{episode_index:06d}_{subtask_id}_f{current_frame:06d}"
                group_dir = output_root / "ranking_actions" / group_id
                candidates = []

                for candidate_id, builder in CANDIDATE_BUILDERS.items():
                    candidate_xy = builder(gt_xy)
                    action_path = group_dir / f"{candidate_id}.json"
                    cosmos_action = save_cosmos_action(candidate_xy, action_path)

                    candidates.append({
                        "candidate_id": candidate_id,
                        "nominal_rank": NOMINAL_RANK[candidate_id],
                        "nominal_label": "strong_positive" if candidate_id == "gt" else "synthetic_candidate",
                        "action_path": str(action_path),
                        "latent_chunk": None,
                        "latent_index": None,
                        "latent_shape": None,
                        "latent_dtype": None,
                        "teacher_score": None,
                        "teacher_reason": None,
                        "xy_final": candidate_xy[-1].astype(float).tolist(),
                        "cosmos_final_translation": cosmos_action[-1, :3].astype(float).tolist(),
                        "perturbation": candidate_id,
                    })

                rows.append({
                    "schema_version": "trajectory_ranking_xy_v2",
                    "group_id": group_id,
                    "episode_id": episode_id,
                    "episode_index": episode_index,
                    "instruction": ann.get("instruction", ""),
                    "subtask_id": subtask_id,
                    "subtask_text": subtask_text,
                    "current_frame": current_frame,
                    "subtask_range": {"start_frame": s, "end_frame": e},
                    "sampling": {
                        "starts_per_subtask": starts_per_subtask,
                        "min_future_frames": min_future_frames,
                        "remaining_gt_frames": int(e - current_frame),
                    },
                    "source": {
                        "parquet_path": str(paths["parquet"]),
                        "video_path": str(paths["video"]),
                        "video_key": video_key,
                        "nav_key": nav_key,
                        "local_traj_frame": current_frame,
                        "local_traj_num_points": int(len(raw_local_traj)),
                        "raw_nav_convention": {"dim0": "forward_x_positive", "dim1": "left_y_positive", "dim2": "yaw_ignored"},
                    },
                    "cosmos": {
                        "action_chunk_size": action_chunk_size,
                        "action_dim": 9,
                        "pose_convention": "backward_anchored",
                        "rotation_format": "rot6d",
                        "translation_convention": {"x": "right", "y": "vertical", "z": "forward"},
                    },
                    "latent_storage": {"format": "torch_chunk", "root": "ranking_latents"},
                    "nominal_ranking": ["gt", "mild", "medium", "bad"],
                    "candidates": candidates,
                    "quality": {
                        "annotation_confidence": quality.get("confidence"),
                        "annotation_usable_for_training": quality.get("usable_for_training", True),
                        "needs_review": quality.get("needs_review", False),
                        "ranking_supervision": "weak_synthetic_except_gt",
                    },
                })

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=Path("lerobot_data_r2r_50"))
    p.add_argument("--output-root", type=Path, default=Path("outputs/trajectory_ranking_xy"))
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--video-key", type=str, default="observation.images.chest_rgb")
    p.add_argument("--nav-key", type=str, default="local_traj")
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--action-chunk-size", type=int, default=60)
    p.add_argument("--starts-per-subtask", type=int, default=3)
    p.add_argument("--min-future-frames", type=int, default=20)
    p.add_argument("--only-usable", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    annotations = load_annotations(args.annotations)
    print("loaded annotations:", len(annotations))

    rows = build_ranking_manifest(
        annotations, args.dataset_root, args.output_root, args.video_key, args.nav_key, args.chunk_size,
        args.action_chunk_size, args.starts_per_subtask, args.min_future_frames, args.only_usable,
    )

    manifest = args.manifest or args.output_root / "trajectory_ranking_manifest.jsonl"
    count = write_jsonl(manifest, rows)
    print("saved ranking groups:", count)
    print("manifest:", manifest)
    if rows:
        print("example group:")
        print(json.dumps(rows[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
