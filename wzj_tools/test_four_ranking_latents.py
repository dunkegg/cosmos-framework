#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import json
import re
from pathlib import Path

import cv2
import numpy as np

from wzj_tools.cosmos_latent_client import cosmos_predict_latent, load_action_json


MANIFEST = "datasets/rxr_sub/trajectory_ranking_xy_chunksize_30/trajectory_ranking_manifest.jsonl"
GROUP_ID = "ep000000_s0_f000000"

IMAGE_ROOT = Path(
    "lerobot_data_r2r_50/videos/chunk-000/observation.images.rgb.100cm_0deg"
)

OUTPUT_DIR = Path(
    "datasets/rxr_sub/trajectory_ranking_xy_chunksize_30/tmp_latents"
)

VIDEO_OUTPUT_DIR = Path(
    "datasets/rxr_sub/trajectory_ranking_xy_chunksize_30/tmp_videos"
)


def natural_key(path: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.name)]


def read_image_frame(image_root: str | Path, episode_idx: int, frame_idx: int) -> np.ndarray:
    image_root = Path(image_root)
    prefix = f"episode_{episode_idx:06d}_"

    image_paths = sorted(
        image_root.glob(f"{prefix}*"),
        key=natural_key,
    )

    if not image_paths:
        raise RuntimeError(f"No images found for episode {episode_idx} in {image_root}")

    if not 0 <= frame_idx < len(image_paths):
        raise IndexError(
            f"episode={episode_idx}, frame_idx={frame_idx}, "
            f"num_images={len(image_paths)}"
        )

    image_path = image_paths[frame_idx]
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if frame is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    return frame


def load_group(manifest_path: str | Path, group_id: str) -> dict:
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            if row["group_id"] == group_id:
                return row

    raise RuntimeError(f"group_id not found: {group_id}")


def decode_video_frames(video_b64: list[str]) -> list[np.ndarray]:
    frames = []

    for i, frame_str in enumerate(video_b64):
        raw = base64.b64decode(frame_str)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise RuntimeError(f"Failed to decode frame {i}")

        frames.append(frame)

    return frames


def save_video(frames: list[np.ndarray], output_path: str | Path, fps: float):
    if not frames:
        raise ValueError("Empty frames")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = frames[0].shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    for frame in frames:
        writer.write(frame)

    writer.release()

    print("saved video:", output_path)


def main():
    group = load_group(MANIFEST, GROUP_ID)

    print("=" * 80)
    print("group:", group["group_id"])
    print("subtask:", group["subtask_text"])
    print("current_frame:", group["current_frame"])
    print("=" * 80)

    frame_idx = int(group["current_frame"])
    episode_idx =int(group["episode_index"])
    frame = read_image_frame(IMAGE_ROOT, episode_idx, frame_idx)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    for candidate in group["candidates"]:
        candidate_id = candidate["candidate_id"]
        action_path = candidate["action_path"]

        latent_path = OUTPUT_DIR / f"{GROUP_ID}_{candidate_id}.pt"
        video_path = VIDEO_OUTPUT_DIR / f"{GROUP_ID}_{candidate_id}.mp4"

        print()
        print("-" * 80)
        print("candidate:", candidate_id)
        print("action:", action_path)

        action = load_action_json(action_path)

        result = cosmos_predict_latent(
            frame_bgr=frame,
            action=action,
            latent_output_path=latent_path,
            output_mode="both",
        )

        video_frames = decode_video_frames(result["video"])
        fps = float(result.get("fps", 10.0))

        save_video(
            video_frames,
            video_path,
            fps,
        )

        results[candidate_id] = result

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    for candidate_id, result in results.items():
        print(
            f"{candidate_id:>8} | "
            f"shape={result.get('latent_shape')} | "
            f"dtype={result.get('latent_dtype')} | "
            f"frames={len(result.get('video', []))}"
        )


if __name__ == "__main__":
    main()