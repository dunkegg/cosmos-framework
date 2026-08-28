#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


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
    raise TypeError(f"Unsupported root type: {type(obj)}")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def lerobot_relative_paths(
    episode_index: int,
    video_key: str,
    chunk_size: int = 1000,
) -> dict[str, str]:
    chunk_idx = episode_index // chunk_size
    return {
        "parquet_path": f"data/chunk-{chunk_idx:03d}/episode_{episode_index:06d}.parquet",
        "video_path": f"videos/chunk-{chunk_idx:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }


def build_phase_lookup(episode_annotation: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = {}
    for semantic in episode_annotation.get("semantic_subtasks", []):
        sid = semantic.get("semantic_id")
        if sid is not None:
            lookup[str(sid)] = list(semantic.get("execution_phases", []))
    return lookup


def find_phase_for_frame(
    phases: list[dict[str, Any]],
    frame_idx: int,
) -> dict[str, Any] | None:
    for phase in phases:
        r = phase.get("range") or {}
        s, e = r.get("start_frame"), r.get("end_frame")
        if s is None or e is None:
            continue
        if int(s) <= frame_idx <= int(e):
            b = phase.get("start_boundary") or {}
            return {
                "phase_id": phase.get("phase_id"),
                "phase_type": phase.get("phase_type"),
                "phase_text": phase.get("text"),
                "phase_start_frame": int(s),
                "phase_end_frame": int(e),
                "phase_boundary_confidence": b.get("confidence"),
                "phase_boundary_uncertainty_frames": b.get("boundary_uncertainty_frames"),
                "phase_usable_for_training": b.get("usable_for_training"),
            }
    return None


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
    samples_per_subtask: int,
    include_start: bool,
    include_end: bool,
) -> list[int]:
    frames = linspace_int(start_frame, end_frame, samples_per_subtask)
    if not include_start:
        frames = [x for x in frames if x != start_frame]
    if not include_end:
        frames = [x for x in frames if x != end_frame]
    return frames


def temporal_progress_target(frame_idx: int, start_frame: int, end_frame: int) -> float:
    if end_frame <= start_frame:
        return 1.0
    return float(min(1.0, max(0.0, (frame_idx - start_frame) / (end_frame - start_frame))))


def completion_target(frame_idx: int, end_frame: int, completion_margin_frames: int) -> int:
    return int(frame_idx >= end_frame - completion_margin_frames)


def build_reward_samples(
    annotations: list[dict[str, Any]],
    dataset_root: str,
    video_key: str,
    action_key: str,
    samples_per_subtask: int,
    horizon_frames: int,
    chunk_size: int,
    completion_margin_frames: int,
    include_start: bool,
    include_end: bool,
    only_usable: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for ann in annotations:
        quality = ann.get("quality") or {}
        if only_usable and quality.get("usable_for_training", True) is False:
            continue

        episode_index = int(ann["episode_index"])
        episode_id = str(ann.get("episode_id", f"episode_{episode_index:06d}"))
        dataset_name = episode_id.split(":", 1)[0]

        timing = ann.get("timing") or {}
        sampling_hz = float(timing.get("sampling_hz", 10.0))

        paths = lerobot_relative_paths(episode_index, video_key, chunk_size)
        if dataset_root:
            paths = {k: str(Path(dataset_root) / v) for k, v in paths.items()}

        phase_lookup = build_phase_lookup(ann)

        for subtask in ann.get("subtasks", []):
            if only_usable and subtask.get("usable_for_training", True) is False:
                continue
            if not subtask.get("resolved", True):
                continue

            subtask_id = str(subtask["subtask_id"])
            subtask_text = str(subtask["text"])
            start_frame = int(subtask["start_frame"])
            end_frame = int(subtask["end_frame"])
            if end_frame < start_frame:
                continue

            current_frames = sample_current_frames(
                start_frame,
                end_frame,
                samples_per_subtask,
                include_start,
                include_end,
            )
            phases = phase_lookup.get(subtask_id, [])

            for current_frame in current_frames:
                future_frame = min(current_frame + horizon_frames, end_frame)

                current_progress = temporal_progress_target(current_frame, start_frame, end_frame)
                future_progress = temporal_progress_target(future_frame, start_frame, end_frame)

                row: dict[str, Any] = {
                    "schema_version": "reward_sample_v1",
                    "dataset_name": dataset_name,
                    "episode_id": episode_id,
                    "episode_index": episode_index,
                    "lerobot": {
                        "parquet_path": paths["parquet_path"],
                        "video_path": paths["video_path"],
                        "video_key": video_key,
                        "action_key": action_key,
                        "sampling_hz": sampling_hz,
                    },
                    "instruction": ann.get("instruction", ""),
                    "subtask_id": subtask_id,
                    "subtask_text": subtask_text,
                    "subtask_range": {
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "start_time": start_frame / sampling_hz,
                        "end_time": end_frame / sampling_hz,
                    },
                    "current": {
                        "frame": current_frame,
                        "time": current_frame / sampling_hz,
                        "progress_target": current_progress,
                        "completion_target": completion_target(
                            current_frame, end_frame, completion_margin_frames
                        ),
                    },
                    "future": {
                        "frame": future_frame,
                        "time": future_frame / sampling_hz,
                        "progress_target": future_progress,
                        "completion_target": completion_target(
                            future_frame, end_frame, completion_margin_frames
                        ),
                        "progress_gain_target": float(future_progress - current_progress),
                        "horizon_frames": int(future_frame - current_frame),
                        "action_start_frame": current_frame,
                        "action_end_frame_exclusive": future_frame,
                    },
                    "reward_training": {
                        "absolute_progress_target": future_progress,
                        "progress_gain_target": float(future_progress - current_progress),
                        "completion_target": completion_target(
                            future_frame, end_frame, completion_margin_frames
                        ),
                    },
                    "cosmos": {
                        "latent_path": None,
                        "seed": None,
                        "action_chunk_size": int(future_frame - current_frame),
                    },
                }

                phase = find_phase_for_frame(phases, current_frame)
                if phase is not None:
                    row["current"]["execution_phase"] = phase

                future_phase = find_phase_for_frame(phases, future_frame)
                if future_phase is not None:
                    row["future"]["execution_phase"] = future_phase

                rows.append(row)

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("outputs/reward_samples.jsonl"))
    p.add_argument("--dataset-root", type=str, default="lerobot_data_r2r_50")
    p.add_argument("--video-key", type=str, default="observation.images.chest_rgb")
    p.add_argument("--action-key", type=str, default="action")
    p.add_argument("--samples-per-subtask", type=int, default=10)
    p.add_argument("--horizon-frames", type=int, default=30)
    p.add_argument("--completion-margin-frames", type=int, default=2)
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--include-start", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--include-end", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--only-usable", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    annotations = load_annotations(args.annotations)
    print("loaded annotations:", len(annotations))

    rows = build_reward_samples(
        annotations=annotations,
        dataset_root=args.dataset_root,
        video_key=args.video_key,
        action_key=args.action_key,
        samples_per_subtask=args.samples_per_subtask,
        horizon_frames=args.horizon_frames,
        chunk_size=args.chunk_size,
        completion_margin_frames=args.completion_margin_frames,
        include_start=args.include_start,
        include_end=args.include_end,
        only_usable=args.only_usable,
    )

    count = write_jsonl(args.output, rows)
    print("saved reward samples:", count)
    print("output:", args.output)

    if rows:
        print("example:")
        print(json.dumps(rows[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
