import argparse
import json
import os
import subprocess
from pathlib import Path

import cv2
import pandas as pd

# 改成你 r2r_lerobot_dataset.py 实际所在模块
from cosmos_framework.data.generator.action.datasets.r2r_lerobot_dataset import R2RLeRobotDataset


def find_parquet(dataset_root: Path, episode_idx: int) -> Path:
    paths = list(dataset_root.glob(f"data/chunk-*/episode_{episode_idx:06d}.parquet"))
    if len(paths) == 0:
        raise FileNotFoundError(f"Cannot find episode_{episode_idx:06d}.parquet under {dataset_root}")
    if len(paths) > 1:
        raise RuntimeError(f"Multiple parquet files found: {paths}")
    return paths[0]
def find_valid_episodes(
    dataset_root: Path,
    start_episode: int,
    num_episodes: int,
    chunk_length: int,
):
    meta_path = dataset_root / "meta" / "episodes.jsonl"

    meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                info = json.loads(line)
                meta[int(info["episode_index"])] = info

    parquet_files = sorted(
        dataset_root.glob("data/chunk-*/episode_*.parquet"),
        key=lambda p: int(p.stem.replace("episode_", "")),
    )

    selected = []

    for parquet_path in parquet_files:
        episode_idx = int(
            parquet_path.stem.replace("episode_", "")
        )

        if episode_idx < start_episode:
            continue

        if episode_idx in meta:
            num_frames = int(meta[episode_idx]["length"])
        else:
            df = pd.read_parquet(
                parquet_path,
                columns=["local_traj"],
            )
            num_frames = len(df)

        # 需要 chunk_length + 1 帧
        if num_frames <= chunk_length:
            print(
                f"[skip] episode={episode_idx} "
                f"length={num_frames} <= {chunk_length}"
            )
            continue

        selected.append(
            (
                episode_idx,
                parquet_path,
                num_frames,
            )
        )

        if len(selected) >= num_episodes:
            break

    if len(selected) < num_episodes:
        print(
            f"[warning] requested={num_episodes}, "
            f"found={len(selected)}"
        )

    if not selected:
        raise RuntimeError(
            "No valid episodes found"
        )

    print(
        "[episodes] "
        + ", ".join(
            f"{episode_idx}({num_frames})"
            for episode_idx, _, num_frames in selected
        )
    )

    return selected

def find_image_dir(dataset_root: Path, parquet_path: Path) -> Path:
    chunk_name = parquet_path.parent.name
    image_dir = (
        dataset_root
        / "videos"
        / chunk_name
        / "observation.images.rgb.100cm_0deg"
    )

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    return image_dir


def make_video(
    image_dir: Path,
    episode_idx: int,
    start_frame: int,
    num_frames: int,
    output_path: Path,
    fps: int,
):
    frames = []

    for frame_idx in range(start_frame, start_frame + num_frames):
        image_path = image_dir / f"episode_{episode_idx:06d}_{frame_idx:06d}.png"

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        frame = cv2.imread(str(image_path))

        if frame is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        frames.append(frame)

    h, w = frames[0].shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video: {output_path}")

    for frame in frames:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))

        writer.write(frame)

    writer.release()

    print(
        f"[video] {output_path} "
        f"frames={num_frames} size={w}x{h} fps={fps}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num-episodes",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/r2r_fd_test",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=480,
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Navigate according to the given trajectory.",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29501,
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
    )

    args = parser.parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"[dataset] root={dataset_root}")

    episodes = find_valid_episodes(
        dataset_root=dataset_root,
        start_episode=args.start_episode,
        num_episodes=args.num_episodes,
        chunk_length=args.chunk_length,
    )

    #
    # Dataset 只初始化一次。
    # action 转换完全使用训练时的实现。
    #
    dataset = R2RLeRobotDataset(
        root=str(dataset_root),
        fps=float(args.fps),
        chunk_length=args.chunk_length,
        split="full",
        mode="forward_dynamics",
        viewpoint="ego_view",
        use_image_augmentation=False,
    )

    samples = []

    for episode_idx, parquet_path, num_frames in episodes:

        start_frame = args.start_frame

        if start_frame + args.chunk_length >= num_frames:
            print(
                f"[skip] episode={episode_idx} "
                f"start={start_frame} "
                f"length={num_frames}"
            )
            continue

        sample_dir = (
            output_root
            / f"episode_{episode_idx:06d}"
            / f"frame_{start_frame:06d}"
        )

        sample_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        image_dir = find_image_dir(
            dataset_root,
            parquet_path,
        )

        print("")
        print(
            f"[prepare] episode={episode_idx} "
            f"start={start_frame} "
            f"length={num_frames}"
        )
        print(f"[parquet] {parquet_path}")
        print(f"[images] {image_dir}")

        df = pd.read_parquet(
            parquet_path
        )

        if "local_traj" not in df.columns:
            raise KeyError(
                f"local_traj not found: {parquet_path}"
            )

        local_traj = df.iloc[
            start_frame
        ]["local_traj"]

        action = dataset._local_traj_to_action(
            local_traj
        )

        if tuple(action.shape) != (
            args.chunk_length,
            9,
        ):
            raise RuntimeError(
                f"Unexpected action shape: "
                f"{tuple(action.shape)}"
            )

        action_path = (
            sample_dir / "action.json"
        )

        with open(
            action_path,
            "w",
        ) as f:
            json.dump(
                action.cpu().numpy().tolist(),
                f,
            )

        print(
            f"[action] {action_path} "
            f"shape={tuple(action.shape)}"
        )

        #
        # GT 61-frame video
        #
        input_video_path = (
            sample_dir / "input_gt.mp4"
        )

        make_video(
            image_dir=image_dir,
            episode_idx=episode_idx,
            start_frame=start_frame,
            num_frames=args.chunk_length + 1,
            output_path=input_video_path,
            fps=args.fps,
        )

        #
        # 如果 meta 里面有 task，
        # 可以直接用真实 instruction。
        #
        prompt = args.prompt

        meta_path = (
            dataset_root
            / "meta"
            / "episodes.jsonl"
        )

        sample_config = {
            "action_chunk_size": args.chunk_length,
            "action_path": str(action_path),
            "domain_name": "av",
            "fps": args.fps,
            "image_size": args.image_size,
            "view_point": "ego_view",
            "model_mode": "forward_dynamics",
            "name": (
                f"r2r_ep{episode_idx:06d}_"
                f"frame{start_frame:06d}"
            ),
            "prompt": prompt,
            "seed": 0,
            "vision_path": str(
                input_video_path
            ),
        }

        samples.append(
            sample_config
        )


    if not samples:
        raise RuntimeError(
            "No samples prepared"
        )

    #
    # 一次性写 batch jsonl
    #
    inference_jsonl_path = (
        output_root / "samples.jsonl"
    )

    with open(
        inference_jsonl_path,
        "w",
        encoding="utf-8",
    ) as f:

        for sample in samples:
            f.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("")
    print(
        f"[batch] {inference_jsonl_path} "
        f"samples={len(samples)}"
    )

    if args.prepare_only:
        print("[done] prepare-only")
        return


    #
    # 所有 episode 共用一个 prediction 目录
    #
    inference_output_dir = (
        output_root / "prediction"
    )

    inference_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cmd = [
        "torchrun",
        f"--nproc-per-node={args.num_gpus}",
        f"--master-port={args.master_port}",
        "-m",
        "cosmos_framework.scripts.inference",
        "--parallelism-preset=throughput",
        "-i",
        str(inference_jsonl_path),
        "-o",
        str(inference_output_dir),
        "--checkpoint-path",
        str(
            Path(
                args.checkpoint
            ).resolve()
        ),
        "--seed=0",
    ]

    if args.config_file is not None:
        cmd.extend(
            [
                "--config-file",
                str(
                    Path(
                        args.config_file
                    ).resolve()
                ),
            ]
        )

    print("")
    print("[run]")
    print(" ".join(cmd))

    env = os.environ.copy()

    subprocess.run(
        cmd,
        env=env,
        check=True,
    )

    print(
        f"[done] output="
        f"{inference_output_dir}"
    )

if __name__ == "__main__":
    main()