from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import (
    ActionNormalization,
    ActionSpec,
    BaseActionLeRobotDataset,
    Pos,
    Rot,
    build_action_spec,
)
from cosmos_framework.data.generator.action.pose_utils import (
    build_abs_pose_from_components,
    pose_abs_to_rel,
)


class R2RLeRobotDataset(BaseActionLeRobotDataset):
    """
    R2R indoor-navigation forward-dynamics dataset.

    Expected layout:
        root/
        ├── data/chunk-000/episode_000000.parquet
        └── videos/chunk-000/observation.images.rgb.100cm_0deg/
            ├── episode_000000_000000.png
            ├── episode_000000_000001.png
            └── ...

    Raw local_traj convention:
        [forward, left, yaw]

    Cosmos action convention:
        translation = [right, vertical, forward]
        rotation = rot6d
        action_dim = 9

    For chunk_length=T:
        video  = T + 1 frames
        action = T steps
    """

    # Reuse the existing AV/navigation domain first.
    # If your domain_utils.py uses another registered name, change this value.
    EMBODIMENT_TYPE: str = "av"
    IMAGE_FEATURE_DIR: str = "observation.images.rgb.100cm_0deg"

    def __init__(
        self,
        root: str,
        fps: float = 10.0,
        chunk_length: int = 60,
        split_seed: int = 42,
        split_val_ratio: float = 0.03,
        split: str = "train",
        mode: str = "forward_dynamics",
        viewpoint: str = "ego_view",
        action_normalization: ActionNormalization | None = None,
        use_image_augmentation: bool = False,
        parquet_cache_size: int = 32,
        caption_fallback: str = "You are an autonomous vehicle planning system.",
    ) -> None:
        super().__init__(
            fps=fps,
            chunk_length=chunk_length,
            split_seed=split_seed,
            split_val_ratio=split_val_ratio,
            split=split,
            mode=mode,
            embodiment_type=self.EMBODIMENT_TYPE,
            viewpoint=viewpoint,
            pose_convention="backward_anchored",
            rotation_format="rot6d",
            action_normalization=action_normalization,
        )

        self.root = Path(root)
        self._use_image_augmentation = bool(use_image_augmentation)
        self._parquet_cache_size = max(1, int(parquet_cache_size))
        self._caption_fallback = str(caption_fallback)

        self._parquet_cache: OrderedDict[int, pd.DataFrame] = OrderedDict()
        self._episode_info: dict[int, dict[str, Any]] = {}

        self._build_r2r_index()

    def _build_action_spec(self) -> ActionSpec:
        return build_action_spec(Pos(), Rot("rot6d"))

    @property
    def action_dim(self) -> int:
        return 9

    def _build_r2r_index(self) -> None:
        """Build the flat index structures expected by BaseActionLeRobotDataset."""
        parquet_files = sorted(self.root.glob("data/chunk-*/episode_*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found under {self.root / 'data'}. "
                "Expected data/chunk-*/episode_*.parquet"
            )

        episodes: list[tuple[int, Path, int]] = []

        for parquet_path in parquet_files:
            try:
                episode_idx = int(parquet_path.stem.replace("episode_", ""))
            except ValueError as exc:
                raise ValueError(f"Unexpected parquet filename: {parquet_path}") from exc

            df = pd.read_parquet(parquet_path, columns=["local_traj"])
            num_frames = len(df)

            if num_frames <= self._chunk_length:
                continue

            episodes.append((episode_idx, parquet_path, num_frames))

        if not episodes:
            raise RuntimeError(
                f"No episodes are longer than chunk_length={self._chunk_length}"
            )

        g = torch.Generator().manual_seed(self._split_seed)
        order = torch.randperm(len(episodes), generator=g).tolist()
        num_val = int(round(len(episodes) * self._split_val_ratio))

        if self._split == "train":
            selected = order[num_val:]
        elif self._split == "val":
            selected = order[:num_val]
        else:
            selected = order

        self._episode_records = []
        self._episode_cum_ends = []
        self._num_valid_indices = 0
        self._episode_info = {}

        for episode_list_idx in selected:
            episode_idx, parquet_path, num_frames = episodes[episode_list_idx]
            valid_len = num_frames - self._chunk_length

            if valid_len <= 0:
                continue

            self._episode_info[episode_idx] = {
                "parquet_path": parquet_path,
                "num_frames": num_frames,
            }

            # Base _resolve_index() expects:
            # (dataset_idx, row_start, valid_len, episode_id)
            # We do not use dataset_idx here. row_start=0 means row_idx is the
            # episode-local start frame.
            self._episode_records.append((0, 0, valid_len, episode_idx))
            self._num_valid_indices += valid_len
            self._episode_cum_ends.append(self._num_valid_indices)

        if self._num_valid_indices == 0:
            raise RuntimeError(
                f"No valid windows found for split={self._split!r}, "
                f"chunk_length={self._chunk_length}"
            )

    def _load_parquet(self, episode_idx: int) -> pd.DataFrame:
        """Small LRU cache for per-episode parquet files."""
        if episode_idx in self._parquet_cache:
            self._parquet_cache.move_to_end(episode_idx)
            return self._parquet_cache[episode_idx]

        info = self._episode_info.get(episode_idx)
        if info is None:
            raise KeyError(f"Unknown episode_idx={episode_idx}")

        df = pd.read_parquet(info["parquet_path"])
        self._parquet_cache[episode_idx] = df
        self._parquet_cache.move_to_end(episode_idx)

        while len(self._parquet_cache) > self._parquet_cache_size:
            self._parquet_cache.popitem(last=False)

        return df

    def _image_path(self, episode_idx: int, frame_idx: int) -> Path:
        chunk_idx = episode_idx // 1000
        return (
            self.root
            / "videos"
            / f"chunk-{chunk_idx:03d}"
            / self.IMAGE_FEATURE_DIR
            / f"episode_{episode_idx:06d}_{frame_idx:06d}.png"
        )

    def _load_image(self, episode_idx: int, frame_idx: int) -> torch.Tensor:
        """Return one RGB image as float tensor [C,H,W] in [0,1]."""
        image_path = self._image_path(episode_idx, frame_idx)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise RuntimeError(
                f"Failed to load image: episode={episode_idx}, "
                f"frame={frame_idx}, path={image_path}"
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(image)
        return torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)

    def _load_video_window(self, episode_idx: int, start_frame: int) -> torch.Tensor:
        """Return [T+1,C,H,W] float video in [0,1]."""
        frames = [
            self._load_image(episode_idx, start_frame + i)
            for i in range(self._chunk_length + 1)
        ]
        video = torch.stack(frames, dim=0)

        expected_t = self._chunk_length + 1
        if video.ndim != 4 or video.shape[0] != expected_t:
            raise RuntimeError(
                f"Unexpected video shape {tuple(video.shape)}, "
                f"expected [{expected_t},C,H,W]"
            )

        return video

    def _interpolate_xy(self, xy: np.ndarray) -> np.ndarray:
        """Interpolate sparse [forward,left] waypoints to chunk_length+1 poses."""
        xy = np.asarray(xy, dtype=np.float32)

        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"Expected xy shape [N,2], got {xy.shape}")
        if len(xy) < 2:
            raise ValueError(f"Need at least two trajectory points, got {len(xy)}")

        segment_length = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cumulative = np.concatenate(
            [np.zeros(1, dtype=np.float32), np.cumsum(segment_length, dtype=np.float32)]
        )

        num_poses = self._chunk_length + 1

        if float(cumulative[-1]) < 1e-6:
            return np.repeat(xy[:1], num_poses, axis=0).astype(np.float32)

        target = np.linspace(
            0.0,
            float(cumulative[-1]),
            num_poses,
            dtype=np.float32,
        )

        forward = np.interp(target, cumulative, xy[:, 0]).astype(np.float32)
        left = np.interp(target, cumulative, xy[:, 1]).astype(np.float32)
        return np.stack([forward, left], axis=-1)

    def _local_traj_to_action(self, local_traj: Any) -> torch.Tensor:
        """
        Convert one sparse R2R local trajectory into [chunk_length,9].

        Input waypoint: [forward, left, yaw]
        Raw yaw is ignored; heading is recomputed from XY.
        """
        points = [np.asarray(p, dtype=np.float32).reshape(-1) for p in local_traj]

        if not points:
            raise ValueError("local_traj is empty")

        traj = np.stack(points, axis=0)

        if traj.ndim != 2 or traj.shape[1] < 2:
            raise ValueError(f"Expected local_traj shape [N,>=2], got {traj.shape}")

        # Raw navigation XY = [forward, left].
        xy = traj[:, :2].astype(np.float32)
        xy = xy - xy[0:1]
        xy = self._interpolate_xy(xy)

        forward = xy[:, 0]
        left = xy[:, 1]
        d_forward = np.diff(forward)
        d_left = np.diff(left)

        heading = np.zeros(len(xy), dtype=np.float32)
        if len(xy) > 1:
            heading[1:] = np.arctan2(-d_left, d_forward).astype(np.float32)
            heading[0] = heading[1]

        # Cosmos camera/AV convention:
        # x = right, y = vertical, z = forward.
        xyz = np.zeros((len(xy), 3), dtype=np.float32)
        xyz[:, 0] = -left
        xyz[:, 1] = 0.0
        xyz[:, 2] = forward

        euler_xyz = np.zeros((len(xy), 3), dtype=np.float32)
        euler_xyz[:, 1] = heading

        poses_abs = build_abs_pose_from_components(
            xyz,
            euler_xyz,
            "euler_xyz",
        )

        action_np = pose_abs_to_rel(
            poses_abs,
            rotation_format="rot6d",
            pose_convention="backward_anchored",
        )
        action_np = np.asarray(action_np, dtype=np.float32)

        expected_shape = (self._chunk_length, 9)
        if action_np.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected action shape {action_np.shape}, expected {expected_shape}"
            )

        return torch.from_numpy(action_np).float()

    def _get_caption(self, df: pd.DataFrame, start_frame: int) -> str:
        """Prefer an instruction/task column when present."""
        for key in ("instruction", "task", "language_instruction", "caption"):
            if key not in df.columns:
                continue

            value = df.iloc[start_frame][key]
            if value is None:
                continue

            text = str(value).strip()
            if text and text.lower() != "nan":
                return text

        return self._caption_fallback

    def __getitem__(self, idx: int) -> dict[str, Any]:
        _, _, episode_idx, start_frame = self._resolve_index(idx)

        df = self._load_parquet(episode_idx)

        if start_frame < 0 or start_frame >= len(df):
            raise IndexError(
                f"Invalid start_frame={start_frame} for episode={episode_idx}, len={len(df)}"
            )

        end_frame = start_frame + self._chunk_length
        if end_frame >= len(df):
            raise IndexError(
                f"Window exceeds episode: episode={episode_idx}, "
                f"start={start_frame}, end={end_frame}, len={len(df)}"
            )

        video = self._load_video_window(
            episode_idx=episode_idx,
            start_frame=start_frame,
        )

        local_traj = df.iloc[start_frame]["local_traj"]
        action = self._local_traj_to_action(local_traj)
        ai_caption = self._get_caption(df=df, start_frame=start_frame)

        # Base _build_result() converts video [T,C,H,W] float [0,1]
        # to [C,T,H,W] uint8 and adds fps/mode/domain/viewpoint fields.
        # backward_anchored currently skips base idle-frame detection, so use
        # append_idle_frames=False in ActionTransformPipeline.
        return self._build_result(
            mode=self._mode,
            video=video,
            action=action,
            ai_caption=ai_caption,
        )
