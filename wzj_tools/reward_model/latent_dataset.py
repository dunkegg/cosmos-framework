from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset


class LatentChunkCache:
    def __init__(self, max_chunks: int = 8):
        self.max_chunks = max_chunks
        self.cache = OrderedDict()

    def load(self, path: str | Path):
        path = str(path)

        if path in self.cache:
            chunk = self.cache.pop(path)
            self.cache[path] = chunk
            return chunk

        chunk = torch.load(path, map_location="cpu")
        self.cache[path] = chunk

        while len(self.cache) > self.max_chunks:
            self.cache.popitem(last=False)

        return chunk


def get_episode_index(row: dict) -> int:
    if "episode_index" not in row:
        raise KeyError(
            f"Missing episode_index, group_id={row.get('group_id')}"
        )

    return int(row["episode_index"])


def candidate_has_latent(candidate: dict) -> bool:
    return (
        candidate.get("latent_chunk") is not None
        and candidate.get("latent_index") is not None
    )


class RankingLatentDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str = "train",
        val_ratio: float = 0.1,
        val_episode_ids: list[int] | None = None,
        split_seed: int = 42,
        max_groups: int = 0,
        max_cached_chunks: int = 8,
    ):
        super().__init__()

        self.manifest_path = Path(manifest_path)
        self.split = split
        self.cache = LatentChunkCache(max_cached_chunks)

        rows = []

        with self.manifest_path.open("r") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                row = json.loads(line)

                candidates = {
                    c.get("candidate_id"): c
                    for c in row.get("candidates", [])
                }

                training_pairs = row.get(
                    "training_pairs_v0",
                    [],
                )

                if not training_pairs:
                    continue

                required_candidates = {
                    candidate_id
                    for pair in training_pairs
                    for candidate_id in pair
                }

                if not all(
                    candidate_id in candidates
                    for candidate_id in required_candidates
                ):
                    continue

                if not all(
                    candidate_has_latent(
                        candidates[candidate_id]
                    )
                    for candidate_id in required_candidates
                ):
                    continue

                row["_candidates"] = candidates
                row["_training_pairs"] = training_pairs

                rows.append(row)

        if not rows:
            raise RuntimeError(
                f"No completed latent groups found in "
                f"{self.manifest_path}"
            )

        episodes = sorted({
            get_episode_index(row)
            for row in rows
        })

        if len(episodes) < 2 and split != "full":
            raise RuntimeError(
                f"Need at least 2 completed episodes for "
                f"train/val split, got {len(episodes)}: "
                f"{episodes}"
            )

        rng = random.Random(split_seed)
        rng.shuffle(episodes)
        if split == "full":
            selected_episodes = set(episodes)

        elif val_episode_ids is not None:
            val_episodes = set(val_episode_ids)

            if split == "val":
                selected_episodes = val_episodes

            elif split == "train":
                selected_episodes = (
                    set(episodes) - val_episodes
                )

            else:
                raise ValueError(
                    f"split must be train/val/full, got {split}"
                )

        else:
            rng = random.Random(split_seed)
            rng.shuffle(episodes)

            n_val = max(
                1,
                int(round(
                    len(episodes) * val_ratio
                )),
            )

            n_val = min(
                n_val,
                len(episodes) - 1,
            )

            val_episodes = set(
                episodes[:n_val]
            )

            train_episodes = set(
                episodes[n_val:]
            )

            if split == "train":
                selected_episodes = train_episodes
            elif split == "val":
                selected_episodes = val_episodes
            else:
                raise ValueError(
                    f"split must be train/val/full, got {split}"
                )

        self.rows = [
            row
            for row in rows
            if get_episode_index(row)
            in selected_episodes
        ]

        if max_groups > 0:
            self.rows = self.rows[:max_groups]

        print(
            f"[dataset] split={split} "
            f"groups={len(self.rows)} "
            f"episodes={len(selected_episodes)} "
            f"episode_ids={sorted(selected_episodes)} "
            f"total_completed_groups={len(rows)}"
        )

    def __len__(self):
        return len(self.rows)

    def _load_latent(
        self,
        candidate: dict,
    ) -> torch.Tensor:

        chunk_path = candidate[
            "latent_chunk"
        ]

        latent_index = int(
            candidate["latent_index"]
        )

        chunk = self.cache.load(
            chunk_path
        )

        latent = chunk[
            "latents"
        ][latent_index]

        # 单条 latent:
        # [1,48,16,30,33]
        if (
            latent.ndim == 5
            and latent.shape[0] == 1
        ):
            latent = latent[0]

        # 最终必须:
        # [48,16,30,33]
        if latent.ndim != 4:
            raise ValueError(
                f"Expected [C,T,H,W], "
                f"got {tuple(latent.shape)} "
                f"from {chunk_path}:"
                f"{latent_index}"
            )

        return latent

    def __getitem__(
        self,
        idx: int,
    ):

        row = self.rows[idx]

        candidates = row[
            "_candidates"
        ]

        training_pairs = row[
            "_training_pairs"
        ]

        # V0 当前 manifest:
        #
        # [
        #   ["gt", "mild"],
        #   ["gt", "medium"],
        #   ["gt", "bad"],
        # ]
        #
        # 当前 train.py 仍然使用固定
        # gt/mild/medium/bad 四路输入。
        required = {
            candidate_id
            for pair in training_pairs
            for candidate_id in pair
        }

        latents = {
            candidate_id:
            self._load_latent(
                candidates[candidate_id]
            )
            for candidate_id in required
        }

        sample = {
            "group_id":
                row["group_id"],

            "episode":
                get_episode_index(row),

            "instruction":
                row.get(
                    "instruction",
                    "",
                ),

            "subtask_text":
                row.get(
                    "subtask_text",
                    "",
                ),

            "training_pairs":
                training_pairs,

            "nominal_ranking":
                row.get(
                    "nominal_ranking",
                    [],
                ),
        }

        sample.update(latents)

        return sample