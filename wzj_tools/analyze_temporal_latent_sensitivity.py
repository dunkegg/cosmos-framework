from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class ChunkReader:
    def __init__(self):
        self.cache = {}

    def load_chunk(self, path: str | Path):
        path = str(path)
        if path not in self.cache:
            self.cache[path] = torch.load(path, map_location="cpu")
        return self.cache[path]

    def get_latent(self, candidate: dict) -> torch.Tensor:
        chunk_path = candidate.get("latent_chunk")
        latent_index = candidate.get("latent_index")

        if chunk_path is None or latent_index is None:
            raise ValueError(f"missing latent info: {candidate.get('candidate_id')}")

        chunk = self.load_chunk(chunk_path)
        latent = chunk["latents"][int(latent_index)].float()

        # LatentChunkWriter 取出单条后通常为 [1, C, T, H, W]
        if latent.ndim == 5 and latent.shape[0] == 1:
            latent = latent[0]

        if latent.ndim != 4:
            raise ValueError(f"expected [C,T,H,W], got {tuple(latent.shape)}")

        return latent


def temporal_metrics(gt: torch.Tensor, x: torch.Tensor) -> dict[str, np.ndarray]:
    if gt.shape != x.shape:
        raise ValueError(f"shape mismatch: {tuple(gt.shape)} vs {tuple(x.shape)}")

    # [C,T,H,W]
    diff = x - gt

    # 对 C,H,W 聚合，保留 T
    reduce_dims = (0, 2, 3)

    rms = torch.sqrt(torch.mean(diff.square(), dim=reduce_dims))
    gt_rms = torch.sqrt(torch.mean(gt.square(), dim=reduce_dims))
    rel_rms = rms / (gt_rms + 1e-8)

    # 每个 timestep 单独算 cosine
    T = gt.shape[1]
    cos = []

    for t in range(T):
        gt_t = gt[:, t].reshape(-1)
        x_t = x[:, t].reshape(-1)

        cos_t = torch.nn.functional.cosine_similarity(
            gt_t.unsqueeze(0),
            x_t.unsqueeze(0),
            dim=1,
        )[0]

        cos.append(float(cos_t))

    return {
        "rms": rms.cpu().numpy(),
        "rel_rms": rel_rms.cpu().numpy(),
        "cos": np.asarray(cos, dtype=np.float64),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-groups", type=int, default=100)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    rows = load_manifest(args.manifest)

    if args.max_groups > 0:
        rows = rows[:args.max_groups]

    reader = ChunkReader()

    names = ["mild", "medium", "bad"]

    all_rel = {name: [] for name in names}
    all_cos = {name: [] for name in names}

    valid_groups = 0
    skipped = 0

    for i, row in enumerate(rows):
        group_id = row.get("group_id", f"group_{i}")

        candidates = {
            c.get("candidate_id"): c
            for c in row.get("candidates", [])
        }

        if not {"gt", "mild", "medium", "bad"}.issubset(candidates):
            skipped += 1
            continue

        try:
            gt = reader.get_latent(candidates["gt"])

            for name in names:
                x = reader.get_latent(candidates[name])
                metrics = temporal_metrics(gt, x)
                all_rel[name].append(metrics["rel_rms"])
                all_cos[name].append(metrics["cos"])

            valid_groups += 1

        except (FileNotFoundError, KeyError, ValueError, IndexError) as e:
            skipped += 1
            print(f"[skip] {group_id}: {e}")
            continue

        if valid_groups % 10 == 0:
            print(f"processed={valid_groups}")

    if valid_groups == 0:
        raise RuntimeError("No valid groups.")

    rel = {
        name: np.stack(all_rel[name], axis=0)
        for name in names
    }

    cos = {
        name: np.stack(all_cos[name], axis=0)
        for name in names
    }

    T = rel["mild"].shape[1]

    print()
    print("=" * 125)
    print(f"valid_groups={valid_groups} skipped={skipped} T={T}")
    print("=" * 125)

    print(
        f"{'t':>3} "
        f"{'time':>7} "
        f"{'mild_rel':>12} "
        f"{'medium_rel':>12} "
        f"{'bad_rel':>12} "
        f"{'mild_cos':>12} "
        f"{'medium_cos':>12} "
        f"{'bad_cos':>12}"
    )

    for t in range(T):
        # 注意：latent T=16 不一定与原视频 10fps 一一对应。
        # 这里只给归一化 rollout 位置，不强行解释成真实视频帧时间。
        progress = t / max(T - 1, 1)

        print(
            f"{t:>3d} "
            f"{progress:>7.3f} "
            f"{rel['mild'][:, t].mean():>12.6f} "
            f"{rel['medium'][:, t].mean():>12.6f} "
            f"{rel['bad'][:, t].mean():>12.6f} "
            f"{cos['mild'][:, t].mean():>12.6f} "
            f"{cos['medium'][:, t].mean():>12.6f} "
            f"{cos['bad'][:, t].mean():>12.6f}"
        )

    print()
    print("=" * 125)
    print("TEMPORAL MEDIAN REL_RMS")
    print("=" * 125)

    print(
        f"{'t':>3} "
        f"{'mild':>12} "
        f"{'medium':>12} "
        f"{'bad':>12}"
    )

    for t in range(T):
        print(
            f"{t:>3d} "
            f"{np.median(rel['mild'][:, t]):>12.6f} "
            f"{np.median(rel['medium'][:, t]):>12.6f} "
            f"{np.median(rel['bad'][:, t]):>12.6f}"
        )

    print()
    print("=" * 125)
    print("EARLY / MIDDLE / LATE REL_RMS")
    print("=" * 125)

    splits = {
        "early": np.arange(0, max(1, T // 3)),
        "middle": np.arange(max(1, T // 3), max(2, 2 * T // 3)),
        "late": np.arange(max(2, 2 * T // 3), T),
    }

    for name in names:
        print(f"{name}:")
        for stage, idx in splits.items():
            value = rel[name][:, idx].mean()
            print(f"  {stage:<6} mean_rel_rms={value:.6f}")

    print()
    print("=" * 125)
    print("TEMPORAL SEPARATION")
    print("=" * 125)

    for t in range(T):
        p_medium_mild = np.mean(rel["medium"][:, t] > rel["mild"][:, t])
        p_bad_mild = np.mean(rel["bad"][:, t] > rel["mild"][:, t])

        print(
            f"t={t:02d} "
            f"P(medium>mild)={p_medium_mild:.3f} "
            f"P(bad>mild)={p_bad_mild:.3f}"
        )


if __name__ == "__main__":
    main()