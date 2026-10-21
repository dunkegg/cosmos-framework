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
            self.cache[path] = torch.load(
                path,
                map_location="cpu",
            )

        return self.cache[path]

    def get_latent(self, candidate: dict) -> torch.Tensor:
        chunk_path = candidate.get("latent_chunk")
        latent_index = candidate.get("latent_index")

        if chunk_path is None:
            raise ValueError(
                f"candidate missing latent_chunk: "
                f"{candidate.get('candidate_id')}"
            )

        if latent_index is None:
            raise ValueError(
                f"candidate missing latent_index: "
                f"{candidate.get('candidate_id')}"
            )

        chunk = self.load_chunk(chunk_path)

        if "latents" not in chunk:
            raise KeyError(
                f"'latents' not found in {chunk_path}"
            )

        latent = chunk["latents"][int(latent_index)]

        return latent.float()


def latent_diff(
    gt: torch.Tensor,
    x: torch.Tensor,
) -> dict[str, float]:
    if gt.shape != x.shape:
        raise ValueError(
            f"shape mismatch: "
            f"{tuple(gt.shape)} vs {tuple(x.shape)}"
        )

    gt = gt.float()
    x = x.float()

    diff = x - gt

    l1 = diff.abs().mean()
    rms = torch.sqrt(
        torch.mean(diff.square())
    )

    gt_rms = torch.sqrt(
        torch.mean(gt.square())
    )

    rel_rms = rms / (
        gt_rms + 1e-8
    )

    gt_flat = gt.reshape(-1)
    x_flat = x.reshape(-1)

    cos = torch.nn.functional.cosine_similarity(
        gt_flat.unsqueeze(0),
        x_flat.unsqueeze(0),
        dim=1,
    )[0]

    return {
        "l1": float(l1),
        "rms": float(rms),
        "rel_rms": float(rel_rms),
        "cos": float(cos),
    }


def summarize(
    name: str,
    values: list[dict],
):
    rel = np.asarray(
        [x["rel_rms"] for x in values],
        dtype=np.float64,
    )

    cos = np.asarray(
        [x["cos"] for x in values],
        dtype=np.float64,
    )

    print(
        f"{name:<10} "
        f"N={len(values):>5d} "
        f"rel_mean={rel.mean():.6f} "
        f"rel_median={np.median(rel):.6f} "
        f"rel_p10={np.percentile(rel, 10):.6f} "
        f"rel_p90={np.percentile(rel, 90):.6f} "
        f"cos_mean={cos.mean():.6f} "
        f"cos_median={np.median(cos):.6f}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    rows = load_manifest(args.manifest)

    if args.max_groups > 0:
        rows = rows[:args.max_groups]

    reader = ChunkReader()

    results = {
        "mild": [],
        "medium": [],
        "bad": [],
    }

    per_group = []

    skipped = 0

    for i, row in enumerate(rows):
        group_id = row.get(
            "group_id",
            f"group_{i}",
        )

        candidates = {
            c.get("candidate_id"): c
            for c in row.get(
                "candidates",
                [],
            )
        }

        required = {
            "gt",
            "mild",
            "medium",
            "bad",
        }

        if not required.issubset(
            candidates.keys()
        ):
            skipped += 1
            continue

        try:
            gt = reader.get_latent(
                candidates["gt"]
            )

            group_result = {
                "group_id": group_id,
            }

            for name in [
                "mild",
                "medium",
                "bad",
            ]:
                latent = reader.get_latent(
                    candidates[name]
                )

                metrics = latent_diff(
                    gt,
                    latent,
                )

                results[name].append(
                    metrics
                )

                group_result[name] = (
                    metrics
                )

            per_group.append(
                group_result
            )

        except (
            FileNotFoundError,
            KeyError,
            ValueError,
            IndexError,
        ) as e:
            skipped += 1
            print(
                f"[skip] {group_id}: {e}"
            )
            continue

        if (
            len(per_group) % 10 == 0
        ):
            print(
                f"processed="
                f"{len(per_group)}"
            )

    print()
    print("=" * 120)
    print(
        f"valid_groups={len(per_group)} "
        f"skipped={skipped}"
    )
    print("=" * 120)

    for name in [
        "mild",
        "medium",
        "bad",
    ]:
        if results[name]:
            summarize(
                name,
                results[name],
            )

    if not per_group:
        return

    mild = np.asarray(
        [
            x["mild"]["rel_rms"]
            for x in per_group
        ]
    )

    medium = np.asarray(
        [
            x["medium"]["rel_rms"]
            for x in per_group
        ]
    )

    bad = np.asarray(
        [
            x["bad"]["rel_rms"]
            for x in per_group
        ]
    )

    print()
    print("=" * 120)
    print("REL_RMS ORDERING")
    print("=" * 120)

    print(
        f"P(medium > mild) = "
        f"{np.mean(medium > mild):.4f}"
    )

    print(
        f"P(bad > mild) = "
        f"{np.mean(bad > mild):.4f}"
    )

    print(
        f"P(bad > medium) = "
        f"{np.mean(bad > medium):.4f}"
    )

    print(
        f"P(mild < medium < bad) = "
        f"{np.mean((mild < medium) & (medium < bad)):.4f}"
    )

    print(
        f"P(mild is closest) = "
        f"{np.mean((mild < medium) & (mild < bad)):.4f}"
    )

    print()
    print("=" * 120)
    print("DISTANCE RATIOS")
    print("=" * 120)

    medium_mild = (
        medium /
        np.maximum(mild, 1e-8)
    )

    bad_mild = (
        bad /
        np.maximum(mild, 1e-8)
    )

    print(
        f"median medium/mild = "
        f"{np.median(medium_mild):.3f}"
    )

    print(
        f"median bad/mild = "
        f"{np.median(bad_mild):.3f}"
    )

    print()
    print("=" * 120)
    print("10 MILD CANDIDATES CLOSEST TO GT")
    print("=" * 120)

    closest = sorted(
        per_group,
        key=lambda x:
            x["mild"]["rel_rms"],
    )[:10]

    for x in closest:
        print(
            f"{x['group_id']} "
            f"mild={x['mild']['rel_rms']:.6f} "
            f"medium={x['medium']['rel_rms']:.6f} "
            f"bad={x['bad']['rel_rms']:.6f}"
        )

    print()
    print("=" * 120)
    print("10 GROUPS: MEDIUM MUCH FARTHER THAN BAD")
    print("=" * 120)

    weird = sorted(
        per_group,
        key=lambda x:
            x["medium"]["rel_rms"]
            - x["bad"]["rel_rms"],
        reverse=True,
    )[:10]

    for x in weird:
        print(
            f"{x['group_id']} "
            f"mild={x['mild']['rel_rms']:.6f} "
            f"medium={x['medium']['rel_rms']:.6f} "
            f"bad={x['bad']['rel_rms']:.6f}"
        )


if __name__ == "__main__":
    main()